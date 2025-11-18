"""
Paper Trading Simulator

Simulates trade execution without real capital.
Tracks positions, applies exit rules, calculates P&L.

Usage:
    trader = PaperTrader(initial_capital=100000)
    trader.process_signals(signals)
    trader.check_exits()
    performance = trader.get_performance()
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config.stock_config_loader import get_stock_exit_params


class Position:
    """
    Represents an open trading position.
    """

    def __init__(
        self,
        ticker: str,
        entry_date: str,
        entry_price: float,
        shares: float,
        signal_info: Dict,
        use_trailing_stop: bool = True,
        trailing_activation_pct: float = 1.5,
        trailing_distance_pct: float = 1.0
    ):
        self.ticker = ticker
        self.entry_date = pd.to_datetime(entry_date)
        self.entry_price = entry_price
        self.shares = shares
        self.signal_info = signal_info
        self.hold_days = 0
        self.current_price = entry_price
        self.current_return = 0.0

        # EXP-101: Store ATR for volatility-based stop loss
        self.atr_pct = signal_info.get('atr_pct', None)

        # EXP-098: Signal strength-based trailing stop activation
        # Adjust activation threshold based on signal quality (same tiers as EXP-096 position sizing)
        signal_strength = signal_info.get('signal_strength', None)
        if signal_strength is not None and signal_strength >= 65.0:
            if signal_strength >= 90:  # ELITE
                trailing_activation_pct = 1.0   # Activate early - let high-confidence winners run
            elif signal_strength >= 80:  # STRONG
                trailing_activation_pct = 1.25
            elif signal_strength >= 70:  # GOOD
                trailing_activation_pct = 1.5   # Baseline
            else:  # ACCEPTABLE (65-69)
                trailing_activation_pct = 2.0   # Conservative - only at profit target

        # EXP-097: Adaptive trailing stop-loss
        self.use_trailing_stop = use_trailing_stop
        self.trailing_activation_pct = trailing_activation_pct  # Profit % to activate trailing
        self.trailing_distance_pct = trailing_distance_pct       # Distance % below peak
        self.peak_price = entry_price                            # Track highest price reached
        self.trailing_stop_active = False                        # Whether trailing is active
        self.trailing_stop_price = None                          # Current trailing stop level

    def update(self, current_price: float, current_date: str):
        """
        Update position with current price and trailing stop logic.

        Args:
            current_price: Current stock price
            current_date: Current date
        """
        self.current_price = current_price
        self.current_return = ((current_price - self.entry_price) / self.entry_price) * 100
        self.hold_days = (pd.to_datetime(current_date) - self.entry_date).days

        # EXP-097: Update trailing stop logic
        if self.use_trailing_stop:
            # Update peak price if new high reached
            if current_price > self.peak_price:
                self.peak_price = current_price

            # Check if we should activate trailing stop
            if not self.trailing_stop_active:
                if self.current_return >= self.trailing_activation_pct:
                    self.trailing_stop_active = True
                    # Set initial trailing stop at trailing_distance_pct below current peak
                    self.trailing_stop_price = self.peak_price * (1 - self.trailing_distance_pct / 100)

            # Update trailing stop price if active (only moves up, never down)
            if self.trailing_stop_active:
                new_stop = self.peak_price * (1 - self.trailing_distance_pct / 100)
                if self.trailing_stop_price is None or new_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_stop

    def get_atr_based_stop_loss(self, default_stop_loss: float = -2.0) -> float:
        """
        EXP-101: Calculate ATR-based stop loss threshold.

        Normalizes stop loss to volatility for consistent risk management:
        - Low volatility (ATR < 1.5%): -1.5% stop (tighter for stable stocks)
        - Medium volatility (1.5-2.5%): -2.0% stop (baseline, current default)
        - High volatility (2.5-3.5%): -2.5% stop (wider for volatile stocks)
        - Very high volatility (≥3.5%): -3.0% stop (widest for extreme volatility)

        This prevents premature stop-outs on volatile stocks while maintaining
        tighter risk control on stable stocks.

        Args:
            default_stop_loss: Fallback stop loss if ATR unavailable (default: -2.0%)

        Returns:
            Stop loss threshold percentage (negative value)
        """
        if self.atr_pct is None or self.atr_pct <= 0:
            return default_stop_loss  # Fallback to default if ATR unavailable

        if self.atr_pct < 1.5:
            return -1.5  # Low volatility - tighter stop
        elif self.atr_pct < 2.5:
            return -2.0  # Medium volatility - baseline
        elif self.atr_pct < 3.5:
            return -2.5  # High volatility - wider stop
        else:
            return -3.0  # Very high volatility - widest stop

    def get_atr_based_profit_target(self, default_profit_target: float = 2.0) -> float:
        """
        EXP-102: Calculate ATR-based profit target threshold.

        Normalizes profit targets to volatility, complementing EXP-101 stop loss:
        - Low volatility (ATR < 1.5%): 1.5% target (realistic for stable stocks)
        - Medium volatility (1.5-2.5%): 2.0% target (baseline, current default)
        - High volatility (2.5-3.5%): 2.5% target (let volatile stocks run)
        - Very high volatility (≥3.5%): 3.0% target (maximize winners on big movers)

        This captures more profit from high-volatility winners while exiting
        low-volatility positions faster before mean reversion.

        Args:
            default_profit_target: Fallback target if ATR unavailable (default: 2.0%)

        Returns:
            Profit target threshold percentage (positive value)
        """
        if self.atr_pct is None or self.atr_pct <= 0:
            return default_profit_target  # Fallback to default if ATR unavailable

        if self.atr_pct < 1.5:
            return 1.5  # Low volatility - faster exit
        elif self.atr_pct < 2.5:
            return 2.0  # Medium volatility - baseline
        elif self.atr_pct < 3.5:
            return 2.5  # High volatility - let it run
        else:
            return 3.0  # Very high volatility - maximize big winners

    def check_exit(self, profit_target: float = 2.0, stop_loss: float = -2.0, max_hold_days: int = 2) -> Optional[str]:
        """
        Check if position should be exited.

        EXP-097: Includes trailing stop logic - if trailing stop is active and triggered,
        takes precedence over profit target (locks in gains).

        EXP-101: Uses ATR-based stop loss instead of fixed threshold for better
        volatility normalization.

        EXP-102: Uses ATR-based profit target to capture more profit from volatile
        winners while exiting low-volatility positions faster.

        Args:
            profit_target: Default profit target percentage - overridden by ATR-based calculation
            stop_loss: Default stop loss percentage (negative) - overridden by ATR-based calculation
            max_hold_days: Maximum hold period

        Returns:
            Exit reason or None
        """
        # EXP-097: Check trailing stop first (if active)
        if self.use_trailing_stop and self.trailing_stop_active and self.trailing_stop_price:
            if self.current_price <= self.trailing_stop_price:
                return 'trailing_stop'

        # EXP-102: Check ATR-based profit target (volatility-normalized)
        atr_profit_target = self.get_atr_based_profit_target(profit_target)
        if self.current_return >= atr_profit_target:
            return 'profit_target'

        # EXP-101: Check ATR-based stop loss (volatility-normalized)
        atr_stop_loss = self.get_atr_based_stop_loss(stop_loss)
        if self.current_return <= atr_stop_loss:
            return 'stop_loss'

        # Check max hold days
        if self.hold_days >= max_hold_days:
            return 'max_hold'

        return None

    def get_pnl(self) -> float:
        """Get position P&L in dollars."""
        return (self.current_price - self.entry_price) * self.shares

    def to_dict(self) -> Dict:
        """Convert position to dictionary."""
        return {
            'ticker': self.ticker,
            'entry_date': self.entry_date.strftime('%Y-%m-%d'),
            'entry_price': self.entry_price,
            'shares': self.shares,
            'hold_days': self.hold_days,
            'current_price': self.current_price,
            'current_return': self.current_return,
            'pnl': self.get_pnl(),
            'signal_info': self.signal_info
        }


class PaperTrader:
    """
    Paper trading simulator with position tracking and performance calculation.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        profit_target: float = 2.0,
        stop_loss: float = -2.0,
        max_hold_days: int = 2,
        position_size: float = 0.1,  # 10% of capital per position (base size)
        max_positions: int = 5,
        data_dir: str = 'data/paper_trading',
        use_limit_orders: bool = True,           # EXP-080: Use limit orders for entry (default: True)
        use_dynamic_sizing: bool = True,         # EXP-096: Signal-strength-based position sizing
        max_portfolio_heat: float = 0.50,        # EXP-096: Max 50% capital deployed at once
        max_position_pct: float = 0.20,          # EXP-103: Max 20% of capital per single position
        reentry_cooldown_days: int = 3,          # EXP-107: Days to wait before re-entering same stock
        max_daily_loss_pct: float = 3.0,         # EXP-108: Max daily loss (circuit breaker: stop trading)
        use_trailing_stop: bool = True,          # EXP-097: Adaptive trailing stop-loss
        trailing_activation_pct: float = 1.5,    # EXP-097: Activate trailing at +1.5% profit
        trailing_distance_pct: float = 1.0       # EXP-097: Trail 1% below peak
    ):
        """
        Initialize paper trader.

        Args:
            initial_capital: Starting capital
            profit_target: Profit target percentage
            stop_loss: Stop loss percentage
            max_hold_days: Maximum hold period
            position_size: Base position size as fraction of capital
            max_positions: Maximum concurrent positions
            data_dir: Directory to store trading data
            use_limit_orders: Use limit order entry strategy (EXP-080: +29% improvement)
            use_dynamic_sizing: Use signal-strength-based position sizing (EXP-096)
            max_portfolio_heat: Maximum fraction of capital deployed simultaneously
            max_position_pct: Maximum per-stock position size (EXP-103: prevents concentration risk)
            reentry_cooldown_days: Days to wait before re-entering same stock (EXP-107: prevents overtrading)
            max_daily_loss_pct: Maximum daily loss percentage (EXP-108: circuit breaker stops trading)
            use_trailing_stop: Use adaptive trailing stop-loss (EXP-097)
            trailing_activation_pct: Profit % to activate trailing stop
            trailing_distance_pct: Distance % below peak price for trailing stop
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_hold_days = max_hold_days
        self.position_size = position_size
        self.max_positions = max_positions
        self.data_dir = data_dir
        self.use_limit_orders = use_limit_orders
        self.use_dynamic_sizing = use_dynamic_sizing
        self.max_portfolio_heat = max_portfolio_heat
        self.max_position_pct = max_position_pct
        self.reentry_cooldown_days = reentry_cooldown_days  # EXP-107
        self.max_daily_loss_pct = max_daily_loss_pct  # EXP-108
        self.use_trailing_stop = use_trailing_stop
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_distance_pct = trailing_distance_pct

        # Trading state
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.daily_equity: List[Dict] = []
        self.last_exit_dates: Dict[str, str] = {}  # EXP-107: Track last exit date by ticker
        self.daily_start_equity: Dict[str, float] = {}  # EXP-108: Track equity at start of each day

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Create data directory
        os.makedirs(data_dir, exist_ok=True)

        # Load existing state if available
        self._load_state()

    def process_signals(self, signals: List[Dict], current_date: str = None) -> List[Dict]:
        """
        Process new trading signals.

        Args:
            signals: List of signal dictionaries from SignalScanner
            current_date: Current date (default: today)

        Returns:
            List of executed trades
        """
        if current_date is None:
            current_date = datetime.now().strftime('%Y-%m-%d')

        executed_trades = []

        for signal in signals:
            # Check if already in position
            if signal['ticker'] in self.positions:
                print(f"[SKIP] {signal['ticker']}: Already in position")
                continue

            # EXP-099: Smart position replacement when at max capacity
            if len(self.positions) >= self.max_positions:
                replacement_made = self._attempt_position_replacement(signal, current_date, executed_trades)
                if replacement_made:
                    continue  # Replacement handled, move to next signal
                else:
                    print(f"[SKIP] {signal['ticker']}: Max positions ({self.max_positions}) reached, no suitable replacement")
                    continue

            # Execute trade
            trade = self._execute_entry(signal, current_date)
            if trade:
                executed_trades.append(trade)
                print(f"[ENTRY] {signal['ticker']}: {trade['shares']:.2f} shares @ ${trade['entry_price']:.2f}")

        return executed_trades

    def _attempt_position_replacement(self, new_signal: Dict, current_date: str, executed_trades: List[Dict]) -> bool:
        """
        EXP-099: Attempt to replace weakest position with stronger signal when at max capacity.

        Replacement criteria:
        - New signal strength > weakest position strength + 15 points (minimum gap)
        - Weakest position not yet profitable (no locked gains to give up)
        - Weakest position held < 1 day (fresh enough to exit without time penalty)

        Args:
            new_signal: New signal to potentially enter
            current_date: Current date
            executed_trades: List to append trades to

        Returns:
            True if replacement made, False otherwise
        """
        if not self.positions:
            return False

        new_signal_strength = new_signal.get('signal_strength', 0)

        # Find weakest position by signal strength
        weakest_ticker = None
        weakest_strength = float('inf')

        for ticker, position in self.positions.items():
            pos_signal_strength = position.signal_info.get('signal_strength', 0)
            if pos_signal_strength < weakest_strength:
                weakest_strength = pos_signal_strength
                weakest_ticker = ticker

        if weakest_ticker is None:
            return False

        weakest_position = self.positions[weakest_ticker]

        # Check replacement criteria
        strength_gap = new_signal_strength - weakest_strength
        MIN_STRENGTH_GAP = 15.0  # Require significant quality improvement

        if strength_gap < MIN_STRENGTH_GAP:
            return False  # New signal not significantly stronger

        # Don't replace profitable positions (would give up locked gains)
        if weakest_position.current_return > 0:
            return False

        # Don't replace positions held >= 1 day (time-based exit penalty)
        if weakest_position.hold_days >= 1:
            return False

        # Criteria met - execute replacement
        print(f"[REPLACE] Swapping {weakest_ticker} (strength={weakest_strength:.0f}) "
              f"for {new_signal['ticker']} (strength={new_signal_strength:.0f}, gap=+{strength_gap:.0f})")

        # Exit weakest position
        exit_trade = {
            'ticker': weakest_ticker,
            'entry_date': weakest_position.entry_date.strftime('%Y-%m-%d'),
            'exit_date': current_date,
            'entry_price': weakest_position.entry_price,
            'exit_price': weakest_position.current_price,
            'shares': weakest_position.shares,
            'pnl': weakest_position.get_pnl(),
            'return': weakest_position.current_return,
            'hold_days': weakest_position.hold_days,
            'exit_reason': 'position_replacement',
            'signal_info': weakest_position.signal_info
        }

        # Remove position
        del self.positions[weakest_ticker]
        self.trade_history.append(exit_trade)
        self.total_trades += 1

        if exit_trade['pnl'] >= 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        print(f"[EXIT] {weakest_ticker}: Closed for replacement, "
              f"P&L: ${exit_trade['pnl']:+.2f} ({exit_trade['return']:+.2f}%)")

        # Enter new position
        entry_trade = self._execute_entry(new_signal, current_date)
        if entry_trade:
            executed_trades.append(entry_trade)
            print(f"[ENTRY] {new_signal['ticker']}: {entry_trade['shares']:.2f} shares @ ${entry_trade['entry_price']:.2f}")
            return True

        return False

    def _get_position_size_multiplier(self, signal_strength: float) -> float:
        """
        Get position size multiplier based on signal strength.

        EXP-096: Tiered position sizing by signal quality
        - ELITE (90-100): 1.5x (15% capital) - highest conviction
        - STRONG (80-89): 1.25x (12.5%) - strong signals
        - GOOD (70-79): 1.0x (10%) - baseline
        - ACCEPTABLE (65-69): 0.75x (7.5%) - minimum Q4

        Args:
            signal_strength: Signal strength score (0-100)

        Returns:
            Position size multiplier
        """
        if not self.use_dynamic_sizing:
            return 1.0

        if signal_strength >= 90:
            return 1.5  # ELITE
        elif signal_strength >= 80:
            return 1.25  # STRONG
        elif signal_strength >= 70:
            return 1.0  # GOOD
        else:
            return 0.75  # ACCEPTABLE (65-69)

    def _get_volatility_multiplier(self, atr_pct: Optional[float]) -> float:
        """
        EXP-100: Calculate position size multiplier based on stock volatility (ATR%).

        Adjusts position size inversely to volatility to normalize risk:
        - Low volatility stocks (<  1.5% ATR): 1.2x (size up stable stocks)
        - Medium volatility (1.5-2.5% ATR): 1.0x (baseline)
        - High volatility (2.5-3.5% ATR): 0.8x (size down volatile stocks)
        - Very high volatility (>= 3.5% ATR): 0.6x (minimal exposure to extreme volatility)

        Args:
            atr_pct: Average True Range as percentage of price (e.g., 2.5 means 2.5% ATR)

        Returns:
            Volatility-based position size multiplier (0.6-1.2x)
        """
        if atr_pct is None or atr_pct <= 0:
            return 1.0  # Default to baseline if ATR unavailable

        if atr_pct < 1.5:
            return 1.2  # Low volatility - size up
        elif atr_pct < 2.5:
            return 1.0  # Medium volatility - baseline
        elif atr_pct < 3.5:
            return 0.8  # High volatility - size down
        else:
            return 0.6  # Very high volatility - minimal size

    def _get_regime_multiplier(self, regime: Optional[str]) -> float:
        """
        EXP-106: Calculate position size multiplier based on market regime.

        Adapts position sizing to market conditions for better risk management:
        - BULL: 1.0x (full size - favorable trending conditions)
        - SIDEWAYS: 0.8x (reduce size - choppy range-bound conditions)
        - BEAR: 0.0x (no trading - already filtered upstream)

        Args:
            regime: Market regime ('BULL', 'SIDEWAYS', 'BEAR', 'UNKNOWN')

        Returns:
            Regime-based position size multiplier (0.0-1.0x)
        """
        if regime is None or regime == 'UNKNOWN':
            return 1.0  # Default to full size if regime unknown

        if regime == 'BULL':
            return 1.0  # Full size in trending markets
        elif regime == 'SIDEWAYS':
            return 0.8  # Reduce size in choppy markets
        elif regime == 'BEAR':
            return 0.0  # No trading (should be filtered upstream)
        else:
            return 1.0  # Default

    def _execute_entry(self, signal: Dict, entry_date: str) -> Optional[Dict]:
        """
        Execute entry trade.

        Args:
            signal: Signal dictionary
            entry_date: Entry date

        Returns:
            Trade dictionary or None
        """
        ticker = signal['ticker']

        # EXP-108: Check daily loss limit (circuit breaker)
        current_total_value = self.get_current_total_value()

        # Track start-of-day equity (reset on new day)
        if entry_date not in self.daily_start_equity:
            self.daily_start_equity[entry_date] = current_total_value

        # Calculate today's P&L
        daily_start = self.daily_start_equity[entry_date]
        daily_pnl = current_total_value - daily_start
        daily_pnl_pct = (daily_pnl / daily_start) * 100

        # Block trading if daily loss limit exceeded
        if daily_pnl_pct < -self.max_daily_loss_pct:
            print(f"[CIRCUIT BREAKER] {ticker}: Daily loss limit reached ({daily_pnl_pct:.2f}% < -{self.max_daily_loss_pct:.1f}%)")
            print(f"                  Trading suspended for remainder of day. Start: ${daily_start:.2f}, Current: ${current_total_value:.2f}")
            return None

        # EXP-107: Check re-entry cooldown period
        if ticker in self.last_exit_dates:
            last_exit = pd.to_datetime(self.last_exit_dates[ticker])
            current_date = pd.to_datetime(entry_date)
            days_since_exit = (current_date - last_exit).days

            if days_since_exit < self.reentry_cooldown_days:
                days_remaining = self.reentry_cooldown_days - days_since_exit
                print(f"[COOLDOWN] {ticker}: Re-entry blocked ({days_since_exit} days since exit, {days_remaining} days remaining)")
                return None

        # EXP-080: Use limit order pricing if enabled
        if self.use_limit_orders and 'limit_price' in signal:
            # Check if limit order would fill (intraday low must be below limit price)
            if not signal.get('limit_would_fill', False):
                print(f"[SKIP] {ticker}: Limit order @ ${signal['limit_price']:.2f} would not fill (low: ${signal.get('intraday_low', 0):.2f})")
                return None

            entry_price = signal['limit_price']
            print(f"[LIMIT ORDER] {ticker}: Entry @ ${entry_price:.2f} (vs close ${signal['price']:.2f}, saved {((signal['price'] - entry_price)/signal['price']*100):.2f}%)")
        else:
            # Traditional close-only entry
            entry_price = signal['price']

        # EXP-096: Calculate position size with signal-strength multiplier
        signal_strength = signal.get('signal_strength', 70.0)  # Default to GOOD tier
        signal_multiplier = self._get_position_size_multiplier(signal_strength)

        # EXP-100: Apply volatility-based multiplier
        atr_pct = signal.get('atr_pct')
        volatility_multiplier = self._get_volatility_multiplier(atr_pct)

        # EXP-106: Apply regime-based multiplier
        regime = signal.get('regime', None)
        regime_multiplier = self._get_regime_multiplier(regime)

        # Combined sizing: signal strength × volatility × regime
        combined_multiplier = signal_multiplier * volatility_multiplier * regime_multiplier
        position_capital = self.capital * self.position_size * combined_multiplier
        shares = position_capital / entry_price

        # EXP-103: Cap position at max_position_pct of total capital (prevents concentration risk)
        current_positions_value = sum(pos.shares * pos.entry_price for pos in self.positions.values())
        total_capital = self.capital + current_positions_value
        max_position_capital = total_capital * self.max_position_pct
        max_shares = max_position_capital / entry_price

        if shares * entry_price > max_position_capital:
            original_shares = shares
            shares = max_shares
            position_capital = shares * entry_price
            print(f"[CAP] {ticker}: Position capped at {self.max_position_pct:.0%} of capital (${max_position_capital:.2f})")
            print(f"      Original: {original_shares:.2f} shares (${original_shares * entry_price:.2f})")
            print(f"      Capped:   {shares:.2f} shares (${position_capital:.2f})")

        # EXP-096: Check portfolio heat limit (total deployed capital)
        current_positions_value = sum(pos.shares * pos.entry_price for pos in self.positions.values())
        proposed_total = current_positions_value + (shares * entry_price)
        total_capital = self.capital + current_positions_value
        portfolio_heat = proposed_total / total_capital

        if portfolio_heat > self.max_portfolio_heat:
            print(f"[SKIP] {ticker}: Portfolio heat limit reached ({portfolio_heat:.1%} > {self.max_portfolio_heat:.1%})")
            return None

        # Check if we have enough capital
        cost = shares * entry_price
        if cost > self.capital:
            print(f"[ERROR] Insufficient capital for {ticker}: Need ${cost:.2f}, have ${self.capital:.2f}")
            return None

        # Create position
        position = Position(
            ticker=ticker,
            entry_date=entry_date,
            entry_price=entry_price,
            shares=shares,
            signal_info=signal,
            use_trailing_stop=self.use_trailing_stop,
            trailing_activation_pct=self.trailing_activation_pct,
            trailing_distance_pct=self.trailing_distance_pct
        )

        self.positions[ticker] = position
        self.capital -= cost

        # Record trade
        trade = {
            'ticker': ticker,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'shares': shares,
            'cost': cost,
            'signal': signal,
            'entry_method': 'limit_order' if self.use_limit_orders else 'market_close',
            'signal_strength': signal_strength,
            'signal_multiplier': signal_multiplier,         # EXP-096
            'volatility_multiplier': volatility_multiplier, # EXP-100
            'regime_multiplier': regime_multiplier,         # EXP-106
            'combined_multiplier': combined_multiplier,     # EXP-106: signal × volatility × regime
            'atr_pct': atr_pct,                            # EXP-100
            'regime': regime                                # EXP-106
        }

        # Log position sizing info
        if self.use_dynamic_sizing:
            tier = "ELITE" if signal_strength >= 90 else "STRONG" if signal_strength >= 80 else "GOOD" if signal_strength >= 70 else "ACCEPTABLE"
            if atr_pct:
                vol_tier = "Low" if atr_pct < 1.5 else "Med" if atr_pct < 2.5 else "High" if atr_pct < 3.5 else "VHigh"
                regime_str = f" × {regime}" if regime else ""
                print(f"[SIZE] {ticker}: {tier} signal ({signal_strength:.1f}) × {vol_tier} vol (ATR {atr_pct:.1f}%){regime_str} → {combined_multiplier:.2f}x = ${cost:.2f} ({(cost/total_capital)*100:.1f}%)")
            else:
                print(f"[SIZE] {ticker}: {tier} signal ({signal_strength:.1f}) → {signal_multiplier:.2f}x = ${cost:.2f} ({(cost/total_capital)*100:.1f}%)")

        return trade

    def _get_exit_params(self, ticker: str) -> Dict[str, float]:
        """
        Get exit parameters for a ticker (per-stock or defaults).

        Args:
            ticker: Stock ticker

        Returns:
            Dictionary with profit_target, stop_loss, max_hold_days
        """
        # Try to get per-stock exit params from stock_configs.json
        stock_params = get_stock_exit_params(ticker)

        if stock_params:
            # Use per-stock parameters (EXP-044/EXP-049 optimized)
            return {
                'profit_target': stock_params.get('profit_target', self.profit_target),
                'stop_loss': stock_params.get('stop_loss', self.stop_loss),
                'max_hold_days': stock_params.get('max_hold_days', self.max_hold_days)
            }
        else:
            # Use default parameters
            return {
                'profit_target': self.profit_target,
                'stop_loss': self.stop_loss,
                'max_hold_days': self.max_hold_days
            }

    def check_exits(self, current_prices: Dict[str, float], current_date: str = None) -> List[Dict]:
        """
        Check and execute exits for open positions.

        Uses per-stock exit parameters if available (EXP-044/EXP-049 optimization),
        otherwise falls back to default parameters.

        Args:
            current_prices: Dictionary of {ticker: current_price}
            current_date: Current date (default: today)

        Returns:
            List of executed exits
        """
        if current_date is None:
            current_date = datetime.now().strftime('%Y-%m-%d')

        executed_exits = []
        tickers_to_exit = []

        # Update all positions and check exit conditions
        for ticker, position in self.positions.items():
            if ticker not in current_prices:
                print(f"[WARN] No price data for {ticker}, skipping exit check")
                continue

            current_price = current_prices[ticker]
            position.update(current_price, current_date)

            # Get per-stock or default exit parameters
            exit_params = self._get_exit_params(ticker)

            exit_reason = position.check_exit(
                profit_target=exit_params['profit_target'],
                stop_loss=exit_params['stop_loss'],
                max_hold_days=exit_params['max_hold_days']
            )

            if exit_reason:
                tickers_to_exit.append((ticker, exit_reason))

        # Execute exits
        for ticker, exit_reason in tickers_to_exit:
            exit_trade = self._execute_exit(ticker, exit_reason, current_date)
            if exit_trade:
                executed_exits.append(exit_trade)
                print(f"[EXIT] {ticker}: {exit_reason} @ ${exit_trade['exit_price']:.2f}, P&L: ${exit_trade['pnl']:+.2f} ({exit_trade['return']:+.2f}%)")

        return executed_exits

    def _execute_exit(self, ticker: str, exit_reason: str, exit_date: str) -> Optional[Dict]:
        """
        Execute exit trade.

        Args:
            ticker: Stock ticker
            exit_reason: Reason for exit
            exit_date: Exit date

        Returns:
            Exit trade dictionary or None
        """
        if ticker not in self.positions:
            return None

        position = self.positions[ticker]
        exit_price = position.current_price
        proceeds = position.shares * exit_price
        pnl = position.get_pnl()

        # Update capital
        self.capital += proceeds

        # Record trade
        exit_trade = {
            'ticker': ticker,
            'entry_date': position.entry_date.strftime('%Y-%m-%d'),
            'entry_price': position.entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'shares': position.shares,
            'hold_days': position.hold_days,
            'return': position.current_return,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'signal_info': position.signal_info
        }

        # Update stats
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Add to history
        self.trade_history.append(exit_trade)

        # EXP-107: Track exit date for re-entry cooldown
        self.last_exit_dates[ticker] = exit_date

        # Remove position
        del self.positions[ticker]

        return exit_trade

    def update_daily_equity(self, current_prices: Dict[str, float], current_date: str = None):
        """
        Update daily equity tracking.

        Args:
            current_prices: Dictionary of {ticker: current_price}
            current_date: Current date
        """
        if current_date is None:
            current_date = datetime.now().strftime('%Y-%m-%d')

        # Update positions
        for ticker, position in self.positions.items():
            if ticker in current_prices:
                position.update(current_prices[ticker], current_date)

        # Calculate total equity
        positions_value = sum(pos.shares * pos.current_price for pos in self.positions.values())
        total_equity = self.capital + positions_value

        # Record equity
        self.daily_equity.append({
            'date': current_date,
            'capital': self.capital,
            'positions_value': positions_value,
            'total_equity': total_equity,
            'return': ((total_equity - self.initial_capital) / self.initial_capital) * 100,
            'num_positions': len(self.positions)
        })

    def get_performance(self) -> Dict:
        """
        Calculate performance metrics.

        Returns:
            Performance dictionary
        """
        if self.total_trades == 0:
            # Calculate equity even with no trades
            positions_value = sum(pos.shares * pos.current_price for pos in self.positions.values())
            total_equity = self.capital + positions_value
            total_return = ((total_equity - self.initial_capital) / self.initial_capital) * 100

            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'total_return': total_return,
                'current_capital': self.capital,
                'current_equity': total_equity,
                'positions_value': positions_value,
                'current_positions': len(self.positions)
            }

        total_pnl = sum(trade['pnl'] for trade in self.trade_history)
        avg_pnl = total_pnl / self.total_trades
        win_rate = (self.winning_trades / self.total_trades) * 100

        # Calculate current equity
        positions_value = sum(pos.shares * pos.current_price for pos in self.positions.values())
        total_equity = self.capital + positions_value
        total_return = ((total_equity - self.initial_capital) / self.initial_capital) * 100

        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'total_return': total_return,
            'current_capital': self.capital,
            'current_equity': total_equity,
            'positions_value': positions_value,
            'current_positions': len(self.positions)
        }

    def get_open_positions(self) -> List[Dict]:
        """Get list of open positions."""
        return [pos.to_dict() for pos in self.positions.values()]

    def get_trade_history(self) -> List[Dict]:
        """Get trade history."""
        return self.trade_history

    def _save_state(self):
        """Save trading state to disk."""
        state = {
            'capital': self.capital,
            'positions': [pos.to_dict() for pos in self.positions.values()],
            'trade_history': self.trade_history,
            'daily_equity': self.daily_equity,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        }

        state_file = os.path.join(self.data_dir, 'paper_trading_state.json')
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load trading state from disk."""
        state_file = os.path.join(self.data_dir, 'paper_trading_state.json')
        if not os.path.exists(state_file):
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            self.capital = state.get('capital', self.initial_capital)
            self.trade_history = state.get('trade_history', [])
            self.daily_equity = state.get('daily_equity', [])
            self.total_trades = state.get('total_trades', 0)
            self.winning_trades = state.get('winning_trades', 0)
            self.losing_trades = state.get('losing_trades', 0)

            # Reconstruct positions
            for pos_dict in state.get('positions', []):
                position = Position(
                    ticker=pos_dict['ticker'],
                    entry_date=pos_dict['entry_date'],
                    entry_price=pos_dict['entry_price'],
                    shares=pos_dict['shares'],
                    signal_info=pos_dict['signal_info'],
                    use_trailing_stop=self.use_trailing_stop,
                    trailing_activation_pct=self.trailing_activation_pct,
                    trailing_distance_pct=self.trailing_distance_pct
                )
                position.hold_days = pos_dict['hold_days']
                position.current_price = pos_dict['current_price']
                position.current_return = pos_dict['current_return']
                self.positions[pos_dict['ticker']] = position

            print(f"[LOADED] Paper trading state: {self.total_trades} trades, ${self.capital:.2f} capital")

        except Exception as e:
            print(f"[WARN] Could not load state: {e}")

    def save(self):
        """Save current state to disk."""
        self._save_state()
        print(f"[SAVED] Paper trading state")


if __name__ == "__main__":
    # Example usage
    trader = PaperTrader(
        initial_capital=100000,
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=2,
        position_size=0.1,
        max_positions=5
    )

    # Simulate signal
    test_signal = {
        'ticker': 'NVDA',
        'date': '2025-11-14',
        'signal_type': 'BUY',
        'price': 145.50,
        'z_score': -2.1,
        'rsi': 28.5,
        'expected_return': 3.2
    }

    # Process signal
    print("Processing test signal...")
    trades = trader.process_signals([test_signal])

    if trades:
        print(f"\nExecuted {len(trades)} trade(s)")
        print("\nOpen positions:")
        for pos in trader.get_open_positions():
            print(f"  {pos['ticker']}: {pos['shares']:.2f} shares @ ${pos['entry_price']:.2f}")

        # Simulate price update (profit target hit)
        print("\nSimulating price update (+2.5% gain)...")
        current_prices = {'NVDA': 149.14}  # +2.5% gain
        exits = trader.check_exits(current_prices, '2025-11-15')

        if exits:
            print(f"\nExecuted {len(exits)} exit(s)")
            performance = trader.get_performance()
            print(f"\nPerformance:")
            print(f"  Total trades: {performance['total_trades']}")
            print(f"  Win rate: {performance['win_rate']:.1f}%")
            print(f"  Total P&L: ${performance['total_pnl']:+.2f}")
            print(f"  Total return: {performance['total_return']:+.2f}%")

        # Save state
        trader.save()
    else:
        print("No trades executed")

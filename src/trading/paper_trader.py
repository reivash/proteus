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

    def check_exit(self, profit_target: float = 2.0, stop_loss: float = -2.0, max_hold_days: int = 2) -> Optional[str]:
        """
        Check if position should be exited.

        EXP-097: Includes trailing stop logic - if trailing stop is active and triggered,
        takes precedence over profit target (locks in gains).

        Args:
            profit_target: Profit target percentage
            stop_loss: Stop loss percentage (negative)
            max_hold_days: Maximum hold period

        Returns:
            Exit reason or None
        """
        # EXP-097: Check trailing stop first (if active)
        if self.use_trailing_stop and self.trailing_stop_active and self.trailing_stop_price:
            if self.current_price <= self.trailing_stop_price:
                return 'trailing_stop'

        # Check profit target
        if self.current_return >= profit_target:
            return 'profit_target'
        # Check fixed stop loss (only if trailing not active or price hasn't activated trailing)
        elif self.current_return <= stop_loss:
            return 'stop_loss'
        # Check max hold days
        elif self.hold_days >= max_hold_days:
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
        self.use_trailing_stop = use_trailing_stop
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_distance_pct = trailing_distance_pct

        # Trading state
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.daily_equity: List[Dict] = []

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
            # Check if we can take position
            if len(self.positions) >= self.max_positions:
                print(f"[SKIP] {signal['ticker']}: Max positions ({self.max_positions}) reached")
                continue

            # Check if already in position
            if signal['ticker'] in self.positions:
                print(f"[SKIP] {signal['ticker']}: Already in position")
                continue

            # Execute trade
            trade = self._execute_entry(signal, current_date)
            if trade:
                executed_trades.append(trade)
                print(f"[ENTRY] {signal['ticker']}: {trade['shares']:.2f} shares @ ${trade['entry_price']:.2f}")

        return executed_trades

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
        size_multiplier = self._get_position_size_multiplier(signal_strength)
        position_capital = self.capital * self.position_size * size_multiplier
        shares = position_capital / entry_price

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
            'size_multiplier': size_multiplier
        }

        # Log position sizing info
        if self.use_dynamic_sizing:
            tier = "ELITE" if signal_strength >= 90 else "STRONG" if signal_strength >= 80 else "GOOD" if signal_strength >= 70 else "ACCEPTABLE"
            print(f"[SIZE] {ticker}: {tier} signal ({signal_strength:.1f}) → {size_multiplier:.2f}x multiplier = ${cost:.2f} ({(cost/total_capital)*100:.1f}% of total capital)")

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

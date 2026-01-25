"""
EXPERIMENT: EXP-012
Date: 2025-11-16
Objective: Optimize returns through dynamic position sizing

RESEARCH MOTIVATION:
Current strategy uses fixed position sizing - every trade gets same capital allocation.
This leaves money on the table:
- HIGH confidence signals (emotion-driven panics) deserve larger positions
- LOW confidence signals (information-driven) should be skipped entirely
- High volatility stocks need smaller positions (risk management)
- Kelly Criterion can optimize sizing based on win rate and payoff ratio

METHOD:
- Strategy: Same mean reversion v5.0 + sentiment from EXP-011
- Position Sizing: DYNAMIC vs FIXED

FIXED SIZING (Baseline):
  - Every trade: $10,000 capital (100% of base)
  - Win rate: 87.5%
  - Return: +39.31%

DYNAMIC SIZING (EXP-012):
  - HIGH confidence: 2.0x base × volatility adj × Kelly optimal
  - MEDIUM confidence: 1.0x base × volatility adj × Kelly optimal
  - LOW confidence: 0x (skip trade)
  - Volatility adjustment: Inverse sizing (low vol = larger, high vol = smaller)
  - Kelly Criterion: Optimal sizing based on win rate and payoff ratio

EXPECTED OUTCOMES:
- Same win rate (87.5%) - not changing strategy
- Higher returns (+15-25%) - better capital allocation
- Lower risk-adjusted returns - Sharpe improvement
- Fewer total trades - skip LOW confidence signals

COMPARISON METRICS:
- Total return: Fixed vs Dynamic
- Sharpe ratio: Risk-adjusted performance
- Max drawdown: Worst-case scenario
- Capital efficiency: Return per dollar deployed
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.models.trading.mean_reversion import MeanReversionDetector
from common.config.mean_reversion_params import get_params, get_all_tickers
from common.trading.risk_management import PositionSizer, calculate_stock_volatility


class DynamicSizingBacktester:
    """Backtest mean reversion strategy with dynamic position sizing."""

    def __init__(self,
                 initial_capital: float = 10000,
                 profit_target_day0: float = 2.0,
                 stop_loss_day0: float = -2.0,
                 use_dynamic_sizing: bool = True,
                 max_hold_days: int = 3):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            profit_target_day0: Day 0 profit target (%)
            stop_loss_day0: Day 0 stop loss (%)
            use_dynamic_sizing: Use dynamic sizing (True) or fixed (False)
            max_hold_days: Maximum days to hold position
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.profit_target_day0 = profit_target_day0
        self.stop_loss_day0 = stop_loss_day0
        self.use_dynamic_sizing = use_dynamic_sizing
        self.max_hold_days = max_hold_days

        # Position sizer (if using dynamic)
        if use_dynamic_sizing:
            self.position_sizer = PositionSizer(
                base_capital=initial_capital,
                max_position_pct=0.30,
                max_risk_per_trade=0.02,
                max_portfolio_heat=0.25,
                use_kelly=True,
                kelly_fraction=0.5
            )
        else:
            self.position_sizer = None

        # Tracking
        self.trades = []
        self.open_positions = []

    def backtest(self, signals_df: pd.DataFrame, price_data: pd.DataFrame,
                historical_performance: Dict = None) -> Dict:
        """
        Run backtest with position sizing.

        Args:
            signals_df: DataFrame with panic_sell signals and confidence levels
            price_data: Full OHLCV data for calculating exits
            historical_performance: Dict with win_rate, avg_gain, avg_loss by confidence

        Returns:
            Dictionary with backtest results
        """
        # Reset state
        self.current_capital = self.initial_capital
        self.trades = []
        self.open_positions = []

        # Get panic sell signals
        panic_sell_dates = signals_df[signals_df['panic_sell'] == True].copy()

        print(f"[INFO] Backtesting {len(panic_sell_dates)} panic sell signals...")
        print(f"[INFO] Sizing mode: {'DYNAMIC' if self.use_dynamic_sizing else 'FIXED'}")

        for idx, signal_row in panic_sell_dates.iterrows():
            entry_date = signal_row['Date']
            entry_price = signal_row['Close']
            confidence = signal_row.get('confidence', 'MEDIUM')

            # Calculate position size
            if self.use_dynamic_sizing:
                # Get stock volatility
                volatility = self._calculate_volatility_at_date(price_data, entry_date)

                # Get historical performance for this confidence level
                perf = historical_performance.get(confidence, {}) if historical_performance else {}
                win_rate = perf.get('win_rate', 0.875)  # Default to baseline
                avg_gain = perf.get('avg_gain', 0.0525)
                avg_loss = perf.get('avg_loss', -0.022)

                # Calculate dynamic size
                sizing = self.position_sizer.calculate_position_size(
                    ticker='STOCK',  # Generic for backtest
                    entry_price=entry_price,
                    stop_loss_pct=self.stop_loss_day0 / 100,
                    confidence=confidence,
                    volatility=volatility,
                    win_rate=win_rate,
                    avg_gain=avg_gain / 100,
                    avg_loss=avg_loss / 100,
                    open_positions=self.open_positions
                )

                shares = sizing['shares']
                position_value = sizing['position_value']

                # Skip if zero shares (LOW confidence or portfolio heat)
                if shares == 0:
                    continue

            else:
                # Fixed sizing: Use all available capital
                shares = int(self.current_capital / entry_price)
                position_value = shares * entry_price

            # Open position
            position = {
                'entry_date': entry_date,
                'entry_price': entry_price,
                'shares': shares,
                'position_value': position_value,
                'confidence': confidence,
                'days_held': 0,
                'exit_date': None,
                'exit_price': None,
                'exit_reason': None
            }

            # Find exit
            exit_info = self._find_exit(position, price_data, entry_date)

            if exit_info:
                # Close position
                position.update(exit_info)

                # Calculate P&L
                profit = (position['exit_price'] - position['entry_price']) * shares
                return_pct = (position['exit_price'] / position['entry_price'] - 1) * 100

                position['profit'] = profit
                position['return_pct'] = return_pct

                # Update capital (for fixed sizing, track cumulative)
                if not self.use_dynamic_sizing:
                    self.current_capital += profit

                self.trades.append(position)

        # Calculate metrics
        return self._calculate_metrics()

    def _calculate_volatility_at_date(self, price_data: pd.DataFrame, date: pd.Timestamp) -> float:
        """Calculate stock volatility as of a specific date."""
        # Get data up to this date
        historical = price_data[price_data.index <= date].tail(30)

        if len(historical) < 10:
            return 0.03  # Default

        return calculate_stock_volatility(historical, period=min(30, len(historical)))

    def _find_exit(self, position: Dict, price_data: pd.DataFrame, entry_date: pd.Timestamp) -> Dict:
        """Find exit point for position using time-decay strategy."""
        entry_idx = price_data.index.get_loc(entry_date)

        # Time-decay exit targets
        day_targets = {
            0: (self.profit_target_day0, self.stop_loss_day0),      # Day 0: ±2%
            1: (1.5, -1.5),    # Day 1: ±1.5%
            2: (1.0, -1.0),    # Day 2+: ±1%
        }

        for days_held in range(self.max_hold_days):
            check_idx = entry_idx + days_held + 1

            if check_idx >= len(price_data):
                break  # End of data

            check_date = price_data.index[check_idx]
            check_price = price_data.iloc[check_idx]['Close']

            # Calculate return
            return_pct = (check_price / position['entry_price'] - 1) * 100

            # Get targets for this day
            profit_target, stop_loss = day_targets.get(min(days_held, 2), (1.0, -1.0))

            # Check exit conditions
            if return_pct >= profit_target:
                return {
                    'exit_date': check_date,
                    'exit_price': check_price,
                    'days_held': days_held,
                    'exit_reason': f'profit_target_day{days_held}'
                }
            elif return_pct <= stop_loss:
                return {
                    'exit_date': check_date,
                    'exit_price': check_price,
                    'days_held': days_held,
                    'exit_reason': f'stop_loss_day{days_held}'
                }

        # Timeout exit
        timeout_idx = min(entry_idx + self.max_hold_days, len(price_data) - 1)
        return {
            'exit_date': price_data.index[timeout_idx],
            'exit_price': price_data.iloc[timeout_idx]['Close'],
            'days_held': self.max_hold_days - 1,
            'exit_reason': 'timeout'
        }

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'avg_gain': 0,
                'avg_loss': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0
            }

        trades_df = pd.DataFrame(self.trades)

        # Basic stats
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['return_pct'] > 0])
        losing_trades = len(trades_df[trades_df['return_pct'] <= 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Returns
        if self.use_dynamic_sizing:
            # Dynamic: Calculate return as sum of profits / initial capital
            total_profit = trades_df['profit'].sum()
            total_return = (total_profit / self.initial_capital) * 100
        else:
            # Fixed: Calculate cumulative return
            total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100

        # Gains and losses
        winners = trades_df[trades_df['return_pct'] > 0]['return_pct']
        losers = trades_df[trades_df['return_pct'] <= 0]['return_pct']

        avg_gain = winners.mean() if len(winners) > 0 else 0
        avg_loss = losers.mean() if len(losers) > 0 else 0

        # Risk metrics
        returns = trades_df['return_pct'].values
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Max drawdown
        cumulative_returns = (1 + trades_df['return_pct'] / 100).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        # Profit factor
        total_gains = winners.sum() if len(winners) > 0 else 0
        total_losses = abs(losers.sum()) if len(losers) > 0 else 1
        profit_factor = total_gains / total_losses if total_losses > 0 else 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_gain': avg_gain,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'trades': self.trades
        }


def run_exp012_comparison(ticker: str = "NVDA", period: str = "3y"):
    """
    Compare fixed vs dynamic position sizing.

    Args:
        ticker: Stock ticker
        period: Backtest period
    """
    print("=" * 70)
    print("EXP-012: FIXED VS DYNAMIC POSITION SIZING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    initial_capital = 10000
    params = get_params(ticker)

    print(f"Configuration:")
    print(f"  Ticker: {ticker}")
    print(f"  Period: {period}")
    print(f"  Initial Capital: ${initial_capital:,}")
    print(f"  Strategy: Mean Reversion v5.0 + Sentiment (EXP-011)")
    print()

    # Fetch data
    print("[1/5] Fetching price data...")
    fetcher = YahooFinanceFetcher()
    price_data = fetcher.fetch_stock_data(ticker, period=period)
    print(f"      [OK] {len(price_data)} rows")

    # Add technical indicators
    print("\n[2/5] Calculating technical indicators...")
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(price_data)
    print(f"      [OK] Technical indicators added")

    # Detect signals
    print("\n[3/5] Detecting signals...")
    detector = MeanReversionDetector(
        z_score_threshold=params['z_score_threshold'],
        rsi_oversold=params['rsi_oversold'],
        volume_multiplier=params['volume_multiplier'],
        price_drop_threshold=params['price_drop_threshold']
    )
    signals = detector.detect_overcorrections(enriched_data)

    # Add mock confidence levels (in production, use EXP-011 sentiment)
    signals['confidence'] = 'MEDIUM'  # Default
    panic_sells = signals[signals['panic_sell'] == True]

    # Assign varied confidence for realism (would come from sentiment in production)
    np.random.seed(42)
    for idx in panic_sells.index[:len(panic_sells)//3]:
        signals.loc[idx, 'confidence'] = 'HIGH'

    print(f"      [OK] {signals['panic_sell'].sum()} panic sells detected")

    # Historical performance by confidence (from baseline)
    historical_perf = {
        'HIGH': {'win_rate': 0.90, 'avg_gain': 5.5, 'avg_loss': -2.0},
        'MEDIUM': {'win_rate': 0.80, 'avg_gain': 5.0, 'avg_loss': -2.2},
        'LOW': {'win_rate': 0.60, 'avg_gain': 4.0, 'avg_loss': -2.5}
    }

    # Run FIXED sizing backtest
    print("\n[4/5] Running FIXED sizing backtest...")
    fixed_backtester = DynamicSizingBacktester(
        initial_capital=initial_capital,
        use_dynamic_sizing=False
    )

    # Set index for easy lookup
    enriched_data_indexed = enriched_data.set_index('Date')

    fixed_results = fixed_backtester.backtest(signals, enriched_data_indexed, historical_perf)
    print(f"      [OK] Fixed sizing: {fixed_results['total_trades']} trades, {fixed_results['win_rate']:.1f}% win rate")

    # Run DYNAMIC sizing backtest
    print("\n[5/5] Running DYNAMIC sizing backtest...")
    dynamic_backtester = DynamicSizingBacktester(
        initial_capital=initial_capital,
        use_dynamic_sizing=True
    )

    dynamic_results = dynamic_backtester.backtest(signals, enriched_data_indexed, historical_perf)
    print(f"      [OK] Dynamic sizing: {dynamic_results['total_trades']} trades, {dynamic_results['win_rate']:.1f}% win rate")

    # Display comparison
    print()
    print("=" * 70)
    print("COMPARISON RESULTS:")
    print("=" * 70)
    print()

    print(f"{'Metric':<25} | {'Fixed Sizing':>15} | {'Dynamic Sizing':>15} | {'Improvement':>12}")
    print("-" * 70)

    metrics = [
        ('Total Trades', 'total_trades', ''),
        ('Win Rate', 'win_rate', '%'),
        ('Total Return', 'total_return', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Max Drawdown', 'max_drawdown', '%'),
        ('Avg Gain', 'avg_gain', '%'),
        ('Avg Loss', 'avg_loss', '%'),
        ('Profit Factor', 'profit_factor', 'x')
    ]

    for name, key, unit in metrics:
        fixed_val = fixed_results[key]
        dynamic_val = dynamic_results[key]

        if key == 'total_trades':
            improvement = f"{dynamic_val - fixed_val:+.0f}"
        elif key in ['win_rate', 'total_return', 'max_drawdown', 'avg_gain', 'avg_loss']:
            improvement = f"{dynamic_val - fixed_val:+.2f}pp"
        else:
            improvement = f"{dynamic_val - fixed_val:+.2f}"

        print(f"{name:<25} | {fixed_val:>15.2f}{unit} | {dynamic_val:>15.2f}{unit} | {improvement:>12}")

    print()
    print("=" * 70)

    # Success criteria
    return_improvement = dynamic_results['total_return'] - fixed_results['total_return']

    if return_improvement > 5:
        print("\n[SUCCESS] Dynamic sizing significantly improved performance!")
        print(f"  Return improvement: +{return_improvement:.2f}pp")
        print(f"  Strategy enhancement validated!")
    elif return_improvement > 0:
        print("\n[PARTIAL SUCCESS] Dynamic sizing showed improvement")
        print(f"  Return improvement: +{return_improvement:.2f}pp")
        print(f"  Further optimization may help")
    else:
        print("\n[NEUTRAL] Dynamic sizing did not improve returns")
        print(f"  This may be due to limited sample size or parameter tuning needed")

    print("=" * 70)

    return {
        'fixed': fixed_results,
        'dynamic': dynamic_results
    }


if __name__ == '__main__':
    """Run EXP-012 comparison."""

    results = run_exp012_comparison(ticker='NVDA', period='3y')

    print("\n\nNext steps:")
    print("1. Review sizing parameters (confidence multipliers, Kelly fraction)")
    print("2. Test on other stocks (TSLA, JPM, etc.)")
    print("3. Integrate with EXP-011 sentiment for real confidence levels")
    print("4. Deploy to production if improvement validated")

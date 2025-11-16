"""
EXPERIMENT: EXP-013
Date: 2025-11-16
Objective: Capture momentum after panic sell reversals for higher returns

RESEARCH MOTIVATION:
Current strategy exits positions using time-decay (Day 0: ±2%, Day 1: ±1.5%, Day 2: ±1%).
This works well but may exit too early when panic reversals create momentum bounces.

Research shows:
- Panic sells overshoot (mean reversion) - WE CAPTURE THIS ✓
- Reversal creates short-term momentum (2-5 days) - WE MISS THIS ✗
- Momentum bounces continue +2-4% beyond initial reversion
- Trailing stops can capture full move while protecting gains

HYPOTHESIS:
Switching from time-decay exits to momentum trailing stops when bounce conditions met
will increase average gain from 6.35% to 8-10%+ without reducing win rate.

METHOD:
BASELINE (Current v5.0):
  - Entry: Panic sell detection
  - Exit: Time-decay targets (±2% → ±1.5% → ±1%)
  - Avg gain: 6.35%

EXP-013 (Momentum Hybrid):
  - Entry: Same panic sell detection
  - Phase 1: Mean reversion (days 0-1)
  - Phase 2: Momentum detection
    * If bounce > 1.5% AND momentum confirmed
    * Switch to trailing stop mode
  - Exit: Momentum trailing stop (1% below recent high)
  - Expected avg gain: 8-10%

MOMENTUM DETECTION:
1. Price momentum: Fast EMA > Slow EMA
2. Volume confirmation: Volume > 1.2x average
3. Price acceleration: ROC > 2%
4. Bounce requirement: Minimum 1.5% from entry

EXPECTED OUTCOMES:
- Win rate: Same (87.5%) - not changing entry
- Avg gain: +30-50% improvement (6.35% → 9%)
- Total return: +20-25pp improvement
- Hold time: Slightly longer (capture full bounce)
- Risk: Same or better (trailing stops protect)

COMPARISON METRICS:
- Total return: Baseline vs Momentum
- Avg gain per trade
- Hold time distribution
- Momentum detection rate
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.models.trading.mean_reversion import MeanReversionDetector
from src.config.mean_reversion_params import get_params
from src.trading.exits.momentum_detector import DynamicExitManager


class MomentumCaptureBacktester:
    """Backtest mean reversion with momentum-aware exits."""

    def __init__(self,
                 initial_capital: float = 10000,
                 use_momentum_exits: bool = True,
                 max_hold_days: int = 7):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            use_momentum_exits: Use momentum trailing stops (True) or time-decay only (False)
            max_hold_days: Maximum days to hold position
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.use_momentum_exits = use_momentum_exits
        self.max_hold_days = max_hold_days

        # Exit manager
        self.exit_manager = DynamicExitManager(
            profit_target_day0=2.0,
            stop_loss_day0=-2.0,
            enable_momentum_mode=use_momentum_exits
        )

        # Tracking
        self.trades = []

    def backtest(self, signals_df: pd.DataFrame, price_data: pd.DataFrame) -> Dict:
        """
        Run backtest with momentum-aware exits.

        Args:
            signals_df: DataFrame with panic_sell signals
            price_data: Full OHLCV data

        Returns:
            Dictionary with backtest results
        """
        # Reset state
        self.current_capital = self.initial_capital
        self.trades = []

        # Get panic sell signals
        panic_sell_dates = signals_df[signals_df['panic_sell'] == True].copy()

        print(f"[INFO] Backtesting {len(panic_sell_dates)} panic sell signals...")
        print(f"[INFO] Exit mode: {'MOMENTUM-AWARE' if self.use_momentum_exits else 'TIME-DECAY ONLY'}")

        # Set index for easy lookup
        price_data_indexed = price_data.set_index('Date')

        for idx, signal_row in panic_sell_dates.iterrows():
            entry_date = signal_row['Date']
            entry_price = signal_row['Close']

            # Get entry index
            try:
                entry_idx = price_data_indexed.index.get_loc(entry_date)
            except KeyError:
                continue

            # Find exit
            exit_info = self._find_exit(price_data_indexed, entry_price, entry_idx)

            if exit_info:
                # Calculate position size (100% of capital)
                shares = int(self.current_capital / entry_price)
                position_value = shares * entry_price

                # Calculate P&L
                profit = (exit_info['exit_price'] - entry_price) * shares
                return_pct = ((exit_info['exit_price'] / entry_price) - 1) * 100

                # Update capital
                self.current_capital += profit

                trade = {
                    'entry_date': entry_date,
                    'exit_date': exit_info['exit_date'],
                    'entry_price': entry_price,
                    'exit_price': exit_info['exit_price'],
                    'shares': shares,
                    'position_value': position_value,
                    'profit': profit,
                    'return_pct': return_pct,
                    'days_held': exit_info['days_held'],
                    'exit_reason': exit_info['exit_reason'],
                    'exit_mode': exit_info.get('exit_mode', 'time_decay'),
                    'bounce_captured': exit_info.get('bounce_captured', 0)
                }

                self.trades.append(trade)

        # Calculate metrics
        return self._calculate_metrics()

    def _find_exit(self, price_data: pd.DataFrame, entry_price: float, entry_idx: int) -> Dict:
        """Find exit point using dynamic exit manager."""

        for days_held in range(self.max_hold_days):
            current_idx = entry_idx + days_held + 1

            if current_idx >= len(price_data):
                break  # End of data

            exit_signal = self.exit_manager.check_exit(
                price_data.reset_index(),
                entry_price,
                entry_idx,
                current_idx,
                days_held
            )

            if exit_signal and exit_signal['should_exit']:
                return {
                    'exit_date': price_data.index[current_idx],
                    'exit_price': exit_signal['exit_price'],
                    'days_held': days_held,
                    'exit_reason': exit_signal['exit_reason'],
                    'exit_mode': exit_signal['exit_mode'],
                    'bounce_captured': exit_signal.get('bounce_captured', 0)
                }

        # Timeout exit
        timeout_idx = min(entry_idx + self.max_hold_days, len(price_data) - 1)
        return {
            'exit_date': price_data.index[timeout_idx],
            'exit_price': price_data.iloc[timeout_idx]['Close'],
            'days_held': self.max_hold_days - 1,
            'exit_reason': 'timeout',
            'exit_mode': 'time_decay',
            'bounce_captured': 0
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
                'avg_hold_days': 0,
                'momentum_exits': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }

        trades_df = pd.DataFrame(self.trades)

        # Basic stats
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['return_pct'] > 0])
        losing_trades = len(trades_df[trades_df['return_pct'] <= 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Returns
        total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100

        # Gains and losses
        winners = trades_df[trades_df['return_pct'] > 0]['return_pct']
        losers = trades_df[trades_df['return_pct'] <= 0]['return_pct']

        avg_gain = winners.mean() if len(winners) > 0 else 0
        avg_loss = losers.mean() if len(losers) > 0 else 0

        # Hold time
        avg_hold_days = trades_df['days_held'].mean()

        # Momentum stats
        momentum_exits = len(trades_df[trades_df['exit_mode'] == 'momentum_trailing'])

        # Risk metrics
        returns = trades_df['return_pct'].values
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Max drawdown
        cumulative_returns = (1 + trades_df['return_pct'] / 100).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_gain': avg_gain,
            'avg_loss': avg_loss,
            'avg_hold_days': avg_hold_days,
            'momentum_exits': momentum_exits,
            'momentum_exit_rate': (momentum_exits / total_trades * 100) if total_trades > 0 else 0,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': self.trades
        }


def run_exp013_comparison(ticker: str = "NVDA", period: str = "3y"):
    """
    Compare time-decay vs momentum-aware exits.

    Args:
        ticker: Stock ticker
        period: Backtest period
    """
    print("=" * 70)
    print("EXP-013: MOMENTUM CAPTURE COMPARISON")
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
    print(f"  Strategy: Mean Reversion v5.0")
    print()

    # Fetch data
    print("[1/4] Fetching price data...")
    fetcher = YahooFinanceFetcher()
    price_data = fetcher.fetch_stock_data(ticker, period=period)
    print(f"      [OK] {len(price_data)} rows")

    # Add technical indicators
    print("\n[2/4] Calculating technical indicators...")
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(price_data)
    print(f"      [OK] Technical indicators added")

    # Detect signals
    print("\n[3/4] Detecting signals...")
    detector = MeanReversionDetector(
        z_score_threshold=params['z_score_threshold'],
        rsi_oversold=params['rsi_oversold'],
        volume_multiplier=params['volume_multiplier'],
        price_drop_threshold=params['price_drop_threshold']
    )
    signals = detector.detect_overcorrections(enriched_data)
    print(f"      [OK] {signals['panic_sell'].sum()} panic sells detected")

    # Run TIME-DECAY backtest
    print("\n[4/4] Running backtests...")
    print()
    print("  [A] TIME-DECAY EXITS (Baseline v5.0)...")
    baseline_backtester = MomentumCaptureBacktester(
        initial_capital=initial_capital,
        use_momentum_exits=False
    )
    baseline_results = baseline_backtester.backtest(signals, enriched_data)
    print(f"      [OK] {baseline_results['total_trades']} trades, {baseline_results['avg_gain']:.2f}% avg gain")

    # Run MOMENTUM backtest
    print()
    print("  [B] MOMENTUM-AWARE EXITS (EXP-013)...")
    momentum_backtester = MomentumCaptureBacktester(
        initial_capital=initial_capital,
        use_momentum_exits=True
    )
    momentum_results = momentum_backtester.backtest(signals, enriched_data)
    print(f"      [OK] {momentum_results['total_trades']} trades, {momentum_results['avg_gain']:.2f}% avg gain")

    # Display comparison
    print()
    print("=" * 70)
    print("COMPARISON RESULTS:")
    print("=" * 70)
    print()

    print(f"{'Metric':<30} | {'Time-Decay':>15} | {'Momentum':>15} | {'Improvement':>12}")
    print("-" * 75)

    metrics = [
        ('Total Trades', 'total_trades', ''),
        ('Win Rate', 'win_rate', '%'),
        ('Total Return', 'total_return', '%'),
        ('Avg Gain (Winners)', 'avg_gain', '%'),
        ('Avg Loss (Losers)', 'avg_loss', '%'),
        ('Avg Hold Days', 'avg_hold_days', ' days'),
        ('Momentum Exits', 'momentum_exits', ''),
        ('Momentum Exit Rate', 'momentum_exit_rate', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Max Drawdown', 'max_drawdown', '%')
    ]

    for name, key, unit in metrics:
        baseline_val = baseline_results.get(key, 0)
        momentum_val = momentum_results.get(key, 0)

        if key in ['total_trades', 'winning_trades', 'momentum_exits']:
            improvement = f"{momentum_val - baseline_val:+.0f}"
        elif key in ['win_rate', 'total_return', 'avg_gain', 'avg_loss', 'max_drawdown', 'momentum_exit_rate']:
            improvement = f"{momentum_val - baseline_val:+.2f}pp"
        elif key == 'avg_hold_days':
            improvement = f"{momentum_val - baseline_val:+.1f}"
        else:
            improvement = f"{momentum_val - baseline_val:+.2f}"

        print(f"{name:<30} | {baseline_val:>15.2f}{unit} | {momentum_val:>15.2f}{unit} | {improvement:>12}")

    print()
    print("=" * 70)

    # Success analysis
    return_improvement = momentum_results['total_return'] - baseline_results['total_return']
    gain_improvement_pct = ((momentum_results['avg_gain'] / baseline_results['avg_gain']) - 1) * 100 if baseline_results['avg_gain'] > 0 else 0

    print()
    if return_improvement > 10:
        print("[SUCCESS] Momentum capture significantly improved performance!")
        print(f"  Return improvement: +{return_improvement:.2f}pp")
        print(f"  Avg gain improvement: +{gain_improvement_pct:.1f}%")
        print(f"  Momentum exits: {momentum_results['momentum_exit_rate']:.1f}% of trades")
        print()
        print("  RECOMMENDATION: Deploy momentum-aware exits to production!")
    elif return_improvement > 0:
        print("[PARTIAL SUCCESS] Momentum capture showed improvement")
        print(f"  Return improvement: +{return_improvement:.2f}pp")
        print(f"  May benefit from parameter tuning")
    else:
        print("[NEUTRAL] Momentum capture did not improve returns")
        print(f"  Time-decay exits may already be optimal for this stock")
        print(f"  Consider testing on higher momentum stocks (TSLA, etc.)")

    print("=" * 70)

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    results = {
        'experiment_id': 'EXP-013',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ticker': ticker,
        'period': period,
        'baseline': {k: v for k, v in baseline_results.items() if k != 'trades'},
        'momentum': {k: v for k, v in momentum_results.items() if k != 'trades'},
        'improvement': {
            'return_pp': return_improvement,
            'avg_gain_pct': gain_improvement_pct,
            'momentum_exit_rate': momentum_results['momentum_exit_rate']
        }
    }

    results_file = os.path.join(results_dir, f'exp013_{ticker.lower()}_momentum_capture.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    return results


if __name__ == '__main__':
    """Run EXP-013 comparison."""

    results = run_exp013_comparison(ticker='TSLA', period='3y')

    print("\n\nNext steps:")
    print("1. Test on high-momentum stocks (TSLA, TSLA)")
    print("2. Tune momentum detection parameters")
    print("3. Analyze momentum vs time-decay trade-by-trade")
    print("4. Deploy if improvement validated across multiple stocks")

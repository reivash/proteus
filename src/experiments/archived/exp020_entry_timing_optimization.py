"""
EXPERIMENT: EXP-020
Date: 2025-11-16
Objective: Optimize entry timing after panic sell detection

RESEARCH MOTIVATION:
Current strategy enters at CLOSE on panic sell detection day.
But panic sells often gap down at open, then recover intraday.

Questions:
- Should we buy at close (current) or next day open?
- Can we get better entry by waiting for intraday dip?
- Does immediate entry vs delayed entry affect returns?

EXP-013 showed avg hold time is 0.5-0.8 days (exits on Day 0-1).
This means entry timing is CRITICAL - even small improvements matter.

HYPOTHESIS:
Buying at NEXT DAY OPEN instead of PANIC DAY CLOSE improves returns because:
- Panic often continues overnight (gap down at open)
- Opens provide better entry prices
- Captures full reversal from lower price

Expected improvement: +1-3% avg gain per trade (+15-25% improvement)

METHOD:
Test 3 entry timing strategies:

STRATEGY A (Baseline): PANIC DAY CLOSE
- Detect panic sell at market close
- Enter immediately at closing price
- Current approach

STRATEGY B: NEXT DAY OPEN
- Detect panic sell at close
- Wait overnight
- Enter at next day opening price
- Hypothesis: Better entry if gap down continues

STRATEGY C: NEXT DAY LOW
- Detect panic sell at close
- Wait for next day
- Enter at next day low (best possible entry)
- Theoretical maximum (not executable in practice)

COMPARISON METRICS:
- Entry price vs panic close
- Avg gain per trade
- Total return
- Win rate (should stay same)
- Risk-adjusted returns

EXPECTED OUTCOMES:
- Next day open: +1-2% better entry price
- Next day low: +2-4% better entry (theoretical max)
- Total return improvement: +15-25%
- Same win rate (timing doesn't affect reversal)
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


class EntryTimingBacktester:
    """Backtest different entry timing strategies."""

    def __init__(self,
                 initial_capital: float = 10000,
                 entry_timing: str = 'close'):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            entry_timing: 'close' (panic day close), 'open' (next day open), or 'low' (next day low)
        """
        self.initial_capital = initial_capital
        self.entry_timing = entry_timing

        # Time-decay exit targets (same as v6.0)
        self.time_decay_targets = {
            0: (2.0, -2.0),   # Day 0: ±2%
            1: (1.5, -1.5),   # Day 1: ±1.5%
            2: (1.0, -1.0),   # Day 2+: ±1%
        }
        self.max_hold_days = 7

    def backtest(self, signals_df: pd.DataFrame, price_data: pd.DataFrame) -> Dict:
        """
        Run backtest with specified entry timing.

        Args:
            signals_df: DataFrame with panic_sell signals
            price_data: Full OHLCV data

        Returns:
            Dictionary with backtest results
        """
        capital = self.initial_capital
        trades = []

        # Get panic sell signals
        panic_sells = signals_df[signals_df['panic_sell'] == True].copy()
        price_data_indexed = price_data.set_index('Date')

        for idx, signal_row in panic_sells.iterrows():
            panic_date = signal_row['Date']
            panic_close = signal_row['Close']

            try:
                panic_idx = price_data_indexed.index.get_loc(panic_date)
            except KeyError:
                continue

            # Determine entry price based on timing strategy
            if self.entry_timing == 'close':
                # STRATEGY A: Enter at panic day close (baseline)
                entry_idx = panic_idx
                entry_price = panic_close
                entry_date = panic_date

            elif self.entry_timing == 'open':
                # STRATEGY B: Enter at next day open
                entry_idx = panic_idx + 1
                if entry_idx >= len(price_data_indexed):
                    continue  # No next day data

                entry_date = price_data_indexed.index[entry_idx]
                entry_price = price_data_indexed.iloc[entry_idx]['Open']

            elif self.entry_timing == 'low':
                # STRATEGY C: Enter at next day low (theoretical best)
                entry_idx = panic_idx + 1
                if entry_idx >= len(price_data_indexed):
                    continue

                entry_date = price_data_indexed.index[entry_idx]
                entry_price = price_data_indexed.iloc[entry_idx]['Low']

            else:
                raise ValueError(f"Invalid entry_timing: {self.entry_timing}")

            # Find exit using time-decay targets
            for days_held in range(self.max_hold_days):
                current_idx = entry_idx + days_held + 1

                if current_idx >= len(price_data_indexed):
                    break

                current_price = price_data_indexed.iloc[current_idx]['Close']
                return_pct = ((current_price / entry_price) - 1) * 100

                # Check time-decay targets
                profit_target, stop_loss = self.time_decay_targets.get(
                    min(days_held, 2),
                    (1.0, -1.0)
                )

                if return_pct >= profit_target or return_pct <= stop_loss:
                    # Exit trade
                    shares = int(capital / entry_price)
                    profit = (current_price - entry_price) * shares
                    capital += profit

                    trades.append({
                        'panic_date': panic_date,
                        'panic_close': panic_close,
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'entry_improvement_pct': ((entry_price / panic_close) - 1) * 100,
                        'exit_date': price_data_indexed.index[current_idx],
                        'exit_price': current_price,
                        'return_pct': return_pct,
                        'days_held': days_held,
                        'exit_reason': 'profit' if return_pct > 0 else 'loss'
                    })
                    break
            else:
                # Timeout exit
                timeout_idx = min(entry_idx + self.max_hold_days, len(price_data_indexed) - 1)
                current_price = price_data_indexed.iloc[timeout_idx]['Close']
                return_pct = ((current_price / entry_price) - 1) * 100

                shares = int(capital / entry_price)
                profit = (current_price - entry_price) * shares
                capital += profit

                trades.append({
                    'panic_date': panic_date,
                    'panic_close': panic_close,
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'entry_improvement_pct': ((entry_price / panic_close) - 1) * 100,
                    'exit_date': price_data_indexed.index[timeout_idx],
                    'exit_price': current_price,
                    'return_pct': return_pct,
                    'days_held': self.max_hold_days - 1,
                    'exit_reason': 'timeout'
                })

        # Calculate metrics
        return self._calculate_metrics(trades, capital)

    def _calculate_metrics(self, trades: List[Dict], final_capital: float) -> Dict:
        """Calculate performance metrics from trades."""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'avg_gain': 0,
                'avg_loss': 0,
                'avg_entry_improvement': 0,
                'avg_hold_days': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }

        trades_df = pd.DataFrame(trades)

        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['return_pct'] > 0])
        losing_trades = len(trades_df[trades_df['return_pct'] <= 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100

        winners = trades_df[trades_df['return_pct'] > 0]['return_pct']
        losers = trades_df[trades_df['return_pct'] <= 0]['return_pct']

        avg_gain = winners.mean() if len(winners) > 0 else 0
        avg_loss = losers.mean() if len(losers) > 0 else 0
        avg_entry_improvement = trades_df['entry_improvement_pct'].mean()
        avg_hold_days = trades_df['days_held'].mean()

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
            'avg_entry_improvement': avg_entry_improvement,
            'avg_hold_days': avg_hold_days,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }


def run_exp020_entry_timing(ticker: str = "NVDA", period: str = "3y"):
    """
    Compare entry timing strategies.

    Args:
        ticker: Stock ticker (Tier A)
        period: Backtest period
    """
    print("=" * 70)
    print("EXP-020: ENTRY TIMING OPTIMIZATION")
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
    print(f"  Strategy: Mean Reversion v6.0")
    print()

    # Fetch price data
    print("[1/4] Fetching price data...")
    fetcher = YahooFinanceFetcher()
    price_data = fetcher.fetch_stock_data(ticker, period=period)
    print(f"      [OK] {len(price_data)} rows")

    # Add technical indicators
    print("\n[2/4] Calculating technical indicators...")
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(price_data)
    print(f"      [OK] Technical indicators added")

    # Detect panic sell signals
    print("\n[3/4] Detecting panic sell signals...")
    detector = MeanReversionDetector(
        z_score_threshold=params['z_score_threshold'],
        rsi_oversold=params['rsi_oversold'],
        volume_multiplier=params['volume_multiplier'],
        price_drop_threshold=params['price_drop_threshold']
    )
    signals = detector.detect_overcorrections(enriched_data)
    num_signals = signals['panic_sell'].sum()
    print(f"      [OK] {num_signals} panic sell signals detected")

    # Run backtests
    print("\n[4/4] Running backtests...")
    print()

    # STRATEGY A: Panic day close (baseline)
    print("  [A] PANIC DAY CLOSE (Baseline)...")
    close_backtester = EntryTimingBacktester(initial_capital, entry_timing='close')
    close_results = close_backtester.backtest(signals, enriched_data)
    print(f"      [OK] {close_results['total_trades']} trades, {close_results['avg_gain']:.2f}% avg gain")

    # STRATEGY B: Next day open
    print("\n  [B] NEXT DAY OPEN...")
    open_backtester = EntryTimingBacktester(initial_capital, entry_timing='open')
    open_results = open_backtester.backtest(signals, enriched_data)
    print(f"      [OK] {open_results['total_trades']} trades, {open_results['avg_gain']:.2f}% avg gain")
    print(f"      Entry improvement: {open_results['avg_entry_improvement']:+.2f}% vs panic close")

    # STRATEGY C: Next day low (theoretical max)
    print("\n  [C] NEXT DAY LOW (Theoretical Best)...")
    low_backtester = EntryTimingBacktester(initial_capital, entry_timing='low')
    low_results = low_backtester.backtest(signals, enriched_data)
    print(f"      [OK] {low_results['total_trades']} trades, {low_results['avg_gain']:.2f}% avg gain")
    print(f"      Entry improvement: {low_results['avg_entry_improvement']:+.2f}% vs panic close")

    # Display comparison
    print()
    print("=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print()

    print(f"{'Metric':<30} | {'Panic Close':>15} | {'Next Open':>15} | {'Next Low':>15}")
    print("-" * 80)

    metrics = [
        ('Total Trades', 'total_trades', ''),
        ('Win Rate', 'win_rate', '%'),
        ('Entry Improvement', 'avg_entry_improvement', '%'),
        ('Total Return', 'total_return', '%'),
        ('Avg Gain (Winners)', 'avg_gain', '%'),
        ('Avg Loss (Losers)', 'avg_loss', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Max Drawdown', 'max_drawdown', '%')
    ]

    for name, key, unit in metrics:
        close_val = close_results.get(key, 0)
        open_val = open_results.get(key, 0)
        low_val = low_results.get(key, 0)

        print(f"{name:<30} | {close_val:>15.2f}{unit} | {open_val:>15.2f}{unit} | {low_val:>15.2f}{unit}")

    print()
    print("=" * 70)

    # Analysis
    return_improvement = open_results['total_return'] - close_results['total_return']
    entry_improvement = open_results['avg_entry_improvement']

    print()
    if return_improvement > 5:
        print("[SUCCESS] Next day open significantly improved returns!")
        print(f"  Entry price improvement: {entry_improvement:+.2f}%")
        print(f"  Return improvement: {return_improvement:+.2f}pp")
        print()
        print("  RECOMMENDATION: Change entry timing to next day open!")
    elif return_improvement > 0:
        print("[PARTIAL SUCCESS] Next day open showed improvement")
        print(f"  Entry price improvement: {entry_improvement:+.2f}%")
        print(f"  Return improvement: {return_improvement:+.2f}pp")
        print(f"  Consider deploying based on risk tolerance")
    else:
        print("[NEUTRAL] Panic day close remains optimal")
        print(f"  Possible reasons:")
        print(f"  - Next day gaps UP (not down)")
        print(f"  - Immediate entry captures full reversal")
        print(f"  - Waiting introduces execution risk")

    print("=" * 70)

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    results_data = {
        'experiment_id': 'EXP-020',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ticker': ticker,
        'period': period,
        'panic_close': {k: v for k, v in close_results.items()},
        'next_open': {k: v for k, v in open_results.items()},
        'next_low': {k: v for k, v in low_results.items()},
        'improvement': {
            'return_pp': return_improvement,
            'entry_improvement_pct': entry_improvement
        }
    }

    results_file = os.path.join(results_dir, f'exp020_{ticker.lower()}_entry_timing.json')
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    return results_data


if __name__ == '__main__':
    """Run EXP-020 entry timing optimization."""

    # Test on NVDA (best Tier A performer)
    results = run_exp020_entry_timing(ticker='NVDA', period='3y')

    print("\n\nNext steps:")
    print("1. Test on V and MA (other Tier A stocks)")
    print("2. If next day open wins: Update scanner to enter at open")
    print("3. Deploy if improvement validated across all Tier A")
    print("4. Consider limit orders at specific prices")

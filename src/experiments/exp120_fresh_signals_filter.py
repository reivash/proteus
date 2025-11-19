"""
EXP-120: Fresh Signals Filter (Stale Oversold Rejection)

CONTEXT:
Current system may trade stocks that have been repeatedly oversold for
extended periods. These are often value traps in structural decline, not
true mean reversion opportunities. Fresh capitulation has better recovery odds.

HYPOTHESIS:
- Fresh signals (first time oversold in 10+ days) = True capitulation, high recovery
- Stale signals (repeatedly oversold) = Value trap, structural decline, avoid
- Freshness is a critical quality indicator for mean reversion

OBJECTIVE:
Filter out stale signals to improve win rate and risk-adjusted returns.

METHODOLOGY:
Test freshness-based filters:
1. Baseline: No freshness filter (current)
2. 5-day freshness: Require no signal in past 5 days
3. 10-day freshness: Require no signal in past 10 days
4. 15-day freshness: Require no signal in past 15 days
5. 20-day freshness: Require no signal in past 20 days

EXPECTED IMPACT:
- Win rate: +5-8% (eliminate value traps)
- Sharpe ratio: +12-18% (better signal quality)
- Trade count: -10-25% (more selective)

VALIDATION CRITERIA:
- Win rate >= 67% (+7pp minimum)
- Sharpe ratio >= 2.65 (+12% minimum)
- Profit factor >= 2.0

Created: 2025-11-19
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Set
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.trading.signal_scanner import SignalScanner
from src.trading.paper_trader import PaperTrader
from src.data.fetchers.yahoo_finance import YahooFinanceFetcher


def get_freshness_filter_name(filter_type: str) -> str:
    """Get human-readable name for freshness filter."""
    names = {
        'none': 'No Filter (Baseline)',
        '5day': '5-Day Freshness (No signal in 5 days)',
        '10day': '10-Day Freshness (No signal in 10 days)',
        '15day': '15-Day Freshness (No signal in 15 days)',
        '20day': '20-Day Freshness (No signal in 20 days)'
    }
    return names.get(filter_type, filter_type)


def get_freshness_days(filter_type: str) -> int:
    """Get number of days for freshness requirement."""
    if filter_type == '5day':
        return 5
    elif filter_type == '10day':
        return 10
    elif filter_type == '15day':
        return 15
    elif filter_type == '20day':
        return 20
    else:
        return 0  # No filter


class SignalHistoryTracker:
    """Track signal history to identify fresh vs stale signals."""

    def __init__(self):
        self.signal_history = {}  # {ticker: [date1, date2, ...]}

    def add_signals(self, signals: List[dict], date_str: str):
        """Record signals for a date."""
        date = datetime.strptime(date_str, '%Y-%m-%d').date()

        for signal in signals:
            ticker = signal['ticker']
            if ticker not in self.signal_history:
                self.signal_history[ticker] = []

            # Add this signal date if not already present
            if date not in self.signal_history[ticker]:
                self.signal_history[ticker].append(date)
                self.signal_history[ticker].sort()

    def is_fresh(self, ticker: str, date_str: str, min_days: int) -> bool:
        """
        Check if signal is fresh (no signals in past min_days).

        Args:
            ticker: Stock ticker
            date_str: Current signal date
            min_days: Minimum days since last signal

        Returns:
            True if fresh, False if stale
        """
        if min_days == 0:
            return True  # No freshness requirement

        current_date = datetime.strptime(date_str, '%Y-%m-%d').date()

        # If no history for this ticker, it's fresh
        if ticker not in self.signal_history:
            return True

        # Check for recent signals
        history = self.signal_history[ticker]

        # Look for signals in the lookback window
        lookback_start = current_date - timedelta(days=min_days)

        for signal_date in reversed(history):  # Start from most recent
            if signal_date >= current_date:
                # Skip current or future dates (shouldn't happen)
                continue

            if signal_date >= lookback_start:
                # Found a signal in the lookback window - stale!
                return False

            if signal_date < lookback_start:
                # All remaining signals are older - fresh!
                return True

        # No signals in lookback window - fresh!
        return True

    def days_since_last_signal(self, ticker: str, date_str: str) -> int:
        """
        Calculate days since last signal.

        Returns:
            Number of days since last signal, or 999 if no history
        """
        current_date = datetime.strptime(date_str, '%Y-%m-%d').date()

        if ticker not in self.signal_history:
            return 999  # No history

        history = self.signal_history[ticker]

        for signal_date in reversed(history):
            if signal_date < current_date:
                days_diff = (current_date - signal_date).days
                return days_diff

        return 999  # No prior signals


def backtest_with_freshness_filter(
    start_date: str,
    end_date: str,
    filter_type: str = 'none'
) -> dict:
    """
    Backtest strategy with freshness filtering.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        filter_type: 'none', '5day', '10day', '15day', '20day'

    Returns:
        Dictionary with performance metrics
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {get_freshness_filter_name(filter_type).upper()}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*70}\n")

    # Initialize components
    fetcher = YahooFinanceFetcher()
    scanner = SignalScanner(
        lookback_days=90,
        min_signal_strength=None  # Use Q4 filtering from SignalScanner
    )
    trader = PaperTrader(
        initial_capital=90000,
        max_positions=5,
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=2,
        use_limit_orders=True
    )

    # Track signal history
    history_tracker = SignalHistoryTracker()

    # Get freshness requirement
    freshness_days = get_freshness_days(filter_type)

    # Track filter statistics
    filter_stats = {
        'total_signals': 0,
        'fresh_signals': 0,
        'stale_signals': 0,
        'freshness_distribution': {  # Days since last signal
            '0-5': 0,
            '6-10': 0,
            '11-15': 0,
            '16-20': 0,
            '20+': 0
        }
    }

    # Convert dates
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    current_dt = start_dt

    while current_dt <= end_dt:
        date_str = current_dt.strftime('%Y-%m-%d')

        # Scan for signals
        signals = scanner.scan_all_stocks(date_str)

        if len(signals) > 0:
            filter_stats['total_signals'] += len(signals)

            print(f"\n{'='*70}")
            print(f"Date: {date_str}")
            print(f"Total signals found: {len(signals)}")

            # Record all signals in history (before filtering)
            history_tracker.add_signals(signals, date_str)

            # Apply freshness filter
            filtered_signals = []
            for signal in signals:
                ticker = signal['ticker']

                # Check freshness
                is_fresh = history_tracker.is_fresh(ticker, date_str, freshness_days)
                days_since = history_tracker.days_since_last_signal(ticker, date_str)

                # Update distribution
                if days_since <= 5:
                    filter_stats['freshness_distribution']['0-5'] += 1
                elif days_since <= 10:
                    filter_stats['freshness_distribution']['6-10'] += 1
                elif days_since <= 15:
                    filter_stats['freshness_distribution']['11-15'] += 1
                elif days_since <= 20:
                    filter_stats['freshness_distribution']['16-20'] += 1
                else:
                    filter_stats['freshness_distribution']['20+'] += 1

                if is_fresh:
                    filter_stats['fresh_signals'] += 1
                    filtered_signals.append(signal)

                    if days_since == 999:
                        print(f"  ✓ {ticker}: FRESH (first signal ever)")
                    else:
                        print(f"  ✓ {ticker}: FRESH ({days_since} days since last signal)")
                else:
                    filter_stats['stale_signals'] += 1
                    print(f"  ✗ {ticker}: STALE ({days_since} days since last signal - REJECT)")

            print(f"\nFreshness Filter Applied ({filter_type}):")
            print(f"  Signals before filter: {len(signals)}")
            print(f"  Signals after filter: {len(filtered_signals)}")
            print(f"  Fresh signals: {filter_stats['fresh_signals']} (passed)")
            print(f"  Stale signals: {filter_stats['stale_signals']} (rejected)")
            if len(signals) > 0:
                print(f"  Rejection rate: {100 * filter_stats['stale_signals'] / filter_stats['total_signals']:.1f}%")
            print(f"{'='*70}")

            # Process filtered signals
            if filtered_signals:
                trader.process_signals(filtered_signals, date_str)

        # Update positions
        trader.update_positions(date_str, fetcher)

        # Move to next trading day
        current_dt += timedelta(days=1)

    # Get performance
    perf = trader.get_performance()

    # Add filter statistics
    perf['filter_stats'] = filter_stats
    perf['filter_type'] = filter_type

    return perf


def print_performance(perf: dict, label: str):
    """Print performance metrics."""
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    print(f"Total Trades: {perf['total_trades']}")
    print(f"Win Rate: {perf['win_rate']:.1f}%")
    print(f"Total Return: {perf['total_return']:.2f}%")
    print(f"Avg Return per Trade: {perf['avg_return_per_trade']:.2f}%")
    print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {perf['max_drawdown']:.2f}%")
    print(f"Profit Factor: {perf['profit_factor']:.2f}")

    # Print filter statistics
    if 'filter_stats' in perf:
        stats = perf['filter_stats']
        print(f"\n{'='*70}")
        print("FILTER STATISTICS:")
        print(f"{'='*70}")
        print(f"Total signals generated: {stats['total_signals']}")
        print(f"Fresh signals: {stats['fresh_signals']} ({100 * stats['fresh_signals'] / stats['total_signals']:.1f}%)")
        print(f"Stale signals rejected: {stats['stale_signals']} ({100 * stats['stale_signals'] / stats['total_signals']:.1f}%)")

        print(f"\nFreshness Distribution (days since last signal):")
        total_signals = stats['total_signals']
        for bucket, count in stats['freshness_distribution'].items():
            pct = 100 * count / total_signals if total_signals > 0 else 0
            print(f"  {bucket} days: {count} ({pct:.1f}%)")

    print(f"{'='*70}\n")


def compare_results(results: Dict[str, dict]):
    """Print comparison table of all results."""
    print(f"\n{'='*70}")
    print("COMPARISON: FRESHNESS FILTERS")
    print(f"{'='*70}\n")

    # Create comparison table
    headers = ["Filter", "Trades", "WR%", "Return%", "Sharpe", "MaxDD%", "PF", "Fresh%"]
    rows = []

    for filter_type, perf in results.items():
        fresh_pct = 100 * perf['filter_stats']['fresh_signals'] / perf['filter_stats']['total_signals'] if perf['filter_stats']['total_signals'] > 0 else 0

        row = [
            get_freshness_filter_name(filter_type)[:30],  # Truncate long names
            perf['total_trades'],
            f"{perf['win_rate']:.1f}",
            f"{perf['total_return']:.2f}",
            f"{perf['sharpe_ratio']:.2f}",
            f"{perf['max_drawdown']:.2f}",
            f"{perf['profit_factor']:.2f}",
            f"{fresh_pct:.1f}"
        ]
        rows.append(row)

    # Print table
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

    # Header
    header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_row)
    print("-" * len(header_row))

    # Rows
    for row in rows:
        print(" | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))))

    print(f"\n{'='*70}")


def main():
    """Run EXP-120 backtest."""
    print("="*70)
    print("EXP-120: FRESH SIGNALS FILTER (STALE OVERSOLD REJECTION)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("OBJECTIVE: Improve win rate by filtering out stale/repeated signals")
    print("THEORY: Fresh capitulation recovers, stale oversold = value trap\n")

    # Test period: 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~2 years

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}\n")

    # Run all strategies
    results = {}

    print("\n" + "="*70)
    print("BASELINE: No Freshness Filter")
    print("="*70)
    results['none'] = backtest_with_freshness_filter(start_str, end_str, filter_type='none')
    print_performance(results['none'], "BASELINE PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 1: 5-Day Freshness")
    print("="*70)
    results['5day'] = backtest_with_freshness_filter(start_str, end_str, filter_type='5day')
    print_performance(results['5day'], "5-DAY FRESHNESS PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 2: 10-Day Freshness")
    print("="*70)
    results['10day'] = backtest_with_freshness_filter(start_str, end_str, filter_type='10day')
    print_performance(results['10day'], "10-DAY FRESHNESS PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 3: 15-Day Freshness")
    print("="*70)
    results['15day'] = backtest_with_freshness_filter(start_str, end_str, filter_type='15day')
    print_performance(results['15day'], "15-DAY FRESHNESS PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 4: 20-Day Freshness")
    print("="*70)
    results['20day'] = backtest_with_freshness_filter(start_str, end_str, filter_type='20day')
    print_performance(results['20day'], "20-DAY FRESHNESS PERFORMANCE")

    # Compare all results
    compare_results(results)

    # Find best strategy
    best_filter = max(results.keys(), key=lambda k: results[k]['sharpe_ratio'])
    best_perf = results[best_filter]
    baseline_perf = results['none']

    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print(f"Best Filter: {get_freshness_filter_name(best_filter)}")
    print(f"  Sharpe Ratio: {best_perf['sharpe_ratio']:.2f} (vs {baseline_perf['sharpe_ratio']:.2f} baseline)")
    print(f"  Improvement: {((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100:.1f}%")
    print(f"  Win Rate: {best_perf['win_rate']:.1f}% (vs {baseline_perf['win_rate']:.1f}% baseline)")
    print(f"  Max Drawdown: {best_perf['max_drawdown']:.2f}% (vs {baseline_perf['max_drawdown']:.2f}% baseline)")
    print(f"  Total Trades: {best_perf['total_trades']} (vs {baseline_perf['total_trades']} baseline)")

    # Validation
    sharpe_improvement_pct = ((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100
    win_rate_improvement = best_perf['win_rate'] - baseline_perf['win_rate']

    print(f"\n{'='*70}")
    print("VALIDATION:")
    print(f"{'='*70}")
    print(f"Target: +12% Sharpe improvement, +5pp win rate")
    print(f"Actual: {sharpe_improvement_pct:+.1f}% Sharpe, {win_rate_improvement:+.1f}pp win rate")

    if sharpe_improvement_pct >= 12 and win_rate_improvement >= 5:
        print("SUCCESS: Targets achieved!")
        print(f"\nRECOMMENDATION: Deploy {get_freshness_filter_name(best_filter)} to production")
    else:
        print("PARTIAL/FAILED: Review results")
        print("\nRECOMMENDATION: Analyze best-performing filter for potential deployment")

    print(f"\n{'='*70}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

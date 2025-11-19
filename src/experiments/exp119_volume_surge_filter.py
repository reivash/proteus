"""
EXP-119: Volume Surge Filter

CONTEXT:
Current system ignores volume data when generating signals. High volume on down
days indicates genuine capitulation/panic selling, while low volume suggests
lack of conviction and higher false positive risk.

HYPOTHESIS:
- High volume + oversold = True capitulation (high quality signal)
- Low volume + oversold = Weak signal (noise, low follow-through)
- Volume surge = Market conviction, stronger mean reversion catalyst

OBJECTIVE:
Filter signals based on volume to improve signal quality and win rate.

METHODOLOGY:
Test volume-based filters:
1. Baseline: No volume filter (current)
2. Absolute volume: Require min absolute volume threshold
3. Relative volume: Require volume > 20-day average
4. Volume surge: Require volume > 1.5x 20-day average
5. Extreme surge: Require volume > 2.0x 20-day average

EXPECTED IMPACT:
- Win rate: +5-10% (eliminate low-conviction signals)
- Sharpe ratio: +10-15% (better signal quality)
- Trade count: -20-40% (more selective)

VALIDATION CRITERIA:
- Win rate >= 65% (+5pp minimum)
- Sharpe ratio >= 2.7 (+13% minimum)
- Profit factor >= 2.0

Created: 2025-11-19
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.trading.signal_scanner import SignalScanner
from src.trading.paper_trader import PaperTrader
from src.data.fetchers.yahoo_finance import YahooFinanceFetcher


def get_volume_filter_name(filter_type: str) -> str:
    """Get human-readable name for volume filter."""
    names = {
        'none': 'No Filter (Baseline)',
        'absolute': 'Minimum Absolute Volume (100K)',
        'relative_1.0x': 'Volume > 1.0x Average',
        'relative_1.5x': 'Volume > 1.5x Average (Surge)',
        'relative_2.0x': 'Volume > 2.0x Average (Extreme Surge)',
        'relative_2.5x': 'Volume > 2.5x Average (Panic)'
    }
    return names.get(filter_type, filter_type)


def calculate_volume_metrics(ticker: str, date_str: str, fetcher: YahooFinanceFetcher) -> Optional[Dict]:
    """
    Calculate volume metrics for a signal.

    Returns:
        Dictionary with volume metrics or None if data unavailable
    """
    try:
        # Get 30 days of data for volume calculation
        end_date = datetime.strptime(date_str, '%Y-%m-%d')
        start_date = end_date - timedelta(days=40)  # Extra buffer for weekends

        df = fetcher.get_stock_data(
            ticker,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        if df is None or len(df) < 20:
            return None

        # Get current day volume (last row)
        current_volume = df['Volume'].iloc[-1]

        # Calculate 20-day average volume (excluding current day)
        avg_volume_20d = df['Volume'].iloc[-21:-1].mean()

        # Calculate volume ratio
        volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 0

        return {
            'current_volume': current_volume,
            'avg_volume_20d': avg_volume_20d,
            'volume_ratio': volume_ratio
        }

    except Exception as e:
        return None


def apply_volume_filter(
    signal: dict,
    date_str: str,
    fetcher: YahooFinanceFetcher,
    filter_type: str
) -> bool:
    """
    Apply volume filter to signal.

    Returns:
        True if signal passes filter, False otherwise
    """
    if filter_type == 'none':
        return True

    # Get volume metrics
    volume_metrics = calculate_volume_metrics(signal['ticker'], date_str, fetcher)

    if volume_metrics is None:
        # Skip signal if volume data unavailable
        return False

    current_volume = volume_metrics['current_volume']
    volume_ratio = volume_metrics['volume_ratio']

    # Apply filter
    if filter_type == 'absolute':
        # Require minimum 100K shares traded
        return current_volume >= 100000

    elif filter_type == 'relative_1.0x':
        # Require above-average volume
        return volume_ratio >= 1.0

    elif filter_type == 'relative_1.5x':
        # Require 1.5x surge
        return volume_ratio >= 1.5

    elif filter_type == 'relative_2.0x':
        # Require 2.0x surge (extreme)
        return volume_ratio >= 2.0

    elif filter_type == 'relative_2.5x':
        # Require 2.5x surge (panic)
        return volume_ratio >= 2.5

    else:
        return True


def backtest_with_volume_filter(
    start_date: str,
    end_date: str,
    filter_type: str = 'none'
) -> dict:
    """
    Backtest strategy with volume filtering.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        filter_type: 'none', 'absolute', 'relative_1.0x', 'relative_1.5x', 'relative_2.0x', 'relative_2.5x'

    Returns:
        Dictionary with performance metrics
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {get_volume_filter_name(filter_type).upper()}")
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
        position_size_pct=0.20,
        max_positions=5,
        profit_target_pct=2.0,
        stop_loss_pct=-2.0,
        max_hold_days=2,
        use_limit_orders=True,
        limit_order_discount_pct=0.5
    )

    # Track filter statistics
    filter_stats = {
        'total_signals': 0,
        'passed_filter': 0,
        'failed_filter': 0,
        'no_volume_data': 0
    }

    # Convert dates
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    current_dt = start_dt

    while current_dt <= end_dt:
        date_str = current_dt.strftime('%Y-%m-%d')

        # Scan for signals
        signals = scanner.scan_all_stocks(date_str, fetcher)

        if len(signals) > 0:
            filter_stats['total_signals'] += len(signals)

            print(f"\n{'='*70}")
            print(f"Date: {date_str}")
            print(f"Total signals found: {len(signals)}")

            # Apply volume filter
            filtered_signals = []
            for signal in signals:
                passes_filter = apply_volume_filter(signal, date_str, fetcher, filter_type)

                if passes_filter:
                    filter_stats['passed_filter'] += 1
                    filtered_signals.append(signal)

                    # Add volume info for logging
                    volume_metrics = calculate_volume_metrics(signal['ticker'], date_str, fetcher)
                    if volume_metrics:
                        signal['volume_info'] = volume_metrics
                        print(f"  ✓ {signal['ticker']}: Volume ratio {volume_metrics['volume_ratio']:.2f}x (PASS)")
                else:
                    # Check if filter failed due to missing data or threshold
                    volume_metrics = calculate_volume_metrics(signal['ticker'], date_str, fetcher)
                    if volume_metrics is None:
                        filter_stats['no_volume_data'] += 1
                        print(f"  ✗ {signal['ticker']}: No volume data available (SKIP)")
                    else:
                        filter_stats['failed_filter'] += 1
                        print(f"  ✗ {signal['ticker']}: Volume ratio {volume_metrics['volume_ratio']:.2f}x (REJECT)")

            print(f"\nVolume Filter Applied ({filter_type}):")
            print(f"  Signals before filter: {len(signals)}")
            print(f"  Signals after filter: {len(filtered_signals)}")
            print(f"  Filtered out: {len(signals) - len(filtered_signals)} ({100 * (len(signals) - len(filtered_signals)) / len(signals):.1f}%)")
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
        print(f"Passed filter: {stats['passed_filter']} ({100 * stats['passed_filter'] / stats['total_signals']:.1f}%)")
        print(f"Failed filter: {stats['failed_filter']} ({100 * stats['failed_filter'] / stats['total_signals']:.1f}%)")
        print(f"No volume data: {stats['no_volume_data']} ({100 * stats['no_volume_data'] / stats['total_signals']:.1f}%)")

    print(f"{'='*70}\n")


def compare_results(results: Dict[str, dict]):
    """Print comparison table of all results."""
    print(f"\n{'='*70}")
    print("COMPARISON: VOLUME SURGE FILTERS")
    print(f"{'='*70}\n")

    # Create comparison table
    headers = ["Filter", "Trades", "WR%", "Return%", "Sharpe", "MaxDD%", "PF", "Pass%"]
    rows = []

    for filter_type, perf in results.items():
        pass_rate = 100 * perf['filter_stats']['passed_filter'] / perf['filter_stats']['total_signals'] if perf['filter_stats']['total_signals'] > 0 else 0

        row = [
            get_volume_filter_name(filter_type)[:30],  # Truncate long names
            perf['total_trades'],
            f"{perf['win_rate']:.1f}",
            f"{perf['total_return']:.2f}",
            f"{perf['sharpe_ratio']:.2f}",
            f"{perf['max_drawdown']:.2f}",
            f"{perf['profit_factor']:.2f}",
            f"{pass_rate:.1f}"
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
    """Run EXP-119 backtest."""
    print("="*70)
    print("EXP-119: VOLUME SURGE FILTER")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("OBJECTIVE: Improve signal quality via volume conviction filtering")
    print("THEORY: High volume on down days = true capitulation vs low volume = noise\n")

    # Test period: 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~2 years

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}\n")

    # Run all strategies
    results = {}

    print("\n" + "="*70)
    print("BASELINE: No Volume Filter")
    print("="*70)
    results['none'] = backtest_with_volume_filter(start_str, end_str, filter_type='none')
    print_performance(results['none'], "BASELINE PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 1: Minimum Absolute Volume (100K)")
    print("="*70)
    results['absolute'] = backtest_with_volume_filter(start_str, end_str, filter_type='absolute')
    print_performance(results['absolute'], "ABSOLUTE VOLUME PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 2: Volume > 1.0x Average")
    print("="*70)
    results['relative_1.0x'] = backtest_with_volume_filter(start_str, end_str, filter_type='relative_1.0x')
    print_performance(results['relative_1.0x'], "1.0X RELATIVE VOLUME PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 3: Volume > 1.5x Average (Surge)")
    print("="*70)
    results['relative_1.5x'] = backtest_with_volume_filter(start_str, end_str, filter_type='relative_1.5x')
    print_performance(results['relative_1.5x'], "1.5X SURGE PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 4: Volume > 2.0x Average (Extreme Surge)")
    print("="*70)
    results['relative_2.0x'] = backtest_with_volume_filter(start_str, end_str, filter_type='relative_2.0x')
    print_performance(results['relative_2.0x'], "2.0X EXTREME SURGE PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 5: Volume > 2.5x Average (Panic)")
    print("="*70)
    results['relative_2.5x'] = backtest_with_volume_filter(start_str, end_str, filter_type='relative_2.5x')
    print_performance(results['relative_2.5x'], "2.5X PANIC PERFORMANCE")

    # Compare all results
    compare_results(results)

    # Find best strategy
    best_filter = max(results.keys(), key=lambda k: results[k]['sharpe_ratio'])
    best_perf = results[best_filter]
    baseline_perf = results['none']

    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print(f"Best Filter: {get_volume_filter_name(best_filter)}")
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
    print(f"Target: +10% Sharpe improvement, +5pp win rate")
    print(f"Actual: {sharpe_improvement_pct:+.1f}% Sharpe, {win_rate_improvement:+.1f}pp win rate")

    if sharpe_improvement_pct >= 10 and best_perf['win_rate'] >= 65:
        print("SUCCESS: Targets achieved!")
        print(f"\nRECOMMENDATION: Deploy {get_volume_filter_name(best_filter)} to production")
    else:
        print("PARTIAL/FAILED: Review results")
        print("\nRECOMMENDATION: Analyze best-performing filter for potential deployment")

    print(f"\n{'='*70}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

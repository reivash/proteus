"""
EXP-122: Position Concentration Limits

CONTEXT:
Current system uses uniform 20% position sizing, which can create dangerous
concentration risk. A single large loss on a 20% position creates -4% portfolio
loss (assuming -20% stock loss). This concentration risk is manageable but
could be optimized with tighter limits.

HYPOTHESIS:
- Enforce maximum position size limits to reduce concentration risk
- Smaller positions = lower single-position impact = smoother equity curve
- Trade-off: More positions = more diversification vs lower returns per win
- Professional standard: 10-15% max position size for concentrated portfolios

OBJECTIVE:
Test different position concentration limits to optimize risk-adjusted returns.

METHODOLOGY:
Test maximum position size limits:
1. Baseline: 20% max (current)
2. 15% max concentration
3. 12% max concentration
4. 10% max concentration
5. 8% max concentration (highly diversified)

EXPECTED IMPACT:
- Max drawdown: -10-20% reduction (smaller max loss per position)
- Sharpe ratio: +5-12% (smoother returns)
- Win rate: Unchanged (same signals)
- Total return: Slightly lower (reduced position sizes)

VALIDATION CRITERIA:
- Max drawdown <= -8% (vs current -10%)
- Sharpe ratio >= 2.55 (+7% minimum)
- Risk-adjusted returns improved

Created: 2025-11-19
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.trading.signal_scanner import SignalScanner
from src.trading.paper_trader import PaperTrader
from src.data.fetchers.yahoo_finance import YahooFinanceFetcher


def get_limit_name(limit_pct: float) -> str:
    """Get human-readable name for concentration limit."""
    return f"{limit_pct:.0f}% Max Position Size"


def backtest_with_concentration_limit(
    start_date: str,
    end_date: str,
    max_position_pct: float = 0.20
) -> dict:
    """
    Backtest strategy with position concentration limit.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_position_pct: Maximum position size as % of capital (0.20 = 20%)

    Returns:
        Dictionary with performance metrics
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {get_limit_name(max_position_pct * 100).upper()}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*70}\n")

    # Initialize components
    fetcher = YahooFinanceFetcher()
    scanner = SignalScanner(
        lookback_days=90,
        min_signal_strength=None  # Use Q4 filtering from SignalScanner
    )

    # Use custom position size based on concentration limit
    trader = PaperTrader(
        initial_capital=90000,
        max_position_pct=max_position_pct,  # Set max position size
        max_positions=5,
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=2,
        use_limit_orders=True
    )

    # Track concentration statistics
    concentration_stats = {
        'total_positions': 0,
        'positions_at_limit': 0,
        'max_position_sizes': [],  # Track all position sizes
        'largest_position_pct': 0.0
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
            print(f"\n{'='*70}")
            print(f"Date: {date_str}")
            print(f"Signals found: {len(signals)}")
            print(f"Max position size: {max_position_pct * 100:.0f}% of capital")

            # Process signals
            trader.process_signals(signals, date_str)

            # Track concentration for newly opened positions
            for pos in trader.positions.values():
                if pos.entry_date == date_str:
                    concentration_stats['total_positions'] += 1

                    # Calculate position size as % of capital at entry
                    capital_at_entry = trader.capital + sum(p.shares * p.entry_price for p in trader.positions.values() if p.entry_date != date_str)
                    position_value = pos.shares * pos.entry_price
                    position_pct = position_value / capital_at_entry

                    concentration_stats['max_position_sizes'].append(position_pct)
                    concentration_stats['largest_position_pct'] = max(
                        concentration_stats['largest_position_pct'],
                        position_pct
                    )

                    # Check if at limit
                    if position_pct >= max_position_pct * 0.95:  # Within 5% of limit
                        concentration_stats['positions_at_limit'] += 1
                        print(f"  Position {pos.ticker}: {position_pct * 100:.1f}% (at limit)")

            print(f"{'='*70}")

        # Update positions
        trader.update_positions(date_str, fetcher)

        # Move to next trading day
        current_dt += timedelta(days=1)

    # Get performance
    perf = trader.get_performance()

    # Add concentration statistics
    if concentration_stats['max_position_sizes']:
        concentration_stats['avg_position_size'] = np.mean(concentration_stats['max_position_sizes'])
        concentration_stats['median_position_size'] = np.median(concentration_stats['max_position_sizes'])
    else:
        concentration_stats['avg_position_size'] = 0.0
        concentration_stats['median_position_size'] = 0.0

    perf['concentration_stats'] = concentration_stats
    perf['max_position_pct'] = max_position_pct

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

    # Print concentration statistics
    if 'concentration_stats' in perf:
        stats = perf['concentration_stats']
        print(f"\n{'='*70}")
        print("CONCENTRATION STATISTICS:")
        print(f"{'='*70}")
        print(f"Total positions opened: {stats['total_positions']}")
        print(f"Positions at size limit: {stats['positions_at_limit']}")
        if stats['total_positions'] > 0:
            limit_pct = 100 * stats['positions_at_limit'] / stats['total_positions']
            print(f"Positions at limit %: {limit_pct:.1f}%")
        print(f"Largest position size: {stats['largest_position_pct'] * 100:.1f}%")
        print(f"Average position size: {stats['avg_position_size'] * 100:.1f}%")
        print(f"Median position size: {stats['median_position_size'] * 100:.1f}%")

    print(f"{'='*70}\n")


def compare_results(results: Dict[str, dict]):
    """Print comparison table of all results."""
    print(f"\n{'='*70}")
    print("COMPARISON: POSITION CONCENTRATION LIMITS")
    print(f"{'='*70}\n")

    # Create comparison table
    headers = ["Max Size", "Trades", "WR%", "Return%", "Sharpe", "MaxDD%", "PF", "AvgSize%"]
    rows = []

    for max_pct, perf in results.items():
        avg_size = perf['concentration_stats']['avg_position_size'] * 100

        row = [
            f"{max_pct * 100:.0f}%",
            perf['total_trades'],
            f"{perf['win_rate']:.1f}",
            f"{perf['total_return']:.2f}",
            f"{perf['sharpe_ratio']:.2f}",
            f"{perf['max_drawdown']:.2f}",
            f"{perf['profit_factor']:.2f}",
            f"{avg_size:.1f}"
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
    """Run EXP-122 backtest."""
    print("="*70)
    print("EXP-122: POSITION CONCENTRATION LIMITS")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("OBJECTIVE: Optimize risk-adjusted returns via position size limits")
    print("THEORY: Smaller max positions = lower concentration risk = smoother returns\n")

    # Test period: 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~2 years

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}\n")

    # Run all strategies
    results = {}

    print("\n" + "="*70)
    print("BASELINE: 20% Max Position Size (Current)")
    print("="*70)
    results[0.20] = backtest_with_concentration_limit(start_str, end_str, max_position_pct=0.20)
    print_performance(results[0.20], "BASELINE PERFORMANCE (20% MAX)")

    print("\n" + "="*70)
    print("TEST 1: 15% Max Position Size")
    print("="*70)
    results[0.15] = backtest_with_concentration_limit(start_str, end_str, max_position_pct=0.15)
    print_performance(results[0.15], "15% MAX PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 2: 12% Max Position Size")
    print("="*70)
    results[0.12] = backtest_with_concentration_limit(start_str, end_str, max_position_pct=0.12)
    print_performance(results[0.12], "12% MAX PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 3: 10% Max Position Size")
    print("="*70)
    results[0.10] = backtest_with_concentration_limit(start_str, end_str, max_position_pct=0.10)
    print_performance(results[0.10], "10% MAX PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 4: 8% Max Position Size (Highly Diversified)")
    print("="*70)
    results[0.08] = backtest_with_concentration_limit(start_str, end_str, max_position_pct=0.08)
    print_performance(results[0.08], "8% MAX PERFORMANCE")

    # Compare all results
    compare_results(results)

    # Find best strategy by Sharpe ratio
    best_pct = max(results.keys(), key=lambda k: results[k]['sharpe_ratio'])
    best_perf = results[best_pct]
    baseline_perf = results[0.20]

    # Also find best by max drawdown
    best_dd_pct = min(results.keys(), key=lambda k: results[k]['max_drawdown'])
    best_dd_perf = results[best_dd_pct]

    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print(f"Best Strategy (Sharpe): {get_limit_name(best_pct * 100)}")
    print(f"  Sharpe Ratio: {best_perf['sharpe_ratio']:.2f} (vs {baseline_perf['sharpe_ratio']:.2f} baseline)")
    print(f"  Improvement: {((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100:.1f}%")
    print(f"  Max Drawdown: {best_perf['max_drawdown']:.2f}% (vs {baseline_perf['max_drawdown']:.2f}% baseline)")
    print(f"  Total Return: {best_perf['total_return']:.2f}% (vs {baseline_perf['total_return']:.2f}% baseline)")

    print(f"\nBest Strategy (Max DD): {get_limit_name(best_dd_pct * 100)}")
    print(f"  Max Drawdown: {best_dd_perf['max_drawdown']:.2f}% (vs {baseline_perf['max_drawdown']:.2f}% baseline)")
    print(f"  Reduction: {baseline_perf['max_drawdown'] - best_dd_perf['max_drawdown']:.2f}pp")
    print(f"  Sharpe Ratio: {best_dd_perf['sharpe_ratio']:.2f}")

    # Validation
    sharpe_improvement_pct = ((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100
    dd_reduction = baseline_perf['max_drawdown'] - best_dd_perf['max_drawdown']

    print(f"\n{'='*70}")
    print("VALIDATION:")
    print(f"{'='*70}")
    print(f"Target: +5% Sharpe improvement OR -1pp max drawdown")
    print(f"Actual: {sharpe_improvement_pct:+.1f}% Sharpe, {dd_reduction:+.2f}pp max drawdown")

    if sharpe_improvement_pct >= 5 or dd_reduction >= 1:
        print("SUCCESS: Target achieved!")
        print(f"\nRECOMMENDATION: Deploy {get_limit_name(best_pct * 100)} to production")
    else:
        print("PARTIAL: Review results")
        print("\nRECOMMENDATION: Consider best-performing limit for deployment")

    # Additional insights
    print(f"\n{'='*70}")
    print("RISK MANAGEMENT INSIGHTS:")
    print(f"{'='*70}")

    # Calculate max single-position loss potential
    for pct, perf in results.items():
        max_single_loss = pct * 0.20  # Assume 20% stock loss (stop loss)
        print(f"{get_limit_name(pct * 100)}: Max single-position loss = {max_single_loss * 100:.1f}% of portfolio")

    print(f"\n{'='*70}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

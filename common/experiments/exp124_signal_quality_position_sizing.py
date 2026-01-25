"""
EXP-124: Signal Quality-Based Position Sizing

CRITICAL INSIGHT FROM EXP-091:
Q4 signals (top 25% by signal_strength) achieve 77.8% win rate vs 63.7% overall.
Instead of filtering low-quality signals (EXP-093), SIZE positions based on quality.

HYPOTHESIS:
Adaptive position sizing > binary filtering because:
- Captures ALL opportunities (no missed trades)
- Allocates more capital to high-quality signals
- Reduces exposure to low-quality signals
- Better capital efficiency

METHODOLOGY:
Test position sizing strategies:
1. Baseline: Uniform 20% position size (current)
2. Quartile-based: Size by signal_strength quartile
   - Q4 (top 25%): 20% position size (full size)
   - Q3: 15% position size (-25% reduction)
   - Q2: 10% position size (-50% reduction)
   - Q1 (bottom 25%): 5% position size (-75% reduction)
3. Linear scaling: Size = 5% + (signal_strength_percentile * 15%)
4. Aggressive: Q4=20%, Q3=10%, Q2=5%, Q1=0% (hybrid with filtering)

EXPECTED IMPACT:
- Win rate: Improved weighted average (more $ in winners)
- Sharpe ratio: +15-25% (better capital allocation)
- Max drawdown: -15-20% (less capital in losers)
- Total return: +10-20% (optimize risk/reward)
- Trade frequency: Unchanged (all signals traded)

VALIDATION CRITERIA:
- Sharpe ratio >= 2.75 (+16% minimum over baseline 2.37)
- Max drawdown <= -8.5% (vs current -10%)
- Deploy if both criteria met

COMPLEMENTARY TO:
- EXP-093: Binary filtering (all-or-nothing)
- EXP-094: Portfolio correlation limits (risk management)
- EXP-122: Position concentration limits (max size caps)

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

from common.trading.signal_scanner import SignalScanner
from common.trading.paper_trader import PaperTrader
from common.data.fetchers.yahoo_finance import YahooFinanceFetcher


def calculate_signal_strength_percentile(signal_strength: float, historical_strengths: List[float]) -> float:
    """
    Calculate percentile rank of signal strength.

    Args:
        signal_strength: Current signal strength
        historical_strengths: List of historical signal strengths

    Returns:
        Percentile rank (0.0 to 1.0)
    """
    if not historical_strengths:
        return 0.5  # Default to median if no history

    rank = sum(1 for s in historical_strengths if s < signal_strength)
    percentile = rank / len(historical_strengths)
    return percentile


def get_position_size_multiplier(
    signal_strength: float,
    historical_strengths: List[float],
    sizing_strategy: str
) -> float:
    """
    Get position size multiplier based on signal strength and strategy.

    Args:
        signal_strength: Current signal strength
        historical_strengths: Historical signal strengths for percentile calculation
        sizing_strategy: 'uniform', 'quartile', 'linear', or 'aggressive'

    Returns:
        Position size multiplier (0.0 to 1.0)
    """
    if sizing_strategy == 'uniform':
        return 1.0  # Baseline: uniform sizing

    # Calculate signal strength percentile
    percentile = calculate_signal_strength_percentile(signal_strength, historical_strengths)

    if sizing_strategy == 'quartile':
        # Quartile-based sizing
        if percentile >= 0.75:  # Q4 (top 25%)
            return 1.0  # Full size (20%)
        elif percentile >= 0.50:  # Q3
            return 0.75  # 15%
        elif percentile >= 0.25:  # Q2
            return 0.50  # 10%
        else:  # Q1 (bottom 25%)
            return 0.25  # 5%

    elif sizing_strategy == 'linear':
        # Linear scaling: 25% to 100% based on percentile
        return 0.25 + (percentile * 0.75)

    elif sizing_strategy == 'aggressive':
        # Aggressive: Heavy concentration on top signals
        if percentile >= 0.75:  # Q4
            return 1.0  # Full size (20%)
        elif percentile >= 0.50:  # Q3
            return 0.50  # 10%
        elif percentile >= 0.25:  # Q2
            return 0.25  # 5%
        else:  # Q1
            return 0.0  # Skip trade

    return 1.0  # Default


def backtest_signal_quality_sizing(
    start_date: str,
    end_date: str,
    sizing_strategy: str = 'uniform'
) -> dict:
    """
    Backtest strategy with signal quality-based position sizing.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        sizing_strategy: 'uniform', 'quartile', 'linear', or 'aggressive'

    Returns:
        Dictionary with performance metrics
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {sizing_strategy.upper()} SIZING STRATEGY")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*70}\n")

    # Initialize components
    fetcher = YahooFinanceFetcher()
    scanner = SignalScanner(
        lookback_days=90,
        min_signal_strength=None  # Use all signals, size by quality
    )
    trader = PaperTrader(
        initial_capital=90000,
        max_positions=5,
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=2,
        use_limit_orders=True
    )

    # Track historical signal strengths for percentile calculation
    historical_strengths = []

    # Track sizing statistics
    sizing_stats = {
        'total_signals': 0,
        'q4_signals': 0,  # Top 25%
        'q3_signals': 0,
        'q2_signals': 0,
        'q1_signals': 0,  # Bottom 25%
        'avg_position_size_pct': [],
        'signals_skipped': 0
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

            # Apply signal quality-based sizing
            sized_signals = []
            for signal in signals:
                signal_strength = signal.get('signal_strength', 0.0)
                sizing_stats['total_signals'] += 1

                # Calculate position size multiplier
                multiplier = get_position_size_multiplier(
                    signal_strength,
                    historical_strengths,
                    sizing_strategy
                )

                # Add to historical strengths
                historical_strengths.append(signal_strength)

                # Calculate percentile for stats
                percentile = calculate_signal_strength_percentile(
                    signal_strength,
                    historical_strengths[:-1]  # Exclude current
                )

                # Track quartile
                if percentile >= 0.75:
                    sizing_stats['q4_signals'] += 1
                    quartile = 'Q4'
                elif percentile >= 0.50:
                    sizing_stats['q3_signals'] += 1
                    quartile = 'Q3'
                elif percentile >= 0.25:
                    sizing_stats['q2_signals'] += 1
                    quartile = 'Q2'
                else:
                    sizing_stats['q1_signals'] += 1
                    quartile = 'Q1'

                if multiplier == 0.0:
                    # Skip this signal
                    sizing_stats['signals_skipped'] += 1
                    print(f"  [SKIP] {signal['ticker']}: {quartile}, strength={signal_strength:.3f}")
                    continue

                # Apply position size multiplier
                if multiplier != 1.0:
                    signal['position_size_multiplier'] = multiplier
                    effective_size = 0.20 * multiplier  # Base 20% * multiplier
                    sizing_stats['avg_position_size_pct'].append(effective_size * 100)
                    print(f"  [SIZE] {signal['ticker']}: {quartile}, {effective_size*100:.1f}% position (multiplier={multiplier:.2f}, strength={signal_strength:.3f})")
                else:
                    sizing_stats['avg_position_size_pct'].append(20.0)

                sized_signals.append(signal)

            print(f"\nSignals after sizing: {len(sized_signals)}/{len(signals)}")
            print(f"{'='*70}")

            # Process sized signals
            if sized_signals:
                trader.process_signals(sized_signals, date_str)

        # Update positions and check exits
        open_positions = trader.get_open_positions()
        if open_positions:
            tickers = [pos['ticker'] for pos in open_positions]
            prices = {}
            for ticker in tickers:
                try:
                    data = fetcher.get_stock_data(ticker, date_str, date_str)
                    if data is not None and not data.empty:
                        prices[ticker] = float(data['Close'].iloc[-1])
                except:
                    pass
            if prices:
                trader.update_daily_equity(prices, date_str)
                trader.check_exits(prices, date_str)

        # Move to next trading day
        current_dt += timedelta(days=1)

    # Get performance
    perf = trader.get_performance()

    # Add sizing statistics
    if sizing_stats['avg_position_size_pct']:
        sizing_stats['avg_position_size'] = np.mean(sizing_stats['avg_position_size_pct'])
        sizing_stats['median_position_size'] = np.median(sizing_stats['avg_position_size_pct'])
    else:
        sizing_stats['avg_position_size'] = 0.0
        sizing_stats['median_position_size'] = 0.0

    perf['sizing_stats'] = sizing_stats
    perf['sizing_strategy'] = sizing_strategy

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

    # Print sizing statistics
    if 'sizing_stats' in perf:
        stats = perf['sizing_stats']
        print(f"\n{'='*70}")
        print("POSITION SIZING STATISTICS:")
        print(f"{'='*70}")
        print(f"Total signals processed: {stats['total_signals']}")
        print(f"Signals skipped: {stats['signals_skipped']}")
        if stats['total_signals'] > 0:
            print(f"\nSignal Distribution by Quartile:")
            print(f"  Q4 (top 25%): {stats['q4_signals']} ({100*stats['q4_signals']/stats['total_signals']:.1f}%)")
            print(f"  Q3: {stats['q3_signals']} ({100*stats['q3_signals']/stats['total_signals']:.1f}%)")
            print(f"  Q2: {stats['q2_signals']} ({100*stats['q2_signals']/stats['total_signals']:.1f}%)")
            print(f"  Q1 (bottom 25%): {stats['q1_signals']} ({100*stats['q1_signals']/stats['total_signals']:.1f}%)")
        print(f"\nAverage position size: {stats['avg_position_size']:.1f}%")
        print(f"Median position size: {stats['median_position_size']:.1f}%")

    print(f"{'='*70}\n")


def compare_results(results: Dict[str, dict]):
    """Print comparison table of all results."""
    print(f"\n{'='*70}")
    print("COMPARISON: SIGNAL QUALITY-BASED POSITION SIZING")
    print(f"{'='*70}\n")

    # Create comparison table
    headers = ["Strategy", "Trades", "WR%", "Return%", "Sharpe", "MaxDD%", "PF", "AvgSize%"]
    rows = []

    for strategy, perf in results.items():
        avg_size = perf['sizing_stats']['avg_position_size']

        row = [
            strategy.capitalize(),
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
    """Run EXP-124 backtest."""
    print("="*70)
    print("EXP-124: SIGNAL QUALITY-BASED POSITION SIZING")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("OBJECTIVE: Optimize capital allocation based on signal quality")
    print("THEORY: Size positions proportionally to signal strength percentile\n")

    # Test period: 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~2 years

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}\n")

    # Run all strategies
    results = {}

    print("\n" + "="*70)
    print("BASELINE: Uniform Position Sizing (20% for all)")
    print("="*70)
    results['uniform'] = backtest_signal_quality_sizing(start_str, end_str, sizing_strategy='uniform')
    print_performance(results['uniform'], "BASELINE PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 1: Quartile-Based Sizing (Q4=20%, Q3=15%, Q2=10%, Q1=5%)")
    print("="*70)
    results['quartile'] = backtest_signal_quality_sizing(start_str, end_str, sizing_strategy='quartile')
    print_performance(results['quartile'], "QUARTILE SIZING PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 2: Linear Scaling (5% to 20% based on percentile)")
    print("="*70)
    results['linear'] = backtest_signal_quality_sizing(start_str, end_str, sizing_strategy='linear')
    print_performance(results['linear'], "LINEAR SIZING PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 3: Aggressive Sizing (Q4=20%, Q3=10%, Q2=5%, Q1=0%)")
    print("="*70)
    results['aggressive'] = backtest_signal_quality_sizing(start_str, end_str, sizing_strategy='aggressive')
    print_performance(results['aggressive'], "AGGRESSIVE SIZING PERFORMANCE")

    # Compare all results
    compare_results(results)

    # Find best strategy
    best_strategy = max(results.keys(), key=lambda k: results[k]['sharpe_ratio'])
    best_perf = results[best_strategy]
    baseline_perf = results['uniform']

    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print(f"Best Strategy: {best_strategy.capitalize()} Sizing")
    print(f"  Sharpe Ratio: {best_perf['sharpe_ratio']:.2f} (vs {baseline_perf['sharpe_ratio']:.2f} baseline)")
    print(f"  Improvement: {((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100:.1f}%")
    print(f"  Win Rate: {best_perf['win_rate']:.1f}% (vs {baseline_perf['win_rate']:.1f}% baseline)")
    print(f"  Max Drawdown: {best_perf['max_drawdown']:.2f}% (vs {baseline_perf['max_drawdown']:.2f}% baseline)")
    print(f"  Total Return: {best_perf['total_return']:.2f}% (vs {baseline_perf['total_return']:.2f}% baseline)")

    # Validation
    sharpe_improvement_pct = ((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100
    dd_reduction = baseline_perf['max_drawdown'] - best_perf['max_drawdown']

    print(f"\n{'='*70}")
    print("VALIDATION:")
    print(f"{'='*70}")
    print(f"Target: +16% Sharpe improvement AND -1.5pp max drawdown")
    print(f"Actual: {sharpe_improvement_pct:+.1f}% Sharpe, {dd_reduction:+.2f}pp max drawdown")

    if sharpe_improvement_pct >= 16 and dd_reduction >= 1.5:
        print("SUCCESS: Both targets achieved!")
        print(f"\nRECOMMENDATION: Deploy {best_strategy.capitalize()} Sizing to production")
    elif sharpe_improvement_pct >= 16 or dd_reduction >= 1.5:
        print("PARTIAL: One target achieved")
        print(f"\nRECOMMENDATION: Consider deploying {best_strategy.capitalize()} Sizing")
    else:
        print("REVIEW: Targets not fully met")
        print("\nRECOMMENDATION: Review results and consider hybrid approach")

    print(f"\n{'='*70}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

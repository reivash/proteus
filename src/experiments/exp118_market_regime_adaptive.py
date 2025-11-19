"""
EXP-118: Market Regime-Adaptive Strategy

CONTEXT:
Current system detects market regimes (BULL/BEAR/SIDEWAYS) but treats all
signals identically regardless of regime. Mean reversion behaves differently
across market conditions.

HYPOTHESIS:
- SIDEWAYS markets: Best for mean reversion (range-bound oscillation)
- BULL markets: Good for mean reversion (dips buy on strength)
- BEAR markets: Risky for mean reversion (falling knife catching)

OBJECTIVE:
Adapt trading strategy based on market regime to improve risk-adjusted returns.

METHODOLOGY:
Test regime-based filters:
1. Baseline: Trade all regimes equally (current)
2. Sideways-only: Only trade SIDEWAYS markets
3. No-bear: Trade BULL + SIDEWAYS, skip BEAR
4. Adaptive sizing: Full size SIDEWAYS, 0.7x BULL, 0.4x BEAR
5. Adaptive thresholds: Tighter exits in BEAR, wider in SIDEWAYS

EXPECTED IMPACT:
- Sharpe ratio: +15-25% (avoid unfavorable regimes)
- Max drawdown: -20-30% (skip bear market losses)
- Win rate: +3-5pp (better regime selection)

VALIDATION CRITERIA:
- Sharpe ratio >= 2.8 (+18% minimum)
- Max drawdown <= -8% (vs current -10%)

Created: 2025-11-18
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.trading.signal_scanner import SignalScanner
from src.trading.paper_trader import PaperTrader
from src.data.fetchers.yahoo_finance import YahooFinanceFetcher


def get_regime_filter_name(filter_type: str) -> str:
    """Get human-readable name for regime filter."""
    names = {
        'none': 'No Filter (Baseline)',
        'sideways_only': 'Sideways Markets Only',
        'no_bear': 'Bull + Sideways (No Bear)',
        'adaptive_sizing': 'Regime-Adaptive Position Sizing',
        'adaptive_exits': 'Regime-Adaptive Exit Thresholds'
    }
    return names.get(filter_type, filter_type)


def get_regime_multiplier(regime: str, filter_type: str) -> float:
    """
    Get position size multiplier based on regime and filter type.

    Returns:
        Multiplier (1.0 = full size, 0.0 = skip trade)
    """
    if filter_type == 'sideways_only':
        return 1.0 if regime == 'SIDEWAYS' else 0.0
    elif filter_type == 'no_bear':
        return 1.0 if regime != 'BEAR' else 0.0
    elif filter_type == 'adaptive_sizing':
        if regime == 'SIDEWAYS':
            return 1.0  # Full size - optimal regime
        elif regime == 'BULL':
            return 0.7  # Reduced size - still workable
        else:  # BEAR
            return 0.4  # Minimal size - high risk
    else:  # 'none' or 'adaptive_exits'
        return 1.0  # Trade all regimes


def get_regime_exits(regime: str, filter_type: str) -> Tuple[float, float]:
    """
    Get exit thresholds based on regime and filter type.

    Returns:
        (profit_target_pct, stop_loss_pct)
    """
    if filter_type == 'adaptive_exits':
        if regime == 'SIDEWAYS':
            return (2.0, -2.0)  # Standard - optimal regime
        elif regime == 'BULL':
            return (2.5, -1.5)  # Wider target, tighter stop - ride strength
        else:  # BEAR
            return (1.5, -1.5)  # Quick exit - reduce exposure
    else:
        return (2.0, -2.0)  # Baseline exits


def backtest_with_regime_filter(
    start_date: str,
    end_date: str,
    filter_type: str = 'none'
) -> dict:
    """
    Backtest strategy with regime-based filtering.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        filter_type: 'none', 'sideways_only', 'no_bear', 'adaptive_sizing', 'adaptive_exits'

    Returns:
        Dictionary with performance metrics
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {get_regime_filter_name(filter_type).upper()}")
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
        profit_target_pct=2.0,  # May be overridden by regime
        stop_loss_pct=-2.0,     # May be overridden by regime
        max_hold_days=2,
        use_limit_orders=True,
        limit_order_discount_pct=0.5
    )

    # Track regime statistics
    regime_stats = {
        'BULL': {'signals': 0, 'trades': 0, 'skipped': 0},
        'BEAR': {'signals': 0, 'trades': 0, 'skipped': 0},
        'SIDEWAYS': {'signals': 0, 'trades': 0, 'skipped': 0}
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
            print(f"\n{'='*70}")
            print(f"Total signals found: {len(signals)}")

            # Get market regime from first signal (all signals on same day have same regime)
            if signals:
                regime = signals[0].get('market_regime', 'SIDEWAYS')
                regime_stats[regime]['signals'] += len(signals)

                # Apply regime filter
                filtered_signals = []
                for signal in signals:
                    multiplier = get_regime_multiplier(regime, filter_type)

                    if multiplier == 0.0:
                        # Skip this signal
                        regime_stats[regime]['skipped'] += 1
                        print(f"  [REGIME FILTER] {signal['ticker']}: Skipped ({regime} regime, filter={filter_type})")
                        continue

                    # Apply position size multiplier
                    if multiplier != 1.0:
                        signal['position_size_multiplier'] = multiplier
                        print(f"  [REGIME SIZING] {signal['ticker']}: {multiplier:.1f}x size ({regime} regime)")

                    # Apply regime-specific exits
                    profit_target, stop_loss = get_regime_exits(regime, filter_type)
                    if (profit_target, stop_loss) != (2.0, -2.0):
                        signal['profit_target_pct'] = profit_target
                        signal['stop_loss_pct'] = stop_loss
                        print(f"  [REGIME EXITS] {signal['ticker']}: {profit_target:.1f}% target, {stop_loss:.1f}% stop ({regime} regime)")

                    filtered_signals.append(signal)
                    regime_stats[regime]['trades'] += 1

                print(f"\nRegime Filter Applied ({filter_type}):")
                print(f"  Signals before filter: {len(signals)}")
                print(f"  Signals after filter: {len(filtered_signals)}")
                print(f"  Filtered out: {len(signals) - len(filtered_signals)} ({100 * (len(signals) - len(filtered_signals)) / len(signals):.1f}%)")
                print(f"  Current regime: {regime}")
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

    # Add regime statistics
    perf['regime_stats'] = regime_stats
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

    # Print regime statistics
    if 'regime_stats' in perf:
        print(f"\n{'='*70}")
        print("REGIME BREAKDOWN:")
        print(f"{'='*70}")
        for regime in ['BULL', 'SIDEWAYS', 'BEAR']:
            stats = perf['regime_stats'][regime]
            if stats['signals'] > 0:
                skip_pct = 100 * stats['skipped'] / stats['signals']
                print(f"{regime}:")
                print(f"  Signals generated: {stats['signals']}")
                print(f"  Trades executed: {stats['trades']}")
                print(f"  Skipped by filter: {stats['skipped']} ({skip_pct:.1f}%)")

    print(f"{'='*70}\n")


def compare_results(results: Dict[str, dict]):
    """Print comparison table of all results."""
    print(f"\n{'='*70}")
    print("COMPARISON: REGIME-ADAPTIVE STRATEGIES")
    print(f"{'='*70}\n")

    # Create comparison table
    headers = ["Strategy", "Trades", "WR%", "Return%", "Sharpe", "MaxDD%", "PF"]
    rows = []

    for filter_type, perf in results.items():
        row = [
            get_regime_filter_name(filter_type),
            perf['total_trades'],
            f"{perf['win_rate']:.1f}",
            f"{perf['total_return']:.2f}",
            f"{perf['sharpe_ratio']:.2f}",
            f"{perf['max_drawdown']:.2f}",
            f"{perf['profit_factor']:.2f}"
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
    print("REGIME-SPECIFIC TRADE DISTRIBUTION:")
    print(f"{'='*70}\n")

    # Print regime breakdown for each strategy
    for filter_type, perf in results.items():
        print(f"\n{get_regime_filter_name(filter_type)}:")
        if 'regime_stats' in perf:
            total_signals = sum(perf['regime_stats'][r]['signals'] for r in ['BULL', 'BEAR', 'SIDEWAYS'])
            total_trades = sum(perf['regime_stats'][r]['trades'] for r in ['BULL', 'BEAR', 'SIDEWAYS'])

            print(f"  Total signals: {total_signals}, Total trades: {total_trades}")
            for regime in ['BULL', 'SIDEWAYS', 'BEAR']:
                stats = perf['regime_stats'][regime]
                if stats['signals'] > 0:
                    trade_pct = 100 * stats['trades'] / total_trades if total_trades > 0 else 0
                    print(f"  {regime}: {stats['trades']} trades ({trade_pct:.1f}% of total)")

    print(f"\n{'='*70}")


def main():
    """Run EXP-118 backtest."""
    print("="*70)
    print("EXP-118: MARKET REGIME-ADAPTIVE STRATEGY")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("OBJECTIVE: Improve risk-adjusted returns via regime-aware trading")
    print("THEORY: Mean reversion performs differently across market regimes\n")

    # Test period: 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~2 years

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}\n")

    # Run all strategies
    results = {}

    print("\n" + "="*70)
    print("BASELINE: No Regime Filter")
    print("="*70)
    results['none'] = backtest_with_regime_filter(start_str, end_str, filter_type='none')
    print_performance(results['none'], "BASELINE PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 1: Sideways Markets Only")
    print("="*70)
    results['sideways_only'] = backtest_with_regime_filter(start_str, end_str, filter_type='sideways_only')
    print_performance(results['sideways_only'], "SIDEWAYS-ONLY PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 2: Bull + Sideways (No Bear)")
    print("="*70)
    results['no_bear'] = backtest_with_regime_filter(start_str, end_str, filter_type='no_bear')
    print_performance(results['no_bear'], "NO-BEAR PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 3: Adaptive Position Sizing by Regime")
    print("="*70)
    results['adaptive_sizing'] = backtest_with_regime_filter(start_str, end_str, filter_type='adaptive_sizing')
    print_performance(results['adaptive_sizing'], "ADAPTIVE-SIZING PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 4: Adaptive Exit Thresholds by Regime")
    print("="*70)
    results['adaptive_exits'] = backtest_with_regime_filter(start_str, end_str, filter_type='adaptive_exits')
    print_performance(results['adaptive_exits'], "ADAPTIVE-EXITS PERFORMANCE")

    # Compare all results
    compare_results(results)

    # Find best strategy
    best_filter = max(results.keys(), key=lambda k: results[k]['sharpe_ratio'])
    best_perf = results[best_filter]
    baseline_perf = results['none']

    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print(f"Best Strategy: {get_regime_filter_name(best_filter)}")
    print(f"  Sharpe Ratio: {best_perf['sharpe_ratio']:.2f} (vs {baseline_perf['sharpe_ratio']:.2f} baseline)")
    print(f"  Improvement: {((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100:.1f}%")
    print(f"  Win Rate: {best_perf['win_rate']:.1f}% (vs {baseline_perf['win_rate']:.1f}% baseline)")
    print(f"  Max Drawdown: {best_perf['max_drawdown']:.2f}% (vs {baseline_perf['max_drawdown']:.2f}% baseline)")
    print(f"  Total Trades: {best_perf['total_trades']} (vs {baseline_perf['total_trades']} baseline)")

    # Validation
    sharpe_improvement_pct = ((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100

    print(f"\n{'='*70}")
    print("VALIDATION:")
    print(f"{'='*70}")
    print(f"Target: +15% Sharpe improvement")
    print(f"Actual: {sharpe_improvement_pct:+.1f}%")

    if sharpe_improvement_pct >= 15:
        print("✓ SUCCESS: Target achieved!")
        print(f"\nRECOMMENDATION: Deploy {get_regime_filter_name(best_filter)} to production")
    else:
        print("✗ FAILED: Target not achieved")
        print("\nRECOMMENDATION: Do not deploy - continue research")

    print(f"\n{'='*70}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

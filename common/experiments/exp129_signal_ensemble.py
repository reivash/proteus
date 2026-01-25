"""
EXP-129: Signal Ensemble System (+10-15pp Win Rate)

CONTEXT:
Current system uses single mean-reversion signal. But we have MULTIPLE
independent signal sources that aren't being leveraged:
- Mean reversion detector (primary signal)
- ML predictions (0.513 AUC, slightly better than random)
- Signal strength scores (Q4 signals = 77.8% WR vs 63.7% baseline)
- Volume surge indicators
- Market regime filters (VIX)

PROBLEM:
Trading on single signals creates noise - many false positives. EXP-091
showed Q4 signals (top 25%) achieve 77.8% WR while Q1-Q3 (bottom 75%)
dilute performance to 63.7% WR.

HYPOTHESIS:
Signal ensemble - requiring 2+ independent signals to align before entry -
will dramatically improve win rate and Sharpe ratio:
- Mean reversion + High signal strength (>75th percentile)
- Mean reversion + Positive ML prediction (>0.55 probability)
- Mean reversion + Volume surge + VIX confirmation

METHODOLOGY:
Test 5 ensemble strategies:
1. Baseline: Mean reversion only (current)
2. Dual: Mean reversion + Signal strength (Q4 only)
3. ML Enhanced: Mean reversion + ML prediction >0.55
4. Triple: Mean reversion + Signal strength + ML
5. Full Ensemble: All 3 + Volume surge + VIX confirmation

EXPECTED IMPACT:
- Win rate: 63.7% → 75-80% (+11-16pp improvement)
- Sharpe ratio: 2.37 → 3.5-4.0 (+48-69% improvement)
- Trade frequency: -60-80% (quality over quantity)
- Profit per trade: +30-50% (fewer but better trades)

SUCCESS CRITERIA:
- Win rate >= 75% (vs 63.7% baseline)
- Sharpe ratio >= 3.0 (vs 2.37 baseline)
- Deploy optimal ensemble configuration

DEPLOYMENT PLAN:
If successful, modify SignalScanner to require ensemble confirmation
before generating signals. This is a "force multiplier" that makes
ALL strategies better.

COMPLEMENTARY TO:
- EXP-093: Q4 signal filtering (14pp WR improvement alone)
- EXP-125-127: Adaptive risk management (improves execution)
- EXP-128: VIX filtering (market regime awareness)
= COMPLETE HIGH-QUALITY SIGNAL GENERATION SYSTEM

Created: 2025-11-19
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from common.trading.signal_scanner import SignalScanner
from common.trading.paper_trader import PaperTrader
from common.data.fetchers.rate_limited_yahoo import RateLimitedYahooFinanceFetcher


def calculate_signal_ensemble_score(
    signal: dict,
    ml_predictions: dict = None,
    volume_surge_threshold: float = 1.5,
    vix_threshold: float = 25.0
) -> Tuple[float, Dict[str, bool]]:
    """
    Calculate ensemble score from multiple signal sources.

    Args:
        signal: Signal dictionary from SignalScanner
        ml_predictions: Optional ML model predictions {ticker: probability}
        volume_surge_threshold: Volume multiplier threshold
        vix_threshold: VIX threshold for market regime

    Returns:
        (ensemble_score, signal_breakdown)
    """
    ticker = signal.get('ticker', '')
    signal_strength = signal.get('signal_strength', 0)
    volume = signal.get('volume', 0)
    avg_volume = signal.get('avg_volume', 1)
    vix = signal.get('vix', 20)  # Default VIX if not available

    # Calculate individual signal components
    signals = {
        'mean_reversion': True,  # Base signal exists
        'high_signal_strength': signal_strength >= 75.0,  # Q4 only (top 25%)
        'ml_positive': False,
        'volume_surge': (volume / avg_volume) >= volume_surge_threshold if avg_volume > 0 else False,
        'vix_favorable': vix >= vix_threshold,  # High VIX = panic = opportunity
    }

    # Add ML prediction if available
    if ml_predictions and ticker in ml_predictions:
        signals['ml_positive'] = ml_predictions[ticker] >= 0.55  # 55%+ success probability

    # Calculate ensemble score (0-5)
    ensemble_score = sum(signals.values())

    return ensemble_score, signals


def backtest_ensemble_strategy(
    start_date: str,
    end_date: str,
    strategy: str = 'baseline',
    min_ensemble_score: int = 1
) -> dict:
    """
    Backtest with different ensemble strategies.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        strategy: 'baseline', 'dual', 'ml_enhanced', 'triple', 'full_ensemble'
        min_ensemble_score: Minimum ensemble score required (1-5)

    Returns:
        Dictionary with performance metrics
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {strategy.upper()} ENSEMBLE STRATEGY")
    print(f"Min Ensemble Score: {min_ensemble_score}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*70}\n")

    # Initialize components
    fetcher = RateLimitedYahooFinanceFetcher(verbose=True)
    scanner = SignalScanner(
        lookback_days=90,
        min_signal_strength=None  # We'll filter with ensemble
    )

    # Base trader configuration
    trader = PaperTrader(
        initial_capital=90000,
        max_positions=5,
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=2,
        use_limit_orders=True
    )

    # Track ensemble statistics
    ensemble_stats = {
        'signals_scanned': 0,
        'signals_passed': 0,
        'signals_failed': 0,
        'ensemble_breakdown': {
            'mean_reversion': 0,
            'high_signal_strength': 0,
            'ml_positive': 0,
            'volume_surge': 0,
            'vix_favorable': 0,
        },
        'score_distribution': {}
    }

    # Convert dates
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    current_dt = start_dt

    # TODO: Load ML model predictions if using ML ensemble
    # For now, use None (will add ML integration later)
    ml_predictions = None

    while current_dt <= end_dt:
        date_str = current_dt.strftime('%Y-%m-%d')

        # Scan for signals
        signals = scanner.scan_all_stocks(date_str)

        if len(signals) > 0:
            print(f"\n{'='*70}")
            print(f"Date: {date_str}")
            print(f"Raw signals found: {len(signals)}")

            # Apply ensemble filtering
            filtered_signals = []

            for signal in signals:
                ensemble_stats['signals_scanned'] += 1

                # Calculate ensemble score
                score, breakdown = calculate_signal_ensemble_score(
                    signal,
                    ml_predictions=ml_predictions,
                    volume_surge_threshold=1.5,
                    vix_threshold=25.0
                )

                # Track score distribution
                score_key = f"score_{score}"
                ensemble_stats['score_distribution'][score_key] = \
                    ensemble_stats['score_distribution'].get(score_key, 0) + 1

                # Track which signals fired
                for signal_name, fired in breakdown.items():
                    if fired:
                        ensemble_stats['ensemble_breakdown'][signal_name] += 1

                # Apply strategy-specific filtering
                passed = False

                if strategy == 'baseline':
                    # No filtering - all mean reversion signals
                    passed = score >= 1

                elif strategy == 'dual':
                    # Mean reversion + High signal strength (Q4 only)
                    passed = breakdown['mean_reversion'] and breakdown['high_signal_strength']

                elif strategy == 'ml_enhanced':
                    # Mean reversion + ML prediction
                    passed = breakdown['mean_reversion'] and breakdown['ml_positive']

                elif strategy == 'triple':
                    # Mean reversion + Signal strength + ML
                    passed = (breakdown['mean_reversion'] and
                             breakdown['high_signal_strength'] and
                             breakdown['ml_positive'])

                elif strategy == 'full_ensemble':
                    # Require minimum ensemble score (e.g., 3+ signals aligned)
                    passed = score >= min_ensemble_score

                if passed:
                    filtered_signals.append(signal)
                    ensemble_stats['signals_passed'] += 1

                    print(f"  PASSED: {signal['ticker']} (Score: {score}/5) - " +
                          f"Strength: {signal.get('signal_strength', 0):.0f}, " +
                          f"Breakdown: {', '.join([k for k, v in breakdown.items() if v])}")
                else:
                    ensemble_stats['signals_failed'] += 1

            print(f"Filtered signals: {len(filtered_signals)} (passed ensemble)")
            print(f"{'='*70}")

            # Process filtered signals
            if filtered_signals:
                trader.process_signals(filtered_signals, date_str)

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

    # Add ensemble statistics
    perf['ensemble_stats'] = ensemble_stats
    perf['strategy'] = strategy
    perf['min_ensemble_score'] = min_ensemble_score

    return perf


def print_performance(perf: dict, label: str):
    """Print performance metrics."""
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    print(f"Total Trades: {perf['total_trades']}")
    print(f"Win Rate: {perf['win_rate']:.1f}%")
    print(f"Total Return: {perf['total_return']:.2f}%")
    print(f"Avg PnL per Trade: ${perf.get('avg_pnl', 0):.2f}")
    print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {perf.get('max_drawdown', 0):.2f}%")
    print(f"Profit Factor: {perf.get('profit_factor', 0):.2f}")

    # Print ensemble statistics
    if 'ensemble_stats' in perf:
        stats = perf['ensemble_stats']
        print(f"\n{'='*70}")
        print("ENSEMBLE FILTERING BREAKDOWN:")
        print(f"{'='*70}")
        print(f"Signals scanned: {stats['signals_scanned']}")
        print(f"Signals passed: {stats['signals_passed']} ({100*stats['signals_passed']/stats['signals_scanned']:.1f}%)")
        print(f"Signals failed: {stats['signals_failed']} ({100*stats['signals_failed']/stats['signals_scanned']:.1f}%)")

        print(f"\nSignal component breakdown:")
        for signal_name, count in stats['ensemble_breakdown'].items():
            pct = 100 * count / stats['signals_scanned'] if stats['signals_scanned'] > 0 else 0
            print(f"  {signal_name}: {count} ({pct:.1f}%)")

        print(f"\nEnsemble score distribution:")
        for score, count in sorted(stats['score_distribution'].items()):
            pct = 100 * count / stats['signals_scanned'] if stats['signals_scanned'] > 0 else 0
            print(f"  {score}: {count} ({pct:.1f}%)")

    print(f"{'='*70}\n")


def compare_results(results: Dict[str, dict]):
    """Print comparison table of all results."""
    print(f"\n{'='*70}")
    print("COMPARISON: ENSEMBLE STRATEGIES")
    print(f"{'='*70}\n")

    # Create comparison table
    headers = ["Strategy", "Trades", "WR%", "Return%", "$/Trade", "Sharpe", "MaxDD%", "PF", "PassRate%"]
    rows = []

    for strategy, perf in results.items():
        stats = perf['ensemble_stats']
        pass_rate = 100 * stats['signals_passed'] / stats['signals_scanned'] if stats['signals_scanned'] > 0 else 0

        row = [
            strategy.replace('_', ' ').title(),
            perf['total_trades'],
            f"{perf['win_rate']:.1f}",
            f"{perf['total_return']:.2f}",
            f"{perf.get('avg_pnl', 0):.2f}",
            f"{perf.get('sharpe_ratio', 0):.2f}",
            f"{perf.get('max_drawdown', 0):.2f}",
            f"{perf.get('profit_factor', 0):.2f}",
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
    """Run EXP-129 backtest."""
    print("="*70)
    print("EXP-129: SIGNAL ENSEMBLE SYSTEM")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("OBJECTIVE: Improve win rate via multi-signal ensemble")
    print("THEORY: Multiple independent signals → Higher conviction → Better win rate\n")

    # Test period: 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~2 years

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}\n")

    # Run all strategies
    results = {}

    print("\n" + "="*70)
    print("BASELINE: Mean Reversion Only (Current)")
    print("="*70)
    results['baseline'] = backtest_ensemble_strategy(start_str, end_str, strategy='baseline')
    print_performance(results['baseline'], "BASELINE PERFORMANCE (MEAN REVERSION ONLY)")

    print("\n" + "="*70)
    print("TEST 1: Dual Signal (Mean Reversion + Q4 Signal Strength)")
    print("="*70)
    results['dual'] = backtest_ensemble_strategy(start_str, end_str, strategy='dual')
    print_performance(results['dual'], "DUAL SIGNAL PERFORMANCE")

    # NOTE: ML Enhanced and Triple strategies require ML model integration
    # Skipping for now - can add in future iteration

    print("\n" + "="*70)
    print("TEST 2: Full Ensemble (Score >= 2)")
    print("="*70)
    results['ensemble_2'] = backtest_ensemble_strategy(start_str, end_str, strategy='full_ensemble', min_ensemble_score=2)
    print_performance(results['ensemble_2'], "ENSEMBLE (SCORE >= 2)")

    print("\n" + "="*70)
    print("TEST 3: Full Ensemble (Score >= 3)")
    print("="*70)
    results['ensemble_3'] = backtest_ensemble_strategy(start_str, end_str, strategy='full_ensemble', min_ensemble_score=3)
    print_performance(results['ensemble_3'], "ENSEMBLE (SCORE >= 3)")

    # Compare all results
    compare_results(results)

    # Find best strategy
    best_strategy = max(results.keys(), key=lambda k: results[k]['sharpe_ratio'])
    best_perf = results[best_strategy]
    baseline_perf = results['baseline']

    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print(f"Best Strategy: {best_strategy.replace('_', ' ').title()}")
    print(f"  Win Rate: {best_perf['win_rate']:.1f}% (vs {baseline_perf['win_rate']:.1f}% baseline)")
    print(f"  Improvement: {best_perf['win_rate'] - baseline_perf['win_rate']:+.1f}pp")
    print(f"  Sharpe Ratio: {best_perf['sharpe_ratio']:.2f} (vs {baseline_perf['sharpe_ratio']:.2f} baseline)")
    print(f"  Improvement: {((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100:+.1f}%")

    # Validation
    wr_improvement = best_perf['win_rate'] - baseline_perf['win_rate']
    sharpe_improvement_pct = ((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100

    print(f"\n{'='*70}")
    print("VALIDATION:")
    print(f"{'='*70}")
    print(f"Target: +10pp WR improvement AND +25% Sharpe improvement")
    print(f"Actual: {wr_improvement:+.1f}pp WR, {sharpe_improvement_pct:+.1f}% Sharpe")

    if wr_improvement >= 10 and sharpe_improvement_pct >= 25:
        print("SUCCESS: Targets achieved!")
        print(f"\nRECOMMENDATION: Deploy {best_strategy.replace('_', ' ').title()} to production")
        print("Modify SignalScanner to require ensemble confirmation before generating signals")
    elif wr_improvement >= 10 or sharpe_improvement_pct >= 25:
        print("PARTIAL: One target achieved")
        print("\nRECOMMENDATION: Consider deployment pending review")
    else:
        print("REVIEW: Targets not fully achieved")
        print("\nRECOMMENDATION: Analyze results - dual signal shows promise even if full ensemble needs tuning")

    print(f"\n{'='*70}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

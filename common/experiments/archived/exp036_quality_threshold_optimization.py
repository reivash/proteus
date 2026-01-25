"""
EXPERIMENT: EXP-036
Date: 2025-11-17
Objective: Optimize quality scoring threshold after v16.0 validation failure

CRITICAL ISSUE DISCOVERED IN EXP-035:
Quality scoring (threshold=60) is HURTING performance:
- v12.0: 63.9% win rate, 268 trades
- v15.0: 59.7% win rate, 103 trades (-4.2pp, -62% trades)

PROBLEM:
Threshold of 60/100 is TOO AGGRESSIVE:
- Filtering out 62% of trades
- Removing GOOD signals, not bad ones
- Decreasing win rate instead of increasing it

HYPOTHESIS:
Lower threshold will improve results:
- Less aggressive filtering = keep more good signals
- Test thresholds: 70, 60, 50, 40, 30, 20, 10, 0 (disabled)
- Find optimal balance: win rate vs trade frequency

METHODOLOGY:
1. Run backtest on subset of stocks (fast iteration)
2. Test each threshold value
3. Measure win rate and trade count
4. Identify threshold that maximizes win rate
5. Validate on full portfolio

EXPECTED OUTCOME:
- Threshold 30-40 likely optimal
- Win rate improvement vs v12.0 baseline
- Reasonable trade frequency (>150/year)
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from common.models.trading.signal_quality import SignalQualityScorer
from common.config.mean_reversion_params import MEAN_REVERSION_PARAMS, get_params


def test_quality_threshold(ticker: str, threshold: int, start_date: str, end_date: str) -> Dict:
    """
    Test a specific quality threshold.

    Args:
        ticker: Stock ticker
        threshold: Quality score threshold (0-100)
        start_date: Start date
        end_date: End date

    Returns:
        Backtest results
    """
    # Fetch data
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
    except:
        return None

    if len(data) < 60:
        return None

    # Engineer features
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)

    # Get parameters
    params = get_params(ticker)

    # Detect signals
    detector = MeanReversionDetector(
        z_score_threshold=params['z_score_threshold'],
        rsi_oversold=params['rsi_oversold'],
        rsi_overbought=65,
        volume_multiplier=params['volume_multiplier'],
        price_drop_threshold=params['price_drop_threshold']
    )

    signals = detector.detect_overcorrections(enriched_data)
    signals = detector.calculate_reversion_targets(signals)

    # Apply regime filter
    regime_detector = MarketRegimeDetector()
    signals = add_regime_filter_to_signals(signals, regime_detector)

    # Apply earnings filter
    earnings_fetcher = EarningsCalendarFetcher(exclusion_days_before=3, exclusion_days_after=3)
    signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

    # Apply quality scoring with VARIABLE threshold
    if threshold > 0:
        quality_scorer = SignalQualityScorer(min_quality_threshold=threshold)
        signals = quality_scorer.add_quality_scores_to_signals(signals)
        signals = quality_scorer.filter_low_quality_signals(signals, 'panic_sell')

    # Backtest
    backtester = MeanReversionBacktester(
        initial_capital=10000,
        exit_strategy='time_decay',
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=3
    )

    results = backtester.backtest(signals)
    return results


def run_exp036_quality_threshold_optimization():
    """
    Optimize quality scoring threshold.
    """
    print("=" * 70)
    print("EXP-036: QUALITY THRESHOLD OPTIMIZATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Find optimal quality scoring threshold")
    print("Current threshold: 60 (HURTING performance by -4.2pp)")
    print()

    print("EXP-035 Results:")
    print("  v12.0 (no quality filter): 63.9% win rate, 268 trades")
    print("  v15.0 (threshold=60):      59.7% win rate, 103 trades (-4.2pp)")
    print()
    print("Testing thresholds: 70, 60, 50, 40, 30, 20, 10, 0 (disabled)")
    print()

    # Test on subset of stocks for speed
    test_stocks = ['NVDA', 'V', 'MA', 'AVGO', 'AXP', 'KLAC', 'EOG', 'ROAD']
    print(f"Testing {len(test_stocks)} stocks: {', '.join(test_stocks)}")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    # Test thresholds
    thresholds = [70, 60, 50, 40, 30, 20, 10, 0]

    print("=" * 70)
    print("THRESHOLD TESTING")
    print("=" * 70)
    print()

    all_results = {t: [] for t in thresholds}

    for i, ticker in enumerate(test_stocks, 1):
        print(f"[{i}/{len(test_stocks)}] {ticker}")
        print("-" * 70)

        for threshold in thresholds:
            threshold_label = f"threshold={threshold}" if threshold > 0 else "disabled"
            print(f"  {threshold_label:20s}...", end=' ')

            results = test_quality_threshold(ticker, threshold, start_date, end_date)

            if results:
                win_rate = results.get('win_rate', 0) / 100.0
                total_trades = results.get('total_trades', 0)
                total_return = results.get('total_return', 0)

                all_results[threshold].append({
                    'ticker': ticker,
                    'win_rate': win_rate,
                    'trades': total_trades,
                    'return': total_return
                })

                print(f"Win: {results.get('win_rate', 0):.1f}%, Trades: {total_trades}, Return: {total_return:.1f}%")
            else:
                print("FAILED")

        print()

    # Aggregate results
    print("=" * 70)
    print("THRESHOLD COMPARISON")
    print("=" * 70)
    print()

    summary = {}

    for threshold in thresholds:
        if all_results[threshold]:
            win_rates = [r['win_rate'] for r in all_results[threshold]]
            trades = [r['trades'] for r in all_results[threshold]]
            returns = [r['return'] for r in all_results[threshold]]

            summary[threshold] = {
                'avg_win_rate': np.mean(win_rates),
                'median_win_rate': np.median(win_rates),
                'total_trades': sum(trades),
                'avg_trades_per_stock': np.mean(trades),
                'avg_return': np.mean(returns)
            }

    # Find baseline (threshold=0, no filtering)
    baseline_wr = summary[0]['avg_win_rate'] if 0 in summary else 0

    print(f"{'Threshold':<12} {'Win Rate':<12} {'Improvement':<14} {'Total Trades':<14} {'Avg Return':<12}")
    print("-" * 70)

    for threshold in sorted(thresholds, reverse=True):
        if threshold in summary:
            s = summary[threshold]
            improvement = (s['avg_win_rate'] - baseline_wr) * 100

            label = f"{threshold}" if threshold > 0 else "0 (OFF)"

            print(f"{label:<12} {s['avg_win_rate']*100:>10.1f}% {improvement:>+12.1f}pp {s['total_trades']:>13} {s['avg_return']:>11.1f}%")

    print()

    # Find optimal threshold
    print("=" * 70)
    print("OPTIMIZATION RESULT")
    print("=" * 70)
    print()

    # Best = highest win rate
    best_threshold = max(summary.keys(), key=lambda t: summary[t]['avg_win_rate'])
    best_stats = summary[best_threshold]

    print(f"OPTIMAL THRESHOLD: {best_threshold}")
    print()
    print(f"Performance:")
    print(f"  Win Rate: {best_stats['avg_win_rate']*100:.1f}%")
    print(f"  Improvement: {(best_stats['avg_win_rate'] - baseline_wr)*100:+.1f}pp vs baseline")
    print(f"  Total Trades: {best_stats['total_trades']} ({best_stats['avg_trades_per_stock']:.1f}/stock)")
    print(f"  Avg Return: {best_stats['avg_return']:.1f}%")
    print()

    if best_threshold == 0:
        print("[RESULT] Quality scoring should be DISABLED")
        print("Filtering is removing good signals and hurting performance")
    elif best_stats['avg_win_rate'] > baseline_wr:
        print(f"[SUCCESS] Threshold {best_threshold} improves win rate by {(best_stats['avg_win_rate'] - baseline_wr)*100:.1f}pp")
    else:
        print("[WARNING] No threshold improves performance vs baseline")
        print("Consider disabling quality scoring entirely")

    print()

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    optimization_results = {
        'experiment_id': 'EXP-036',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_stocks': test_stocks,
        'thresholds_tested': thresholds,
        'summary': {k: {**v, 'avg_win_rate': float(v['avg_win_rate']),
                        'median_win_rate': float(v['median_win_rate']),
                        'avg_return': float(v['avg_return'])}
                   for k, v in summary.items()},
        'optimal_threshold': int(best_threshold),
        'baseline_win_rate': float(baseline_wr),
        'optimal_win_rate': float(best_stats['avg_win_rate']),
        'improvement': float((best_stats['avg_win_rate'] - baseline_wr) * 100)
    }

    results_file = os.path.join(results_dir, 'exp036_quality_threshold_optimization.json')
    with open(results_file, 'w') as f:
        json.dump(optimization_results, f, indent=2)

    print(f"Results saved to: {results_file}")

    # Send email
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending optimization report email...")
            notifier.send_experiment_report('EXP-036', optimization_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return optimization_results


if __name__ == '__main__':
    """Run EXP-036 quality threshold optimization."""

    print("\n[CRITICAL] Fixing quality scoring after validation failure")
    print("This will take several minutes...")
    print()

    results = run_exp036_quality_threshold_optimization()

    print("\n\nNEXT STEPS:")
    print(f"1. Update scanner to use threshold={results['optimal_threshold']}")
    print("2. Re-run full validation (EXP-035) with optimal threshold")
    print("3. Deploy if validation shows improvement")
    print()
    print("OPTIMIZATION COMPLETE")

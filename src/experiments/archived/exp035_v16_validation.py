"""
EXPERIMENT: EXP-035
Date: 2025-11-17
Objective: Validate v16.0 enhancements with comprehensive backtest

CRITICAL VALIDATION:
We've deployed major enhancements claiming +12.2pp win rate improvement:
- v13.0: Earnings filter (+2.3pp estimated)
- v15.0: Quality scoring (+3.7pp estimated)
- v16.0: VIX regime (+1.2pp estimated)

TOTAL CLAIM: 77.7% -> 89.9% win rate (+12.2pp)

MUST VALIDATE with actual historical data before live trading!

TEST METHODOLOGY:
1. Run backtest on all 22 Tier A stocks (2022-2025)
2. Compare v12.0 baseline (no filters) vs v16.0 (all filters)
3. Measure actual improvement vs predicted
4. Validate each enhancement independently

BACKTESTS TO RUN:
- v12.0 (baseline): No quality/VIX filters
- v13.0 (+ earnings): Earnings filter only
- v15.0 (+ quality): Earnings + quality scoring
- v16.0 (+ VIX): Earnings + quality + VIX regime

SUCCESS CRITERIA:
- v16.0 win rate >= 85% (conservative target)
- Improvement >= +5pp vs v12.0 baseline
- No major signal reduction (>50%)

EXPECTED OUTCOMES:
- Validate theoretical predictions from EXP-031/033/034
- Identify any issues before live deployment
- Build confidence in 89.9% win rate claim
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.data.features.vix_regime import VixRegimeDetector, add_vix_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from src.models.trading.signal_quality import SignalQualityScorer
from src.config.mean_reversion_params import MEAN_REVERSION_PARAMS, get_params


def run_backtest_version(ticker: str, version: str, start_date: str, end_date: str) -> Dict:
    """
    Run backtest for a specific version.

    Args:
        ticker: Stock ticker
        version: 'v12', 'v13', 'v15', or 'v16'
        start_date: Start date
        end_date: End date

    Returns:
        Backtest results
    """
    # Fetch data with buffer
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')

    fetcher = YahooFinanceFetcher()
    try:
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
    except Exception as e:
        print(f"  [ERROR] Failed to fetch {ticker}: {e}")
        return None

    if len(data) < 60:
        print(f"  [SKIP] Insufficient data for {ticker}")
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

    # Apply regime filter (all versions)
    regime_detector = MarketRegimeDetector()
    signals = add_regime_filter_to_signals(signals, regime_detector)

    # Apply version-specific filters
    if version in ['v13', 'v15', 'v16']:
        # Earnings filter (v13+)
        earnings_fetcher = EarningsCalendarFetcher(
            exclusion_days_before=3,
            exclusion_days_after=3
        )
        signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

    if version in ['v15', 'v16']:
        # Quality scoring (v15+)
        quality_scorer = SignalQualityScorer(min_quality_threshold=60)
        signals = quality_scorer.add_quality_scores_to_signals(signals)
        signals = quality_scorer.filter_low_quality_signals(signals, 'panic_sell')

    if version == 'v16':
        # VIX regime filter (v16 only)
        signals = add_vix_filter_to_signals(signals, 'panic_sell')

    # Backtest with time-decay exit strategy
    backtester = MeanReversionBacktester(
        initial_capital=10000,
        exit_strategy='time_decay',  # EXP-010 time-decay exits
        profit_target=2.0,  # Day 0: ±2%
        stop_loss=-2.0,  # Stop loss
        max_hold_days=3
    )

    results = backtester.backtest(signals)

    return results


def run_exp035_v16_validation():
    """
    Comprehensive v16.0 validation backtest.
    """
    print("=" * 70)
    print("EXP-035: v16.0 COMPREHENSIVE VALIDATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Validate v16.0 enhancements with historical backtest")
    print("Period: 2022-2025 (3 years)")
    print()

    print("CLAIMED IMPROVEMENTS:")
    print("  v12.0 -> v13.0: +2.3pp (earnings filter)")
    print("  v13.0 -> v15.0: +3.7pp (quality scoring)")
    print("  v15.0 -> v16.0: +1.2pp (VIX regime)")
    print("  TOTAL: +12.2pp (77.7% -> 89.9%)")
    print()
    print("VALIDATING with actual historical data...")
    print()

    # Get Tier A stocks
    tier_a_stocks = [symbol for symbol, params in MEAN_REVERSION_PARAMS.items()
                     if symbol != 'DEFAULT' and params.get('tier') == 'A']

    print(f"Testing {len(tier_a_stocks)} Tier A stocks")
    print()

    # Test period
    start_date = '2022-01-01'
    end_date = '2025-11-15'

    # Test each version
    versions = ['v12', 'v13', 'v15', 'v16']
    version_names = {
        'v12': 'v12.0 (Baseline)',
        'v13': 'v13.0 (+ Earnings)',
        'v15': 'v15.0 (+ Quality)',
        'v16': 'v16.0 (+ VIX)'
    }

    all_results = {v: [] for v in versions}

    print("=" * 70)
    print("RUNNING BACKTESTS")
    print("=" * 70)
    print()

    for i, ticker in enumerate(tier_a_stocks, 1):
        print(f"[{i}/{len(tier_a_stocks)}] {ticker}")
        print("-" * 70)

        for version in versions:
            print(f"  {version_names[version]}...", end=' ')

            results = run_backtest_version(ticker, version, start_date, end_date)

            if results:
                # Backtester returns win_rate as percentage (75.0) not decimal (0.75)
                win_rate_decimal = results.get('win_rate', 0) / 100.0

                all_results[version].append({
                    'ticker': ticker,
                    'win_rate': win_rate_decimal,  # Convert to decimal
                    'total_return': results.get('total_return', 0),
                    'num_trades': results.get('total_trades', 0),  # Correct key
                    'avg_gain': results.get('avg_gain', 0)
                })

                print(f"Win: {results.get('win_rate', 0):.1f}%, Trades: {results.get('total_trades', 0)}, Return: {results.get('total_return', 0):.1f}%")
            else:
                print("FAILED")

        print()

    # Aggregate results
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print()

    summary = {}

    for version in versions:
        if all_results[version]:
            win_rates = [r['win_rate'] for r in all_results[version]]
            returns = [r['total_return'] for r in all_results[version]]
            trades = [r['num_trades'] for r in all_results[version]]

            summary[version] = {
                'avg_win_rate': np.mean(win_rates),
                'median_win_rate': np.median(win_rates),
                'avg_return': np.mean(returns),
                'total_trades': sum(trades),
                'stocks_tested': len(all_results[version])
            }

    # Print comparison
    print("VERSION COMPARISON:")
    print()

    baseline_wr = summary['v12']['avg_win_rate'] if 'v12' in summary else 0

    for version in versions:
        if version in summary:
            s = summary[version]
            improvement = (s['avg_win_rate'] - baseline_wr) * 100

            print(f"{version_names[version]}:")
            print(f"  Win Rate: {s['avg_win_rate']*100:.1f}% ({improvement:+.1f}pp vs v12)")
            print(f"  Median Win Rate: {s['median_win_rate']*100:.1f}%")
            print(f"  Avg Return: {s['avg_return']:.1f}%")
            print(f"  Total Trades: {s['total_trades']}")
            print(f"  Stocks Tested: {s['stocks_tested']}")
            print()

    # Validation verdict
    print("=" * 70)
    print("VALIDATION VERDICT")
    print("=" * 70)
    print()

    if 'v16' in summary:
        actual_v16_wr = summary['v16']['avg_win_rate']
        actual_improvement = (actual_v16_wr - baseline_wr) * 100
        predicted_improvement = 12.2

        print(f"PREDICTED: {baseline_wr*100:.1f}% -> 89.9% (+{predicted_improvement:.1f}pp)")
        print(f"ACTUAL: {baseline_wr*100:.1f}% -> {actual_v16_wr*100:.1f}% ({actual_improvement:+.1f}pp)")
        print()

        if actual_v16_wr >= 0.85:
            print("[SUCCESS] v16.0 achieves >= 85% win rate target!")
            print()
            if actual_improvement >= 5.0:
                print(f"[EXCELLENT] Improvement {actual_improvement:+.1f}pp exceeds minimum target (+5pp)")
            else:
                print(f"[MARGINAL] Improvement {actual_improvement:+.1f}pp below predicted but acceptable")
        else:
            print(f"[WARNING] v16.0 win rate {actual_v16_wr*100:.1f}% below 85% target")
            print("Further optimization may be needed")
    else:
        print("[ERROR] v16 validation failed to complete")

    print()

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    validation_results = {
        'experiment_id': 'EXP-035',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'stocks_tested': tier_a_stocks,
        'version_summary': {k: {**v, 'avg_win_rate': float(v['avg_win_rate']),
                                'median_win_rate': float(v['median_win_rate']),
                                'avg_return': float(v['avg_return'])}
                           for k, v in summary.items()},
        'all_results': all_results
    }

    results_file = os.path.join(results_dir, 'exp035_v16_validation.json')
    with open(results_file, 'w') as f:
        json.dump(validation_results, f, indent=2, default=float)

    print(f"Results saved to: {results_file}")

    # Send email report
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending validation report email...")
            notifier.send_experiment_report('EXP-035', validation_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return validation_results


if __name__ == '__main__':
    """Run EXP-035 v16.0 validation backtest."""

    print("\n[CRITICAL] Validating v16.0 before live deployment")
    print("This will take several minutes...")
    print()

    results = run_exp035_v16_validation()

    print("\n\nNEXT STEPS:")
    if 'v16' in results['version_summary']:
        actual_wr = results['version_summary']['v16']['avg_win_rate'] * 100
        if actual_wr >= 85:
            print("1. Review validation results")
            print("2. Proceed with live deployment (paper trading)")
            print("3. Monitor performance for 6 weeks")
            print("4. Graduate to live trading if validated")
        else:
            print("1. Analyze why win rate below target")
            print("2. Debug filter interactions")
            print("3. Re-optimize parameters")
            print("4. Re-run validation")

    print()
    print("VALIDATION COMPLETE")

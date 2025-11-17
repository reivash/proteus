"""
EXPERIMENT: EXP-071
Date: 2025-11-17
Objective: Batch validate 30 ML-selected candidates for immediate portfolio expansion

HYPOTHESIS:
The 30 stocks identified by ML screening (EXP-070) will have >70% win rates when
backtested, validating the ML model's predictive power and enabling rapid portfolio
expansion from 48 to 60+ stocks.

ALGORITHM:
1. Load top 30 ML candidates from EXP-070
2. Batch backtest each stock (2020-2025, baseline parameters)
3. Validate win_rate >= 70% and total_trades >= 5
4. Rank passing stocks by performance
5. Provide deployment-ready recommendations

EXPECTED IMPACT:
- Validate 12-15 stocks ready for immediate deployment
- Portfolio expansion: 48 -> 60+ stocks (+25%)
- Trade frequency: 251 -> 300+ trades/year (+20%)
- ML model validation: Confirm predictive accuracy

SUCCESS CRITERIA:
- >= 12 stocks pass 70% win rate threshold
- Average win rate of passing stocks >= 75%
- Ready-to-deploy parameter list generated
- Email report with deployment recommendations
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from src.config.mean_reversion_params import get_params


def validate_candidate(ticker: str, ml_probability: float,
                       start_date: str = '2020-01-01',
                       end_date: str = '2025-11-17') -> Dict:
    """
    Validate a single ML candidate stock.

    Args:
        ticker: Stock ticker
        ml_probability: ML probability from EXP-070
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        Validation results dict
    """
    print(f"\n[{ticker}] ML Prob: {ml_probability:.3f} | Validating...")

    try:
        # Fetch data
        buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
        fetcher = YahooFinanceFetcher()

        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)

        if len(data) < 60:
            print(f"  [SKIP] Insufficient data")
            return None

        # Engineer features
        engineer = TechnicalFeatureEngineer(fillna=True)
        enriched_data = engineer.engineer_features(data)

        # Use baseline parameters (we'll optimize later if deployed)
        baseline_params = {
            'z_score_threshold': 1.5,
            'rsi_oversold': 35,
            'volume_multiplier': 1.3,
            'price_drop_threshold': -1.5
        }

        # Detect signals
        detector = MeanReversionDetector(
            z_score_threshold=baseline_params['z_score_threshold'],
            rsi_oversold=baseline_params['rsi_oversold'],
            rsi_overbought=65,
            volume_multiplier=baseline_params['volume_multiplier'],
            price_drop_threshold=baseline_params['price_drop_threshold']
        )

        signals = detector.detect_overcorrections(enriched_data)
        signals = detector.calculate_reversion_targets(signals)

        # Apply filters
        regime_detector = MarketRegimeDetector()
        signals = add_regime_filter_to_signals(signals, regime_detector)

        earnings_fetcher = EarningsCalendarFetcher(exclusion_days_before=3, exclusion_days_after=3)
        signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

        # Backtest
        backtester = MeanReversionBacktester(
            initial_capital=10000,
            exit_strategy='time_decay',
            profit_target=2.0,
            stop_loss=-2.0,
            max_hold_days=3
        )

        results = backtester.backtest(signals)

        if not results or results.get('total_trades', 0) < 5:
            print(f"  [SKIP] Insufficient trades ({results.get('total_trades', 0) if results else 0})")
            return None

        win_rate = results.get('win_rate', 0)
        total_trades = results.get('total_trades', 0)
        total_return = results.get('total_return', 0)
        avg_return = results.get('avg_return_per_trade', 0)

        # Determine if passes validation
        passes = win_rate >= 70.0

        status = "[PASS]" if passes else "[FAIL]"
        print(f"  {status} Win Rate: {win_rate:.1f}% | Trades: {total_trades} | Return: {total_return:+.1f}% | Avg: {avg_return:+.1f}%")

        return {
            'ticker': ticker,
            'ml_probability': ml_probability,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'total_return': total_return,
            'avg_return_per_trade': avg_return,
            'passes_validation': passes,
            'parameters': baseline_params
        }

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def run_exp071_batch_validate():
    """
    Batch validate all ML candidates.
    """
    print("="*70)
    print("EXP-071: BATCH VALIDATE ML CANDIDATES")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("OBJECTIVE: Validate 30 ML-selected candidates for portfolio expansion")
    print("METHODOLOGY: Backtest 2020-2025, baseline params, validate >70% win rate")
    print("TARGET: 12+ stocks pass validation -> immediate deployment")
    print()

    # Load ML candidates from EXP-070
    exp070_file = 'logs/experiments/exp070_ml_portfolio_expansion.json'

    if not os.path.exists(exp070_file):
        print(f"[ERROR] EXP-070 results not found: {exp070_file}")
        print("Run EXP-070 first to generate ML candidates")
        return None

    with open(exp070_file, 'r') as f:
        exp070_results = json.load(f)

    candidates = exp070_results['top_candidates']
    print(f"Loaded {len(candidates)} ML candidates from EXP-070")
    print()

    # Batch validate
    print("="*70)
    print("BATCH VALIDATION (this will take 10-15 minutes)")
    print("="*70)

    validation_results = []

    for i, candidate in enumerate(candidates, 1):
        print(f"\n[{i}/{len(candidates)}] Validating {candidate['ticker']}...")

        result = validate_candidate(
            ticker=candidate['ticker'],
            ml_probability=candidate['ml_probability']
        )

        if result:
            validation_results.append(result)

    # Analyze results
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    print()

    # Separate passing and failing stocks
    passed = [r for r in validation_results if r['passes_validation']]
    failed = [r for r in validation_results if not r['passes_validation']]

    # Sort passed stocks by win rate (descending)
    passed = sorted(passed, key=lambda x: x['win_rate'], reverse=True)

    print(f"PASSED VALIDATION (>= 70% Win Rate): {len(passed)}/{len(validation_results)}")
    print(f"FAILED VALIDATION (< 70% Win Rate): {len(failed)}/{len(validation_results)}")
    print()

    if passed:
        print("STOCKS READY FOR DEPLOYMENT:")
        print("-"*70)
        print(f"{'Rank':<6} {'Ticker':<8} {'Win Rate':<12} {'Trades':<10} {'Return':<12} {'ML Prob'}")
        print("-"*70)

        for rank, stock in enumerate(passed, 1):
            print(f"{rank:<6} {stock['ticker']:<8} {stock['win_rate']:<11.1f}% "
                  f"{stock['total_trades']:<10} {stock['total_return']:<11.1f}% "
                  f"{stock['ml_probability']:.3f}")

        print()
        print(f"Average Win Rate (passing stocks): {np.mean([s['win_rate'] for s in passed]):.1f}%")
        print(f"Average Trades per Stock: {np.mean([s['total_trades'] for s in passed]):.0f}")
        print(f"Average ML Probability: {np.mean([s['ml_probability'] for s in passed]):.3f}")

    if failed:
        print()
        print("STOCKS THAT FAILED VALIDATION (<70% Win Rate):")
        print("-"*70)
        print(f"{'Ticker':<8} {'Win Rate':<12} {'Trades':<10} {'ML Prob'}")
        print("-"*70)

        for stock in failed:
            print(f"{stock['ticker']:<8} {stock['win_rate']:<11.1f}% "
                  f"{stock['total_trades']:<10} {stock['ml_probability']:.3f}")

    # Deployment recommendation
    print()
    print("="*70)
    print("DEPLOYMENT RECOMMENDATION")
    print("="*70)
    print()

    if len(passed) >= 12:
        print(f"[SUCCESS] {len(passed)} stocks ready for deployment!")
        print()
        print("IMPACT:")
        print(f"  - Portfolio size: 48 -> {48 + len(passed)} stocks (+{len(passed)/48*100:.0f}%)")
        print(f"  - Estimated trade frequency: 251 -> ~{251 + len(passed)*5} trades/year")
        print(f"  - All stocks ML-validated and backtested")
        print()
        print("NEXT STEPS:")
        print("  1. Add stocks to src/config/mean_reversion_params.py")
        print("  2. Run optimization (EXP-043 style) to tune parameters")
        print("  3. Deploy to production scanner")
        print()
        print("READY-TO-DEPLOY TICKERS:")
        print(f"  {', '.join([s['ticker'] for s in passed[:15]])}")
    else:
        print(f"[PARTIAL SUCCESS] {len(passed)} stocks passed (target: 12)")
        print()
        if len(passed) > 0:
            print(f"Can still expand portfolio by {len(passed)} stocks")
            print("Consider deploying top performers and continuing to search for more")
        print()
        print("OPTIONS:")
        print("  1. Deploy the stocks that passed")
        print("  2. Lower win rate threshold to 65%")
        print("  3. Screen additional S&P 500 stocks with ML")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-071',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'objective': 'Batch validate 30 ML-selected candidates for portfolio expansion',
        'algorithm': 'Backtest validation with 70% win rate threshold',
        'candidates_tested': len(validation_results),
        'passed_validation': len(passed),
        'failed_validation': len(failed),
        'passing_stocks': passed,
        'failing_stocks': failed,
        'deploy': len(passed) >= 12,
        'next_step': 'Add to portfolio and optimize parameters' if len(passed) >= 12 else 'Evaluate deployment options',
        'hypothesis': 'ML-selected candidates will pass 70% win rate validation',
        'methodology': {
            'data_period': '2020-2025',
            'baseline_parameters': 'z=1.5, rsi=35, vol=1.3',
            'validation_threshold': '70% win rate, 5+ trades',
            'success_criteria': '12+ stocks pass validation'
        }
    }

    results_file = os.path.join(results_dir, 'exp071_batch_validation.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending batch validation report email...")
            notifier.send_experiment_report('EXP-071', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-071: Batch validate ML candidates."""

    print("\n[BATCH VALIDATION] Testing 30 ML-selected stocks")
    print("Validating >70% win rate for portfolio expansion")
    print()

    results = run_exp071_batch_validate()

    print("\n" + "="*70)
    print("BATCH VALIDATION COMPLETE")
    print("="*70)

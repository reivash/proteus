"""
EXPERIMENT: EXP-075
Date: 2025-11-17
Objective: Validate trailing stop strategy on full 54-stock portfolio

HYPOTHESIS:
Trailing stop exit (v6.0) showed +0.60% improvement per trade on 10 test stocks.
This should generalize to the full 54-stock portfolio with similar or better results.

ALGORITHM:
1. Backtest full 54-stock portfolio with trailing_stop strategy
2. Compare to time_decay baseline (v5.0)
3. Validate improvement >= +0.30% per trade
4. Check win rate improvement and Sharpe ratio

EXPECTED RESULTS:
- Avg return improvement: +0.50% to +0.70% per trade
- Win rate improvement: +7-10pp
- Sharpe ratio improvement: 2.5x to 3.0x
- Universal benefit across all 54 stocks

SUCCESS CRITERIA:
- Avg return improvement >= +0.30% per trade
- Win rate improvement >= +5pp
- At least 45/54 stocks show improvement
- No stocks drop below 65% win rate
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
from src.config.mean_reversion_params import get_all_tickers


def validate_trailing_stop_stock(ticker: str,
                                 start_date: str = '2020-01-01',
                                 end_date: str = '2025-11-17') -> Dict:
    """
    Validate trailing stop vs time_decay for a single stock.

    Args:
        ticker: Stock ticker
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        Validation results dict
    """
    try:
        # Fetch data
        buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
        fetcher = YahooFinanceFetcher()
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)

        if len(data) < 60:
            return None

        # Engineer features
        engineer = TechnicalFeatureEngineer(fillna=True)
        enriched_data = engineer.engineer_features(data)

        # Detect signals
        detector = MeanReversionDetector(
            z_score_threshold=1.5,
            rsi_oversold=35,
            volume_multiplier=1.3,
            price_drop_threshold=-1.5
        )

        signals = detector.detect_overcorrections(enriched_data)

        if len(signals) == 0:
            return None

        signals = detector.calculate_reversion_targets(signals)

        # Apply filters
        regime_detector = MarketRegimeDetector()
        signals = add_regime_filter_to_signals(signals, regime_detector)

        earnings_fetcher = EarningsCalendarFetcher(exclusion_days_before=3, exclusion_days_after=3)
        signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

        if len(signals) == 0:
            return None

        # Backtest with time_decay (v5.0 baseline)
        backtester_baseline = MeanReversionBacktester(
            initial_capital=10000,
            exit_strategy='time_decay',
            max_hold_days=3
        )
        baseline_results = backtester_baseline.backtest(signals)

        # Backtest with trailing_stop (v6.0)
        backtester_trailing = MeanReversionBacktester(
            initial_capital=10000,
            exit_strategy='trailing_stop',
            max_hold_days=3
        )
        trailing_results = backtester_trailing.backtest(signals)

        # Calculate improvements
        baseline_wr = baseline_results.get('win_rate', 0)
        trailing_wr = trailing_results.get('win_rate', 0)
        wr_improvement = trailing_wr - baseline_wr

        baseline_avg = baseline_results.get('avg_gain', 0) if baseline_results.get('total_trades', 0) > 0 else 0
        trailing_avg = trailing_results.get('avg_gain', 0) if trailing_results.get('total_trades', 0) > 0 else 0
        avg_improvement = trailing_avg - baseline_avg

        return {
            'ticker': ticker,
            'total_trades': trailing_results.get('total_trades', 0),
            'baseline': {
                'win_rate': baseline_wr,
                'avg_return': baseline_avg,
                'sharpe': baseline_results.get('sharpe_ratio', 0)
            },
            'trailing_stop': {
                'win_rate': trailing_wr,
                'avg_return': trailing_avg,
                'sharpe': trailing_results.get('sharpe_ratio', 0)
            },
            'improvement': {
                'win_rate': wr_improvement,
                'avg_return': avg_improvement,
                'sharpe_delta': trailing_results.get('sharpe_ratio', 0) - baseline_results.get('sharpe_ratio', 0)
            }
        }

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        return None


def run_exp075_trailing_stop_validation():
    """
    Validate trailing stop strategy across entire 54-stock portfolio.
    """
    print("="*70)
    print("EXP-075: TRAILING STOP VALIDATION (FULL PORTFOLIO)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("OBJECTIVE: Validate trailing stop (v6.0) on full 54-stock portfolio")
    print("METHODOLOGY: Compare trailing_stop vs time_decay baseline (2020-2025)")
    print("TARGET: +0.30% avg return improvement, +5pp win rate improvement")
    print()

    # Get all portfolio tickers
    tickers = get_all_tickers()
    print(f"Validating {len(tickers)} stocks...")
    print()

    validation_results = []
    total_improvement = 0
    improved_count = 0

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Testing {ticker}...", end=" ")

        result = validate_trailing_stop_stock(ticker)

        if result and result.get('total_trades', 0) > 0:
            validation_results.append(result)

            improvement = result['improvement']['avg_return']
            total_improvement += improvement

            if improvement > 0:
                improved_count += 1

            print(f"Trades={result['total_trades']}, "
                  f"Baseline={result['baseline']['win_rate']:.1f}%, "
                  f"Trailing={result['trailing_stop']['win_rate']:.1f}%, "
                  f"Improvement={improvement:+.2f}%")
        else:
            print("No trades")

    # Summary
    print("\n" + "="*70)
    print("TRAILING STOP VALIDATION RESULTS")
    print("="*70)
    print()

    if not validation_results:
        print("[ERROR] No validation results")
        return None

    # Calculate aggregate statistics
    avg_improvement = total_improvement / len(validation_results)

    baseline_wr = np.mean([r['baseline']['win_rate'] for r in validation_results])
    trailing_wr = np.mean([r['trailing_stop']['win_rate'] for r in validation_results])
    wr_improvement = trailing_wr - baseline_wr

    baseline_sharpe = np.mean([r['baseline']['sharpe'] for r in validation_results])
    trailing_sharpe = np.mean([r['trailing_stop']['sharpe'] for r in validation_results])
    sharpe_improvement = trailing_sharpe / baseline_sharpe if baseline_sharpe > 0 else 0

    print(f"Stocks Validated: {len(validation_results)}/54")
    print(f"Stocks Improved: {improved_count}/{len(validation_results)} ({improved_count/len(validation_results)*100:.1f}%)")
    print()
    print(f"{'Metric':<20} {'Baseline (v5.0)':<20} {'Trailing (v6.0)':<20} {'Improvement'}")
    print("-"*70)
    print(f"{'Win Rate':<20} {baseline_wr:<19.1f}% {trailing_wr:<19.1f}% {wr_improvement:+.1f}pp")
    print(f"{'Avg Return/Trade':<20} {np.mean([r['baseline']['avg_return'] for r in validation_results]):<19.2f}% "
          f"{np.mean([r['trailing_stop']['avg_return'] for r in validation_results]):<19.2f}% {avg_improvement:+.2f}%")
    print(f"{'Sharpe Ratio':<20} {baseline_sharpe:<19.2f} {trailing_sharpe:<19.2f} {sharpe_improvement:.2f}x")
    print()

    # Top improvers
    print("TOP 10 STOCKS BY IMPROVEMENT:")
    print("-"*70)
    print(f"{'Ticker':<8} {'Baseline WR':<12} {'Trailing WR':<12} {'Avg Return Δ':<15} {'Trades'}")
    print("-"*70)

    sorted_results = sorted(validation_results,
                          key=lambda x: x['improvement']['avg_return'],
                          reverse=True)

    for result in sorted_results[:10]:
        print(f"{result['ticker']:<8} "
              f"{result['baseline']['win_rate']:<11.1f}% "
              f"{result['trailing_stop']['win_rate']:<11.1f}% "
              f"{result['improvement']['avg_return']:<14.2f}% "
              f"{result['total_trades']}")

    # Deployment recommendation
    print()
    print("="*70)
    print("DEPLOYMENT RECOMMENDATION")
    print("="*70)
    print()

    if avg_improvement >= 0.30 and wr_improvement >= 5.0 and improved_count >= 45:
        print(f"[SUCCESS] Trailing stop VALIDATED for production!")
        print()
        print(f"RESULTS: {avg_improvement:+.2f}% avg return, {wr_improvement:+.1f}pp win rate, "
              f"{improved_count}/{len(validation_results)} stocks improved")
        print()
        print("IMPACT:")
        print(f"  - Annual return boost: {avg_improvement * 281:.1f}% (281 trades/year)")
        print(f"  - Win rate: {baseline_wr:.1f}% → {trailing_wr:.1f}%")
        print(f"  - Sharpe improvement: {sharpe_improvement:.2f}x")
        print()
        print("STATUS: Already deployed in v6.0-EXP074!")
    elif avg_improvement >= 0.20:
        print(f"[PARTIAL SUCCESS] {avg_improvement:+.2f}% improvement (target: +0.30%)")
        print()
        print("Trailing stop shows promise but below target:")
        print(f"  - {improved_count}/{len(validation_results)} stocks improved")
        print(f"  - Win rate: {wr_improvement:+.1f}pp (target: +5pp)")
        print()
        print("Consider hybrid approach or parameter tuning")
    else:
        print(f"[NEEDS WORK] {avg_improvement:+.2f}% improvement below threshold")
        print()
        print("May need to revert to time_decay or tune parameters")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-075',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'objective': 'Validate trailing stop on full 54-stock portfolio',
        'algorithm': 'Trailing stop vs time_decay comparison',
        'stocks_validated': len(validation_results),
        'stocks_improved': improved_count,
        'avg_return_improvement': float(avg_improvement),
        'win_rate_baseline': float(baseline_wr),
        'win_rate_trailing': float(trailing_wr),
        'win_rate_improvement': float(wr_improvement),
        'sharpe_improvement_factor': float(sharpe_improvement),
        'stock_results': validation_results,
        'validated': avg_improvement >= 0.30 and wr_improvement >= 5.0,
        'next_step': 'Already deployed in v6.0-EXP074' if avg_improvement >= 0.30 else 'Consider parameter tuning',
        'methodology': {
            'data_period': '2020-2025',
            'baseline_strategy': 'time_decay',
            'test_strategy': 'trailing_stop',
            'success_criteria': {
                'avg_improvement': '+0.30%',
                'win_rate_improvement': '+5pp',
                'stocks_improved': '45/54'
            }
        }
    }

    results_file = os.path.join(results_dir, 'exp075_trailing_stop_validation.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending trailing stop validation report...")
            notifier.send_experiment_report('EXP-075', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-075: Trailing stop validation."""

    print("\n[TRAILING STOP VALIDATION] Testing v6.0 on full 54-stock portfolio")
    print("Validating +0.60% improvement claim from EXP-074")
    print()

    results = run_exp075_trailing_stop_validation()

    print("\n" + "="*70)
    print("TRAILING STOP VALIDATION COMPLETE")
    print("="*70)

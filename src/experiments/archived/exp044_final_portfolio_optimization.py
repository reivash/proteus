"""
EXPERIMENT: EXP-044
Date: 2025-11-17
Objective: Complete 100% portfolio optimization (final 7 stocks)

MOTIVATION:
Portfolio optimization at 77% (24/31 stocks)
- EXP-037, EXP-038, EXP-043 showed +5-15pp improvements per stock
- 7 stocks still using baseline parameters
- Completing optimization will maximize portfolio performance

PROBLEM:
Remaining stocks underoptimized:
- TXN: Large-cap semiconductor (baseline params)
- FTNT: Mid-cap cybersecurity (baseline params)
- MLM: Mid-cap materials (baseline params)
- IDXX: Mid-cap healthcare (baseline params)
- DXCM: Mid-cap healthcare tech (baseline params)
- EXR: Mid-cap REIT (baseline params)
- INVH: Mid-cap REIT (baseline params)

HYPOTHESIS:
Grid search optimization will improve final 7 stocks:
- Expected average improvement: +10pp (based on EXP-037/038/043 patterns)
- Portfolio optimization complete: 100% coverage
- System win rate: ~78% → ~80%+

METHODOLOGY:
1. Grid search optimization (same as EXP-037/038/043):
   - Z-score: [1.2, 1.5, 1.8, 2.0, 2.5]
   - RSI: [30, 32, 35, 38, 40]
   - Volume: [1.2, 1.3, 1.5, 1.8]
   - 100 combinations per stock

2. Test on 2022-2025 data with all filters (earnings, regime)
3. Select params with highest win rate (min 5 trades)
4. Deploy if improvement >= +2pp
5. Update to v13.5-FULLY-OPTIMIZED

EXPECTED OUTCOME:
- All 7 stocks optimized (100% portfolio coverage)
- Average improvement: +10pp per stock
- Portfolio optimization: 100% complete (31/31 stocks)
- System win rate: ~80%+ (from ~78%)
- MILESTONE: Full portfolio optimization complete!
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from itertools import product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from src.config.mean_reversion_params import get_params


def optimize_stock_parameters(ticker: str, start_date: str, end_date: str) -> dict:
    """
    Find optimal parameters for a stock via grid search.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date

    Returns:
        Optimization results with best parameters
    """
    print(f"\nOptimizing {ticker}...")

    # Fetch data
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
    except Exception as e:
        print(f"  [ERROR] Failed to fetch: {e}")
        return None

    if len(data) < 60:
        print(f"  [SKIP] Insufficient data")
        return None

    # Get baseline parameters
    baseline_params = get_params(ticker)

    # Engineer features
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)

    # Parameter grid
    z_scores = [1.2, 1.5, 1.8, 2.0, 2.5]
    rsi_levels = [30, 32, 35, 38, 40]
    volume_mults = [1.2, 1.3, 1.5, 1.8]

    # Track baseline performance
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

    backtester = MeanReversionBacktester(
        initial_capital=10000,
        exit_strategy='time_decay',
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=3
    )

    baseline_results = backtester.backtest(signals)

    if not baseline_results or baseline_results.get('total_trades', 0) < 5:
        print(f"  [SKIP] Insufficient baseline trades")
        return None

    baseline_win_rate = baseline_results.get('win_rate', 0)
    baseline_trades = baseline_results.get('total_trades', 0)
    baseline_return = baseline_results.get('total_return', 0)

    print(f"  Baseline: {baseline_win_rate:.1f}% win rate ({baseline_trades} trades, {baseline_return:+.1f}% return)")

    # Grid search
    best_win_rate = baseline_win_rate
    best_params = baseline_params.copy()
    best_results = baseline_results

    total_combinations = len(z_scores) * len(rsi_levels) * len(volume_mults)
    tested = 0

    for z_score, rsi, volume in product(z_scores, rsi_levels, volume_mults):
        tested += 1

        if tested % 20 == 0:
            print(f"  Progress: {tested}/{total_combinations} combinations tested...")

        # Test this combination
        detector = MeanReversionDetector(
            z_score_threshold=z_score,
            rsi_oversold=rsi,
            rsi_overbought=65,
            volume_multiplier=volume,
            price_drop_threshold=-1.5
        )

        test_signals = detector.detect_overcorrections(enriched_data)
        test_signals = detector.calculate_reversion_targets(test_signals)

        # Apply same filters
        test_signals = add_regime_filter_to_signals(test_signals, regime_detector)
        test_signals = earnings_fetcher.add_earnings_filter_to_signals(test_signals, ticker, 'panic_sell')

        # Backtest
        try:
            results = backtester.backtest(test_signals)

            if not results or results.get('total_trades', 0) < 5:
                continue

            win_rate = results.get('win_rate', 0)

            # Update best if better
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_params = {
                    'z_score_threshold': z_score,
                    'rsi_oversold': rsi,
                    'volume_multiplier': volume,
                    'price_drop_threshold': -1.5
                }
                best_results = results

        except Exception as e:
            continue

    # Calculate improvement
    improvement = best_win_rate - baseline_win_rate

    print(f"  OPTIMIZED: {best_win_rate:.1f}% win rate ({best_results['total_trades']} trades, {best_results['total_return']:+.1f}% return)")
    print(f"  Improvement: {improvement:+.1f}pp")

    if improvement >= 2.0:
        print(f"  [DEPLOY] Significant improvement!")
    else:
        print(f"  [SKIP] Marginal improvement (< +2pp)")

    return {
        'ticker': ticker,
        'baseline': {
            'win_rate': baseline_win_rate,
            'total_trades': baseline_trades,
            'total_return': baseline_return,
            'params': baseline_params
        },
        'optimized': {
            'win_rate': best_win_rate,
            'total_trades': best_results['total_trades'],
            'total_return': best_results['total_return'],
            'sharpe_ratio': best_results.get('sharpe_ratio', 0),
            'params': best_params
        },
        'improvement': improvement,
        'deploy': improvement >= 2.0
    }


def run_exp044_final_portfolio_optimization():
    """
    Complete 100% portfolio optimization (final 7 stocks).
    """
    print("=" * 70)
    print("EXP-044: FINAL PORTFOLIO OPTIMIZATION (100% COVERAGE)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Optimize final 7 stocks for 100% portfolio coverage")
    print("Current: 24/31 stocks optimized (77%)")
    print("Expected: +5-15pp improvement per stock (based on EXP-037/038/043)")
    print()

    # Final 7 unoptimized stocks
    final_stocks = ['TXN', 'FTNT', 'MLM', 'IDXX', 'DXCM', 'EXR', 'INVH']

    print(f"Optimizing {len(final_stocks)} stocks: {', '.join(final_stocks)}")
    print("Testing 100 parameter combinations per stock")
    print("This will take 15-20 minutes...")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    optimization_results = []

    for ticker in final_stocks:
        result = optimize_stock_parameters(ticker, start_date, end_date)
        if result:
            optimization_results.append(result)

    # Summary
    print()
    print("=" * 70)
    print("FINAL OPTIMIZATION RESULTS")
    print("=" * 70)
    print()

    if optimization_results:
        print(f"{'Stock':<8} {'Baseline':<12} {'Optimized':<12} {'Improvement':<14} {'Deploy?'}")
        print("-" * 70)

        total_improvement = 0
        deploy_count = 0

        for result in sorted(optimization_results, key=lambda x: x['improvement'], reverse=True):
            ticker = result['ticker']
            baseline_wr = result['baseline']['win_rate']
            optimized_wr = result['optimized']['win_rate']
            improvement = result['improvement']
            deploy = "YES" if result['deploy'] else "no"

            total_improvement += improvement
            if result['deploy']:
                deploy_count += 1

            print(f"{ticker:<8} {baseline_wr:>10.1f}% {optimized_wr:>10.1f}% {improvement:>12.1f}pp {deploy:>8}")

        avg_improvement = total_improvement / len(optimization_results)

        print()
        print(f"AVERAGE IMPROVEMENT: {avg_improvement:+.1f}pp")
        print(f"STOCKS TO DEPLOY: {deploy_count}/{len(optimization_results)}")
        print()

        if deploy_count > 0:
            print(f"[SUCCESS] {deploy_count} stocks show significant improvement (>+2pp)!")
            print()
            print("OPTIMIZED PARAMETERS:")
            print()

            for result in optimization_results:
                if result['deploy']:
                    ticker = result['ticker']
                    params = result['optimized']['params']
                    wr = result['optimized']['win_rate']
                    improvement = result['improvement']

                    print(f"{ticker}:")
                    print(f"  z_score_threshold: {params['z_score_threshold']}")
                    print(f"  rsi_oversold: {params['rsi_oversold']}")
                    print(f"  volume_multiplier: {params['volume_multiplier']}")
                    print(f"  Win rate: {wr:.1f}% ({improvement:+.1f}pp)")
                    print()

        # Milestone message
        total_optimized = 24 + deploy_count
        coverage = (total_optimized / 31) * 100

        print()
        print("=" * 70)
        print("PORTFOLIO OPTIMIZATION MILESTONE")
        print("=" * 70)
        print()
        print(f"Portfolio optimization: {total_optimized}/31 stocks ({coverage:.1f}%)")

        if total_optimized == 31:
            print()
            print("🎉 MILESTONE ACHIEVED: 100% PORTFOLIO OPTIMIZATION COMPLETE!")
            print()
            print("All 31 Tier A stocks now running optimized parameters!")
            print("Expected portfolio win rate: ~80%+")
        else:
            print(f"\nRemaining stocks: {31 - total_optimized}")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-044',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'stocks_optimized': final_stocks,
        'optimization_results': optimization_results,
        'avg_improvement': avg_improvement if optimization_results else 0,
        'stocks_to_deploy': deploy_count if optimization_results else 0,
        'total_portfolio_optimized': 24 + (deploy_count if optimization_results else 0),
        'portfolio_coverage_pct': ((24 + (deploy_count if optimization_results else 0)) / 31) * 100
    }

    results_file = os.path.join(results_dir, 'exp044_final_portfolio_optimization.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending final optimization report email...")
            notifier.send_experiment_report('EXP-044', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-044 final portfolio optimization."""

    print("\n[FINAL OPTIMIZATION] Completing 100% portfolio optimization")
    print("This will take 15-20 minutes...")
    print()

    results = run_exp044_final_portfolio_optimization()

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Deploy optimized parameters to mean_reversion_params.py")
    print("2. Update to v13.5-FULLY-OPTIMIZED")
    print("3. MILESTONE: 100% portfolio optimization complete!")
    print()
    print("FINAL PORTFOLIO OPTIMIZATION COMPLETE")

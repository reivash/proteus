"""
EXPERIMENT: EXP-063
Date: 2025-11-17
Objective: Optimize parameters for unoptimized Tier A stocks

RESEARCH MOTIVATION:
After 62 experiments and multiple expansions, 22 out of 46 stocks (48%)
still use default or un-optimized parameters.

Parameter optimization has PROVEN ROI:
- EXP-037: 5 stocks, +16.6pp avg improvement
- EXP-038: 12 stocks, +9.1pp avg improvement
- EXP-043: 7 stocks, +13.2pp avg improvement (3 hit 100% win rate!)
- EXP-044: 4 stocks, +5.8pp avg improvement (TXN hit 100% win rate!)

THEORY FOUNDATION:
One-size-fits-all parameters are suboptimal:
- High-volatility stocks (tech) need higher z-score thresholds
- Low-volatility stocks (utilities) need lower z-score thresholds
- Volume characteristics vary by stock
- Per-stock optimization captures these differences

HYPOTHESIS:
Optimizing parameters for the 22 unoptimized stocks will improve win rates
- Expected improvement: +5-15pp per stock (based on historical results)
- Focus on stocks with highest trading frequency first (max impact)
- Deploy incrementally to validate improvements

PRIORITY STOCKS (Round 1 - recently added/high frequency):
1. SYK - Just re-added in EXP-024 (70% win rate with defaults)
2. KLAC - Added in EXP-024, has default params (80% win rate)
3. MA - Original Tier A, never optimized (71.4% win rate)
4. AXP - Original Tier A, never optimized (71.4% win rate)
5. JNJ - Added in EXP-042, never optimized (77.8% win rate)

METHODOLOGY:
1. Grid search for each stock:
   - Z-score: [1.2, 1.5, 1.8, 2.0, 2.5]
   - RSI: [30, 32, 35, 38, 40]
   - Volume: [1.2, 1.3, 1.5, 1.8]
   - Total: 100 combinations per stock
2. Select parameters that maximize win rate (min 5 trades)
3. Validate improvement > +1pp before deployment
4. Document results for mean_reversion_params.py

EXPECTED OUTCOME:
- 5 stocks optimized with +5-15pp improvement each
- Average portfolio win rate increase: +0.5-1.0pp
- Validation for deploying optimizations to remaining 17 stocks
- Quick deployment (Round 1 = 500 backtests, ~30 minutes)
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from itertools import product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from common.config.mean_reversion_params import MEAN_REVERSION_PARAMS, get_params


def optimize_stock_parameters(ticker: str, start_date: str, end_date: str) -> Dict:
    """
    Find optimal parameters for a stock via grid search.

    Args:
        ticker: Stock ticker
        start_date: Start date for backtest
        end_date: End date for backtest

    Returns:
        Optimization results with best parameters
    """
    print(f"\n{'='*70}")
    print(f"OPTIMIZING: {ticker}")
    print(f"{'='*70}")

    # Fetch data
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
    except Exception as e:
        print(f"[ERROR] Failed to fetch {ticker}: {e}")
        return None

    if len(data) < 60:
        print(f"[SKIP] Insufficient data")
        return None

    # Parameter grid
    z_scores = [1.2, 1.5, 1.8, 2.0, 2.5]
    rsi_levels = [30, 32, 35, 38, 40]
    volume_mults = [1.2, 1.3, 1.5, 1.8]

    # Get baseline (current parameters)
    baseline_params = get_params(ticker)

    print(f"\nBaseline parameters:")
    print(f"  Z-score: {baseline_params['z_score_threshold']}")
    print(f"  RSI: {baseline_params['rsi_oversold']}")
    print(f"  Volume: {baseline_params['volume_multiplier']}x")
    print()

    print(f"Grid search: {len(z_scores)}×{len(rsi_levels)}×{len(volume_mults)} = {len(z_scores)*len(rsi_levels)*len(volume_mults)} combinations")
    print()

    best_result = None
    best_win_rate = 0
    best_params = None
    baseline_result = None

    total_combos = len(list(product(z_scores, rsi_levels, volume_mults)))

    for i, (z_score, rsi, volume) in enumerate(product(z_scores, rsi_levels, volume_mults), 1):
        if i % 20 == 0:
            print(f"  Progress: {i}/{total_combos} ({i/total_combos*100:.0f}%)")

        # Engineer features
        engineer = TechnicalFeatureEngineer(fillna=True)
        enriched_data = engineer.engineer_features(data.copy())

        # Detect signals with current parameters
        detector = MeanReversionDetector(
            z_score_threshold=z_score,
            rsi_oversold=rsi,
            rsi_overbought=65,
            volume_multiplier=volume,
            price_drop_threshold=-1.5
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

        try:
            result = backtester.backtest(signals)

            if result is None or result.get('total_trades', 0) < 5:
                continue

            # Check if this is baseline
            if (z_score == baseline_params['z_score_threshold'] and
                rsi == baseline_params['rsi_oversold'] and
                volume == baseline_params['volume_multiplier']):
                baseline_result = result

            # Track best
            win_rate = result['win_rate']
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_result = result
                best_params = {
                    'z_score_threshold': z_score,
                    'rsi_oversold': rsi,
                    'volume_multiplier': volume
                }

        except Exception as e:
            continue

    if best_result is None:
        print(f"[SKIP] No valid parameter combinations found")
        return None

    # Calculate improvement
    if baseline_result:
        baseline_wr = baseline_result['win_rate']
        improvement = best_win_rate - baseline_wr
    else:
        baseline_wr = None
        improvement = None

    print()
    print(f"RESULTS:")
    if baseline_result:
        print(f"  Baseline: {baseline_wr:.1f}% win rate ({baseline_result['total_trades']} trades)")
    print(f"  Optimized: {best_win_rate:.1f}% win rate ({best_result['total_trades']} trades)")
    if improvement is not None:
        print(f"  Improvement: {improvement:+.1f}pp")
    print()
    print(f"Best parameters:")
    print(f"  Z-score: {best_params['z_score_threshold']}")
    print(f"  RSI: {best_params['rsi_oversold']}")
    print(f"  Volume: {best_params['volume_multiplier']}x")

    return {
        'ticker': ticker,
        'baseline_win_rate': baseline_wr,
        'baseline_trades': baseline_result['total_trades'] if baseline_result else None,
        'optimized_win_rate': best_win_rate,
        'optimized_trades': best_result['total_trades'],
        'improvement': improvement,
        'best_params': best_params,
        'result': best_result
    }


def run_exp063_optimize_unoptimized(period: str = "3y"):
    """
    Optimize parameters for unoptimized Tier A stocks (Round 1: 5 stocks).

    Args:
        period: Backtest period
    """
    print("="*70)
    print("EXP-063: PARAMETER OPTIMIZATION FOR UNOPTIMIZED STOCKS (ROUND 1)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("ROUND 1 STOCKS (Priority - recently added/high frequency):")
    print("  1. SYK - Just re-added (70% baseline)")
    print("  2. KLAC - EXP-024 addition (80% baseline)")
    print("  3. MA - Original Tier A (71.4% baseline)")
    print("  4. AXP - Original Tier A (71.4% baseline)")
    print("  5. JNJ - EXP-042 addition (77.8% baseline)")
    print()
    print("Expected time: ~30 minutes (500 backtests)")
    print()

    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    if period == "3y":
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    # Priority stocks
    priority_stocks = ['SYK', 'KLAC', 'MA', 'AXP', 'JNJ']

    results = []
    for ticker in priority_stocks:
        result = optimize_stock_parameters(ticker, start_date, end_date)
        if result:
            results.append(result)

    # Summary
    print()
    print("="*70)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*70)
    print()

    for result in results:
        ticker = result['ticker']
        baseline_wr = result.get('baseline_win_rate')
        optimized_wr = result['optimized_win_rate']
        improvement = result.get('improvement')

        print(f"{ticker}:")
        if baseline_wr:
            print(f"  Baseline: {baseline_wr:.1f}% → Optimized: {optimized_wr:.1f}% ({improvement:+.1f}pp)")
        else:
            print(f"  Optimized: {optimized_wr:.1f}%")
        print(f"  Best params: z={result['best_params']['z_score_threshold']}, rsi={result['best_params']['rsi_oversold']}, vol={result['best_params']['volume_multiplier']}x")
        print()

    # Aggregate stats
    total_improvement = sum(r.get('improvement', 0) for r in results if r.get('improvement'))
    avg_improvement = total_improvement / len([r for r in results if r.get('improvement')])

    print("AGGREGATE STATS:")
    print(f"  Stocks optimized: {len(results)}")
    print(f"  Average improvement: {avg_improvement:+.1f}pp")
    print(f"  Total improvement: {total_improvement:+.1f}pp")
    print()

    # Deployment recommendation
    deployable = [r for r in results if r.get('improvement', 0) >= 1.0]
    print(f"Stocks with >+1pp improvement (deployable): {len(deployable)}")
    for r in deployable:
        print(f"  - {r['ticker']}: {r['improvement']:+.1f}pp improvement")
    print()

    # Save results
    output_file = f'logs/experiments/exp063_parameter_optimization_round1.json'
    os.makedirs('logs/experiments', exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            'experiment_id': 'EXP-063',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'period': period,
            'stocks_tested': len(priority_stocks),
            'results': results,
            'aggregate': {
                'avg_improvement': avg_improvement,
                'total_improvement': total_improvement,
                'deployable_count': len(deployable)
            }
        }, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    return results


if __name__ == "__main__":
    results = run_exp063_optimize_unoptimized()

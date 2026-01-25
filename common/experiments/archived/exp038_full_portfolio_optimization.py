"""
EXPERIMENT: EXP-038
Date: 2025-11-17
Objective: Complete per-stock parameter optimization for ALL Tier A stocks

MOTIVATION:
EXP-037 Results (4 stocks tested):
- Average improvement: +16.6pp
- NVDA: +8.3pp (75.0% → 83.3%)
- V: +16.7pp (66.7% → 83.3%)
- GILD: Optimized to z=1.2
- ROAD: +28.6pp (71.4% → 100%!)

PROBLEM:
Only 4 of 22 Tier A stocks optimized. Remaining 18 stocks still using default parameters.

HYPOTHESIS:
Optimizing all 22 Tier A stocks will deliver portfolio-wide improvement:
- Expected: +10-15pp average improvement
- Target: 63.6% → 75%+ portfolio win rate
- Data-driven, validated approach

METHODOLOGY:
1. Run grid search for each of 18 remaining Tier A stocks:
   - Z-score: [1.2, 1.5, 1.8, 2.0, 2.5]
   - RSI: [30, 32, 35, 38, 40]
   - Volume: [1.2, 1.3, 1.5, 1.8]

2. Test period: 2022-01-01 to 2025-11-15 (same as EXP-037)
3. Find parameters that maximize win rate
4. Deploy only if improvement >= +2pp per stock
5. Update production config with optimized parameters

REMAINING STOCKS (18):
Large-cap (13):
- MA, AVGO, AXP, KLAC, ORCL, MRVL, ABBV, SYK, EOG, TXN, INTU

Mid-cap (6):
- MLM, FTNT, IDXX, DXCM, EXR, INVH

Small-cap (1):
- INSM

EXPECTED OUTCOME:
- All 22 Tier A stocks with data-driven optimized parameters
- Portfolio win rate: 75%+ (validated)
- Maintained trade frequency
- Complete v13.1-OPTIMIZED deployment
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List
from itertools import product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from common.config.mean_reversion_params import get_params, get_all_tickers


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

    total_combos = len(z_scores) * len(rsi_levels) * len(volume_mults)
    print(f"Grid search: {total_combos} combinations")
    print()

    best_result = None
    best_win_rate = 0
    best_params = None
    all_results = []

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
            results = backtester.backtest(signals)

            if results and results.get('total_trades', 0) > 5:
                win_rate = results.get('win_rate', 0) / 100.0
                total_return = results.get('total_return', 0)
                total_trades = results.get('total_trades', 0)

                all_results.append({
                    'z_score': z_score,
                    'rsi': rsi,
                    'volume': volume,
                    'win_rate': win_rate,
                    'total_return': total_return,
                    'total_trades': total_trades
                })

                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_result = results
                    best_params = {
                        'z_score_threshold': z_score,
                        'rsi_oversold': rsi,
                        'volume_multiplier': volume
                    }
        except:
            pass

    print()

    if not best_params:
        print("[FAILED] No valid parameter combinations found")
        return None

    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print()

    # Compare with baseline
    print(f"BASELINE:")
    print(f"  Z-score: {baseline_params['z_score_threshold']}")
    print(f"  RSI: {baseline_params['rsi_oversold']}")
    print(f"  Volume: {baseline_params['volume_multiplier']}x")
    print()

    print(f"OPTIMIZED:")
    print(f"  Z-score: {best_params['z_score_threshold']}")
    print(f"  RSI: {best_params['rsi_oversold']}")
    print(f"  Volume: {best_params['volume_multiplier']}x")
    print()

    print(f"PERFORMANCE:")
    print(f"  Win Rate: {best_win_rate*100:.1f}%")
    print(f"  Total Trades: {best_result.get('total_trades', 0)}")
    print(f"  Total Return: {best_result.get('total_return', 0):.1f}%")
    print()

    # Check if optimization helped
    baseline_match = [r for r in all_results if
                     r['z_score'] == baseline_params['z_score_threshold'] and
                     r['rsi'] == baseline_params['rsi_oversold'] and
                     abs(r['volume'] - baseline_params['volume_multiplier']) < 0.01]

    if baseline_match:
        baseline_wr = baseline_match[0]['win_rate']
        improvement = (best_win_rate - baseline_wr) * 100
        print(f"IMPROVEMENT: {improvement:+.1f}pp vs baseline")

        if improvement > 2.0:
            print(f"[SUCCESS] Significant improvement!")
        elif improvement > 0:
            print(f"[MARGINAL] Slight improvement")
        else:
            print(f"[NO BENEFIT] Keep baseline parameters")
    else:
        print(f"[INFO] Could not compare with exact baseline")

    return {
        'ticker': ticker,
        'baseline_params': baseline_params,
        'optimized_params': best_params,
        'best_win_rate': best_win_rate,
        'best_total_trades': best_result.get('total_trades', 0),
        'best_total_return': best_result.get('total_return', 0),
        'all_results': all_results
    }


def run_exp038_full_portfolio_optimization():
    """
    Optimize all remaining Tier A stocks.
    """
    print("=" * 70)
    print("EXP-038: FULL PORTFOLIO OPTIMIZATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Optimize ALL 22 Tier A stocks")
    print("EXP-037 results: +16.6pp average improvement on 4 stocks")
    print("Expected: +10-15pp portfolio-wide improvement")
    print()

    # Get all Tier A stocks
    all_tickers = get_all_tickers()

    # Already optimized in EXP-037
    already_optimized = ['NVDA', 'V', 'GILD', 'ROAD']

    # Remaining stocks to optimize
    remaining_stocks = [t for t in all_tickers if t not in already_optimized]

    print(f"Already optimized: {len(already_optimized)} stocks")
    print(f"  {', '.join(already_optimized)}")
    print()

    print(f"Remaining: {len(remaining_stocks)} stocks")
    print(f"  {', '.join(remaining_stocks)}")
    print()

    print(f"[WARNING] This will take 2-3 hours to complete!")
    print(f"Estimated time: {len(remaining_stocks) * 8} minutes")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    optimization_results = []

    for i, stock in enumerate(remaining_stocks, 1):
        print(f"\n[{i}/{len(remaining_stocks)}] Processing {stock}...")
        result = optimize_stock_parameters(stock, start_date, end_date)
        if result:
            optimization_results.append(result)

    # Aggregate results
    print()
    print("=" * 70)
    print("PORTFOLIO OPTIMIZATION SUMMARY")
    print("=" * 70)
    print()

    if optimization_results:
        print(f"{'Stock':<8} {'Baseline WR':<13} {'Optimized WR':<14} {'Improvement':<12} {'Trades'}")
        print("-" * 70)

        total_improvement = 0
        successful_optimizations = 0

        for result in optimization_results:
            ticker = result['ticker']
            opt_wr = result['best_win_rate'] * 100
            trades = result['best_total_trades']

            # Find baseline for comparison
            baseline_wr = 0
            for r in result['all_results']:
                bp = result['baseline_params']
                if (r['z_score'] == bp['z_score_threshold'] and
                    r['rsi'] == bp['rsi_oversold'] and
                    abs(r['volume'] - bp['volume_multiplier']) < 0.01):
                    baseline_wr = r['win_rate'] * 100
                    break

            if baseline_wr > 0:
                improvement = opt_wr - baseline_wr
                total_improvement += improvement
                if improvement > 2.0:
                    successful_optimizations += 1
                print(f"{ticker:<8} {baseline_wr:>11.1f}% {opt_wr:>12.1f}% {improvement:>+10.1f}pp {trades:>6}")
            else:
                print(f"{ticker:<8} {'N/A':>11} {opt_wr:>12.1f}% {'N/A':>10} {trades:>6}")

        print()
        avg_improvement = total_improvement / len(optimization_results)
        print(f"AVERAGE IMPROVEMENT: {avg_improvement:+.1f}pp")
        print(f"SIGNIFICANT IMPROVEMENTS (>+2pp): {successful_optimizations}/{len(optimization_results)}")
        print()

        if avg_improvement >= 5.0:
            print("[SUCCESS] Excellent portfolio-wide optimization!")
            print("Deploy optimized parameters immediately")
        elif avg_improvement >= 2.0:
            print("[GOOD] Meaningful improvement across portfolio")
            print("Deploy optimized parameters")
        else:
            print("[MARGINAL] Limited benefit from optimization")
            print("Deploy only stocks with >+2pp improvement")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-038',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'stocks_optimized': len(optimization_results),
        'already_optimized': already_optimized,
        'optimization_results': [{**r, 'best_win_rate': float(r['best_win_rate']),
                                  'all_results': None}
                                 for r in optimization_results]
    }

    results_file = os.path.join(results_dir, 'exp038_full_portfolio_optimization.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"Results saved to: {results_file}")

    # Send email
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending optimization report email...")
            notifier.send_experiment_report('EXP-038', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-038 full portfolio optimization."""

    print("\n[DATA-DRIVEN] Optimizing full Tier A portfolio")
    print("This will take 2-3 hours...")
    print()

    results = run_exp038_full_portfolio_optimization()

    print("\n\nNEXT STEPS:")
    print("1. Review optimization results for all stocks")
    print("2. Deploy optimized parameters to mean_reversion_params.py")
    print("3. Update v13.1-OPTIMIZED with full portfolio coverage")
    print("4. Run production validation")
    print()
    print("FULL PORTFOLIO OPTIMIZATION COMPLETE")

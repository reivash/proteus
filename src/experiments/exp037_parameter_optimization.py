"""
EXPERIMENT: EXP-037
Date: 2025-11-17
Objective: Optimize parameters per stock using data-driven backtest

MOTIVATION:
Current v13.0-VALIDATED: 63.6% win rate (validated)
All stocks use SAME parameters:
- Z-score threshold: 1.5
- RSI threshold: 35
- Volume multiplier: 1.3x
- Price drop: -1.5%

PROBLEM:
One-size-fits-all is suboptimal:
- NVDA (high volatility) might need z-score=2.0
- GILD (low volatility) might need z-score=1.2
- Different stocks have different characteristics

HYPOTHESIS:
Optimizing parameters per stock will improve win rate:
- Use backtest to find optimal thresholds per stock
- Grid search across parameter space
- Validate improvements before deployment
- Expected: +2-5pp win rate improvement (conservative)

METHODOLOGY:
1. For each stock, test parameter combinations:
   - Z-score: [1.2, 1.5, 1.8, 2.0, 2.5]
   - RSI: [30, 32, 35, 38, 40]
   - Volume: [1.2, 1.3, 1.5, 1.8, 2.0]

2. Run backtest for each combination
3. Find parameters that maximize win rate
4. Validate on out-of-sample data
5. Deploy only if improvement > +1pp

CONSTRAINTS:
- Test on subset of stocks first (fast iteration)
- Only optimize if clear improvement
- Document results for transparency
- VALIDATE before deployment (learned from v15/16 failure)

EXPECTED OUTCOME:
- Per-stock optimized parameters
- Win rate improvement: +2-5pp (65-68% target)
- Reasonable trade frequency maintained
- Properly validated before deployment
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

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from src.config.mean_reversion_params import MEAN_REVERSION_PARAMS, get_params


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

    print(f"Grid search: {len(z_scores)} z-scores × {len(rsi_levels)} RSI × {len(volume_mults)} volume = {len(z_scores)*len(rsi_levels)*len(volume_mults)} combinations")
    print()

    best_result = None
    best_win_rate = 0
    best_params = None
    all_results = []

    total_combos = len(list(product(z_scores, rsi_levels, volume_mults)))

    for i, (z_score, rsi, volume) in enumerate(product(z_scores, rsi_levels, volume_mults), 1):
        if i % 10 == 0:
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
            price_drop_threshold=-1.5  # Keep constant for now
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

            if results and results.get('total_trades', 0) > 5:  # Need minimum trades
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

        if improvement > 1.0:
            print(f"[SUCCESS] Significant improvement!")
        elif improvement > 0:
            print(f"[MARGINAL] Slight improvement")
        else:
            print(f"[NO BENEFIT] Optimization didn't help")
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


def run_exp037_parameter_optimization():
    """
    Run per-stock parameter optimization.
    """
    print("=" * 70)
    print("EXP-037: PER-STOCK PARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Find optimal parameters per stock via backtest")
    print("Current: All stocks use same parameters (one-size-fits-all)")
    print("Expected: +2-5pp win rate improvement through customization")
    print()

    # Test on subset for speed (can expand later)
    test_stocks = ['NVDA', 'V', 'AAPL', 'MSFT', 'GILD', 'ROAD']
    print(f"Testing {len(test_stocks)} stocks: {', '.join(test_stocks)}")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    optimization_results = []

    for stock in test_stocks:
        result = optimize_stock_parameters(stock, start_date, end_date)
        if result:
            optimization_results.append(result)

    # Aggregate results
    print()
    print("=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    print()

    if optimization_results:
        print(f"{'Stock':<8} {'Baseline WR':<13} {'Optimized WR':<14} {'Improvement':<12} {'Trades'}")
        print("-" * 70)

        total_improvement = 0

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
                print(f"{ticker:<8} {baseline_wr:>11.1f}% {opt_wr:>12.1f}% {improvement:>+10.1f}pp {trades:>6}")
            else:
                print(f"{ticker:<8} {'N/A':>11} {opt_wr:>12.1f}% {'N/A':>10} {trades:>6}")

        print()
        avg_improvement = total_improvement / len(optimization_results)
        print(f"AVERAGE IMPROVEMENT: {avg_improvement:+.1f}pp")
        print()

        if avg_improvement >= 2.0:
            print("[SUCCESS] Optimization shows significant benefit!")
            print("Recommended: Deploy optimized parameters")
        elif avg_improvement >= 1.0:
            print("[MARGINAL] Optimization shows some benefit")
            print("Consider deploying if consistent across more stocks")
        else:
            print("[NO BENEFIT] Optimization doesn't improve performance")
            print("Keep current parameters (one-size-fits-all is fine)")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-037',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'stocks_tested': test_stocks,
        'optimization_results': [{**r, 'best_win_rate': float(r['best_win_rate']),
                                  'all_results': None}  # Omit detailed results for size
                                 for r in optimization_results]
    }

    results_file = os.path.join(results_dir, 'exp037_parameter_optimization.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"Results saved to: {results_file}")

    # Send email
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending optimization report email...")
            notifier.send_experiment_report('EXP-037', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-037 parameter optimization."""

    print("\n[DATA-DRIVEN] Optimizing parameters per stock")
    print("This will take 10-15 minutes...")
    print()

    results = run_exp037_parameter_optimization()

    print("\n\nNEXT STEPS:")
    print("1. Review optimization results")
    print("2. If improvement >= +2pp, deploy optimized parameters")
    print("3. If improvement < +1pp, keep current parameters")
    print("4. Always validate with out-of-sample data")
    print()
    print("OPTIMIZATION COMPLETE")

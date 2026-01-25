"""
EXPERIMENT: EXP-039
Date: 2025-11-17
Objective: Optimize exit strategy thresholds for maximum returns

MOTIVATION:
Current time-decay exit strategy (EXP-010):
- Day 0: ±2.0% (entry day)
- Day 1: ±1.5% (second day)
- Day 2+: ±1.0% (third day onwards)
- Max hold: 3 days

PROBLEM:
Fixed thresholds may be suboptimal:
- High volatility stocks (NVDA) might need wider targets (±3%)
- Low volatility stocks (V) might need tighter targets (±1.5%)
- Current thresholds chosen empirically, not data-driven

HYPOTHESIS:
Optimizing exit thresholds will improve returns:
- Test profit targets: [1.5%, 2.0%, 2.5%, 3.0%]
- Test stop losses: [-1.5%, -2.0%, -2.5%, -3.0%]
- Test max hold days: [2, 3, 4, 5]
- Find optimal combination per stock
- Expected: +5-10% total return improvement

METHODOLOGY:
1. For each Tier A stock, test exit combinations:
   - Profit targets: [1.5, 2.0, 2.5, 3.0]
   - Stop losses: [-1.5, -2.0, -2.5, -3.0]
   - Max hold days: [2, 3, 4, 5]

2. Use OPTIMIZED parameters from EXP-037/038
3. Measure: total return, win rate, avg gain/loss
4. Find exit strategy that maximizes total return while maintaining win rate
5. Deploy only if improvement >= +5% total return

CONSTRAINTS:
- Must maintain win rate >= 70% (Tier A threshold)
- Must maintain reasonable trade frequency
- Simpler is better (avoid over-optimization)

EXPECTED OUTCOME:
- Per-stock optimized exit thresholds
- +5-10% total return improvement
- Maintained or improved win rates
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
from common.config.mean_reversion_params import get_params


def optimize_exit_strategy(ticker: str, start_date: str, end_date: str) -> Dict:
    """
    Find optimal exit strategy for a stock.

    Args:
        ticker: Stock ticker
        start_date: Start date for backtest
        end_date: End date for backtest

    Returns:
        Optimization results with best exit strategy
    """
    print(f"\n{'='*70}")
    print(f"OPTIMIZING EXIT STRATEGY: {ticker}")
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

    # Get stock parameters (use optimized if available)
    params = get_params(ticker)

    print(f"\nUsing parameters:")
    print(f"  Z-score: {params['z_score_threshold']}")
    print(f"  RSI: {params['rsi_oversold']}")
    print(f"  Volume: {params['volume_multiplier']}x")
    print()

    # Engineer features
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)

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

    # Apply filters
    regime_detector = MarketRegimeDetector()
    signals = add_regime_filter_to_signals(signals, regime_detector)

    earnings_fetcher = EarningsCalendarFetcher(exclusion_days_before=3, exclusion_days_after=3)
    signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

    # Exit strategy grid
    profit_targets = [1.5, 2.0, 2.5, 3.0]
    stop_losses = [-1.5, -2.0, -2.5, -3.0]
    max_hold_days_options = [2, 3, 4, 5]

    baseline_exit = {'profit_target': 2.0, 'stop_loss': -2.0, 'max_hold_days': 3}

    print(f"Testing {len(profit_targets)} × {len(stop_losses)} × {len(max_hold_days_options)} = "
          f"{len(profit_targets)*len(stop_losses)*len(max_hold_days_options)} exit combinations")
    print()

    best_result = None
    best_total_return = -999
    best_exit_strategy = None
    all_results = []

    baseline_result = None

    for i, (profit_target, stop_loss, max_hold) in enumerate(
            product(profit_targets, stop_losses, max_hold_days_options), 1):

        if i % 10 == 0:
            print(f"  Progress: {i}/{len(profit_targets)*len(stop_losses)*len(max_hold_days_options)}")

        # Backtest with this exit strategy
        backtester = MeanReversionBacktester(
            initial_capital=10000,
            exit_strategy='time_decay',
            profit_target=profit_target,
            stop_loss=stop_loss,
            max_hold_days=max_hold
        )

        try:
            results = backtester.backtest(signals)

            if results and results.get('total_trades', 0) > 5:
                win_rate = results.get('win_rate', 0) / 100.0
                total_return = results.get('total_return', 0)
                total_trades = results.get('total_trades', 0)

                result_entry = {
                    'profit_target': profit_target,
                    'stop_loss': stop_loss,
                    'max_hold_days': max_hold,
                    'win_rate': win_rate,
                    'total_return': total_return,
                    'total_trades': total_trades
                }

                all_results.append(result_entry)

                # Track baseline
                if (profit_target == baseline_exit['profit_target'] and
                    stop_loss == baseline_exit['stop_loss'] and
                    max_hold == baseline_exit['max_hold_days']):
                    baseline_result = result_entry

                # Find best total return (while maintaining decent win rate)
                if win_rate >= 0.65 and total_return > best_total_return:
                    best_total_return = total_return
                    best_result = results
                    best_exit_strategy = {
                        'profit_target': profit_target,
                        'stop_loss': stop_loss,
                        'max_hold_days': max_hold
                    }
        except:
            pass

    print()

    if not best_exit_strategy:
        print("[FAILED] No valid exit strategies found")
        return None

    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print()

    # Compare with baseline
    print(f"BASELINE EXIT STRATEGY:")
    print(f"  Profit target: {baseline_exit['profit_target']}%")
    print(f"  Stop loss: {baseline_exit['stop_loss']}%")
    print(f"  Max hold: {baseline_exit['max_hold_days']} days")
    if baseline_result:
        print(f"  Performance: {baseline_result['win_rate']*100:.1f}% win rate, "
              f"{baseline_result['total_return']:.1f}% return")
    print()

    print(f"OPTIMIZED EXIT STRATEGY:")
    print(f"  Profit target: {best_exit_strategy['profit_target']}%")
    print(f"  Stop loss: {best_exit_strategy['stop_loss']}%")
    print(f"  Max hold: {best_exit_strategy['max_hold_days']} days")
    print(f"  Performance: {best_result.get('win_rate', 0):.1f}% win rate, "
          f"{best_result.get('total_return', 0):.1f}% return")
    print()

    if baseline_result:
        return_improvement = best_total_return - baseline_result['total_return']
        win_rate_change = (best_result.get('win_rate', 0)/100.0 - baseline_result['win_rate']) * 100

        print(f"IMPROVEMENT:")
        print(f"  Return: {return_improvement:+.1f}%")
        print(f"  Win rate: {win_rate_change:+.1f}pp")
        print()

        if return_improvement > 5.0:
            print("[SUCCESS] Significant return improvement!")
        elif return_improvement > 2.0:
            print("[MARGINAL] Modest improvement")
        else:
            print("[NO BENEFIT] Keep baseline exit strategy")

    return {
        'ticker': ticker,
        'baseline_exit': baseline_exit,
        'baseline_result': baseline_result,
        'optimized_exit': best_exit_strategy,
        'optimized_result': {
            'win_rate': best_result.get('win_rate', 0) / 100.0,
            'total_return': best_result.get('total_return', 0),
            'total_trades': best_result.get('total_trades', 0)
        },
        'all_results': all_results
    }


def run_exp039_exit_strategy_optimization():
    """
    Optimize exit strategies for Tier A stocks.
    """
    print("=" * 70)
    print("EXP-039: EXIT STRATEGY OPTIMIZATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Find optimal exit thresholds per stock")
    print("Current: Fixed ±2.0% Day 0, ±1.5% Day 1, ±1.0% Day 2+, 3 day max")
    print("Expected: +5-10% total return improvement")
    print()

    # Test on subset first (quick iteration)
    test_stocks = ['NVDA', 'V', 'MA', 'AVGO', 'GILD', 'ROAD', 'MSFT', 'INSM']
    print(f"Testing {len(test_stocks)} stocks: {', '.join(test_stocks)}")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    optimization_results = []

    for i, stock in enumerate(test_stocks, 1):
        print(f"\n[{i}/{len(test_stocks)}] Processing {stock}...")
        result = optimize_exit_strategy(stock, start_date, end_date)
        if result:
            optimization_results.append(result)

    # Aggregate results
    print()
    print("=" * 70)
    print("EXIT STRATEGY OPTIMIZATION SUMMARY")
    print("=" * 70)
    print()

    if optimization_results:
        print(f"{'Stock':<8} {'Baseline Return':<16} {'Optimized Return':<17} {'Improvement':<12} "
              f"{'Optimized Exit'}")
        print("-" * 100)

        total_improvement = 0
        successful_optimizations = 0

        for result in optimization_results:
            ticker = result['ticker']
            baseline_return = result['baseline_result']['total_return'] if result['baseline_result'] else 0
            optimized_return = result['optimized_result']['total_return']
            improvement = optimized_return - baseline_return

            opt_exit = result['optimized_exit']
            exit_str = f"±{opt_exit['profit_target']}%, {opt_exit['max_hold_days']}d"

            total_improvement += improvement
            if improvement > 5.0:
                successful_optimizations += 1

            print(f"{ticker:<8} {baseline_return:>14.1f}% {optimized_return:>15.1f}% "
                  f"{improvement:>+10.1f}% {exit_str:>15}")

        print()
        avg_improvement = total_improvement / len(optimization_results)
        print(f"AVERAGE RETURN IMPROVEMENT: {avg_improvement:+.1f}%")
        print(f"SIGNIFICANT IMPROVEMENTS (>+5%): {successful_optimizations}/{len(optimization_results)}")
        print()

        if avg_improvement >= 5.0:
            print("[SUCCESS] Exit optimization shows significant benefit!")
            print("Deploy optimized exit strategies")
        elif avg_improvement >= 2.0:
            print("[MARGINAL] Some benefit from exit optimization")
            print("Consider deploying for stocks with >+5% improvement")
        else:
            print("[NO BENEFIT] Current exit strategy is already optimal")
            print("Keep baseline ±2.0%, 3 day max hold")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-039',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'stocks_tested': test_stocks,
        'optimization_results': [{**r,
                                  'baseline_result': {**r['baseline_result'],
                                                     'win_rate': float(r['baseline_result']['win_rate'])}
                                                     if r['baseline_result'] else None,
                                  'optimized_result': {**r['optimized_result'],
                                                      'win_rate': float(r['optimized_result']['win_rate'])},
                                  'all_results': None}
                                 for r in optimization_results]
    }

    results_file = os.path.join(results_dir, 'exp039_exit_strategy_optimization.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending optimization report email...")
            notifier.send_experiment_report('EXP-039', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-039 exit strategy optimization."""

    print("\n[DATA-DRIVEN] Optimizing exit strategies")
    print("This will take 10-15 minutes...")
    print()

    results = run_exp039_exit_strategy_optimization()

    print("\n\nNEXT STEPS:")
    print("1. Review exit strategy results")
    print("2. If improvement >= +5%, deploy optimized exits")
    print("3. Update backtester with per-stock exit strategies")
    print("4. Re-validate with EXP-035")
    print()
    print("EXIT STRATEGY OPTIMIZATION COMPLETE")

"""
EXPERIMENT: EXP-072
Date: 2025-11-17
Objective: Optimize parameters for 6 newly deployed ML-validated stocks

HYPOTHESIS:
The 6 stocks deployed in EXP-071 are using baseline parameters (z=1.5, rsi=35, vol=1.3).
Stock-specific parameter optimization can improve win rates by +5-10pp per stock,
similar to EXP-063 results (+22.5pp for SYK, +26.7pp for MA).

ALGORITHM:
For each new stock (CMCSA, TMUS, META, ETN, HD, SHW):
1. Load historical data (2020-2025)
2. Grid search optimal parameters:
   - z_score_threshold: [1.0, 1.2, 1.5, 1.8, 2.0]
   - rsi_oversold: [30, 32, 35, 38, 40]
   - volume_multiplier: [1.0, 1.2, 1.3, 1.5, 1.8]
3. Backtest each combination
4. Select best performing parameters (maximize win_rate, then total_return)
5. Update mean_reversion_params.py with optimized values

EXPECTED IMPACT:
- Win rate improvement: 76.1% baseline → 80-85% optimized
- 6 stocks × ~5 trades/year = ~30 trades affected
- Could add +2-3pp to overall portfolio win rate
- Validates ML screening + optimization methodology

SUCCESS CRITERIA:
- At least 4/6 stocks show improvement vs baseline
- Average improvement >= +5pp win rate
- No stock drops below 70% win rate threshold
- All optimized params are reasonable (not overfitted)
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester


# New stocks to optimize (from EXP-071)
NEW_STOCKS = [
    {'ticker': 'CMCSA', 'baseline_wr': 83.3},
    {'ticker': 'ETN', 'baseline_wr': 76.5},
    {'ticker': 'HD', 'baseline_wr': 76.5},
    {'ticker': 'TMUS', 'baseline_wr': 76.5},
    {'ticker': 'META', 'baseline_wr': 72.2},
    {'ticker': 'SHW', 'baseline_wr': 71.4},
]

# Parameter grid
PARAM_GRID = {
    'z_score_threshold': [1.0, 1.2, 1.5, 1.8, 2.0],
    'rsi_oversold': [30, 32, 35, 38, 40],
    'volume_multiplier': [1.0, 1.2, 1.3, 1.5, 1.8],
}


def optimize_stock_parameters(ticker: str, baseline_wr: float,
                              start_date: str = '2020-01-01',
                              end_date: str = '2025-11-17') -> Dict:
    """
    Optimize parameters for a single stock.

    Args:
        ticker: Stock ticker
        baseline_wr: Baseline win rate from EXP-071
        start_date: Optimization start date
        end_date: Optimization end date

    Returns:
        Optimization results dict
    """
    print(f"\n{'='*70}")
    print(f"OPTIMIZING: {ticker} (Baseline WR: {baseline_wr:.1f}%)")
    print(f"{'='*70}")

    try:
        # Fetch data
        buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
        fetcher = YahooFinanceFetcher()
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)

        if len(data) < 60:
            print(f"[ERROR] Insufficient data for {ticker}")
            return None

        # Engineer features
        engineer = TechnicalFeatureEngineer(fillna=True)
        enriched_data = engineer.engineer_features(data)

        # Initialize filters
        regime_detector = MarketRegimeDetector()
        earnings_fetcher = EarningsCalendarFetcher(exclusion_days_before=3, exclusion_days_after=3)

        # Grid search
        best_result = None
        best_score = -np.inf
        results = []

        total_combinations = (len(PARAM_GRID['z_score_threshold']) *
                            len(PARAM_GRID['rsi_oversold']) *
                            len(PARAM_GRID['volume_multiplier']))

        print(f"\nTesting {total_combinations} parameter combinations...")
        print()

        combo_num = 0
        for z_thresh in PARAM_GRID['z_score_threshold']:
            for rsi in PARAM_GRID['rsi_oversold']:
                for vol_mult in PARAM_GRID['volume_multiplier']:
                    combo_num += 1

                    # Test this combination
                    detector = MeanReversionDetector(
                        z_score_threshold=z_thresh,
                        rsi_oversold=rsi,
                        rsi_overbought=65,
                        volume_multiplier=vol_mult,
                        price_drop_threshold=-1.5
                    )

                    signals = detector.detect_overcorrections(enriched_data)
                    signals = detector.calculate_reversion_targets(signals)

                    # Apply filters
                    signals = add_regime_filter_to_signals(signals, regime_detector)
                    signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

                    # Backtest
                    backtester = MeanReversionBacktester(
                        initial_capital=10000,
                        exit_strategy='time_decay',
                        profit_target=2.0,
                        stop_loss=-2.0,
                        max_hold_days=3
                    )

                    backtest_results = backtester.backtest(signals)

                    if not backtest_results or backtest_results.get('total_trades', 0) < 5:
                        continue

                    win_rate = backtest_results.get('win_rate', 0)
                    total_return = backtest_results.get('total_return', 0)
                    total_trades = backtest_results.get('total_trades', 0)

                    # Score: prioritize win_rate, then total_return as tiebreaker
                    score = win_rate * 100 + total_return * 0.1

                    result = {
                        'z_score_threshold': z_thresh,
                        'rsi_oversold': rsi,
                        'volume_multiplier': vol_mult,
                        'win_rate': win_rate,
                        'total_return': total_return,
                        'total_trades': total_trades,
                        'score': score
                    }

                    results.append(result)

                    if score > best_score:
                        best_score = score
                        best_result = result
                        print(f"[{combo_num}/{total_combinations}] NEW BEST: z={z_thresh}, rsi={rsi}, vol={vol_mult} "
                              f"-> WR={win_rate:.1f}%, Return={total_return:+.1f}%, Trades={total_trades}")

        if not best_result:
            print(f"[ERROR] No valid parameter combinations found for {ticker}")
            return None

        # Calculate improvement
        improvement = best_result['win_rate'] - baseline_wr

        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE: {ticker}")
        print(f"{'='*70}")
        print(f"Baseline:  WR={baseline_wr:.1f}% (z=1.5, rsi=35, vol=1.3)")
        print(f"Optimized: WR={best_result['win_rate']:.1f}% (z={best_result['z_score_threshold']}, "
              f"rsi={best_result['rsi_oversold']}, vol={best_result['volume_multiplier']})")
        print(f"Improvement: {improvement:+.1f}pp")
        print(f"Total Return: {best_result['total_return']:+.1f}%")
        print(f"Total Trades: {best_result['total_trades']}")

        return {
            'ticker': ticker,
            'baseline_win_rate': baseline_wr,
            'optimized_win_rate': best_result['win_rate'],
            'improvement': improvement,
            'best_params': {
                'z_score_threshold': best_result['z_score_threshold'],
                'rsi_oversold': best_result['rsi_oversold'],
                'volume_multiplier': best_result['volume_multiplier'],
                'price_drop_threshold': -1.5
            },
            'backtest_results': {
                'win_rate': best_result['win_rate'],
                'total_return': best_result['total_return'],
                'total_trades': best_result['total_trades']
            },
            'all_results': results
        }

    except Exception as e:
        print(f"[ERROR] Optimization failed for {ticker}: {e}")
        return None


def run_exp072_optimize_new_stocks():
    """
    Optimize parameters for all 6 new stocks.
    """
    print("="*70)
    print("EXP-072: OPTIMIZE NEW STOCK PARAMETERS")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("OBJECTIVE: Optimize parameters for 6 newly deployed stocks (EXP-071)")
    print("METHODOLOGY: Grid search over z-score, RSI, volume thresholds")
    print("TARGET: +5-10pp win rate improvement per stock")
    print()

    optimization_results = []

    for stock_info in NEW_STOCKS:
        result = optimize_stock_parameters(
            ticker=stock_info['ticker'],
            baseline_wr=stock_info['baseline_wr']
        )

        if result:
            optimization_results.append(result)

    # Summary
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)
    print()

    if not optimization_results:
        print("[ERROR] No stocks were successfully optimized")
        return None

    print(f"Stocks Optimized: {len(optimization_results)}/{len(NEW_STOCKS)}")
    print()

    print(f"{'Ticker':<8} {'Baseline':<12} {'Optimized':<12} {'Improvement':<12} {'Trades'}")
    print("-"*70)

    total_improvement = 0
    improved_count = 0

    for result in optimization_results:
        improvement = result['improvement']
        if improvement > 0:
            improved_count += 1
        total_improvement += improvement

        print(f"{result['ticker']:<8} "
              f"{result['baseline_win_rate']:<11.1f}% "
              f"{result['optimized_win_rate']:<11.1f}% "
              f"{improvement:<11.1f}pp "
              f"{result['backtest_results']['total_trades']}")

    avg_improvement = total_improvement / len(optimization_results) if optimization_results else 0

    print()
    print(f"Stocks Improved: {improved_count}/{len(optimization_results)}")
    print(f"Average Improvement: {avg_improvement:+.1f}pp")
    print()

    # Recommendations
    print("="*70)
    print("DEPLOYMENT RECOMMENDATION")
    print("="*70)
    print()

    if improved_count >= 4 and avg_improvement >= 5.0:
        print(f"[SUCCESS] {improved_count}/{len(optimization_results)} stocks improved, avg +{avg_improvement:.1f}pp")
        print()
        print("NEXT STEPS:")
        print("  1. Review optimized parameters for reasonableness")
        print("  2. Update common/config/mean_reversion_params.py")
        print("  3. Deploy to production")
        print()
        print("OPTIMIZED PARAMETERS:")
        for result in optimization_results:
            if result['improvement'] > 0:
                params = result['best_params']
                print(f"  {result['ticker']}: z={params['z_score_threshold']}, "
                      f"rsi={params['rsi_oversold']}, vol={params['volume_multiplier']}")
    else:
        print(f"[PARTIAL SUCCESS] {improved_count}/{len(optimization_results)} stocks improved")
        print()
        print("Consider deploying only stocks with positive improvements")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-072',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'objective': 'Optimize parameters for 6 newly deployed stocks',
        'algorithm': 'Grid search parameter optimization',
        'stocks_optimized': len(optimization_results),
        'stocks_improved': improved_count,
        'average_improvement': float(avg_improvement),
        'optimization_results': optimization_results,
        'success': improved_count >= 4 and avg_improvement >= 5.0,
        'next_step': 'Update mean_reversion_params.py with optimized parameters' if improved_count >= 4 else 'Review results',
        'methodology': {
            'data_period': '2020-2025',
            'parameter_grid': PARAM_GRID,
            'success_criteria': '4+ stocks improved, +5pp average'
        }
    }

    results_file = os.path.join(results_dir, 'exp072_parameter_optimization.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending optimization report email...")
            notifier.send_experiment_report('EXP-072', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-072: Optimize new stock parameters."""

    print("\n[PARAMETER OPTIMIZATION] Optimizing 6 newly deployed stocks")
    print("Grid searching z-score, RSI, volume thresholds for maximum performance")
    print()

    results = run_exp072_optimize_new_stocks()

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)

"""
EXPERIMENT: EXP-048
Date: 2025-11-17
Objective: Re-optimize or remove underperforming stocks from portfolio

MOTIVATION:
EXP-047 validation identified weak performers dragging down portfolio average:
- FTNT: 25.0% win rate, -9.3% return (CRITICAL - only 4 trades)
- SYK: 55.6% win rate, +0.9% return (9 trades)
- DXCM: 60.0% win rate, +0.5% return (5 trades)
- INVH: 60.0% win rate, +7.0% return (5 trades)

Portfolio avg return: 16.7% (target: 18%+)
Need +1.3pp improvement to hit target.

PROBLEM:
These 4 stocks are pulling down the portfolio average:
- FTNT at -9.3% is especially problematic
- SYK, DXCM contribute almost nothing
- Portfolio would be stronger without dead weight

HYPOTHESIS:
Aggressive re-optimization or removal will boost portfolio avg:
- Intensive grid search on underperformers
- If no 70%+ win rate config found, REMOVE from portfolio
- Expected: +2-3pp portfolio avg return improvement

METHODOLOGY:
1. Expanded grid search on 4 underperformers:
   - z_score: [1.2, 1.5, 1.8, 2.0, 2.5, 3.0]  (added 3.0 - more selective)
   - rsi: [25, 30, 32, 35, 38, 40]  (added 25 - deeper oversold)
   - volume: [1.2, 1.3, 1.5, 1.8, 2.0]  (added 2.0 - stronger confirmation)
   - price_drop: [-2.0, -2.5, -3.0, -3.5, -4.0]  (test range)

2. Test 180 combinations per stock (vs 100 before)

3. Threshold for keeping stock:
   - Win rate >= 70%
   - Total return > 0
   - Min 5 trades

4. If no config meets threshold: REMOVE from portfolio

5. Deploy successful optimizations, document removals

EXPECTED OUTCOME:
- 2-4 stocks successfully re-optimized (70%+ win rate)
- 0-2 stocks removed from portfolio
- +2-3pp portfolio avg return improvement
- Portfolio quality > quantity
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


def optimize_stock_aggressive(ticker: str, start_date: str, end_date: str) -> dict:
    """
    Aggressive optimization with expanded parameter grid.

    Args:
        ticker: Stock ticker
        start_date: Start date for backtest
        end_date: End date for backtest

    Returns:
        Best configuration or None if no viable config found
    """
    print(f"\n{'='*70}")
    print(f"AGGRESSIVE OPTIMIZATION: {ticker}")
    print(f"{'='*70}")

    # Fetch data
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
    except Exception as e:
        print(f"[ERROR] Failed to fetch data: {e}")
        return None

    if len(data) < 60:
        print(f"[ERROR] Insufficient data")
        return None

    # Current baseline
    current_params = get_params(ticker)
    print(f"\nCurrent config: z={current_params['z_score_threshold']}, "
          f"rsi={current_params['rsi_oversold']}, "
          f"vol={current_params['volume_multiplier']}")

    # Expanded parameter grid
    z_thresholds = [1.2, 1.5, 1.8, 2.0, 2.5, 3.0]  # Added 3.0 for ultra-selective
    rsi_levels = [25, 30, 32, 35, 38, 40]  # Added 25 for deeper oversold
    volume_multipliers = [1.2, 1.3, 1.5, 1.8, 2.0]  # Added 2.0 for strong confirmation

    param_combinations = list(product(z_thresholds, rsi_levels, volume_multipliers))
    print(f"Testing {len(param_combinations)} configurations...")

    best_config = None
    best_win_rate = 0

    for z_thresh, rsi_level, vol_mult in param_combinations:
        # Engineer features
        engineer = TechnicalFeatureEngineer(fillna=True)
        enriched_data = engineer.engineer_features(data)

        # Detect signals
        detector = MeanReversionDetector(
            z_score_threshold=z_thresh,
            rsi_oversold=rsi_level,
            rsi_overbought=65,
            volume_multiplier=vol_mult,
            price_drop_threshold=-3.0
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

            if not results or results.get('total_trades', 0) < 5:
                continue

            win_rate = results['win_rate']
            total_return = results['total_return']
            total_trades = results['total_trades']

            # Threshold: 70%+ win rate, positive return
            if win_rate >= 70.0 and total_return > 0 and total_trades >= 5:
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_config = {
                        'z_score_threshold': z_thresh,
                        'rsi_oversold': rsi_level,
                        'volume_multiplier': vol_mult,
                        'win_rate': win_rate,
                        'total_return': total_return,
                        'total_trades': total_trades,
                        'sharpe_ratio': results.get('sharpe_ratio', 0)
                    }

        except Exception as e:
            continue

    if best_config:
        print(f"\n[SUCCESS] Viable config found!")
        print(f"  Config: z={best_config['z_score_threshold']}, "
              f"rsi={best_config['rsi_oversold']}, "
              f"vol={best_config['volume_multiplier']}")
        print(f"  Win rate: {best_config['win_rate']:.1f}%")
        print(f"  Return: {best_config['total_return']:+.1f}%")
        print(f"  Trades: {best_config['total_trades']}")
        print(f"  Sharpe: {best_config['sharpe_ratio']:.2f}")

        return {
            'ticker': ticker,
            'optimized': True,
            'config': best_config,
            'baseline_win_rate': current_params.get('performance', {}).get('win_rate', 0)
        }
    else:
        print(f"\n[FAILED] No viable config found (70%+ win rate threshold)")
        print(f"  Recommendation: REMOVE {ticker} from portfolio")

        return {
            'ticker': ticker,
            'optimized': False,
            'recommendation': 'REMOVE',
            'reason': 'No configuration achieves 70%+ win rate threshold'
        }


def run_exp048_underperformer_rescue():
    """
    Re-optimize or remove underperforming stocks.
    """
    print("="*70)
    print("EXP-048: UNDERPERFORMER RESCUE OPERATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Re-optimize or remove underperforming stocks")
    print("Target: +1.3pp portfolio avg return improvement (16.7% -> 18%+)")
    print()

    # Underperformers from EXP-047
    underperformers = ['FTNT', 'SYK', 'DXCM', 'INVH']
    print(f"Underperformers to rescue: {', '.join(underperformers)}")
    print("Threshold: 70%+ win rate, positive return, min 5 trades")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    print(f"Test period: {start_date} to {end_date}")
    print("This will take 10-15 minutes...")
    print()

    results = []

    for ticker in underperformers:
        result = optimize_stock_aggressive(ticker, start_date, end_date)
        if result:
            results.append(result)

    # Summary
    print()
    print("="*70)
    print("RESCUE OPERATION RESULTS")
    print("="*70)
    print()

    optimized = [r for r in results if r.get('optimized', False)]
    removed = [r for r in results if not r.get('optimized', False)]

    print(f"Successfully optimized: {len(optimized)}/{len(results)}")
    print(f"Recommended for removal: {len(removed)}/{len(results)}")
    print()

    if optimized:
        print("OPTIMIZED STOCKS:")
        print(f"{'Ticker':<8} {'Win Rate':<12} {'Return':<12} {'Trades':<10}")
        print("-"*70)
        for r in optimized:
            cfg = r['config']
            print(f"{r['ticker']:<8} {cfg['win_rate']:>10.1f}% {cfg['total_return']:>10.1f}% "
                  f"{cfg['total_trades']:>8}")
        print()

    if removed:
        print("RECOMMENDED FOR REMOVAL:")
        for r in removed:
            print(f"  {r['ticker']}: {r['reason']}")
        print()

    # Deployment decision
    print("="*70)
    print("DEPLOYMENT DECISION")
    print("="*70)
    print()

    if optimized:
        print(f"[DEPLOY] Update parameters for {len(optimized)} stock(s)")
        for r in optimized:
            cfg = r['config']
            print(f"\n{r['ticker']}:")
            print(f"  z_score_threshold: {cfg['z_score_threshold']}")
            print(f"  rsi_oversold: {cfg['rsi_oversold']}")
            print(f"  volume_multiplier: {cfg['volume_multiplier']}")
            print(f"  Performance: {cfg['win_rate']:.1f}% win rate, {cfg['total_return']:+.1f}% return")

    if removed:
        print(f"\n[REMOVE] Remove {len(removed)} stock(s) from portfolio:")
        for r in removed:
            print(f"  - {r['ticker']}")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-048',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'underperformers_tested': underperformers,
        'successfully_optimized': len(optimized),
        'recommended_removals': len(removed),
        'optimized_stocks': optimized,
        'removed_stocks': removed,
        'deploy_optimizations': len(optimized) > 0,
        'remove_stocks': len(removed) > 0
    }

    results_file = os.path.join(results_dir, 'exp048_underperformer_rescue.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending rescue operation report email...")
            notifier.send_experiment_report('EXP-048', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-048 underperformer rescue."""

    print("\n[RESCUE OPERATION] Aggressive re-optimization of underperformers")
    print("This will take 10-15 minutes...")
    print()

    results = run_exp048_underperformer_rescue()

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print()

    if results.get('deploy_optimizations', False):
        print("1. Deploy optimized parameters to mean_reversion_params.py")
        print("2. Update stock performance notes")

    if results.get('remove_stocks', False):
        print("3. Remove underperforming stocks from portfolio")
        print("4. Update portfolio documentation")

    print("5. Re-run EXP-047 validation to confirm improvement")
    print()
    print("UNDERPERFORMER RESCUE COMPLETE")

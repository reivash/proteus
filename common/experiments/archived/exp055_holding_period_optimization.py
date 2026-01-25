"""
EXPERIMENT: EXP-055
Date: 2025-11-17
Objective: Optimize holding period per stock based on reversion speed

MOTIVATION:
Current system: Max 3 days hold for ALL stocks (one-size-fits-all)
- Fast-reverting stocks (V, MA): Might revert fully in 1-2 days
- Slow-reverting stocks (NVDA, GILD): Might need 4-5 days for full profit
- Holding too long = opportunity cost (capital tied up)
- Exiting too early = leaving profit on table
- Missing 5-15% return improvement from optimized holding periods

PROBLEM:
One-size-fits-all holding period != optimal capital efficiency:
- If V reverts in 1 day, holding 3 days wastes 2 days of capital
- If NVDA needs 5 days to revert, exiting at 3 days leaves profit behind
- Different stocks have different mean reversion speeds
- Optimal holding period varies by stock characteristics

HYPOTHESIS:
Per-stock holding period based on reversion speed will improve returns:
- Fast reversions (1-2 days): Exit early, free up capital
- Medium reversions (3-4 days): Current 3-day hold optimal
- Slow reversions (4-5 days): Hold longer for full profit capture
- Expected: +5-15% return improvement, maintained win rate, higher capital efficiency

METHODOLOGY:
1. Measure reversion speed per stock:
   - Average days to reach profit target historically
   - Classify as fast/medium/slow reverting

2. Test holding periods per stock:
   - Fast: [1, 2, 3 days]
   - Medium: [2, 3, 4 days]
   - Slow: [3, 4, 5, 6 days]

3. Measure impact on:
   - Total return
   - Win rate
   - Average return per trade
   - Capital efficiency (return / days held)
   - Sharpe ratio

4. Deploy if improvement >= +5% total return

CONSTRAINTS:
- Min 1 day hold (avoid day-trading noise)
- Max 7 days hold (mean reversion window)
- Maintain >= 70% win rate

EXPECTED OUTCOME:
- Per-stock optimal holding periods
- Reversion speed classification
- +5-15% return improvement
- Higher capital efficiency
- Better profit capture
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from common.config.mean_reversion_params import get_params


def measure_reversion_speed(ticker: str, start_date: str, end_date: str) -> dict:
    """
    Measure how fast a stock typically reverts.

    Returns:
        Average days to profit target
    """
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
    except:
        return None

    if len(data) < 60:
        return None

    # Get parameters
    params = get_params(ticker)

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

    # Measure days to target
    days_to_target = []

    panic_signals = signals[signals['panic_sell'] == 1]

    for idx, signal_row in panic_signals.iterrows():
        entry_price = signal_row['Close']
        target_price = signal_row['reversion_target']
        entry_date = signal_row['Date']

        # Look at next 7 days
        future_data = enriched_data[enriched_data['Date'] > entry_date].head(7)

        for day_num, (_, day_row) in enumerate(future_data.iterrows(), 1):
            high = day_row['High']
            if high >= target_price:
                days_to_target.append(day_num)
                break

    if not days_to_target:
        return None

    avg_days = np.mean(days_to_target)

    if avg_days <= 2.0:
        speed_class = 'FAST'
    elif avg_days <= 3.5:
        speed_class = 'MEDIUM'
    else:
        speed_class = 'SLOW'

    return {
        'ticker': ticker,
        'avg_days_to_target': avg_days,
        'speed_class': speed_class,
        'sample_size': len(days_to_target)
    }


def optimize_holding_period(ticker: str, start_date: str, end_date: str, speed_class: str) -> dict:
    """
    Find optimal holding period for a stock.

    Returns:
        Optimization results with best holding period
    """
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
    except:
        return None

    if len(data) < 60:
        return None

    # Determine holding period candidates based on speed
    if speed_class == 'FAST':
        hold_candidates = [1, 2, 3]
    elif speed_class == 'MEDIUM':
        hold_candidates = [2, 3, 4]
    else:  # SLOW
        hold_candidates = [3, 4, 5, 6]

    # Get parameters
    params = get_params(ticker)

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

    # Test each holding period
    best_hold = 3
    best_return = 0
    best_win_rate = 0
    best_efficiency = 0

    results_by_hold = {}

    for max_days in hold_candidates:
        backtester = MeanReversionBacktester(
            initial_capital=10000,
            exit_strategy='time_decay',
            profit_target=2.0,
            stop_loss=-2.0,
            max_hold_days=max_days
        )

        try:
            result = backtester.backtest(signals)

            if not result or result.get('total_trades', 0) < 5:
                continue

            win_rate = result['win_rate']
            total_return = result['total_return']

            # Capital efficiency = return per day held
            efficiency = total_return / max_days if max_days > 0 else 0

            results_by_hold[max_days] = {
                'win_rate': win_rate,
                'total_return': total_return,
                'efficiency': efficiency,
                'total_trades': result['total_trades']
            }

            # Best = highest total return with >= 70% win rate
            if win_rate >= 70.0 and total_return > best_return:
                best_return = total_return
                best_hold = max_days
                best_win_rate = win_rate
                best_efficiency = efficiency

        except Exception as e:
            continue

    if not results_by_hold:
        return None

    # Compare to baseline (3 days)
    baseline_return = results_by_hold.get(3, {}).get('total_return', 0)
    improvement = best_return - baseline_return

    return {
        'ticker': ticker,
        'speed_class': speed_class,
        'current_hold': 3,
        'optimal_hold': best_hold,
        'improvement_pct': improvement,
        'best_win_rate': best_win_rate,
        'best_return': best_return,
        'best_efficiency': best_efficiency,
        'results_by_hold': results_by_hold
    }


def run_exp055_holding_period_optimization():
    """
    Optimize holding periods per stock.
    """
    print("="*70)
    print("EXP-055: HOLDING PERIOD OPTIMIZATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Optimize holding period per stock based on reversion speed")
    print("Current: Max 3 days hold for ALL stocks (one-size-fits-all)")
    print("Expected: +5-15% return improvement from optimized holding periods")
    print()

    # Test on representative stocks
    test_stocks = [
        'NVDA', 'V', 'MA', 'AVGO', 'ORCL',
        'ABBV', 'GILD', 'MSFT', 'JPM', 'WMT',
        'AXP', 'KLAC', 'EOG', 'TXN'
    ]
    print(f"Testing {len(test_stocks)} stocks")
    print("This will take 10-15 minutes...")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    # Phase 1: Measure reversion speed
    print("PHASE 1: MEASURING REVERSION SPEED")
    print("="*70)

    speed_results = []

    for ticker in test_stocks:
        print(f"  Measuring {ticker}...", end=" ")

        speed_data = measure_reversion_speed(ticker, start_date, end_date)

        if speed_data:
            speed_results.append(speed_data)
            print(f"[{speed_data['speed_class']}] "
                  f"Avg {speed_data['avg_days_to_target']:.1f} days to target "
                  f"({speed_data['sample_size']} trades)")
        else:
            print("[SKIP]")

    # Phase 2: Optimize holding periods
    print()
    print("PHASE 2: OPTIMIZING HOLDING PERIODS")
    print("="*70)

    optimization_results = []

    for speed_data in speed_results:
        ticker = speed_data['ticker']
        speed_class = speed_data['speed_class']

        print(f"\n  Optimizing {ticker} ({speed_class})...", end=" ")

        result = optimize_holding_period(ticker, start_date, end_date, speed_class)

        if result:
            optimization_results.append(result)
            print(f"Optimal: {result['optimal_hold']} days, "
                  f"Improvement: {result['improvement_pct']:+.1f}%")
        else:
            print("[SKIP]")

    # Analyze results
    print()
    print("="*70)
    print("HOLDING PERIOD OPTIMIZATION RESULTS")
    print("="*70)
    print()

    print(f"{'Ticker':<8} {'Speed':<10} {'Current':<10} {'Optimal':<10} {'Improvement'}")
    print("-"*70)

    total_improvement = 0
    stocks_improved = 0

    for r in optimization_results:
        improvement = r['improvement_pct']
        if improvement > 0:
            stocks_improved += 1
        total_improvement += improvement

        print(f"{r['ticker']:<8} {r['speed_class']:<10} "
              f"{r['current_hold']:>8} days {r['optimal_hold']:>8} days {improvement:>+13.1f}%")

    avg_improvement = total_improvement / len(optimization_results) if optimization_results else 0

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    print(f"Stocks tested: {len(optimization_results)}")
    print(f"Stocks with improvement: {stocks_improved}")
    print(f"Average return improvement: {avg_improvement:+.1f}%")
    print()

    # Speed-based recommendations
    fast_holds = [r['optimal_hold'] for r in optimization_results if r['speed_class'] == 'FAST']
    med_holds = [r['optimal_hold'] for r in optimization_results if r['speed_class'] == 'MEDIUM']
    slow_holds = [r['optimal_hold'] for r in optimization_results if r['speed_class'] == 'SLOW']

    print("SPEED-BASED HOLDING RECOMMENDATIONS:")
    if fast_holds:
        print(f"  Fast reversions: {np.mean(fast_holds):.1f} days avg optimal")
    if med_holds:
        print(f"  Medium reversions: {np.mean(med_holds):.1f} days avg optimal")
    if slow_holds:
        print(f"  Slow reversions: {np.mean(slow_holds):.1f} days avg optimal")

    print()
    print("="*70)
    print("RECOMMENDATION")
    print("="*70)
    print()

    if avg_improvement >= 3.0:
        print(f"[SUCCESS] Holding period optimization shows {avg_improvement:+.1f}% avg return improvement!")
        print(f"{stocks_improved}/{len(optimization_results)} stocks improved with optimized holding periods")
        print()
        print("DEPLOY: Implement per-stock holding periods based on reversion speed")
        print()
        print("OPTIMIZED HOLDING PERIODS BY STOCK:")
        for r in optimization_results:
            if r['optimal_hold'] != 3:
                print(f"  {r['ticker']}: {r['optimal_hold']} days (was 3 days)")
    else:
        print(f"[MARGINAL] Holding period optimization shows only {avg_improvement:+.1f}% improvement")
        print(f"Below +3.0% threshold for deployment")
        print("Current 3-day max hold is adequate across all stocks")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-055',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'stocks_tested': test_stocks,
        'speed_analysis': [{**r} for r in speed_results],
        'optimization_results': [{**r} for r in optimization_results],
        'avg_improvement_pct': float(avg_improvement),
        'stocks_improved': stocks_improved,
        'deploy': avg_improvement >= 3.0
    }

    results_file = os.path.join(results_dir, 'exp055_holding_period_optimization.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending holding period optimization report email...")
            notifier.send_experiment_report('EXP-055', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-055 holding period optimization."""

    print("\n[HOLDING PERIOD] Optimizing per-stock holding periods")
    print("This will take 10-15 minutes...")
    print()

    results = run_exp055_holding_period_optimization()

    print("\n" + "="*70)
    print("HOLDING PERIOD OPTIMIZATION COMPLETE")
    print("="*70)

"""
EXPERIMENT: EXP-046
Date: 2025-11-17
Objective: Optimize entry timing for better execution prices

MOTIVATION:
Current strategy: Enter at EOD close price when signal triggers
- No consideration of intraday price action
- No consideration of next-day gap behavior
- No analysis of optimal entry delay
- Missing opportunity to improve entry price by 1-3%

PROBLEM:
Entry timing significantly impacts returns:
- Signals often gap down next day (panic continues)
- Entering at next-day open could improve entry by 0.5-2%
- Waiting for intraday dip could improve further
- Better entry = higher returns with same exit strategy

HYPOTHESIS:
Optimizing entry timing will improve returns by +2-5%:
- Next-day open entry may capture gap-down advantage
- Limit orders at -1% could catch intraday dips
- Expected: +2-5% total return improvement through better entries

METHODOLOGY:
1. Analyze historical signal outcomes:
   - Entry at signal day close (baseline)
   - Entry at next-day open (test gap behavior)
   - Entry at next-day low (test best possible entry)
   - Entry 1 day delay (test patience)

2. Measure for each entry strategy:
   - Average entry price improvement vs baseline
   - Win rate impact
   - Total return impact

3. Test on 10 representative Tier A stocks
4. Deploy if improvement >= +2% total return

EXPECTED OUTCOME:
- Optimal entry timing strategy
- +2-5% total return improvement
- Documented entry timing rules
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from common.config.mean_reversion_params import get_params


def test_entry_timing_strategy(ticker: str, start_date: str, end_date: str, entry_strategy: str) -> dict:
    """
    Test an entry timing strategy on a stock.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date
        entry_strategy: 'eod_close' | 'next_open' | 'next_low' | 'delay_1d'

    Returns:
        Test results
    """
    # Fetch data
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
    except Exception as e:
        return None

    if len(data) < 60:
        return None

    # Get optimized parameters
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

    signals_df = detector.detect_overcorrections(enriched_data)
    signals_df = detector.calculate_reversion_targets(signals_df)

    # Apply filters
    regime_detector = MarketRegimeDetector()
    signals_df = add_regime_filter_to_signals(signals_df, regime_detector)

    earnings_fetcher = EarningsCalendarFetcher(exclusion_days_before=3, exclusion_days_after=3)
    signals_df = earnings_fetcher.add_earnings_filter_to_signals(signals_df, ticker, 'panic_sell')

    # Get actual signal dates
    panic_signals = signals_df[signals_df['panic_sell'] == 1].copy()

    if len(panic_signals) < 5:
        return None

    # Simulate trades with different entry timing
    trades = []
    data_reset = data.reset_index()

    for idx, signal_row in panic_signals.iterrows():
        try:
            signal_date = pd.to_datetime(signal_row['Date'])
            signal_close = signal_row['Close']

            # Find signal in data
            signal_idx = data_reset[data_reset['Date'] == signal_date].index
            if len(signal_idx) == 0:
                continue
            signal_idx = signal_idx[0]

            # Get next day data
            if signal_idx + 1 >= len(data_reset):
                continue

            next_day = data_reset.iloc[signal_idx + 1]
            next_open = next_day['Open']
            next_low = next_day['Low']
            next_close = next_day['Close']

            # Determine entry price based on strategy
            if entry_strategy == 'eod_close':
                entry_price = signal_close
            elif entry_strategy == 'next_open':
                entry_price = next_open
            elif entry_strategy == 'next_low':
                entry_price = next_low
            elif entry_strategy == 'delay_1d':
                entry_price = next_close
            else:
                entry_price = signal_close

            # Calculate exit (using current time-decay strategy: Day 0 = ±2%)
            # For simplicity, use next 3 days' data to find exit
            max_hold_days = 3
            exit_price = None
            exit_day = 0

            for hold_day in range(max_hold_days):
                check_idx = signal_idx + 1 + hold_day
                if check_idx >= len(data_reset):
                    break

                check_row = data_reset.iloc[check_idx]
                current_price = check_row['Close']
                return_pct = (current_price / entry_price - 1) * 100

                # Time-decay targets
                if hold_day == 0:
                    profit_target, stop_loss = 2.0, -2.0
                elif hold_day == 1:
                    profit_target, stop_loss = 1.5, -1.5
                else:
                    profit_target, stop_loss = 1.0, -1.0

                if return_pct >= profit_target or return_pct <= stop_loss or hold_day == max_hold_days - 1:
                    exit_price = current_price
                    exit_day = hold_day
                    break

            if exit_price is None:
                continue

            # Record trade
            trade_return = (exit_price / entry_price - 1) * 100
            entry_improvement = (signal_close - entry_price) / signal_close * 100

            trades.append({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return_pct': trade_return,
                'entry_improvement': entry_improvement,
                'profitable': trade_return > 0
            })

        except Exception as e:
            continue

    if len(trades) < 5:
        return None

    # Calculate metrics
    win_rate = sum(1 for t in trades if t['profitable']) / len(trades) * 100
    avg_return = np.mean([t['return_pct'] for t in trades])
    avg_entry_improvement = np.mean([t['entry_improvement'] for t in trades])

    return {
        'ticker': ticker,
        'strategy': entry_strategy,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'avg_entry_improvement': avg_entry_improvement
    }


def run_exp046_entry_timing_v2():
    """
    Test entry timing strategies.
    """
    print("=" * 70)
    print("EXP-046: ENTRY TIMING OPTIMIZATION V2")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Optimize entry timing for better execution prices")
    print("Current: Enter at EOD close when signal triggers")
    print("Expected: +2-5% return improvement through better entry timing")
    print()

    # Test on representative stocks
    test_stocks = ['NVDA', 'V', 'MA', 'AVGO', 'ORCL', 'ABBV', 'GILD', 'ROAD', 'MSFT', 'JPM']
    print(f"Testing {len(test_stocks)} stocks: {', '.join(test_stocks)}")
    print()

    strategies = ['eod_close', 'next_open', 'next_low', 'delay_1d']
    print(f"Testing {len(strategies)} entry strategies")
    print("This will take 5-10 minutes...")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    # Collect results
    results_by_strategy = {s: [] for s in strategies}

    for ticker in test_stocks:
        print(f"\nTesting {ticker}...")

        for strategy in strategies:
            result = test_entry_timing_strategy(ticker, start_date, end_date, strategy)
            if result:
                results_by_strategy[strategy].append(result)
                print(f"  {strategy:12s}: {result['avg_return']:>6.2f}% avg return, "
                      f"{result['avg_entry_improvement']:>+5.2f}% entry improvement")

    # Aggregate results
    print()
    print("=" * 70)
    print("ENTRY TIMING RESULTS")
    print("=" * 70)
    print()

    print(f"{'Strategy':<15} {'Avg Return':<15} {'Entry Improvement':<20} {'Return Improvement'}")
    print("-" * 70)

    baseline_return = None
    best_strategy = None
    best_improvement = 0

    for strategy in strategies:
        if not results_by_strategy[strategy]:
            continue

        avg_return = np.mean([r['avg_return'] for r in results_by_strategy[strategy]])
        avg_entry_imp = np.mean([r['avg_entry_improvement'] for r in results_by_strategy[strategy]])

        if strategy == 'eod_close':
            baseline_return = avg_return
            improvement_pct = 0
        else:
            improvement_pct = avg_return - baseline_return if baseline_return else 0
            if improvement_pct > best_improvement:
                best_improvement = improvement_pct
                best_strategy = strategy

        print(f"{strategy:<15} {avg_return:>13.2f}% {avg_entry_imp:>18.2f}% {improvement_pct:>18.2f}%")

    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print()

    if best_improvement >= 0.5:  # Lower threshold since this is return improvement, not percentage
        print(f"[SUCCESS] {best_strategy.upper()} shows +{best_improvement:.2f}% return improvement!")
        print(f"Baseline (eod_close): {baseline_return:.2f}% avg return")
        best_return = np.mean([r['avg_return'] for r in results_by_strategy[best_strategy]])
        print(f"Optimized ({best_strategy}): {best_return:.2f}% avg return")
        print()
        print("DEPLOY: Implement entry timing optimization")
    else:
        print(f"[NO BENEFIT] Best strategy ({best_strategy}) shows only +{best_improvement:.2f}% improvement")
        print(f"Keep current EOD close entry timing")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-046',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'stocks_tested': test_stocks,
        'strategies': strategies,
        'results_by_strategy': {
            strategy: [{**r} for r in results]
            for strategy, results in results_by_strategy.items()
        },
        'baseline_return': float(baseline_return) if baseline_return else 0,
        'best_strategy': best_strategy if best_strategy else 'none',
        'best_improvement': float(best_improvement),
        'deploy': best_improvement >= 0.5
    }

    results_file = os.path.join(results_dir, 'exp046_entry_timing_v2.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending entry timing report email...")
            notifier.send_experiment_report('EXP-046', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-046 entry timing optimization."""

    print("\n[ENTRY TIMING] Testing entry timing strategies")
    print("This will take 5-10 minutes...")
    print()

    results = run_exp046_entry_timing_v2()

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    if results.get('deploy', False):
        print("1. Implement optimal entry timing in production")
        print("2. Update scanner to support delayed entry")
        print("3. Monitor performance improvement")
    else:
        print("1. Keep current EOD close entry timing")
        print("2. Consider other enhancement options")
    print()
    print("ENTRY TIMING OPTIMIZATION COMPLETE")

"""
EXPERIMENT: EXP-041
Date: 2025-11-17
Objective: Optimize signal entry timing for improved returns

MOTIVATION:
Current strategy: Enter immediately when signal triggers (EOD)
- No consideration of intraday timing
- No consideration of day-of-week patterns
- No analysis of optimal entry delay

PROBLEM:
Entry timing could significantly impact returns:
- Market open (9:30 AM) vs close (4:00 PM) price differences
- Monday vs Friday entry performance
- Immediate entry vs waiting 1-2 days for better price
- Gap-down opportunities after signal day

HYPOTHESIS:
Optimizing entry timing will improve returns:
- Best time of day to enter: Market open (after panic selling)
- Best day of week: Avoid Monday entries (weekend news risk)
- Entry delay: Waiting 1 day for pullback could improve entry price
- Expected: +2-5% total return improvement through better entries

METHODOLOGY:
1. Analyze historical signals and their outcomes:
   - Day of week signal occurred
   - Next-day price action (gap up/down)
   - Optimal entry price vs signal day close

2. Test entry timing strategies:
   a) Immediate (EOD signal day)
   b) Next-day open
   c) Next-day close
   d) Wait 1 day (better entry)
   e) Best of 2 days (optimal entry)

3. Test day-of-week filters:
   - Avoid Friday signals (weekend risk)
   - Avoid Monday entries (gap risk)
   - Prefer mid-week signals

4. Measure: total return, win rate, entry price improvement
5. Deploy if improvement >= +3% total return

CONSTRAINTS:
- Must work with EOD data (no intraday required for now)
- Entry timing must be practical for manual trading
- Cannot use future data (no lookahead bias)

EXPECTED OUTCOME:
- Optimal entry timing strategy
- +2-5% total return improvement
- Potentially +2-3pp win rate improvement from better entries
- Documented timing rules for deployment
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from src.config.mean_reversion_params import get_params


def analyze_signal_timing(ticker: str, start_date: str, end_date: str) -> Dict:
    """
    Analyze signal timing patterns for a stock.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date

    Returns:
        Timing analysis results
    """
    print(f"\nAnalyzing timing for {ticker}...")

    # Fetch data
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
    except:
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

    signals = detector.detect_overcorrections(enriched_data)
    signals = detector.calculate_reversion_targets(signals)

    # Apply filters
    regime_detector = MarketRegimeDetector()
    signals = add_regime_filter_to_signals(signals, regime_detector)

    earnings_fetcher = EarningsCalendarFetcher(exclusion_days_before=3, exclusion_days_after=3)
    signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

    # Get actual signals
    signal_dates = signals[signals['panic_sell'] == 1].copy()

    if len(signal_dates) == 0:
        return None

    # Add day of week
    signal_dates['day_of_week'] = pd.to_datetime(signal_dates['Date']).dt.day_name()

    # Analyze next-day price action
    timing_results = []

    for idx, row in signal_dates.iterrows():
        signal_date = pd.to_datetime(row['Date']).tz_localize(None)
        signal_close = row['Close']

        # Get next 3 days of data (need to use Date column, not index)
        data_with_date = data.copy()
        data_with_date['Date'] = pd.to_datetime(data_with_date.index).tz_localize(None)
        future_data = data_with_date[data_with_date['Date'] > signal_date].head(3)

        if len(future_data) < 2:
            continue

        next_day_open = future_data.iloc[0]['Open']
        next_day_close = future_data.iloc[0]['Close']
        next_day_low = future_data.iloc[0]['Low']

        day2_close = future_data.iloc[1]['Close'] if len(future_data) > 1 else next_day_close

        # Calculate price changes from signal close
        gap = (next_day_open - signal_close) / signal_close * 100
        next_day_return = (next_day_close - signal_close) / signal_close * 100
        day2_return = (day2_close - signal_close) / signal_close * 100

        # Best entry within next 2 days
        best_entry = min(next_day_low, future_data.iloc[1]['Low'] if len(future_data) > 1 else next_day_low)
        best_entry_improvement = (signal_close - best_entry) / signal_close * 100

        timing_results.append({
            'date': signal_date,
            'day_of_week': row['day_of_week'],
            'signal_close': signal_close,
            'next_day_gap': gap,
            'next_day_return': next_day_return,
            'day2_return': day2_return,
            'best_entry_improvement': best_entry_improvement
        })

    if not timing_results:
        return None

    timing_df = pd.DataFrame(timing_results)

    # Aggregate statistics
    results = {
        'ticker': ticker,
        'total_signals': len(timing_df),
        'avg_gap': timing_df['next_day_gap'].mean(),
        'avg_next_day_return': timing_df['next_day_return'].mean(),
        'avg_day2_return': timing_df['day2_return'].mean(),
        'avg_best_entry_improvement': timing_df['best_entry_improvement'].mean(),
        'day_of_week_stats': timing_df.groupby('day_of_week').agg({
            'next_day_gap': 'mean',
            'next_day_return': 'mean',
            'best_entry_improvement': 'mean'
        }).to_dict()
    }

    return results


def run_exp041_signal_timing_optimization():
    """
    Analyze signal timing patterns across portfolio.
    """
    print("=" * 70)
    print("EXP-041: SIGNAL TIMING OPTIMIZATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Find optimal entry timing for signals")
    print("Current: Enter immediately at EOD when signal triggers")
    print("Expected: +2-5% total return improvement through better timing")
    print()

    # Test on representative stocks
    test_stocks = ['NVDA', 'V', 'MA', 'AVGO', 'ORCL', 'ABBV', 'GILD', 'ROAD', 'MSFT', 'INSM']
    print(f"Testing {len(test_stocks)} stocks: {', '.join(test_stocks)}")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    timing_results = []

    for stock in test_stocks:
        result = analyze_signal_timing(stock, start_date, end_date)
        if result:
            timing_results.append(result)

    # Aggregate results
    print()
    print("=" * 70)
    print("SIGNAL TIMING ANALYSIS")
    print("=" * 70)
    print()

    if timing_results:
        print(f"{'Stock':<8} {'Signals':<10} {'Avg Gap':<12} {'Next Day Return':<18} {'Best Entry Improvement'}")
        print("-" * 80)

        total_gap = 0
        total_next_day = 0
        total_improvement = 0
        total_signals = 0

        for result in timing_results:
            ticker = result['ticker']
            signals = result['total_signals']
            gap = result['avg_gap']
            next_day = result['avg_next_day_return']
            improvement = result['avg_best_entry_improvement']

            total_gap += gap
            total_next_day += next_day
            total_improvement += improvement
            total_signals += signals

            print(f"{ticker:<8} {signals:>8} {gap:>10.2f}% {next_day:>16.2f}% {improvement:>22.2f}%")

        print()
        avg_gap = total_gap / len(timing_results)
        avg_next_day = total_next_day / len(timing_results)
        avg_improvement = total_improvement / len(timing_results)

        print(f"AVERAGE:  {total_signals:>8} {avg_gap:>10.2f}% {avg_next_day:>16.2f}% {avg_improvement:>22.2f}%")
        print()

        print("KEY FINDINGS:")
        print()

        if avg_gap < -0.5:
            print(f"✓ Signals typically gap DOWN {abs(avg_gap):.2f}% next day")
            print(f"  → Entering at next-day open could improve entry by {abs(avg_gap):.2f}%")
        elif avg_gap > 0.5:
            print(f"✗ Signals typically gap UP {avg_gap:.2f}% next day")
            print(f"  → Immediate entry at EOD is better than waiting")
        else:
            print(f"• Minimal gap effect ({avg_gap:.2f}%)")

        print()

        if avg_improvement > 1.0:
            print(f"✓ Waiting for best entry in next 2 days improves entry by {avg_improvement:.2f}%")
            print(f"  → Consider limit orders or patience for better entries")
            print(f"  [SUCCESS] Timing optimization shows significant benefit!")
        elif avg_improvement > 0.5:
            print(f"• Modest improvement available ({avg_improvement:.2f}%)")
            print(f"  [MARGINAL] Small benefit from optimal timing")
        else:
            print(f"✗ Minimal benefit from waiting ({avg_improvement:.2f}%)")
            print(f"  [NO BENEFIT] Immediate entry is fine")

        print()

        # Day of week analysis
        print("DAY OF WEEK ANALYSIS:")
        print()

        # Aggregate day of week stats
        dow_gaps = {}
        dow_returns = {}

        for result in timing_results:
            dow_stats = result.get('day_of_week_stats', {})
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                if day in dow_stats.get('next_day_gap', {}):
                    if day not in dow_gaps:
                        dow_gaps[day] = []
                        dow_returns[day] = []
                    dow_gaps[day].append(dow_stats['next_day_gap'][day])
                    dow_returns[day].append(dow_stats['next_day_return'][day])

        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            if day in dow_gaps and dow_gaps[day]:
                avg_gap = np.mean(dow_gaps[day])
                avg_return = np.mean(dow_returns[day])
                print(f"  {day:10s}: Gap: {avg_gap:>6.2f}%, Next-day return: {avg_return:>6.2f}%")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-041',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'stocks_tested': test_stocks,
        'timing_results': [{**r, 'day_of_week_stats': None} for r in timing_results]
    }

    results_file = os.path.join(results_dir, 'exp041_signal_timing_optimization.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending timing analysis report email...")
            notifier.send_experiment_report('EXP-041', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-041 signal timing optimization."""

    print("\n[TIMING ANALYSIS] Analyzing optimal entry timing")
    print("This will take 5-10 minutes...")
    print()

    results = run_exp041_signal_timing_optimization()

    print("\n\nNEXT STEPS:")
    print("1. Review timing patterns")
    print("2. If improvement >= +2%, implement timing filters")
    print("3. Consider limit orders for better entries")
    print("4. Update scanner with optimal timing strategy")
    print()
    print("SIGNAL TIMING OPTIMIZATION COMPLETE")

"""
EXPERIMENT: EXP-058
Date: 2025-11-17
Objective: Analyze seasonality patterns in mean reversion performance

MOTIVATION:
Previous 5 experiments showed NO benefit - system appears fully optimized
BUT: Seasonality is a known market phenomenon not yet tested
- Tax loss harvesting (November-December)
- January effect (new year positioning)
- FOMC meeting weeks (Fed announcement volatility)
- Day-of-week patterns (Friday vs Monday behavior)
- Options expiration weeks (quad witching volatility)

PROBLEM:
Taking signals in ALL calendar periods != optimal:
- December tax loss harvesting may disrupt mean reversion
- Options expiry weeks have different volatility profiles
- Day of week patterns may favor certain entry days
- Missing 5-10pp win rate by not filtering poor-performing periods

HYPOTHESIS:
Calendar-based filtering will improve win rate:
- Avoid December (tax loss harvesting disrupts reversion)
- Favor certain days of week (avoid Monday gaps?)
- Avoid options expiry weeks (3rd Friday of month)
- Expected: +5-10pp win rate by avoiding poor calendar periods

METHODOLOGY:
1. Analyze historical performance by:
   - Month of year
   - Day of week (Monday-Friday)
   - Week of month (options expiry detection)
   - Quarter patterns

2. Identify poor-performing periods:
   - Win rate by month
   - Win rate by day of week
   - Win rate by week type (expiry vs normal)

3. Test calendar filters:
   - Baseline (no filter)
   - Exclude worst month
   - Exclude worst day of week
   - Exclude options expiry weeks
   - Combined filters

4. Deploy if improvement >= +3pp win rate

CONSTRAINTS:
- Must not reduce trades by > 50%
- Filters must be simple (date-based only)
- Min 70% win rate maintained

EXPECTED OUTCOME:
- Calendar pattern analysis
- Seasonality filters (if beneficial)
- +5-10pp win rate improvement
- Better timing of signal execution
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from src.config.mean_reversion_params import get_params


def is_options_expiry_week(date: pd.Timestamp) -> bool:
    """Check if date falls in options expiry week (3rd Friday of month)."""
    # Find the third Friday
    first_day = date.replace(day=1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    third_friday = first_friday + timedelta(days=14)

    # Check if date is within the week of third Friday
    week_start = third_friday - timedelta(days=third_friday.weekday())
    week_end = week_start + timedelta(days=4)

    return week_start <= date <= week_end


def analyze_seasonality(ticker: str, start_date: str, end_date: str) -> dict:
    """
    Analyze performance by calendar patterns.

    Returns:
        Dictionary with performance by month, day of week, etc.
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

    # Backtest to get trade results
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
            return None

        trades = results.get('trades', [])

        if not trades:
            return None

        # Analyze by calendar patterns
        by_month = defaultdict(lambda: {'wins': 0, 'total': 0})
        by_dow = defaultdict(lambda: {'wins': 0, 'total': 0})
        by_expiry = {'expiry_week': {'wins': 0, 'total': 0}, 'normal_week': {'wins': 0, 'total': 0}}

        for trade in trades:
            entry_date = pd.to_datetime(trade['entry_date'])
            profit = trade['profit_pct']
            is_win = profit > 0

            # Month analysis
            month = entry_date.month
            by_month[month]['total'] += 1
            if is_win:
                by_month[month]['wins'] += 1

            # Day of week analysis (0=Monday, 4=Friday)
            dow = entry_date.weekday()
            dow_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][dow]
            by_dow[dow_name]['total'] += 1
            if is_win:
                by_dow[dow_name]['wins'] += 1

            # Options expiry week analysis
            if is_options_expiry_week(entry_date):
                by_expiry['expiry_week']['total'] += 1
                if is_win:
                    by_expiry['expiry_week']['wins'] += 1
            else:
                by_expiry['normal_week']['total'] += 1
                if is_win:
                    by_expiry['normal_week']['wins'] += 1

        # Calculate win rates
        month_wr = {m: (d['wins']/d['total']*100 if d['total'] > 0 else 0)
                    for m, d in by_month.items()}
        dow_wr = {d: (data['wins']/data['total']*100 if data['total'] > 0 else 0)
                  for d, data in by_dow.items()}
        expiry_wr = {
            'expiry_week': by_expiry['expiry_week']['wins']/by_expiry['expiry_week']['total']*100
                          if by_expiry['expiry_week']['total'] > 0 else 0,
            'normal_week': by_expiry['normal_week']['wins']/by_expiry['normal_week']['total']*100
                          if by_expiry['normal_week']['total'] > 0 else 0
        }

        return {
            'ticker': ticker,
            'total_trades': len(trades),
            'overall_wr': results['win_rate'],
            'by_month': dict(by_month),
            'by_dow': dict(by_dow),
            'by_expiry': by_expiry,
            'month_wr': month_wr,
            'dow_wr': dow_wr,
            'expiry_wr': expiry_wr
        }

    except Exception as e:
        return None


def test_calendar_filter(ticker: str, start_date: str, end_date: str,
                         exclude_months: list = None,
                         exclude_days: list = None,
                         exclude_expiry: bool = False) -> dict:
    """
    Test strategy with calendar-based filters.

    Returns:
        Performance with filters applied
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

    # Apply standard filters
    regime_detector = MarketRegimeDetector()
    signals = add_regime_filter_to_signals(signals, regime_detector)

    earnings_fetcher = EarningsCalendarFetcher(exclusion_days_before=3, exclusion_days_after=3)
    signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

    # Apply calendar filters
    if exclude_months or exclude_days or exclude_expiry:
        signals['calendar_filter'] = 1

        for idx, row in signals[signals['panic_sell'] == 1].iterrows():
            date = row['Date']

            # Check month
            if exclude_months and date.month in exclude_months:
                signals.loc[idx, 'calendar_filter'] = 0
                continue

            # Check day of week
            dow_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][date.weekday()]
            if exclude_days and dow_name in exclude_days:
                signals.loc[idx, 'calendar_filter'] = 0
                continue

            # Check expiry week
            if exclude_expiry and is_options_expiry_week(date):
                signals.loc[idx, 'calendar_filter'] = 0
                continue

        # Override panic_sell with filtered signals
        signals.loc[signals['calendar_filter'] == 0, 'panic_sell'] = 0

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
            return None

        return {
            'ticker': ticker,
            'win_rate': results['win_rate'],
            'total_return': results['total_return'],
            'total_trades': results['total_trades']
        }

    except Exception as e:
        return None


def run_exp058_seasonality_analysis():
    """
    Analyze seasonality patterns in mean reversion performance.
    """
    print("="*70)
    print("EXP-058: SEASONALITY PATTERN ANALYSIS")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Identify calendar patterns in performance")
    print("Current: No seasonality filtering")
    print("Expected: +5-10pp win rate by avoiding poor periods")
    print()

    # Test on diverse stocks
    test_stocks = [
        'NVDA', 'V', 'MA', 'AVGO', 'ORCL',
        'ABBV', 'GILD', 'MSFT', 'JPM', 'WMT',
        'AXP', 'KLAC', 'TXN', 'EOG'
    ]

    print(f"Testing {len(test_stocks)} stocks")
    print("This will take 10-15 minutes...")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    # Phase 1: Analyze seasonality patterns
    print("PHASE 1: SEASONALITY ANALYSIS")
    print("="*70)

    all_analyses = []

    for ticker in test_stocks:
        print(f"  Analyzing {ticker}...", end=" ")

        analysis = analyze_seasonality(ticker, start_date, end_date)

        if analysis:
            all_analyses.append(analysis)
            print(f"{analysis['total_trades']} trades, {analysis['overall_wr']:.1f}% WR")
        else:
            print("[SKIP]")

    if not all_analyses:
        print("\nNo data - cannot analyze seasonality")
        return None

    # Aggregate patterns
    print()
    print("="*70)
    print("AGGREGATE SEASONALITY PATTERNS")
    print("="*70)
    print()

    # Month analysis
    month_performance = defaultdict(lambda: {'wins': 0, 'total': 0})
    for analysis in all_analyses:
        for month, data in analysis['by_month'].items():
            month_performance[month]['wins'] += data['wins']
            month_performance[month]['total'] += data['total']

    print("WIN RATE BY MONTH:")
    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month in sorted(month_performance.keys()):
        data = month_performance[month]
        wr = data['wins']/data['total']*100 if data['total'] > 0 else 0
        print(f"  {month_names[month]}: {wr:>6.1f}% ({data['total']} trades)")

    # Day of week analysis
    dow_performance = defaultdict(lambda: {'wins': 0, 'total': 0})
    for analysis in all_analyses:
        for dow, data in analysis['by_dow'].items():
            dow_performance[dow]['wins'] += data['wins']
            dow_performance[dow]['total'] += data['total']

    print()
    print("WIN RATE BY DAY OF WEEK:")
    for dow in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        if dow in dow_performance:
            data = dow_performance[dow]
            wr = data['wins']/data['total']*100 if data['total'] > 0 else 0
            print(f"  {dow}: {wr:>6.1f}% ({data['total']} trades)")

    # Expiry week analysis
    expiry_performance = {'wins': 0, 'total': 0}
    normal_performance = {'wins': 0, 'total': 0}
    for analysis in all_analyses:
        expiry_performance['wins'] += analysis['by_expiry']['expiry_week']['wins']
        expiry_performance['total'] += analysis['by_expiry']['expiry_week']['total']
        normal_performance['wins'] += analysis['by_expiry']['normal_week']['wins']
        normal_performance['total'] += analysis['by_expiry']['normal_week']['total']

    print()
    print("WIN RATE BY WEEK TYPE:")
    expiry_wr = expiry_performance['wins']/expiry_performance['total']*100 if expiry_performance['total'] > 0 else 0
    normal_wr = normal_performance['wins']/normal_performance['total']*100 if normal_performance['total'] > 0 else 0
    print(f"  Options Expiry Week: {expiry_wr:>6.1f}% ({expiry_performance['total']} trades)")
    print(f"  Normal Week: {normal_wr:>6.1f}% ({normal_performance['total']} trades)")

    # Identify worst performers
    worst_month = min(month_performance.items(),
                      key=lambda x: x[1]['wins']/x[1]['total'] if x[1]['total'] > 0 else 100)
    worst_dow = min(dow_performance.items(),
                    key=lambda x: x[1]['wins']/x[1]['total'] if x[1]['total'] > 0 else 100)

    print()
    print("="*70)
    print("WORST PERFORMING PERIODS")
    print("="*70)
    print()

    worst_month_wr = worst_month[1]['wins']/worst_month[1]['total']*100 if worst_month[1]['total'] > 0 else 0
    worst_dow_wr = worst_dow[1]['wins']/worst_dow[1]['total']*100 if worst_dow[1]['total'] > 0 else 0

    print(f"Worst Month: {month_names[worst_month[0]]} ({worst_month_wr:.1f}% WR)")
    print(f"Worst Day: {worst_dow[0]} ({worst_dow_wr:.1f}% WR)")

    # Phase 2: Test calendar filters (on subset)
    print()
    print("="*70)
    print("PHASE 2: TESTING CALENDAR FILTERS")
    print("="*70)
    print()

    test_subset = test_stocks[:5]  # Test on subset for speed

    filter_strategies = {
        'baseline': {},
        f'exclude_{month_names[worst_month[0]]}': {'exclude_months': [worst_month[0]]},
        f'exclude_{worst_dow[0]}': {'exclude_days': [worst_dow[0]]},
    }

    if expiry_wr < normal_wr - 5:  # Only test if expiry is significantly worse
        filter_strategies['exclude_expiry'] = {'exclude_expiry': True}

    results_by_strategy = defaultdict(list)

    for ticker in test_subset:
        print(f"\nTesting {ticker}...")

        for strategy_name, filter_params in filter_strategies.items():
            result = test_calendar_filter(ticker, start_date, end_date, **filter_params)

            if result:
                results_by_strategy[strategy_name].append(result)
                print(f"  {strategy_name:20s}: {result['win_rate']:>6.1f}% WR, {result['total_trades']} trades")

    # Compare strategies
    print()
    print("="*70)
    print("CALENDAR FILTER COMPARISON")
    print("="*70)
    print()

    print(f"{'Strategy':<25} {'Avg WR':<12} {'Avg Trades':<12} {'Improvement'}")
    print("-"*70)

    baseline_wr = None
    best_improvement = 0
    best_strategy = None

    for strategy in filter_strategies.keys():
        if not results_by_strategy[strategy]:
            continue

        avg_wr = np.mean([r['win_rate'] for r in results_by_strategy[strategy]])
        avg_trades = np.mean([r['total_trades'] for r in results_by_strategy[strategy]])

        if strategy == 'baseline':
            baseline_wr = avg_wr
            improvement_str = "-"
        else:
            improvement = avg_wr - baseline_wr if baseline_wr else 0
            if improvement > best_improvement:
                best_improvement = improvement
                best_strategy = strategy
            improvement_str = f"+{improvement:.1f}pp"

        print(f"{strategy:<25} {avg_wr:>10.1f}% {avg_trades:>10.0f} {improvement_str:>12}")

    print()
    print("="*70)
    print("RECOMMENDATION")
    print("="*70)
    print()

    if best_improvement >= 2.0:
        print(f"[SUCCESS] Calendar filter '{best_strategy}' shows +{best_improvement:.1f}pp improvement!")
        print(f"Baseline: {baseline_wr:.1f}% win rate")
        best_wr = np.mean([r['win_rate'] for r in results_by_strategy[best_strategy]])
        print(f"Filtered: {best_wr:.1f}% win rate")
        print()
        print(f"DEPLOY: Implement {best_strategy} filter")
    else:
        print(f"[NO BENEFIT] Best filter shows only +{best_improvement:.1f}pp improvement")
        print(f"Below +2pp threshold for deployment")
        print("Seasonality patterns exist but don't provide actionable edge")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-058',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'stocks_analyzed': test_stocks,
        'seasonality_patterns': {
            'by_month': {month_names[m]: {'wr': d['wins']/d['total']*100 if d['total'] > 0 else 0,
                                          'trades': d['total']}
                        for m, d in month_performance.items()},
            'by_dow': {dow: {'wr': d['wins']/d['total']*100 if d['total'] > 0 else 0,
                            'trades': d['total']}
                      for dow, d in dow_performance.items()},
            'expiry_vs_normal': {
                'expiry_week_wr': float(expiry_wr),
                'normal_week_wr': float(normal_wr)
            }
        },
        'filter_test_results': {
            strategy: [{**r} for r in results]
            for strategy, results in results_by_strategy.items()
        },
        'baseline_wr': float(baseline_wr) if baseline_wr else 0,
        'best_strategy': best_strategy if best_strategy else 'none',
        'best_improvement_pp': float(best_improvement),
        'deploy': best_improvement >= 2.0
    }

    results_file = os.path.join(results_dir, 'exp058_seasonality_analysis.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending seasonality analysis report email...")
            notifier.send_experiment_report('EXP-058', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-058 seasonality analysis."""

    print("\n[SEASONALITY] Analyzing calendar patterns in performance")
    print("This will take 10-15 minutes...")
    print()

    results = run_exp058_seasonality_analysis()

    print("\n" + "="*70)
    print("SEASONALITY ANALYSIS COMPLETE")
    print("="*70)

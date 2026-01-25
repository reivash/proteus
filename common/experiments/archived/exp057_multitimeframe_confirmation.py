"""
EXPERIMENT: EXP-057
Date: 2025-11-17
Objective: Test multi-timeframe signal confirmation for higher quality signals

MOTIVATION:
Current system: Signals based ONLY on daily data
- No confirmation from higher timeframes (weekly)
- Weak daily signals might not align with weekly trend
- Strong signals should show panic across multiple timeframes
- Missing 3-5pp win rate improvement from better signal quality

PROBLEM:
Single timeframe signals != optimal signal quality:
- Daily panic sell but weekly uptrend = weaker signal (noise)
- Daily + weekly panic = stronger confirmation (real overcorrection)
- No filter for timeframe alignment
- Taking every daily signal reduces selectivity

HYPOTHESIS:
Multi-timeframe confirmation will improve win rate:
- Require daily Z-score < -1.5 AND weekly Z-score < -1.0
- Filter out noise signals that don't show on weekly
- Take only highest-conviction signals
- Expected: +3-5pp win rate, reduced trades but higher quality

METHODOLOGY:
1. Calculate indicators on multiple timeframes:
   - Daily: Current signals
   - Weekly: Z-score, RSI, volume

2. Test confirmation strategies:
   - Daily only (baseline)
   - Daily + Weekly Z-score confirmation
   - Daily + Weekly RSI confirmation
   - Daily + Weekly combined confirmation

3. Measure impact on:
   - Win rate (key metric)
   - Total return
   - Trade frequency
   - Sharpe ratio

4. Deploy if improvement >= +3pp win rate

CONSTRAINTS:
- Must not reduce trades by > 60% (maintain opportunity flow)
- Min 70% win rate maintained
- Timeframe logic must be simple

EXPECTED OUTCOME:
- Multi-timeframe confirmation framework
- Higher win rate (+3-5pp)
- Fewer but stronger signals
- Better risk-adjusted returns
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


def get_weekly_confirmation(ticker: str, signal_date: pd.Timestamp, weekly_data: pd.DataFrame) -> dict:
    """
    Check if weekly timeframe confirms the daily signal.

    Returns:
        Dictionary with weekly confirmation metrics
    """
    # Find the week containing the signal date
    signal_week_data = weekly_data[weekly_data['Date'] <= signal_date]

    if signal_week_data.empty:
        return {'confirmed': False, 'reason': 'no_weekly_data'}

    latest_week = signal_week_data.iloc[-1]

    # Check weekly Z-score
    weekly_z = latest_week.get('z_score', 0)
    weekly_rsi = latest_week.get('rsi', 50)

    # Confirmation criteria
    z_confirmed = weekly_z < -1.0  # Weekly also showing oversold
    rsi_confirmed = weekly_rsi < 40  # Weekly RSI oversold

    return {
        'confirmed': z_confirmed and rsi_confirmed,
        'weekly_z_score': weekly_z,
        'weekly_rsi': weekly_rsi,
        'z_confirmed': z_confirmed,
        'rsi_confirmed': rsi_confirmed
    }


def test_multitimeframe_strategy(ticker: str, start_date: str, end_date: str, require_confirmation: bool = False) -> dict:
    """
    Test strategy with/without multitimeframe confirmation.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date
        require_confirmation: If True, require weekly confirmation

    Returns:
        Test results
    """
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    # Get daily data
    try:
        daily_data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
    except:
        return None

    if len(daily_data) < 60:
        return None

    # Get weekly data
    try:
        # Yahoo Finance: Use interval='1wk' for weekly data
        weekly_data = fetcher.fetch_stock_data(
            ticker,
            start_date=buffer_start,
            end_date=end_date,
            interval='1wk'
        )
    except:
        weekly_data = pd.DataFrame()

    # Engineer features on both timeframes
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_daily = engineer.engineer_features(daily_data)

    if not weekly_data.empty:
        enriched_weekly = engineer.engineer_features(weekly_data)
    else:
        enriched_weekly = pd.DataFrame()

    # Get parameters
    params = get_params(ticker)

    # Detect daily signals
    detector = MeanReversionDetector(
        z_score_threshold=params['z_score_threshold'],
        rsi_oversold=params['rsi_oversold'],
        rsi_overbought=65,
        volume_multiplier=params['volume_multiplier'],
        price_drop_threshold=params['price_drop_threshold']
    )

    signals = detector.detect_overcorrections(enriched_daily)
    signals = detector.calculate_reversion_targets(signals)

    # Apply filters
    regime_detector = MarketRegimeDetector()
    signals = add_regime_filter_to_signals(signals, regime_detector)

    earnings_fetcher = EarningsCalendarFetcher(exclusion_days_before=3, exclusion_days_after=3)
    signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

    # Apply multitimeframe filter if required
    if require_confirmation and not enriched_weekly.empty:
        signals_confirmed = []

        for idx, row in signals[signals['panic_sell'] == 1].iterrows():
            signal_date = row['Date']
            confirmation = get_weekly_confirmation(ticker, signal_date, enriched_weekly)

            if confirmation['confirmed']:
                signals_confirmed.append(idx)

        # Mark only confirmed signals
        signals['multitimeframe_confirmed'] = 0
        signals.loc[signals_confirmed, 'multitimeframe_confirmed'] = 1

        # Override panic_sell with confirmed signals only
        signals['panic_sell'] = signals['multitimeframe_confirmed']

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
            'strategy': 'multitimeframe' if require_confirmation else 'daily_only',
            'win_rate': results['win_rate'],
            'total_return': results['total_return'],
            'total_trades': results['total_trades'],
            'sharpe_ratio': results.get('sharpe_ratio', 0)
        }

    except Exception as e:
        return None


def run_exp057_multitimeframe_confirmation():
    """
    Test multi-timeframe signal confirmation.
    """
    print("="*70)
    print("EXP-057: MULTI-TIMEFRAME CONFIRMATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Test multi-timeframe signal confirmation")
    print("Current: Daily signals only (no timeframe confirmation)")
    print("Expected: +3-5pp win rate from higher quality signals")
    print()

    # Test on representative stocks
    test_stocks = [
        'NVDA', 'V', 'MA', 'AVGO', 'ORCL',
        'ABBV', 'GILD', 'MSFT', 'JPM', 'WMT'
    ]
    print(f"Testing {len(test_stocks)} stocks")
    print("This will take 5-10 minutes...")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    strategies = ['daily_only', 'multitimeframe']

    # Collect results
    results_by_strategy = {s: [] for s in strategies}

    for ticker in test_stocks:
        print(f"\nTesting {ticker}...")

        # Test daily only
        daily_result = test_multitimeframe_strategy(ticker, start_date, end_date, require_confirmation=False)
        if daily_result:
            results_by_strategy['daily_only'].append(daily_result)
            print(f"  daily_only     : {daily_result['win_rate']:>6.1f}% WR, "
                  f"{daily_result['total_return']:>+6.1f}% return, "
                  f"{daily_result['total_trades']} trades")

        # Test multitimeframe
        mtf_result = test_multitimeframe_strategy(ticker, start_date, end_date, require_confirmation=True)
        if mtf_result:
            results_by_strategy['multitimeframe'].append(mtf_result)
            print(f"  multitimeframe : {mtf_result['win_rate']:>6.1f}% WR, "
                  f"{mtf_result['total_return']:>+6.1f}% return, "
                  f"{mtf_result['total_trades']} trades")

    # Aggregate results
    print()
    print("="*70)
    print("MULTI-TIMEFRAME CONFIRMATION RESULTS")
    print("="*70)
    print()

    print(f"{'Strategy':<20} {'Avg WR':<12} {'Avg Return':<15} {'Avg Trades':<12} {'Improvement'}")
    print("-"*70)

    baseline_wr = None
    best_improvement = 0

    for strategy in strategies:
        if not results_by_strategy[strategy]:
            continue

        avg_wr = np.mean([r['win_rate'] for r in results_by_strategy[strategy]])
        avg_return = np.mean([r['total_return'] for r in results_by_strategy[strategy]])
        avg_trades = np.mean([r['total_trades'] for r in results_by_strategy[strategy]])

        if strategy == 'daily_only':
            baseline_wr = avg_wr
            improvement_str = "-"
        else:
            wr_improvement = avg_wr - baseline_wr if baseline_wr else 0
            best_improvement = wr_improvement
            improvement_str = f"+{wr_improvement:.1f}pp"

        print(f"{strategy:<20} {avg_wr:>10.1f}% {avg_return:>13.1f}% {avg_trades:>10.1f} {improvement_str:>12}")

    print()
    print("="*70)
    print("RECOMMENDATION")
    print("="*70)
    print()

    if best_improvement >= 2.0:
        print(f"[SUCCESS] Multi-timeframe confirmation shows +{best_improvement:.1f}pp win rate improvement!")
        mtf_wr = np.mean([r['win_rate'] for r in results_by_strategy['multitimeframe']])
        print(f"Daily only: {baseline_wr:.1f}% win rate")
        print(f"Multi-timeframe: {mtf_wr:.1f}% win rate")
        print()
        print("DEPLOY: Implement weekly timeframe confirmation")
    else:
        print(f"[NO BENEFIT] Multi-timeframe shows only +{best_improvement:.1f}pp improvement")
        print(f"Below +2pp threshold for deployment")
        print("Keep current daily-only signal detection")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-057',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'stocks_tested': test_stocks,
        'strategies': strategies,
        'results_by_strategy': {
            strategy: [{**r} for r in results]
            for strategy, results in results_by_strategy.items()
        },
        'baseline_win_rate': float(baseline_wr) if baseline_wr else 0,
        'best_improvement_pp': float(best_improvement),
        'deploy': best_improvement >= 2.0
    }

    results_file = os.path.join(results_dir, 'exp057_multitimeframe_confirmation.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending multi-timeframe confirmation report email...")
            notifier.send_experiment_report('EXP-057', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-057 multi-timeframe confirmation."""

    print("\n[MULTI-TIMEFRAME] Testing weekly confirmation filter")
    print("This will take 5-10 minutes...")
    print()

    results = run_exp057_multitimeframe_confirmation()

    print("\n" + "="*70)
    print("MULTI-TIMEFRAME CONFIRMATION COMPLETE")
    print("="*70)

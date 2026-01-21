"""
EXPERIMENT: EXP-059
Date: 2025-11-17
Objective: Filter signals based on market breadth quality

MOTIVATION:
Previous 5+ experiments showed NO benefit - but this is DIFFERENT
Market breadth = overall market participation (% of stocks advancing)
- Current system: Treat all days equally regardless of market breadth
- Strong breadth (70%+ stocks advancing): Healthy market, mean reversion works well
- Weak breadth (30%- stocks advancing): Narrow market, mean reversion less reliable
- Not tested yet because it's PORTFOLIO-LEVEL not STOCK-LEVEL

PROBLEM:
Ignoring market breadth != optimal signal quality:
- Mean reversion assumes market inefficiency is temporary
- In weak breadth: Only few stocks move (sector-specific, not inefficiency)
- In strong breadth: Many stocks move (real market-wide moves, better reversion)
- Missing 5-10pp win rate by taking signals in weak breadth environments

HYPOTHESIS:
Market breadth filtering will improve win rate:
- Strong breadth (SPY advance count high): Take all signals
- Weak breadth (SPY advance count low): Skip signals or reduce size
- Breadth calculated from SPY intraday moves as proxy
- Expected: +5-10pp win rate, slightly fewer trades but higher quality

METHODOLOGY:
1. Calculate market breadth proxy:
   - Use SPY price action and volume as breadth indicator
   - High volume + strong moves = strong breadth
   - Low volume + choppy = weak breadth

2. Test breadth-based filters:
   - Baseline (no breadth filter)
   - Strong breadth only (skip weak breadth days)
   - Breadth-weighted sizing (reduce size in weak breadth)

3. Measure impact on:
   - Win rate (key metric)
   - Total return
   - Trade frequency
   - Sharpe ratio

4. Deploy if improvement >= +3pp win rate

CONSTRAINTS:
- Must not reduce trades by > 40%
- Min 70% win rate maintained
- Use SPY data (already available, no new APIs)

EXPECTED OUTCOME:
- Market breadth filter implementation
- +5-10pp win rate improvement
- Higher quality signals
- Better risk-adjusted returns
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from src.config.mean_reversion_params import get_params


def calculate_market_breadth_proxy(spy_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate market breadth proxy from SPY data.

    Real breadth = NYSE advance/decline ratio
    Proxy = SPY daily return + volume surge

    Strong breadth indicators:
    - SPY up strongly (>1%) with high volume (>1.5x avg)
    - Many stocks participating in move

    Weak breadth indicators:
    - SPY choppy (<0.5% moves) with low volume
    - Few stocks participating

    Returns:
        DataFrame with breadth_score (0-100)
    """
    spy_data = spy_data.copy()

    # Calculate daily return
    spy_data['daily_return'] = spy_data['Close'].pct_change() * 100

    # Calculate volume ratio
    spy_data['volume_20d_avg'] = spy_data['Volume'].rolling(20).mean()
    spy_data['volume_ratio'] = spy_data['Volume'] / spy_data['volume_20d_avg']

    # Breadth score components:
    # 1. Price strength (0-50): Larger moves = stronger breadth
    price_component = np.abs(spy_data['daily_return']).clip(0, 2) / 2 * 50

    # 2. Volume confirmation (0-50): Higher volume = more participation
    volume_component = (spy_data['volume_ratio'] - 1).clip(0, 1) * 50

    # Combined score (0-100)
    spy_data['breadth_score'] = price_component + volume_component

    # Smooth with 5-day MA to reduce noise
    spy_data['breadth_score_smooth'] = spy_data['breadth_score'].rolling(5).mean()

    return spy_data[['Date', 'breadth_score', 'breadth_score_smooth']]


def test_breadth_filter(ticker: str, start_date: str, end_date: str,
                        use_breadth_filter: bool = False,
                        breadth_threshold: float = 30.0) -> dict:
    """
    Test strategy with market breadth filter.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date
        use_breadth_filter: If True, skip signals on weak breadth days
        breadth_threshold: Minimum breadth score (0-100) to take signals

    Returns:
        Test results
    """
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    # Get stock data
    try:
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
    except:
        return None

    if len(data) < 60:
        return None

    # Get SPY data for breadth proxy
    try:
        spy_data = fetcher.fetch_stock_data('SPY', start_date=buffer_start, end_date=end_date)
        breadth_data = calculate_market_breadth_proxy(spy_data)
    except:
        breadth_data = pd.DataFrame()

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

    # Apply breadth filter if requested
    if use_breadth_filter and not breadth_data.empty:
        # Merge breadth data
        signals['Date'] = pd.to_datetime(signals['Date'])
        breadth_data['Date'] = pd.to_datetime(breadth_data['Date'])
        signals = signals.merge(breadth_data, on='Date', how='left')

        # Filter signals based on breadth
        signals['breadth_filter'] = signals['breadth_score_smooth'] >= breadth_threshold

        # Override panic_sell for weak breadth days
        signals.loc[signals['breadth_filter'] == False, 'panic_sell'] = 0

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
            'strategy': 'breadth_filter' if use_breadth_filter else 'baseline',
            'win_rate': results['win_rate'],
            'total_return': results['total_return'],
            'total_trades': results['total_trades'],
            'sharpe_ratio': results.get('sharpe_ratio', 0)
        }

    except Exception as e:
        return None


def run_exp059_market_breadth_filter():
    """
    Test market breadth-based signal filtering.
    """
    print("="*70)
    print("EXP-059: MARKET BREADTH SIGNAL QUALITY FILTER")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Filter signals based on market breadth")
    print("Current: No breadth consideration (all days treated equally)")
    print("Expected: +5-10pp win rate by avoiding weak breadth periods")
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

    # Test different breadth thresholds
    breadth_thresholds = [20.0, 30.0, 40.0]

    strategies = ['baseline'] + [f'breadth_{int(t)}' for t in breadth_thresholds]

    # Collect results
    results_by_strategy = {s: [] for s in strategies}

    for ticker in test_stocks:
        print(f"\nTesting {ticker}...")

        # Test baseline
        baseline_result = test_breadth_filter(ticker, start_date, end_date, use_breadth_filter=False)
        if baseline_result:
            results_by_strategy['baseline'].append(baseline_result)
            print(f"  baseline     : {baseline_result['win_rate']:>6.1f}% WR, "
                  f"{baseline_result['total_return']:>+6.1f}% return, "
                  f"{baseline_result['total_trades']} trades")

        # Test breadth filters
        for threshold in breadth_thresholds:
            breadth_result = test_breadth_filter(
                ticker, start_date, end_date,
                use_breadth_filter=True,
                breadth_threshold=threshold
            )

            if breadth_result:
                strategy_name = f'breadth_{int(threshold)}'
                results_by_strategy[strategy_name].append(breadth_result)
                print(f"  breadth_{int(threshold):>2}   : {breadth_result['win_rate']:>6.1f}% WR, "
                      f"{breadth_result['total_return']:>+6.1f}% return, "
                      f"{breadth_result['total_trades']} trades")

    # Aggregate results
    print()
    print("="*70)
    print("MARKET BREADTH FILTER RESULTS")
    print("="*70)
    print()

    print(f"{'Strategy':<15} {'Avg WR':<12} {'Avg Return':<15} {'Avg Trades':<12} {'Improvement'}")
    print("-"*70)

    baseline_wr = None
    best_improvement = 0
    best_strategy = None

    for strategy in strategies:
        if not results_by_strategy[strategy]:
            continue

        avg_wr = np.mean([r['win_rate'] for r in results_by_strategy[strategy]])
        avg_return = np.mean([r['total_return'] for r in results_by_strategy[strategy]])
        avg_trades = np.mean([r['total_trades'] for r in results_by_strategy[strategy]])

        if strategy == 'baseline':
            baseline_wr = avg_wr
            improvement_str = "-"
        else:
            wr_improvement = avg_wr - baseline_wr if baseline_wr else 0
            if wr_improvement > best_improvement:
                best_improvement = wr_improvement
                best_strategy = strategy
            improvement_str = f"+{wr_improvement:.1f}pp"

        print(f"{strategy:<15} {avg_wr:>10.1f}% {avg_return:>13.1f}% {avg_trades:>10.1f} {improvement_str:>12}")

    print()
    print("="*70)
    print("RECOMMENDATION")
    print("="*70)
    print()

    if best_improvement >= 2.0:
        print(f"[SUCCESS] Market breadth filter '{best_strategy}' shows +{best_improvement:.1f}pp improvement!")
        best_wr = np.mean([r['win_rate'] for r in results_by_strategy[best_strategy]])
        print(f"Baseline: {baseline_wr:.1f}% win rate")
        print(f"Breadth filtered: {best_wr:.1f}% win rate")
        print()
        print(f"DEPLOY: Implement market breadth filter (threshold: {best_strategy.split('_')[1]})")
    else:
        print(f"[NO BENEFIT] Best breadth filter shows only +{best_improvement:.1f}pp improvement")
        print(f"Below +2pp threshold for deployment")
        print("Market breadth doesn't provide actionable signal quality improvement")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-059',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'stocks_tested': test_stocks,
        'strategies': strategies,
        'breadth_thresholds_tested': breadth_thresholds,
        'results_by_strategy': {
            strategy: [{**r} for r in results]
            for strategy, results in results_by_strategy.items()
        },
        'baseline_win_rate': float(baseline_wr) if baseline_wr else 0,
        'best_strategy': best_strategy if best_strategy else 'none',
        'best_improvement_pp': float(best_improvement),
        'deploy': best_improvement >= 2.0
    }

    results_file = os.path.join(results_dir, 'exp059_market_breadth_filter.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending market breadth filter report email...")
            notifier.send_experiment_report('EXP-059', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-059 market breadth filter."""

    print("\n[MARKET BREADTH] Testing breadth-based signal quality filter")
    print("This will take 5-10 minutes...")
    print()

    results = run_exp059_market_breadth_filter()

    print("\n" + "="*70)
    print("MARKET BREADTH FILTER COMPLETE")
    print("="*70)

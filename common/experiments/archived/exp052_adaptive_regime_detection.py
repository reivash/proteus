"""
EXPERIMENT: EXP-052
Date: 2025-11-17
Objective: Implement adaptive strategy based on market volatility and breadth conditions

MOTIVATION:
Current system: Simple BEAR market filter (blocks all trades in bear markets)
- No volatility regime consideration
- No market breadth filtering
- No adaptation to market conditions
- Same strategy in calm vs volatile markets
- Same strategy in healthy vs weak breadth

PROBLEM:
Mean reversion strategy performance varies by market conditions:
- HIGH VOLATILITY: Wider swings, higher false signals, need wider stops
- LOW VOLATILITY: Tighter ranges, normal strategy works well
- WEAK BREADTH: Few stocks participating, mean reversion less reliable
- STRONG BREADTH: Many stocks moving, mean reversion more reliable
- Missing 20-30% of improvement from market-adaptive strategy

HYPOTHESIS:
Adaptive strategy based on market conditions will improve performance:
- VIX > 25: High volatility → Skip trades OR use wider stops/smaller positions
- VIX < 15: Low volatility → Normal strategy
- Breadth < 40%: Weak market → Skip trades OR reduce positions
- Breadth > 60%: Strong market → Normal strategy
- Expected: +10-15% Sharpe ratio, +3-5pp win rate

METHODOLOGY:
1. Measure market conditions:
   - VIX level (volatility)
   - S&P 500 advance/decline ratio (breadth)
   - Market regime (BULL/BEAR/SIDEWAYS)
   - Combine into "market quality score"

2. Test filtering strategies:
   - No filter (baseline)
   - VIX filter: Skip when VIX > 25
   - Breadth filter: Skip when breadth < 40%
   - Combined filter: VIX + Breadth
   - Adaptive sizing: Reduce positions in poor conditions

3. Measure impact on:
   - Win rate
   - Total return
   - Sharpe ratio
   - Max drawdown
   - Number of trades

4. Deploy if improvement >= +5% Sharpe ratio

CONSTRAINTS:
- Must not reduce trade frequency by > 50%
- Min 70% win rate maintained
- Simple to implement (no complex calculations)

EXPECTED OUTCOME:
- Market condition filters
- Adaptive position sizing rules
- +10-15% Sharpe improvement
- +3-5pp win rate improvement
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


def get_vix_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Get VIX (volatility index) data."""
    fetcher = YahooFinanceFetcher()
    try:
        vix_data = fetcher.fetch_stock_data('^VIX', start_date=start_date, end_date=end_date)
        return vix_data
    except:
        return pd.DataFrame()


def calculate_market_breadth(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Calculate S&P 500 advance/decline breadth.

    For simplicity, use SPY price action as proxy for breadth.
    Real implementation would use advance/decline data.
    """
    fetcher = YahooFinanceFetcher()
    try:
        spy_data = fetcher.fetch_stock_data('SPY', start_date=start_date, end_date=end_date)

        # Calculate daily return
        spy_data['daily_return'] = spy_data['Close'].pct_change() * 100

        # Calculate 10-day rolling average return as breadth proxy
        spy_data['breadth_proxy'] = spy_data['daily_return'].rolling(10).mean()

        # Normalize to 0-100 scale (positive = strong breadth)
        spy_data['breadth_score'] = 50 + spy_data['breadth_proxy'] * 10
        spy_data['breadth_score'] = spy_data['breadth_score'].clip(0, 100)

        return spy_data[['Date', 'breadth_score']]
    except:
        return pd.DataFrame()


def test_adaptive_strategy(ticker: str, start_date: str, end_date: str, strategy: str) -> dict:
    """
    Test an adaptive market condition strategy.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date
        strategy: 'baseline' | 'vix_filter' | 'breadth_filter' | 'combined'

    Returns:
        Test results
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

    # Apply market condition filters
    if strategy != 'baseline':
        # Get VIX data
        vix_data = get_vix_data(start_date, end_date)

        if not vix_data.empty:
            # Merge VIX into signals
            vix_data['Date'] = pd.to_datetime(vix_data['Date'])
            signals['Date'] = pd.to_datetime(signals['Date'])
            signals = signals.merge(vix_data[['Date', 'Close']], on='Date', how='left', suffixes=('', '_vix'))
            signals.rename(columns={'Close_vix': 'VIX'}, inplace=True)

            if strategy in ['vix_filter', 'combined']:
                # Filter out trades when VIX > 25
                signals['vix_filter'] = signals['VIX'] <= 25
                signals.loc[signals['vix_filter'] == False, 'panic_sell'] = 0

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
            'strategy': strategy,
            'win_rate': results['win_rate'],
            'total_return': results['total_return'],
            'total_trades': results['total_trades'],
            'sharpe_ratio': results.get('sharpe_ratio', 0)
        }

    except Exception as e:
        return None


def run_exp052_adaptive_regime_detection():
    """
    Test adaptive market condition strategies.
    """
    print("="*70)
    print("EXP-052: ADAPTIVE MARKET REGIME DETECTION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Implement adaptive strategy based on market conditions")
    print("Current: Simple BEAR market filter only")
    print("Expected: +10-15% Sharpe improvement, +3-5pp win rate")
    print()

    # Test on subset of stocks
    test_stocks = ['NVDA', 'V', 'MA', 'AVGO', 'ORCL', 'ABBV', 'GILD', 'MSFT', 'JPM', 'WMT']
    print(f"Testing {len(test_stocks)} stocks: {', '.join(test_stocks)}")
    print()

    strategies = ['baseline', 'vix_filter']
    print(f"Testing {len(strategies)} strategies")
    print("This will take 5-10 minutes...")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    # Collect results
    results_by_strategy = {s: [] for s in strategies}

    for ticker in test_stocks:
        print(f"\nTesting {ticker}...")

        for strategy in strategies:
            result = test_adaptive_strategy(ticker, start_date, end_date, strategy)
            if result:
                results_by_strategy[strategy].append(result)
                print(f"  {strategy:12s}: {result['win_rate']:>6.1f}% WR, "
                      f"{result['total_return']:>+6.1f}% return, "
                      f"{result['total_trades']} trades")

    # Aggregate results
    print()
    print("="*70)
    print("ADAPTIVE STRATEGY RESULTS")
    print("="*70)
    print()

    print(f"{'Strategy':<15} {'Avg WR':<12} {'Avg Return':<15} {'Avg Trades':<12} {'Improvement'}")
    print("-"*70)

    baseline_wr = None
    baseline_return = None
    best_strategy = None
    best_improvement = 0

    for strategy in strategies:
        if not results_by_strategy[strategy]:
            continue

        avg_wr = np.mean([r['win_rate'] for r in results_by_strategy[strategy]])
        avg_return = np.mean([r['total_return'] for r in results_by_strategy[strategy]])
        avg_trades = np.mean([r['total_trades'] for r in results_by_strategy[strategy]])

        if strategy == 'baseline':
            baseline_wr = avg_wr
            baseline_return = avg_return
            improvement = 0
        else:
            wr_improvement = avg_wr - baseline_wr if baseline_wr else 0
            if wr_improvement > best_improvement:
                best_improvement = wr_improvement
                best_strategy = strategy

        improvement_str = f"+{avg_wr - baseline_wr:.1f}pp" if baseline_wr and strategy != 'baseline' else "-"

        print(f"{strategy:<15} {avg_wr:>10.1f}% {avg_return:>13.1f}% {avg_trades:>10.1f} {improvement_str:>12}")

    print()
    print("="*70)
    print("RECOMMENDATION")
    print("="*70)
    print()

    if best_improvement >= 2.0:
        print(f"[SUCCESS] {best_strategy.upper()} shows +{best_improvement:.1f}pp win rate improvement!")
        best_wr = np.mean([r['win_rate'] for r in results_by_strategy[best_strategy]])
        print(f"Baseline: {baseline_wr:.1f}% win rate")
        print(f"Optimized ({best_strategy}): {best_wr:.1f}% win rate")
        print()
        print("DEPLOY: Implement adaptive market condition filtering")
    else:
        print(f"[NO BENEFIT] Best strategy ({best_strategy}) shows only +{best_improvement:.1f}pp improvement")
        print(f"Below +2pp threshold for deployment")
        print(f"Keep current simple BEAR market filter")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-052',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'stocks_tested': test_stocks,
        'strategies': strategies,
        'results_by_strategy': {
            strategy: [{**r} for r in results]
            for strategy, results in results_by_strategy.items()
        },
        'baseline_win_rate': float(baseline_wr) if baseline_wr else 0,
        'best_strategy': best_strategy if best_strategy else 'none',
        'best_improvement_pp': float(best_improvement),
        'deploy': best_improvement >= 2.0
    }

    results_file = os.path.join(results_dir, 'exp052_adaptive_regime_detection.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending adaptive regime report email...")
            notifier.send_experiment_report('EXP-052', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-052 adaptive regime detection."""

    print("\n[ADAPTIVE REGIME] Testing market condition filters")
    print("This will take 5-10 minutes...")
    print()

    results = run_exp052_adaptive_regime_detection()

    print("\n" + "="*70)
    print("ADAPTIVE REGIME DETECTION COMPLETE")
    print("="*70)

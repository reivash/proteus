"""
EXPERIMENT: EXP-008-VIX
Date: 2025-11-14
Objective: Test VIX filter for extreme volatility protection

HYPOTHESIS:
During extreme fear (VIX > 30), mean reversion breaks down.
VIX filter should prevent trading during market crashes and extreme volatility.

EXPECTED IMPACT:
- Fewer trades (blocked during extreme fear)
- Higher win rate (avoid crisis periods)
- Better risk-adjusted returns
- Protection against tail risk events

HISTORICAL EXTREME VIX PERIODS:
- 2020 COVID Crash: Feb-May 2020 (VIX 30-85)
- 2022 Bear Market: Multiple spikes (VIX 30-35)
- 2018 Correction: Feb 2018 (VIX 30-50)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.fetchers.vix_fetcher import VIXFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from common.config.mean_reversion_params import get_params


def test_vix_periods():
    """Test VIX fetcher and analyze historical periods."""

    print("=" * 70)
    print("VIX ANALYSIS: EXTREME FEAR PERIODS")
    print("=" * 70)
    print()

    fetcher = VIXFetcher(extreme_threshold=30)

    test_periods = {
        'COVID_crash': ('2020-01-01', '2020-06-30', 'COVID-19 Crash'),
        'bear_2022': ('2022-01-01', '2022-12-31', '2022 Bear Market'),
        'bull_2023_2024': ('2023-01-01', '2024-12-31', '2023-2024 Bull Market'),
        'recent': ('2024-01-01', '2025-11-14', '2024-2025 Recent')
    }

    for period_name, (start, end, description) in test_periods.items():
        print(f"{description} ({start} to {end}):")
        print("-" * 70)

        stats = fetcher.analyze_vix_periods(start, end)

        if stats['data_available']:
            print(f"  Total days: {stats['total_days']}")
            print(f"  Extreme fear (VIX > 30): {stats['extreme_days']} days ({stats['extreme_pct']:.1f}%)")
            print(f"  Elevated (VIX 20-30): {stats['elevated_days']} days ({stats['elevated_pct']:.1f}%)")
            print(f"  Normal (VIX < 20): {stats['normal_days']} days ({stats['normal_pct']:.1f}%)")
            print(f"  Average VIX: {stats['avg_vix']:.1f}")
            print(f"  VIX range: {stats['min_vix']:.1f} - {stats['max_vix']:.1f}")

            if stats['extreme_pct'] > 10:
                print(f"  [CRITICAL] {stats['extreme_pct']:.1f}% extreme fear - filter will block many trades")
            elif stats['extreme_pct'] > 0:
                print(f"  [OK] Some extreme fear periods detected")
            else:
                print(f"  [OK] No extreme fear periods (VIX < 30)")
        else:
            print("  [FAIL] VIX data not available")

        print()

    print("=" * 70)


def test_stock_with_vix_filter(ticker, start_date, end_date, use_vix_filter=True):
    """
    Test mean reversion with optional VIX filter.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date
        use_vix_filter: Whether to apply VIX filter

    Returns:
        Results dictionary or None
    """
    # Fetch data
    fetcher = YahooFinanceFetcher()
    try:
        data = fetcher.fetch_stock_data(ticker, start_date=start_date, end_date=end_date)
    except Exception as e:
        return None

    if len(data) < 60:
        return None

    # Engineer features
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)

    # Get stock-specific parameters
    params = get_params(ticker)

    # Detect overcorrections
    detector = MeanReversionDetector(
        z_score_threshold=params['z_score_threshold'],
        rsi_oversold=params['rsi_oversold'],
        rsi_overbought=65,
        volume_multiplier=params['volume_multiplier'],
        price_drop_threshold=params['price_drop_threshold']
    )

    signals = detector.detect_overcorrections(enriched_data)
    signals = detector.calculate_reversion_targets(signals)

    # Count signals before filtering
    signals_before = signals['panic_sell'].sum()

    # Apply regime filter (always)
    regime_detector = MarketRegimeDetector()
    signals = add_regime_filter_to_signals(signals, regime_detector)

    signals_after_regime = signals['panic_sell'].sum()

    # Apply earnings filter (always)
    earnings_fetcher = EarningsCalendarFetcher(
        exclusion_days_before=3,
        exclusion_days_after=3
    )
    signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

    signals_after_earnings = signals['panic_sell'].sum()

    # Apply VIX filter if requested
    if use_vix_filter:
        vix_fetcher = VIXFetcher(extreme_threshold=30)
        signals = vix_fetcher.add_vix_filter_to_signals(signals, 'panic_sell')

    signals_after_vix = signals['panic_sell'].sum()

    # Backtest
    backtester = MeanReversionBacktester(
        initial_capital=10000,
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=2
    )

    results = backtester.backtest(signals)

    if results['total_trades'] == 0:
        return None

    return {
        'ticker': ticker,
        'signals_original': int(signals_before),
        'signals_after_regime': int(signals_after_regime),
        'signals_after_earnings': int(signals_after_earnings),
        'signals_after_vix': int(signals_after_vix),
        'signals_blocked_by_vix': int(signals_after_earnings - signals_after_vix),
        'trades': results['total_trades'],
        'win_rate': results['win_rate'],
        'return': results['total_return'],
        'sharpe': results['sharpe_ratio'],
        'winning_trades': results['winning_trades'],
        'losing_trades': results['losing_trades']
    }


def compare_with_and_without_vix_filter():
    """
    Compare performance with and without VIX filter.
    """
    print("\n" + "=" * 70)
    print("COMPARING: WITH vs WITHOUT VIX FILTER")
    print("=" * 70)
    print()

    # Test on full period (includes COVID and 2022 bear market)
    test_period = ('2020-01-01', '2025-11-14')

    # Test representative stocks from each sector
    test_stocks = [
        ("NVDA", "NVIDIA"),
        ("TSLA", "Tesla"),
        ("JPM", "JPMorgan"),
        ("JNJ", "Johnson & Johnson"),
    ]

    print("Testing on 2020-2025 period (includes COVID crash + 2022 bear market)")
    print(f"Filters: Regime + Earnings + VIX")
    print()

    results_summary = []

    for ticker, name in test_stocks:
        print(f"\n{name} ({ticker})")
        print("-" * 70)

        # WITHOUT VIX filter
        print("  WITHOUT VIX filter:")
        result_without = test_stock_with_vix_filter(ticker, *test_period, use_vix_filter=False)

        if result_without:
            print(f"    Signals: {result_without['signals_after_earnings']}")
            print(f"    Trades: {result_without['trades']}")
            print(f"    Win rate: {result_without['win_rate']:.1f}%")
            print(f"    Return: {result_without['return']:+.2f}%")

        # WITH VIX filter
        print("  WITH VIX filter:")
        result_with = test_stock_with_vix_filter(ticker, *test_period, use_vix_filter=True)

        if result_with:
            print(f"    Signals: {result_with['signals_after_vix']}")
            print(f"    Signals blocked (VIX): {result_with['signals_blocked_by_vix']}")
            print(f"    Trades: {result_with['trades']}")
            print(f"    Win rate: {result_with['win_rate']:.1f}%")
            print(f"    Return: {result_with['return']:+.2f}%")

        # Calculate impact
        if result_without and result_with:
            win_rate_change = result_with['win_rate'] - result_without['win_rate']
            return_change = result_with['return'] - result_without['return']

            print(f"\n  IMPACT:")
            print(f"    Win rate change: {win_rate_change:+.1f}%")
            print(f"    Return change: {return_change:+.2f}%")

            if win_rate_change > 3 and return_change > 0:
                print(f"    [SUCCESS] VIX filter significantly improves performance!")
            elif win_rate_change > 0:
                print(f"    [GOOD] VIX filter improves win rate")
            elif abs(return_change) < 2:
                print(f"    [NEUTRAL] Minimal impact (good - no extreme periods)")
            else:
                print(f"    [MIXED] Trade-off between trades and quality")

            results_summary.append({
                'ticker': ticker,
                'name': name,
                'signals_blocked': result_with['signals_blocked_by_vix'],
                'win_rate_without': result_without['win_rate'],
                'win_rate_with': result_with['win_rate'],
                'win_rate_change': win_rate_change,
                'return_without': result_without['return'],
                'return_with': result_with['return'],
                'return_change': return_change
            })

    # Summary
    if results_summary:
        print("\n" + "=" * 70)
        print("SUMMARY: VIX FILTER IMPACT")
        print("=" * 70)
        print(f"{'Stock':<12} {'Blocked':>8} {'Win Rate':>18} {'Return':>18}")
        print(f"{'':12} {'Signals':>8} {'Without':>8} {'With':>8} {'Without':>8} {'With':>8}")
        print("-" * 70)

        for r in results_summary:
            print(f"{r['ticker']:<12} {r['signals_blocked']:>8} "
                  f"{r['win_rate_without']:>7.1f}% {r['win_rate_with']:>7.1f}% "
                  f"{r['return_without']:>+7.2f}% {r['return_with']:>+7.2f}%")

        avg_blocked = sum(r['signals_blocked'] for r in results_summary) / len(results_summary)
        avg_win_rate_change = sum(r['win_rate_change'] for r in results_summary) / len(results_summary)
        avg_return_change = sum(r['return_change'] for r in results_summary) / len(results_summary)

        print("-" * 70)
        print(f"{'AVERAGE':<12} {avg_blocked:>8.1f} "
              f"{avg_win_rate_change:>16.1f}% {avg_return_change:>18.2f}%")
        print("=" * 70)

        print("\nCONCLUSION:")
        if avg_win_rate_change > 3:
            print("[SUCCESS] VIX filter significantly improves performance!")
            print(f"Average win rate improvement: {avg_win_rate_change:+.1f}%")
            print("Recommendation: ALWAYS use VIX filter")
        elif avg_win_rate_change > 1:
            print("[GOOD] VIX filter improves win rate with minimal trade-off")
            print("Recommendation: Use VIX filter for risk management")
        elif abs(avg_blocked) < 1:
            print("[NEUTRAL] Few signals blocked (no extreme periods in test range)")
            print("VIX filter provides insurance for future crises")
        else:
            print("[MIXED] VIX filter blocks signals but impact unclear")
            print("Consider testing on longer period with more crises")

        print()
        print("NOTE: 2020-2025 period includes:")
        print("  - COVID crash (Feb-May 2020, VIX 30-85)")
        print("  - 2022 bear market (VIX 30-35 spikes)")
        print("  - VIX filter protects against future tail risk events")

    print("=" * 70)


if __name__ == "__main__":
    # First analyze VIX periods
    test_vix_periods()

    print("\n\n")

    # Then compare with/without filter
    compare_with_and_without_vix_filter()

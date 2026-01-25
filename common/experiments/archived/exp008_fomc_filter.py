"""
EXPERIMENT: EXP-008-FOMC
Date: 2025-11-14
Objective: Test FOMC meeting filter for policy-driven volatility protection

HYPOTHESIS:
During FOMC meetings, market moves are driven by Fed policy decisions.
FOMC filter should prevent trading during policy-driven volatility.

EXPECTED IMPACT:
- Fewer trades (blocked during Fed meetings)
- Higher win rate (avoid policy-driven moves)
- Better risk-adjusted returns
- Protection against Fed-driven volatility

FOMC MEETING SCHEDULE:
- 8 meetings per year (every ~6 weeks)
- Scheduled in advance
- Major market-moving events (rate decisions, QE/QT)

TEST METHODOLOGY:
- Compare WITH vs WITHOUT FOMC filter
- Test period: 2020-2025 (includes major Fed policy shifts)
- Exclusion window: 1 day before + 1 day after meeting
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.fetchers.fomc_calendar import FOMCCalendarFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from common.config.mean_reversion_params import get_params


def test_fomc_periods():
    """Test FOMC calendar and analyze meeting frequency."""

    print("=" * 70)
    print("FOMC MEETING ANALYSIS")
    print("=" * 70)
    print()

    fetcher = FOMCCalendarFetcher(exclusion_days_before=1, exclusion_days_after=1)

    test_periods = {
        'covid_2020': ('2020-01-01', '2020-12-31', '2020 (COVID + Emergency Cuts)'),
        'tightening_2022': ('2022-01-01', '2022-12-31', '2022 (Aggressive Tightening)'),
        'plateau_2023': ('2023-01-01', '2023-12-31', '2023 (Rates Plateau)'),
        'recent_2024': ('2024-01-01', '2025-11-14', '2024-2025 (Recent)'),
    }

    for period_name, (start, end, description) in test_periods.items():
        print(f"{description}:")
        print("-" * 70)

        stats = fetcher.analyze_fomc_periods(start, end)

        print(f"  FOMC meetings: {stats['fomc_meetings']}")
        print(f"  Total days: {stats['total_days']}")
        print(f"  Exclusion days: {stats['exclusion_days']} ({stats['exclusion_pct']:.1f}% of period)")
        print(f"  Avg days between meetings: {stats['avg_days_between_meetings']:.0f}")
        print(f"  Exclusion window: {stats['exclusion_window']}")

        if stats['exclusion_pct'] > 10:
            print(f"  [NOTE] {stats['exclusion_pct']:.1f}% of trading days excluded")
        else:
            print(f"  [OK] Moderate exclusion ({stats['exclusion_pct']:.1f}%)")

        print()

    print("=" * 70)


def test_stock_with_fomc_filter(ticker, start_date, end_date, use_fomc_filter=True):
    """
    Test mean reversion with optional FOMC filter.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date
        use_fomc_filter: Whether to apply FOMC filter

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

    # Apply FOMC filter if requested
    if use_fomc_filter:
        fomc_fetcher = FOMCCalendarFetcher(
            exclusion_days_before=1,
            exclusion_days_after=1
        )
        signals = fomc_fetcher.add_fomc_filter_to_signals(signals, 'panic_sell')

    signals_after_fomc = signals['panic_sell'].sum()

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
        'signals_after_fomc': int(signals_after_fomc),
        'signals_blocked_by_fomc': int(signals_after_earnings - signals_after_fomc),
        'trades': results['total_trades'],
        'win_rate': results['win_rate'],
        'return': results['total_return'],
        'sharpe': results['sharpe_ratio'],
        'winning_trades': results['winning_trades'],
        'losing_trades': results['losing_trades']
    }


def compare_with_and_without_fomc_filter():
    """
    Compare performance with and without FOMC filter.
    """
    print("\n" + "=" * 70)
    print("COMPARING: WITH vs WITHOUT FOMC FILTER")
    print("=" * 70)
    print()

    # Test on full period (includes Fed policy shifts: emergency cuts, aggressive tightening)
    test_period = ('2020-01-01', '2025-11-14')

    # Test representative stocks from each sector
    test_stocks = [
        ("NVDA", "NVIDIA"),
        ("TSLA", "Tesla"),
        ("JPM", "JPMorgan"),
        ("JNJ", "Johnson & Johnson"),
    ]

    print("Testing on 2020-2025 period (includes Fed emergency cuts + aggressive tightening)")
    print(f"Filters: Regime + Earnings + FOMC")
    print(f"FOMC exclusion window: 1 day before + 1 day after meeting")
    print()

    results_summary = []

    for ticker, name in test_stocks:
        print(f"\n{name} ({ticker})")
        print("-" * 70)

        # WITHOUT FOMC filter
        print("  WITHOUT FOMC filter:")
        result_without = test_stock_with_fomc_filter(ticker, *test_period, use_fomc_filter=False)

        if result_without:
            print(f"    Signals: {result_without['signals_after_earnings']}")
            print(f"    Trades: {result_without['trades']}")
            print(f"    Win rate: {result_without['win_rate']:.1f}%")
            print(f"    Return: {result_without['return']:+.2f}%")

        # WITH FOMC filter
        print("  WITH FOMC filter:")
        result_with = test_stock_with_fomc_filter(ticker, *test_period, use_fomc_filter=True)

        if result_with:
            print(f"    Signals: {result_with['signals_after_fomc']}")
            print(f"    Signals blocked (FOMC): {result_with['signals_blocked_by_fomc']}")
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
                print(f"    [SUCCESS] FOMC filter significantly improves performance!")
            elif win_rate_change > 0 and return_change > 0:
                print(f"    [GOOD] FOMC filter improves both win rate and return")
            elif win_rate_change > 0:
                print(f"    [MIXED] Higher win rate but lower return")
            elif abs(return_change) < 2:
                print(f"    [NEUTRAL] Minimal impact")
            else:
                print(f"    [NEGATIVE] FOMC filter reduces performance")

            results_summary.append({
                'ticker': ticker,
                'name': name,
                'signals_blocked': result_with['signals_blocked_by_fomc'],
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
        print("SUMMARY: FOMC FILTER IMPACT")
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
        if avg_win_rate_change > 3 and avg_return_change > 0:
            print("[SUCCESS] FOMC filter significantly improves performance!")
            print(f"Average win rate improvement: {avg_win_rate_change:+.1f}%")
            print(f"Average return improvement: {avg_return_change:+.2f}%")
            print("Recommendation: ALWAYS use FOMC filter")
        elif avg_win_rate_change > 1 and avg_return_change > 0:
            print("[GOOD] FOMC filter improves both win rate and return")
            print("Recommendation: Use FOMC filter for better performance")
        elif avg_win_rate_change > 0:
            print("[MIXED] Higher win rate but trade-off on returns")
            print("Consider using for risk-adjusted improvement")
        elif abs(avg_return_change) < 1:
            print("[NEUTRAL] Minimal impact (few signals near FOMC meetings)")
            print("FOMC filter provides insurance but limited benefit in test period")
        else:
            print("[NEGATIVE] FOMC filter reduces performance")
            print("Recommendation: DO NOT use FOMC filter")

        print()
        print(f"NOTE: Test period includes major Fed policy events:")
        print(f"  - 2020: Emergency rate cuts (COVID)")
        print(f"  - 2022: Aggressive tightening (40-year high inflation)")
        print(f"  - 2023-2025: Rate plateau and potential cuts")

    print("=" * 70)


if __name__ == "__main__":
    # First analyze FOMC meeting frequency
    test_fomc_periods()

    print("\n\n")

    # Then compare with/without filter
    compare_with_and_without_fomc_filter()

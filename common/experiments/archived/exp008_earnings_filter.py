"""
EXPERIMENT: EXP-008-EARNINGS
Date: 2025-11-14
Objective: Test mean reversion with earnings date filter

PROBLEM:
Panic sells near earnings are often justified by poor results/guidance.
These are NOT technical overcorrections, but fundamental events.

SOLUTION:
Exclude trades within 3 days before and after earnings announcements.

This should reduce false signals and improve win rate.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from datetime import datetime

def test_earnings_calendar():
    """Test earnings calendar fetcher."""

    print("=" * 70)
    print("TESTING EARNINGS CALENDAR FETCHER")
    print("=" * 70)
    print()

    fetcher = EarningsCalendarFetcher()

    test_stocks = [
        ("NVDA", "NVIDIA"),
        ("TSLA", "Tesla"),
        ("AAPL", "Apple")
    ]

    for ticker, name in test_stocks:
        print(f"{name} ({ticker}):")
        print("-" * 70)

        stats = fetcher.get_earnings_stats(ticker, start_date='2022-01-01', end_date='2025-11-13')

        if stats['data_available']:
            print(f"  Total earnings reports: {stats['total_earnings']}")
            print(f"  Earnings per year: {stats['earnings_per_year']:.1f}")
            print(f"  Total exclusion days: {stats['total_exclusion_days']}")
            print(f"  Exclusion window: {stats['exclusion_days_before']} days before + {stats['exclusion_days_after']} days after")
            print(f"  [OK] Earnings data available")
        else:
            print(f"  [WARN] No earnings data available")

        print()

    print("=" * 70)


def run_mean_reversion_with_earnings_filter(ticker, ticker_name, start_date, end_date, use_earnings_filter=True):
    """
    Run mean reversion with earnings filter.

    Args:
        ticker: Stock ticker
        ticker_name: Stock name
        start_date: Start date
        end_date: End date
        use_earnings_filter: Whether to apply earnings filter

    Returns:
        Results dictionary
    """
    # Fetch data
    fetcher = YahooFinanceFetcher()
    try:
        data = fetcher.fetch_stock_data(ticker, start_date=start_date, end_date=end_date)
    except Exception as e:
        print(f"  [FAIL] Could not fetch data: {e}")
        return None

    if len(data) < 60:
        print(f"  [FAIL] Insufficient data ({len(data)} rows)")
        return None

    # Engineer features
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)

    # Detect overcorrections
    mean_rev_detector = MeanReversionDetector(
        z_score_threshold=1.5,
        rsi_oversold=35,
        rsi_overbought=65,
        volume_multiplier=1.5,
        price_drop_threshold=-2.0
    )

    signals = mean_rev_detector.detect_overcorrections(enriched_data)
    signals = mean_rev_detector.calculate_reversion_targets(signals)

    # Count signals before filtering
    signals_before = signals['panic_sell'].sum()

    # Apply regime filter (always)
    regime_detector = MarketRegimeDetector()
    signals = add_regime_filter_to_signals(signals, regime_detector)

    signals_after_regime = signals['panic_sell'].sum()

    # Apply earnings filter if requested
    if use_earnings_filter:
        earnings_fetcher = EarningsCalendarFetcher(
            exclusion_days_before=3,
            exclusion_days_after=3
        )
        signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

    signals_after_earnings = signals['panic_sell'].sum()

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
        'ticker_name': ticker_name,
        'signals_original': int(signals_before),
        'signals_after_regime': int(signals_after_regime),
        'signals_after_earnings': int(signals_after_earnings),
        'signals_blocked_by_earnings': int(signals_after_regime - signals_after_earnings),
        'trades': results['total_trades'],
        'win_rate': results['win_rate'],
        'return': results['total_return'],
        'sharpe': results['sharpe_ratio'],
        'winning_trades': results['winning_trades'],
        'losing_trades': results['losing_trades']
    }


def compare_with_and_without_earnings_filter():
    """Compare performance with and without earnings filter."""

    print("\n" + "=" * 70)
    print("COMPARING: WITH vs WITHOUT EARNINGS FILTER")
    print("=" * 70)
    print()

    test_stocks = [
        ("NVDA", "NVIDIA"),
        ("TSLA", "Tesla"),
        ("AAPL", "Apple")
    ]

    # Test on full period (2022-2025)
    test_period = ('2022-01-01', '2025-11-13')

    print("Testing on full period: 2022-2025")
    print(f"Filters applied: Regime filter + Earnings filter")
    print()

    results_summary = []

    for ticker, name in test_stocks:
        print(f"\n{name} ({ticker})")
        print("-" * 70)

        # WITHOUT earnings filter (but with regime filter)
        print("  WITHOUT earnings filter:")
        result_without = run_mean_reversion_with_earnings_filter(
            ticker, name, *test_period, use_earnings_filter=False
        )

        if result_without:
            print(f"    Signals: {result_without['signals_after_regime']}")
            print(f"    Trades: {result_without['trades']}")
            print(f"    Win rate: {result_without['win_rate']:.1f}%")
            print(f"    Return: {result_without['return']:+.2f}%")

        # WITH earnings filter (and regime filter)
        print("  WITH earnings filter:")
        result_with = run_mean_reversion_with_earnings_filter(
            ticker, name, *test_period, use_earnings_filter=True
        )

        if result_with:
            print(f"    Signals: {result_with['signals_after_earnings']}")
            print(f"    Signals blocked (earnings): {result_with['signals_blocked_by_earnings']}")
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

            if win_rate_change > 0 and return_change > 0:
                print(f"    [OK] Earnings filter IMPROVED performance")
            elif win_rate_change > 0:
                print(f"    [MIXED] Higher win rate but lower return (fewer trades)")
            else:
                print(f"    [NEUTRAL] No significant improvement")

            results_summary.append({
                'ticker': ticker,
                'name': name,
                'signals_blocked': result_with['signals_blocked_by_earnings'],
                'win_rate_without': result_without['win_rate'],
                'win_rate_with': result_with['win_rate'],
                'win_rate_change': win_rate_change,
                'return_without': result_without['return'],
                'return_with': result_with['return'],
                'return_change': return_change
            })

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: EARNINGS FILTER IMPACT")
    print("=" * 70)
    print(f"{'Stock':<12} {'Blocked':>8} {'Win Rate':>18} {'Return':>18}")
    print(f"{'':12} {'Signals':>8} {'Without':>8} {'With':>8} {'Without':>8} {'With':>8}")
    print("-" * 70)

    for r in results_summary:
        print(f"{r['ticker']:<12} {r['signals_blocked']:>8} "
              f"{r['win_rate_without']:>7.1f}% {r['win_rate_with']:>7.1f}% "
              f"{r['return_without']:>+7.2f}% {r['return_with']:>+7.2f}%")

    # Overall assessment
    avg_win_rate_change = sum(r['win_rate_change'] for r in results_summary) / len(results_summary)
    avg_return_change = sum(r['return_change'] for r in results_summary) / len(results_summary)
    total_blocked = sum(r['signals_blocked'] for r in results_summary)

    print("-" * 70)
    print(f"{'AVERAGE':<12} {total_blocked/len(results_summary):>8.1f} "
          f"{avg_win_rate_change:>16.1f}% {avg_return_change:>18.2f}%")
    print("=" * 70)

    print("\nCONCLUSION:")
    if avg_win_rate_change > 2 and avg_return_change > 0:
        print("[SUCCESS] Earnings filter significantly improves performance!")
        print("Recommendation: ALWAYS use earnings filter with mean reversion.")
    elif avg_win_rate_change > 0:
        print("[GOOD] Earnings filter improves win rate with acceptable return trade-off.")
        print("Recommendation: Use earnings filter for better risk-adjusted returns.")
    elif avg_win_rate_change < -2:
        print("[MIXED] Earnings filter blocks too many good signals.")
        print("Consider reducing exclusion window (2 days instead of 3).")
    else:
        print("[NEUTRAL] Earnings filter has minimal impact.")
        print("May still be worth using for risk reduction.")

    print("=" * 70)


if __name__ == "__main__":
    # First test earnings calendar
    test_earnings_calendar()

    print("\n\n")

    # Then compare with and without filter
    compare_with_and_without_earnings_filter()

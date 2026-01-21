"""
EXPERIMENT: EXP-008-PARAMS-COMPARE
Date: 2025-11-14
Objective: Compare universal vs optimized parameters

Show the performance improvement from using stock-specific parameters.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester


def test_with_parameters(ticker, ticker_name, start_date, end_date, params):
    """Test strategy with specific parameters."""

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

    # Apply filters
    regime_detector = MarketRegimeDetector()
    signals = add_regime_filter_to_signals(signals, regime_detector)

    earnings_fetcher = EarningsCalendarFetcher(
        exclusion_days_before=3,
        exclusion_days_after=3
    )
    signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

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
        'trades': results['total_trades'],
        'win_rate': results['win_rate'],
        'return': results['total_return'],
        'sharpe': results['sharpe_ratio'],
        'max_drawdown': results['max_drawdown']
    }


def compare_universal_vs_optimized():
    """
    Direct comparison: Universal vs Optimized parameters.
    """
    print("=" * 70)
    print("PERFORMANCE COMPARISON: UNIVERSAL vs OPTIMIZED PARAMETERS")
    print("=" * 70)
    print()

    test_period = ('2022-01-01', '2025-11-14')

    # Universal parameters (baseline)
    universal = {
        'z_score_threshold': 1.5,
        'rsi_oversold': 35,
        'volume_multiplier': 1.5,
        'price_drop_threshold': -2.0
    }

    # Optimized parameters (from optimization run)
    optimized = {
        'NVDA': {
            'z_score_threshold': 1.5,
            'rsi_oversold': 35,
            'volume_multiplier': 1.3,
            'price_drop_threshold': -1.5
        },
        'TSLA': {
            'z_score_threshold': 1.75,
            'rsi_oversold': 32,
            'volume_multiplier': 1.3,
            'price_drop_threshold': -1.5
        },
        'AAPL': {
            'z_score_threshold': 1.0,
            'rsi_oversold': 35,
            'volume_multiplier': 1.7,
            'price_drop_threshold': -1.5
        }
    }

    test_stocks = [
        ("NVDA", "NVIDIA"),
        ("TSLA", "Tesla"),
        ("AAPL", "Apple")
    ]

    comparisons = []

    for ticker, name in test_stocks:
        print(f"\n{name} ({ticker})")
        print("-" * 70)

        # Test with universal parameters
        print("  Testing with UNIVERSAL parameters...")
        universal_result = test_with_parameters(ticker, name, *test_period, universal)

        # Test with optimized parameters
        print("  Testing with OPTIMIZED parameters...")
        optimized_result = test_with_parameters(ticker, name, *test_period, optimized[ticker])

        if universal_result and optimized_result:
            print(f"    UNIVERSAL: {universal_result['trades']} trades, {universal_result['win_rate']:.1f}% win rate, {universal_result['return']:+.2f}% return")
            print(f"    OPTIMIZED: {optimized_result['trades']} trades, {optimized_result['win_rate']:.1f}% win rate, {optimized_result['return']:+.2f}% return")

            win_rate_change = optimized_result['win_rate'] - universal_result['win_rate']
            return_change = optimized_result['return'] - universal_result['return']
            sharpe_change = optimized_result['sharpe'] - universal_result['sharpe']

            print(f"    IMPROVEMENT: {win_rate_change:+.1f}% win rate, {return_change:+.2f}% return, {sharpe_change:+.2f} Sharpe")

            if win_rate_change > 2 or return_change > 5:
                print("    [SUCCESS] Significant improvement!")
            elif win_rate_change > 0:
                print("    [GOOD] Modest improvement")
            else:
                print("    [NEUTRAL] No improvement")

            comparisons.append({
                'ticker': ticker,
                'name': name,
                'universal': universal_result,
                'optimized': optimized_result,
                'win_rate_change': win_rate_change,
                'return_change': return_change,
                'sharpe_change': sharpe_change
            })

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: UNIVERSAL vs OPTIMIZED")
    print("=" * 70)
    print(f"{'Stock':<12} {'Win Rate':<24} {'Return':<24} {'Sharpe':<18}")
    print(f"{'':12} {'Univ':>10} {'Opt':>10} {'Univ':>10} {'Opt':>10} {'Univ':>8} {'Opt':>8}")
    print("-" * 70)

    for comp in comparisons:
        u = comp['universal']
        o = comp['optimized']

        print(f"{comp['ticker']:<12} "
              f"{u['win_rate']:>9.1f}% {o['win_rate']:>9.1f}% "
              f"{u['return']:>+9.2f}% {o['return']:>+9.2f}% "
              f"{u['sharpe']:>8.2f} {o['sharpe']:>8.2f}")

    # Calculate averages
    avg_win_rate_change = sum(c['win_rate_change'] for c in comparisons) / len(comparisons)
    avg_return_change = sum(c['return_change'] for c in comparisons) / len(comparisons)
    avg_sharpe_change = sum(c['sharpe_change'] for c in comparisons) / len(comparisons)

    print("-" * 70)
    print(f"{'IMPROVEMENT':<12} {avg_win_rate_change:>19.1f}% {avg_return_change:>19.2f}% {avg_sharpe_change:>16.2f}")
    print("=" * 70)

    print("\nCONCLUSION:")
    if avg_win_rate_change > 3:
        print("[SUCCESS] Stock-specific parameters significantly improve performance!")
        print(f"Average win rate improvement: {avg_win_rate_change:+.1f}%")
        print("Recommendation: USE stock-specific parameters in production.")
    elif avg_win_rate_change > 1:
        print("[GOOD] Stock-specific parameters provide modest improvement.")
        print(f"Average win rate improvement: {avg_win_rate_change:+.1f}%")
        print("Recommendation: Consider stock-specific parameters for key stocks.")
    else:
        print("[NEUTRAL] Stock-specific parameters have minimal impact.")
        print("Universal parameters are sufficient.")

    print()
    print("KEY INSIGHTS:")

    # Find stocks with biggest improvements
    best_improvement = max(comparisons, key=lambda x: x['win_rate_change'])
    print(f"  - Biggest winner: {best_improvement['name']} (+{best_improvement['win_rate_change']:.1f}% win rate)")

    # Show parameter differences
    print()
    print("OPTIMAL PARAMETERS BY STOCK:")
    for ticker in ['NVDA', 'TSLA', 'AAPL']:
        params = optimized[ticker]
        print(f"  {ticker}: z-score={params['z_score_threshold']:.2f}, RSI={params['rsi_oversold']}, "
              f"volume={params['volume_multiplier']:.1f}x, drop={params['price_drop_threshold']:.1f}%")

    print()
    print("VARIANCE IN OPTIMAL PARAMETERS:")
    z_scores = [optimized[t]['z_score_threshold'] for t in ['NVDA', 'TSLA', 'AAPL']]
    rsi_values = [optimized[t]['rsi_oversold'] for t in ['NVDA', 'TSLA', 'AAPL']]

    print(f"  Z-score range: {min(z_scores):.2f} - {max(z_scores):.2f} (diff={max(z_scores)-min(z_scores):.2f})")
    print(f"  RSI range: {min(rsi_values)} - {max(rsi_values)} (diff={max(rsi_values)-min(rsi_values)})")

    if max(z_scores) - min(z_scores) > 0.5:
        print("  → Z-score varies significantly! Stock-specific tuning is important.")

    print("=" * 70)


if __name__ == "__main__":
    compare_universal_vs_optimized()

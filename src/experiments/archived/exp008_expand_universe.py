"""
EXPERIMENT: EXP-008-EXPAND
Date: 2025-11-14
Objective: Expand stock universe with optimized parameters

APPROACH:
Apply parameter optimization framework to additional stocks:
- MSFT: Microsoft (mega-cap tech, moderate volatility)
- GOOGL: Google (mega-cap tech, moderate volatility)
- META: Meta/Facebook (large-cap tech, higher volatility)
- AMZN: Amazon (mega-cap tech, moderate volatility)

Determine which stocks are suitable for mean reversion strategy.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester


def optimize_parameters_for_stock(ticker, ticker_name, start_date, end_date):
    """
    Find optimal parameters for a specific stock.
    Simplified version - tests key parameter combinations.
    """
    print(f"\n{'=' * 70}")
    print(f"OPTIMIZING: {ticker_name} ({ticker})")
    print(f"{'=' * 70}")

    # Fetch data
    fetcher = YahooFinanceFetcher()
    try:
        data = fetcher.fetch_stock_data(ticker, start_date=start_date, end_date=end_date)
    except Exception as e:
        print(f"[FAIL] Could not fetch data: {e}")
        return None

    if len(data) < 60:
        print(f"[FAIL] Insufficient data ({len(data)} rows)")
        return None

    # Engineer features
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)

    # Parameter grid (focused on key parameters)
    param_grid = {
        'z_score_threshold': [1.0, 1.25, 1.5, 1.75, 2.0],
        'rsi_oversold': [30, 32, 35, 38, 40],
        'volume_multiplier': [1.3, 1.5, 1.7, 2.0],
        'price_drop_threshold': [-1.5, -2.0, -2.5, -3.0]
    }

    print(f"Testing 400 parameter combinations...")

    results = []

    for z_threshold in param_grid['z_score_threshold']:
        for rsi_oversold in param_grid['rsi_oversold']:
            for volume_mult in param_grid['volume_multiplier']:
                for price_drop in param_grid['price_drop_threshold']:

                    detector = MeanReversionDetector(
                        z_score_threshold=z_threshold,
                        rsi_oversold=rsi_oversold,
                        rsi_overbought=65,
                        volume_multiplier=volume_mult,
                        price_drop_threshold=price_drop
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

                    if signals['panic_sell'].sum() == 0:
                        continue

                    backtester = MeanReversionBacktester(
                        initial_capital=10000,
                        profit_target=2.0,
                        stop_loss=-2.0,
                        max_hold_days=2
                    )

                    backtest_results = backtester.backtest(signals)

                    if backtest_results['total_trades'] == 0:
                        continue

                    results.append({
                        'z_score': z_threshold,
                        'rsi_oversold': rsi_oversold,
                        'volume_mult': volume_mult,
                        'price_drop': price_drop,
                        'signals': int(signals['panic_sell'].sum()),
                        'trades': backtest_results['total_trades'],
                        'win_rate': backtest_results['win_rate'],
                        'return': backtest_results['total_return'],
                        'sharpe': backtest_results['sharpe_ratio'],
                        'max_drawdown': backtest_results['max_drawdown']
                    })

    if len(results) == 0:
        print("[FAIL] No valid parameter combinations found")
        return None

    # Sort by win rate and Sharpe
    valid_results = [r for r in results if r['win_rate'] >= 60 and r['trades'] >= 5]

    if len(valid_results) == 0:
        print("[WARN] No combinations with 60%+ win rate and 5+ trades")
        print("This stock may not be suitable for mean reversion strategy.")
        # Show best available
        best_available = sorted(results, key=lambda x: (x['win_rate'], x['sharpe']), reverse=True)[:5]
        print(f"\nBEST AVAILABLE (out of {len(results)} combinations):")
        print("-" * 70)
        for i, r in enumerate(best_available, 1):
            print(f"{i}. z={r['z_score']:.2f}, RSI={r['rsi_oversold']}, vol={r['volume_mult']:.1f}x, drop={r['price_drop']:.1f}% "
                  f"-> {r['trades']} trades, {r['win_rate']:.1f}% win, {r['return']:+.2f}% return")
        return None

    valid_results = sorted(valid_results, key=lambda x: (x['win_rate'], x['sharpe']), reverse=True)
    top_5 = valid_results[:5]

    print(f"\nTOP 5 PARAMETERS (out of {len(results)} valid):")
    print("-" * 70)
    print(f"{'Rank':<6} {'Z-score':<10} {'RSI':<8} {'Volume':<10} {'Drop':<10} {'Trades':<8} {'Win%':<8} {'Return':<10} {'Sharpe':<8}")
    print("-" * 70)

    for i, result in enumerate(top_5, 1):
        print(f"{i:<6} {result['z_score']:<10.2f} {result['rsi_oversold']:<8} "
              f"{result['volume_mult']:<10.1f} {result['price_drop']:<10.1f} "
              f"{result['trades']:<8} {result['win_rate']:<7.1f}% "
              f"{result['return']:>+8.2f}% {result['sharpe']:<8.2f}")

    best = top_5[0]

    print(f"\nBEST PARAMETERS:")
    print(f"  Z-score: {best['z_score']}")
    print(f"  RSI: {best['rsi_oversold']}")
    print(f"  Volume: {best['volume_mult']}x")
    print(f"  Drop: {best['price_drop']}%")
    print()
    print(f"PERFORMANCE:")
    print(f"  Trades: {best['trades']}")
    print(f"  Win rate: {best['win_rate']:.1f}%")
    print(f"  Return: {best['return']:+.2f}%")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  Max drawdown: {best['max_drawdown']:.2f}%")

    # Assess suitability
    if best['win_rate'] >= 75 and best['sharpe'] > 3:
        print(f"  [EXCELLENT] Strong candidate for mean reversion!")
    elif best['win_rate'] >= 65 and best['sharpe'] > 1:
        print(f"  [GOOD] Suitable for mean reversion")
    elif best['win_rate'] >= 60:
        print(f"  [ACCEPTABLE] Marginal performance, monitor closely")
    else:
        print(f"  [POOR] Consider excluding from strategy")

    return {
        'ticker': ticker,
        'ticker_name': ticker_name,
        'best_params': {
            'z_score_threshold': best['z_score'],
            'rsi_oversold': best['rsi_oversold'],
            'volume_multiplier': best['volume_mult'],
            'price_drop_threshold': best['price_drop']
        },
        'performance': {
            'signals': best['signals'],
            'trades': best['trades'],
            'win_rate': best['win_rate'],
            'return': best['return'],
            'sharpe': best['sharpe'],
            'max_drawdown': best['max_drawdown']
        },
        'suitable': best['win_rate'] >= 60
    }


def expand_stock_universe():
    """
    Expand stock universe with new optimized stocks.
    """
    print("=" * 70)
    print("EXPANDING STOCK UNIVERSE")
    print("=" * 70)
    print()

    test_period = ('2022-01-01', '2025-11-14')

    # Existing stocks (for reference)
    existing_stocks = [
        ("NVDA", "NVIDIA", "87.5% win, +45.75% return"),
        ("TSLA", "Tesla", "87.5% win, +12.81% return"),
        ("AAPL", "Apple", "60.0% win, -0.25% return")
    ]

    # New stocks to test
    new_stocks = [
        ("MSFT", "Microsoft"),
        ("GOOGL", "Google"),
        ("META", "Meta"),
        ("AMZN", "Amazon")
    ]

    print("EXISTING UNIVERSE (optimized):")
    for ticker, name, perf in existing_stocks:
        print(f"  {ticker:<6} {name:<12} - {perf}")
    print()

    print("TESTING NEW STOCKS:")
    print()

    optimization_results = []

    for ticker, name in new_stocks:
        result = optimize_parameters_for_stock(ticker, name, *test_period)
        if result:
            optimization_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("EXPANDED UNIVERSE SUMMARY")
    print("=" * 70)
    print()

    # Filter suitable stocks
    suitable = [r for r in optimization_results if r['suitable']]
    unsuitable = [r for r in optimization_results if not r['suitable']]

    if suitable:
        print("SUITABLE FOR MEAN REVERSION:")
        print("-" * 70)
        print(f"{'Stock':<12} {'Z-score':<10} {'RSI':<8} {'Volume':<10} {'Win Rate':<12} {'Return':<12} {'Sharpe':<10}")
        print("-" * 70)

        for result in sorted(suitable, key=lambda x: x['performance']['win_rate'], reverse=True):
            params = result['best_params']
            perf = result['performance']

            print(f"{result['ticker']:<12} "
                  f"{params['z_score_threshold']:<10.2f} "
                  f"{params['rsi_oversold']:<8} "
                  f"{params['volume_multiplier']:<10.1f} "
                  f"{perf['win_rate']:<11.1f}% "
                  f"{perf['return']:>+10.2f}% "
                  f"{perf['sharpe']:<10.2f}")

        print("-" * 70)
        print(f"Total suitable stocks: {len(suitable)}")

    if unsuitable:
        print()
        print("NOT SUITABLE (exclude from strategy):")
        for result in unsuitable:
            perf = result['performance']
            print(f"  {result['ticker']:<6} {result['ticker_name']:<12} - "
                  f"{perf['win_rate']:.1f}% win, {perf['return']:+.2f}% return (below threshold)")

    print()
    print("=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)

    if suitable:
        print(f"1. ADD {len(suitable)} new stocks to trading universe")
        print(f"2. Update src/config/mean_reversion_params.py with optimal parameters")
        print(f"3. Current universe: 3 stocks -> {3 + len(suitable)} stocks")
        print()
        print("EXPECTED DIVERSIFICATION BENEFIT:")
        print("  - More trading opportunities")
        print("  - Reduced single-stock risk")
        print("  - Better portfolio performance")
    else:
        print("[WARN] No new stocks meet performance threshold (60% win rate)")
        print("Consider:")
        print("  - Testing different stock sectors (energy, finance, healthcare)")
        print("  - Lowering acceptance threshold (55% win rate)")
        print("  - Different strategy for stable mega-caps")

    print("=" * 70)

    return optimization_results


if __name__ == "__main__":
    expand_stock_universe()

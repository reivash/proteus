"""
EXPERIMENT: EXP-008-PARAMS
Date: 2025-11-14
Objective: Optimize mean reversion parameters per stock

HYPOTHESIS:
Different stocks have different volatility characteristics.
Stock-specific parameter tuning should improve performance over universal parameters.

APPROACH:
Test parameter combinations for each stock:
- z_score_threshold: How many standard deviations = panic sell
- rsi_oversold: RSI threshold for oversold condition
- volume_multiplier: Volume spike threshold
- price_drop_threshold: Minimum price drop %

STOCKS TESTED:
- NVDA: High volatility GPU/AI stock
- TSLA: Extremely volatile EV stock
- AAPL: More stable mega-cap tech stock
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from datetime import datetime
import itertools


def optimize_parameters_for_stock(ticker, ticker_name, start_date, end_date):
    """
    Find optimal parameters for a specific stock.

    Args:
        ticker: Stock ticker
        ticker_name: Stock name
        start_date: Start date
        end_date: End date

    Returns:
        Best parameters and performance metrics
    """
    print(f"\n{'=' * 70}")
    print(f"OPTIMIZING PARAMETERS: {ticker_name} ({ticker})")
    print(f"{'=' * 70}")
    print(f"Period: {start_date} to {end_date}")
    print()

    # Fetch data once
    fetcher = YahooFinanceFetcher()
    try:
        data = fetcher.fetch_stock_data(ticker, start_date=start_date, end_date=end_date)
    except Exception as e:
        print(f"[FAIL] Could not fetch data: {e}")
        return None

    if len(data) < 60:
        print(f"[FAIL] Insufficient data ({len(data)} rows)")
        return None

    # Engineer features once
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)

    # Parameter grid
    param_grid = {
        'z_score_threshold': [1.0, 1.25, 1.5, 1.75, 2.0],
        'rsi_oversold': [30, 32, 35, 38, 40],
        'volume_multiplier': [1.3, 1.5, 1.7, 2.0],
        'price_drop_threshold': [-1.5, -2.0, -2.5, -3.0]
    }

    print(f"Testing {len(param_grid['z_score_threshold']) * len(param_grid['rsi_oversold']) * len(param_grid['volume_multiplier']) * len(param_grid['price_drop_threshold'])} parameter combinations...")
    print()

    # Test each combination
    results = []

    for z_threshold in param_grid['z_score_threshold']:
        for rsi_oversold in param_grid['rsi_oversold']:
            for volume_mult in param_grid['volume_multiplier']:
                for price_drop in param_grid['price_drop_threshold']:

                    # Detect overcorrections with these parameters
                    detector = MeanReversionDetector(
                        z_score_threshold=z_threshold,
                        rsi_oversold=rsi_oversold,
                        rsi_overbought=65,  # Keep constant
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

                    # Skip if no signals
                    if signals['panic_sell'].sum() == 0:
                        continue

                    # Backtest
                    backtester = MeanReversionBacktester(
                        initial_capital=10000,
                        profit_target=2.0,
                        stop_loss=-2.0,
                        max_hold_days=2
                    )

                    backtest_results = backtester.backtest(signals)

                    # Skip if no trades
                    if backtest_results['total_trades'] == 0:
                        continue

                    # Store results
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
                        'max_drawdown': backtest_results['max_drawdown'],
                        'winning_trades': backtest_results['winning_trades'],
                        'losing_trades': backtest_results['losing_trades']
                    })

    if len(results) == 0:
        print("[FAIL] No valid parameter combinations found")
        return None

    # Sort by multiple criteria
    # Primary: Win rate (must be > 60%)
    # Secondary: Sharpe ratio
    # Tertiary: Total return

    valid_results = [r for r in results if r['win_rate'] >= 60 and r['trades'] >= 5]

    if len(valid_results) == 0:
        print("[WARN] No combinations with 60%+ win rate and 5+ trades")
        # Fall back to best available
        valid_results = sorted(results, key=lambda x: (x['win_rate'], x['sharpe']), reverse=True)
    else:
        valid_results = sorted(valid_results, key=lambda x: (x['win_rate'], x['sharpe']), reverse=True)

    # Get top 5 combinations
    top_5 = valid_results[:5]

    print(f"TOP 5 PARAMETER COMBINATIONS (out of {len(results)} valid):")
    print("-" * 70)
    print(f"{'Rank':<6} {'Z-score':<10} {'RSI':<8} {'Volume':<10} {'Drop':<10} {'Trades':<8} {'Win%':<8} {'Return':<10} {'Sharpe':<8}")
    print("-" * 70)

    for i, result in enumerate(top_5, 1):
        print(f"{i:<6} {result['z_score']:<10.2f} {result['rsi_oversold']:<8} "
              f"{result['volume_mult']:<10.1f} {result['price_drop']:<10.1f} "
              f"{result['trades']:<8} {result['win_rate']:<7.1f}% "
              f"{result['return']:>+8.2f}% {result['sharpe']:<8.2f}")

    print("-" * 70)

    # Best combination
    best = top_5[0]

    print(f"\nBEST PARAMETERS:")
    print(f"  Z-score threshold: {best['z_score']}")
    print(f"  RSI oversold: {best['rsi_oversold']}")
    print(f"  Volume multiplier: {best['volume_mult']}")
    print(f"  Price drop: {best['price_drop']}%")
    print()
    print(f"PERFORMANCE:")
    print(f"  Signals: {best['signals']}")
    print(f"  Trades: {best['trades']}")
    print(f"  Win rate: {best['win_rate']:.1f}%")
    print(f"  Return: {best['return']:+.2f}%")
    print(f"  Sharpe ratio: {best['sharpe']:.2f}")
    print(f"  Max drawdown: {best['max_drawdown']:.2f}%")

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
        'top_5': top_5
    }


def compare_universal_vs_optimized():
    """
    Compare universal parameters vs stock-specific optimized parameters.
    """
    print("\n" + "=" * 70)
    print("COMPARING: UNIVERSAL vs STOCK-SPECIFIC PARAMETERS")
    print("=" * 70)
    print()

    test_period = ('2022-01-01', '2025-11-14')

    test_stocks = [
        ("NVDA", "NVIDIA"),
        ("TSLA", "Tesla"),
        ("AAPL", "Apple")
    ]

    # Universal parameters (current)
    universal_params = {
        'z_score_threshold': 1.5,
        'rsi_oversold': 35,
        'volume_multiplier': 1.5,
        'price_drop_threshold': -2.0
    }

    print("UNIVERSAL PARAMETERS (baseline):")
    print(f"  Z-score: {universal_params['z_score_threshold']}")
    print(f"  RSI: {universal_params['rsi_oversold']}")
    print(f"  Volume: {universal_params['volume_multiplier']}")
    print(f"  Price drop: {universal_params['price_drop_threshold']}%")
    print()

    # Optimize for each stock
    optimization_results = []

    for ticker, name in test_stocks:
        result = optimize_parameters_for_stock(ticker, name, *test_period)
        if result:
            optimization_results.append(result)

    # Summary comparison
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    print()

    print(f"{'Stock':<12} {'Z-score':<12} {'RSI':<8} {'Volume':<10} {'Drop':<10} {'Win Rate':<12} {'Return':<12} {'Sharpe':<10}")
    print("-" * 70)

    for result in optimization_results:
        params = result['best_params']
        perf = result['performance']

        print(f"{result['ticker']:<12} "
              f"{params['z_score_threshold']:<12.2f} "
              f"{params['rsi_oversold']:<8} "
              f"{params['volume_multiplier']:<10.1f} "
              f"{params['price_drop_threshold']:<10.1f} "
              f"{perf['win_rate']:<11.1f}% "
              f"{perf['return']:>+10.2f}% "
              f"{perf['sharpe']:<10.2f}")

    print("=" * 70)

    # Analysis
    print("\nKEY FINDINGS:")

    # Check if parameters vary significantly
    z_scores = [r['best_params']['z_score_threshold'] for r in optimization_results]
    rsi_values = [r['best_params']['rsi_oversold'] for r in optimization_results]

    z_range = max(z_scores) - min(z_scores)
    rsi_range = max(rsi_values) - min(rsi_values)

    if z_range > 0.5:
        print(f"  [IMPORTANT] Z-score varies significantly: {min(z_scores):.2f} - {max(z_scores):.2f}")
        print(f"              Stock-specific tuning is valuable!")
    else:
        print(f"  [NEUTRAL] Z-score fairly consistent: {min(z_scores):.2f} - {max(z_scores):.2f}")

    if rsi_range > 5:
        print(f"  [IMPORTANT] RSI varies significantly: {min(rsi_values)} - {max(rsi_values)}")
        print(f"              Stock-specific RSI thresholds recommended!")
    else:
        print(f"  [NEUTRAL] RSI fairly consistent: {min(rsi_values)} - {max(rsi_values)}")

    # Performance improvement
    avg_win_rate = sum(r['performance']['win_rate'] for r in optimization_results) / len(optimization_results)
    avg_sharpe = sum(r['performance']['sharpe'] for r in optimization_results) / len(optimization_results)

    print()
    print(f"AVERAGE PERFORMANCE WITH OPTIMIZED PARAMETERS:")
    print(f"  Win rate: {avg_win_rate:.1f}%")
    print(f"  Sharpe ratio: {avg_sharpe:.2f}")

    return optimization_results


if __name__ == "__main__":
    compare_universal_vs_optimized()

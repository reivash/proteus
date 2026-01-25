"""
EXPERIMENT: EXP-008-BEAR
Date: 2025-11-14
Objective: Test mean reversion strategy in 2022 bear market

CONTEXT:
2022 was a significant bear market due to:
- Fed rate hikes (0% to 4.5%)
- High inflation (8-9%)
- Tech selloff (-30% NASDAQ)
- Recession fears

QUESTION:
Does mean reversion strategy work in bear markets, or only in bull markets?

This tests if strategy is robust across different market regimes.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from exp008_mean_reversion import run_mean_reversion_experiment
from datetime import datetime

def compare_market_regimes():
    """Compare mean reversion performance across bull and bear markets."""

    print("=" * 70)
    print(" " * 10 + "MEAN REVERSION: BULL VS BEAR MARKET TEST")
    print("=" * 70)
    print()

    # Test configurations
    test_stocks = [
        ("NVDA", "NVIDIA"),
        ("TSLA", "Tesla"),
        ("AAPL", "Apple")
    ]

    # Market periods
    periods = {
        'bear_2022': {
            'start': '2022-01-01',
            'end': '2022-12-31',
            'description': '2022 Bear Market (Fed hikes, inflation, tech selloff)'
        },
        'bull_2023_2024': {
            'start': '2023-01-01',
            'end': '2024-12-31',
            'description': '2023-2024 Bull Market (AI boom, rate pause)'
        },
        'recent_2024_2025': {
            'start': '2024-01-01',
            'end': '2025-11-13',
            'description': '2024-2025 Recent Period (continued AI growth)'
        }
    }

    results_by_period = {}

    for period_name, period_config in periods.items():
        print(f"\n{'=' * 70}")
        print(f"TESTING PERIOD: {period_config['description']}")
        print(f"Date Range: {period_config['start']} to {period_config['end']}")
        print(f"{'=' * 70}\n")

        period_results = []

        for ticker, name in test_stocks:
            print(f"\nTesting {name} ({ticker})...")
            print("-" * 70)

            # Run experiment with custom date range
            result = run_mean_reversion_experiment_custom_dates(
                ticker=ticker,
                ticker_name=name,
                start_date=period_config['start'],
                end_date=period_config['end']
            )

            if result and result['backtest_results']['total_trades'] > 0:
                period_results.append({
                    'ticker': ticker,
                    'name': name,
                    'trades': result['backtest_results']['total_trades'],
                    'win_rate': result['backtest_results']['win_rate'],
                    'return': result['backtest_results']['total_return'],
                    'sharpe': result['backtest_results']['sharpe_ratio']
                })

                print(f"  [OK] {result['backtest_results']['total_trades']} trades, "
                      f"{result['backtest_results']['win_rate']:.1f}% win rate, "
                      f"{result['backtest_results']['total_return']:+.2f}% return")
            else:
                period_results.append({
                    'ticker': ticker,
                    'name': name,
                    'trades': 0,
                    'win_rate': 0,
                    'return': 0,
                    'sharpe': 0
                })
                print(f"  [WARN] No trades or failed")

        results_by_period[period_name] = {
            'config': period_config,
            'results': period_results
        }

    # Print comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON: MEAN REVERSION ACROSS MARKET REGIMES")
    print("=" * 70)

    for period_name, data in results_by_period.items():
        config = data['config']
        results = data['results']

        print(f"\n{config['description']}")
        print("-" * 70)
        print(f"{'Stock':<12} {'Trades':>8} {'Win Rate':>10} {'Return':>10} {'Sharpe':>10}")
        print("-" * 70)

        total_trades = 0
        total_wins = 0
        total_returns = []

        for r in results:
            if r['trades'] > 0:
                print(f"{r['ticker']:<12} {r['trades']:>8} {r['win_rate']:>9.1f}% "
                      f"{r['return']:>9.2f}% {r['sharpe']:>10.2f}")
                total_trades += r['trades']
                total_wins += r['trades'] * (r['win_rate'] / 100)
                total_returns.append(r['return'])

        if total_trades > 0:
            avg_win_rate = (total_wins / total_trades) * 100
            avg_return = sum(total_returns) / len(total_returns)
            print("-" * 70)
            print(f"{'AVERAGE':<12} {total_trades/len(results):>8.1f} {avg_win_rate:>9.1f}% {avg_return:>9.2f}%")
        else:
            print("[NO TRADES IN THIS PERIOD]")

    # Final assessment
    print("\n" + "=" * 70)
    print("REGIME ANALYSIS:")
    print("=" * 70)

    # Compare bear vs bull
    bear_results = results_by_period['bear_2022']['results']
    bull_results = results_by_period['bull_2023_2024']['results']

    bear_avg = sum(r['return'] for r in bear_results if r['trades'] > 0) / max(1, len([r for r in bear_results if r['trades'] > 0]))
    bull_avg = sum(r['return'] for r in bull_results if r['trades'] > 0) / max(1, len([r for r in bull_results if r['trades'] > 0]))

    print(f"\nBear Market (2022):     {bear_avg:+.2f}% avg return")
    print(f"Bull Market (2023-24):  {bull_avg:+.2f}% avg return")

    if bear_avg > 0 and bull_avg > 0:
        print("\n[SUCCESS] Strategy works in BOTH bull and bear markets!")
        print("Strategy is regime-independent - robust across market conditions.")
    elif bull_avg > 0 and bear_avg <= 0:
        print("\n[CAUTION] Strategy only works in bull markets.")
        print("Do NOT use in bear market conditions.")
    elif bear_avg > 0 and bull_avg <= 0:
        print("\n[INTERESTING] Strategy works better in bear markets!")
        print("Counter-trend strategy thrives on volatility.")
    else:
        print("\n[FAILED] Strategy doesn't work in either regime.")
        print("Need to reconsider approach.")

    print("=" * 70)


def run_mean_reversion_experiment_custom_dates(ticker, ticker_name, start_date, end_date):
    """
    Run mean reversion with custom date range.

    This is a wrapper around the main experiment that allows specifying exact dates.
    """
    from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
    from common.data.features.technical_indicators import TechnicalFeatureEngineer
    from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester

    # Fetch data
    fetcher = YahooFinanceFetcher()
    try:
        data = fetcher.fetch_stock_data(ticker, start_date=start_date, end_date=end_date)
    except Exception as e:
        print(f"      [FAIL] Could not fetch data: {e}")
        return None

    # Need minimum data
    if len(data) < 60:
        print(f"      [FAIL] Insufficient data ({len(data)} rows)")
        return None

    # Engineer features
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)

    # Detect overcorrections
    detector = MeanReversionDetector(
        z_score_threshold=1.5,
        rsi_oversold=35,
        rsi_overbought=65,
        volume_multiplier=1.5,
        price_drop_threshold=-2.0
    )

    signals = detector.detect_overcorrections(enriched_data)
    signals = detector.calculate_reversion_targets(signals)

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
        'experiment_id': 'EXP-008-BEAR',
        'ticker': ticker,
        'ticker_name': ticker_name,
        'start_date': start_date,
        'end_date': end_date,
        'backtest_results': results
    }


if __name__ == "__main__":
    compare_market_regimes()

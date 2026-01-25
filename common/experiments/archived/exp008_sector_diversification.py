"""
EXPERIMENT: EXP-008-SECTORS
Date: 2025-11-14
Objective: Diversify trading universe across sectors

CURRENT STATE:
5 stocks, all tech sector (100% concentration risk)
- NVDA, TSLA, AAPL, AMZN, MSFT

GOAL:
Expand to 8-10 stocks across multiple sectors
- Energy (oil & gas, renewables)
- Finance (banks, investment firms)
- Healthcare (pharma, insurance)
- Keep best tech performers

APPROACH:
Test high-quality stocks from each sector with parameter optimization.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester


def optimize_stock(ticker, ticker_name, sector, start_date, end_date):
    """
    Optimize parameters for a stock from any sector.
    """
    print(f"\n{'=' * 70}")
    print(f"OPTIMIZING: {ticker_name} ({ticker}) - {sector}")
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

    # Parameter grid
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
        print(f"[WARN] No combinations with 60%+ win rate and 5+ trades")
        print(f"This stock may not be suitable for mean reversion.")
        best_available = sorted(results, key=lambda x: (x['win_rate'], x['sharpe']), reverse=True)[:3]
        print(f"\nBEST AVAILABLE (out of {len(results)} combinations):")
        for i, r in enumerate(best_available, 1):
            print(f"{i}. z={r['z_score']:.2f}, RSI={r['rsi_oversold']}, vol={r['volume_mult']:.1f}x "
                  f"-> {r['trades']} trades, {r['win_rate']:.1f}% win, {r['return']:+.2f}%")
        return None

    valid_results = sorted(valid_results, key=lambda x: (x['win_rate'], x['sharpe']), reverse=True)
    best = valid_results[0]

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

    # Assessment
    if best['win_rate'] >= 75 and best['sharpe'] > 3:
        assessment = "EXCELLENT"
        print(f"  [{assessment}] Strong candidate for mean reversion!")
    elif best['win_rate'] >= 65 and best['sharpe'] > 1:
        assessment = "GOOD"
        print(f"  [{assessment}] Suitable for mean reversion")
    elif best['win_rate'] >= 60:
        assessment = "ACCEPTABLE"
        print(f"  [{assessment}] Marginal performance, monitor closely")
    else:
        assessment = "POOR"
        print(f"  [{assessment}] Consider excluding from strategy")

    return {
        'ticker': ticker,
        'ticker_name': ticker_name,
        'sector': sector,
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
        'assessment': assessment,
        'suitable': best['win_rate'] >= 60 and assessment in ['EXCELLENT', 'GOOD', 'ACCEPTABLE']
    }


def test_sector_diversification():
    """
    Test stocks across multiple sectors for diversification.
    """
    print("=" * 70)
    print("SECTOR DIVERSIFICATION EXPERIMENT")
    print("=" * 70)
    print()

    test_period = ('2022-01-01', '2025-11-14')

    # Sector candidates
    candidates = [
        # Energy sector
        ("XOM", "ExxonMobil", "Energy"),
        ("CVX", "Chevron", "Energy"),

        # Finance sector
        ("JPM", "JPMorgan Chase", "Finance"),
        ("BAC", "Bank of America", "Finance"),

        # Healthcare sector
        ("UNH", "UnitedHealth", "Healthcare"),
        ("JNJ", "Johnson & Johnson", "Healthcare"),

        # Additional tech (for comparison)
        ("INTC", "Intel", "Tech - Semiconductor"),
        ("AMD", "AMD", "Tech - Semiconductor"),
    ]

    print("CURRENT UNIVERSE (Tech-only):")
    print("  NVDA: 87.5% win, +45.75% return")
    print("  TSLA: 87.5% win, +12.81% return")
    print("  AMZN: 80.0% win, +10.20% return")
    print("  MSFT: 66.7% win, +2.06% return")
    print("  AAPL: 60.0% win, -0.25% return")
    print()
    print("TESTING NEW SECTORS:")
    print()

    results_by_sector = {
        'Energy': [],
        'Finance': [],
        'Healthcare': [],
        'Tech - Semiconductor': []
    }

    for ticker, name, sector in candidates:
        result = optimize_stock(ticker, name, sector, *test_period)
        if result:
            results_by_sector[sector].append(result)

    # Summary by sector
    print("\n" + "=" * 70)
    print("SECTOR ANALYSIS")
    print("=" * 70)
    print()

    for sector in ['Energy', 'Finance', 'Healthcare', 'Tech - Semiconductor']:
        sector_results = results_by_sector[sector]

        if not sector_results:
            print(f"{sector.upper()}:")
            print("  [FAIL] No suitable stocks found")
            print()
            continue

        suitable = [r for r in sector_results if r['suitable']]

        print(f"{sector.upper()}:")
        if suitable:
            print(f"  Suitable stocks: {len(suitable)}/{len(sector_results)}")
            for result in sorted(suitable, key=lambda x: x['performance']['win_rate'], reverse=True):
                perf = result['performance']
                print(f"    {result['ticker']:<6} {result['ticker_name']:<20} - "
                      f"{perf['win_rate']:.1f}% win, {perf['return']:+.2f}% return, "
                      f"Sharpe {perf['sharpe']:.2f}")
        else:
            print(f"  [WARN] No suitable stocks from {len(sector_results)} tested")
        print()

    # Overall summary
    all_suitable = []
    for sector_results in results_by_sector.values():
        all_suitable.extend([r for r in sector_results if r['suitable']])

    if all_suitable:
        print("=" * 70)
        print("RECOMMENDED ADDITIONS (sorted by win rate)")
        print("=" * 70)
        print(f"{'Ticker':<8} {'Name':<20} {'Sector':<20} {'Win%':<8} {'Return':<10} {'Sharpe':<8}")
        print("-" * 70)

        for result in sorted(all_suitable, key=lambda x: x['performance']['win_rate'], reverse=True):
            perf = result['performance']
            print(f"{result['ticker']:<8} {result['ticker_name']:<20} {result['sector']:<20} "
                  f"{perf['win_rate']:<7.1f}% {perf['return']:>+8.2f}% {perf['sharpe']:<8.2f}")

        print("=" * 70)
        print()

        # Sector breakdown
        sector_counts = {}
        for r in all_suitable:
            sector = r['sector']
            if sector not in sector_counts:
                sector_counts[sector] = 0
            sector_counts[sector] += 1

        print("SECTOR DISTRIBUTION:")
        for sector, count in sorted(sector_counts.items()):
            print(f"  {sector}: {count} stock(s)")

        total_stocks = len(all_suitable)
        print()
        print(f"PROPOSED UNIVERSE: 5 (current) + {total_stocks} (new) = {5 + total_stocks} stocks")

        # Diversification metrics
        tech_count = 5 + sector_counts.get('Tech - Semiconductor', 0)
        tech_pct = tech_count / (5 + total_stocks) * 100

        print()
        print("DIVERSIFICATION METRICS:")
        print(f"  Tech stocks: {tech_count}/{5 + total_stocks} ({tech_pct:.1f}%)")
        print(f"  Non-tech stocks: {total_stocks - sector_counts.get('Tech - Semiconductor', 0)}/{5 + total_stocks} ({100-tech_pct:.1f}%)")

        if tech_pct < 70:
            print("  [GOOD] Reduced tech concentration from 100% to {tech_pct:.1f}%")
        else:
            print("  [WARN] Still high tech concentration ({tech_pct:.1f}%)")
    else:
        print("=" * 70)
        print("[WARN] NO SUITABLE STOCKS FOUND")
        print("=" * 70)
        print()
        print("RECOMMENDATIONS:")
        print("1. Mean reversion may work best on tech sector")
        print("2. Consider different strategies for other sectors")
        print("3. Focus on optimizing existing tech universe")

    print("=" * 70)

    return all_suitable


if __name__ == "__main__":
    test_sector_diversification()

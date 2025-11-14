"""
EXPERIMENT: EXP-008-UNIVERSE-OPT
Date: 2025-11-14
Objective: Review marginal performers (AAPL, CVX) for potential removal

HYPOTHESIS:
Removing marginal performers (60% win rate) may improve portfolio metrics.

CURRENT UNIVERSE (10 stocks):
- Tier 1 (80%+ win): NVDA, TSLA, AMZN, JPM, JNJ, UNH (6 stocks)
- Tier 2 (65-75% win): MSFT, INTC (2 stocks)
- Tier 3 (60% win): AAPL, CVX (2 stocks) <- CANDIDATES FOR REMOVAL

PROPOSED UNIVERSE (8 stocks):
- Remove AAPL (60% win, -0.25% return, -0.13 Sharpe)
- Remove CVX (60% win, +2.50% return, 2.68 Sharpe)
- Keep: Tier 1 + Tier 2 only (8 stocks, all 65%+ win rate)

EXPECTED IMPACT:
- Higher average win rate (remove 60% performers)
- Higher average return (remove negative/low performers)
- Better Sharpe ratio (better risk-adjusted returns)
- More focused portfolio (quality > quantity)

TEST METHODOLOGY:
- Compare 10-stock vs 8-stock portfolio metrics
- Analyze contribution of AAPL and CVX
- Determine optimal universe size
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from src.config.mean_reversion_params import get_params


def test_stock(ticker, start_date, end_date):
    """
    Test mean reversion on a single stock.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date

    Returns:
        Results dictionary or None
    """
    # Fetch data
    fetcher = YahooFinanceFetcher()
    try:
        data = fetcher.fetch_stock_data(ticker, start_date=start_date, end_date=end_date)
    except Exception as e:
        print(f"[WARN] Could not fetch data for {ticker}: {e}")
        return None

    if len(data) < 60:
        print(f"[WARN] Insufficient data for {ticker}")
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

    # Apply regime filter
    regime_detector = MarketRegimeDetector()
    signals = add_regime_filter_to_signals(signals, regime_detector)

    # Apply earnings filter
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

    results['ticker'] = ticker
    return results


def compare_universe_sizes():
    """
    Compare 10-stock vs 8-stock universe.
    """
    print("=" * 70)
    print("UNIVERSE OPTIMIZATION: 10-STOCK vs 8-STOCK")
    print("=" * 70)
    print()

    test_period = ('2022-01-01', '2025-11-14')

    # Define universes
    universe_10 = ['NVDA', 'TSLA', 'AAPL', 'AMZN', 'MSFT', 'JPM', 'JNJ', 'UNH', 'INTC', 'CVX']
    universe_8 = ['NVDA', 'TSLA', 'AMZN', 'MSFT', 'JPM', 'JNJ', 'UNH', 'INTC']  # Remove AAPL, CVX

    print(f"Test period: {test_period[0]} to {test_period[1]}")
    print()

    # Test 10-stock universe
    print("=" * 70)
    print("10-STOCK UNIVERSE (Current)")
    print("=" * 70)
    print()

    results_10 = []
    for ticker in universe_10:
        print(f"Testing {ticker}...", end=" ")
        result = test_stock(ticker, *test_period)
        if result:
            results_10.append(result)
            print(f"[OK] {result['total_trades']} trades, {result['win_rate']:.1f}% win, {result['total_return']:+.2f}%")
        else:
            print("[SKIP] No trades")

    # Test 8-stock universe
    print()
    print("=" * 70)
    print("8-STOCK UNIVERSE (Remove AAPL, CVX)")
    print("=" * 70)
    print()

    results_8 = []
    for ticker in universe_8:
        print(f"Testing {ticker}...", end=" ")
        result = test_stock(ticker, *test_period)
        if result:
            results_8.append(result)
            print(f"[OK] {result['total_trades']} trades, {result['win_rate']:.1f}% win, {result['total_return']:+.2f}%")
        else:
            print("[SKIP] No trades")

    # Portfolio-level comparison
    print()
    print("=" * 70)
    print("PORTFOLIO-LEVEL COMPARISON")
    print("=" * 70)
    print()

    def calculate_portfolio_metrics(results, universe_size):
        """Calculate portfolio-level metrics."""
        total_trades = sum(r['total_trades'] for r in results)
        total_winning = sum(r['winning_trades'] for r in results)
        total_losing = sum(r['losing_trades'] for r in results)

        # Weighted average by number of trades
        avg_win_rate = np.average(
            [r['win_rate'] for r in results],
            weights=[r['total_trades'] for r in results]
        )

        # Portfolio return (sum of individual returns)
        portfolio_return = sum(r['total_return'] for r in results)
        avg_return_per_stock = portfolio_return / len(results)

        # Weighted Sharpe ratio
        avg_sharpe = np.average(
            [r['sharpe_ratio'] for r in results],
            weights=[r['total_trades'] for r in results]
        )

        # Max drawdown
        avg_max_drawdown = np.average([r['max_drawdown'] for r in results])

        return {
            'universe_size': universe_size,
            'stocks_tested': len(results),
            'total_trades': total_trades,
            'avg_win_rate': avg_win_rate,
            'portfolio_return': portfolio_return,
            'avg_return_per_stock': avg_return_per_stock,
            'avg_sharpe': avg_sharpe,
            'avg_max_drawdown': avg_max_drawdown,
            'winning_trades': total_winning,
            'losing_trades': total_losing
        }

    portfolio_10 = calculate_portfolio_metrics(results_10, 10)
    portfolio_8 = calculate_portfolio_metrics(results_8, 8)

    # Print comparison
    print(f"{'Metric':<30} {'10-Stock':<20} {'8-Stock':<20} {'Change':<15}")
    print("-" * 85)
    print(f"{'Universe Size':<30} {portfolio_10['universe_size']:<20} {portfolio_8['universe_size']:<20} {'-':<15}")
    print(f"{'Stocks Tested':<30} {portfolio_10['stocks_tested']:<20} {portfolio_8['stocks_tested']:<20} {'-':<15}")
    print(f"{'Total Trades':<30} {portfolio_10['total_trades']:<20} {portfolio_8['total_trades']:<20} {'-':<15}")

    win_rate_change = portfolio_8['avg_win_rate'] - portfolio_10['avg_win_rate']
    print(f"{'Avg Win Rate':<30} {portfolio_10['avg_win_rate']:<19.1f}% {portfolio_8['avg_win_rate']:<19.1f}% {win_rate_change:+.1f}%")

    portfolio_return_change = portfolio_8['portfolio_return'] - portfolio_10['portfolio_return']
    print(f"{'Portfolio Return (Sum)':<30} {portfolio_10['portfolio_return']:<19.2f}% {portfolio_8['portfolio_return']:<19.2f}% {portfolio_return_change:+.2f}%")

    avg_return_change = portfolio_8['avg_return_per_stock'] - portfolio_10['avg_return_per_stock']
    print(f"{'Avg Return Per Stock':<30} {portfolio_10['avg_return_per_stock']:<19.2f}% {portfolio_8['avg_return_per_stock']:<19.2f}% {avg_return_change:+.2f}%")

    sharpe_change = portfolio_8['avg_sharpe'] - portfolio_10['avg_sharpe']
    print(f"{'Avg Sharpe Ratio':<30} {portfolio_10['avg_sharpe']:<19.2f} {portfolio_8['avg_sharpe']:<19.2f} {sharpe_change:+.2f}")

    drawdown_change = portfolio_8['avg_max_drawdown'] - portfolio_10['avg_max_drawdown']
    print(f"{'Avg Max Drawdown':<30} {portfolio_10['avg_max_drawdown']:<19.2f}% {portfolio_8['avg_max_drawdown']:<19.2f}% {drawdown_change:+.2f}%")

    print("=" * 85)

    # AAPL and CVX contribution analysis
    print()
    print("=" * 70)
    print("AAPL & CVX CONTRIBUTION ANALYSIS")
    print("=" * 70)
    print()

    aapl_result = next((r for r in results_10 if r['ticker'] == 'AAPL'), None)
    cvx_result = next((r for r in results_10 if r['ticker'] == 'CVX'), None)

    if aapl_result:
        print(f"AAPL Contribution:")
        print(f"  Trades: {aapl_result['total_trades']}")
        print(f"  Win rate: {aapl_result['win_rate']:.1f}%")
        print(f"  Return: {aapl_result['total_return']:+.2f}%")
        print(f"  Sharpe: {aapl_result['sharpe_ratio']:.2f}")
        print(f"  Assessment: {'NEGATIVE' if aapl_result['total_return'] < 0 else 'POSITIVE'} contribution")

    print()

    if cvx_result:
        print(f"CVX Contribution:")
        print(f"  Trades: {cvx_result['total_trades']}")
        print(f"  Win rate: {cvx_result['win_rate']:.1f}%")
        print(f"  Return: {cvx_result['total_return']:+.2f}%")
        print(f"  Sharpe: {cvx_result['sharpe_ratio']:.2f}")
        print(f"  Assessment: POSITIVE contribution (but marginal)")

    print()

    combined_contribution = 0
    if aapl_result:
        combined_contribution += aapl_result['total_return']
    if cvx_result:
        combined_contribution += cvx_result['total_return']

    print(f"Combined AAPL + CVX contribution: {combined_contribution:+.2f}%")
    print(f"Impact of removal: {-combined_contribution:+.2f}% portfolio return")

    print("=" * 70)

    # Detailed stock comparison
    print()
    print("=" * 70)
    print("REMAINING STOCKS (8-STOCK UNIVERSE)")
    print("=" * 70)
    print()

    print(f"{'Stock':<8} {'Trades':<8} {'Win Rate':<12} {'Return':<12} {'Sharpe':<10} {'Tier':<10}")
    print("-" * 70)

    # Sort by win rate
    sorted_results = sorted(results_8, key=lambda x: x['win_rate'], reverse=True)

    for result in sorted_results:
        ticker = result['ticker']
        win_rate = result['win_rate']

        # Determine tier
        if win_rate >= 80:
            tier = "Tier 1"
        elif win_rate >= 65:
            tier = "Tier 2"
        else:
            tier = "Tier 3"

        print(f"{ticker:<8} {result['total_trades']:<8} {result['win_rate']:<11.1f}% {result['total_return']:>+10.2f}% {result['sharpe_ratio']:<9.2f} {tier:<10}")

    print("=" * 70)

    # Conclusion
    print()
    print("CONCLUSION:")
    print()

    if portfolio_return_change > 0 and win_rate_change > 0:
        print("[WORSE] Removing AAPL/CVX REDUCES portfolio performance")
        print(f"Portfolio return loss: {portfolio_return_change:+.2f}%")
        print(f"Win rate change: {win_rate_change:+.1f}%")
        print("Recommendation: KEEP 10-stock universe (AAPL and CVX contribute value)")
    elif portfolio_return_change < -2:
        print("[BETTER] Removing AAPL/CVX IMPROVES portfolio performance")
        print(f"Portfolio return gain: {-portfolio_return_change:+.2f}%")
        print("Recommendation: REMOVE AAPL and CVX, use 8-stock universe")
    else:
        print("[NEUTRAL] Minimal impact from removing AAPL/CVX")
        print(f"Portfolio return change: {portfolio_return_change:+.2f}%")
        print("Consider diversification benefits vs marginal performance")

    print()
    print("KEY INSIGHTS:")
    print(f"- 10-stock universe: {portfolio_10['stocks_tested']} stocks, {portfolio_10['avg_win_rate']:.1f}% win rate")
    print(f"- 8-stock universe: {portfolio_8['stocks_tested']} stocks, {portfolio_8['avg_win_rate']:.1f}% win rate")
    print(f"- AAPL/CVX combined: {combined_contribution:+.2f}% return contribution")
    print(f"- Trade-off: Diversification (10 stocks) vs Quality (8 stocks)")

    print()
    print("=" * 70)


if __name__ == "__main__":
    compare_universe_sizes()

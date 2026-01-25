"""
EXPERIMENT: EXP-008-POSITION-SIZING
Date: 2025-11-14
Objective: Test dynamic position sizing vs fixed position sizing

HYPOTHESIS:
Allocating larger positions to higher-performing stocks (higher win rate, better Sharpe)
will improve portfolio-level returns and risk-adjusted performance.

APPROACH:
- FIXED sizing: All stocks get 1.0x position (equal weighting)
- DYNAMIC sizing: Tier-based position sizing
  - Tier 1 (80%+ win): 1.0x position (full)
  - Tier 2 (65-75% win): 0.75x position (reduced)
  - Tier 3 (60-65% win): 0.5x position (minimal)

EXPECTED IMPACT:
- Higher portfolio returns (concentrate capital in best performers)
- Better Sharpe ratio (better risk-adjusted returns)
- Reduced exposure to marginal stocks
- Natural risk management

TEST METHODOLOGY:
- Run backtests on all 10 stocks (2022-2025 period)
- Calculate portfolio-level metrics
- Compare FIXED vs DYNAMIC approaches
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from common.config.mean_reversion_params import get_params, get_all_tickers
from common.config.position_sizing import get_position_size, get_tier, get_position_info


def test_stock_with_position_sizing(ticker, start_date, end_date, position_size_multiplier=1.0):
    """
    Test mean reversion with dynamic position sizing.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date
        position_size_multiplier: Position size multiplier (default 1.0)

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

    # Backtest with position sizing
    backtester = MeanReversionBacktester(
        initial_capital=10000,
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=2,
        position_size_multiplier=position_size_multiplier
    )

    results = backtester.backtest(signals)

    if results['total_trades'] == 0:
        return None

    # Add position sizing info
    results['ticker'] = ticker
    results['position_size'] = position_size_multiplier
    results['tier'] = get_tier(ticker)

    return results


def compare_fixed_vs_dynamic_sizing():
    """
    Compare portfolio performance with fixed vs dynamic position sizing.
    """
    print("=" * 70)
    print("DYNAMIC POSITION SIZING TEST")
    print("=" * 70)
    print()

    test_period = ('2022-01-01', '2025-11-14')
    tickers = get_all_tickers()

    print(f"Test period: {test_period[0]} to {test_period[1]}")
    print(f"Testing {len(tickers)} stocks")
    print()

    # FIXED POSITION SIZING (1.0x for all stocks)
    print("=" * 70)
    print("FIXED POSITION SIZING (1.0x for all stocks)")
    print("=" * 70)
    print()

    fixed_results = []
    for ticker in tickers:
        print(f"Testing {ticker} (Fixed 1.0x)...", end=" ")
        result = test_stock_with_position_sizing(ticker, *test_period, position_size_multiplier=1.0)
        if result:
            fixed_results.append(result)
            print(f"[OK] {result['total_trades']} trades, {result['win_rate']:.1f}% win, {result['total_return']:+.2f}%")
        else:
            print("[SKIP] No trades")

    # DYNAMIC POSITION SIZING (tier-based)
    print()
    print("=" * 70)
    print("DYNAMIC POSITION SIZING (tier-based)")
    print("=" * 70)
    print()

    dynamic_results = []
    for ticker in tickers:
        position_size = get_position_size(ticker)
        tier = get_tier(ticker)
        print(f"Testing {ticker} (Tier {tier}, {position_size:.2f}x)...", end=" ")
        result = test_stock_with_position_sizing(ticker, *test_period, position_size_multiplier=position_size)
        if result:
            dynamic_results.append(result)
            print(f"[OK] {result['total_trades']} trades, {result['win_rate']:.1f}% win, {result['total_return']:+.2f}%")
        else:
            print("[SKIP] No trades")

    # Portfolio-level comparison
    print()
    print("=" * 70)
    print("PORTFOLIO-LEVEL COMPARISON")
    print("=" * 70)
    print()

    if not fixed_results or not dynamic_results:
        print("[ERROR] Insufficient results for comparison")
        return

    # Calculate portfolio metrics
    def calculate_portfolio_metrics(results):
        """Calculate portfolio-level metrics from individual stock results."""
        total_capital = 10000 * len(results)  # $10k per stock
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

        # Average return per stock
        avg_return_per_stock = portfolio_return / len(results)

        # Weighted Sharpe ratio
        avg_sharpe = np.average(
            [r['sharpe_ratio'] for r in results],
            weights=[r['total_trades'] for r in results]
        )

        return {
            'total_stocks': len(results),
            'total_trades': total_trades,
            'avg_win_rate': avg_win_rate,
            'portfolio_return': portfolio_return,
            'avg_return_per_stock': avg_return_per_stock,
            'avg_sharpe': avg_sharpe,
            'winning_trades': total_winning,
            'losing_trades': total_losing
        }

    fixed_portfolio = calculate_portfolio_metrics(fixed_results)
    dynamic_portfolio = calculate_portfolio_metrics(dynamic_results)

    # Print comparison table
    print(f"{'Metric':<30} {'Fixed (1.0x)':<20} {'Dynamic (Tier)':<20} {'Change':<15}")
    print("-" * 85)
    print(f"{'Stocks Tested':<30} {fixed_portfolio['total_stocks']:<20} {dynamic_portfolio['total_stocks']:<20} {'-':<15}")
    print(f"{'Total Trades':<30} {fixed_portfolio['total_trades']:<20} {dynamic_portfolio['total_trades']:<20} {'-':<15}")
    print(f"{'Winning Trades':<30} {fixed_portfolio['winning_trades']:<20} {dynamic_portfolio['winning_trades']:<20} {'-':<15}")
    print(f"{'Losing Trades':<30} {fixed_portfolio['losing_trades']:<20} {dynamic_portfolio['losing_trades']:<20} {'-':<15}")

    win_rate_change = dynamic_portfolio['avg_win_rate'] - fixed_portfolio['avg_win_rate']
    print(f"{'Avg Win Rate':<30} {fixed_portfolio['avg_win_rate']:<19.1f}% {dynamic_portfolio['avg_win_rate']:<19.1f}% {win_rate_change:+.1f}%")

    portfolio_return_change = dynamic_portfolio['portfolio_return'] - fixed_portfolio['portfolio_return']
    print(f"{'Portfolio Return (Sum)':<30} {fixed_portfolio['portfolio_return']:<19.2f}% {dynamic_portfolio['portfolio_return']:<19.2f}% {portfolio_return_change:+.2f}%")

    avg_return_change = dynamic_portfolio['avg_return_per_stock'] - fixed_portfolio['avg_return_per_stock']
    print(f"{'Avg Return Per Stock':<30} {fixed_portfolio['avg_return_per_stock']:<19.2f}% {dynamic_portfolio['avg_return_per_stock']:<19.2f}% {avg_return_change:+.2f}%")

    sharpe_change = dynamic_portfolio['avg_sharpe'] - fixed_portfolio['avg_sharpe']
    print(f"{'Avg Sharpe Ratio':<30} {fixed_portfolio['avg_sharpe']:<19.2f} {dynamic_portfolio['avg_sharpe']:<19.2f} {sharpe_change:+.2f}")

    print("=" * 85)

    # Detailed stock-by-stock comparison
    print()
    print("=" * 70)
    print("STOCK-BY-STOCK COMPARISON")
    print("=" * 70)
    print()

    print(f"{'Stock':<8} {'Tier':<6} {'Size':<8} {'Return (Fixed)':<16} {'Return (Dynamic)':<16} {'Change':<10}")
    print("-" * 70)

    for fixed_res, dynamic_res in zip(fixed_results, dynamic_results):
        ticker = fixed_res['ticker']
        tier = dynamic_res['tier']
        size = dynamic_res['position_size']
        return_change = dynamic_res['total_return'] - fixed_res['total_return']

        print(f"{ticker:<8} {tier:<6} {size:<7.2f}x {fixed_res['total_return']:>+14.2f}% {dynamic_res['total_return']:>+14.2f}% {return_change:>+8.2f}%")

    print("=" * 70)

    # Conclusion
    print()
    print("CONCLUSION:")
    print()

    if portfolio_return_change > 5 and win_rate_change >= 0:
        print("[SUCCESS] Dynamic position sizing significantly improves portfolio returns!")
        print(f"Portfolio return improvement: {portfolio_return_change:+.2f}%")
        print(f"Win rate change: {win_rate_change:+.1f}%")
        print("Recommendation: DEPLOY dynamic position sizing in production")
    elif portfolio_return_change > 2:
        print("[GOOD] Dynamic position sizing improves portfolio performance")
        print(f"Portfolio return improvement: {portfolio_return_change:+.2f}%")
        print("Recommendation: Use dynamic position sizing for better returns")
    elif abs(portfolio_return_change) < 2:
        print("[NEUTRAL] Minimal impact from dynamic position sizing")
        print("Consider using for risk management even without return improvement")
    else:
        print("[NEGATIVE] Dynamic position sizing reduces portfolio returns")
        print(f"Portfolio return change: {portfolio_return_change:+.2f}%")
        print("Recommendation: Stick with fixed position sizing")

    print()
    print("KEY INSIGHTS:")
    print(f"- Fixed sizing allocates equal capital to all stocks (good and bad)")
    print(f"- Dynamic sizing concentrates capital in Tier 1 stocks (80%+ win rate)")
    print(f"- Tier 1: {len([r for r in dynamic_results if r['tier'] == 1])} stocks at 1.0x position")
    print(f"- Tier 2: {len([r for r in dynamic_results if r['tier'] == 2])} stocks at 0.75x position")
    print(f"- Tier 3: {len([r for r in dynamic_results if r['tier'] == 3])} stocks at 0.5x position")

    print()
    print("=" * 70)


if __name__ == "__main__":
    compare_fixed_vs_dynamic_sizing()

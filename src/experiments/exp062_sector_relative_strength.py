"""
EXPERIMENT: EXP-062
Date: 2025-11-17
Objective: Enhance signal quality using sector relative strength

RESEARCH MOTIVATION:
After 61 experiments, what works = QUALITY FILTERING, not more constraints
- v15.0 quality scoring: FAILED (-4.2pp)
- v16.0 VIX regime: FAILED (+0.0pp)
- EXP-061 gap filter: FAILED (-5.7pp)

THEORY FOUNDATION:
Stocks that are OVERSOLD but OUTPERFORMING their sector = higher-quality signals
- Mean reversion + relative strength = dual confirmation
- Sector weakness can drag down even strong stocks
- Relative strength shows underlying demand

HYPOTHESIS:
Filtering for stocks with positive sector-relative performance will improve win rate
- Only take signals when stock is beating its sector (20-day relative performance)
- Expected improvement: +3-5pp win rate
- Trade reduction: 20-30% (acceptable)

METHODOLOGY:
1. Calculate sector performance (SPY sectors: XLF, XLK, XLV, XLE, XLY, XLP, XLI, XLB, XLU)
2. Calculate stock vs sector relative performance (20-day)
3. Test baseline vs sector-enhanced signals
4. Validate on full 45-stock portfolio

SECTOR MAPPINGS:
- XLF (Finance): AXP, SCHW, AIG, USB, JPM, MS, PNC
- XLK (Technology): NVDA, AVGO, V, MA, ORCL, MRVL, MSFT, QCOM, AMAT, ADI, NOW, CRM, ADBE
- XLV (Healthcare): ABBV, GILD, JNJ, PFE, INTU
- XLE (Energy): EOG, COP, SLB, XOM, MPC
- XLY (Consumer Discretionary): TGT, LOW, NKE
- XLI (Industrials): LMT, CAT, HON
- XLB (Materials): MLM, APD
- XLRE (Real Estate): EXR, IDXX
- XLP (Consumer Staples): WMT, CVS
- Uncategorized: TXN (use XLK), KLAC (use XLK), INSM (use XLV), ROAD (use XLI), ECL (use XLB)

EXPECTED OUTCOME:
- Confirmation of sector-relative strength filter (+3-5pp win rate)
- Deployment to v16.0-SECTOR-ENHANCED
- OR: Discovery that relative strength doesn't matter for mean reversion
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from src.config.mean_reversion_params import get_params, get_all_tickers


# Sector ETF mappings
SECTOR_ETFS = {
    'XLF': ['AXP', 'SCHW', 'AIG', 'USB', 'JPM', 'MS', 'PNC'],
    'XLK': ['NVDA', 'AVGO', 'V', 'MA', 'ORCL', 'MRVL', 'MSFT', 'QCOM', 'AMAT', 'ADI', 'NOW', 'CRM', 'ADBE', 'TXN', 'KLAC'],
    'XLV': ['ABBV', 'GILD', 'JNJ', 'PFE', 'INTU', 'INSM'],
    'XLE': ['EOG', 'COP', 'SLB', 'XOM', 'MPC'],
    'XLY': ['TGT', 'LOW'],
    'XLI': ['LMT', 'CAT', 'ROAD'],
    'XLB': ['MLM', 'APD', 'ECL'],
    'XLRE': ['EXR', 'IDXX'],
    'XLP': ['WMT', 'CVS']
}

# Reverse mapping: ticker -> sector ETF
TICKER_TO_SECTOR = {}
for sector, tickers in SECTOR_ETFS.items():
    for ticker in tickers:
        TICKER_TO_SECTOR[ticker] = sector


def get_sector_for_ticker(ticker: str) -> str:
    """Get sector ETF for a ticker."""
    return TICKER_TO_SECTOR.get(ticker, 'XLK')  # Default to tech


def calculate_relative_strength(stock_data: pd.DataFrame, sector_data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate stock performance relative to sector.

    Returns:
        DataFrame with 'relative_strength' column (positive = outperforming sector)
    """
    data = stock_data.copy()

    # Calculate returns
    data['stock_return'] = data['Close'].pct_change(window)

    # Align sector data
    sector_aligned = sector_data.reindex(data.index, method='ffill')
    data['sector_return'] = sector_aligned['Close'].pct_change(window)

    # Relative strength = stock return - sector return
    data['relative_strength'] = data['stock_return'] - data['sector_return']

    return data


def validate_sector_enhancement(ticker: str, start_date: str, end_date: str) -> dict:
    """
    Validate sector relative strength enhancement on a single stock.

    Returns:
        Comparison of baseline vs sector-enhanced
    """
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    try:
        # Fetch stock data
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
        if len(data) < 60:
            return None

        # Fetch sector data
        sector_etf = get_sector_for_ticker(ticker)
        sector_data = fetcher.fetch_stock_data(sector_etf, start_date=buffer_start, end_date=end_date)

        # Calculate relative strength
        data = calculate_relative_strength(data, sector_data, window=20)

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

    # Get parameters
    params = get_params(ticker)

    # Engineer features
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)
    enriched_data['relative_strength'] = data['relative_strength']

    # Detect signals
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

    earnings_fetcher = EarningsCalendarFetcher(exclusion_days_before=3, exclusion_days_after=3)
    signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

    # BASELINE: No sector filter
    backtester = MeanReversionBacktester(
        initial_capital=10000,
        exit_strategy='time_decay',
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=3
    )

    try:
        baseline_results = backtester.backtest(signals)

        if not baseline_results or baseline_results.get('total_trades', 0) < 5:
            return None

        # SECTOR-ENHANCED: Only signals with positive relative strength
        sector_signals = signals.copy()
        sector_signals['has_relative_strength'] = enriched_data['relative_strength'] > 0
        sector_signals.loc[sector_signals['has_relative_strength'] == False, 'panic_sell'] = 0

        sector_results = backtester.backtest(sector_signals)

        if not sector_results or sector_results.get('total_trades', 0) < 3:
            return {
                'ticker': ticker,
                'sector': sector_etf,
                'baseline_wr': baseline_results['win_rate'],
                'baseline_trades': baseline_results['total_trades'],
                'sector_wr': None,
                'sector_trades': 0,
                'improvement': None,
                'trade_reduction': 100.0
            }

        baseline_wr = baseline_results['win_rate']
        sector_wr = sector_results['win_rate']
        baseline_trades = baseline_results['total_trades']
        sector_trades = sector_results['total_trades']

    except Exception as e:
        return None

    return {
        'ticker': ticker,
        'sector': sector_etf,
        'baseline_wr': baseline_wr,
        'baseline_trades': len(baseline_trades),
        'sector_wr': sector_wr,
        'sector_trades': len(sector_trades),
        'improvement': sector_wr - baseline_wr,
        'trade_reduction': (1 - len(sector_trades) / len(baseline_trades)) * 100 if len(baseline_trades) > 0 else 0
    }


def run_exp062_sector_relative_strength(period: str = "3y"):
    """
    Validate sector relative strength enhancement across full portfolio.

    Args:
        period: Backtest period
    """
    print("[SECTOR RELATIVE STRENGTH] Comprehensive validation")
    print("This will take 15-20 minutes...\n")

    print("=" * 70)
    print("EXP-062: SECTOR RELATIVE STRENGTH ENHANCEMENT VALIDATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("HYPOTHESIS:")
    print("Stocks showing oversold conditions BUT outperforming their sector")
    print("should have higher-quality mean reversion signals")
    print()

    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    if period == "3y":
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    # Get all tickers
    all_tickers = get_all_tickers()

    print(f"Testing {len(all_tickers)} stocks")
    print("This will take 15-20 minutes...\n")

    results = []
    for ticker in all_tickers:
        try:
            print(f"Validating {ticker}...", end=" ")
            result = validate_sector_enhancement(ticker, start_date, end_date)

            if result is None:
                print("No data")
                continue

            results.append(result)

            baseline_str = f"Baseline: {result['baseline_wr']:.1f}% ({result['baseline_trades']} trades)"
            sector_str = f"Sector: {result['sector_wr']:.1f}% ({result['sector_trades']} trades)"
            improvement_str = f"Improvement: {result['improvement']:+.1f}pp"

            print(f"{baseline_str} | {sector_str} | {improvement_str}")

        except Exception as e:
            print(f"Error: {e}")
            continue

    # Aggregate analysis
    print()
    print("=" * 70)
    print("SECTOR RELATIVE STRENGTH VALIDATION RESULTS")
    print("=" * 70)
    print()

    print(f"Total stocks tested: {len(all_tickers)}")
    print(f"Stocks with results: {len(results)}")
    print()

    if len(results) == 0:
        print("ERROR: No stocks returned valid results")
        print("Check for data availability issues or API errors")
        return results

    # Aggregate metrics
    total_baseline_trades = sum(r['baseline_trades'] for r in results)
    total_sector_trades = sum(r['sector_trades'] for r in results if r['sector_trades'] is not None and r['sector_trades'] > 0)

    baseline_wins = sum(r['baseline_wr'] * r['baseline_trades'] / 100 for r in results if r['baseline_wr'] is not None)
    sector_wins = sum(r['sector_wr'] * r['sector_trades'] / 100 for r in results if r['sector_wr'] is not None and r['sector_trades'] is not None)

    baseline_wr = baseline_wins / total_baseline_trades * 100 if total_baseline_trades > 0 else 0
    sector_wr = sector_wins / total_sector_trades * 100 if total_sector_trades > 0 else 0

    trade_reduction = (1 - total_sector_trades / total_baseline_trades) * 100 if total_baseline_trades > 0 else 0

    print("AGGREGATE PERFORMANCE:")
    print(f"  Baseline win rate: {baseline_wr:.1f}%")
    print(f"  Sector-enhanced win rate: {sector_wr:.1f}%")
    print(f"  Improvement: {sector_wr - baseline_wr:+.1f}pp")
    print()
    print(f"  Baseline trades/stock: {total_baseline_trades / len(results):.1f}")
    print(f"  Sector trades/stock: {total_sector_trades / len([r for r in results if r['sector_trades'] is not None and r['sector_trades'] > 0]):.1f}" if len([r for r in results if r['sector_trades'] is not None and r['sector_trades'] > 0]) > 0 else "  Sector trades/stock: 0.0")
    print(f"  Trade reduction: {trade_reduction:.1f}%")
    print()

    # Top performers (filter out None improvements)
    valid_results = [r for r in results if r['improvement'] is not None]
    sorted_results = sorted(valid_results, key=lambda x: x['improvement'], reverse=True)

    print("TOP 10 STOCKS BY SECTOR ENHANCEMENT:")
    for i, r in enumerate(sorted_results[:10], 1):
        print(f"  {i}. {r['ticker']} ({r['sector']}): {r['baseline_wr']:.1f}% -> {r['sector_wr']:.1f}% ({r['improvement']:+.1f}pp)")
    print()

    print("BOTTOM 10 STOCKS (SECTOR FILTER HURT):")
    for i, r in enumerate(sorted_results[-10:], 1):
        print(f"  {i}. {r['ticker']} ({r['sector']}): {r['baseline_wr']:.1f}% -> {r['sector_wr']:.1f}% ({r['improvement']:+.1f}pp)")
    print()

    # Decision
    print("=" * 70)
    print("DEPLOYMENT DECISION:")
    print("=" * 70)

    if sector_wr - baseline_wr >= 3.0 and trade_reduction <= 40:
        print(f"APPROVED: Sector enhancement provides {sector_wr - baseline_wr:+.1f}pp improvement")
        print(f"Trade reduction of {trade_reduction:.1f}% is acceptable")
        print("Ready for deployment to v16.0-SECTOR-ENHANCED")
    elif sector_wr - baseline_wr >= 1.0:
        print(f"MARGINAL: Only {sector_wr - baseline_wr:+.1f}pp improvement")
        print("Needs further optimization before deployment")
    else:
        print(f"REJECTED: Sector enhancement shows {sector_wr - baseline_wr:+.1f}pp change")
        print("No benefit to deployment")

    print()
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_exp062_sector_relative_strength()

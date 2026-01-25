"""
EXPERIMENT: EXP-064
Date: 2025-11-17
Objective: Screen underrepresented sectors for Tier A stock candidates

RESEARCH MOTIVATION:
Current portfolio (46 stocks) has significant sector concentration:
- Semiconductors: 8 stocks (17%)
- Finance: 7 stocks (15%)
- Tech/Software: 6 stocks (13%)

But MISSING entirely:
- Utilities: 0 stocks
- Telecom: 0 stocks

UNDERREPRESENTED:
- REIT: 1 stock (should have 3-5 for diversification)
- Payments: 2 stocks (V, MA dominate but room for more)
- Healthcare services: 0 stocks (only pharma/devices)

THEORY FOUNDATION:
Sector diversification reduces concentration risk and provides:
1. Uncorrelated trading opportunities (utilities vs tech have different cycles)
2. Defensive exposure (utilities/telecom stable during market stress)
3. Income generation (REITs/telecom high dividend = price stability = better mean reversion)

HYPOTHESIS:
Defensive sectors (utilities, telecom, REITs) will show strong mean reversion:
- Lower volatility = more predictable rebounds
- High dividend yield = price support
- Stable businesses = less noise in signals
- Expected: 2-4 new Tier A stocks (>70% win rate)

TEST CANDIDATES:
Utilities (5 stocks):
- NEE (NextEra Energy) - Largest utility, renewable leader
- DUK (Duke Energy) - Traditional utility, stable dividend
- SO (Southern Company) - Southeastern utility, consistent
- D (Dominion Energy) - Mid-Atlantic utility
- AEP (American Electric Power) - Midwest utility

Telecom (3 stocks):
- T (AT&T) - High dividend, stable
- VZ (Verizon) - Industry leader, 5G deployment
- TMUS (T-Mobile) - Growth telecom, aggressive pricing

REITs (4 stocks):
- PLD (Prologis) - Logistics/warehouses (e-commerce tailwind)
- AMT (American Tower) - Cell towers (5G tailwind)
- EQIX (Equinix) - Data centers (cloud tailwind)
- PSA (Public Storage) - Self-storage (complements EXR)

Healthcare Services (3 stocks):
- UNH (UnitedHealth) - Largest health insurer
- CI (Cigna) - Health services
- HCA (HCA Healthcare) - Hospital operator

Aerospace/Defense (2 stocks):
- NOC (Northrop Grumman) - Complement to LMT
- LHX (L3Harris) - Defense electronics

TOTAL: 17 candidates
Expected time: ~20 minutes (17 stocks × 1 min = 17 min)

METHODOLOGY:
1. Test each candidate with DEFAULT parameters (z=1.5, rsi=35, vol=1.3)
2. Run 3-year backtest
3. Classify as Tier A (>70%), B (55-70%), or C (<55%)
4. Deploy Tier A candidates immediately
5. Save Tier B candidates for future parameter optimization

EXPECTED OUTCOME:
- 2-4 new Tier A stocks from defensive sectors
- Better sector diversification (reduce concentration risk)
- More trading opportunities in different market cycles
- Validation that mean reversion works across ALL stable sectors
"""

import sys
import os
import pandas as pd
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester


def test_stock(ticker: str, start_date: str, end_date: str):
    """Test a stock with default mean reversion parameters."""
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
        if len(data) < 60:
            return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

    # Engineer features
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)

    # Detect signals with DEFAULT parameters
    detector = MeanReversionDetector(
        z_score_threshold=1.5,
        rsi_oversold=35,
        rsi_overbought=65,
        volume_multiplier=1.3,
        price_drop_threshold=-1.5
    )

    signals = detector.detect_overcorrections(enriched_data)
    signals = detector.calculate_reversion_targets(signals)

    # Apply filters
    regime_detector = MarketRegimeDetector()
    signals = add_regime_filter_to_signals(signals, regime_detector)

    earnings_fetcher = EarningsCalendarFetcher(exclusion_days_before=3, exclusion_days_after=3)
    signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

    # Backtest
    backtester = MeanReversionBacktester(
        initial_capital=10000,
        exit_strategy='time_decay',
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=3
    )

    try:
        result = backtester.backtest(signals)
        if result is None or result.get('total_trades', 0) < 5:
            return None
        return result
    except Exception as e:
        print(f"  [ERROR] Backtest failed: {e}")
        return None


def run_exp064_sector_diversification():
    """Screen underrepresented sectors for Tier A candidates."""
    print("="*70)
    print("EXP-064: SECTOR DIVERSIFICATION SCREENING")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("OBJECTIVE: Find Tier A stocks in underrepresented sectors")
    print("Target: 2-4 new stocks with >70% win rate")
    print("Focus: Utilities, Telecom, REITs, Healthcare Services")
    print()

    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')

    candidates = {
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP'],
        'Telecom': ['T', 'VZ', 'TMUS'],
        'REIT': ['PLD', 'AMT', 'EQIX', 'PSA'],
        'Healthcare Services': ['UNH', 'CI', 'HCA'],
        'Aerospace/Defense': ['NOC', 'LHX']
    }

    all_results = []

    for sector, tickers in candidates.items():
        print()
        print(f"{'='*70}")
        print(f"SECTOR: {sector} ({len(tickers)} candidates)")
        print(f"{'='*70}")
        print()

        for ticker in tickers:
            print(f"Testing {ticker}...", end=" ")
            result = test_stock(ticker, start_date, end_date)

            if result is None:
                print("No data / insufficient trades")
                continue

            win_rate = result['win_rate']
            total_return = result['total_return']
            trades = result['total_trades']
            sharpe = result['sharpe_ratio']

            tier = 'A' if win_rate >= 70 else ('B' if win_rate >= 55 else 'C')

            print(f"{win_rate:.1f}% WR, {total_return:+.1f}% return, {trades} trades, Sharpe {sharpe:.2f} [TIER {tier}]")

            all_results.append({
                'ticker': ticker,
                'sector': sector,
                'win_rate': win_rate,
                'total_return': total_return,
                'total_trades': trades,
                'sharpe_ratio': sharpe,
                'tier': tier
            })

    # Summary
    print()
    print("="*70)
    print("SECTOR DIVERSIFICATION SCREENING RESULTS")
    print("="*70)
    print()

    tier_a = [r for r in all_results if r['tier'] == 'A']
    tier_b = [r for r in all_results if r['tier'] == 'B']
    tier_c = [r for r in all_results if r['tier'] == 'C']

    print(f"TIER A (>70% win rate): {len(tier_a)} stocks")
    for r in sorted(tier_a, key=lambda x: x['win_rate'], reverse=True):
        print(f"  {r['ticker']} ({r['sector']}): {r['win_rate']:.1f}% WR, {r['total_return']:+.1f}% return, {r['total_trades']} trades")
    print()

    print(f"TIER B (55-70% win rate): {len(tier_b)} stocks")
    for r in sorted(tier_b, key=lambda x: x['win_rate'], reverse=True):
        print(f"  {r['ticker']} ({r['sector']}): {r['win_rate']:.1f}% WR, {r['total_return']:+.1f}% return")
    print()

    print(f"TIER C (<55% win rate): {len(tier_c)} stocks")
    for r in sorted(tier_c, key=lambda x: x['win_rate'], reverse=True):
        print(f"  {r['ticker']} ({r['sector']}): {r['win_rate']:.1f}% WR, {r['total_return']:+.1f}% return")
    print()

    # Deployment recommendation
    print("="*70)
    print("DEPLOYMENT RECOMMENDATION")
    print("="*70)
    print()

    if len(tier_a) > 0:
        print(f"APPROVED: {len(tier_a)} new Tier A stocks ready for deployment!")
        print()
        print("Add to mean_reversion_params.py:")
        for r in tier_a:
            print(f"  {r['ticker']}: z=1.5, rsi=35, vol=1.3 (default params)")
        print()
        print(f"Portfolio expansion: 46 -> {46 + len(tier_a)} stocks (+{len(tier_a)/46*100:.1f}%)")
    else:
        print("No Tier A stocks found. Consider:")
        print("1. Parameter optimization on Tier B candidates")
        print("2. Testing additional sectors")
    print()

    if len(tier_b) > 0:
        print(f"FUTURE OPTIMIZATION: {len(tier_b)} Tier B stocks")
        print("These candidates could reach Tier A with parameter tuning (EXP-037 style)")
        print()

    # Save results
    output_file = 'logs/experiments/exp064_sector_diversification.json'
    os.makedirs('logs/experiments', exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            'experiment_id': 'EXP-064',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'candidates_tested': len(all_results),
            'tier_a': len(tier_a),
            'tier_b': len(tier_b),
            'tier_c': len(tier_c),
            'results': all_results
        }, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    return all_results


if __name__ == "__main__":
    results = run_exp064_sector_diversification()

"""
EXPERIMENT: EXP-042
Date: 2025-11-17
Objective: Screen for new Tier A stocks to expand portfolio

MOTIVATION:
Current portfolio: 23 Tier A stocks
- Previous expansions: MSFT found via EXP-037 (77.8% win rate)
- More high-quality stocks likely exist
- Portfolio expansion = more trading opportunities

PROBLEM:
Limited portfolio coverage:
- Only 23 stocks actively monitored
- Many quality large-caps not tested (GOOGL, AMZN, META, etc.)
- Mid-cap opportunities unexplored
- Potential 5-10 additional Tier A candidates

HYPOTHESIS:
Screening quality stocks will find 3-5 new Tier A additions:
- Focus on large-cap stability (market cap >$100B)
- Strong fundamentals
- Moderate volatility (good for mean reversion)
- Expected: 3-5 stocks with >70% win rate

METHODOLOGY:
1. Test high-quality candidates:
   - Mega-cap tech: GOOGL, AMZN, META, NFLX, TSLA
   - Mega-cap finance: JPM, BAC, WFC, GS
   - Mega-cap healthcare: JNJ, UNH, PFE, TMO
   - Mega-cap consumer: PG, KO, PEP, WMT, COST

2. Run 3-year backtest (2022-2025) with optimized parameters
3. Classify as Tier A (>70%), B (55-70%), C (<55%)
4. Add Tier A stocks to portfolio
5. Optimize parameters for new Tier A stocks

TIER A REQUIREMENTS:
- Win rate: >70% (CRITICAL)
- Total trades: >5 (sufficient data)
- Total return: >5% (minimum profitability)
- Sharpe ratio: >3.0 (preferred)

EXPECTED OUTCOME:
- 3-5 new Tier A stocks identified
- Portfolio expansion: 23 → 26-28 stocks
- +15-25 trades/year from new stocks
- Increased diversification
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from common.config.mean_reversion_params import get_params


def test_stock_candidate(ticker: str, start_date: str, end_date: str) -> dict:
    """
    Test a stock candidate for Tier A qualification.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date

    Returns:
        Test results
    """
    print(f"\nTesting {ticker}...")

    # Fetch data
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
    except Exception as e:
        print(f"  [ERROR] Failed to fetch: {e}")
        return None

    if len(data) < 60:
        print(f"  [SKIP] Insufficient data")
        return None

    # Use DEFAULT parameters for initial test
    params = get_params('DEFAULT')

    # Engineer features
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)

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

    # Backtest
    backtester = MeanReversionBacktester(
        initial_capital=10000,
        exit_strategy='time_decay',
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=3
    )

    try:
        results = backtester.backtest(signals)

        if not results or results.get('total_trades', 0) < 5:
            print(f"  [SKIP] Insufficient trades ({results.get('total_trades', 0) if results else 0})")
            return None

        win_rate = results.get('win_rate', 0)
        total_return = results.get('total_return', 0)
        total_trades = results.get('total_trades', 0)
        sharpe = results.get('sharpe_ratio', 0)
        avg_gain = results.get('avg_gain', 0)

        # Classify tier
        if win_rate >= 70.0:
            tier = 'A'
            tier_label = 'TIER A'
        elif win_rate >= 55.0:
            tier = 'B'
            tier_label = 'TIER B'
        else:
            tier = 'C'
            tier_label = 'TIER C'

        print(f"  [{tier_label}] Win: {win_rate:.1f}%, Return: {total_return:+.1f}%, "
              f"Sharpe: {sharpe:.2f}, Trades: {total_trades}")

        return {
            'ticker': ticker,
            'tier': tier,
            'win_rate': win_rate / 100.0,
            'total_return': total_return,
            'total_trades': total_trades,
            'sharpe_ratio': sharpe,
            'avg_gain': avg_gain,
            'baseline_params': params
        }

    except Exception as e:
        print(f"  [ERROR] Backtest failed: {e}")
        return None


def run_exp042_tier_a_screening():
    """
    Screen candidates for Tier A qualification.
    """
    print("=" * 70)
    print("EXP-042: TIER A STOCK SCREENING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Find new Tier A stocks (>70% win rate)")
    print("Current portfolio: 23 Tier A stocks")
    print("Expected: 3-5 new Tier A additions")
    print()

    # Candidate stocks
    candidates = {
        'Mega-cap Tech': ['GOOGL', 'AMZN', 'META', 'NFLX', 'TSLA', 'AMD', 'QCOM'],
        'Mega-cap Finance': ['JPM', 'BAC', 'WFC', 'C', 'MS'],
        'Mega-cap Healthcare': ['JNJ', 'UNH', 'PFE', 'TMO', 'DHR', 'LLY'],
        'Mega-cap Consumer': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'MCD'],
        'Semiconductors': ['AMAT', 'LRCX', 'ADI', 'MU', 'INTC'],
        'Other Large-cap': ['CRM', 'ADBE', 'NOW', 'SHOP', 'SNOW']
    }

    all_candidates = []
    for category, tickers in candidates.items():
        all_candidates.extend(tickers)

    print(f"Testing {len(all_candidates)} candidates across {len(candidates)} categories")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    results_by_category = {cat: [] for cat in candidates.keys()}
    all_results = []

    for category, tickers in candidates.items():
        print(f"\n{'='*70}")
        print(f"{category.upper()}")
        print(f"{'='*70}")

        for ticker in tickers:
            result = test_stock_candidate(ticker, start_date, end_date)
            if result:
                results_by_category[category].append(result)
                all_results.append(result)

    # Aggregate results
    print()
    print("=" * 70)
    print("SCREENING RESULTS")
    print("=" * 70)
    print()

    tier_a_stocks = [r for r in all_results if r['tier'] == 'A']
    tier_b_stocks = [r for r in all_results if r['tier'] == 'B']
    tier_c_stocks = [r for r in all_results if r['tier'] == 'C']

    print(f"TIER A (>70% win rate): {len(tier_a_stocks)} stocks")
    if tier_a_stocks:
        print()
        print(f"{'Ticker':<8} {'Win Rate':<12} {'Return':<12} {'Sharpe':<10} {'Trades':<10}")
        print("-" * 70)
        for r in sorted(tier_a_stocks, key=lambda x: x['win_rate'], reverse=True):
            print(f"{r['ticker']:<8} {r['win_rate']*100:>10.1f}% {r['total_return']:>10.1f}% "
                  f"{r['sharpe_ratio']:>8.2f} {r['total_trades']:>8}")

    print()
    print(f"TIER B (55-70% win rate): {len(tier_b_stocks)} stocks")
    if tier_b_stocks:
        print("  " + ", ".join([r['ticker'] for r in tier_b_stocks]))

    print()
    print(f"TIER C (<55% win rate): {len(tier_c_stocks)} stocks")
    if tier_c_stocks:
        print("  " + ", ".join([r['ticker'] for r in tier_c_stocks]))

    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print()

    if tier_a_stocks:
        print(f"[SUCCESS] Found {len(tier_a_stocks)} new Tier A stocks!")
        print()
        print("NEXT STEPS:")
        print(f"1. Add {len(tier_a_stocks)} stocks to mean_reversion_params.py")
        print("2. Run parameter optimization (EXP-037 style) for each stock")
        print("3. Update portfolio: 23 → " + str(23 + len(tier_a_stocks)) + " stocks")
        print(f"4. Expected trades/year: +{len(tier_a_stocks) * 10}-{len(tier_a_stocks) * 15}")
    else:
        print("[NO ADDITIONS] No new Tier A stocks found")
        print("Current 23 stocks remain optimal portfolio")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-042',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'candidates_tested': len(all_results),
        'tier_a_found': len(tier_a_stocks),
        'tier_b_found': len(tier_b_stocks),
        'tier_c_found': len(tier_c_stocks),
        'tier_a_stocks': [{**r, 'win_rate': float(r['win_rate'])} for r in tier_a_stocks],
        'all_results': [{**r, 'win_rate': float(r['win_rate'])} for r in all_results]
    }

    results_file = os.path.join(results_dir, 'exp042_tier_a_screening.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending screening report email...")
            notifier.send_experiment_report('EXP-042', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-042 Tier A screening."""

    print("\n[PORTFOLIO EXPANSION] Screening for new Tier A stocks")
    print("This will take 15-20 minutes...")
    print()

    results = run_exp042_tier_a_screening()

    print("\nTIER A STOCK SCREENING COMPLETE")

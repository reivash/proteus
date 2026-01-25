"""
EXPERIMENT: EXP-050
Date: 2025-11-17
Objective: Expand Tier A universe with 5-10 additional high-quality stocks

MOTIVATION:
Current portfolio: 27 curated Tier A stocks (79.3% win rate, 19.2% avg return)
- System is PROVEN - 79.3% win rate validates our approach
- Quality bar established - 70%+ win rate threshold
- Limited trading opportunities - ~200 trades/year
- Could handle larger portfolio with same quality standards

OPPORTUNITY:
Expand universe to find more high-quality stocks:
- Each new stock adds ~7-10 trades/year
- 5-10 new stocks = +35-100 trades/year
- +20-50% more total return with same win rate
- Multiplicative impact vs incremental optimizations

HYPOTHESIS:
Can find 5-10 additional stocks meeting quality criteria:
- Win rate >= 70% (our proven threshold)
- Min 5 trades over test period (sufficient data)
- Similar market cap/liquidity to current portfolio
- Diverse sectors (avoid concentration risk)
- Expected: +10-30% total portfolio return improvement

METHODOLOGY:
1. Screen S&P 500 + NASDAQ 100 for candidates:
   - Market cap: $10B+ (liquidity requirement)
   - Avg volume: 1M+ shares/day (tradeable)
   - Price: $50+ (avoid penny stocks)
   - Exclude current 27 stocks

2. Test each candidate on 2022-2025 data:
   - Use baseline parameters initially
   - Apply all filters (earnings, regime)
   - Measure: win rate, total trades, avg return
   - Threshold: 60%+ win rate (for optimization potential)

3. Optimize promising candidates:
   - Grid search if baseline >= 60%
   - Optimize to 70%+ win rate
   - Document optimal parameters

4. Deploy if find >= 5 stocks meeting 70%+ threshold

CANDIDATE SECTORS TO EXPLORE:
- Financial Services: GS, MS, AIG, BLK, SCHW (we have JPM, AXP)
- Technology: AMD, INTC, CRM, ADBE, CSCO (we have NVDA, MSFT, ORCL, etc.)
- Healthcare: UNH, CVS, AMGN, BMY, LLY (we have ABBV, GILD, JNJ, PFE)
- Consumer: COST, TGT, HD, NKE (we have WMT)
- Industrials: CAT, DE, GE, HON (we have MLM, EXR, ROAD)
- Energy: XOM, CVX, COP (we have EOG)

EXPECTED OUTCOME:
- 5-10 new Tier A stocks
- All with 70%+ win rate
- +35-100 trades/year
- +10-30% portfolio return improvement
- Documented parameters for each
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from itertools import product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from common.config.mean_reversion_params import get_params, get_all_tickers


# Candidate stocks to screen (high-quality large caps not in current portfolio)
CANDIDATE_STOCKS = [
    # Financials
    'GS', 'MS', 'BLK', 'SCHW', 'AIG', 'USB', 'PNC', 'TFC',
    # Technology
    'AMD', 'INTC', 'CRM', 'ADBE', 'CSCO', 'NXPI', 'LRCX', 'SNPS',
    # Healthcare
    'UNH', 'CVS', 'AMGN', 'BMY', 'LLY', 'REGN', 'VRTX', 'ISRG',
    # Consumer
    'COST', 'TGT', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'DIS',
    # Industrials
    'CAT', 'DE', 'GE', 'HON', 'UPS', 'RTX', 'LMT', 'BA',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'PXD', 'MPC', 'PSX',
    # Materials
    'LIN', 'APD', 'ECL', 'NEM', 'FCX'
]


def screen_candidate(ticker: str, start_date: str, end_date: str) -> dict:
    """
    Screen a candidate stock with baseline parameters.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date

    Returns:
        Screening results or None
    """
    print(f"  Screening {ticker}...", end=" ")

    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
    except Exception as e:
        print(f"[SKIP] Fetch failed")
        return None

    if len(data) < 60:
        print(f"[SKIP] Insufficient data")
        return None

    # Use baseline parameters for initial screen
    baseline_params = {
        'z_score_threshold': 1.5,
        'rsi_oversold': 35,
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5
    }

    # Engineer features
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)

    # Detect signals
    detector = MeanReversionDetector(
        z_score_threshold=baseline_params['z_score_threshold'],
        rsi_oversold=baseline_params['rsi_oversold'],
        rsi_overbought=65,
        volume_multiplier=baseline_params['volume_multiplier'],
        price_drop_threshold=baseline_params['price_drop_threshold']
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
            print(f"[SKIP] Insufficient trades ({results.get('total_trades', 0)})")
            return None

        win_rate = results['win_rate']
        total_return = results['total_return']
        total_trades = results['total_trades']
        sharpe = results.get('sharpe_ratio', 0)

        # Threshold for optimization: 60%+ win rate (potential to reach 70%+)
        if win_rate >= 60.0:
            print(f"[CANDIDATE] WR: {win_rate:.1f}%, Return: {total_return:+.1f}%, Trades: {total_trades}")
            return {
                'ticker': ticker,
                'baseline_win_rate': win_rate,
                'baseline_return': total_return,
                'total_trades': total_trades,
                'sharpe_ratio': sharpe,
                'needs_optimization': win_rate < 70.0
            }
        else:
            print(f"[REJECT] WR: {win_rate:.1f}% (below 60% threshold)")
            return None

    except Exception as e:
        print(f"[ERROR] {str(e)[:50]}")
        return None


def optimize_candidate(ticker: str, start_date: str, end_date: str) -> dict:
    """
    Optimize parameters for a promising candidate.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date

    Returns:
        Optimization results or None
    """
    print(f"\n  Optimizing {ticker}...")

    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
    except:
        return None

    # Grid search
    z_thresholds = [1.2, 1.5, 1.8, 2.0, 2.5]
    rsi_levels = [30, 32, 35, 38, 40]
    volume_multipliers = [1.2, 1.3, 1.5, 1.8]

    combinations = list(product(z_thresholds, rsi_levels, volume_multipliers))

    best_config = None
    best_win_rate = 0

    for z_thresh, rsi_level, vol_mult in combinations:
        engineer = TechnicalFeatureEngineer(fillna=True)
        enriched_data = engineer.engineer_features(data)

        detector = MeanReversionDetector(
            z_score_threshold=z_thresh,
            rsi_oversold=rsi_level,
            rsi_overbought=65,
            volume_multiplier=vol_mult,
            price_drop_threshold=-1.5
        )

        signals = detector.detect_overcorrections(enriched_data)
        signals = detector.calculate_reversion_targets(signals)

        regime_detector = MarketRegimeDetector()
        signals = add_regime_filter_to_signals(signals, regime_detector)

        earnings_fetcher = EarningsCalendarFetcher(exclusion_days_before=3, exclusion_days_after=3)
        signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

        backtester = MeanReversionBacktester(
            initial_capital=10000,
            exit_strategy='time_decay',
            profit_target=2.0,
            stop_loss=-2.0,
            max_hold_days=3
        )

        try:
            results = backtester.backtest(signals)

            if results and results.get('total_trades', 0) >= 5:
                win_rate = results['win_rate']
                total_return = results['total_return']

                if win_rate >= 70.0 and win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_config = {
                        'ticker': ticker,
                        'z_score_threshold': z_thresh,
                        'rsi_oversold': rsi_level,
                        'volume_multiplier': vol_mult,
                        'win_rate': win_rate,
                        'total_return': total_return,
                        'total_trades': results['total_trades'],
                        'sharpe_ratio': results.get('sharpe_ratio', 0)
                    }
        except:
            continue

    if best_config:
        print(f"    [SUCCESS] Optimized to {best_config['win_rate']:.1f}% win rate!")
        return best_config
    else:
        print(f"    [FAILED] Could not reach 70%+ win rate")
        return None


def run_exp050_tier_a_expansion():
    """
    Screen and optimize new Tier A stock candidates.
    """
    print("="*70)
    print("EXP-050: TIER A PORTFOLIO EXPANSION V2")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Find 5-10 additional high-quality Tier A stocks")
    print("Current portfolio: 27 stocks (79.3% win rate, 19.2% avg return)")
    print("Quality threshold: 70%+ win rate")
    print("Expected: +10-30% portfolio return improvement")
    print()

    # Filter out stocks we already have
    current_tickers = get_all_tickers()
    candidates = [t for t in CANDIDATE_STOCKS if t not in current_tickers]

    print(f"Screening {len(candidates)} candidates...")
    print("This will take 10-15 minutes...")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    # Phase 1: Initial screening
    print("PHASE 1: BASELINE SCREENING")
    print("="*70)

    promising_candidates = []

    for ticker in candidates:
        result = screen_candidate(ticker, start_date, end_date)
        if result:
            promising_candidates.append(result)

    print()
    print(f"Found {len(promising_candidates)} promising candidates (60%+ win rate)")
    print()

    # Phase 2: Optimization
    if promising_candidates:
        print("PHASE 2: PARAMETER OPTIMIZATION")
        print("="*70)

        optimized_stocks = []
        needs_optimization = [c for c in promising_candidates if c.get('needs_optimization', False)]
        already_qualified = [c for c in promising_candidates if not c.get('needs_optimization', False)]

        # Add already qualified stocks
        for stock in already_qualified:
            optimized_stocks.append(stock)
            print(f"  {stock['ticker']}: Already meets 70%+ threshold ({stock['baseline_win_rate']:.1f}%)")

        # Optimize those that need it
        for candidate in needs_optimization:
            result = optimize_candidate(candidate['ticker'], start_date, end_date)
            if result:
                optimized_stocks.append(result)

        print()
        print("="*70)
        print("EXPANSION RESULTS")
        print("="*70)
        print()

        print(f"New Tier A stocks found: {len(optimized_stocks)}")
        print()

        if optimized_stocks:
            print(f"{'Ticker':<8} {'Win Rate':<12} {'Return':<12} {'Trades':<10} {'Sharpe'}")
            print("-"*70)
            for stock in optimized_stocks:
                wr = stock.get('win_rate', stock.get('baseline_win_rate', 0))
                ret = stock.get('total_return', stock.get('baseline_return', 0))
                trades = stock.get('total_trades', 0)
                sharpe = stock.get('sharpe_ratio', 0)
                print(f"{stock['ticker']:<8} {wr:>10.1f}% {ret:>10.1f}% {trades:>8} {sharpe:>10.2f}")

            print()

            # Calculate projected impact
            new_trades = sum([s['total_trades'] for s in optimized_stocks])
            avg_new_return = np.mean([s.get('total_return', s.get('baseline_return', 0)) for s in optimized_stocks])

            print("PROJECTED IMPACT:")
            print(f"  Additional trades/year: ~{int(new_trades * 365 / (3*365))} (+{int(new_trades * 365 / (3*365) / 200 * 100)}%)")
            print(f"  New stocks avg return: {avg_new_return:.1f}%")
            print(f"  New portfolio size: {27 + len(optimized_stocks)} stocks")

            if len(optimized_stocks) >= 5:
                print()
                print("[SUCCESS] Found 5+ high-quality stocks! Deploy to production.")
            else:
                print()
                print(f"[PARTIAL] Found {len(optimized_stocks)} stocks. Consider deploying if quality is high.")

        # Save results
        results_dir = 'logs/experiments'
        os.makedirs(results_dir, exist_ok=True)

        exp_results = {
            'experiment_id': 'EXP-050',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_period': f'{start_date} to {end_date}',
            'candidates_screened': len(candidates),
            'promising_candidates': len(promising_candidates),
            'stocks_qualified': len(optimized_stocks),
            'new_stocks': optimized_stocks,
            'deploy': len(optimized_stocks) >= 5
        }

        results_file = os.path.join(results_dir, 'exp050_tier_a_expansion_v2.json')
        with open(results_file, 'w') as f:
            json.dump(exp_results, f, indent=2, default=float)

        print(f"\nResults saved to: {results_file}")

        # Send email
        try:
            from common.notifications.sendgrid_notifier import SendGridNotifier
            notifier = SendGridNotifier()
            if notifier.is_enabled():
                print("\nSending expansion report email...")
                notifier.send_experiment_report('EXP-050', exp_results)
            else:
                print("\n[INFO] Email not configured")
        except Exception as e:
            print(f"\n[WARNING] Email error: {e}")

        return exp_results

    else:
        print("[NO CANDIDATES] No stocks met 60%+ baseline threshold")
        return None


if __name__ == '__main__':
    """Run EXP-050 Tier A expansion."""

    print("\n[PORTFOLIO EXPANSION] Screening for new Tier A stocks")
    print("This will take 10-15 minutes...")
    print()

    results = run_exp050_tier_a_expansion()

    print("\n" + "="*70)
    print("TIER A EXPANSION COMPLETE")
    print("="*70)

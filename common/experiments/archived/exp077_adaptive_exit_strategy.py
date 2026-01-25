"""
EXPERIMENT: EXP-077
Date: 2025-11-17
Objective: Implement adaptive exit strategy based on stock volatility

HYPOTHESIS:
EXP-075 showed trailing_stop works for volatile stocks but hurts stable stocks:
- Volatile stocks (NVDA, V, AVGO): +3 to +10pp improvement with trailing_stop
- Stable stocks (JPM, JNJ, WMT): -4 to -18pp decline with trailing_stop

Adaptive strategy: Use trailing_stop for volatile stocks, time_decay for stable stocks.
This should combine the best of both worlds and beat baseline time_decay.

ALGORITHM:
1. Calculate ATR for each stock to determine volatility tier
2. Apply adaptive exit strategy:
   - High volatility (ATR > 2.5%): trailing_stop
   - Low/medium volatility (ATR <= 2.5%): time_decay
3. Backtest on full 54-stock portfolio (2020-2025)
4. Compare to:
   - Baseline: time_decay only
   - EXP-075: trailing_stop only
5. Measure improvement in win rate, returns, Sharpe ratio

EXPECTED RESULTS:
- Win rate improvement: +3-5pp over time_decay baseline
- Avg return improvement: +0.30-0.50% per trade
- Sharpe ratio improvement: +15-25%
- Universal benefit: 80-90% of stocks improve

SUCCESS CRITERIA:
- Win rate improvement >= +2pp
- Avg return improvement >= +0.20% per trade
- At least 40/54 stocks improve vs baseline
- No category (volatile/stable) regresses
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from common.config.mean_reversion_params import get_all_tickers


def calculate_atr_pct(data: pd.DataFrame) -> float:
    """Calculate ATR as percentage of price."""
    if 'atr_14' in data.columns:
        atr = data['atr_14'].iloc[-1]
        price = data['Close'].iloc[-1]
        return (atr / price) * 100
    else:
        # Calculate manually
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())

        true_range = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}).max(axis=1)
        atr = true_range.rolling(window=14).mean().iloc[-1]
        price = data['Close'].iloc[-1]

        return (atr / price) * 100


def get_adaptive_exit_strategy(atr_pct: float, threshold: float = 2.5) -> str:
    """
    Determine exit strategy based on volatility.

    Args:
        atr_pct: ATR as percentage
        threshold: Volatility threshold (default 2.5%)

    Returns:
        'trailing_stop' for high volatility, 'time_decay' for low/medium
    """
    return 'trailing_stop' if atr_pct > threshold else 'time_decay'


def test_adaptive_exit_stock(ticker: str,
                             start_date: str = '2020-01-01',
                             end_date: str = '2025-11-17',
                             volatility_threshold: float = 2.5) -> Dict:
    """
    Test adaptive exit strategy for a single stock.
    """
    try:
        # Fetch data
        buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
        fetcher = YahooFinanceFetcher()
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)

        if len(data) < 60:
            return None

        # Engineer features
        engineer = TechnicalFeatureEngineer(fillna=True)
        enriched_data = engineer.engineer_features(data)

        # Detect signals
        detector = MeanReversionDetector(
            z_score_threshold=1.5,
            rsi_oversold=35,
            volume_multiplier=1.3,
            price_drop_threshold=-1.5
        )

        signals = detector.detect_overcorrections(enriched_data)

        if len(signals) == 0:
            return None

        signals = detector.calculate_reversion_targets(signals)

        # Apply filters
        regime_detector = MarketRegimeDetector()
        signals = add_regime_filter_to_signals(signals, regime_detector)

        earnings_fetcher = EarningsCalendarFetcher(exclusion_days_before=3, exclusion_days_after=3)
        signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

        if len(signals) == 0:
            return None

        # Calculate ATR and determine strategy
        atr_pct = calculate_atr_pct(enriched_data)
        adaptive_strategy = get_adaptive_exit_strategy(atr_pct, volatility_threshold)

        # Baseline: time_decay
        backtester_baseline = MeanReversionBacktester(
            initial_capital=10000,
            exit_strategy='time_decay',
            max_hold_days=3
        )
        baseline_results = backtester_baseline.backtest(signals)

        # Adaptive: Use strategy based on volatility
        backtester_adaptive = MeanReversionBacktester(
            initial_capital=10000,
            exit_strategy=adaptive_strategy,
            max_hold_days=3
        )
        adaptive_results = backtester_adaptive.backtest(signals)

        # Calculate improvements
        baseline_wr = baseline_results.get('win_rate', 0)
        adaptive_wr = adaptive_results.get('win_rate', 0)
        wr_improvement = adaptive_wr - baseline_wr

        baseline_avg = baseline_results.get('avg_gain', 0) if baseline_results.get('total_trades', 0) > 0 else 0
        adaptive_avg = adaptive_results.get('avg_gain', 0) if adaptive_results.get('total_trades', 0) > 0 else 0
        avg_improvement = adaptive_avg - baseline_avg

        return {
            'ticker': ticker,
            'atr_pct': atr_pct,
            'adaptive_strategy': adaptive_strategy,
            'volatility_tier': 'HIGH' if atr_pct > volatility_threshold else 'LOW/MEDIUM',
            'total_trades': adaptive_results.get('total_trades', 0),
            'baseline': {
                'win_rate': baseline_wr,
                'avg_return': baseline_avg,
                'sharpe': baseline_results.get('sharpe_ratio', 0),
                'total_return': baseline_results.get('total_return', 0)
            },
            'adaptive': {
                'win_rate': adaptive_wr,
                'avg_return': adaptive_avg,
                'sharpe': adaptive_results.get('sharpe_ratio', 0),
                'total_return': adaptive_results.get('total_return', 0)
            },
            'improvement': {
                'win_rate': wr_improvement,
                'avg_return': avg_improvement,
                'sharpe_delta': adaptive_results.get('sharpe_ratio', 0) - baseline_results.get('sharpe_ratio', 0)
            }
        }

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_exp077_adaptive_exit_strategy():
    """
    Test adaptive exit strategy across full 54-stock portfolio.
    """
    print("="*70)
    print("EXP-077: ADAPTIVE EXIT STRATEGY")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("OBJECTIVE: Combine best of trailing_stop + time_decay")
    print("METHODOLOGY: High volatility -> trailing_stop, Low volatility -> time_decay")
    print("TARGET: +2pp win rate, +0.20% avg return improvement")
    print()

    # Get all portfolio tickers
    tickers = get_all_tickers()
    print(f"Testing {len(tickers)} stocks...")
    print()

    test_results = []
    improved_count = 0
    total_wr_improvement = 0
    total_return_improvement = 0

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Testing {ticker}...", end=" ")

        result = test_adaptive_exit_stock(ticker)

        if result and result.get('total_trades', 0) > 0:
            test_results.append(result)

            wr_improvement = result['improvement']['win_rate']
            return_improvement = result['improvement']['avg_return']

            total_wr_improvement += wr_improvement
            total_return_improvement += return_improvement

            if return_improvement > 0:
                improved_count += 1

            print(f"ATR={result['atr_pct']:.2f}%, Strategy={result['adaptive_strategy']}, "
                  f"WR Delta={wr_improvement:+.1f}pp, Return Delta={return_improvement:+.2f}%")
        else:
            print("No trades")

    # Summary
    print("\n" + "="*70)
    print("ADAPTIVE EXIT STRATEGY RESULTS")
    print("="*70)
    print()

    if not test_results:
        print("[ERROR] No test results")
        return None

    # Calculate aggregate statistics
    avg_wr_improvement = total_wr_improvement / len(test_results)
    avg_return_improvement = total_return_improvement / len(test_results)

    baseline_wr = np.mean([r['baseline']['win_rate'] for r in test_results])
    adaptive_wr = np.mean([r['adaptive']['win_rate'] for r in test_results])

    baseline_return = np.mean([r['baseline']['avg_return'] for r in test_results])
    adaptive_return = np.mean([r['adaptive']['avg_return'] for r in test_results])

    baseline_sharpe = np.mean([r['baseline']['sharpe'] for r in test_results])
    adaptive_sharpe = np.mean([r['adaptive']['sharpe'] for r in test_results])

    # Breakdown by volatility tier
    high_vol_stocks = [r for r in test_results if r['volatility_tier'] == 'HIGH']
    low_vol_stocks = [r for r in test_results if r['volatility_tier'] == 'LOW/MEDIUM']

    print(f"Stocks Tested: {len(test_results)}/54")
    print(f"Stocks Improved: {improved_count}/{len(test_results)} ({improved_count/len(test_results)*100:.1f}%)")
    print()

    print("VOLATILITY TIER BREAKDOWN:")
    print(f"  High Volatility (trailing_stop): {len(high_vol_stocks)} stocks")
    print(f"  Low/Medium Volatility (time_decay): {len(low_vol_stocks)} stocks")
    print()

    print(f"{'Metric':<25} {'Baseline':<20} {'Adaptive':<20} {'Change'}")
    print("-"*70)
    print(f"{'Win Rate':<25} {baseline_wr:<19.1f}% {adaptive_wr:<19.1f}% {avg_wr_improvement:+.1f}pp")
    print(f"{'Avg Return/Trade':<25} {baseline_return:<19.2f}% {adaptive_return:<19.2f}% {avg_return_improvement:+.2f}%")
    print(f"{'Sharpe Ratio':<25} {baseline_sharpe:<19.2f} {adaptive_sharpe:<19.2f} "
          f"{((adaptive_sharpe/baseline_sharpe - 1) * 100) if baseline_sharpe > 0 else 0:+.1f}%")
    print()

    # Compare to EXP-075 (trailing_stop only)
    print("COMPARISON TO EXP-075 (Trailing Stop Only):")
    print(f"  EXP-075 Result: 67.3% -> 64.7% (-2.6pp, -0.54%)")
    print(f"  EXP-077 Result: {baseline_wr:.1f}% -> {adaptive_wr:.1f}% ({avg_wr_improvement:+.1f}pp, {avg_return_improvement:+.2f}%)")
    print(f"  Improvement over EXP-075: {avg_wr_improvement + 2.6:+.1f}pp, {avg_return_improvement + 0.54:+.2f}%")
    print()

    # Top improvers
    print("TOP 10 STOCKS BY IMPROVEMENT:")
    print("-"*70)
    print(f"{'Ticker':<8} {'Vol Tier':<12} {'Strategy':<15} {'WR Change':<12} {'Return Change'}")
    print("-"*70)

    sorted_results = sorted(test_results,
                          key=lambda x: x['improvement']['avg_return'],
                          reverse=True)

    for result in sorted_results[:10]:
        print(f"{result['ticker']:<8} "
              f"{result['volatility_tier']:<12} "
              f"{result['adaptive_strategy']:<15} "
              f"{result['improvement']['win_rate']:<9.1f}pp "
              f"{result['improvement']['avg_return']:+.2f}%")

    # Deployment recommendation
    print()
    print("="*70)
    print("DEPLOYMENT RECOMMENDATION")
    print("="*70)
    print()

    if avg_wr_improvement >= 2.0 and avg_return_improvement >= 0.20 and improved_count >= 40:
        print(f"[SUCCESS] Adaptive exit strategy VALIDATED for deployment!")
        print()
        print(f"RESULTS: {avg_wr_improvement:+.1f}pp WR improvement, {avg_return_improvement:+.2f}% return improvement")
        print(f"IMPACT: {improved_count}/{len(test_results)} stocks improved")
        print()
        print("NEXT STEPS:")
        print("  1. Add ATR-based strategy selection to MeanReversionBacktester")
        print("  2. Deploy adaptive exit to production")
        print("  3. Monitor performance split by volatility tier")
        print()
        print(f"ESTIMATED ANNUAL IMPACT: {avg_return_improvement * 281:.1f}% cumulative return boost")
    elif avg_return_improvement >= 0.10:
        print(f"[PARTIAL SUCCESS] {avg_return_improvement:+.2f}% improvement (target: +0.20%)")
        print()
        print("Adaptive strategy shows promise:")
        print(f"  - {improved_count}/{len(test_results)} stocks improved")
        print(f"  - Win rate: {avg_wr_improvement:+.1f}pp")
        print()
        print("Consider deploying with refined threshold (currently 2.5% ATR)")
    else:
        print(f"[NEEDS WORK] {avg_return_improvement:+.2f}% improvement below threshold")
        print()
        print("Adaptive strategy may need different approach")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-077',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'objective': 'Test adaptive exit strategy based on volatility',
        'algorithm': 'High vol -> trailing_stop, Low vol -> time_decay',
        'stocks_tested': len(test_results),
        'stocks_improved': improved_count,
        'avg_wr_improvement_pp': float(avg_wr_improvement),
        'avg_return_improvement_pct': float(avg_return_improvement),
        'baseline_win_rate': float(baseline_wr),
        'adaptive_win_rate': float(adaptive_wr),
        'high_volatility_stocks': len(high_vol_stocks),
        'low_volatility_stocks': len(low_vol_stocks),
        'stock_results': test_results,
        'validated': avg_wr_improvement >= 2.0 and avg_return_improvement >= 0.20,
        'next_step': 'Deploy to production' if avg_return_improvement >= 0.20 else 'Refine threshold',
        'methodology': {
            'data_period': '2020-2025',
            'volatility_threshold': 2.5,
            'high_volatility_strategy': 'trailing_stop',
            'low_volatility_strategy': 'time_decay',
            'baseline_comparison': 'time_decay only'
        }
    }

    results_file = os.path.join(results_dir, 'exp077_adaptive_exit_strategy.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending adaptive exit strategy report...")
            notifier.send_experiment_report('EXP-077', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-077: Adaptive exit strategy."""

    print("\n[ADAPTIVE EXIT STRATEGY] Combining best of trailing_stop + time_decay")
    print("High volatility stocks -> trailing_stop, Low volatility -> time_decay")
    print()

    results = run_exp077_adaptive_exit_strategy()

    print("\n" + "="*70)
    print("ADAPTIVE EXIT STRATEGY TEST COMPLETE")
    print("="*70)

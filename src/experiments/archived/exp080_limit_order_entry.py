"""
EXPERIMENT: EXP-080
Date: 2025-11-17
Objective: Implement limit order entry for better intraday pricing

LEARNING FROM EXP-078:
- Perfect timing (low of day): +1.32% improvement (IMPOSSIBLE)
- Open entry: +0.03% improvement (MINIMAL)
- Realistic solution: Limit orders at discount to close

HYPOTHESIS:
Instead of entering at close, place limit order at 0.5% below close price.
- If filled during day: Better entry price
- If not filled: Skip trade (acts as additional filter)
- Historical analysis can determine optimal discount level (0.3% to 1.0%)

ALGORITHM:
1. For each historical signal, check if limit order would have filled:
   - Entry price = Close × (1 - discount%)
   - Filled if: Intraday Low <= Entry price
2. Compare to close-only entry:
   - Win rate (may improve due to filtering)
   - Avg return per trade (better entry = higher return)
   - Fill rate (what % of signals execute)
3. Test discount levels: 0.3%, 0.5%, 0.7%, 1.0%
4. Find optimal balance: fill rate vs return improvement

EXPECTED RESULTS:
- 0.5% discount: 70-80% fill rate, +0.3-0.5% return improvement
- Acts as volatility filter (only fills on true panic)
- Win rate improvement: +1-3pp (filtered signals)
- Annual impact: +84-140% cumulative (281 trades × 0.3-0.5%)

SUCCESS CRITERIA:
- Fill rate >= 65% (avoid losing too many signals)
- Avg return improvement >= +0.25% per filled trade
- Win rate improvement >= +1pp
- Universal benefit across 70%+ of stocks
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

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from src.config.mean_reversion_params import get_all_tickers


def test_limit_order_entry(ticker: str,
                             discount_pct: float = 0.5,
                             start_date: str = '2020-01-01',
                             end_date: str = '2024-11-17') -> Dict:
    """
    Test limit order entry strategy for a single stock.

    Args:
        ticker: Stock ticker
        discount_pct: Discount below close for limit order (0.5 = 0.5%)
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        Test results dict
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

        # Backtest with close entry (baseline)
        backtester_baseline = MeanReversionBacktester(
            initial_capital=10000,
            exit_strategy='time_decay',
            max_hold_days=3
        )
        baseline_results = backtester_baseline.backtest(signals)

        # Backtest with limit order entry
        # Simulate: Only execute if intraday low <= limit price
        filtered_signals = []
        total_signals = 0
        filled_signals = 0

        for signal_date, signal_row in signals.iterrows():
            if signal_row['panic_sell'] != 1:
                continue

            total_signals += 1

            close_price = signal_row['Close']
            low_price = signal_row['Low']
            limit_price = close_price * (1 - discount_pct / 100)

            # Check if limit order would have filled
            if low_price <= limit_price:
                filled_signals += 1
                # Modify entry price in signal
                modified_signal = signal_row.copy()
                modified_signal['Close'] = limit_price  # Enter at limit price
                filtered_signals.append(modified_signal)

        if len(filtered_signals) == 0:
            return {
                'ticker': ticker,
                'total_signals': total_signals,
                'filled_signals': 0,
                'fill_rate': 0,
                'baseline_wr': baseline_results.get('win_rate', 0),
                'limit_wr': 0,
                'baseline_avg_return': baseline_results.get('avg_gain', 0),
                'limit_avg_return': 0,
                'improvement': 0
            }

        # Create DataFrame for limit order signals
        limit_signals = pd.DataFrame(filtered_signals)
        limit_signals.index = pd.to_datetime(limit_signals.index) if not isinstance(limit_signals.index, pd.DatetimeIndex) else limit_signals.index

        # Backtest limit order strategy
        backtester_limit = MeanReversionBacktester(
            initial_capital=10000,
            exit_strategy='time_decay',
            max_hold_days=3
        )
        limit_results = backtester_limit.backtest(limit_signals)

        # Calculate metrics
        fill_rate = (filled_signals / total_signals) * 100 if total_signals > 0 else 0

        baseline_wr = baseline_results.get('win_rate', 0)
        limit_wr = limit_results.get('win_rate', 0)

        baseline_avg = baseline_results.get('avg_gain', 0)
        limit_avg = limit_results.get('avg_gain', 0)

        improvement = limit_avg - baseline_avg

        return {
            'ticker': ticker,
            'total_signals': total_signals,
            'filled_signals': filled_signals,
            'fill_rate': fill_rate,
            'baseline': {
                'win_rate': baseline_wr,
                'avg_return': baseline_avg,
                'total_trades': baseline_results.get('total_trades', 0)
            },
            'limit_order': {
                'win_rate': limit_wr,
                'avg_return': limit_avg,
                'total_trades': limit_results.get('total_trades', 0)
            },
            'improvement': {
                'win_rate': limit_wr - baseline_wr,
                'avg_return': improvement
            }
        }

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_exp080_limit_order_entry():
    """
    Test limit order entry strategy across full 54-stock portfolio.
    """
    print("="*70)
    print("EXP-080: LIMIT ORDER ENTRY OPTIMIZATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("OBJECTIVE: Better intraday entry via limit orders")
    print("METHODOLOGY: Place limit at 0.5% below close, only fill if hit")
    print("TARGET: +0.25% return improvement with 65%+ fill rate")
    print()

    # Test different discount levels
    discount_levels = [0.3, 0.5, 0.7, 1.0]

    print(f"Testing discount levels: {discount_levels}%")
    print()

    all_results = {}

    for discount in discount_levels:
        print(f"\n{'='*70}")
        print(f"TESTING: {discount}% DISCOUNT")
        print(f"{'='*70}\n")

        tickers = get_all_tickers()
        results = []
        total_improvement = 0
        improved_count = 0

        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] Testing {ticker}...", end=" ")

            result = test_limit_order_entry(ticker, discount_pct=discount)

            if result and result.get('filled_signals', 0) > 0:
                results.append(result)

                improvement = result['improvement']['avg_return']
                fill_rate = result['fill_rate']
                total_improvement += improvement

                if improvement > 0:
                    improved_count += 1

                print(f"Fill={fill_rate:.1f}%, Improvement={improvement:+.2f}%")
            else:
                print("No fills")

        all_results[discount] = {
            'results': results,
            'avg_improvement': total_improvement / len(results) if results else 0,
            'improved_count': improved_count,
            'avg_fill_rate': np.mean([r['fill_rate'] for r in results]) if results else 0
        }

    # Summary across all discount levels
    print("\n" + "="*70)
    print("LIMIT ORDER ENTRY OPTIMIZATION RESULTS")
    print("="*70)
    print()

    print(f"{'Discount':<12} {'Fill Rate':<15} {'Avg Improvement':<20} {'Stocks Improved'}")
    print("-"*70)

    for discount in discount_levels:
        data = all_results[discount]
        print(f"{discount:<11.1f}% {data['avg_fill_rate']:<14.1f}% "
              f"{data['avg_improvement']:<19.2f}% "
              f"{data['improved_count']}/{len(data['results'])}")

    # Find optimal discount
    optimal_discount = max(all_results.items(),
                          key=lambda x: x[1]['avg_improvement'] if x[1]['avg_fill_rate'] >= 65 else -999)[0]

    optimal_data = all_results[optimal_discount]

    print()
    print(f"OPTIMAL DISCOUNT: {optimal_discount}%")
    print(f"  Fill Rate: {optimal_data['avg_fill_rate']:.1f}%")
    print(f"  Avg Improvement: {optimal_data['avg_improvement']:+.2f}%")
    print(f"  Stocks Improved: {optimal_data['improved_count']}/{len(optimal_data['results'])}")
    print()

    # Detailed stats for optimal discount
    optimal_results = optimal_data['results']

    baseline_wr = np.mean([r['baseline']['win_rate'] for r in optimal_results])
    limit_wr = np.mean([r['limit_order']['win_rate'] for r in optimal_results])

    baseline_avg = np.mean([r['baseline']['avg_return'] for r in optimal_results])
    limit_avg = np.mean([r['limit_order']['avg_return'] for r in optimal_results])

    print(f"{'Metric':<25} {'Baseline':<20} {'Limit Order':<20} {'Improvement'}")
    print("-"*70)
    print(f"{'Win Rate':<25} {baseline_wr:<19.1f}% {limit_wr:<19.1f}% {limit_wr - baseline_wr:+.1f}pp")
    print(f"{'Avg Return/Trade':<25} {baseline_avg:<19.2f}% {limit_avg:<19.2f}% {optimal_data['avg_improvement']:+.2f}%")
    print()

    # Top 10 improvers
    print("TOP 10 STOCKS BY IMPROVEMENT:")
    print("-"*70)
    print(f"{'Ticker':<8} {'Fill Rate':<12} {'WR Delta':<12} {'Return Delta'}")
    print("-"*70)

    sorted_results = sorted(optimal_results,
                          key=lambda x: x['improvement']['avg_return'],
                          reverse=True)

    for result in sorted_results[:10]:
        print(f"{result['ticker']:<8} "
              f"{result['fill_rate']:<11.1f}% "
              f"{result['improvement']['win_rate']:<11.1f}pp "
              f"{result['improvement']['avg_return']:+.2f}%")

    # Deployment recommendation
    print()
    print("="*70)
    print("DEPLOYMENT RECOMMENDATION")
    print("="*70)
    print()

    if optimal_data['avg_improvement'] >= 0.25 and optimal_data['avg_fill_rate'] >= 65:
        print(f"[SUCCESS] Limit order entry VALIDATED for production!")
        print()
        print(f"RESULTS: {optimal_data['avg_improvement']:+.2f}% avg return improvement")
        print(f"FILL RATE: {optimal_data['avg_fill_rate']:.1f}% (maintains signal volume)")
        print(f"IMPACT: {optimal_data['improved_count']}/{len(optimal_results)} stocks improved")
        print()
        print("NEXT STEPS:")
        print(f"  1. Implement limit orders at {optimal_discount}% below close in production")
        print("  2. Use Good-Till-Day (GTD) orders that expire at close")
        print("  3. Monitor fill rates and adjust discount if needed")
        print()
        print(f"ESTIMATED ANNUAL IMPACT: {optimal_data['avg_improvement'] * 281 * (optimal_data['avg_fill_rate']/100):.1f}% cumulative")
    elif optimal_data['avg_improvement'] >= 0.15:
        print(f"[PARTIAL SUCCESS] {optimal_data['avg_improvement']:+.2f}% improvement (target: +0.25%)")
        print()
        print(f"Limit orders show promise but below target:")
        print(f"  - Fill rate: {optimal_data['avg_fill_rate']:.1f}%")
        print(f"  - {optimal_data['improved_count']}/{len(optimal_results)} stocks improved")
        print()
        print("Consider implementing with monitoring")
    else:
        print(f"[NEEDS WORK] {optimal_data['avg_improvement']:+.2f}% improvement too small")
        print()
        print("Limit orders may not provide significant edge")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-080',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'objective': 'Test limit order entry for better intraday pricing',
        'algorithm': 'Limit orders at discount to close price',
        'discount_levels_tested': discount_levels,
        'optimal_discount': float(optimal_discount),
        'optimal_fill_rate': float(optimal_data['avg_fill_rate']),
        'optimal_improvement': float(optimal_data['avg_improvement']),
        'stocks_improved': optimal_data['improved_count'],
        'baseline_win_rate': float(baseline_wr),
        'limit_win_rate': float(limit_wr),
        'all_discount_results': {
            str(k): {
                'avg_improvement': float(v['avg_improvement']),
                'avg_fill_rate': float(v['avg_fill_rate']),
                'improved_count': v['improved_count']
            } for k, v in all_results.items()
        },
        'validated': optimal_data['avg_improvement'] >= 0.25 and optimal_data['avg_fill_rate'] >= 65,
        'next_step': f'Deploy {optimal_discount}% limit orders' if optimal_data['avg_improvement'] >= 0.25 else 'Needs refinement'
    }

    results_file = os.path.join(results_dir, 'exp080_limit_order_entry.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending limit order entry report...")
            notifier.send_experiment_report('EXP-080', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-080: Limit order entry optimization."""

    print("\n[LIMIT ORDER ENTRY] Testing practical intraday entry improvement")
    print("Place limit orders at discount to close, fill only if hit")
    print()

    results = run_exp080_limit_order_entry()

    print("\n" + "="*70)
    print("LIMIT ORDER ENTRY OPTIMIZATION COMPLETE")
    print("="*70)

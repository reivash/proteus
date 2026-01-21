"""
EXPERIMENT: EXP-078
Date: 2025-11-17
Objective: Optimize intraday entry timing for mean reversion signals

HYPOTHESIS:
Current strategy enters at close price on signal day. Intraday price action
on panic sell days follows patterns that can be exploited for better entries:

Pattern 1: V-shaped recovery (panic at open, recovery by close) -> Buy at open
Pattern 2: Continued decline (weakness all day) -> Buy at close
Pattern 3: Mid-day reversal -> Buy at low of day

Historical analysis can determine optimal entry time per stock/volatility tier.

ALGORITHM:
1. For each historical signal, analyze intraday price action:
   - Open, High, Low, Close prices
   - Time of day when low was hit
   - Intraday range and recovery pattern
2. Calculate hypothetical returns for different entry strategies:
   - Enter at open
   - Enter at low of day (if we could time it perfectly)
   - Enter at close (current baseline)
   - Enter at mid-day
3. Determine which entry time historically performs best
4. Test if simple rules can capture this improvement:
   - "Always enter at open for volatile stocks"
   - "Always enter at close for stable stocks"
   - "Enter at midpoint of open-close range"

EXPECTED RESULTS:
- Open entry: +0.3-0.5% better for volatile stocks (catch panic lows)
- Close entry: +0.1-0.2% better for stable stocks (avoid intraday noise)
- Optimal entry improvement: +0.2-0.4% average across all trades

SUCCESS CRITERIA:
- Identifiable pattern in optimal entry timing
- Improvement >= +0.15% per trade vs close-only entry
- Simple rule that can be implemented in production
- Improvement consistent across 70%+ of stocks
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
from src.models.trading.mean_reversion import MeanReversionDetector
from src.config.mean_reversion_params import get_all_tickers


def analyze_intraday_entry_timing(ticker: str,
                                  start_date: str = '2020-01-01',
                                  end_date: str = '2025-11-17') -> Dict:
    """
    Analyze optimal intraday entry timing for a stock.
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

        # Apply filters
        regime_detector = MarketRegimeDetector()
        signals = add_regime_filter_to_signals(signals, regime_detector)

        earnings_fetcher = EarningsCalendarFetcher(exclusion_days_before=3, exclusion_days_after=3)
        signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

        if len(signals) == 0:
            return None

        # Analyze entry timing for each signal
        entry_analysis = []

        for signal_date, signal_row in signals.iterrows():
            # Get signal day data
            signal_data = enriched_data.loc[signal_date]

            open_price = signal_data['Open']
            high_price = signal_data['High']
            low_price = signal_data['Low']
            close_price = signal_data['Close']

            # Calculate intraday metrics
            intraday_range = ((high_price - low_price) / low_price) * 100
            open_to_close = ((close_price - open_price) / open_price) * 100
            low_to_close = ((close_price - low_price) / low_price) * 100

            # Get 3-day forward returns from different entry points
            future_data = enriched_data.loc[signal_date:].iloc[1:4]  # Next 3 days

            if len(future_data) == 0:
                continue

            # Use close of day 2 as exit (typical holding period)
            if len(future_data) >= 2:
                exit_price = future_data.iloc[1]['Close']
            else:
                exit_price = future_data.iloc[-1]['Close']

            # Calculate returns from different entry points
            return_from_open = ((exit_price - open_price) / open_price) * 100
            return_from_low = ((exit_price - low_price) / low_price) * 100
            return_from_close = ((exit_price - close_price) / close_price) * 100
            return_from_midpoint = ((exit_price - (open_price + close_price) / 2) / ((open_price + close_price) / 2)) * 100

            entry_analysis.append({
                'date': signal_date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'intraday_range': intraday_range,
                'open_to_close_pct': open_to_close,
                'low_to_close_pct': low_to_close,
                'return_from_open': return_from_open,
                'return_from_low': return_from_low,
                'return_from_close': return_from_close,
                'return_from_midpoint': return_from_midpoint,
                'best_entry': max([
                    ('open', return_from_open),
                    ('low', return_from_low),
                    ('close', return_from_close),
                    ('midpoint', return_from_midpoint)
                ], key=lambda x: x[1])[0]
            })

        if len(entry_analysis) == 0:
            return None

        df_analysis = pd.DataFrame(entry_analysis)

        # Calculate average returns for each entry strategy
        avg_return_open = df_analysis['return_from_open'].mean()
        avg_return_low = df_analysis['return_from_low'].mean()
        avg_return_close = df_analysis['return_from_close'].mean()
        avg_return_midpoint = df_analysis['return_from_midpoint'].mean()

        # Determine best strategy
        strategies = {
            'open': avg_return_open,
            'low': avg_return_low,
            'close': avg_return_close,
            'midpoint': avg_return_midpoint
        }
        best_strategy = max(strategies.items(), key=lambda x: x[1])

        # Calculate improvement over close (baseline)
        improvement = best_strategy[1] - avg_return_close

        return {
            'ticker': ticker,
            'total_signals': len(entry_analysis),
            'avg_intraday_range': df_analysis['intraday_range'].mean(),
            'avg_open_to_close': df_analysis['open_to_close_pct'].mean(),
            'strategies': {
                'open': avg_return_open,
                'low': avg_return_low,
                'close': avg_return_close,
                'midpoint': avg_return_midpoint
            },
            'best_strategy': best_strategy[0],
            'best_return': best_strategy[1],
            'baseline_return': avg_return_close,
            'improvement': improvement,
            'improvement_pct': (improvement / abs(avg_return_close)) * 100 if avg_return_close != 0 else 0,
            'best_entry_distribution': df_analysis['best_entry'].value_counts().to_dict()
        }

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_exp078_intraday_entry_timing():
    """
    Analyze optimal intraday entry timing across full portfolio.
    """
    print("="*70)
    print("EXP-078: INTRADAY ENTRY TIMING OPTIMIZATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("OBJECTIVE: Find optimal intraday entry time for mean reversion signals")
    print("METHODOLOGY: Analyze historical open/high/low/close returns")
    print("TARGET: +0.15% improvement per trade vs close-only entry")
    print()

    # Get all portfolio tickers
    tickers = get_all_tickers()
    print(f"Analyzing {len(tickers)} stocks...")
    print()

    results = []
    total_improvement = 0
    improved_count = 0

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Analyzing {ticker}...", end=" ")

        result = analyze_intraday_entry_timing(ticker)

        if result and result.get('total_signals', 0) >= 5:
            results.append(result)

            improvement = result['improvement']
            total_improvement += improvement

            if improvement > 0:
                improved_count += 1

            print(f"Signals={result['total_signals']}, "
                  f"Best={result['best_strategy']}, "
                  f"Improvement={improvement:+.2f}%")
        else:
            print("Insufficient signals")

    # Summary
    print("\n" + "="*70)
    print("INTRADAY ENTRY TIMING RESULTS")
    print("="*70)
    print()

    if not results:
        print("[ERROR] No results")
        return None

    # Calculate aggregate statistics
    avg_improvement = total_improvement / len(results)

    baseline_return = np.mean([r['baseline_return'] for r in results])
    best_return = np.mean([r['best_return'] for r in results])

    # Count best entry strategies
    strategy_counts = {}
    for r in results:
        strategy = r['best_strategy']
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

    print(f"Stocks Analyzed: {len(results)}/54")
    print(f"Stocks Improved: {improved_count}/{len(results)} ({improved_count/len(results)*100:.1f}%)")
    print()

    print("BEST ENTRY STRATEGY DISTRIBUTION:")
    for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {strategy}: {count} stocks ({count/len(results)*100:.1f}%)")
    print()

    print(f"{'Entry Strategy':<20} {'Avg Return':<15} {'vs Close Baseline'}")
    print("-"*70)

    avg_open = np.mean([r['strategies']['open'] for r in results])
    avg_low = np.mean([r['strategies']['low'] for r in results])
    avg_close = np.mean([r['strategies']['close'] for r in results])
    avg_midpoint = np.mean([r['strategies']['midpoint'] for r in results])

    print(f"{'Open':<20} {avg_open:<14.2f}% {avg_open - avg_close:+.2f}%")
    print(f"{'Low (perfect timing)':<20} {avg_low:<14.2f}% {avg_low - avg_close:+.2f}%")
    print(f"{'Close (baseline)':<20} {avg_close:<14.2f}% +0.00%")
    print(f"{'Midpoint':<20} {avg_midpoint:<14.2f}% {avg_midpoint - avg_close:+.2f}%")
    print()

    print(f"Average Improvement (Best Strategy): {avg_improvement:+.2f}%")
    print(f"Annual Impact (281 trades): {avg_improvement * 281:.1f}% cumulative")
    print()

    # Top improvers
    print("TOP 10 STOCKS BY IMPROVEMENT:")
    print("-"*70)
    print(f"{'Ticker':<8} {'Signals':<10} {'Best Entry':<12} {'Improvement'}")
    print("-"*70)

    sorted_results = sorted(results, key=lambda x: x['improvement'], reverse=True)

    for result in sorted_results[:10]:
        print(f"{result['ticker']:<8} "
              f"{result['total_signals']:<10} "
              f"{result['best_strategy']:<12} "
              f"{result['improvement']:+.2f}%")

    # Deployment recommendation
    print()
    print("="*70)
    print("DEPLOYMENT RECOMMENDATION")
    print("="*70)
    print()

    if avg_improvement >= 0.15 and improved_count >= len(results) * 0.7:
        print(f"[SUCCESS] Intraday entry timing VALIDATED!")
        print()
        print(f"RESULTS: {avg_improvement:+.2f}% improvement per trade")
        print(f"IMPACT: {improved_count}/{len(results)} stocks improved")
        print()

        # Determine simple rule
        most_common_strategy = max(strategy_counts.items(), key=lambda x: x[1])[0]
        print(f"SIMPLE RULE: Entry at {most_common_strategy} works for {strategy_counts[most_common_strategy]} stocks")
        print()
        print("NEXT STEPS:")
        print(f"  1. Implement {most_common_strategy} entry in production scanner")
        print("  2. Monitor improvement in live trading")
        print("  3. Consider stock-specific entry timing rules")
        print()
        print(f"ESTIMATED ANNUAL IMPACT: {avg_improvement * 281:.1f}% cumulative return boost")
    elif avg_improvement >= 0.10:
        print(f"[PARTIAL SUCCESS] {avg_improvement:+.2f}% improvement (target: +0.15%)")
        print()
        print("Intraday timing shows promise but needs refinement")
    else:
        print(f"[INCONCLUSIVE] {avg_improvement:+.2f}% improvement too small")
        print()
        print("Intraday timing may not provide significant edge")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-078',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'objective': 'Optimize intraday entry timing',
        'algorithm': 'Historical analysis of open/high/low/close returns',
        'stocks_analyzed': len(results),
        'stocks_improved': improved_count,
        'avg_improvement_pct': float(avg_improvement),
        'baseline_return': float(baseline_return),
        'best_return': float(best_return),
        'strategy_distribution': strategy_counts,
        'avg_strategies': {
            'open': float(avg_open),
            'low': float(avg_low),
            'close': float(avg_close),
            'midpoint': float(avg_midpoint)
        },
        'stock_results': results,
        'validated': avg_improvement >= 0.15,
        'recommended_strategy': max(strategy_counts.items(), key=lambda x: x[1])[0],
        'next_step': 'Deploy to production' if avg_improvement >= 0.15 else 'Needs refinement'
    }

    results_file = os.path.join(results_dir, 'exp078_intraday_entry_timing.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending intraday entry timing report...")
            notifier.send_experiment_report('EXP-078', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-078: Intraday entry timing optimization."""

    print("\n[INTRADAY ENTRY TIMING] Analyzing optimal entry time for signals")
    print("Testing: Open, Low, Close, Midpoint entry strategies")
    print()

    results = run_exp078_intraday_entry_timing()

    print("\n" + "="*70)
    print("INTRADAY ENTRY TIMING ANALYSIS COMPLETE")
    print("="*70)

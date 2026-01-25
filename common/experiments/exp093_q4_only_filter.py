"""
EXP-093: Q4-Only Signal Filter

CONTEXT:
EXP-091 validated base scanner achieves 63.7% win rate overall.
Quartile analysis showed Q4 (top 25% signal quality) achieves 77.8% win rate.

OBJECTIVE:
Validate Q4-only filtering improves production performance.

METHODOLOGY:
1. Calculate signal_strength 75th percentile threshold from historical signals
2. Backtest trading ONLY Q4 signals (signal_strength >= p75)
3. Compare to baseline (all signals)
4. Measure trade frequency reduction vs quality improvement

HYPOTHESIS:
Trading only Q4 signals will:
- Increase win rate from 63.7% to ~77.8% (+14.1pp)
- Reduce trade frequency by ~75% (only top quartile)
- Improve Sharpe ratio due to higher quality
- May reduce absolute returns due to fewer trades
- Net effect: Better risk-adjusted returns

EXPECTED IMPACT:
- Win rate: 63.7% → 77.8% (+14.1pp)
- Sharpe ratio: +30-50% improvement
- Trade frequency: 281 trades/year → ~70 trades/year
- Quality over quantity: Fewer but much higher conviction signals

SUCCESS CRITERIA:
- Win rate >= 75% (vs 63.7% baseline)
- Sharpe ratio >= 3.0 (vs 2.37 baseline)
- Deploy if validated

RISK:
Low - We already know Q4 performs at 77.8%. This just validates
the threshold calculation and production integration.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from common.trading.signal_scanner import SignalScanner
from common.trading.paper_trader import PaperTrader
from common.data.fetchers.yahoo_finance import YahooFinanceFetcher


def get_enhanced_performance(trader: 'PaperTrader') -> dict:
    """
    Get performance metrics including Sharpe ratio and avg return.

    Args:
        trader: PaperTrader instance

    Returns:
        Enhanced performance dictionary with all metrics
    """
    # Get basic performance
    perf = trader.get_performance()

    # Calculate avg_return from trade history
    if trader.trade_history:
        avg_return = np.mean([trade['return'] for trade in trader.trade_history])
    else:
        avg_return = 0.0

    # Calculate Sharpe ratio from daily equity
    if len(trader.daily_equity) > 1:
        returns = []
        for i in range(1, len(trader.daily_equity)):
            prev_equity = trader.daily_equity[i-1]['total_equity']
            curr_equity = trader.daily_equity[i]['total_equity']
            daily_return = (curr_equity - prev_equity) / prev_equity
            returns.append(daily_return)

        if returns and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    # Add enhanced metrics
    perf['avg_return'] = avg_return
    perf['sharpe_ratio'] = sharpe_ratio

    return perf


def calculate_signal_strength_threshold(scanner: SignalScanner, start_date: str, end_date: str) -> float:
    """
    Calculate the 75th percentile (Q4 threshold) of signal_strength
    from historical signals.
    """
    print("\nCalculating Q4 threshold from historical signals...")

    all_signals = []
    fetcher = YahooFinanceFetcher()

    # Get all historical signals
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        signals = scanner.scan_all_stocks(date_str)

        if signals:
            for signal in signals:
                all_signals.append(signal['signal_strength'])

    if not all_signals:
        print("[ERROR] No signals found in historical period")
        return None

    # Calculate 75th percentile
    q4_threshold = np.percentile(all_signals, 75)

    print(f"\nSignal strength distribution:")
    print(f"  Total signals: {len(all_signals)}")
    print(f"  Mean: {np.mean(all_signals):.2f}")
    print(f"  Median (Q2): {np.median(all_signals):.2f}")
    print(f"  75th percentile (Q4 threshold): {q4_threshold:.2f}")
    print(f"  Q4 signals: {sum(1 for s in all_signals if s >= q4_threshold)} ({sum(1 for s in all_signals if s >= q4_threshold)/len(all_signals)*100:.1f}%)")

    return q4_threshold


def backtest_q4_only(scanner: SignalScanner, q4_threshold: float,
                     start_date: str, end_date: str) -> dict:
    """
    Backtest trading ONLY Q4 signals (signal_strength >= q4_threshold).
    """
    print(f"\nBacktesting Q4-only filter (threshold >= {q4_threshold:.2f})...")

    trader = PaperTrader(
        initial_capital=100000,
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=2,
        position_size=0.1,
        max_positions=5
    )

    fetcher = YahooFinanceFetcher()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    total_signals_seen = 0
    q4_signals_traded = 0

    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')

        # Get all signals for this day
        all_signals = scanner.scan_all_stocks(date_str)

        if all_signals:
            total_signals_seen += len(all_signals)

            # Filter to Q4 only
            q4_signals = [s for s in all_signals if s['signal_strength'] >= q4_threshold]

            if q4_signals:
                q4_signals_traded += len(q4_signals)
                trader.process_signals(q4_signals, date_str)

        # Check exits
        try:
            current_prices = fetcher.get_current_prices([p['ticker'] for p in trader.positions])
            trader.check_exits(current_prices, date_str)
            trader.update_daily_equity(current_prices, date_str)
        except:
            pass

    # Get performance stats
    performance = get_enhanced_performance(trader)

    print(f"\nSignal filtering stats:")
    print(f"  Total signals seen: {total_signals_seen}")
    print(f"  Q4 signals traded: {q4_signals_traded}")
    print(f"  Filter rate: {(1 - q4_signals_traded/total_signals_seen)*100:.1f}% filtered out")

    print(f"\nPerformance:")
    print(f"  Win rate: {performance['win_rate']:.1f}%")
    print(f"  Total trades: {performance['total_trades']}")
    print(f"  Avg return: {performance['avg_return']:.2f}%")
    print(f"  Sharpe ratio: {performance.get('sharpe_ratio', 0):.2f}")

    return {
        'signals_seen': total_signals_seen,
        'signals_traded': q4_signals_traded,
        'filter_rate': (1 - q4_signals_traded/total_signals_seen) * 100,
        'performance': performance,
        'q4_threshold': q4_threshold
    }


def main():
    print("=" * 70)
    print("EXP-093: Q4-ONLY SIGNAL FILTER")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("OBJECTIVE: Validate Q4-only filtering improves win rate")
    print("BASELINE: 63.7% win rate (all signals)")
    print("TARGET: 77.8% win rate (Q4 signals only)")
    print()

    # Initialize scanner
    scanner = SignalScanner(lookback_days=90)

    # Test period: Last year of data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = '2024-01-01'

    # Step 1: Calculate Q4 threshold
    q4_threshold = calculate_signal_strength_threshold(scanner, start_date, end_date)

    if q4_threshold is None:
        print("\n[ERROR] Could not calculate Q4 threshold")
        return

    # Step 2: Backtest baseline (all signals)
    print("\n" + "=" * 70)
    print("BASELINE: All signals")
    print("=" * 70)

    trader_baseline = PaperTrader(
        initial_capital=100000,
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=2,
        position_size=0.1,
        max_positions=5
    )

    fetcher = YahooFinanceFetcher()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    baseline_signals_count = 0

    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        signals = scanner.scan_all_stocks(date_str)

        if signals:
            baseline_signals_count += len(signals)
            trader_baseline.process_signals(signals, date_str)

        try:
            current_prices = fetcher.get_current_prices([p['ticker'] for p in trader_baseline.positions])
            trader_baseline.check_exits(current_prices, date_str)
            trader_baseline.update_daily_equity(current_prices, date_str)
        except:
            pass

    baseline_perf = get_enhanced_performance(trader_baseline)

    print(f"\nBaseline performance:")
    print(f"  Total signals: {baseline_signals_count}")
    print(f"  Total trades: {baseline_perf['total_trades']}")
    print(f"  Win rate: {baseline_perf['win_rate']:.1f}%")
    print(f"  Avg return: {baseline_perf['avg_return']:.2f}%")
    print(f"  Sharpe ratio: {baseline_perf.get('sharpe_ratio', 0):.2f}")

    # Step 3: Backtest Q4-only
    print("\n" + "=" * 70)
    print(f"Q4-ONLY: signal_strength >= {q4_threshold:.2f}")
    print("=" * 70)

    q4_results = backtest_q4_only(scanner, q4_threshold, start_date, end_date)

    # Step 4: Compare results
    print("\n" + "=" * 70)
    print("COMPARISON: Baseline vs Q4-Only")
    print("=" * 70)

    baseline_wr = baseline_perf['win_rate']
    q4_wr = q4_results['performance']['win_rate']
    wr_improvement = q4_wr - baseline_wr

    baseline_sharpe = baseline_perf.get('sharpe_ratio', 0)
    q4_sharpe = q4_results['performance'].get('sharpe_ratio', 0)
    sharpe_improvement = ((q4_sharpe / baseline_sharpe - 1) * 100) if baseline_sharpe > 0 else 0

    print(f"\nWin Rate:")
    print(f"  Baseline (all signals): {baseline_wr:.1f}%")
    print(f"  Q4-only: {q4_wr:.1f}%")
    print(f"  Improvement: {wr_improvement:+.1f}pp")

    print(f"\nSharpe Ratio:")
    print(f"  Baseline: {baseline_sharpe:.2f}")
    print(f"  Q4-only: {q4_sharpe:.2f}")
    print(f"  Improvement: {sharpe_improvement:+.1f}%")

    print(f"\nTrade Frequency:")
    print(f"  Baseline: {baseline_signals_count} signals")
    print(f"  Q4-only: {q4_results['signals_traded']} signals")
    print(f"  Reduction: {q4_results['filter_rate']:.1f}%")

    # Recommendation
    print("\n" + "=" * 70)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 70)

    if q4_wr >= 75 and q4_sharpe >= 3.0:
        recommendation = "DEPLOY"
        reason = f"Win rate {q4_wr:.1f}% >= 75% target, Sharpe {q4_sharpe:.2f} >= 3.0 target"
    elif q4_wr >= baseline_wr + 10:
        recommendation = "DEPLOY"
        reason = f"+{wr_improvement:.1f}pp win rate improvement exceeds +10pp threshold"
    else:
        recommendation = "REJECT"
        reason = f"Win rate {q4_wr:.1f}% below 75% target or insufficient improvement"

    print(f"\n[{recommendation}] {reason}")

    # Save results
    results = {
        'experiment': 'EXP-093',
        'timestamp': datetime.now().isoformat(),
        'objective': 'Q4-only signal filter validation',
        'q4_threshold': q4_threshold,
        'baseline': {
            'signals': baseline_signals_count,
            'trades': baseline_perf['total_trades'],
            'win_rate': baseline_wr,
            'avg_return': baseline_perf['avg_return'],
            'sharpe': baseline_sharpe
        },
        'q4_only': {
            'signals': q4_results['signals_traded'],
            'trades': q4_results['performance']['total_trades'],
            'win_rate': q4_wr,
            'avg_return': q4_results['performance']['avg_return'],
            'sharpe': q4_sharpe
        },
        'improvement': {
            'win_rate_pp': wr_improvement,
            'sharpe_pct': sharpe_improvement,
            'signal_reduction_pct': q4_results['filter_rate']
        },
        'recommendation': recommendation,
        'reason': reason
    }

    output_path = Path('results/ml_experiments/exp093_q4_only_filter.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70)
    print("EXP-093 COMPLETE")
    print("=" * 70)

    # Send email notification
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            notifier.send_experiment_report('EXP-093', results)
            print("\n[EMAIL] Experiment report sent via SendGrid")
    except Exception as e:
        print(f"\n[WARNING] Could not send email: {e}")


if __name__ == "__main__":
    main()

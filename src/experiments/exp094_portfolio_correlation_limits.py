"""
EXP-094: Portfolio Correlation Limits

CONTEXT:
Current system can hold up to 5 concurrent positions, but doesn't consider
correlation between them. High correlation = higher portfolio risk.

OBJECTIVE:
Validate if limiting portfolio correlation improves risk-adjusted returns.

METHODOLOGY:
1. Baseline: No correlation limits (current system)
2. Test Strategy: Reject new signals if correlation with existing positions > threshold
3. Correlation calculation: Rolling 30-day price correlation
4. Test thresholds: 0.7, 0.8, 0.9 (reject if avg correlation exceeds threshold)

HYPOTHESIS:
Diversified portfolios have lower volatility for same returns:
- Lower correlation → Lower portfolio volatility
- Same win rate → Same returns
- Result: Higher Sharpe ratio (+10-20% improvement expected)

EXPECTED IMPACT:
- Sharpe ratio: 2.37 → 2.6-2.8 (+10-18% improvement)
- Max drawdown: -15% reduction via diversification
- Win rate: Unchanged (same quality signals)
- Trade frequency: Slightly lower (some signals rejected due to correlation)

RISK:
Low - Only affects position selection, not signal quality.
Worst case: No improvement (correlation already low naturally).

SUCCESS CRITERIA:
- Sharpe ratio >= 2.6 (+10% minimum)
- Max drawdown improved (lower)
- Deploy optimal threshold if validated
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.trading.signal_scanner import SignalScanner
from src.trading.paper_trader import PaperTrader
from src.data.fetchers.yahoo_finance import YahooFinanceFetcher


def calculate_correlation(ticker1: str, ticker2: str, date: str, window_days: int = 30) -> float:
    """
    Calculate rolling price correlation between two tickers.

    Args:
        ticker1: First ticker
        ticker2: Second ticker
        date: Current date
        window_days: Lookback window for correlation (default 30 days)

    Returns:
        Correlation coefficient (-1 to 1), or None if insufficient data
    """
    fetcher = YahooFinanceFetcher()
    end_date = pd.to_datetime(date)
    start_date = end_date - timedelta(days=window_days + 10)  # Extra buffer

    try:
        data1 = fetcher.fetch_stock_data(ticker1, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        data2 = fetcher.fetch_stock_data(ticker2, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if len(data1) < window_days or len(data2) < window_days:
            return None

        # Align dates
        merged = pd.merge(data1[['Close']], data2[['Close']], left_index=True, right_index=True, suffixes=('_1', '_2'))

        if len(merged) < window_days:
            return None

        # Calculate correlation on last window_days
        recent = merged.tail(window_days)
        corr = recent['Close_1'].corr(recent['Close_2'])

        return corr

    except Exception as e:
        print(f"[WARN] Failed to calculate correlation between {ticker1} and {ticker2}: {e}")
        return None


def should_accept_signal(signal: Dict, current_positions: List[Dict], correlation_threshold: float) -> tuple:
    """
    Determine if signal should be accepted based on portfolio correlation.

    Args:
        signal: Signal dictionary
        current_positions: List of current position dictionaries
        correlation_threshold: Maximum average correlation allowed

    Returns:
        (accept: bool, avg_correlation: float)
    """
    if not current_positions:
        return True, 0.0

    ticker = signal['ticker']
    date = signal['date']

    correlations = []
    for pos in current_positions:
        pos_ticker = pos['ticker']
        if ticker != pos_ticker:
            corr = calculate_correlation(ticker, pos_ticker, date, window_days=30)
            if corr is not None:
                correlations.append(abs(corr))  # Use absolute correlation

    if not correlations:
        return True, 0.0

    avg_corr = np.mean(correlations)
    accept = avg_corr <= correlation_threshold

    return accept, avg_corr


def backtest_with_correlation_limit(correlation_threshold: float, start_date: str, end_date: str) -> dict:
    """
    Backtest strategy with correlation limits.

    Args:
        correlation_threshold: Maximum average correlation (0.7, 0.8, 0.9, or None for baseline)
        start_date: Start date
        end_date: End date

    Returns:
        Performance results dictionary
    """
    print(f"\nBacktesting correlation_threshold={correlation_threshold if correlation_threshold else 'None (baseline)'}...")

    scanner = SignalScanner(lookback_days=90)
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

    total_signals = 0
    accepted_signals = 0
    rejected_signals = 0
    rejection_reasons = []

    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')

        # Get signals for this day
        signals = scanner.scan_all_stocks(date_str)

        if signals:
            total_signals += len(signals)

            # Filter signals based on correlation if threshold set
            if correlation_threshold is not None:
                filtered_signals = []
                for signal in signals:
                    accept, avg_corr = should_accept_signal(signal, trader.positions, correlation_threshold)
                    if accept:
                        filtered_signals.append(signal)
                        accepted_signals += 1
                    else:
                        rejected_signals += 1
                        rejection_reasons.append({
                            'ticker': signal['ticker'],
                            'date': date_str,
                            'avg_correlation': avg_corr,
                            'threshold': correlation_threshold
                        })
                signals = filtered_signals
            else:
                accepted_signals += len(signals)

            trader.process_signals(signals, date_str)

        # Check exits
        try:
            current_prices = fetcher.get_current_prices([p['ticker'] for p in trader.positions])
            trader.check_exits(current_prices, date_str)
            trader.update_daily_equity(current_prices, date_str)
        except:
            pass

    # Get performance
    performance = trader.get_performance_summary()

    print(f"  Total signals: {total_signals}")
    print(f"  Accepted: {accepted_signals}")
    print(f"  Rejected: {rejected_signals} ({rejected_signals/total_signals*100:.1f}% filtered)")
    print(f"  Win rate: {performance['win_rate']:.1f}%")
    print(f"  Sharpe ratio: {performance.get('sharpe_ratio', 0):.2f}")

    return {
        'correlation_threshold': correlation_threshold,
        'total_signals': total_signals,
        'accepted_signals': accepted_signals,
        'rejected_signals': rejected_signals,
        'rejection_rate': rejected_signals / total_signals if total_signals > 0 else 0,
        'performance': performance,
        'rejection_reasons': rejection_reasons
    }


def main():
    print("=" * 70)
    print("EXP-094: PORTFOLIO CORRELATION LIMITS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("OBJECTIVE: Improve Sharpe ratio via portfolio diversification")
    print("BASELINE: No correlation limits (current system)")
    print("TEST: Reject signals with avg correlation > threshold")
    print()

    # Test period: Last year of data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = '2024-01-01'

    print(f"Test Period: {start_date} to {end_date}")
    print()

    # Test multiple thresholds
    thresholds = [None, 0.9, 0.8, 0.7]  # None = baseline
    results = []

    for threshold in thresholds:
        result = backtest_with_correlation_limit(threshold, start_date, end_date)
        results.append(result)

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON: Baseline vs Correlation Limits")
    print("=" * 70)

    baseline = results[0]
    baseline_sharpe = baseline['performance'].get('sharpe_ratio', 0)
    baseline_wr = baseline['performance']['win_rate']

    print(f"\nBaseline (no limits):")
    print(f"  Win rate: {baseline_wr:.1f}%")
    print(f"  Sharpe ratio: {baseline_sharpe:.2f}")
    print(f"  Total trades: {baseline['performance']['total_trades']}")

    best_result = baseline
    best_improvement = 0

    for result in results[1:]:
        threshold = result['correlation_threshold']
        sharpe = result['performance'].get('sharpe_ratio', 0)
        wr = result['performance']['win_rate']

        sharpe_improvement = ((sharpe / baseline_sharpe - 1) * 100) if baseline_sharpe > 0 else 0
        wr_change = wr - baseline_wr

        print(f"\nThreshold {threshold}:")
        print(f"  Win rate: {wr:.1f}% ({wr_change:+.1f}pp)")
        print(f"  Sharpe ratio: {sharpe:.2f} ({sharpe_improvement:+.1f}%)")
        print(f"  Total trades: {result['performance']['total_trades']}")
        print(f"  Rejection rate: {result['rejection_rate']*100:.1f}%")

        if sharpe_improvement > best_improvement:
            best_improvement = sharpe_improvement
            best_result = result

    # Recommendation
    print("\n" + "=" * 70)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 70)

    if best_improvement >= 10:
        recommendation = "DEPLOY"
        threshold = best_result['correlation_threshold']
        reason = f"Sharpe ratio improved by {best_improvement:.1f}% with threshold {threshold}"
    else:
        recommendation = "REJECT"
        reason = f"Sharpe improvement {best_improvement:.1f}% below 10% target"

    print(f"\n[{recommendation}] {reason}")

    # Save results
    output = {
        'experiment': 'EXP-094',
        'timestamp': datetime.now().isoformat(),
        'objective': 'Portfolio correlation limits for diversification',
        'test_period': {
            'start': start_date,
            'end': end_date
        },
        'baseline': {
            'win_rate': baseline_wr,
            'sharpe': baseline_sharpe,
            'total_trades': baseline['performance']['total_trades']
        },
        'best_result': {
            'threshold': best_result['correlation_threshold'],
            'win_rate': best_result['performance']['win_rate'],
            'sharpe': best_result['performance'].get('sharpe_ratio', 0),
            'total_trades': best_result['performance']['total_trades'],
            'rejection_rate': best_result['rejection_rate']
        },
        'improvement': {
            'sharpe_pct': best_improvement
        },
        'recommendation': recommendation,
        'reason': reason
    }

    output_path = Path('results/ml_experiments/exp094_portfolio_correlation_limits.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70)
    print("EXP-094 COMPLETE")
    print("=" * 70)

    # Send email notification
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            notifier.send_experiment_report('EXP-094', output)
            print("\n[EMAIL] Experiment report sent via SendGrid")
    except Exception as e:
        print(f"\n[WARNING] Could not send email: {e}")


if __name__ == "__main__":
    main()

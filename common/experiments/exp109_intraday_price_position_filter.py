"""
EXP-109: Intraday Price Position Filter

OBJECTIVE: Improve signal quality by filtering out stocks with intraday recovery

THEORY:
- Mean reversion works best when stocks close near their intraday low (true panic)
- Stocks that recover intraday show underlying strength (poor mean reversion candidates)
- Intraday price position = (Close - Low) / (High - Low)
  * 0.0 = closed at low (true panic, best for mean reversion)
  * 1.0 = closed at high (full recovery, worst for mean reversion)

METHODOLOGY:
- Baseline: No filter (current system)
- Enhanced: Filter out signals where price_position > 0.30 (closed >30% above low)
- Backtest both strategies on historical data (2 years)
- Compare win rates, Sharpe ratios, and trade frequency

EXPECTED IMPACT:
- Win rate: +3-7pp improvement (filter out weak signals)
- Sharpe ratio: +10-20% improvement
- Trade frequency: -20-30% (quality over quantity)

SUCCESS CRITERIA:
- Win rate >= +3pp
- Sharpe ratio >= +10%
- Deploy if validated
"""

import sys
import os
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.trading.signal_scanner import SignalScanner
from common.trading.paper_trader import PaperTrader
from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.notifications.sendgrid_notifier import SendGridNotifier
from common.config.mean_reversion_params import get_all_tickers


# No longer needed - filter is now configurable via threshold parameter


def backtest_strategy(
    start_date: str,
    end_date: str,
    use_filter: bool = False,
    price_position_threshold: float = 0.30
) -> dict:
    """
    Backtest strategy with or without intraday price position filter.

    Args:
        start_date: Start date for backtest
        end_date: End date for backtest
        use_filter: Whether to use intraday price position filter
        price_position_threshold: Max price position (0.0-1.0)

    Returns:
        Performance metrics dictionary
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {'WITH FILTER' if use_filter else 'NO FILTER (BASELINE)'}")
    if use_filter:
        print(f"Filter: price_position <= {price_position_threshold}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*70}\n")

    # Initialize components
    if use_filter:
        scanner = SignalScanner(
            lookback_days=90,
            min_signal_strength=None,
            intraday_price_position_threshold=price_position_threshold
        )
    else:
        # Baseline: No filter (set threshold to None)
        scanner = SignalScanner(
            lookback_days=90,
            min_signal_strength=None,
            intraday_price_position_threshold=None
        )

    trader = PaperTrader(
        initial_capital=100000,
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=2,
        position_size=0.1,
        max_positions=5,
        use_limit_orders=True,
        use_dynamic_sizing=True,
        max_portfolio_heat=0.50
    )

    fetcher = YahooFinanceFetcher()

    # Generate trading dates
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days

    signals_found = 0
    signals_filtered = 0
    entries_executed = 0
    exits_executed = 0

    for date in dates:
        date_str = date.strftime('%Y-%m-%d')

        # Get market regime
        regime = scanner.get_market_regime()

        if regime == 'BEAR':
            continue  # Skip bear market days

        # Scan for signals
        signals = scanner.scan_all_stocks(date_str)
        signals_found += len(signals)

        # Apply intraday price position filter if enabled
        if use_filter and signals:
            pre_filter = len(signals)
            signals = [
                s for s in signals
                if s.get('intraday_price_position', 0.5) <= price_position_threshold
            ]
            signals_filtered += (pre_filter - len(signals))

        # Process entries
        if signals:
            entries = trader.process_signals(signals, date_str)
            entries_executed += len(entries)

        # Fetch current prices for exit checks
        open_positions = trader.get_open_positions()
        if open_positions:
            tickers = [pos['ticker'] for pos in open_positions]
            prices = {}

            for ticker in tickers:
                try:
                    data = fetcher.fetch_stock_data(ticker, start_date=date_str, end_date=date_str)
                    if not data.empty:
                        prices[ticker] = float(data['Close'].iloc[-1])
                except:
                    pass

            # Update positions and check exits
            if prices:
                trader.update_daily_equity(prices, date_str)
                exits = trader.check_exits(prices, date_str)
                exits_executed += len(exits)

    # Calculate performance metrics
    stats = trader.get_performance_stats()

    result = {
        'use_filter': use_filter,
        'price_position_threshold': price_position_threshold if use_filter else None,
        'total_signals_found': signals_found,
        'signals_filtered': signals_filtered if use_filter else 0,
        'filter_rate': (signals_filtered / signals_found * 100) if signals_found > 0 else 0,
        'entries_executed': entries_executed,
        'exits_executed': exits_executed,
        'total_trades': stats.get('total_trades', 0),
        'win_rate': stats.get('win_rate', 0),
        'avg_win': stats.get('avg_win', 0),
        'avg_loss': stats.get('avg_loss', 0),
        'total_return': stats.get('total_return', 0),
        'sharpe_ratio': stats.get('sharpe_ratio', 0),
        'max_drawdown': stats.get('max_drawdown', 0),
        'final_equity': stats.get('current_equity', 100000)
    }

    return result


def main():
    """Run EXP-109 experiment."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("=" * 70)
    print("EXP-109: INTRADAY PRICE POSITION FILTER")
    print("=" * 70)
    print(f"Started: {timestamp}")
    print()
    print("OBJECTIVE: Improve signal quality by filtering intraday recovery signals")
    print("THEORY: Mean reversion works best when stocks close near their lows")
    print()

    # Test period: Last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~2 years

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}")
    print()

    # Test different thresholds
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40]

    # Baseline (no filter)
    print("\n" + "=" * 70)
    print("BASELINE: No Filter")
    print("=" * 70)
    baseline = backtest_strategy(start_str, end_str, use_filter=False)

    print(f"\nBaseline Results:")
    print(f"  Signals found:     {baseline['total_signals_found']}")
    print(f"  Trades executed:   {baseline['total_trades']}")
    print(f"  Win rate:          {baseline['win_rate']:.1f}%")
    print(f"  Avg win:           {baseline['avg_win']:.2f}%")
    print(f"  Avg loss:          {baseline['avg_loss']:.2f}%")
    print(f"  Total return:      {baseline['total_return']:.2f}%")
    print(f"  Sharpe ratio:      {baseline['sharpe_ratio']:.2f}")
    print(f"  Max drawdown:      {baseline['max_drawdown']:.2f}%")

    # Test each threshold
    results = [baseline]
    best_result = baseline
    best_improvement = 0

    for threshold in thresholds:
        print("\n" + "=" * 70)
        print(f"ENHANCED: Price Position <= {threshold}")
        print("=" * 70)

        result = backtest_strategy(start_str, end_str, use_filter=True, price_position_threshold=threshold)
        results.append(result)

        # Calculate improvements
        win_rate_improvement = result['win_rate'] - baseline['win_rate']
        sharpe_improvement = ((result['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100) if baseline['sharpe_ratio'] != 0 else 0

        print(f"\nResults:")
        print(f"  Signals found:     {result['total_signals_found']}")
        print(f"  Signals filtered:  {result['signals_filtered']} ({result['filter_rate']:.1f}%)")
        print(f"  Trades executed:   {result['total_trades']}")
        print(f"  Win rate:          {result['win_rate']:.1f}% ({win_rate_improvement:+.1f}pp)")
        print(f"  Avg win:           {result['avg_win']:.2f}%")
        print(f"  Avg loss:          {result['avg_loss']:.2f}%")
        print(f"  Total return:      {result['total_return']:.2f}%")
        print(f"  Sharpe ratio:      {result['sharpe_ratio']:.2f} ({sharpe_improvement:+.1f}%)")
        print(f"  Max drawdown:      {result['max_drawdown']:.2f}%")

        # Track best result
        if win_rate_improvement > best_improvement:
            best_improvement = win_rate_improvement
            best_result = result

    # Summary and recommendation
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nThreshold Comparison:")
    print(f"{'Threshold':<12} {'Win Rate':<12} {'Improvement':<15} {'Sharpe':<12} {'Trades':<10}")
    print("-" * 70)

    for result in results:
        if result['use_filter']:
            threshold = result['price_position_threshold']
            win_rate_imp = result['win_rate'] - baseline['win_rate']
            print(f"{threshold:<12.2f} {result['win_rate']:<12.1f} {win_rate_imp:+<15.1f}pp {result['sharpe_ratio']:<12.2f} {result['total_trades']:<10}")
        else:
            print(f"{'Baseline':<12} {result['win_rate']:<12.1f} {'-':<15} {result['sharpe_ratio']:<12.2f} {result['total_trades']:<10}")

    # Deployment recommendation
    print("\n" + "=" * 70)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 70)

    win_rate_improvement = best_result['win_rate'] - baseline['win_rate']
    sharpe_improvement = ((best_result['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100) if baseline['sharpe_ratio'] != 0 else 0

    deploy = win_rate_improvement >= 3.0 and sharpe_improvement >= 10.0

    if deploy:
        print(f"\n[DEPLOY] Enhancement validated!")
        print(f"  Best threshold: {best_result['price_position_threshold']}")
        print(f"  Win rate improvement: +{win_rate_improvement:.1f}pp")
        print(f"  Sharpe improvement: +{sharpe_improvement:.1f}%")
        print(f"  Trade frequency: {best_result['filter_rate']:.1f}% filtered (quality over quantity)")
    else:
        print(f"\n[REJECT] Insufficient improvement")
        print(f"  Win rate improvement: +{win_rate_improvement:.1f}pp (need +3.0pp)")
        print(f"  Sharpe improvement: +{sharpe_improvement:.1f}% (need +10.0%)")

    # Save results
    experiment_results = {
        'experiment': 'EXP-109',
        'timestamp': timestamp,
        'objective': 'Intraday price position filter for signal quality',
        'test_period': f"{start_str} to {end_str}",
        'baseline': baseline,
        'thresholds_tested': thresholds,
        'results': results,
        'best_result': best_result,
        'win_rate_improvement': win_rate_improvement,
        'sharpe_improvement': sharpe_improvement,
        'recommendation': 'DEPLOY' if deploy else 'REJECT',
        'reason': f"Win rate: +{win_rate_improvement:.1f}pp, Sharpe: +{sharpe_improvement:.1f}%" if deploy else 'Insufficient improvement'
    }

    # Save to file
    os.makedirs('results/ml_experiments', exist_ok=True)
    output_file = 'results/ml_experiments/exp109_intraday_price_position_filter.json'
    with open(output_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Email notification
    try:
        notifier = SendGridNotifier()
        subject = f"EXP-109: Intraday Price Position Filter - {'DEPLOY' if deploy else 'REJECT'}"

        # Build email body
        body_lines = [
            "=" * 70,
            "EXP-109: INTRADAY PRICE POSITION FILTER",
            "=" * 70,
            "",
            "OBJECTIVE: Improve signal quality by filtering intraday recovery signals",
            "",
            f"Test period: {start_str} to {end_str}",
            "",
            "BASELINE (No Filter):",
            f"  Win rate: {baseline['win_rate']:.1f}%",
            f"  Sharpe: {baseline['sharpe_ratio']:.2f}",
            f"  Trades: {baseline['total_trades']}",
            "",
            f"BEST RESULT (Threshold: {best_result['price_position_threshold']}):",
            f"  Win rate: {best_result['win_rate']:.1f}% (+{win_rate_improvement:.1f}pp)",
            f"  Sharpe: {best_result['sharpe_ratio']:.2f} (+{sharpe_improvement:.1f}%)",
            f"  Trades: {best_result['total_trades']} ({best_result['filter_rate']:.1f}% filtered)",
            "",
            "=" * 70,
            "RECOMMENDATION",
            "=" * 70,
            "",
            f"[{'DEPLOY' if deploy else 'REJECT'}] {experiment_results['reason']}",
            ""
        ]

        email_body = "\n".join(body_lines)
        notifier.send_experiment_report(subject, email_body, experiment_results)
        print("\n[EMAIL] Experiment report sent via SendGrid")
    except Exception as e:
        print(f"\n[EMAIL] Failed to send notification: {e}")

    print("\n" + "=" * 70)
    print("EXP-109 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

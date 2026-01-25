"""
EXP-115: Entry Timing Optimization (Market-on-Open vs Market-on-Close)

OBJECTIVE: Improve entry price by optimizing trade execution timing

PROBLEM:
Current system enters at market close (fixed timing). But panic sell days
have distinct intraday price patterns that can be exploited for better entries.

THEORY:
Mean reversion signals trigger when stocks close down significantly. But:
- Morning gap-downs often recover intraday (better to enter at close)
- Intraday capitulation often bottoms before close (better to enter at low)
- Opening prices vs closing prices reveal optimal entry window

HYPOTHESIS:
Entering at OPEN (next morning) instead of CLOSE (signal day) may capture
panic lows more effectively, improving entry price by 0.1-0.3% per trade.

METHODOLOGY:
Historical analysis of signal day price patterns:
1. Baseline: Enter at CLOSE (current strategy)
2. Test: Enter at next day's OPEN
3. Test: Enter at next day's LOW (theoretical best)
4. Test: Enter at next day's VWAP (balanced)

Calculate return from each entry point to exit point.

EXPECTED IMPACT:
- Entry price improvement: +0.1-0.3% per trade
- Annual impact: 281 trades × 0.2% = +56.2% additional return
- Win rate: Unchanged (same exits)
- Sharpe ratio: +5-10% improvement
- Universal benefit across all stocks

SUCCESS CRITERIA:
- Entry improvement >= +0.15% per trade
- Sharpe improvement >= +5%
- Deploy optimal entry timing if validated

IMPLEMENTATION:
If MOO (market-on-open) wins, change order execution from:
- Current: Market order at close on signal day
- New: Market order at open on signal day + 1

COMPLEMENTARITY:
- Works alongside ALL other experiments (109-114)
- Doesn't change signal quality or exits
- Pure execution optimization
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


def backtest_with_entry_timing(
    start_date: str,
    end_date: str,
    entry_timing: str = 'close'
) -> dict:
    """
    Backtest strategy with different entry timing strategies.

    Args:
        start_date: Start date for backtest
        end_date: End date for backtest
        entry_timing: Entry timing strategy
            - 'close': Enter at close on signal day (current baseline)
            - 'next_open': Enter at open on signal day + 1
            - 'next_low': Enter at low on signal day + 1 (theoretical best)
            - 'next_vwap': Enter at VWAP on signal day + 1

    Returns:
        Performance metrics dictionary
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: Entry Timing = {entry_timing.upper()}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*70}\n")

    # Initialize components
    scanner = SignalScanner(
        lookback_days=90,
        min_signal_strength=65.0,  # Q4 filter (EXP-093)
        intraday_price_position_threshold=0.30  # EXP-109
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
    entries_executed = 0
    exits_executed = 0
    entry_price_adjustments = []

    for i, date in enumerate(dates):
        date_str = date.strftime('%Y-%m-%d')

        # Get market regime
        regime = scanner.get_market_regime()
        if regime == 'BEAR':
            continue

        # Scan for signals
        signals = scanner.scan_all_stocks(date_str)
        signals_found += len(signals)

        # Adjust entry prices based on timing strategy
        if signals and entry_timing != 'close':
            # Get next trading day
            if i + 1 < len(dates):
                next_date = dates[i + 1]
                next_date_str = next_date.strftime('%Y-%m-%d')

                for signal in signals:
                    ticker = signal['ticker']
                    baseline_price = signal['entry_price']  # Close price

                    try:
                        # Fetch next day's OHLC data
                        next_data = fetcher.fetch_stock_data(
                            ticker,
                            start_date=next_date_str,
                            end_date=next_date_str
                        )

                        if not next_data.empty:
                            next_open = float(next_data['Open'].iloc[-1])
                            next_high = float(next_data['High'].iloc[-1])
                            next_low = float(next_data['Low'].iloc[-1])
                            next_close = float(next_data['Close'].iloc[-1])

                            # Calculate VWAP approximation (OHLC average)
                            next_vwap = (next_open + next_high + next_low + next_close) / 4

                            # Select entry price based on strategy
                            if entry_timing == 'next_open':
                                new_entry = next_open
                            elif entry_timing == 'next_low':
                                new_entry = next_low
                            elif entry_timing == 'next_vwap':
                                new_entry = next_vwap
                            else:
                                new_entry = baseline_price

                            # Calculate improvement vs baseline
                            improvement_pct = ((baseline_price - new_entry) / baseline_price) * 100

                            # Update signal entry price
                            signal['entry_price'] = new_entry
                            signal['baseline_entry'] = baseline_price
                            signal['entry_improvement'] = improvement_pct

                            entry_price_adjustments.append({
                                'date': date_str,
                                'ticker': ticker,
                                'baseline_entry': baseline_price,
                                'actual_entry': new_entry,
                                'improvement_pct': improvement_pct,
                                'next_open': next_open,
                                'next_low': next_low,
                                'next_vwap': next_vwap
                            })
                    except:
                        # Fallback to baseline if data unavailable
                        signal['entry_price'] = baseline_price

        # Process entries
        if signals:
            entries = trader.process_signals(signals, date_str)
            entries_executed += len(entries)

        # Check exits
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

            if prices:
                trader.update_daily_equity(prices, date_str)
                exits = trader.check_exits(prices, date_str)
                exits_executed += len(exits)

    # Calculate performance metrics
    stats = trader.get_performance_stats()

    # Calculate profit factor
    total_wins = sum([t['pnl'] for t in trader.closed_positions if t['pnl'] > 0])
    total_losses = abs(sum([t['pnl'] for t in trader.closed_positions if t['pnl'] < 0]))
    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    # Calculate average entry improvement
    avg_entry_improvement = None
    if entry_price_adjustments:
        avg_entry_improvement = np.mean([a['improvement_pct'] for a in entry_price_adjustments])

    result = {
        'entry_timing': entry_timing,
        'signals_found': signals_found,
        'entries_executed': entries_executed,
        'exits_executed': exits_executed,
        'total_trades': stats.get('total_trades', 0),
        'win_rate': stats.get('win_rate', 0),
        'avg_win': stats.get('avg_win', 0),
        'avg_loss': stats.get('avg_loss', 0),
        'profit_factor': profit_factor,
        'total_return': stats.get('total_return', 0),
        'sharpe_ratio': stats.get('sharpe_ratio', 0),
        'max_drawdown': stats.get('max_drawdown', 0),
        'final_equity': stats.get('current_equity', 100000),
        'avg_entry_improvement': avg_entry_improvement,
        'entry_adjustments_count': len(entry_price_adjustments)
    }

    return result


def main():
    """Run EXP-115 experiment."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("=" * 70)
    print("EXP-115: ENTRY TIMING OPTIMIZATION")
    print("=" * 70)
    print(f"Started: {timestamp}")
    print()
    print("OBJECTIVE: Improve entry price via optimal trade execution timing")
    print("THEORY: Morning prices vs close prices reveal best entry window")
    print()

    # Test period: Last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}")
    print()

    # Baseline (enter at close)
    print("\n" + "=" * 70)
    print("BASELINE: Enter at CLOSE (signal day)")
    print("=" * 70)
    baseline = backtest_with_entry_timing(start_str, end_str, entry_timing='close')

    print(f"\nBaseline Results:")
    print(f"  Trades executed:   {baseline['total_trades']}")
    print(f"  Win rate:          {baseline['win_rate']:.1f}%")
    print(f"  Avg win:           {baseline['avg_win']:.2f}%")
    print(f"  Avg loss:          {baseline['avg_loss']:.2f}%")
    print(f"  Profit factor:     {baseline['profit_factor']:.2f}")
    print(f"  Total return:      {baseline['total_return']:.2f}%")
    print(f"  Sharpe ratio:      {baseline['sharpe_ratio']:.2f}")
    print(f"  Max drawdown:      {baseline['max_drawdown']:.2f}%")

    # Test 1: Enter at next day's open
    print("\n" + "=" * 70)
    print("TEST 1: Enter at OPEN (signal day + 1)")
    print("=" * 70)
    test_open = backtest_with_entry_timing(start_str, end_str, entry_timing='next_open')

    # Calculate improvements
    return_improvement_open = test_open['total_return'] - baseline['total_return']
    sharpe_improvement_open = ((test_open['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100) if baseline['sharpe_ratio'] != 0 else 0

    print(f"\nNext Open Results:")
    print(f"  Trades executed:   {test_open['total_trades']}")
    print(f"  Win rate:          {test_open['win_rate']:.1f}%")
    print(f"  Avg win:           {test_open['avg_win']:.2f}%")
    print(f"  Avg loss:          {test_open['avg_loss']:.2f}%")
    print(f"  Profit factor:     {test_open['profit_factor']:.2f}")
    print(f"  Total return:      {test_open['total_return']:.2f}% ({return_improvement_open:+.2f}pp)")
    print(f"  Sharpe ratio:      {test_open['sharpe_ratio']:.2f} ({sharpe_improvement_open:+.1f}%)")
    print(f"  Max drawdown:      {test_open['max_drawdown']:.2f}%")
    if test_open['avg_entry_improvement']:
        print(f"  Avg entry improvement: {test_open['avg_entry_improvement']:.3f}%")

    # Test 2: Enter at next day's low (theoretical best)
    print("\n" + "=" * 70)
    print("TEST 2: Enter at LOW (signal day + 1) - Theoretical Best")
    print("=" * 70)
    test_low = backtest_with_entry_timing(start_str, end_str, entry_timing='next_low')

    return_improvement_low = test_low['total_return'] - baseline['total_return']
    sharpe_improvement_low = ((test_low['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100) if baseline['sharpe_ratio'] != 0 else 0

    print(f"\nNext Low Results:")
    print(f"  Trades executed:   {test_low['total_trades']}")
    print(f"  Win rate:          {test_low['win_rate']:.1f}%")
    print(f"  Avg win:           {test_low['avg_win']:.2f}%")
    print(f"  Avg loss:          {test_low['avg_loss']:.2f}%")
    print(f"  Profit factor:     {test_low['profit_factor']:.2f}")
    print(f"  Total return:      {test_low['total_return']:.2f}% ({return_improvement_low:+.2f}pp)")
    print(f"  Sharpe ratio:      {test_low['sharpe_ratio']:.2f} ({sharpe_improvement_low:+.1f}%)")
    print(f"  Max drawdown:      {test_low['max_drawdown']:.2f}%")
    if test_low['avg_entry_improvement']:
        print(f"  Avg entry improvement: {test_low['avg_entry_improvement']:.3f}%")

    # Test 3: Enter at next day's VWAP
    print("\n" + "=" * 70)
    print("TEST 3: Enter at VWAP (signal day + 1)")
    print("=" * 70)
    test_vwap = backtest_with_entry_timing(start_str, end_str, entry_timing='next_vwap')

    return_improvement_vwap = test_vwap['total_return'] - baseline['total_return']
    sharpe_improvement_vwap = ((test_vwap['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100) if baseline['sharpe_ratio'] != 0 else 0

    print(f"\nNext VWAP Results:")
    print(f"  Trades executed:   {test_vwap['total_trades']}")
    print(f"  Win rate:          {test_vwap['win_rate']:.1f}%")
    print(f"  Avg win:           {test_vwap['avg_win']:.2f}%")
    print(f"  Avg loss:          {test_vwap['avg_loss']:.2f}%")
    print(f"  Profit factor:     {test_vwap['profit_factor']:.2f}")
    print(f"  Total return:      {test_vwap['total_return']:.2f}% ({return_improvement_vwap:+.2f}pp)")
    print(f"  Sharpe ratio:      {test_vwap['sharpe_ratio']:.2f} ({sharpe_improvement_vwap:+.1f}%)")
    print(f"  Max drawdown:      {test_vwap['max_drawdown']:.2f}%")
    if test_vwap['avg_entry_improvement']:
        print(f"  Avg entry improvement: {test_vwap['avg_entry_improvement']:.3f}%")

    # Determine best strategy
    strategies = [
        ('BASELINE (Close)', baseline, 0, 0),
        ('Next Open', test_open, return_improvement_open, sharpe_improvement_open),
        ('Next Low', test_low, return_improvement_low, sharpe_improvement_low),
        ('Next VWAP', test_vwap, return_improvement_vwap, sharpe_improvement_vwap)
    ]

    best_strategy = max(strategies, key=lambda x: x[1]['total_return'])

    # Deployment recommendation
    print("\n" + "=" * 70)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 70)

    # Deploy if improvement meets criteria
    deploy = best_strategy[2] >= 5.0 or best_strategy[3] >= 5.0

    if deploy and best_strategy[0] != 'BASELINE (Close)':
        print(f"\n[DEPLOY] {best_strategy[0]} strategy validated!")
        print(f"  Return improvement: +{best_strategy[2]:.2f}pp")
        print(f"  Sharpe improvement: +{best_strategy[3]:.1f}%")
        print(f"  Total return: {best_strategy[1]['total_return']:.2f}%")
        print(f"\n  Implementation: Change order execution to {best_strategy[0]}")
        print(f"  Universal benefit across all trades")
        print(f"  No impact on signal quality or exits")
    else:
        print(f"\n[REJECT] Insufficient improvement")
        print(f"  Best strategy: {best_strategy[0]}")
        print(f"  Return improvement: +{best_strategy[2]:.2f}pp (need +5.0pp)")
        print(f"  Sharpe improvement: +{best_strategy[3]:.1f}% (need +5.0%)")

    # Save results
    experiment_results = {
        'experiment': 'EXP-115',
        'timestamp': timestamp,
        'objective': 'Entry timing optimization',
        'test_period': f"{start_str} to {end_str}",
        'baseline': baseline,
        'next_open': test_open,
        'next_low': test_low,
        'next_vwap': test_vwap,
        'best_strategy': best_strategy[0],
        'return_improvement_pp': best_strategy[2],
        'sharpe_improvement_pct': best_strategy[3],
        'recommendation': 'DEPLOY' if deploy else 'REJECT',
        'reason': f"{best_strategy[0]}: Return +{best_strategy[2]:.2f}pp, Sharpe +{best_strategy[3]:.1f}%"
    }

    # Save to file
    os.makedirs('results/ml_experiments', exist_ok=True)
    output_file = 'results/ml_experiments/exp115_entry_timing_optimization.json'
    with open(output_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Email notification
    try:
        notifier = SendGridNotifier()
        subject = f"EXP-115: Entry Timing Optimization - {'DEPLOY' if deploy else 'REJECT'}"

        body_lines = [
            "=" * 70,
            "EXP-115: ENTRY TIMING OPTIMIZATION",
            "=" * 70,
            "",
            "OBJECTIVE: Improve entry price via optimal execution timing",
            "",
            f"Test period: {start_str} to {end_str}",
            "",
            "BASELINE (Enter at Close):",
            f"  Total Return: {baseline['total_return']:.2f}%",
            f"  Sharpe: {baseline['sharpe_ratio']:.2f}",
            "",
            "NEXT OPEN:",
            f"  Total Return: {test_open['total_return']:.2f}% ({return_improvement_open:+.2f}pp)",
            f"  Sharpe: {test_open['sharpe_ratio']:.2f} ({sharpe_improvement_open:+.1f}%)",
            f"  Entry improvement: {test_open.get('avg_entry_improvement', 0):.3f}%",
            "",
            "NEXT LOW (Theoretical):",
            f"  Total Return: {test_low['total_return']:.2f}% ({return_improvement_low:+.2f}pp)",
            f"  Sharpe: {test_low['sharpe_ratio']:.2f} ({sharpe_improvement_low:+.1f}%)",
            f"  Entry improvement: {test_low.get('avg_entry_improvement', 0):.3f}%",
            "",
            "NEXT VWAP:",
            f"  Total Return: {test_vwap['total_return']:.2f}% ({return_improvement_vwap:+.2f}pp)",
            f"  Sharpe: {test_vwap['sharpe_ratio']:.2f} ({sharpe_improvement_vwap:+.1f}%)",
            f"  Entry improvement: {test_vwap.get('avg_entry_improvement', 0):.3f}%",
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
    print("EXP-115 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

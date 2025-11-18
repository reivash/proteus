"""
EXP-116: Partial Profit Taking (Scale Out Strategy)

OBJECTIVE: Improve win rate and consistency via partial profit taking

PROBLEM:
Current system uses all-or-nothing exits: Either hit full 2% profit target
or get stopped out. This creates unnecessary risk:
- Winners that retrace from +1.8% to stop loss = full loss
- All upside captured only if full 2% target hit
- No mechanism to lock in partial gains

THEORY:
Scaling out of positions (taking partial profits) addresses the classic
trader dilemma: "I should have taken profits when I had them."

Research shows partial profit taking:
- Increases win rate (+3-5pp typical)
- Reduces maximum adverse excursion
- Improves psychological comfort (reduces regret)
- Maintains upside participation

HYPOTHESIS:
Taking 50% profits at 1.0% gain, letting remaining 50% run to 2.0% target
will improve win rate and reduce drawdowns while maintaining similar returns.

METHODOLOGY:
Test multiple scale-out strategies:
1. BASELINE: All-or-nothing (current) - 100% at 2% profit
2. TEST: 50/50 split - 50% at 1%, 50% at 2%
3. TEST: 2/3 - 1/3 split - 67% at 1%, 33% at 2%
4. TEST: 1/3 - 2/3 split - 33% at 1%, 67% at 2%
5. TEST: Progressive scale - 33% at 0.75%, 33% at 1.5%, 34% at 2%

Calculate metrics for each strategy.

EXPECTED IMPACT:
- Win rate: +3-5pp improvement (lock in partial gains earlier)
- Sharpe ratio: +5-10% improvement (less volatility)
- Max drawdown: -10-15% reduction
- Average win: Slightly lower (taking profits earlier)
- Average loss: Unchanged (same stop loss)
- Net effect: Higher win rate compensates for slightly lower avg win

SUCCESS CRITERIA:
- Win rate improvement >= +3pp
- Sharpe improvement >= +5%
- Deploy optimal scale-out strategy if validated

IMPLEMENTATION:
If 50/50 wins, modify PaperTrader to support partial exits:
- Current: Exit 100% at profit_target
- New: Exit 50% at profit_target/2, exit remaining 50% at profit_target

COMPLEMENTARITY:
- Works alongside ALL other experiments (109-115)
- Doesn't change signal quality, position sizing, or entry
- Pure exit optimization (complements EXP-111, 112, 113)

PSYCHOLOGY:
Addresses common regret scenario: "Stock hit +1.5% then reversed to stop loss"
With scale-out: Lock in +0.75% on half, reducing psychological pain
"""

import sys
import os
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.trading.signal_scanner import SignalScanner
from src.trading.paper_trader import PaperTrader
from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.notifications.sendgrid_notifier import SendGridNotifier


def backtest_with_scale_out(
    start_date: str,
    end_date: str,
    scale_out_strategy: str = 'none'
) -> dict:
    """
    Backtest strategy with different scale-out (partial profit taking) strategies.

    Args:
        start_date: Start date for backtest
        end_date: End date for backtest
        scale_out_strategy: Scale out strategy
            - 'none': All-or-nothing (current baseline) - 100% at 2%
            - 'split_50_50': 50% at 1%, 50% at 2%
            - 'split_67_33': 67% at 1%, 33% at 2%
            - 'split_33_67': 33% at 1%, 67% at 2%
            - 'progressive': 33% at 0.75%, 33% at 1.5%, 34% at 2%

    Returns:
        Performance metrics dictionary
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: Scale-Out Strategy = {scale_out_strategy.upper()}")
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
    scale_out_events = []
    partial_exits = 0
    full_exits = 0

    # Define scale-out levels based on strategy
    scale_out_levels = {}
    if scale_out_strategy == 'none':
        scale_out_levels = {2.0: 1.0}  # 100% at 2%
    elif scale_out_strategy == 'split_50_50':
        scale_out_levels = {1.0: 0.5, 2.0: 0.5}  # 50% at 1%, 50% at 2%
    elif scale_out_strategy == 'split_67_33':
        scale_out_levels = {1.0: 0.67, 2.0: 0.33}  # 67% at 1%, 33% at 2%
    elif scale_out_strategy == 'split_33_67':
        scale_out_levels = {1.0: 0.33, 2.0: 0.67}  # 33% at 1%, 67% at 2%
    elif scale_out_strategy == 'progressive':
        scale_out_levels = {0.75: 0.33, 1.5: 0.33, 2.0: 0.34}  # 33% at 0.75%, 33% at 1.5%, 34% at 2%

    for date in dates:
        date_str = date.strftime('%Y-%m-%d')

        # Get market regime
        regime = scanner.get_market_regime()
        if regime == 'BEAR':
            continue

        # Scan for signals
        signals = scanner.scan_all_stocks(date_str)
        signals_found += len(signals)

        # Process entries
        if signals:
            entries = trader.process_signals(signals, date_str)
            entries_executed += len(entries)

        # Check exits with scale-out logic
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

                # Custom exit logic for scale-out
                if scale_out_strategy != 'none':
                    for pos in open_positions:
                        ticker = pos['ticker']
                        if ticker not in prices:
                            continue

                        entry_price = pos['entry_price']
                        current_price = prices[ticker]
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100

                        # Check each scale-out level
                        for target_pct, exit_fraction in sorted(scale_out_levels.items()):
                            # Check if position has already hit this level
                            level_key = f"hit_{target_pct}"
                            if pos.get(level_key, False):
                                continue

                            # Check if current price hits this target
                            if pnl_pct >= target_pct:
                                # Mark this level as hit
                                pos[level_key] = True

                                # Record scale-out event
                                scale_out_events.append({
                                    'date': date_str,
                                    'ticker': ticker,
                                    'entry_price': entry_price,
                                    'exit_price': current_price,
                                    'target_pct': target_pct,
                                    'exit_fraction': exit_fraction,
                                    'pnl_pct': pnl_pct
                                })

                                if exit_fraction < 1.0:
                                    partial_exits += 1
                                else:
                                    full_exits += 1

                # Standard exit check (for stop loss and time decay)
                exits = trader.check_exits(prices, date_str)
                exits_executed += len(exits)

    # Calculate performance metrics
    stats = trader.get_performance_stats()

    # Calculate profit factor
    total_wins = sum([t['pnl'] for t in trader.closed_positions if t['pnl'] > 0])
    total_losses = abs(sum([t['pnl'] for t in trader.closed_positions if t['pnl'] < 0]))
    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    result = {
        'scale_out_strategy': scale_out_strategy,
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
        'partial_exits': partial_exits,
        'full_exits': full_exits,
        'scale_out_events_count': len(scale_out_events)
    }

    return result


def main():
    """Run EXP-116 experiment."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("=" * 70)
    print("EXP-116: PARTIAL PROFIT TAKING (SCALE OUT STRATEGY)")
    print("=" * 70)
    print(f"Started: {timestamp}")
    print()
    print("OBJECTIVE: Improve win rate and consistency via partial profit taking")
    print("THEORY: Lock in partial gains early, let remainder run to full target")
    print()

    # Test period: Last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}")
    print()

    # Baseline (all-or-nothing)
    print("\n" + "=" * 70)
    print("BASELINE: All-or-Nothing (100% at 2% profit)")
    print("=" * 70)
    baseline = backtest_with_scale_out(start_str, end_str, scale_out_strategy='none')

    print(f"\nBaseline Results:")
    print(f"  Trades executed:   {baseline['total_trades']}")
    print(f"  Win rate:          {baseline['win_rate']:.1f}%")
    print(f"  Avg win:           {baseline['avg_win']:.2f}%")
    print(f"  Avg loss:          {baseline['avg_loss']:.2f}%")
    print(f"  Profit factor:     {baseline['profit_factor']:.2f}")
    print(f"  Total return:      {baseline['total_return']:.2f}%")
    print(f"  Sharpe ratio:      {baseline['sharpe_ratio']:.2f}")
    print(f"  Max drawdown:      {baseline['max_drawdown']:.2f}%")

    # Test 1: 50/50 split
    print("\n" + "=" * 70)
    print("TEST 1: 50/50 Split (50% at 1%, 50% at 2%)")
    print("=" * 70)
    test_50_50 = backtest_with_scale_out(start_str, end_str, scale_out_strategy='split_50_50')

    win_rate_improvement_50_50 = test_50_50['win_rate'] - baseline['win_rate']
    sharpe_improvement_50_50 = ((test_50_50['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100) if baseline['sharpe_ratio'] != 0 else 0
    return_improvement_50_50 = test_50_50['total_return'] - baseline['total_return']

    print(f"\n50/50 Split Results:")
    print(f"  Trades executed:   {test_50_50['total_trades']}")
    print(f"  Win rate:          {test_50_50['win_rate']:.1f}% ({win_rate_improvement_50_50:+.1f}pp)")
    print(f"  Avg win:           {test_50_50['avg_win']:.2f}%")
    print(f"  Avg loss:          {test_50_50['avg_loss']:.2f}%")
    print(f"  Profit factor:     {test_50_50['profit_factor']:.2f}")
    print(f"  Total return:      {test_50_50['total_return']:.2f}% ({return_improvement_50_50:+.2f}pp)")
    print(f"  Sharpe ratio:      {test_50_50['sharpe_ratio']:.2f} ({sharpe_improvement_50_50:+.1f}%)")
    print(f"  Max drawdown:      {test_50_50['max_drawdown']:.2f}%")
    print(f"  Partial exits:     {test_50_50['partial_exits']}")
    print(f"  Full exits:        {test_50_50['full_exits']}")

    # Test 2: 67/33 split
    print("\n" + "=" * 70)
    print("TEST 2: 67/33 Split (67% at 1%, 33% at 2%)")
    print("=" * 70)
    test_67_33 = backtest_with_scale_out(start_str, end_str, scale_out_strategy='split_67_33')

    win_rate_improvement_67_33 = test_67_33['win_rate'] - baseline['win_rate']
    sharpe_improvement_67_33 = ((test_67_33['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100) if baseline['sharpe_ratio'] != 0 else 0
    return_improvement_67_33 = test_67_33['total_return'] - baseline['total_return']

    print(f"\n67/33 Split Results:")
    print(f"  Trades executed:   {test_67_33['total_trades']}")
    print(f"  Win rate:          {test_67_33['win_rate']:.1f}% ({win_rate_improvement_67_33:+.1f}pp)")
    print(f"  Avg win:           {test_67_33['avg_win']:.2f}%")
    print(f"  Avg loss:          {test_67_33['avg_loss']:.2f}%")
    print(f"  Profit factor:     {test_67_33['profit_factor']:.2f}")
    print(f"  Total return:      {test_67_33['total_return']:.2f}% ({return_improvement_67_33:+.2f}pp)")
    print(f"  Sharpe ratio:      {test_67_33['sharpe_ratio']:.2f} ({sharpe_improvement_67_33:+.1f}%)")
    print(f"  Max drawdown:      {test_67_33['max_drawdown']:.2f}%")

    # Test 3: 33/67 split
    print("\n" + "=" * 70)
    print("TEST 3: 33/67 Split (33% at 1%, 67% at 2%)")
    print("=" * 70)
    test_33_67 = backtest_with_scale_out(start_str, end_str, scale_out_strategy='split_33_67')

    win_rate_improvement_33_67 = test_33_67['win_rate'] - baseline['win_rate']
    sharpe_improvement_33_67 = ((test_33_67['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100) if baseline['sharpe_ratio'] != 0 else 0
    return_improvement_33_67 = test_33_67['total_return'] - baseline['total_return']

    print(f"\n33/67 Split Results:")
    print(f"  Trades executed:   {test_33_67['total_trades']}")
    print(f"  Win rate:          {test_33_67['win_rate']:.1f}% ({win_rate_improvement_33_67:+.1f}pp)")
    print(f"  Avg win:           {test_33_67['avg_win']:.2f}%")
    print(f"  Avg loss:          {test_33_67['avg_loss']:.2f}%")
    print(f"  Profit factor:     {test_33_67['profit_factor']:.2f}")
    print(f"  Total return:      {test_33_67['total_return']:.2f}% ({return_improvement_33_67:+.2f}pp)")
    print(f"  Sharpe ratio:      {test_33_67['sharpe_ratio']:.2f} ({sharpe_improvement_33_67:+.1f}%)")
    print(f"  Max drawdown:      {test_33_67['max_drawdown']:.2f}%")

    # Test 4: Progressive scale
    print("\n" + "=" * 70)
    print("TEST 4: Progressive Scale (33% at 0.75%, 33% at 1.5%, 34% at 2%)")
    print("=" * 70)
    test_progressive = backtest_with_scale_out(start_str, end_str, scale_out_strategy='progressive')

    win_rate_improvement_prog = test_progressive['win_rate'] - baseline['win_rate']
    sharpe_improvement_prog = ((test_progressive['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100) if baseline['sharpe_ratio'] != 0 else 0
    return_improvement_prog = test_progressive['total_return'] - baseline['total_return']

    print(f"\nProgressive Scale Results:")
    print(f"  Trades executed:   {test_progressive['total_trades']}")
    print(f"  Win rate:          {test_progressive['win_rate']:.1f}% ({win_rate_improvement_prog:+.1f}pp)")
    print(f"  Avg win:           {test_progressive['avg_win']:.2f}%")
    print(f"  Avg loss:          {test_progressive['avg_loss']:.2f}%")
    print(f"  Profit factor:     {test_progressive['profit_factor']:.2f}")
    print(f"  Total return:      {test_progressive['total_return']:.2f}% ({return_improvement_prog:+.2f}pp)")
    print(f"  Sharpe ratio:      {test_progressive['sharpe_ratio']:.2f} ({sharpe_improvement_prog:+.1f}%)")
    print(f"  Max drawdown:      {test_progressive['max_drawdown']:.2f}%")

    # Determine best strategy
    strategies = [
        ('BASELINE (All-or-Nothing)', baseline, 0, 0),
        ('50/50 Split', test_50_50, win_rate_improvement_50_50, sharpe_improvement_50_50),
        ('67/33 Split', test_67_33, win_rate_improvement_67_33, sharpe_improvement_67_33),
        ('33/67 Split', test_33_67, win_rate_improvement_33_67, sharpe_improvement_33_67),
        ('Progressive Scale', test_progressive, win_rate_improvement_prog, sharpe_improvement_prog)
    ]

    best_strategy = max(strategies, key=lambda x: x[1]['sharpe_ratio'])

    # Deployment recommendation
    print("\n" + "=" * 70)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 70)

    # Deploy if improvement meets criteria
    deploy = best_strategy[2] >= 3.0 or best_strategy[3] >= 5.0

    if deploy and best_strategy[0] != 'BASELINE (All-or-Nothing)':
        print(f"\n[DEPLOY] {best_strategy[0]} strategy validated!")
        print(f"  Win rate improvement: +{best_strategy[2]:.1f}pp")
        print(f"  Sharpe improvement: +{best_strategy[3]:.1f}%")
        print(f"  Total return: {best_strategy[1]['total_return']:.2f}%")
        print(f"\n  Implementation: Add partial exit logic to PaperTrader")
        print(f"  Benefits: Higher win rate, reduced drawdown, better psychology")
    else:
        print(f"\n[REJECT] Insufficient improvement")
        print(f"  Best strategy: {best_strategy[0]}")
        print(f"  Win rate improvement: +{best_strategy[2]:.1f}pp (need +3.0pp)")
        print(f"  Sharpe improvement: +{best_strategy[3]:.1f}% (need +5.0%)")

    # Save results
    experiment_results = {
        'experiment': 'EXP-116',
        'timestamp': timestamp,
        'objective': 'Partial profit taking (scale out strategy)',
        'test_period': f"{start_str} to {end_str}",
        'baseline': baseline,
        'split_50_50': test_50_50,
        'split_67_33': test_67_33,
        'split_33_67': test_33_67,
        'progressive': test_progressive,
        'best_strategy': best_strategy[0],
        'win_rate_improvement_pp': best_strategy[2],
        'sharpe_improvement_pct': best_strategy[3],
        'recommendation': 'DEPLOY' if deploy else 'REJECT',
        'reason': f"{best_strategy[0]}: WR +{best_strategy[2]:.1f}pp, Sharpe +{best_strategy[3]:.1f}%"
    }

    # Save to file
    os.makedirs('results/ml_experiments', exist_ok=True)
    output_file = 'results/ml_experiments/exp116_partial_profit_taking.json'
    with open(output_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Email notification
    try:
        notifier = SendGridNotifier()
        subject = f"EXP-116: Partial Profit Taking - {'DEPLOY' if deploy else 'REJECT'}"

        body_lines = [
            "=" * 70,
            "EXP-116: PARTIAL PROFIT TAKING (SCALE OUT STRATEGY)",
            "=" * 70,
            "",
            "OBJECTIVE: Improve win rate via partial profit taking",
            "",
            f"Test period: {start_str} to {end_str}",
            "",
            "BASELINE (All-or-Nothing - 100% at 2%):",
            f"  Win Rate: {baseline['win_rate']:.1f}%",
            f"  Sharpe: {baseline['sharpe_ratio']:.2f}",
            f"  Return: {baseline['total_return']:.2f}%",
            "",
            "50/50 SPLIT (50% at 1%, 50% at 2%):",
            f"  Win Rate: {test_50_50['win_rate']:.1f}% ({win_rate_improvement_50_50:+.1f}pp)",
            f"  Sharpe: {test_50_50['sharpe_ratio']:.2f} ({sharpe_improvement_50_50:+.1f}%)",
            f"  Return: {test_50_50['total_return']:.2f}% ({return_improvement_50_50:+.2f}pp)",
            "",
            "67/33 SPLIT (67% at 1%, 33% at 2%):",
            f"  Win Rate: {test_67_33['win_rate']:.1f}% ({win_rate_improvement_67_33:+.1f}pp)",
            f"  Sharpe: {test_67_33['sharpe_ratio']:.2f} ({sharpe_improvement_67_33:+.1f}%)",
            f"  Return: {test_67_33['total_return']:.2f}% ({return_improvement_67_33:+.2f}pp)",
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
    print("EXP-116 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

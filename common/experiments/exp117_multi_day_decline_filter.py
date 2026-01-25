"""
EXP-117: Multi-Day Decline Filter (Capitulation Detection)

OBJECTIVE: Improve signal quality by requiring sustained multi-day declines

PROBLEM:
Current system triggers on single-day drops (e.g., stock drops -3% in one day).
But single-day moves can be noise - true capitulation builds over multiple days.

THEORY:
Mean reversion works best when selling pressure is SUSTAINED, not just a single panic day.

Comparison:
- Scenario A: Stock drops -3% today (could reverse tomorrow - weak setup)
- Scenario B: Stock drops -1%, -1%, -1.5% over 3 days (sustained selling - strong setup)

Scenario B indicates:
1. Sellers persistently overwhelming buyers
2. Technical oversold condition building
3. Institutional accumulation likely beginning
4. Higher probability of capitulation and reversal

RESEARCH BASIS:
Academic studies show multi-day declines have higher mean reversion success rates:
- Consecutive down days exhaust sellers
- Creates stronger technical oversold
- Institutional buying often starts on day 3-4
- Higher win rate (+3-5pp typical)

METHODOLOGY:
Test different multi-day decline requirements:
1. BASELINE: No multi-day requirement (current - any single-day drop triggers)
2. TEST: Require 2 consecutive down days
3. TEST: Require 3 consecutive down days
4. TEST: Require 2 of last 3 days down
5. TEST: Require declining 5-day moving average

EXPECTED IMPACT:
- Win rate: +3-5pp (stronger signals)
- Trade frequency: -20-30% (fewer but better signals)
- Sharpe ratio: +5-8%
- Average win: Unchanged
- Profit factor: +8-12%

SUCCESS CRITERIA:
- Win rate improvement >= +3pp
- Sharpe improvement >= +5%
- Deploy optimal multi-day requirement if validated

COMPLEMENTARITY:
- Works alongside EXP-109 (intraday position)
- Both improve signal quality from different angles
- EXP-109: Intraday behavior (closes near low)
- EXP-117: Multi-day behavior (sustained decline)
- Combined: Extremely high-quality signals
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


def check_multi_day_decline(
    ticker: str,
    date_str: str,
    fetcher: YahooFinanceFetcher,
    decline_type: str = 'none'
) -> bool:
    """
    Check if stock meets multi-day decline criteria.

    Args:
        ticker: Stock ticker
        date_str: Current date
        fetcher: Data fetcher
        decline_type: Type of multi-day decline requirement
            - 'none': No requirement (baseline)
            - 'two_consecutive': Require 2 consecutive down days
            - 'three_consecutive': Require 3 consecutive down days
            - 'two_of_three': Require 2 of last 3 days down
            - 'declining_ma': Require declining 5-day moving average

    Returns:
        True if stock meets criteria, False otherwise
    """
    if decline_type == 'none':
        return True  # No filter - all signals pass

    try:
        # Fetch last 10 days of data
        lookback_start = (datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=15)).strftime('%Y-%m-%d')
        data = fetcher.fetch_stock_data(ticker, start_date=lookback_start, end_date=date_str)

        if data is None or data.empty or len(data) < 3:
            return False  # Not enough data

        # Calculate daily returns
        data['return'] = data['Close'].pct_change() * 100

        if decline_type == 'two_consecutive':
            # Require last 2 days both down
            if len(data) < 2:
                return False
            last_two_returns = data['return'].iloc[-2:].values
            return all(r < 0 for r in last_two_returns if not np.isnan(r))

        elif decline_type == 'three_consecutive':
            # Require last 3 days all down
            if len(data) < 3:
                return False
            last_three_returns = data['return'].iloc[-3:].values
            return all(r < 0 for r in last_three_returns if not np.isnan(r))

        elif decline_type == 'two_of_three':
            # Require at least 2 of last 3 days down
            if len(data) < 3:
                return False
            last_three_returns = data['return'].iloc[-3:].values
            down_days = sum(1 for r in last_three_returns if not np.isnan(r) and r < 0)
            return down_days >= 2

        elif decline_type == 'declining_ma':
            # Require 5-day MA is declining
            if len(data) < 5:
                return False
            data['ma5'] = data['Close'].rolling(window=5).mean()
            if len(data) < 6:  # Need at least 6 days to compare 2 MA values
                return False
            ma_current = data['ma5'].iloc[-1]
            ma_previous = data['ma5'].iloc[-2]
            return ma_current < ma_previous

        else:
            return True

    except Exception as e:
        print(f"  [ERROR] Failed to check multi-day decline for {ticker}: {e}")
        return False


def backtest_with_multi_day_filter(
    start_date: str,
    end_date: str,
    decline_type: str = 'none'
) -> dict:
    """
    Backtest strategy with multi-day decline filter.

    Args:
        start_date: Start date for backtest
        end_date: End date for backtest
        decline_type: Multi-day decline requirement type

    Returns:
        Performance metrics dictionary
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: Multi-Day Filter = {decline_type.upper()}")
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
    signals_filtered = 0
    entries_executed = 0
    exits_executed = 0

    for date in dates:
        date_str = date.strftime('%Y-%m-%d')

        # Get market regime
        regime = scanner.get_market_regime()
        if regime == 'BEAR':
            continue

        # Scan for signals
        signals = scanner.scan_all_stocks(date_str)
        signals_found += len(signals)

        # Apply multi-day decline filter
        if decline_type != 'none' and signals:
            filtered_signals = []
            for signal in signals:
                ticker = signal['ticker']
                if check_multi_day_decline(ticker, date_str, fetcher, decline_type):
                    filtered_signals.append(signal)
                else:
                    signals_filtered += 1

            signals = filtered_signals

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

    # Calculate filter rate
    filter_rate = (signals_filtered / signals_found * 100) if signals_found > 0 else 0

    result = {
        'decline_type': decline_type,
        'signals_found': signals_found,
        'signals_filtered': signals_filtered,
        'filter_rate_pct': filter_rate,
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
        'final_equity': stats.get('current_equity', 100000)
    }

    return result


def main():
    """Run EXP-117 experiment."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("=" * 70)
    print("EXP-117: MULTI-DAY DECLINE FILTER (CAPITULATION DETECTION)")
    print("=" * 70)
    print(f"Started: {timestamp}")
    print()
    print("OBJECTIVE: Improve signal quality via multi-day decline requirements")
    print("THEORY: Sustained selling creates stronger mean reversion setups")
    print()

    # Test period: Last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}")
    print()

    # Baseline (no multi-day requirement)
    print("\n" + "=" * 70)
    print("BASELINE: No Multi-Day Requirement")
    print("=" * 70)
    baseline = backtest_with_multi_day_filter(start_str, end_str, decline_type='none')

    print(f"\nBaseline Results:")
    print(f"  Trades executed:   {baseline['total_trades']}")
    print(f"  Win rate:          {baseline['win_rate']:.1f}%")
    print(f"  Avg win:           {baseline['avg_win']:.2f}%")
    print(f"  Avg loss:          {baseline['avg_loss']:.2f}%")
    print(f"  Profit factor:     {baseline['profit_factor']:.2f}")
    print(f"  Total return:      {baseline['total_return']:.2f}%")
    print(f"  Sharpe ratio:      {baseline['sharpe_ratio']:.2f}")
    print(f"  Max drawdown:      {baseline['max_drawdown']:.2f}%")

    # Test 1: 2 consecutive down days
    print("\n" + "=" * 70)
    print("TEST 1: Require 2 Consecutive Down Days")
    print("=" * 70)
    test_two_consec = backtest_with_multi_day_filter(start_str, end_str, decline_type='two_consecutive')

    win_rate_improvement_2c = test_two_consec['win_rate'] - baseline['win_rate']
    sharpe_improvement_2c = ((test_two_consec['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100) if baseline['sharpe_ratio'] != 0 else 0
    return_improvement_2c = test_two_consec['total_return'] - baseline['total_return']

    print(f"\n2 Consecutive Results:")
    print(f"  Trades executed:   {test_two_consec['total_trades']}")
    print(f"  Signals filtered:  {test_two_consec['signals_filtered']} ({test_two_consec['filter_rate_pct']:.1f}%)")
    print(f"  Win rate:          {test_two_consec['win_rate']:.1f}% ({win_rate_improvement_2c:+.1f}pp)")
    print(f"  Avg win:           {test_two_consec['avg_win']:.2f}%")
    print(f"  Avg loss:          {test_two_consec['avg_loss']:.2f}%")
    print(f"  Profit factor:     {test_two_consec['profit_factor']:.2f}")
    print(f"  Total return:      {test_two_consec['total_return']:.2f}% ({return_improvement_2c:+.2f}pp)")
    print(f"  Sharpe ratio:      {test_two_consec['sharpe_ratio']:.2f} ({sharpe_improvement_2c:+.1f}%)")
    print(f"  Max drawdown:      {test_two_consec['max_drawdown']:.2f}%")

    # Test 2: 3 consecutive down days
    print("\n" + "=" * 70)
    print("TEST 2: Require 3 Consecutive Down Days")
    print("=" * 70)
    test_three_consec = backtest_with_multi_day_filter(start_str, end_str, decline_type='three_consecutive')

    win_rate_improvement_3c = test_three_consec['win_rate'] - baseline['win_rate']
    sharpe_improvement_3c = ((test_three_consec['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100) if baseline['sharpe_ratio'] != 0 else 0
    return_improvement_3c = test_three_consec['total_return'] - baseline['total_return']

    print(f"\n3 Consecutive Results:")
    print(f"  Trades executed:   {test_three_consec['total_trades']}")
    print(f"  Signals filtered:  {test_three_consec['signals_filtered']} ({test_three_consec['filter_rate_pct']:.1f}%)")
    print(f"  Win rate:          {test_three_consec['win_rate']:.1f}% ({win_rate_improvement_3c:+.1f}pp)")
    print(f"  Avg win:           {test_three_consec['avg_win']:.2f}%")
    print(f"  Avg loss:          {test_three_consec['avg_loss']:.2f}%")
    print(f"  Profit factor:     {test_three_consec['profit_factor']:.2f}")
    print(f"  Total return:      {test_three_consec['total_return']:.2f}% ({return_improvement_3c:+.2f}pp)")
    print(f"  Sharpe ratio:      {test_three_consec['sharpe_ratio']:.2f} ({sharpe_improvement_3c:+.1f}%)")
    print(f"  Max drawdown:      {test_three_consec['max_drawdown']:.2f}%")

    # Test 3: 2 of 3 days down
    print("\n" + "=" * 70)
    print("TEST 3: Require 2 of Last 3 Days Down")
    print("=" * 70)
    test_two_of_three = backtest_with_multi_day_filter(start_str, end_str, decline_type='two_of_three')

    win_rate_improvement_2o3 = test_two_of_three['win_rate'] - baseline['win_rate']
    sharpe_improvement_2o3 = ((test_two_of_three['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100) if baseline['sharpe_ratio'] != 0 else 0
    return_improvement_2o3 = test_two_of_three['total_return'] - baseline['total_return']

    print(f"\n2 of 3 Days Results:")
    print(f"  Trades executed:   {test_two_of_three['total_trades']}")
    print(f"  Signals filtered:  {test_two_of_three['signals_filtered']} ({test_two_of_three['filter_rate_pct']:.1f}%)")
    print(f"  Win rate:          {test_two_of_three['win_rate']:.1f}% ({win_rate_improvement_2o3:+.1f}pp)")
    print(f"  Avg win:           {test_two_of_three['avg_win']:.2f}%")
    print(f"  Avg loss:          {test_two_of_three['avg_loss']:.2f}%")
    print(f"  Profit factor:     {test_two_of_three['profit_factor']:.2f}")
    print(f"  Total return:      {test_two_of_three['total_return']:.2f}% ({return_improvement_2o3:+.2f}pp)")
    print(f"  Sharpe ratio:      {test_two_of_three['sharpe_ratio']:.2f} ({sharpe_improvement_2o3:+.1f}%)")
    print(f"  Max drawdown:      {test_two_of_three['max_drawdown']:.2f}%")

    # Test 4: Declining 5-day MA
    print("\n" + "=" * 70)
    print("TEST 4: Require Declining 5-Day Moving Average")
    print("=" * 70)
    test_declining_ma = backtest_with_multi_day_filter(start_str, end_str, decline_type='declining_ma')

    win_rate_improvement_ma = test_declining_ma['win_rate'] - baseline['win_rate']
    sharpe_improvement_ma = ((test_declining_ma['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100) if baseline['sharpe_ratio'] != 0 else 0
    return_improvement_ma = test_declining_ma['total_return'] - baseline['total_return']

    print(f"\nDeclining MA Results:")
    print(f"  Trades executed:   {test_declining_ma['total_trades']}")
    print(f"  Signals filtered:  {test_declining_ma['signals_filtered']} ({test_declining_ma['filter_rate_pct']:.1f}%)")
    print(f"  Win rate:          {test_declining_ma['win_rate']:.1f}% ({win_rate_improvement_ma:+.1f}pp)")
    print(f"  Avg win:           {test_declining_ma['avg_win']:.2f}%")
    print(f"  Avg loss:          {test_declining_ma['avg_loss']:.2f}%")
    print(f"  Profit factor:     {test_declining_ma['profit_factor']:.2f}")
    print(f"  Total return:      {test_declining_ma['total_return']:.2f}% ({return_improvement_ma:+.2f}pp)")
    print(f"  Sharpe ratio:      {test_declining_ma['sharpe_ratio']:.2f} ({sharpe_improvement_ma:+.1f}%)")
    print(f"  Max drawdown:      {test_declining_ma['max_drawdown']:.2f}%")

    # Determine best strategy
    strategies = [
        ('BASELINE (No Multi-Day)', baseline, 0, 0),
        ('2 Consecutive Down', test_two_consec, win_rate_improvement_2c, sharpe_improvement_2c),
        ('3 Consecutive Down', test_three_consec, win_rate_improvement_3c, sharpe_improvement_3c),
        ('2 of 3 Days Down', test_two_of_three, win_rate_improvement_2o3, sharpe_improvement_2o3),
        ('Declining 5-Day MA', test_declining_ma, win_rate_improvement_ma, sharpe_improvement_ma)
    ]

    best_strategy = max(strategies, key=lambda x: x[1]['sharpe_ratio'])

    # Deployment recommendation
    print("\n" + "=" * 70)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 70)

    # Deploy if improvement meets criteria
    deploy = best_strategy[2] >= 3.0 or best_strategy[3] >= 5.0

    if deploy and best_strategy[0] != 'BASELINE (No Multi-Day)':
        print(f"\n[DEPLOY] {best_strategy[0]} filter validated!")
        print(f"  Win rate improvement: +{best_strategy[2]:.1f}pp")
        print(f"  Sharpe improvement: +{best_strategy[3]:.1f}%")
        print(f"  Total return: {best_strategy[1]['total_return']:.2f}%")
        print(f"\n  Implementation: Add multi-day decline check to SignalScanner")
        print(f"  Benefits: Higher quality signals, stronger mean reversion setups")
    else:
        print(f"\n[REJECT] Insufficient improvement")
        print(f"  Best strategy: {best_strategy[0]}")
        print(f"  Win rate improvement: +{best_strategy[2]:.1f}pp (need +3.0pp)")
        print(f"  Sharpe improvement: +{best_strategy[3]:.1f}% (need +5.0%)")

    # Save results
    experiment_results = {
        'experiment': 'EXP-117',
        'timestamp': timestamp,
        'objective': 'Multi-day decline filter (capitulation detection)',
        'test_period': f"{start_str} to {end_str}",
        'baseline': baseline,
        'two_consecutive': test_two_consec,
        'three_consecutive': test_three_consec,
        'two_of_three': test_two_of_three,
        'declining_ma': test_declining_ma,
        'best_strategy': best_strategy[0],
        'win_rate_improvement_pp': best_strategy[2],
        'sharpe_improvement_pct': best_strategy[3],
        'recommendation': 'DEPLOY' if deploy else 'REJECT',
        'reason': f"{best_strategy[0]}: WR +{best_strategy[2]:.1f}pp, Sharpe +{best_strategy[3]:.1f}%"
    }

    # Save to file
    os.makedirs('results/ml_experiments', exist_ok=True)
    output_file = 'results/ml_experiments/exp117_multi_day_decline_filter.json'
    with open(output_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Email notification
    try:
        notifier = SendGridNotifier()
        subject = f"EXP-117: Multi-Day Decline Filter - {'DEPLOY' if deploy else 'REJECT'}"

        body_lines = [
            "=" * 70,
            "EXP-117: MULTI-DAY DECLINE FILTER (CAPITULATION DETECTION)",
            "=" * 70,
            "",
            "OBJECTIVE: Improve signal quality via multi-day decline requirements",
            "",
            f"Test period: {start_str} to {end_str}",
            "",
            "BASELINE (No Multi-Day):",
            f"  Win Rate: {baseline['win_rate']:.1f}%",
            f"  Sharpe: {baseline['sharpe_ratio']:.2f}",
            f"  Return: {baseline['total_return']:.2f}%",
            "",
            "2 CONSECUTIVE DOWN:",
            f"  Win Rate: {test_two_consec['win_rate']:.1f}% ({win_rate_improvement_2c:+.1f}pp)",
            f"  Sharpe: {test_two_consec['sharpe_ratio']:.2f} ({sharpe_improvement_2c:+.1f}%)",
            f"  Return: {test_two_consec['total_return']:.2f}% ({return_improvement_2c:+.2f}pp)",
            "",
            "3 CONSECUTIVE DOWN:",
            f"  Win Rate: {test_three_consec['win_rate']:.1f}% ({win_rate_improvement_3c:+.1f}pp)",
            f"  Sharpe: {test_three_consec['sharpe_ratio']:.2f} ({sharpe_improvement_3c:+.1f}%)",
            f"  Return: {test_three_consec['total_return']:.2f}% ({return_improvement_3c:+.2f}pp)",
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
    print("EXP-117 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
EXP-112: Signal-Quality-Based Exit Thresholds

OBJECTIVE: Improve profit factor by giving elite signals room to run, quick exits on weak signals

PROBLEM:
Current system uses fixed 2% profit / -2% stop for ALL signals regardless of quality.
But elite signals (90-100 strength) have higher conviction and deserve wider targets,
while marginal signals (65-69 strength) should exit quickly to minimize risk.

THEORY:
Signal strength already quantifies our conviction in each trade.
- Elite signals: Stronger technical setup → higher expected move → wider targets
- Weak signals: Marginal setup → lower expected move → tighter targets

METHODOLOGY:
- Baseline: Fixed 2% profit / -2% stop (current)
- Enhanced: Quality-based dynamic exits:
  * Elite (90-100 strength): 2.5% profit, -2.5% stop (let winners run)
  * Strong (80-89 strength): 2.0% profit, -2.0% stop (baseline)
  * Good (70-79 strength): 2.0% profit, -2.0% stop (baseline)
  * Acceptable (65-69 strength): 1.5% profit, -1.5% stop (quick exit)

EXPECTED IMPACT:
- Profit factor: +10-15% improvement (capture bigger moves from elite signals)
- Sharpe ratio: +5-8% improvement
- Win rate: Potentially higher (quick exits on weak signals reduce losses)
- Average win: Higher (elite signals allowed to run further)

SUCCESS CRITERIA:
- Profit factor improvement >= +8%
- Sharpe ratio improvement >= +5%
- Deploy if validated

COMPLEMENTARITY:
- EXP-111: Volatility-based exits (ATR dimension)
- EXP-112: Quality-based exits (signal strength dimension)
- Future: Could combine both for multi-factor exit optimization
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


def get_quality_based_exits(signal_strength: float) -> tuple:
    """
    Calculate dynamic profit target and stop loss based on signal quality.

    Args:
        signal_strength: Signal strength score (0-100)

    Returns:
        Tuple of (profit_target, stop_loss)
    """
    if signal_strength >= 90:
        # Elite signals - let winners run
        return (2.5, -2.5)
    elif signal_strength >= 80:
        # Strong signals - baseline
        return (2.0, -2.0)
    elif signal_strength >= 70:
        # Good signals - baseline
        return (2.0, -2.0)
    else:
        # Acceptable signals (65-69) - quick exit
        return (1.5, -1.5)


def backtest_with_quality_exits(
    start_date: str,
    end_date: str,
    use_quality_exits: bool = False
) -> dict:
    """
    Backtest strategy with or without quality-based exit thresholds.

    Args:
        start_date: Start date for backtest
        end_date: End date for backtest
        use_quality_exits: Whether to use signal-quality-based dynamic exits

    Returns:
        Performance metrics dictionary
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {'QUALITY-BASED EXITS' if use_quality_exits else 'FIXED EXITS (BASELINE)'}")
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
        profit_target=2.0,  # Will be overridden if dynamic
        stop_loss=-2.0,     # Will be overridden if dynamic
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
    exit_adjustments = []

    for date in dates:
        date_str = date.strftime('%Y-%m-%d')

        # Get market regime
        regime = scanner.get_market_regime()
        if regime == 'BEAR':
            continue

        # Scan for signals
        signals = scanner.scan_all_stocks(date_str)
        signals_found += len(signals)

        # Set quality-based exits if enabled
        if use_quality_exits and signals:
            for signal in signals:
                signal_strength = signal.get('signal_strength', 70.0)
                profit_target, stop_loss = get_quality_based_exits(signal_strength)

                signal['profit_target'] = profit_target
                signal['stop_loss'] = stop_loss

                exit_adjustments.append({
                    'date': date_str,
                    'ticker': signal['ticker'],
                    'signal_strength': signal_strength,
                    'profit_target': profit_target,
                    'stop_loss': stop_loss
                })

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

    # Calculate average exit adjustments if used
    avg_profit_target = None
    avg_stop_loss = None
    quality_distribution = {}
    if exit_adjustments:
        avg_profit_target = np.mean([a['profit_target'] for a in exit_adjustments])
        avg_stop_loss = np.mean([a['stop_loss'] for a in exit_adjustments])

        # Count signals by quality tier
        for adj in exit_adjustments:
            strength = adj['signal_strength']
            if strength >= 90:
                tier = 'Elite (90-100)'
            elif strength >= 80:
                tier = 'Strong (80-89)'
            elif strength >= 70:
                tier = 'Good (70-79)'
            else:
                tier = 'Acceptable (65-69)'

            quality_distribution[tier] = quality_distribution.get(tier, 0) + 1

    result = {
        'use_quality_exits': use_quality_exits,
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
        'avg_profit_target': avg_profit_target,
        'avg_stop_loss': avg_stop_loss,
        'quality_distribution': quality_distribution,
        'exit_adjustments_count': len(exit_adjustments) if exit_adjustments else 0
    }

    return result


def main():
    """Run EXP-112 experiment."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("=" * 70)
    print("EXP-112: SIGNAL-QUALITY-BASED EXIT THRESHOLDS")
    print("=" * 70)
    print(f"Started: {timestamp}")
    print()
    print("OBJECTIVE: Give elite signals room to run, quick exits on weak signals")
    print("THEORY: Signal strength quantifies conviction - adjust exits accordingly")
    print()

    # Test period: Last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}")
    print()

    # Baseline (fixed exits)
    print("\n" + "=" * 70)
    print("BASELINE: Fixed 2% profit / -2% stop")
    print("=" * 70)
    baseline = backtest_with_quality_exits(start_str, end_str, use_quality_exits=False)

    print(f"\nBaseline Results:")
    print(f"  Trades executed:   {baseline['total_trades']}")
    print(f"  Win rate:          {baseline['win_rate']:.1f}%")
    print(f"  Avg win:           {baseline['avg_win']:.2f}%")
    print(f"  Avg loss:          {baseline['avg_loss']:.2f}%")
    print(f"  Profit factor:     {baseline['profit_factor']:.2f}")
    print(f"  Total return:      {baseline['total_return']:.2f}%")
    print(f"  Sharpe ratio:      {baseline['sharpe_ratio']:.2f}")
    print(f"  Max drawdown:      {baseline['max_drawdown']:.2f}%")

    # Enhanced (quality-based exits)
    print("\n" + "=" * 70)
    print("ENHANCED: Signal-Quality-Based Exits")
    print("=" * 70)
    enhanced = backtest_with_quality_exits(start_str, end_str, use_quality_exits=True)

    # Calculate improvements
    profit_factor_improvement = ((enhanced['profit_factor'] - baseline['profit_factor']) / baseline['profit_factor'] * 100) if baseline['profit_factor'] != 0 else 0
    sharpe_improvement = ((enhanced['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100) if baseline['sharpe_ratio'] != 0 else 0
    return_improvement = enhanced['total_return'] - baseline['total_return']
    win_rate_improvement = enhanced['win_rate'] - baseline['win_rate']

    print(f"\nEnhanced Results:")
    print(f"  Trades executed:   {enhanced['total_trades']}")
    print(f"  Win rate:          {enhanced['win_rate']:.1f}% ({win_rate_improvement:+.1f}pp)")
    print(f"  Avg win:           {enhanced['avg_win']:.2f}%")
    print(f"  Avg loss:          {enhanced['avg_loss']:.2f}%")
    print(f"  Profit factor:     {enhanced['profit_factor']:.2f} ({profit_factor_improvement:+.1f}%)")
    print(f"  Total return:      {enhanced['total_return']:.2f}% ({return_improvement:+.2f}pp)")
    print(f"  Sharpe ratio:      {enhanced['sharpe_ratio']:.2f} ({sharpe_improvement:+.1f}%)")
    print(f"  Max drawdown:      {enhanced['max_drawdown']:.2f}%")

    if enhanced['avg_profit_target']:
        print(f"\n  Avg profit target: {enhanced['avg_profit_target']:.2f}%")
        print(f"  Avg stop loss:     {enhanced['avg_stop_loss']:.2f}%")

    if enhanced['quality_distribution']:
        print(f"\n  Signal Quality Distribution:")
        for tier, count in sorted(enhanced['quality_distribution'].items()):
            print(f"    {tier}: {count} signals")

    # Deployment recommendation
    print("\n" + "=" * 70)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 70)

    deploy = profit_factor_improvement >= 8.0 or sharpe_improvement >= 5.0

    if deploy:
        print(f"\n[DEPLOY] Enhancement validated!")
        print(f"  Profit factor improvement: +{profit_factor_improvement:.1f}%")
        print(f"  Sharpe improvement: +{sharpe_improvement:.1f}%")
        print(f"  Return improvement: +{return_improvement:.2f}pp")
        print(f"\n  Implementation: Add quality-based dynamic exit thresholds to PaperTrader")
        print(f"  Complementary to EXP-111 (ATR-based exits)")
    else:
        print(f"\n[REJECT] Insufficient improvement")
        print(f"  Profit factor improvement: +{profit_factor_improvement:.1f}% (need +8.0%)")
        print(f"  Sharpe improvement: +{sharpe_improvement:.1f}% (need +5.0%)")

    # Save results
    experiment_results = {
        'experiment': 'EXP-112',
        'timestamp': timestamp,
        'objective': 'Signal-quality-based exit thresholds',
        'test_period': f"{start_str} to {end_str}",
        'baseline': baseline,
        'enhanced': enhanced,
        'profit_factor_improvement_pct': profit_factor_improvement,
        'sharpe_improvement_pct': sharpe_improvement,
        'return_improvement_pp': return_improvement,
        'win_rate_improvement_pp': win_rate_improvement,
        'recommendation': 'DEPLOY' if deploy else 'REJECT',
        'reason': f"Profit Factor: +{profit_factor_improvement:.1f}%, Sharpe: +{sharpe_improvement:.1f}%, Return: +{return_improvement:.2f}pp"
    }

    # Save to file
    os.makedirs('results/ml_experiments', exist_ok=True)
    output_file = 'results/ml_experiments/exp112_signal_quality_exit_thresholds.json'
    with open(output_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Email notification
    try:
        notifier = SendGridNotifier()
        subject = f"EXP-112: Quality-Based Exit Thresholds - {'DEPLOY' if deploy else 'REJECT'}"

        body_lines = [
            "=" * 70,
            "EXP-112: SIGNAL-QUALITY-BASED EXIT THRESHOLDS",
            "=" * 70,
            "",
            "OBJECTIVE: Elite signals get wider targets, weak signals exit quickly",
            "",
            f"Test period: {start_str} to {end_str}",
            "",
            "BASELINE (Fixed 2% / -2%):",
            f"  Profit Factor: {baseline['profit_factor']:.2f}",
            f"  Sharpe: {baseline['sharpe_ratio']:.2f}",
            f"  Return: {baseline['total_return']:.2f}%",
            "",
            "ENHANCED (Quality-Based Dynamic):",
            f"  Profit Factor: {enhanced['profit_factor']:.2f} (+{profit_factor_improvement:.1f}%)",
            f"  Sharpe: {enhanced['sharpe_ratio']:.2f} (+{sharpe_improvement:.1f}%)",
            f"  Return: {enhanced['total_return']:.2f}% (+{return_improvement:.2f}pp)",
            "",
            "EXIT THRESHOLDS:",
            "  Elite (90-100): 2.5% profit / -2.5% stop",
            "  Strong (80-89): 2.0% profit / -2.0% stop",
            "  Good (70-79): 2.0% profit / -2.0% stop",
            "  Acceptable (65-69): 1.5% profit / -1.5% stop",
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
    print("EXP-112 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

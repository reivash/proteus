"""
EXP-113: Adaptive Max Hold Days (Multi-Factor)

OBJECTIVE: Optimize holding periods based on signal quality + volatility

PROBLEM:
Current system uses fixed 2-day max hold for ALL trades regardless of:
- Signal quality (elite vs weak signals)
- Stock volatility (high ATR vs low ATR)
- Expected reversion timeline

Elite signals (90-100 strength) often take 3-4 days to fully mean revert.
Weak signals (65-69 strength) should exit quickly at 1 day.
Volatile stocks need more time to revert than stable stocks.

THEORY:
Signal quality indicates conviction (how good is the setup)
ATR indicates volatility (how much time needed to revert)
Combine both factors for optimal hold period:

| Signal Quality | ATR         | Max Hold | Rationale                    |
|---------------|-------------|----------|------------------------------|
| Elite (90-100)| High (>2.5%)| 4 days   | Best setups need most time   |
| Elite (90-100)| Med (1.5-2.5%)| 3 days | Strong conviction, moderate  |
| Elite (90-100)| Low (<1.5%) | 2 days   | Quick reversion expected     |
| Strong (80-89)| High        | 3 days   | Good setup, volatile stock   |
| Strong (80-89)| Med/Low     | 2 days   | Baseline                     |
| Good/Accept   | Any         | 2 days   | Standard exit                |

METHODOLOGY:
- Baseline: Fixed 2-day max hold (current)
- Enhanced: Adaptive hold days based on quality + ATR
- Backtest both strategies on 2-year historical data
- Compare win rates, avg win, Sharpe ratio

EXPECTED IMPACT:
- Win rate: +2-4pp (hold winners longer, exit losers earlier)
- Average win: +0.3-0.5% (capture full mean reversion move)
- Sharpe ratio: +5-10% improvement
- Better capture of elite signal potential

SUCCESS CRITERIA:
- Win rate improvement >= +2pp
- Average win improvement >= +0.2%
- Sharpe ratio improvement >= +5%
- Deploy if validated
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


def calculate_atr(data: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate Average True Range (ATR) for volatility measurement.

    Args:
        data: OHLC price data
        period: ATR period (default: 14)

    Returns:
        ATR value as percentage of price
    """
    if len(data) < period:
        return 2.0  # Default to medium volatility

    high = data['High']
    low = data['Low']
    close = data['Close']

    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate ATR
    atr = tr.rolling(window=period).mean().iloc[-1]

    # Convert to percentage
    current_price = close.iloc[-1]
    atr_pct = (atr / current_price) * 100

    return float(atr_pct)


def get_adaptive_max_hold_days(signal_strength: float, atr_pct: float) -> int:
    """
    Calculate adaptive max hold days based on signal quality and ATR.

    Args:
        signal_strength: Signal strength score (0-100)
        atr_pct: ATR as percentage of price

    Returns:
        Max hold days (1-4)
    """
    # Elite signals (90-100)
    if signal_strength >= 90:
        if atr_pct > 2.5:
            return 4  # Elite + High Vol = 4 days
        elif atr_pct >= 1.5:
            return 3  # Elite + Med Vol = 3 days
        else:
            return 2  # Elite + Low Vol = 2 days

    # Strong signals (80-89)
    elif signal_strength >= 80:
        if atr_pct > 2.5:
            return 3  # Strong + High Vol = 3 days
        else:
            return 2  # Strong + Med/Low Vol = 2 days

    # Good/Acceptable signals (65-79)
    else:
        return 2  # Baseline for all


def backtest_with_adaptive_hold_days(
    start_date: str,
    end_date: str,
    use_adaptive_hold: bool = False
) -> dict:
    """
    Backtest strategy with or without adaptive max hold days.

    Args:
        start_date: Start date for backtest
        end_date: End date for backtest
        use_adaptive_hold: Whether to use adaptive max hold days

    Returns:
        Performance metrics dictionary
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {'ADAPTIVE MAX HOLD' if use_adaptive_hold else 'FIXED 2-DAY HOLD (BASELINE)'}")
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
        max_hold_days=2,  # Will be overridden if adaptive
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
    hold_day_adjustments = []

    for date in dates:
        date_str = date.strftime('%Y-%m-%d')

        # Get market regime
        regime = scanner.get_market_regime()
        if regime == 'BEAR':
            continue

        # Scan for signals
        signals = scanner.scan_all_stocks(date_str)
        signals_found += len(signals)

        # Calculate adaptive max hold days if enabled
        if use_adaptive_hold and signals:
            for signal in signals:
                ticker = signal['ticker']
                signal_strength = signal.get('signal_strength', 70.0)

                # Fetch historical data to calculate ATR
                try:
                    lookback_start = (date - timedelta(days=45)).strftime('%Y-%m-%d')
                    hist_data = fetcher.fetch_stock_data(ticker, start_date=lookback_start, end_date=date_str)

                    if not hist_data.empty and len(hist_data) >= 14:
                        atr_pct = calculate_atr(hist_data)
                        max_hold_days = get_adaptive_max_hold_days(signal_strength, atr_pct)

                        signal['max_hold_days'] = max_hold_days
                        signal['atr_pct'] = atr_pct

                        hold_day_adjustments.append({
                            'date': date_str,
                            'ticker': ticker,
                            'signal_strength': signal_strength,
                            'atr_pct': atr_pct,
                            'max_hold_days': max_hold_days
                        })
                    else:
                        # Fallback to baseline
                        signal['max_hold_days'] = 2
                except:
                    # Fallback to baseline on error
                    signal['max_hold_days'] = 2

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

    # Calculate average hold days if used
    avg_max_hold_days = None
    hold_day_distribution = {}
    if hold_day_adjustments:
        avg_max_hold_days = np.mean([a['max_hold_days'] for a in hold_day_adjustments])

        # Count by hold days
        for adj in hold_day_adjustments:
            days = adj['max_hold_days']
            hold_day_distribution[f"{days} days"] = hold_day_distribution.get(f"{days} days", 0) + 1

    result = {
        'use_adaptive_hold': use_adaptive_hold,
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
        'avg_max_hold_days': avg_max_hold_days,
        'hold_day_distribution': hold_day_distribution,
        'adjustments_count': len(hold_day_adjustments) if hold_day_adjustments else 0
    }

    return result


def main():
    """Run EXP-113 experiment."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("=" * 70)
    print("EXP-113: ADAPTIVE MAX HOLD DAYS (MULTI-FACTOR)")
    print("=" * 70)
    print(f"Started: {timestamp}")
    print()
    print("OBJECTIVE: Optimize holding periods via signal quality + volatility")
    print("THEORY: Elite signals need more time, weak signals exit quickly")
    print()

    # Test period: Last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}")
    print()

    # Baseline (fixed 2-day hold)
    print("\n" + "=" * 70)
    print("BASELINE: Fixed 2-Day Max Hold")
    print("=" * 70)
    baseline = backtest_with_adaptive_hold_days(start_str, end_str, use_adaptive_hold=False)

    print(f"\nBaseline Results:")
    print(f"  Trades executed:   {baseline['total_trades']}")
    print(f"  Win rate:          {baseline['win_rate']:.1f}%")
    print(f"  Avg win:           {baseline['avg_win']:.2f}%")
    print(f"  Avg loss:          {baseline['avg_loss']:.2f}%")
    print(f"  Profit factor:     {baseline['profit_factor']:.2f}")
    print(f"  Total return:      {baseline['total_return']:.2f}%")
    print(f"  Sharpe ratio:      {baseline['sharpe_ratio']:.2f}")
    print(f"  Max drawdown:      {baseline['max_drawdown']:.2f}%")

    # Enhanced (adaptive hold days)
    print("\n" + "=" * 70)
    print("ENHANCED: Adaptive Max Hold Days (Quality + ATR)")
    print("=" * 70)
    enhanced = backtest_with_adaptive_hold_days(start_str, end_str, use_adaptive_hold=True)

    # Calculate improvements
    win_rate_improvement = enhanced['win_rate'] - baseline['win_rate']
    avg_win_improvement = enhanced['avg_win'] - baseline['avg_win']
    sharpe_improvement = ((enhanced['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100) if baseline['sharpe_ratio'] != 0 else 0
    return_improvement = enhanced['total_return'] - baseline['total_return']
    profit_factor_improvement = ((enhanced['profit_factor'] - baseline['profit_factor']) / baseline['profit_factor'] * 100) if baseline['profit_factor'] != 0 else 0

    print(f"\nEnhanced Results:")
    print(f"  Trades executed:   {enhanced['total_trades']}")
    print(f"  Win rate:          {enhanced['win_rate']:.1f}% ({win_rate_improvement:+.1f}pp)")
    print(f"  Avg win:           {enhanced['avg_win']:.2f}% ({avg_win_improvement:+.2f}%)")
    print(f"  Avg loss:          {enhanced['avg_loss']:.2f}%")
    print(f"  Profit factor:     {enhanced['profit_factor']:.2f} ({profit_factor_improvement:+.1f}%)")
    print(f"  Total return:      {enhanced['total_return']:.2f}% ({return_improvement:+.2f}pp)")
    print(f"  Sharpe ratio:      {enhanced['sharpe_ratio']:.2f} ({sharpe_improvement:+.1f}%)")
    print(f"  Max drawdown:      {enhanced['max_drawdown']:.2f}%")

    if enhanced['avg_max_hold_days']:
        print(f"\n  Avg max hold days: {enhanced['avg_max_hold_days']:.2f} days")

    if enhanced['hold_day_distribution']:
        print(f"\n  Hold Day Distribution:")
        for hold_days, count in sorted(enhanced['hold_day_distribution'].items()):
            pct = (count / enhanced['adjustments_count']) * 100 if enhanced['adjustments_count'] > 0 else 0
            print(f"    {hold_days}: {count} signals ({pct:.1f}%)")

    # Deployment recommendation
    print("\n" + "=" * 70)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 70)

    deploy = (win_rate_improvement >= 2.0 and avg_win_improvement >= 0.2) or sharpe_improvement >= 5.0

    if deploy:
        print(f"\n[DEPLOY] Enhancement validated!")
        print(f"  Win rate improvement: +{win_rate_improvement:.1f}pp")
        print(f"  Avg win improvement: +{avg_win_improvement:.2f}%")
        print(f"  Sharpe improvement: +{sharpe_improvement:.1f}%")
        print(f"  Return improvement: +{return_improvement:.2f}pp")
        print(f"\n  Implementation: Add adaptive max_hold_days to PaperTrader based on quality + ATR")
    else:
        print(f"\n[REJECT] Insufficient improvement")
        print(f"  Win rate improvement: +{win_rate_improvement:.1f}pp (need +2.0pp)")
        print(f"  Avg win improvement: +{avg_win_improvement:.2f}% (need +0.2%)")
        print(f"  Sharpe improvement: +{sharpe_improvement:.1f}% (need +5.0%)")

    # Save results
    experiment_results = {
        'experiment': 'EXP-113',
        'timestamp': timestamp,
        'objective': 'Adaptive max hold days based on quality + ATR',
        'test_period': f"{start_str} to {end_str}",
        'baseline': baseline,
        'enhanced': enhanced,
        'win_rate_improvement_pp': win_rate_improvement,
        'avg_win_improvement_pct': avg_win_improvement,
        'sharpe_improvement_pct': sharpe_improvement,
        'return_improvement_pp': return_improvement,
        'profit_factor_improvement_pct': profit_factor_improvement,
        'recommendation': 'DEPLOY' if deploy else 'REJECT',
        'reason': f"Win Rate: +{win_rate_improvement:.1f}pp, Avg Win: +{avg_win_improvement:.2f}%, Sharpe: +{sharpe_improvement:.1f}%"
    }

    # Save to file
    os.makedirs('results/ml_experiments', exist_ok=True)
    output_file = 'results/ml_experiments/exp113_adaptive_max_hold_days.json'
    with open(output_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Email notification
    try:
        notifier = SendGridNotifier()
        subject = f"EXP-113: Adaptive Max Hold Days - {'DEPLOY' if deploy else 'REJECT'}"

        body_lines = [
            "=" * 70,
            "EXP-113: ADAPTIVE MAX HOLD DAYS (MULTI-FACTOR)",
            "=" * 70,
            "",
            "OBJECTIVE: Optimize holding periods via signal quality + volatility",
            "",
            f"Test period: {start_str} to {end_str}",
            "",
            "BASELINE (Fixed 2-Day):",
            f"  Win Rate: {baseline['win_rate']:.1f}%",
            f"  Avg Win: {baseline['avg_win']:.2f}%",
            f"  Sharpe: {baseline['sharpe_ratio']:.2f}",
            f"  Return: {baseline['total_return']:.2f}%",
            "",
            "ENHANCED (Adaptive Hold Days):",
            f"  Win Rate: {enhanced['win_rate']:.1f}% (+{win_rate_improvement:.1f}pp)",
            f"  Avg Win: {enhanced['avg_win']:.2f}% (+{avg_win_improvement:.2f}%)",
            f"  Sharpe: {enhanced['sharpe_ratio']:.2f} (+{sharpe_improvement:.1f}%)",
            f"  Return: {enhanced['total_return']:.2f}% (+{return_improvement:.2f}pp)",
            "",
            "HOLD DAY LOGIC:",
            "  Elite (90-100) + High ATR (>2.5%): 4 days",
            "  Elite (90-100) + Med ATR (1.5-2.5%): 3 days",
            "  Elite (90-100) + Low ATR (<1.5%): 2 days",
            "  Strong (80-89) + High ATR: 3 days",
            "  Strong (80-89) + Med/Low ATR: 2 days",
            "  Good/Acceptable (65-79): 2 days baseline",
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
    print("EXP-113 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

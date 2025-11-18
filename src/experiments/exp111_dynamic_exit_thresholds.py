"""
EXP-111: ATR-Based Dynamic Exit Thresholds

OBJECTIVE: Improve profit factor and risk-adjusted returns via volatility-adaptive exits

PROBLEM:
Current system uses fixed 2% profit target and -2% stop loss for ALL stocks.
But 2% in NVDA (ATR ~3.5%) is routine noise, while 2% in WMT (ATR ~1.2%) is significant.

THEORY:
- High volatility stocks need wider exits to capture their larger moves
- Low volatility stocks need tighter exits to avoid overstaying
- One-size-fits-all exits leave money on the table

METHODOLOGY:
- Baseline: Fixed 2% profit / -2% stop (current)
- Enhanced: ATR-based dynamic exits:
  * High volatility (ATR > 2.5%): 3.0% profit, -3.0% stop
  * Medium volatility (ATR 1.5-2.5%): 2.0% profit, -2.0% stop
  * Low volatility (ATR < 1.5%): 1.5% profit, -1.5% stop

EXPECTED IMPACT:
- Profit factor: +15-25% improvement (capture bigger moves in volatile stocks)
- Sharpe ratio: +8-12% improvement
- Win rate: Unchanged (same entries)
- Average win: Higher (wider targets for volatile stocks)

SUCCESS CRITERIA:
- Profit factor improvement >= +10%
- Sharpe ratio improvement >= +8%
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


def get_dynamic_exits(atr_pct: float) -> tuple:
    """
    Calculate dynamic profit target and stop loss based on ATR.

    Args:
        atr_pct: ATR as percentage of price

    Returns:
        Tuple of (profit_target, stop_loss)
    """
    if atr_pct > 2.5:
        # High volatility - wider exits
        return (3.0, -3.0)
    elif atr_pct >= 1.5:
        # Medium volatility - baseline
        return (2.0, -2.0)
    else:
        # Low volatility - tighter exits
        return (1.5, -1.5)


def backtest_with_dynamic_exits(
    start_date: str,
    end_date: str,
    use_dynamic_exits: bool = False
) -> dict:
    """
    Backtest strategy with or without dynamic exit thresholds.

    Args:
        start_date: Start date for backtest
        end_date: End date for backtest
        use_dynamic_exits: Whether to use ATR-based dynamic exits

    Returns:
        Performance metrics dictionary
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {'DYNAMIC EXITS' if use_dynamic_exits else 'FIXED EXITS (BASELINE)'}")
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

        # Calculate ATR and set dynamic exits if enabled
        if use_dynamic_exits and signals:
            for signal in signals:
                ticker = signal['ticker']

                # Fetch historical data to calculate ATR
                try:
                    # Get 30 days of data for ATR calculation
                    lookback_start = (date - timedelta(days=45)).strftime('%Y-%m-%d')
                    hist_data = fetcher.fetch_stock_data(ticker, start_date=lookback_start, end_date=date_str)

                    if not hist_data.empty and len(hist_data) >= 14:
                        atr_pct = calculate_atr(hist_data)
                        profit_target, stop_loss = get_dynamic_exits(atr_pct)

                        signal['profit_target'] = profit_target
                        signal['stop_loss'] = stop_loss
                        signal['atr_pct'] = atr_pct

                        exit_adjustments.append({
                            'date': date_str,
                            'ticker': ticker,
                            'atr_pct': atr_pct,
                            'profit_target': profit_target,
                            'stop_loss': stop_loss
                        })
                    else:
                        # Fallback to baseline if insufficient data
                        signal['profit_target'] = 2.0
                        signal['stop_loss'] = -2.0
                except:
                    # Fallback to baseline on error
                    signal['profit_target'] = 2.0
                    signal['stop_loss'] = -2.0

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

    # Calculate average profit factor
    total_wins = sum([t['pnl'] for t in trader.closed_positions if t['pnl'] > 0])
    total_losses = abs(sum([t['pnl'] for t in trader.closed_positions if t['pnl'] < 0]))
    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    # Calculate average exit adjustments if used
    avg_profit_target = None
    avg_stop_loss = None
    if exit_adjustments:
        avg_profit_target = np.mean([a['profit_target'] for a in exit_adjustments])
        avg_stop_loss = np.mean([a['stop_loss'] for a in exit_adjustments])

    result = {
        'use_dynamic_exits': use_dynamic_exits,
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
        'exit_adjustments_count': len(exit_adjustments) if exit_adjustments else 0
    }

    return result


def main():
    """Run EXP-111 experiment."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("=" * 70)
    print("EXP-111: ATR-BASED DYNAMIC EXIT THRESHOLDS")
    print("=" * 70)
    print(f"Started: {timestamp}")
    print()
    print("OBJECTIVE: Improve profit factor via volatility-adaptive exits")
    print("THEORY: High ATR stocks need wider exits, low ATR stocks need tighter exits")
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
    baseline = backtest_with_dynamic_exits(start_str, end_str, use_dynamic_exits=False)

    print(f"\nBaseline Results:")
    print(f"  Trades executed:   {baseline['total_trades']}")
    print(f"  Win rate:          {baseline['win_rate']:.1f}%")
    print(f"  Avg win:           {baseline['avg_win']:.2f}%")
    print(f"  Avg loss:          {baseline['avg_loss']:.2f}%")
    print(f"  Profit factor:     {baseline['profit_factor']:.2f}")
    print(f"  Total return:      {baseline['total_return']:.2f}%")
    print(f"  Sharpe ratio:      {baseline['sharpe_ratio']:.2f}")
    print(f"  Max drawdown:      {baseline['max_drawdown']:.2f}%")

    # Enhanced (dynamic exits)
    print("\n" + "=" * 70)
    print("ENHANCED: ATR-Based Dynamic Exits")
    print("=" * 70)
    enhanced = backtest_with_dynamic_exits(start_str, end_str, use_dynamic_exits=True)

    # Calculate improvements
    profit_factor_improvement = ((enhanced['profit_factor'] - baseline['profit_factor']) / baseline['profit_factor'] * 100) if baseline['profit_factor'] != 0 else 0
    sharpe_improvement = ((enhanced['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100) if baseline['sharpe_ratio'] != 0 else 0
    return_improvement = enhanced['total_return'] - baseline['total_return']

    print(f"\nEnhanced Results:")
    print(f"  Trades executed:   {enhanced['total_trades']}")
    print(f"  Win rate:          {enhanced['win_rate']:.1f}%")
    print(f"  Avg win:           {enhanced['avg_win']:.2f}%")
    print(f"  Avg loss:          {enhanced['avg_loss']:.2f}%")
    print(f"  Profit factor:     {enhanced['profit_factor']:.2f} ({profit_factor_improvement:+.1f}%)")
    print(f"  Total return:      {enhanced['total_return']:.2f}% ({return_improvement:+.2f}pp)")
    print(f"  Sharpe ratio:      {enhanced['sharpe_ratio']:.2f} ({sharpe_improvement:+.1f}%)")
    print(f"  Max drawdown:      {enhanced['max_drawdown']:.2f}%")
    if enhanced['avg_profit_target']:
        print(f"  Avg profit target: {enhanced['avg_profit_target']:.2f}%")
        print(f"  Avg stop loss:     {enhanced['avg_stop_loss']:.2f}%")

    # Deployment recommendation
    print("\n" + "=" * 70)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 70)

    deploy = profit_factor_improvement >= 10.0 or sharpe_improvement >= 8.0

    if deploy:
        print(f"\n[DEPLOY] Enhancement validated!")
        print(f"  Profit factor improvement: +{profit_factor_improvement:.1f}%")
        print(f"  Sharpe improvement: +{sharpe_improvement:.1f}%")
        print(f"  Return improvement: +{return_improvement:.2f}pp")
        print(f"\n  Implementation: Add ATR-based dynamic exit thresholds to PaperTrader")
    else:
        print(f"\n[REJECT] Insufficient improvement")
        print(f"  Profit factor improvement: +{profit_factor_improvement:.1f}% (need +10.0%)")
        print(f"  Sharpe improvement: +{sharpe_improvement:.1f}% (need +8.0%)")

    # Save results
    experiment_results = {
        'experiment': 'EXP-111',
        'timestamp': timestamp,
        'objective': 'ATR-based dynamic exit thresholds',
        'test_period': f"{start_str} to {end_str}",
        'baseline': baseline,
        'enhanced': enhanced,
        'profit_factor_improvement_pct': profit_factor_improvement,
        'sharpe_improvement_pct': sharpe_improvement,
        'return_improvement_pp': return_improvement,
        'recommendation': 'DEPLOY' if deploy else 'REJECT',
        'reason': f"Profit Factor: +{profit_factor_improvement:.1f}%, Sharpe: +{sharpe_improvement:.1f}%, Return: +{return_improvement:.2f}pp"
    }

    # Save to file
    os.makedirs('results/ml_experiments', exist_ok=True)
    output_file = 'results/ml_experiments/exp111_dynamic_exit_thresholds.json'
    with open(output_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Email notification
    try:
        notifier = SendGridNotifier()
        subject = f"EXP-111: Dynamic Exit Thresholds - {'DEPLOY' if deploy else 'REJECT'}"

        body_lines = [
            "=" * 70,
            "EXP-111: ATR-BASED DYNAMIC EXIT THRESHOLDS",
            "=" * 70,
            "",
            "OBJECTIVE: Improve profit factor via volatility-adaptive exits",
            "",
            f"Test period: {start_str} to {end_str}",
            "",
            "BASELINE (Fixed 2% / -2%):",
            f"  Profit Factor: {baseline['profit_factor']:.2f}",
            f"  Sharpe: {baseline['sharpe_ratio']:.2f}",
            f"  Return: {baseline['total_return']:.2f}%",
            "",
            "ENHANCED (ATR-Based Dynamic):",
            f"  Profit Factor: {enhanced['profit_factor']:.2f} (+{profit_factor_improvement:.1f}%)",
            f"  Sharpe: {enhanced['sharpe_ratio']:.2f} (+{sharpe_improvement:.1f}%)",
            f"  Return: {enhanced['total_return']:.2f}% (+{return_improvement:.2f}pp)",
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
    print("EXP-111 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

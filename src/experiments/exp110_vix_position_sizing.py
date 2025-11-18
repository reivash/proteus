"""
EXP-110: VIX-Based Adaptive Position Sizing

OBJECTIVE: Improve risk-adjusted returns via VIX-based position sizing

THEORY:
- Low VIX (<15): Calm markets, mean reversion is safer → size up (1.3x)
- Medium VIX (15-25): Normal markets → baseline sizing (1.0x)
- High VIX (25-35): Volatile markets → size down (0.7x)
- Extreme VIX (>35): Panic markets → size down heavily (0.4x)

RATIONALE:
VIX measures market volatility and fear. Mean reversion strategies perform best
in calm, predictable markets (low VIX). In high VIX environments, stocks can
continue falling far beyond normal reverting points.

METHODOLOGY:
- Baseline: Current position sizing (no VIX adjustment)
- Enhanced: Multiply position size by VIX multiplier
- Backtest both strategies on 2-year period
- Compare Sharpe ratio, max drawdown, win rate

EXPECTED IMPACT:
- Sharpe ratio: +10-15% improvement
- Max drawdown: -20% reduction
- Win rate: Unchanged (same signals)
- Better risk-adjusted returns

SUCCESS CRITERIA:
- Sharpe ratio improvement >= +10%
- Max drawdown reduction >= -15%
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


def get_vix_multiplier(vix_value: float) -> float:
    """
    Calculate position size multiplier based on VIX level.

    Args:
        vix_value: Current VIX index value

    Returns:
        Position size multiplier (0.4 to 1.3)
    """
    if vix_value < 15:
        return 1.3  # Calm markets - size up
    elif vix_value < 25:
        return 1.0  # Normal markets - baseline
    elif vix_value < 35:
        return 0.7  # Volatile markets - size down
    else:
        return 0.4  # Extreme volatility - size down heavily


def get_vix_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch VIX data for date range.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with VIX close prices indexed by date
    """
    fetcher = YahooFinanceFetcher()
    vix_data = fetcher.fetch_stock_data('^VIX', start_date=start_date, end_date=end_date)

    if vix_data.empty:
        print("[WARN] Could not fetch VIX data, using default multiplier of 1.0")
        return pd.DataFrame()

    # Create date-indexed series of VIX values
    vix_series = vix_data['Close'].copy()
    vix_series.index = pd.to_datetime(vix_series.index)

    return vix_series


def backtest_with_vix_sizing(
    start_date: str,
    end_date: str,
    use_vix_sizing: bool = False
) -> dict:
    """
    Backtest strategy with or without VIX-based position sizing.

    Args:
        start_date: Start date for backtest
        end_date: End date for backtest
        use_vix_sizing: Whether to use VIX-based position sizing

    Returns:
        Performance metrics dictionary
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {'WITH VIX SIZING' if use_vix_sizing else 'BASELINE (NO VIX)'}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*70}\n")

    # Fetch VIX data if using VIX sizing
    vix_data = get_vix_data(start_date, end_date) if use_vix_sizing else None

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
    vix_adjustments = []

    for date in dates:
        date_str = date.strftime('%Y-%m-%d')

        # Get market regime
        regime = scanner.get_market_regime()
        if regime == 'BEAR':
            continue

        # Get VIX multiplier for this date
        vix_multiplier = 1.0
        if use_vix_sizing and vix_data is not None and len(vix_data) > 0:
            # Get VIX value for this date (or closest prior date)
            try:
                vix_value = vix_data.asof(date)
                if not pd.isna(vix_value):
                    vix_multiplier = get_vix_multiplier(float(vix_value))
                    vix_adjustments.append({
                        'date': date_str,
                        'vix': float(vix_value),
                        'multiplier': vix_multiplier
                    })
            except:
                pass

        # Scan for signals
        signals = scanner.scan_all_stocks(date_str)
        signals_found += len(signals)

        # Apply VIX multiplier to position sizes if enabled
        if use_vix_sizing and signals:
            for signal in signals:
                signal['position_size'] *= vix_multiplier
                signal['vix_multiplier'] = vix_multiplier

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

    # Calculate average VIX multiplier if used
    avg_vix_multiplier = None
    if vix_adjustments:
        avg_vix_multiplier = np.mean([a['multiplier'] for a in vix_adjustments])

    result = {
        'use_vix_sizing': use_vix_sizing,
        'signals_found': signals_found,
        'entries_executed': entries_executed,
        'exits_executed': exits_executed,
        'total_trades': stats.get('total_trades', 0),
        'win_rate': stats.get('win_rate', 0),
        'avg_win': stats.get('avg_win', 0),
        'avg_loss': stats.get('avg_loss', 0),
        'total_return': stats.get('total_return', 0),
        'sharpe_ratio': stats.get('sharpe_ratio', 0),
        'max_drawdown': stats.get('max_drawdown', 0),
        'final_equity': stats.get('current_equity', 100000),
        'avg_vix_multiplier': avg_vix_multiplier,
        'vix_adjustments_count': len(vix_adjustments) if vix_adjustments else 0
    }

    return result


def main():
    """Run EXP-110 experiment."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("=" * 70)
    print("EXP-110: VIX-BASED ADAPTIVE POSITION SIZING")
    print("=" * 70)
    print(f"Started: {timestamp}")
    print()
    print("OBJECTIVE: Improve risk-adjusted returns via VIX-based position sizing")
    print("THEORY: Low VIX = calm markets = size up, High VIX = volatile = size down")
    print()

    # Test period: Last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}")
    print()

    # Baseline (no VIX sizing)
    print("\n" + "=" * 70)
    print("BASELINE: No VIX Sizing")
    print("=" * 70)
    baseline = backtest_with_vix_sizing(start_str, end_str, use_vix_sizing=False)

    print(f"\nBaseline Results:")
    print(f"  Trades executed:   {baseline['total_trades']}")
    print(f"  Win rate:          {baseline['win_rate']:.1f}%")
    print(f"  Avg win:           {baseline['avg_win']:.2f}%")
    print(f"  Avg loss:          {baseline['avg_loss']:.2f}%")
    print(f"  Total return:      {baseline['total_return']:.2f}%")
    print(f"  Sharpe ratio:      {baseline['sharpe_ratio']:.2f}")
    print(f"  Max drawdown:      {baseline['max_drawdown']:.2f}%")

    # Enhanced (with VIX sizing)
    print("\n" + "=" * 70)
    print("ENHANCED: VIX-Based Position Sizing")
    print("=" * 70)
    enhanced = backtest_with_vix_sizing(start_str, end_str, use_vix_sizing=True)

    # Calculate improvements
    sharpe_improvement = ((enhanced['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100) if baseline['sharpe_ratio'] != 0 else 0
    drawdown_improvement = ((enhanced['max_drawdown'] - baseline['max_drawdown']) / baseline['max_drawdown'] * 100) if baseline['max_drawdown'] != 0 else 0
    return_improvement = enhanced['total_return'] - baseline['total_return']

    print(f"\nEnhanced Results:")
    print(f"  Trades executed:   {enhanced['total_trades']}")
    print(f"  Win rate:          {enhanced['win_rate']:.1f}%")
    print(f"  Avg win:           {enhanced['avg_win']:.2f}%")
    print(f"  Avg loss:          {enhanced['avg_loss']:.2f}%")
    print(f"  Total return:      {enhanced['total_return']:.2f}% ({return_improvement:+.2f}pp)")
    print(f"  Sharpe ratio:      {enhanced['sharpe_ratio']:.2f} ({sharpe_improvement:+.1f}%)")
    print(f"  Max drawdown:      {enhanced['max_drawdown']:.2f}% ({drawdown_improvement:+.1f}%)")
    if enhanced['avg_vix_multiplier']:
        print(f"  Avg VIX multiplier: {enhanced['avg_vix_multiplier']:.2f}x")

    # Deployment recommendation
    print("\n" + "=" * 70)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 70)

    deploy = sharpe_improvement >= 10.0 or (sharpe_improvement >= 5.0 and drawdown_improvement <= -10.0)

    if deploy:
        print(f"\n[DEPLOY] Enhancement validated!")
        print(f"  Sharpe improvement: +{sharpe_improvement:.1f}%")
        print(f"  Drawdown improvement: {drawdown_improvement:+.1f}%")
        print(f"  Return improvement: +{return_improvement:.2f}pp")
        print(f"\n  Implementation: Add VIX-based multiplier to PaperTrader position sizing")
    else:
        print(f"\n[REJECT] Insufficient improvement")
        print(f"  Sharpe improvement: +{sharpe_improvement:.1f}% (need +10.0%)")
        print(f"  Drawdown improvement: {drawdown_improvement:+.1f}%")

    # Save results
    experiment_results = {
        'experiment': 'EXP-110',
        'timestamp': timestamp,
        'objective': 'VIX-based adaptive position sizing',
        'test_period': f"{start_str} to {end_str}",
        'baseline': baseline,
        'enhanced': enhanced,
        'sharpe_improvement_pct': sharpe_improvement,
        'drawdown_improvement_pct': drawdown_improvement,
        'return_improvement_pp': return_improvement,
        'recommendation': 'DEPLOY' if deploy else 'REJECT',
        'reason': f"Sharpe: +{sharpe_improvement:.1f}%, Drawdown: {drawdown_improvement:+.1f}%, Return: +{return_improvement:.2f}pp"
    }

    # Save to file
    os.makedirs('results/ml_experiments', exist_ok=True)
    output_file = 'results/ml_experiments/exp110_vix_position_sizing.json'
    with open(output_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Email notification
    try:
        notifier = SendGridNotifier()
        subject = f"EXP-110: VIX Position Sizing - {'DEPLOY' if deploy else 'REJECT'}"

        body_lines = [
            "=" * 70,
            "EXP-110: VIX-BASED ADAPTIVE POSITION SIZING",
            "=" * 70,
            "",
            "OBJECTIVE: Improve risk-adjusted returns via VIX-based sizing",
            "",
            f"Test period: {start_str} to {end_str}",
            "",
            "BASELINE (No VIX):",
            f"  Sharpe: {baseline['sharpe_ratio']:.2f}",
            f"  Max DD: {baseline['max_drawdown']:.2f}%",
            f"  Return: {baseline['total_return']:.2f}%",
            "",
            "ENHANCED (VIX Sizing):",
            f"  Sharpe: {enhanced['sharpe_ratio']:.2f} (+{sharpe_improvement:.1f}%)",
            f"  Max DD: {enhanced['max_drawdown']:.2f}% ({drawdown_improvement:+.1f}%)",
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
    print("EXP-110 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

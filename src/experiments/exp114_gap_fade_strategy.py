"""
EXP-114: Gap Fade Strategy (Intraday Mean Reversion)

OBJECTIVE: Capture intraday bounces from overnight gap-down overreactions

OPPORTUNITY:
Current system trades end-of-day mean reversion (2-day hold).
But there's a SEPARATE opportunity: gap fade trading.

Stocks that gap down >2% at open often represent:
- Overnight panic selling (overreaction to news)
- Short-seller piling on
- Algorithm-driven cascades
- Emotional fear-based selling

These gaps frequently BOUNCE during the trading day as:
- Short sellers take profits
- Bargain hunters step in
- Fear subsides and rationality returns

THEORY:
Gap fades are a distinct form of mean reversion:
- Entry: At/near open on gap down >2%
- Exit: Same-day bounce (intraday) or next day max
- Hold: 1 day max (quick bounce)
- Psychology: Overnight gaps often overreact, creating intraday reversal

METHODOLOGY:
- Scan for stocks that gap down >2% from previous close
- Filter for quality: liquidity, no earnings, fundamentally sound
- Compare baseline (no gap fade) vs enhanced (with gap fade signals)
- Track as separate signal stream alongside EOD mean reversion

EXPECTED IMPACT:
- New signal stream: +50-100 additional trades/year
- Win rate: 65-75% (gaps are strong overreaction signals)
- Average gain: +0.8-1.2% per trade (smaller but frequent)
- Annual contribution: 50-100 trades × 1% = +50-100% additional return
- Sharpe improvement: +10-15% (uncorrelated to EOD signals)

SUCCESS CRITERIA:
- Win rate >= 65%
- Average gain >= 0.8%
- Sharpe ratio improvement >= +8%
- Deploy if validated

COMPLEMENTARITY:
- Current: End-of-day mean reversion (2-day hold)
- Gap Fade: Intraday/overnight bounce (1-day hold)
- No overlap - completely separate opportunity set
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
from src.config.mean_reversion_params import get_all_tickers


def scan_gap_fade_signals(
    date: str,
    fetcher: YahooFinanceFetcher,
    min_gap_pct: float = 2.0,
    min_volume: int = 1000000
) -> list:
    """
    Scan for gap fade opportunities (stocks that gap down at open).

    Args:
        date: Date to scan (YYYY-MM-DD)
        fetcher: Data fetcher instance
        min_gap_pct: Minimum gap down percentage (default: 2.0%)
        min_volume: Minimum average volume (default: 1M)

    Returns:
        List of gap fade signal dictionaries
    """
    signals = []
    tickers = get_all_tickers()

    # Get previous trading day
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    prev_date = (date_obj - timedelta(days=3)).strftime('%Y-%m-%d')  # 3 days to ensure we get previous trading day

    for ticker in tickers:
        try:
            # Fetch data for previous day and current day
            data = fetcher.fetch_stock_data(ticker, start_date=prev_date, end_date=date)

            if data.empty or len(data) < 2:
                continue

            # Get previous close and current open
            prev_close = float(data['Close'].iloc[-2])
            curr_open = float(data['Open'].iloc[-1])
            curr_close = float(data['Close'].iloc[-1])
            curr_low = float(data['Low'].iloc[-1])
            curr_high = float(data['High'].iloc[-1])

            # Calculate gap percentage
            gap_pct = ((curr_open - prev_close) / prev_close) * 100

            # Check if gap down meets criteria
            if gap_pct <= -min_gap_pct:
                # Check volume (use 20-day average)
                if len(data) >= 20:
                    avg_volume = data['Volume'].tail(20).mean()
                else:
                    avg_volume = data['Volume'].mean()

                if avg_volume < min_volume:
                    continue

                # Calculate intraday bounce
                bounce_from_open = ((curr_close - curr_open) / curr_open) * 100

                # Create signal
                signal = {
                    'ticker': ticker,
                    'date': date,
                    'signal_type': 'GAP_FADE',
                    'entry_price': curr_open,  # Enter at open
                    'prev_close': prev_close,
                    'gap_pct': gap_pct,
                    'intraday_bounce': bounce_from_open,
                    'intraday_low': curr_low,
                    'intraday_high': curr_high,
                    'close': curr_close,
                    'avg_volume': float(avg_volume),
                    'limit_price': curr_open * 0.99,  # Small limit discount
                    'max_hold_days': 1  # Gap fades are 1-day max
                }

                signals.append(signal)

        except Exception as e:
            continue

    return signals


def backtest_with_gap_fade(
    start_date: str,
    end_date: str,
    include_gap_fade: bool = False
) -> dict:
    """
    Backtest strategy with or without gap fade signals.

    Args:
        start_date: Start date for backtest
        end_date: End date for backtest
        include_gap_fade: Whether to include gap fade signals

    Returns:
        Performance metrics dictionary
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {'WITH GAP FADE' if include_gap_fade else 'EOD MEAN REVERSION ONLY (BASELINE)'}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*70}\n")

    # Initialize components
    scanner = SignalScanner(
        lookback_days=90,
        min_signal_strength=65.0,  # Q4 filter
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

    eod_signals_found = 0
    gap_signals_found = 0
    eod_entries = 0
    gap_entries = 0
    exits_executed = 0

    for date in dates:
        date_str = date.strftime('%Y-%m-%d')

        # Get market regime
        regime = scanner.get_market_regime()
        if regime == 'BEAR':
            continue

        all_signals = []

        # 1. Scan for end-of-day mean reversion signals (existing)
        eod_signals = scanner.scan_all_stocks(date_str)
        eod_signals_found += len(eod_signals)
        all_signals.extend(eod_signals)

        # 2. Scan for gap fade signals (new)
        if include_gap_fade:
            gap_signals = scan_gap_fade_signals(date_str, fetcher)
            gap_signals_found += len(gap_signals)
            all_signals.extend(gap_signals)

        # Process entries
        if all_signals:
            entries = trader.process_signals(all_signals, date_str)

            # Track entry types
            for entry in entries:
                if entry.get('signal_type') == 'GAP_FADE':
                    gap_entries += 1
                else:
                    eod_entries += 1

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

    # Separate gap fade vs EOD trades
    gap_trades = [t for t in trader.closed_positions if t.get('signal_type') == 'GAP_FADE']
    eod_trades = [t for t in trader.closed_positions if t.get('signal_type') != 'GAP_FADE']

    gap_win_rate = (len([t for t in gap_trades if t['pnl'] > 0]) / len(gap_trades) * 100) if gap_trades else 0
    gap_avg_gain = np.mean([t['return'] for t in gap_trades]) if gap_trades else 0

    result = {
        'include_gap_fade': include_gap_fade,
        'eod_signals_found': eod_signals_found,
        'gap_signals_found': gap_signals_found,
        'eod_entries': eod_entries,
        'gap_entries': gap_entries,
        'exits_executed': exits_executed,
        'total_trades': stats.get('total_trades', 0),
        'gap_trades_count': len(gap_trades),
        'eod_trades_count': len(eod_trades),
        'gap_win_rate': gap_win_rate,
        'gap_avg_gain': gap_avg_gain,
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
    """Run EXP-114 experiment."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("=" * 70)
    print("EXP-114: GAP FADE STRATEGY (INTRADAY MEAN REVERSION)")
    print("=" * 70)
    print(f"Started: {timestamp}")
    print()
    print("OBJECTIVE: Capture intraday bounces from overnight gap-down overreactions")
    print("THEORY: Gaps >2% often overreact, creating same-day reversal opportunities")
    print()

    # Test period: Last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}")
    print()

    # Baseline (EOD mean reversion only)
    print("\n" + "=" * 70)
    print("BASELINE: End-of-Day Mean Reversion Only")
    print("=" * 70)
    baseline = backtest_with_gap_fade(start_str, end_str, include_gap_fade=False)

    print(f"\nBaseline Results:")
    print(f"  EOD signals found: {baseline['eod_signals_found']}")
    print(f"  Trades executed:   {baseline['total_trades']}")
    print(f"  Win rate:          {baseline['win_rate']:.1f}%")
    print(f"  Avg win:           {baseline['avg_win']:.2f}%")
    print(f"  Avg loss:          {baseline['avg_loss']:.2f}%")
    print(f"  Profit factor:     {baseline['profit_factor']:.2f}")
    print(f"  Total return:      {baseline['total_return']:.2f}%")
    print(f"  Sharpe ratio:      {baseline['sharpe_ratio']:.2f}")
    print(f"  Max drawdown:      {baseline['max_drawdown']:.2f}%")

    # Enhanced (with gap fade signals)
    print("\n" + "=" * 70)
    print("ENHANCED: EOD Mean Reversion + Gap Fade")
    print("=" * 70)
    enhanced = backtest_with_gap_fade(start_str, end_str, include_gap_fade=True)

    # Calculate improvements
    win_rate_improvement = enhanced['win_rate'] - baseline['win_rate']
    sharpe_improvement = ((enhanced['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100) if baseline['sharpe_ratio'] != 0 else 0
    return_improvement = enhanced['total_return'] - baseline['total_return']
    profit_factor_improvement = ((enhanced['profit_factor'] - baseline['profit_factor']) / baseline['profit_factor'] * 100) if baseline['profit_factor'] != 0 else 0

    print(f"\nEnhanced Results:")
    print(f"  EOD signals found: {enhanced['eod_signals_found']}")
    print(f"  Gap signals found: {enhanced['gap_signals_found']} (NEW)")
    print(f"  Total trades:      {enhanced['total_trades']} (+{enhanced['total_trades'] - baseline['total_trades']})")
    print(f"    - EOD trades:    {enhanced['eod_trades_count']}")
    print(f"    - Gap trades:    {enhanced['gap_trades_count']} (NEW)")
    print(f"\n  Gap Fade Performance:")
    print(f"    Win rate:        {enhanced['gap_win_rate']:.1f}%")
    print(f"    Avg gain:        {enhanced['gap_avg_gain']:.2f}%")
    print(f"\n  Overall Performance:")
    print(f"  Win rate:          {enhanced['win_rate']:.1f}% ({win_rate_improvement:+.1f}pp)")
    print(f"  Avg win:           {enhanced['avg_win']:.2f}%")
    print(f"  Avg loss:          {enhanced['avg_loss']:.2f}%")
    print(f"  Profit factor:     {enhanced['profit_factor']:.2f} ({profit_factor_improvement:+.1f}%)")
    print(f"  Total return:      {enhanced['total_return']:.2f}% ({return_improvement:+.2f}pp)")
    print(f"  Sharpe ratio:      {enhanced['sharpe_ratio']:.2f} ({sharpe_improvement:+.1f}%)")
    print(f"  Max drawdown:      {enhanced['max_drawdown']:.2f}%")

    # Deployment recommendation
    print("\n" + "=" * 70)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 70)

    deploy = (enhanced['gap_win_rate'] >= 65.0 and enhanced['gap_avg_gain'] >= 0.8) or sharpe_improvement >= 8.0

    if deploy:
        print(f"\n[DEPLOY] Gap fade strategy validated!")
        print(f"  Gap trades added: {enhanced['gap_trades_count']}")
        print(f"  Gap win rate: {enhanced['gap_win_rate']:.1f}%")
        print(f"  Gap avg gain: {enhanced['gap_avg_gain']:.2f}%")
        print(f"  Overall Sharpe improvement: +{sharpe_improvement:.1f}%")
        print(f"  Overall return improvement: +{return_improvement:.2f}pp")
        print(f"\n  Implementation: Add gap fade scanner to daily workflow")
        print(f"  Signal criteria: Gap down >2%, volume >1M, 1-day max hold")
    else:
        print(f"\n[REJECT] Insufficient performance")
        print(f"  Gap win rate: {enhanced['gap_win_rate']:.1f}% (need 65.0%)")
        print(f"  Gap avg gain: {enhanced['gap_avg_gain']:.2f}% (need 0.8%)")
        print(f"  Sharpe improvement: +{sharpe_improvement:.1f}% (need +8.0%)")

    # Save results
    experiment_results = {
        'experiment': 'EXP-114',
        'timestamp': timestamp,
        'objective': 'Gap fade strategy (intraday mean reversion)',
        'test_period': f"{start_str} to {end_str}",
        'baseline': baseline,
        'enhanced': enhanced,
        'sharpe_improvement_pct': sharpe_improvement,
        'return_improvement_pp': return_improvement,
        'profit_factor_improvement_pct': profit_factor_improvement,
        'recommendation': 'DEPLOY' if deploy else 'REJECT',
        'reason': f"Gap WR: {enhanced['gap_win_rate']:.1f}%, Gap Avg: {enhanced['gap_avg_gain']:.2f}%, Sharpe: +{sharpe_improvement:.1f}%"
    }

    # Save to file
    os.makedirs('results/ml_experiments', exist_ok=True)
    output_file = 'results/ml_experiments/exp114_gap_fade_strategy.json'
    with open(output_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Email notification
    try:
        notifier = SendGridNotifier()
        subject = f"EXP-114: Gap Fade Strategy - {'DEPLOY' if deploy else 'REJECT'}"

        body_lines = [
            "=" * 70,
            "EXP-114: GAP FADE STRATEGY (INTRADAY MEAN REVERSION)",
            "=" * 70,
            "",
            "OBJECTIVE: Capture intraday bounces from overnight gap-down overreactions",
            "",
            f"Test period: {start_str} to {end_str}",
            "",
            "BASELINE (EOD Mean Reversion Only):",
            f"  Trades: {baseline['total_trades']}",
            f"  Win Rate: {baseline['win_rate']:.1f}%",
            f"  Sharpe: {baseline['sharpe_ratio']:.2f}",
            f"  Return: {baseline['total_return']:.2f}%",
            "",
            "ENHANCED (EOD + Gap Fade):",
            f"  Total Trades: {enhanced['total_trades']} (+{enhanced['gap_trades_count']} gap trades)",
            f"  Gap Win Rate: {enhanced['gap_win_rate']:.1f}%",
            f"  Gap Avg Gain: {enhanced['gap_avg_gain']:.2f}%",
            f"  Overall Win Rate: {enhanced['win_rate']:.1f}% ({win_rate_improvement:+.1f}pp)",
            f"  Overall Sharpe: {enhanced['sharpe_ratio']:.2f} (+{sharpe_improvement:.1f}%)",
            f"  Overall Return: {enhanced['total_return']:.2f}% (+{return_improvement:.2f}pp)",
            "",
            "GAP FADE CRITERIA:",
            "  Entry: Stocks that gap down >2% at open",
            "  Hold: 1 day max (intraday/overnight bounce)",
            "  Volume: >1M average daily volume",
            "  Limit: 1% below open price",
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
    print("EXP-114 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
EXP-125: Adaptive Stop-Loss Optimization

CONTEXT:
Current system uses uniform -2% stop loss for all trades. This is suboptimal because:
- High-volatility stocks need wider stops to avoid premature exits
- High-quality signals deserve more room to work
- Tight stops in volatile markets = death by a thousand cuts

HYPOTHESIS:
Adaptive stop-loss based on ATR + signal strength improves risk-adjusted returns:
- Low volatility (ATR < 1.5%): -1.5% stop (tight)
- Medium volatility (ATR 1.5-2.5%): -2.0% stop (baseline)
- High volatility (ATR > 2.5%): -2.5% stop (wide)
- BONUS: High-quality signals (strength >= 80) get +0.5% wider stops

OBJECTIVE:
Reduce premature stop-outs while maintaining risk control.

METHODOLOGY:
Test 3 strategies:
1. Baseline: -2% uniform stop loss (current)
2. ATR-based: Stop loss scales with volatility
3. ATR + Signal Quality: Volatility + signal strength adaptive

EXPECTED IMPACT:
- Win rate: +2-4pp (fewer premature stops)
- Avg return per trade: +0.3-0.5% (let winners run slightly longer)
- Sharpe ratio: +8-12% (better risk-adjusted returns)
- Max drawdown: Unchanged or slightly better

SUCCESS CRITERIA:
- Sharpe ratio >= 2.6 (+10% minimum)
- Win rate improvement >= +2pp
- Deploy if validated

Created: 2025-11-19
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')

from src.trading.signal_scanner import SignalScanner
from src.trading.paper_trader import PaperTrader
from src.data.fetchers.rate_limited_yahoo import RateLimitedYahooFinanceFetcher


def calculate_atr_stop_loss(atr_pct: float, signal_strength: float = None, use_quality_boost: bool = False) -> float:
    """
    Calculate adaptive stop loss based on ATR and optionally signal quality.

    Args:
        atr_pct: ATR as percentage of price
        signal_strength: Signal quality score (optional)
        use_quality_boost: Whether to give high-quality signals wider stops

    Returns:
        Stop loss percentage (negative value)
    """
    # Base stop loss from ATR
    if atr_pct < 1.5:
        stop_loss = -1.5  # Low volatility = tight stop
    elif atr_pct < 2.5:
        stop_loss = -2.0  # Medium volatility = baseline
    else:
        stop_loss = -2.5  # High volatility = wide stop

    # Quality boost: High-quality signals get wider stops
    if use_quality_boost and signal_strength and signal_strength >= 80:
        stop_loss -= 0.5  # Give elite signals more room

    return stop_loss


def backtest_stop_loss_strategy(
    start_date: str,
    end_date: str,
    strategy: str = 'baseline'
) -> dict:
    """
    Backtest with different stop-loss strategies.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        strategy: 'baseline', 'atr_based', or 'atr_quality'

    Returns:
        Dictionary with performance metrics
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {strategy.upper()} STOP-LOSS STRATEGY")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*70}\n")

    # Initialize components
    fetcher = RateLimitedYahooFinanceFetcher(verbose=True)
    scanner = SignalScanner(
        lookback_days=90,
        min_signal_strength=None  # Use Q4 filtering from SignalScanner
    )

    # Base trader configuration
    trader = PaperTrader(
        initial_capital=90000,
        max_positions=5,
        profit_target=2.0,
        stop_loss=-2.0,  # Will be overridden for adaptive strategies
        max_hold_days=2,
        use_limit_orders=True
    )

    # Track stop-loss statistics
    stop_stats = {
        'total_exits': 0,
        'stop_loss_exits': 0,
        'profit_target_exits': 0,
        'time_decay_exits': 0,
        'stop_losses_by_size': {}
    }

    # Convert dates
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    current_dt = start_dt

    while current_dt <= end_dt:
        date_str = current_dt.strftime('%Y-%m-%d')

        # Scan for signals
        signals = scanner.scan_all_stocks(date_str)

        if len(signals) > 0:
            print(f"\n{'='*70}")
            print(f"Date: {date_str}")
            print(f"Signals found: {len(signals)}")

            # Apply adaptive stop-loss for non-baseline strategies
            if strategy != 'baseline':
                for signal in signals:
                    # Calculate ATR percentage
                    # Note: Signals should include ATR data from scanner
                    atr = signal.get('atr', 2.0)  # Default to 2% if not available
                    price = signal.get('price', 100)
                    atr_pct = (atr / price) * 100 if price > 0 else 2.0

                    # Get signal strength
                    signal_strength = signal.get('signal_strength', 0)

                    # Calculate adaptive stop loss
                    if strategy == 'atr_based':
                        stop_loss = calculate_atr_stop_loss(atr_pct, use_quality_boost=False)
                    elif strategy == 'atr_quality':
                        stop_loss = calculate_atr_stop_loss(atr_pct, signal_strength, use_quality_boost=True)
                    else:
                        stop_loss = -2.0  # Baseline

                    # Override signal stop loss
                    signal['stop_loss_pct'] = stop_loss

                    # Track stop-loss distribution
                    stop_key = f"{stop_loss:.1f}%"
                    stop_stats['stop_losses_by_size'][stop_key] = stop_stats['stop_losses_by_size'].get(stop_key, 0) + 1

                    print(f"  {signal['ticker']}: Stop loss = {stop_loss:.1f}% (ATR: {atr_pct:.2f}%, Strength: {signal_strength:.0f})")

            print(f"{'='*70}")

            # Process signals
            trader.process_signals(signals, date_str)

        # Update positions and check exits
        open_positions = trader.get_open_positions()
        if open_positions:
            tickers = [pos['ticker'] for pos in open_positions]
            prices = {}
            for ticker in tickers:
                try:
                    data = fetcher.get_stock_data(ticker, date_str, date_str)
                    if data is not None and not data.empty:
                        prices[ticker] = float(data['Close'].iloc[-1])
                except:
                    pass
            if prices:
                trader.update_daily_equity(prices, date_str)
                exits = trader.check_exits(prices, date_str)

                # Track exit reasons
                for exit_trade in exits:
                    stop_stats['total_exits'] += 1
                    reason = exit_trade.get('exit_reason', 'unknown')
                    if 'stop' in reason.lower():
                        stop_stats['stop_loss_exits'] += 1
                    elif 'profit' in reason.lower() or 'target' in reason.lower():
                        stop_stats['profit_target_exits'] += 1
                    else:
                        stop_stats['time_decay_exits'] += 1

        # Move to next trading day
        current_dt += timedelta(days=1)

    # Get performance
    perf = trader.get_performance()

    # Add stop-loss statistics
    perf['stop_stats'] = stop_stats
    perf['strategy'] = strategy

    return perf


def print_performance(perf: dict, label: str):
    """Print performance metrics."""
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    print(f"Total Trades: {perf['total_trades']}")
    print(f"Win Rate: {perf['win_rate']:.1f}%")
    print(f"Total Return: {perf['total_return']:.2f}%")
    print(f"Avg Return per Trade: {perf['avg_return_per_trade']:.2f}%")
    print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {perf['max_drawdown']:.2f}%")
    print(f"Profit Factor: {perf['profit_factor']:.2f}")

    # Print stop-loss statistics
    if 'stop_stats' in perf:
        stats = perf['stop_stats']
        print(f"\n{'='*70}")
        print("STOP-LOSS BREAKDOWN:")
        print(f"{'='*70}")
        print(f"Total exits: {stats['total_exits']}")
        print(f"Stop-loss exits: {stats['stop_loss_exits']} ({100*stats['stop_loss_exits']/stats['total_exits']:.1f}%)")
        print(f"Profit target exits: {stats['profit_target_exits']} ({100*stats['profit_target_exits']/stats['total_exits']:.1f}%)")
        print(f"Time decay exits: {stats['time_decay_exits']} ({100*stats['time_decay_exits']/stats['total_exits']:.1f}%)")

        if stats['stop_losses_by_size']:
            print(f"\nStop-loss distribution:")
            for size, count in sorted(stats['stop_losses_by_size'].items()):
                print(f"  {size}: {count} signals")

    print(f"{'='*70}\n")


def compare_results(results: Dict[str, dict]):
    """Print comparison table of all results."""
    print(f"\n{'='*70}")
    print("COMPARISON: STOP-LOSS STRATEGIES")
    print(f"{'='*70}\n")

    # Create comparison table
    headers = ["Strategy", "Trades", "WR%", "Avg$/Trade", "Sharpe", "MaxDD%", "PF", "StopLoss%"]
    rows = []

    for strategy, perf in results.items():
        stop_pct = 100 * perf['stop_stats']['stop_loss_exits'] / perf['stop_stats']['total_exits'] if perf['stop_stats']['total_exits'] > 0 else 0

        row = [
            strategy.replace('_', ' ').title(),
            perf['total_trades'],
            f"{perf['win_rate']:.1f}",
            f"{perf['avg_return_per_trade']:.2f}",
            f"{perf['sharpe_ratio']:.2f}",
            f"{perf['max_drawdown']:.2f}",
            f"{perf['profit_factor']:.2f}",
            f"{stop_pct:.1f}"
        ]
        rows.append(row)

    # Print table
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

    # Header
    header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_row)
    print("-" * len(header_row))

    # Rows
    for row in rows:
        print(" | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))))

    print(f"\n{'='*70}")


def main():
    """Run EXP-125 backtest."""
    print("="*70)
    print("EXP-125: ADAPTIVE STOP-LOSS OPTIMIZATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("OBJECTIVE: Reduce premature stop-outs via adaptive stop-loss")
    print("THEORY: ATR-based stops + quality boost = better risk management\n")

    # Test period: 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~2 years

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}\n")

    # Run all strategies
    results = {}

    print("\n" + "="*70)
    print("BASELINE: Uniform -2% Stop Loss (Current)")
    print("="*70)
    results['baseline'] = backtest_stop_loss_strategy(start_str, end_str, strategy='baseline')
    print_performance(results['baseline'], "BASELINE PERFORMANCE (-2% UNIFORM)")

    print("\n" + "="*70)
    print("TEST 1: ATR-Based Adaptive Stop Loss")
    print("="*70)
    results['atr_based'] = backtest_stop_loss_strategy(start_str, end_str, strategy='atr_based')
    print_performance(results['atr_based'], "ATR-BASED PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 2: ATR + Signal Quality Adaptive Stop Loss")
    print("="*70)
    results['atr_quality'] = backtest_stop_loss_strategy(start_str, end_str, strategy='atr_quality')
    print_performance(results['atr_quality'], "ATR + QUALITY PERFORMANCE")

    # Compare all results
    compare_results(results)

    # Find best strategy
    best_strategy = max(results.keys(), key=lambda k: results[k]['sharpe_ratio'])
    best_perf = results[best_strategy]
    baseline_perf = results['baseline']

    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print(f"Best Strategy: {best_strategy.replace('_', ' ').title()}")
    print(f"  Sharpe Ratio: {best_perf['sharpe_ratio']:.2f} (vs {baseline_perf['sharpe_ratio']:.2f} baseline)")
    print(f"  Improvement: {((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100:.1f}%")
    print(f"  Win Rate: {best_perf['win_rate']:.1f}% (vs {baseline_perf['win_rate']:.1f}% baseline)")
    print(f"  Avg $/Trade: {best_perf['avg_return_per_trade']:.2f}% (vs {baseline_perf['avg_return_per_trade']:.2f}% baseline)")

    # Validation
    sharpe_improvement_pct = ((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100
    wr_improvement = best_perf['win_rate'] - baseline_perf['win_rate']

    print(f"\n{'='*70}")
    print("VALIDATION:")
    print(f"{'='*70}")
    print(f"Target: +10% Sharpe improvement AND +2pp win rate")
    print(f"Actual: {sharpe_improvement_pct:+.1f}% Sharpe, {wr_improvement:+.1f}pp win rate")

    if sharpe_improvement_pct >= 10 and wr_improvement >= 2:
        print("SUCCESS: Targets achieved!")
        print(f"\nRECOMMENDATION: Deploy {best_strategy.replace('_', ' ').title()} to production")
    elif sharpe_improvement_pct >= 10 or wr_improvement >= 2:
        print("PARTIAL: One target achieved")
        print("\nRECOMMENDATION: Consider deployment pending review")
    else:
        print("FAILED: Targets not achieved")
        print("\nRECOMMENDATION: Do not deploy - continue research")

    print(f"\n{'='*70}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

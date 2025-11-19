"""
EXP-121: Consecutive Loss Circuit Breaker

CONTEXT:
Current system trades continuously regardless of recent performance. Multiple
consecutive losses may indicate unfavorable market conditions, strategy drift,
or regime change. Continuing to trade during these periods can compound losses.

HYPOTHESIS:
- After N consecutive losses, pause trading for a cooldown period
- This prevents loss spirals and protects capital during adverse conditions
- Similar to how professional traders step back after consecutive losses
- Allows market conditions to normalize before resuming

OBJECTIVE:
Implement circuit breaker to reduce max drawdown and improve risk-adjusted returns.

METHODOLOGY:
Test different circuit breaker thresholds:
1. Baseline: No circuit breaker (current)
2. 3-loss breaker: Pause 3 days after 3 consecutive losses
3. 4-loss breaker: Pause 5 days after 4 consecutive losses
4. 5-loss breaker: Pause 7 days after 5 consecutive losses
5. Adaptive breaker: Pause days = (consecutive_losses - 2) * 2

EXPECTED IMPACT:
- Max drawdown: -20-30% reduction (avoid loss spirals)
- Sharpe ratio: +10-15% (better risk management)
- Win rate: Unchanged or slightly higher (skip unfavorable periods)
- Trade count: -5-15% (selective pause periods)

VALIDATION CRITERIA:
- Max drawdown <= -7% (vs current -10%)
- Sharpe ratio >= 2.6 (+9% minimum)
- Profit factor >= 2.0

Created: 2025-11-19
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.trading.signal_scanner import SignalScanner
from src.trading.paper_trader import PaperTrader
from src.data.fetchers.yahoo_finance import YahooFinanceFetcher


def get_breaker_name(breaker_type: str) -> str:
    """Get human-readable name for circuit breaker."""
    names = {
        'none': 'No Circuit Breaker (Baseline)',
        '3loss_3day': '3-Loss Breaker (3-day pause)',
        '4loss_5day': '4-Loss Breaker (5-day pause)',
        '5loss_7day': '5-Loss Breaker (7-day pause)',
        'adaptive': 'Adaptive Breaker (scaling pause)'
    }
    return names.get(breaker_type, breaker_type)


def get_breaker_config(breaker_type: str) -> tuple:
    """
    Get circuit breaker configuration.

    Returns:
        (consecutive_loss_threshold, pause_days)
        Returns None if adaptive (calculated dynamically)
    """
    if breaker_type == '3loss_3day':
        return (3, 3)
    elif breaker_type == '4loss_5day':
        return (4, 5)
    elif breaker_type == '5loss_7day':
        return (5, 7)
    elif breaker_type == 'adaptive':
        return ('adaptive', None)
    else:  # 'none'
        return (None, None)


class CircuitBreakerTracker:
    """Track consecutive losses and enforce trading pauses."""

    def __init__(self, breaker_type: str):
        self.breaker_type = breaker_type
        self.consecutive_losses = 0
        self.pause_until = None  # Date when pause ends
        self.total_pauses = 0
        self.total_pause_days = 0
        self.trades_during_pause = 0

        # Get config
        threshold, pause_days = get_breaker_config(breaker_type)
        self.loss_threshold = threshold
        self.pause_days_fixed = pause_days

    def record_trade_result(self, is_win: bool, date_str: str):
        """
        Record trade result and update consecutive loss counter.

        Args:
            is_win: True if trade was profitable
            date_str: Trade date (YYYY-MM-DD)
        """
        current_date = datetime.strptime(date_str, '%Y-%m-%d').date()

        if is_win:
            # Reset consecutive loss counter on win
            self.consecutive_losses = 0
        else:
            # Increment consecutive losses
            self.consecutive_losses += 1

            # Check if circuit breaker should trigger
            if self.should_trigger_pause():
                self.trigger_pause(current_date)

    def should_trigger_pause(self) -> bool:
        """Check if circuit breaker should trigger."""
        if self.breaker_type == 'none':
            return False

        if self.breaker_type == 'adaptive':
            # Trigger after 3+ consecutive losses
            return self.consecutive_losses >= 3
        else:
            # Trigger if threshold reached
            return self.consecutive_losses >= self.loss_threshold

    def trigger_pause(self, current_date):
        """Trigger circuit breaker pause."""
        if self.breaker_type == 'adaptive':
            # Adaptive: pause_days = (consecutive_losses - 2) * 2
            # 3 losses = 2 days, 4 losses = 4 days, 5 losses = 6 days, etc.
            pause_days = (self.consecutive_losses - 2) * 2
        else:
            pause_days = self.pause_days_fixed

        self.pause_until = current_date + timedelta(days=pause_days)
        self.total_pauses += 1
        self.total_pause_days += pause_days

        print(f"\n{'='*70}")
        print(f"CIRCUIT BREAKER TRIGGERED!")
        print(f"Consecutive losses: {self.consecutive_losses}")
        print(f"Pausing trading until: {self.pause_until}")
        print(f"Pause duration: {pause_days} days")
        print(f"{'='*70}\n")

    def is_trading_paused(self, date_str: str) -> bool:
        """
        Check if trading is currently paused.

        Args:
            date_str: Current date (YYYY-MM-DD)

        Returns:
            True if paused, False otherwise
        """
        if self.pause_until is None:
            return False

        current_date = datetime.strptime(date_str, '%Y-%m-%d').date()

        if current_date < self.pause_until:
            return True
        else:
            # Pause period ended
            if current_date == self.pause_until:
                print(f"\n{'='*70}")
                print(f"CIRCUIT BREAKER PAUSE ENDED")
                print(f"Resuming trading on {date_str}")
                print(f"{'='*70}\n")

            self.pause_until = None
            self.consecutive_losses = 0  # Reset counter after pause
            return False

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            'total_pauses': self.total_pauses,
            'total_pause_days': self.total_pause_days,
            'trades_skipped_during_pause': self.trades_during_pause
        }


def backtest_with_circuit_breaker(
    start_date: str,
    end_date: str,
    breaker_type: str = 'none'
) -> dict:
    """
    Backtest strategy with circuit breaker.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        breaker_type: 'none', '3loss_3day', '4loss_5day', '5loss_7day', 'adaptive'

    Returns:
        Dictionary with performance metrics
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {get_breaker_name(breaker_type).upper()}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*70}\n")

    # Initialize components
    fetcher = YahooFinanceFetcher()
    scanner = SignalScanner(
        lookback_days=90,
        min_signal_strength=None  # Use Q4 filtering from SignalScanner
    )
    trader = PaperTrader(
        initial_capital=90000,
        max_positions=5,
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=2,
        use_limit_orders=True
    )

    # Initialize circuit breaker
    breaker = CircuitBreakerTracker(breaker_type)

    # Track statistics
    signals_skipped_by_breaker = 0
    total_signals = 0

    # Convert dates
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    current_dt = start_dt

    # Track closed positions to update breaker
    last_closed_count = 0

    while current_dt <= end_dt:
        date_str = current_dt.strftime('%Y-%m-%d')

        # Check if trading is paused
        is_paused = breaker.is_trading_paused(date_str)

        # Scan for signals
        signals = scanner.scan_all_stocks(date_str)

        if len(signals) > 0:
            total_signals += len(signals)

            if is_paused:
                # Skip signals during pause
                signals_skipped_by_breaker += len(signals)
                breaker.trades_during_pause += len(signals)

                print(f"\n{'='*70}")
                print(f"Date: {date_str}")
                print(f"Signals found: {len(signals)}")
                print(f"CIRCUIT BREAKER ACTIVE - Skipping all signals")
                print(f"Pause ends: {breaker.pause_until}")
                print(f"{'='*70}")

                signals = []  # Clear signals
            else:
                print(f"\n{'='*70}")
                print(f"Date: {date_str}")
                print(f"Total signals found: {len(signals)}")
                print(f"Circuit breaker status: ACTIVE (trading allowed)")
                print(f"Consecutive losses: {breaker.consecutive_losses}")
                print(f"{'='*70}")

        # Process signals (empty if paused)
        if signals:
            trader.process_signals(signals, date_str)

        # Update positions
        trader.update_positions(date_str, fetcher)

        # Check for newly closed positions and update breaker
        current_closed = trader.get_performance()['total_trades']
        if current_closed > last_closed_count:
            # New trades closed - update breaker with results
            new_closes = current_closed - last_closed_count

            # Get recent closed positions
            closed_positions = [p for p in trader.closed_positions[-new_closes:]]

            for pos in closed_positions:
                is_win = pos.pnl_pct > 0
                breaker.record_trade_result(is_win, date_str)

            last_closed_count = current_closed

        # Move to next trading day
        current_dt += timedelta(days=1)

    # Get performance
    perf = trader.get_performance()

    # Add circuit breaker statistics
    breaker_stats = breaker.get_stats()
    breaker_stats['signals_skipped'] = signals_skipped_by_breaker
    breaker_stats['total_signals'] = total_signals

    perf['breaker_stats'] = breaker_stats
    perf['breaker_type'] = breaker_type

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

    # Print circuit breaker statistics
    if 'breaker_stats' in perf:
        stats = perf['breaker_stats']
        print(f"\n{'='*70}")
        print("CIRCUIT BREAKER STATISTICS:")
        print(f"{'='*70}")
        print(f"Total pauses triggered: {stats['total_pauses']}")
        print(f"Total days paused: {stats['total_pause_days']}")
        print(f"Signals skipped during pause: {stats['signals_skipped']}")
        if stats['total_signals'] > 0:
            skip_pct = 100 * stats['signals_skipped'] / stats['total_signals']
            print(f"Signal skip rate: {skip_pct:.1f}%")

    print(f"{'='*70}\n")


def compare_results(results: Dict[str, dict]):
    """Print comparison table of all results."""
    print(f"\n{'='*70}")
    print("COMPARISON: CIRCUIT BREAKER STRATEGIES")
    print(f"{'='*70}\n")

    # Create comparison table
    headers = ["Strategy", "Trades", "WR%", "Return%", "Sharpe", "MaxDD%", "PF", "Pauses"]
    rows = []

    for breaker_type, perf in results.items():
        pauses = perf['breaker_stats']['total_pauses'] if 'breaker_stats' in perf else 0

        row = [
            get_breaker_name(breaker_type)[:30],  # Truncate long names
            perf['total_trades'],
            f"{perf['win_rate']:.1f}",
            f"{perf['total_return']:.2f}",
            f"{perf['sharpe_ratio']:.2f}",
            f"{perf['max_drawdown']:.2f}",
            f"{perf['profit_factor']:.2f}",
            pauses
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
    """Run EXP-121 backtest."""
    print("="*70)
    print("EXP-121: CONSECUTIVE LOSS CIRCUIT BREAKER")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("OBJECTIVE: Reduce max drawdown via consecutive loss circuit breaker")
    print("THEORY: Pause trading after N losses to avoid loss spirals\n")

    # Test period: 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~2 years

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}\n")

    # Run all strategies
    results = {}

    print("\n" + "="*70)
    print("BASELINE: No Circuit Breaker")
    print("="*70)
    results['none'] = backtest_with_circuit_breaker(start_str, end_str, breaker_type='none')
    print_performance(results['none'], "BASELINE PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 1: 3-Loss Breaker (3-day pause)")
    print("="*70)
    results['3loss_3day'] = backtest_with_circuit_breaker(start_str, end_str, breaker_type='3loss_3day')
    print_performance(results['3loss_3day'], "3-LOSS BREAKER PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 2: 4-Loss Breaker (5-day pause)")
    print("="*70)
    results['4loss_5day'] = backtest_with_circuit_breaker(start_str, end_str, breaker_type='4loss_5day')
    print_performance(results['4loss_5day'], "4-LOSS BREAKER PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 3: 5-Loss Breaker (7-day pause)")
    print("="*70)
    results['5loss_7day'] = backtest_with_circuit_breaker(start_str, end_str, breaker_type='5loss_7day')
    print_performance(results['5loss_7day'], "5-LOSS BREAKER PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 4: Adaptive Breaker (scaling pause)")
    print("="*70)
    results['adaptive'] = backtest_with_circuit_breaker(start_str, end_str, breaker_type='adaptive')
    print_performance(results['adaptive'], "ADAPTIVE BREAKER PERFORMANCE")

    # Compare all results
    compare_results(results)

    # Find best strategy (by Sharpe ratio)
    best_breaker = max(results.keys(), key=lambda k: results[k]['sharpe_ratio'])
    best_perf = results[best_breaker]
    baseline_perf = results['none']

    # Also find best by max drawdown reduction
    best_dd_breaker = min(results.keys(), key=lambda k: results[k]['max_drawdown'])
    best_dd_perf = results[best_dd_breaker]

    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print(f"Best Strategy (Sharpe): {get_breaker_name(best_breaker)}")
    print(f"  Sharpe Ratio: {best_perf['sharpe_ratio']:.2f} (vs {baseline_perf['sharpe_ratio']:.2f} baseline)")
    print(f"  Improvement: {((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100:.1f}%")
    print(f"  Max Drawdown: {best_perf['max_drawdown']:.2f}% (vs {baseline_perf['max_drawdown']:.2f}% baseline)")
    print(f"  Total Trades: {best_perf['total_trades']} (vs {baseline_perf['total_trades']} baseline)")

    print(f"\nBest Strategy (Max DD): {get_breaker_name(best_dd_breaker)}")
    print(f"  Max Drawdown: {best_dd_perf['max_drawdown']:.2f}% (vs {baseline_perf['max_drawdown']:.2f}% baseline)")
    print(f"  Reduction: {baseline_perf['max_drawdown'] - best_dd_perf['max_drawdown']:.2f}pp")

    # Validation
    sharpe_improvement_pct = ((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100
    dd_reduction = baseline_perf['max_drawdown'] - best_dd_perf['max_drawdown']

    print(f"\n{'='*70}")
    print("VALIDATION:")
    print(f"{'='*70}")
    print(f"Target: +10% Sharpe improvement, -2pp max drawdown")
    print(f"Actual: {sharpe_improvement_pct:+.1f}% Sharpe, {dd_reduction:+.2f}pp max drawdown")

    if sharpe_improvement_pct >= 10 or dd_reduction >= 2:
        print("SUCCESS: Target achieved!")
        print(f"\nRECOMMENDATION: Deploy {get_breaker_name(best_breaker)} to production")
    else:
        print("PARTIAL/FAILED: Review results")
        print("\nRECOMMENDATION: Analyze best-performing breaker for potential deployment")

    print(f"\n{'='*70}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

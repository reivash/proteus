"""
EXP-127: Adaptive Max Hold Days Optimization

CONTEXT:
Current system uses uniform 2-day max hold period for ALL trades. This is
capital-inefficient because:
- Elite signals (90+ strength) deserve 3-4 days to maximize returns
- Weak signals (65-70 strength) should exit at 1-2 days to recycle capital
- Volatility affects optimal hold times (high ATR needs more time)

HYPOTHESIS:
Adaptive max hold days based on signal quality + volatility improves both
returns AND capital efficiency:
- ELITE (90-100): 4 days (let winners run)
- STRONG (80-89): 3 days (above average runway)
- GOOD (70-79): 2 days (baseline)
- ACCEPTABLE (65-69): 1 day (quick exit, recycle capital)
- VOLATILITY BOOST: +1 day if ATR > 2.5% (high-vol stocks need time)

OBJECTIVE:
Maximize risk-adjusted returns via signal-quality-based hold periods.

METHODOLOGY:
Test 4 strategies:
1. Baseline: 2-day uniform (current)
2. Signal-based: Hold days scale with signal strength
3. Volatility-based: Hold days scale with ATR
4. Combined: Signal strength + volatility adaptive

EXPECTED IMPACT:
- Total return: +20-30% (elite signals capture more upside)
- Capital efficiency: +15-25% (faster recycling)
- Win rate: +2-3pp (fewer forced exits on strong signals)
- Sharpe ratio: +8-15% (better risk-adjusted returns)

SUCCESS CRITERIA:
- Total return >= +20% vs baseline
- Sharpe ratio >= 2.6 (+10% minimum)
- Deploy if validated

COMPLEMENTARY TO:
- EXP-125: Adaptive stop-loss (prevents bad exits)
- EXP-126: Adaptive profit targets (maximizes good exits)
- EXP-127: Adaptive hold days (optimizes timing)
= COMPLETE ADAPTIVE RISK MANAGEMENT SYSTEM

Created: 2025-11-19
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.trading.signal_scanner import SignalScanner
from src.trading.paper_trader import PaperTrader
from src.data.fetchers.rate_limited_yahoo import RateLimitedYahooFinanceFetcher


def calculate_adaptive_max_hold_days(
    signal_strength: float,
    atr_pct: float = None,
    strategy: str = 'baseline'
) -> int:
    """
    Calculate adaptive max hold days based on signal quality and volatility.

    Args:
        signal_strength: Signal quality score (65-100)
        atr_pct: ATR as percentage of price (optional)
        strategy: 'baseline', 'signal_based', 'volatility_based', or 'combined'

    Returns:
        Max hold days (integer)
    """
    if strategy == 'baseline':
        return 2  # Current uniform

    # Signal-based component
    if signal_strength >= 90:
        base_days = 4  # Elite: maximum runway
    elif signal_strength >= 80:
        base_days = 3  # Strong: above average
    elif signal_strength >= 70:
        base_days = 2  # Good: baseline
    else:
        base_days = 1  # Acceptable: quick exit

    if strategy == 'signal_based':
        return base_days

    # Volatility-based component
    if strategy == 'volatility_based':
        if atr_pct and atr_pct > 2.5:
            return 3  # High volatility needs time
        elif atr_pct and atr_pct < 1.5:
            return 1  # Low volatility = quick moves
        else:
            return 2  # Baseline

    # Combined strategy
    if strategy == 'combined':
        # Add volatility boost
        vol_boost = 0
        if atr_pct and atr_pct > 2.5:
            vol_boost = 1  # High-vol stocks need extra day

        # Cap at 5 days maximum
        return min(base_days + vol_boost, 5)

    return 2  # Default baseline


def backtest_hold_days_strategy(
    start_date: str,
    end_date: str,
    strategy: str = 'baseline'
) -> dict:
    """
    Backtest with different max hold days strategies.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        strategy: 'baseline', 'signal_based', 'volatility_based', or 'combined'

    Returns:
        Dictionary with performance metrics
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {strategy.upper()} MAX HOLD DAYS STRATEGY")
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
        stop_loss=-2.0,
        max_hold_days=2,  # Will be overridden for adaptive strategies
        use_limit_orders=True
    )

    # Track hold days statistics
    hold_stats = {
        'total_trades': 0,
        'hold_days_by_signal': {},  # Track distribution
        'avg_hold_days': [],
        'exits_by_reason': {
            'profit_target': 0,
            'stop_loss': 0,
            'time_decay': 0
        }
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

            # Apply adaptive max hold days for non-baseline strategies
            if strategy != 'baseline':
                for signal in signals:
                    # Get ATR data
                    atr = signal.get('atr', 2.0)
                    price = signal.get('price', 100)
                    atr_pct = (atr / price) * 100 if price > 0 else 2.0

                    # Get signal strength
                    signal_strength = signal.get('signal_strength', 70)

                    # Calculate adaptive max hold days
                    max_days = calculate_adaptive_max_hold_days(
                        signal_strength,
                        atr_pct,
                        strategy
                    )

                    # Override signal max hold days
                    signal['max_hold_days'] = max_days

                    # Track distribution
                    key = f"{max_days}d"
                    hold_stats['hold_days_by_signal'][key] = hold_stats['hold_days_by_signal'].get(key, 0) + 1

                    print(f"  {signal['ticker']}: Max hold = {max_days} days (Strength: {signal_strength:.0f}, ATR: {atr_pct:.2f}%)")

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

                # Track exit statistics
                for exit_trade in exits:
                    hold_stats['total_trades'] += 1

                    # Track hold days
                    entry_date = datetime.strptime(exit_trade.get('entry_date', date_str), '%Y-%m-%d')
                    exit_date = datetime.strptime(date_str, '%Y-%m-%d')
                    days_held = (exit_date - entry_date).days
                    hold_stats['avg_hold_days'].append(days_held)

                    # Track exit reason
                    reason = exit_trade.get('exit_reason', 'unknown')
                    if 'profit' in reason.lower() or 'target' in reason.lower():
                        hold_stats['exits_by_reason']['profit_target'] += 1
                    elif 'stop' in reason.lower():
                        hold_stats['exits_by_reason']['stop_loss'] += 1
                    else:
                        hold_stats['exits_by_reason']['time_decay'] += 1

        # Move to next trading day
        current_dt += timedelta(days=1)

    # Get performance
    perf = trader.get_performance()

    # Add hold days statistics
    if hold_stats['avg_hold_days']:
        hold_stats['avg_hold_days'] = np.mean(hold_stats['avg_hold_days'])
    else:
        hold_stats['avg_hold_days'] = 0.0

    perf['hold_stats'] = hold_stats
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
    print(f"Avg PnL per Trade: ${perf.get('avg_pnl', 0):.2f}")
    print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {perf.get('max_drawdown', 0):.2f}%")
    print(f"Profit Factor: {perf.get('profit_factor', 0):.2f}")

    # Print hold days statistics
    if 'hold_stats' in perf:
        stats = perf['hold_stats']
        print(f"\n{'='*70}")
        print("HOLD DAYS BREAKDOWN:")
        print(f"{'='*70}")
        print(f"Average hold days: {stats['avg_hold_days']:.2f}")

        if stats['total_trades'] > 0:
            exits = stats['exits_by_reason']
            total = stats['total_trades']
            print(f"\nExit reasons:")
            print(f"  Profit target: {exits['profit_target']} ({100*exits['profit_target']/total:.1f}%)")
            print(f"  Stop loss: {exits['stop_loss']} ({100*exits['stop_loss']/total:.1f}%)")
            print(f"  Time decay: {exits['time_decay']} ({100*exits['time_decay']/total:.1f}%)")

        if stats['hold_days_by_signal']:
            print(f"\nMax hold days distribution:")
            for days, count in sorted(stats['hold_days_by_signal'].items()):
                print(f"  {days}: {count} signals")

    print(f"{'='*70}\n")


def compare_results(results: Dict[str, dict]):
    """Print comparison table of all results."""
    print(f"\n{'='*70}")
    print("COMPARISON: MAX HOLD DAYS STRATEGIES")
    print(f"{'='*70}\n")

    # Create comparison table
    headers = ["Strategy", "Trades", "WR%", "Return%", "$/Trade", "Sharpe", "MaxDD%", "PF", "AvgDays"]
    rows = []

    for strategy, perf in results.items():
        avg_days = perf['hold_stats']['avg_hold_days']

        row = [
            strategy.replace('_', ' ').title(),
            perf['total_trades'],
            f"{perf['win_rate']:.1f}",
            f"{perf['total_return']:.2f}",
            f"{perf.get('avg_pnl', 0):.2f}",
            f"{perf.get('sharpe_ratio', 0):.2f}",
            f"{perf.get('max_drawdown', 0):.2f}",
            f"{perf.get('profit_factor', 0):.2f}",
            f"{avg_days:.2f}"
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
    """Run EXP-127 backtest."""
    print("="*70)
    print("EXP-127: ADAPTIVE MAX HOLD DAYS OPTIMIZATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("OBJECTIVE: Optimize capital efficiency + returns via adaptive hold periods")
    print("THEORY: Signal quality + volatility determine optimal hold time\n")

    # Test period: 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~2 years

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}\n")

    # Run all strategies
    results = {}

    print("\n" + "="*70)
    print("BASELINE: Uniform 2-Day Max Hold (Current)")
    print("="*70)
    results['baseline'] = backtest_hold_days_strategy(start_str, end_str, strategy='baseline')
    print_performance(results['baseline'], "BASELINE PERFORMANCE (2-DAY UNIFORM)")

    print("\n" + "="*70)
    print("TEST 1: Signal-Based Adaptive Hold Days")
    print("="*70)
    results['signal_based'] = backtest_hold_days_strategy(start_str, end_str, strategy='signal_based')
    print_performance(results['signal_based'], "SIGNAL-BASED PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 2: Volatility-Based Adaptive Hold Days")
    print("="*70)
    results['volatility_based'] = backtest_hold_days_strategy(start_str, end_str, strategy='volatility_based')
    print_performance(results['volatility_based'], "VOLATILITY-BASED PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 3: Combined (Signal + Volatility) Adaptive Hold Days")
    print("="*70)
    results['combined'] = backtest_hold_days_strategy(start_str, end_str, strategy='combined')
    print_performance(results['combined'], "COMBINED PERFORMANCE")

    # Compare all results
    compare_results(results)

    # Find best strategy
    best_strategy = max(results.keys(), key=lambda k: results[k]['total_return'])
    best_perf = results[best_strategy]
    baseline_perf = results['baseline']

    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print(f"Best Strategy: {best_strategy.replace('_', ' ').title()}")
    print(f"  Total Return: {best_perf['total_return']:.2f}% (vs {baseline_perf['total_return']:.2f}% baseline)")
    print(f"  Improvement: {((best_perf['total_return'] / baseline_perf['total_return']) - 1) * 100:.1f}%")
    print(f"  Sharpe Ratio: {best_perf['sharpe_ratio']:.2f} (vs {baseline_perf['sharpe_ratio']:.2f} baseline)")
    print(f"  Avg Hold Days: {best_perf['hold_stats']['avg_hold_days']:.2f} days")

    # Validation
    return_improvement_pct = ((best_perf['total_return'] / baseline_perf['total_return']) - 1) * 100
    sharpe_improvement_pct = ((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100

    print(f"\n{'='*70}")
    print("VALIDATION:")
    print(f"{'='*70}")
    print(f"Target: +20% return improvement AND +10% Sharpe")
    print(f"Actual: {return_improvement_pct:+.1f}% return, {sharpe_improvement_pct:+.1f}% Sharpe")

    if return_improvement_pct >= 20 and sharpe_improvement_pct >= 10:
        print("SUCCESS: Targets achieved!")
        print(f"\nRECOMMENDATION: Deploy {best_strategy.replace('_', ' ').title()} to production")
    elif return_improvement_pct >= 20 or sharpe_improvement_pct >= 10:
        print("PARTIAL: One target achieved")
        print("\nRECOMMENDATION: Consider deployment pending review")
    else:
        print("REVIEW: Targets not fully achieved")
        print("\nRECOMMENDATION: Analyze results before deployment")

    print(f"\n{'='*70}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

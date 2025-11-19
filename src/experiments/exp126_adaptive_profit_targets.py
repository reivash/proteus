"""
EXP-126: Adaptive Profit Target Optimization

CONTEXT:
Current system uses uniform 2.0% profit target for all trades. This is suboptimal because:
- Elite signals (90+ strength, 77.8% WR) can achieve higher returns
- High-volatility stocks naturally move more, allowing higher targets
- Uniform targets leave money on the table for best opportunities

HYPOTHESIS:
Adaptive profit targets based on signal quality + volatility improve total returns:
- Elite signals (90-100 strength): 3.0% target (high confidence, let it run)
- Strong signals (80-89 strength): 2.5% target (good confidence)
- Good signals (70-79 strength): 2.0% target (baseline)
- Acceptable signals (65-69 strength): 1.5% target (take profit quickly)
- BONUS: High-volatility stocks (ATR > 2.5%) get +0.5% higher targets

OBJECTIVE:
Maximize total returns while maintaining or improving win rate.

METHODOLOGY:
Test 4 strategies:
1. Baseline: 2.0% uniform target (current)
2. Signal-based: Target scales with signal strength only
3. Volatility-based: Target scales with ATR only
4. Combined: Signal strength + ATR adaptive targets

EXPECTED IMPACT:
- Avg return per trade: +0.4-0.7% (elite signals capture more upside)
- Total return: +40-60% annual (on 90-100 trades/year)
- Win rate: Unchanged or slightly lower (higher targets = harder to hit)
- Sharpe ratio: +10-15% (bigger wins offset any WR decline)

SUCCESS CRITERIA:
- Total return >= +30% improvement over baseline
- Sharpe ratio >= 2.6 (+10% minimum)
- Deploy if validated

SYNERGY WITH EXP-125:
- EXP-125: Adaptive stop-loss (reduce bad exits)
- EXP-126: Adaptive profit targets (maximize good exits)
- Together: Optimal risk/reward for each trade

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


def calculate_adaptive_profit_target(
    signal_strength: float,
    atr_pct: float = None,
    strategy: str = 'baseline'
) -> float:
    """
    Calculate adaptive profit target based on signal quality and volatility.

    Args:
        signal_strength: Signal quality score (0-100)
        atr_pct: ATR as percentage of price (optional)
        strategy: 'baseline', 'signal_based', 'volatility_based', or 'combined'

    Returns:
        Profit target percentage (positive value)
    """
    if strategy == 'baseline':
        return 2.0  # Current uniform target

    # Signal-based component
    if signal_strength >= 90:
        signal_target = 3.0  # Elite: aim high
    elif signal_strength >= 80:
        signal_target = 2.5  # Strong: above average
    elif signal_strength >= 70:
        signal_target = 2.0  # Good: baseline
    else:
        signal_target = 1.5  # Acceptable: take profit quickly

    if strategy == 'signal_based':
        return signal_target

    # Volatility-based component
    vol_boost = 0.0
    if atr_pct and atr_pct > 2.5:
        vol_boost = 0.5  # High volatility = higher targets achievable

    if strategy == 'volatility_based':
        return 2.0 + vol_boost  # Baseline + volatility adjustment

    # Combined strategy
    if strategy == 'combined':
        return signal_target + vol_boost

    return 2.0  # Default


def backtest_profit_target_strategy(
    start_date: str,
    end_date: str,
    strategy: str = 'baseline'
) -> dict:
    """
    Backtest with different profit target strategies.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        strategy: 'baseline', 'signal_based', 'volatility_based', or 'combined'

    Returns:
        Dictionary with performance metrics
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {strategy.upper()} PROFIT TARGET STRATEGY")
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
        profit_target=2.0,  # Will be overridden for adaptive strategies
        stop_loss=-2.0,
        max_hold_days=2,
        use_limit_orders=True
    )

    # Track profit target statistics
    target_stats = {
        'total_exits': 0,
        'profit_target_hits': 0,
        'stop_loss_exits': 0,
        'time_decay_exits': 0,
        'targets_by_size': {},
        'avg_target': 0.0,
        'signals_by_tier': {'ELITE': 0, 'STRONG': 0, 'GOOD': 0, 'ACCEPTABLE': 0}
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

            # Apply adaptive profit targets for non-baseline strategies
            if strategy != 'baseline':
                for signal in signals:
                    # Get signal data
                    atr = signal.get('atr', 2.0)
                    price = signal.get('price', 100)
                    atr_pct = (atr / price) * 100 if price > 0 else 2.0
                    signal_strength = signal.get('signal_strength', 0)

                    # Calculate adaptive profit target
                    profit_target = calculate_adaptive_profit_target(
                        signal_strength,
                        atr_pct,
                        strategy
                    )

                    # Override signal profit target
                    signal['profit_target_pct'] = profit_target

                    # Track statistics
                    target_key = f"{profit_target:.1f}%"
                    target_stats['targets_by_size'][target_key] = target_stats['targets_by_size'].get(target_key, 0) + 1

                    # Track signal tiers
                    if signal_strength >= 90:
                        tier = 'ELITE'
                    elif signal_strength >= 80:
                        tier = 'STRONG'
                    elif signal_strength >= 70:
                        tier = 'GOOD'
                    else:
                        tier = 'ACCEPTABLE'
                    target_stats['signals_by_tier'][tier] += 1

                    print(f"  {signal['ticker']}: {tier} ({signal_strength:.0f}) -> Target = {profit_target:.1f}% (ATR: {atr_pct:.2f}%)")

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
                    target_stats['total_exits'] += 1
                    reason = exit_trade.get('exit_reason', 'unknown')
                    if 'profit' in reason.lower() or 'target' in reason.lower():
                        target_stats['profit_target_hits'] += 1
                    elif 'stop' in reason.lower():
                        target_stats['stop_loss_exits'] += 1
                    else:
                        target_stats['time_decay_exits'] += 1

        # Move to next trading day
        current_dt += timedelta(days=1)

    # Get performance
    perf = trader.get_performance()

    # Calculate average target
    if target_stats['targets_by_size']:
        total_targets = sum(target_stats['targets_by_size'].values())
        weighted_sum = sum(
            float(key.replace('%', '')) * count
            for key, count in target_stats['targets_by_size'].items()
        )
        target_stats['avg_target'] = weighted_sum / total_targets if total_targets > 0 else 2.0

    # Add profit target statistics
    perf['target_stats'] = target_stats
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

    # Print profit target statistics
    if 'target_stats' in perf:
        stats = perf['target_stats']
        print(f"\n{'='*70}")
        print("PROFIT TARGET BREAKDOWN:")
        print(f"{'='*70}")
        print(f"Average target: {stats['avg_target']:.2f}%")
        print(f"Total exits: {stats['total_exits']}")
        if stats['total_exits'] > 0:
            print(f"Profit target hits: {stats['profit_target_hits']} ({100*stats['profit_target_hits']/stats['total_exits']:.1f}%)")
            print(f"Stop-loss exits: {stats['stop_loss_exits']} ({100*stats['stop_loss_exits']/stats['total_exits']:.1f}%)")
            print(f"Time decay exits: {stats['time_decay_exits']} ({100*stats['time_decay_exits']/stats['total_exits']:.1f}%)")

        print(f"\nSignal tier distribution:")
        for tier, count in stats['signals_by_tier'].items():
            if count > 0:
                print(f"  {tier}: {count} signals")

        if stats['targets_by_size']:
            print(f"\nTarget distribution:")
            for size, count in sorted(stats['targets_by_size'].items()):
                print(f"  {size}: {count} signals")

    print(f"{'='*70}\n")


def compare_results(results: Dict[str, dict]):
    """Print comparison table of all results."""
    print(f"\n{'='*70}")
    print("COMPARISON: PROFIT TARGET STRATEGIES")
    print(f"{'='*70}\n")

    # Create comparison table
    headers = ["Strategy", "Trades", "WR%", "Avg$/Trade", "Total%", "Sharpe", "MaxDD%", "PF", "AvgTarget%"]
    rows = []

    for strategy, perf in results.items():
        avg_target = perf['target_stats']['avg_target']

        row = [
            strategy.replace('_', ' ').title(),
            perf['total_trades'],
            f"{perf['win_rate']:.1f}",
            f"{perf.get('avg_pnl', 0):.2f}",
            f"{perf['total_return']:.2f}",
            f"{perf.get('sharpe_ratio', 0):.2f}",
            f"{perf.get('max_drawdown', 0):.2f}",
            f"{perf.get('profit_factor', 0):.2f}",
            f"{avg_target:.2f}"
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
    """Run EXP-126 backtest."""
    print("="*70)
    print("EXP-126: ADAPTIVE PROFIT TARGET OPTIMIZATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("OBJECTIVE: Maximize returns via signal-quality-based profit targets")
    print("THEORY: Elite signals deserve higher targets to capture full upside\n")

    # Test period: 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~2 years

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}\n")

    # Run all strategies
    results = {}

    print("\n" + "="*70)
    print("BASELINE: Uniform 2.0% Profit Target (Current)")
    print("="*70)
    results['baseline'] = backtest_profit_target_strategy(start_str, end_str, strategy='baseline')
    print_performance(results['baseline'], "BASELINE PERFORMANCE (2.0% UNIFORM)")

    print("\n" + "="*70)
    print("TEST 1: Signal-Based Adaptive Targets")
    print("="*70)
    results['signal_based'] = backtest_profit_target_strategy(start_str, end_str, strategy='signal_based')
    print_performance(results['signal_based'], "SIGNAL-BASED PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 2: Volatility-Based Adaptive Targets")
    print("="*70)
    results['volatility_based'] = backtest_profit_target_strategy(start_str, end_str, strategy='volatility_based')
    print_performance(results['volatility_based'], "VOLATILITY-BASED PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 3: Combined (Signal + Volatility) Adaptive Targets")
    print("="*70)
    results['combined'] = backtest_profit_target_strategy(start_str, end_str, strategy='combined')
    print_performance(results['combined'], "COMBINED PERFORMANCE")

    # Compare all results
    compare_results(results)

    # Find best strategy by total return
    best_strategy = max(results.keys(), key=lambda k: results[k]['total_return'])
    best_perf = results[best_strategy]
    baseline_perf = results['baseline']

    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print(f"Best Strategy: {best_strategy.replace('_', ' ').title()}")
    print(f"  Total Return: {best_perf['total_return']:.2f}% (vs {baseline_perf['total_return']:.2f}% baseline)")
    print(f"  Improvement: {best_perf['total_return'] - baseline_perf['total_return']:.2f}pp")
    print(f"  Sharpe Ratio: {best_perf.get('sharpe_ratio', 0):.2f} (vs {baseline_perf.get('sharpe_ratio', 0):.2f} baseline)")
    print(f"  Avg PnL/Trade: ${best_perf.get('avg_pnl', 0):.2f} (vs ${baseline_perf.get('avg_pnl', 0):.2f} baseline)")

    # Validation
    return_improvement_pct = ((best_perf['total_return'] / baseline_perf['total_return']) - 1) * 100 if baseline_perf['total_return'] > 0 else 0
    sharpe_improvement_pct = ((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100 if baseline_perf['sharpe_ratio'] > 0 else 0

    print(f"\n{'='*70}")
    print("VALIDATION:")
    print(f"{'='*70}")
    print(f"Target: +30% total return improvement AND +10% Sharpe")
    print(f"Actual: {return_improvement_pct:+.1f}% return, {sharpe_improvement_pct:+.1f}% Sharpe")

    if return_improvement_pct >= 30 and sharpe_improvement_pct >= 10:
        print("SUCCESS: Both targets achieved!")
        print(f"\nRECOMMENDATION: Deploy {best_strategy.replace('_', ' ').title()} to production")
    elif return_improvement_pct >= 30 or sharpe_improvement_pct >= 10:
        print("PARTIAL: One target achieved")
        print("\nRECOMMENDATION: Consider deployment pending review")
    else:
        print("MARGINAL: Review results")
        print("\nRECOMMENDATION: Evaluate cost/benefit before deployment")

    print(f"\n{'='*70}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

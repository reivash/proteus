"""
EXP-128: VIX Entry Filter Optimization

CONTEXT:
Current system trades mean reversion signals in ALL volatility regimes.
Research shows mean reversion alpha is 2-3x higher during elevated VIX:
- Low VIX (< 15): Stable markets, weak mean reversion
- Medium VIX (15-20): Normal volatility, baseline performance
- High VIX (> 20): Fear/uncertainty, strong mean reversion opportunities

HYPOTHESIS:
Only trading when VIX > threshold improves win rate and returns by
concentrating capital on highest-probability setups.

OBJECTIVE:
Maximize risk-adjusted returns via volatility regime filtering.

METHODOLOGY:
Test 5 strategies:
1. Baseline: No VIX filter (current)
2. VIX > 15: Filter out very low volatility periods
3. VIX > 18: Moderate filter (target normal+ volatility)
4. VIX > 20: High filter (only elevated fear)
5. VIX > 25: Extreme filter (crisis-only trading)

EXPECTED IMPACT:
- Win rate: +5-8pp (better signal quality in volatile periods)
- Avg return per trade: +0.5-1.0% (stronger reversions)
- Sharpe ratio: +15-25% (better risk-adjusted returns)
- Trade frequency: -30-60% (quality over quantity)
- Max drawdown: -10-20% (avoid low-vol whipsaws)

SUCCESS CRITERIA:
- Sharpe ratio >= 2.8 (+18% minimum)
- Win rate >= 70% (+5pp minimum)
- Deploy optimal threshold if validated

COMPLEMENTARY TO:
- EXP-093: Q4 signal filter (signal quality)
- EXP-110: VIX position sizing (size adjustment)
- EXP-125/126/127: Adaptive risk (exit optimization)
= COMPLETE MULTI-LAYER RISK MANAGEMENT

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


def get_vix_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch VIX data for the backtest period.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with VIX close prices indexed by date
    """
    fetcher = RateLimitedYahooFinanceFetcher(verbose=True)

    print(f"Fetching VIX data from {start_date} to {end_date}...")

    try:
        vix_df = fetcher.fetch_stock_data(
            '^VIX',
            start_date=start_date,
            end_date=end_date
        )

        if vix_df is not None and not vix_df.empty:
            # Keep only Close column and rename
            vix_df = vix_df[['Close']].copy()
            vix_df.columns = ['VIX']
            print(f"Loaded {len(vix_df)} VIX data points")
            return vix_df
        else:
            print("[WARN] No VIX data found")
            return pd.DataFrame()

    except Exception as e:
        print(f"[ERROR] Failed to fetch VIX: {e}")
        return pd.DataFrame()


def should_trade_on_date(date_str: str, vix_df: pd.DataFrame, vix_threshold: float) -> tuple:
    """
    Check if we should trade on this date based on VIX.

    Args:
        date_str: Date to check (YYYY-MM-DD)
        vix_df: VIX DataFrame
        vix_threshold: Minimum VIX value to trade

    Returns:
        (should_trade: bool, vix_value: float)
    """
    if vix_df.empty:
        return True, None  # If no VIX data, trade anyway (baseline)

    date = pd.to_datetime(date_str)

    # Find closest VIX value (handle weekends/holidays)
    if date in vix_df.index:
        vix_value = float(vix_df.loc[date, 'VIX'])
    else:
        # Get most recent VIX value
        prior_dates = vix_df.index[vix_df.index <= date]
        if len(prior_dates) > 0:
            vix_value = float(vix_df.loc[prior_dates[-1], 'VIX'])
        else:
            return True, None  # No prior data, trade anyway

    should_trade = vix_value >= vix_threshold

    return should_trade, vix_value


def backtest_vix_filter_strategy(
    start_date: str,
    end_date: str,
    vix_threshold: float = None
) -> dict:
    """
    Backtest with VIX entry filter.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        vix_threshold: Minimum VIX to trade (None = no filter)

    Returns:
        Dictionary with performance metrics
    """
    strategy_name = f"VIX > {vix_threshold}" if vix_threshold else "Baseline (No Filter)"

    print(f"\n{'='*70}")
    print(f"BACKTEST: {strategy_name}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*70}\n")

    # Fetch VIX data first
    vix_df = get_vix_data(start_date, end_date)

    # Initialize components
    fetcher = RateLimitedYahooFinanceFetcher(verbose=True)
    scanner = SignalScanner(
        lookback_days=90,
        min_signal_strength=65.0  # Q4 filter deployed
    )

    trader = PaperTrader(
        initial_capital=90000,
        max_positions=5,
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=2,
        use_limit_orders=True
    )

    # Track VIX filter statistics
    vix_stats = {
        'total_signal_days': 0,
        'filtered_out_days': 0,
        'traded_days': 0,
        'vix_values_on_signals': [],
        'vix_values_traded': []
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
            vix_stats['total_signal_days'] += 1

            # Check VIX filter
            should_trade, vix_value = should_trade_on_date(date_str, vix_df, vix_threshold or 0)

            if vix_value is not None:
                vix_stats['vix_values_on_signals'].append(vix_value)

            if should_trade:
                vix_stats['traded_days'] += 1
                if vix_value is not None:
                    vix_stats['vix_values_traded'].append(vix_value)

                print(f"\n{'='*70}")
                print(f"Date: {date_str} | VIX: {vix_value:.2f if vix_value else 'N/A'} | Signals: {len(signals)}")
                print(f"{'='*70}")

                # Process signals
                trader.process_signals(signals, date_str)
            else:
                vix_stats['filtered_out_days'] += 1
                print(f"[FILTER] {date_str}: VIX {vix_value:.2f} < {vix_threshold} threshold - {len(signals)} signal(s) skipped")

        # Update positions and check exits (always check, even if not entering new trades)
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
                trader.check_exits(prices, date_str)

        # Move to next trading day
        current_dt += timedelta(days=1)

    # Get performance
    perf = trader.get_performance()

    # Add VIX filter statistics
    if vix_stats['vix_values_on_signals']:
        vix_stats['avg_vix_on_signals'] = np.mean(vix_stats['vix_values_on_signals'])
        vix_stats['avg_vix_traded'] = np.mean(vix_stats['vix_values_traded']) if vix_stats['vix_values_traded'] else 0
    else:
        vix_stats['avg_vix_on_signals'] = 0
        vix_stats['avg_vix_traded'] = 0

    perf['vix_stats'] = vix_stats
    perf['strategy'] = strategy_name

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

    # Print VIX filter statistics
    if 'vix_stats' in perf:
        stats = perf['vix_stats']
        print(f"\n{'='*70}")
        print("VIX FILTER STATISTICS:")
        print(f"{'='*70}")
        print(f"Total signal days: {stats['total_signal_days']}")
        print(f"Traded days: {stats['traded_days']}")
        print(f"Filtered out days: {stats['filtered_out_days']}")
        if stats['total_signal_days'] > 0:
            filter_rate = 100 * stats['filtered_out_days'] / stats['total_signal_days']
            print(f"Filter rate: {filter_rate:.1f}%")
        print(f"Avg VIX on all signal days: {stats['avg_vix_on_signals']:.2f}")
        print(f"Avg VIX on traded days: {stats['avg_vix_traded']:.2f}")

    print(f"{'='*70}\n")


def compare_results(results: Dict[str, dict]):
    """Print comparison table of all results."""
    print(f"\n{'='*70}")
    print("COMPARISON: VIX ENTRY FILTER STRATEGIES")
    print(f"{'='*70}\n")

    # Create comparison table
    headers = ["Strategy", "Trades", "WR%", "Return%", "$/Trade", "Sharpe", "MaxDD%", "PF", "AvgVIX"]
    rows = []

    for strategy, perf in results.items():
        avg_vix = perf['vix_stats']['avg_vix_traded']

        row = [
            strategy,
            perf['total_trades'],
            f"{perf['win_rate']:.1f}",
            f"{perf['total_return']:.2f}",
            f"{perf['avg_return_per_trade']:.2f}",
            f"{perf['sharpe_ratio']:.2f}",
            f"{perf['max_drawdown']:.2f}",
            f"{perf['profit_factor']:.2f}",
            f"{avg_vix:.1f}"
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
    """Run EXP-128 backtest."""
    print("="*70)
    print("EXP-128: VIX ENTRY FILTER OPTIMIZATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("OBJECTIVE: Improve returns by trading only during elevated volatility")
    print("THEORY: Mean reversion alpha is 2-3x higher when VIX is elevated\n")

    # Test period: 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~2 years

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}\n")

    # Run all strategies
    results = {}

    print("\n" + "="*70)
    print("BASELINE: No VIX Filter (Current)")
    print("="*70)
    results['Baseline'] = backtest_vix_filter_strategy(start_str, end_str, vix_threshold=None)
    print_performance(results['Baseline'], "BASELINE PERFORMANCE (NO FILTER)")

    print("\n" + "="*70)
    print("TEST 1: VIX > 15 Filter (Exclude very low vol)")
    print("="*70)
    results['VIX > 15'] = backtest_vix_filter_strategy(start_str, end_str, vix_threshold=15.0)
    print_performance(results['VIX > 15'], "VIX > 15 PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 2: VIX > 18 Filter (Normal+ volatility)")
    print("="*70)
    results['VIX > 18'] = backtest_vix_filter_strategy(start_str, end_str, vix_threshold=18.0)
    print_performance(results['VIX > 18'], "VIX > 18 PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 3: VIX > 20 Filter (Elevated fear)")
    print("="*70)
    results['VIX > 20'] = backtest_vix_filter_strategy(start_str, end_str, vix_threshold=20.0)
    print_performance(results['VIX > 20'], "VIX > 20 PERFORMANCE")

    print("\n" + "="*70)
    print("TEST 4: VIX > 25 Filter (Crisis mode)")
    print("="*70)
    results['VIX > 25'] = backtest_vix_filter_strategy(start_str, end_str, vix_threshold=25.0)
    print_performance(results['VIX > 25'], "VIX > 25 PERFORMANCE")

    # Compare all results
    compare_results(results)

    # Find best strategy
    best_strategy = max(results.keys(), key=lambda k: results[k]['sharpe_ratio'])
    best_perf = results[best_strategy]
    baseline_perf = results['Baseline']

    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print(f"Best Strategy: {best_strategy}")
    print(f"  Sharpe Ratio: {best_perf['sharpe_ratio']:.2f} (vs {baseline_perf['sharpe_ratio']:.2f} baseline)")
    print(f"  Improvement: {((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100:.1f}%")
    print(f"  Win Rate: {best_perf['win_rate']:.1f}% (vs {baseline_perf['win_rate']:.1f}% baseline)")
    print(f"  Total Trades: {best_perf['total_trades']} (vs {baseline_perf['total_trades']} baseline)")

    # Validation
    sharpe_improvement_pct = ((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100
    wr_improvement = best_perf['win_rate'] - baseline_perf['win_rate']

    print(f"\n{'='*70}")
    print("VALIDATION:")
    print(f"{'='*70}")
    print(f"Target: +18% Sharpe improvement AND +5pp win rate")
    print(f"Actual: {sharpe_improvement_pct:+.1f}% Sharpe, {wr_improvement:+.1f}pp win rate")

    if sharpe_improvement_pct >= 18 and wr_improvement >= 5:
        print("SUCCESS: Targets achieved!")
        print(f"\nRECOMMENDATION: Deploy {best_strategy} to production")
    elif sharpe_improvement_pct >= 18 or wr_improvement >= 5:
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

"""
EXP-131: Multi-Timeframe Signal Convergence

CONTEXT:
Current system only uses daily timeframe for mean reversion signals.
Research shows combining multiple timeframes (daily, weekly, monthly)
significantly improves prediction accuracy.

PROBLEM:
- Daily signals can be noise (intraday volatility)
- Lack of trend context leads to false signals
- Ignoring longer-term support/resistance levels

HYPOTHESIS:
When signals align across multiple timeframes, success rate increases:
- Daily: Mean reversion opportunity
- Weekly: Trend confirmation (not fighting the trend)
- Monthly: Near major support level
→ CONVERGENCE = High-probability setup

METHODOLOGY:
Test 4 strategies:
1. Baseline: Daily signals only (current)
2. Daily + Weekly convergence (2 timeframes)
3. Daily + Monthly convergence (2 timeframes)
4. Full convergence (daily + weekly + monthly, all 3 agree)

TIMEFRAME DEFINITIONS:
- Daily: Current mean reversion signal (oversold bounce)
- Weekly: 5-day trend alignment (price above 5-day MA = uptrend)
- Monthly: 20-day support level (price near 20-day low)

EXPECTED IMPACT:
- Win rate: 63.7% → 75-82% (+12-18pp improvement)
- Sharpe ratio: 2.37 → 3.3-3.8 (+39-60%)
- Trade frequency: -30-50% (higher quality, fewer false signals)
- Risk-adjusted returns: Significantly better

SUCCESS CRITERIA:
- Win rate >= 75%
- Sharpe ratio >= 3.0 (+27% minimum)
- Deploy best convergence strategy if validated

RESEARCH BASIS:
- Bulkowski (2008): Multi-timeframe analysis improves pattern reliability
- Covel (2004): Trend-following requires timeframe alignment
- Murphy (1999): Support/resistance levels strengthen across timeframes

Created: 2025-11-19
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.trading.signal_scanner import SignalScanner
from src.trading.paper_trader import PaperTrader
from src.data.fetchers.rate_limited_yahoo import RateLimitedYahooFinanceFetcher


def analyze_timeframes(
    ticker: str,
    date: str,
    fetcher: RateLimitedYahooFinanceFetcher,
    lookback_days: int = 30
) -> Dict[str, bool]:
    """
    Analyze signal alignment across multiple timeframes.

    Args:
        ticker: Stock ticker
        date: Analysis date (YYYY-MM-DD)
        fetcher: Data fetcher instance
        lookback_days: Lookback period for analysis

    Returns:
        Dictionary with timeframe signals:
        - daily_signal: Mean reversion opportunity exists
        - weekly_uptrend: Price above 5-day MA
        - monthly_support: Price near 20-day low (within 5%)
    """
    # Get historical data
    start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

    try:
        data = fetcher.get_stock_data(ticker, start_date, date)

        if data is None or len(data) < 20:
            return {
                'daily_signal': False,
                'weekly_uptrend': False,
                'monthly_support': False,
            }

        # Calculate indicators
        close_prices = data['Close'].values
        current_price = close_prices[-1]

        # Daily signal: Already provided by SignalScanner (mean reversion)
        daily_signal = True  # Assume signal exists (filtered later)

        # Weekly timeframe: 5-day moving average trend
        if len(close_prices) >= 5:
            ma_5 = np.mean(close_prices[-5:])
            weekly_uptrend = current_price > ma_5  # Uptrend confirmation
        else:
            weekly_uptrend = False

        # Monthly timeframe: Near 20-day low (support level)
        if len(close_prices) >= 20:
            low_20 = np.min(close_prices[-20:])
            monthly_support = current_price <= low_20 * 1.05  # Within 5% of 20-day low
        else:
            monthly_support = False

        return {
            'daily_signal': daily_signal,
            'weekly_uptrend': weekly_uptrend,
            'monthly_support': monthly_support,
        }

    except Exception as e:
        print(f"[ERROR] Failed to analyze timeframes for {ticker}: {e}")
        return {
            'daily_signal': False,
            'weekly_uptrend': False,
            'monthly_support': False,
        }


def calculate_convergence_score(timeframes: Dict[str, bool], strategy: str) -> Tuple[int, bool]:
    """
    Calculate convergence score based on timeframe alignment.

    Args:
        timeframes: Dictionary of timeframe signals
        strategy: 'baseline', 'daily_weekly', 'daily_monthly', or 'full_convergence'

    Returns:
        Tuple of (convergence_score, signal_accepted)
    """
    if strategy == 'baseline':
        # No timeframe filtering
        return 1, timeframes['daily_signal']

    elif strategy == 'daily_weekly':
        # Daily + Weekly convergence
        score = sum([timeframes['daily_signal'], timeframes['weekly_uptrend']])
        accepted = score >= 2
        return score, accepted

    elif strategy == 'daily_monthly':
        # Daily + Monthly convergence
        score = sum([timeframes['daily_signal'], timeframes['monthly_support']])
        accepted = score >= 2
        return score, accepted

    elif strategy == 'full_convergence':
        # All 3 timeframes must agree
        score = sum(timeframes.values())
        accepted = score >= 3
        return score, accepted

    return 0, False


def backtest_timeframe_strategy(
    start_date: str,
    end_date: str,
    strategy: str = 'baseline'
) -> dict:
    """
    Backtest with multi-timeframe convergence filtering.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        strategy: 'baseline', 'daily_weekly', 'daily_monthly', or 'full_convergence'

    Returns:
        Dictionary with performance metrics
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {strategy.upper().replace('_', ' ')} STRATEGY")
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
        max_hold_days=2,
        use_limit_orders=True
    )

    # Track timeframe statistics
    timeframe_stats = {
        'total_signals': 0,
        'signals_accepted': 0,
        'signals_rejected': 0,
        'convergence_scores': [],
        'timeframe_breakdown': {
            'daily_only': 0,
            'weekly_only': 0,
            'monthly_only': 0,
            'daily_weekly': 0,
            'daily_monthly': 0,
            'weekly_monthly': 0,
            'full_convergence': 0,
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
            print(f"Raw signals found: {len(signals)}")

            # Apply timeframe filtering for non-baseline strategies
            filtered_signals = []

            for signal in signals:
                timeframe_stats['total_signals'] += 1
                ticker = signal['ticker']

                # Analyze timeframes
                timeframes = analyze_timeframes(ticker, date_str, fetcher)

                # Calculate convergence
                score, accepted = calculate_convergence_score(timeframes, strategy)
                timeframe_stats['convergence_scores'].append(score)

                # Track breakdown
                if timeframes['daily_signal'] and timeframes['weekly_uptrend'] and timeframes['monthly_support']:
                    timeframe_stats['timeframe_breakdown']['full_convergence'] += 1
                elif timeframes['daily_signal'] and timeframes['weekly_uptrend']:
                    timeframe_stats['timeframe_breakdown']['daily_weekly'] += 1
                elif timeframes['daily_signal'] and timeframes['monthly_support']:
                    timeframe_stats['timeframe_breakdown']['daily_monthly'] += 1
                elif timeframes['weekly_uptrend'] and timeframes['monthly_support']:
                    timeframe_stats['timeframe_breakdown']['weekly_monthly'] += 1
                elif timeframes['daily_signal']:
                    timeframe_stats['timeframe_breakdown']['daily_only'] += 1
                elif timeframes['weekly_uptrend']:
                    timeframe_stats['timeframe_breakdown']['weekly_only'] += 1
                elif timeframes['monthly_support']:
                    timeframe_stats['timeframe_breakdown']['monthly_only'] += 1

                if accepted:
                    timeframe_stats['signals_accepted'] += 1
                    filtered_signals.append(signal)
                    print(f"  {ticker}: ACCEPTED (Score: {score}/3, Daily: {timeframes['daily_signal']}, Weekly: {timeframes['weekly_uptrend']}, Monthly: {timeframes['monthly_support']})")
                else:
                    timeframe_stats['signals_rejected'] += 1
                    print(f"  {ticker}: REJECTED (Score: {score}/3, Daily: {timeframes['daily_signal']}, Weekly: {timeframes['weekly_uptrend']}, Monthly: {timeframes['monthly_support']})")

            print(f"Filtered signals: {len(filtered_signals)}")
            print(f"{'='*70}")

            # Process filtered signals
            if filtered_signals:
                trader.process_signals(filtered_signals, date_str)

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
                trader.check_exits(prices, date_str)

        # Move to next trading day
        current_dt += timedelta(days=1)

    # Get performance
    perf = trader.get_performance()

    # Add timeframe statistics
    if timeframe_stats['convergence_scores']:
        timeframe_stats['avg_convergence_score'] = np.mean(timeframe_stats['convergence_scores'])
    else:
        timeframe_stats['avg_convergence_score'] = 0.0

    if timeframe_stats['total_signals'] > 0:
        timeframe_stats['acceptance_rate'] = (timeframe_stats['signals_accepted'] / timeframe_stats['total_signals']) * 100
    else:
        timeframe_stats['acceptance_rate'] = 0.0

    perf['timeframe_stats'] = timeframe_stats
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

    # Print timeframe statistics
    if 'timeframe_stats' in perf:
        stats = perf['timeframe_stats']
        print(f"\n{'='*70}")
        print("TIMEFRAME CONVERGENCE ANALYSIS:")
        print(f"{'='*70}")
        print(f"Total raw signals: {stats['total_signals']}")
        print(f"Signals accepted: {stats['signals_accepted']}")
        print(f"Signals rejected: {stats['signals_rejected']}")
        print(f"Acceptance rate: {stats['acceptance_rate']:.1f}%")
        print(f"Avg convergence score: {stats['avg_convergence_score']:.2f}/3")

        print(f"\nTimeframe breakdown:")
        breakdown = stats['timeframe_breakdown']
        total = sum(breakdown.values())
        if total > 0:
            for key, count in breakdown.items():
                pct = (count / total) * 100
                print(f"  {key.replace('_', ' ').title()}: {count} ({pct:.1f}%)")

    print(f"{'='*70}\n")


def compare_results(results: Dict[str, dict]):
    """Print comparison table of all results."""
    print(f"\n{'='*70}")
    print("COMPARISON: MULTI-TIMEFRAME CONVERGENCE STRATEGIES")
    print(f"{'='*70}\n")

    # Create comparison table
    headers = ["Strategy", "Trades", "WR%", "Return%", "$/Trade", "Sharpe", "MaxDD%", "PF", "Accept%"]
    rows = []

    for strategy, perf in results.items():
        accept_rate = perf['timeframe_stats']['acceptance_rate']

        row = [
            strategy.replace('_', ' ').title(),
            perf['total_trades'],
            f"{perf['win_rate']:.1f}",
            f"{perf['total_return']:.2f}",
            f"{perf.get('avg_pnl', 0):.2f}",
            f"{perf.get('sharpe_ratio', 0):.2f}",
            f"{perf.get('max_drawdown', 0):.2f}",
            f"{perf.get('profit_factor', 0):.2f}",
            f"{accept_rate:.1f}",
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
    """Run EXP-131 backtest."""
    print("="*70)
    print("EXP-131: MULTI-TIMEFRAME SIGNAL CONVERGENCE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("OBJECTIVE: Improve win rate via timeframe alignment")
    print("THEORY: Signals that align across timeframes have higher success rates\n")

    # Test period: 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~2 years

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}\n")

    # Run all strategies
    results = {}

    print("\n" + "="*70)
    print("BASELINE: Daily Signals Only (No Timeframe Filtering)")
    print("="*70)
    results['baseline'] = backtest_timeframe_strategy(start_str, end_str, strategy='baseline')
    print_performance(results['baseline'], "BASELINE PERFORMANCE (DAILY ONLY)")

    print("\n" + "="*70)
    print("TEST 1: Daily + Weekly Convergence (2 Timeframes)")
    print("="*70)
    results['daily_weekly'] = backtest_timeframe_strategy(start_str, end_str, strategy='daily_weekly')
    print_performance(results['daily_weekly'], "DAILY + WEEKLY CONVERGENCE")

    print("\n" + "="*70)
    print("TEST 2: Daily + Monthly Convergence (2 Timeframes)")
    print("="*70)
    results['daily_monthly'] = backtest_timeframe_strategy(start_str, end_str, strategy='daily_monthly')
    print_performance(results['daily_monthly'], "DAILY + MONTHLY CONVERGENCE")

    print("\n" + "="*70)
    print("TEST 3: Full Convergence (All 3 Timeframes)")
    print("="*70)
    results['full_convergence'] = backtest_timeframe_strategy(start_str, end_str, strategy='full_convergence')
    print_performance(results['full_convergence'], "FULL TIMEFRAME CONVERGENCE")

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
    print(f"  Win Rate: {best_perf['win_rate']:.1f}% (vs {baseline_perf['win_rate']:.1f}% baseline)")
    print(f"  Improvement: {best_perf['win_rate'] - baseline_perf['win_rate']:+.1f}pp")
    print(f"  Sharpe Ratio: {best_perf['sharpe_ratio']:.2f} (vs {baseline_perf['sharpe_ratio']:.2f} baseline)")
    print(f"  Trades: {best_perf['total_trades']} (vs {baseline_perf['total_trades']} baseline)")

    # Validation
    win_rate_improvement = best_perf['win_rate'] - baseline_perf['win_rate']
    sharpe_improvement_pct = ((best_perf['sharpe_ratio'] / baseline_perf['sharpe_ratio']) - 1) * 100

    print(f"\n{'='*70}")
    print("VALIDATION:")
    print(f"{'='*70}")
    print(f"Target: +12pp win rate AND +27% Sharpe")
    print(f"Actual: {win_rate_improvement:+.1f}pp win rate, {sharpe_improvement_pct:+.1f}% Sharpe")

    if win_rate_improvement >= 12 and sharpe_improvement_pct >= 27:
        print("SUCCESS: Targets achieved!")
        print(f"\nRECOMMENDATION: Deploy {best_strategy.replace('_', ' ').title()} to production")
    elif win_rate_improvement >= 12 or sharpe_improvement_pct >= 27:
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

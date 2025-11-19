"""
EXP-132: Short Interest Squeeze Detection

CONTEXT:
Current system ignores short interest data. This misses a powerful catalyst:
high short interest + mean reversion = potential short squeeze.

PROBLEM:
- Short squeezes can amplify mean reversion returns by 50-200%
- High short interest (>15%) creates forced buying pressure
- Shorts covering = accelerated price recovery
- Current system treats all reversals equally

HYPOTHESIS:
Stocks with high short interest that trigger mean reversion signals have:
1. Higher win rates (+5-8pp)
2. Larger returns per trade (+0.5-1.0%)
3. Faster recovery (shorts forced to cover)

OPPORTUNITY:
Short interest data is freely available via Yahoo Finance.
No external paid API required.

METHODOLOGY:
Test 3 strategies:
1. Baseline: All mean reversion signals (current)
2. High Short Interest: Only trade stocks with short interest >15%
3. Moderate Short Interest: Only trade stocks with short interest >10%
4. Low Short Interest: Avoid stocks with short interest <5%

SHORT INTEREST THRESHOLDS:
- EXTREME (>20%): Highest squeeze potential
- HIGH (15-20%): Strong squeeze potential
- MODERATE (10-15%): Medium squeeze potential
- NORMAL (5-10%): Low squeeze potential
- LOW (<5%): Minimal squeeze potential

EXPECTED IMPACT:
- Win rate: 63.7% → 69-72% (+5-8pp on high SI stocks)
- Return per trade: +0.5-1.0% (squeeze amplification)
- Sharpe ratio: 2.37 → 2.7-3.0 (+15-25%)
- Trade frequency: -30-40% (selective filtering)

SUCCESS CRITERIA:
- Win rate >= 70% on high short interest stocks
- Sharpe ratio >= 2.7 (+15% minimum)
- Deploy if validated

RESEARCH BASIS:
- Dechow et al. (2001): Short interest predicts returns
- Asquith et al. (2005): Short squeeze dynamics
- Diether et al. (2009): High short interest = reversal opportunity

Created: 2025-11-19
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.trading.signal_scanner import SignalScanner
from src.trading.paper_trader import PaperTrader
from src.data.fetchers.rate_limited_yahoo import RateLimitedYahooFinanceFetcher


def get_short_interest(
    ticker: str,
    date: str,
    fetcher: RateLimitedYahooFinanceFetcher
) -> Optional[float]:
    """
    Get short interest percentage for a stock.

    Args:
        ticker: Stock ticker
        date: Date (YYYY-MM-DD)
        fetcher: Data fetcher instance

    Returns:
        Short interest as percentage of float, or None if unavailable
    """
    try:
        # Get stock info from Yahoo Finance
        import yfinance as yf

        stock = yf.Ticker(ticker)
        info = stock.info

        # Get short interest metrics
        shares_short = info.get('sharesShort', 0)
        float_shares = info.get('floatShares', 0)

        if float_shares > 0 and shares_short > 0:
            short_interest_pct = (shares_short / float_shares) * 100
            return short_interest_pct

        # Alternative: shortPercentOfFloat
        short_pct = info.get('shortPercentOfFloat')
        if short_pct is not None:
            return short_pct * 100  # Convert to percentage

        return None

    except Exception as e:
        print(f"[WARN] Could not get short interest for {ticker}: {e}")
        return None


def classify_short_interest(short_interest_pct: Optional[float]) -> str:
    """
    Classify short interest level.

    Args:
        short_interest_pct: Short interest as percentage of float

    Returns:
        Classification: 'extreme', 'high', 'moderate', 'normal', 'low', or 'unknown'
    """
    if short_interest_pct is None:
        return 'unknown'

    if short_interest_pct >= 20:
        return 'extreme'  # >20%: Highest squeeze potential
    elif short_interest_pct >= 15:
        return 'high'     # 15-20%: Strong squeeze potential
    elif short_interest_pct >= 10:
        return 'moderate' # 10-15%: Medium squeeze potential
    elif short_interest_pct >= 5:
        return 'normal'   # 5-10%: Low squeeze potential
    else:
        return 'low'      # <5%: Minimal squeeze potential


def calculate_squeeze_score(
    short_interest_pct: Optional[float],
    signal_strength: float,
    volume_surge: float = 1.0
) -> float:
    """
    Calculate short squeeze potential score.

    Args:
        short_interest_pct: Short interest as percentage of float
        signal_strength: Mean reversion signal strength (65-100)
        volume_surge: Volume relative to average (1.0 = average)

    Returns:
        Squeeze score (0-100)
    """
    if short_interest_pct is None:
        return 0.0

    # Base score from short interest (0-40 points)
    if short_interest_pct >= 20:
        si_score = 40
    elif short_interest_pct >= 15:
        si_score = 30
    elif short_interest_pct >= 10:
        si_score = 20
    else:
        si_score = 10

    # Signal strength component (0-40 points)
    # Normalize 65-100 to 0-40
    signal_score = ((signal_strength - 65) / 35) * 40

    # Volume surge component (0-20 points)
    # High volume = shorts covering
    if volume_surge >= 2.0:
        volume_score = 20
    elif volume_surge >= 1.5:
        volume_score = 15
    elif volume_surge >= 1.2:
        volume_score = 10
    else:
        volume_score = 5

    total_score = si_score + signal_score + volume_score
    return min(total_score, 100)


def backtest_short_interest_strategy(
    start_date: str,
    end_date: str,
    strategy: str = 'baseline'
) -> dict:
    """
    Backtest with short interest filtering.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        strategy: 'baseline', 'high_si', 'moderate_si', or 'low_si_avoid'

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

    # Track short interest statistics
    si_stats = {
        'total_signals': 0,
        'signals_accepted': 0,
        'signals_rejected': 0,
        'short_interest_levels': {
            'extreme': 0,
            'high': 0,
            'moderate': 0,
            'normal': 0,
            'low': 0,
            'unknown': 0,
        },
        'squeeze_scores': [],
        'avg_short_interest': [],
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

            # Apply short interest filtering for non-baseline strategies
            filtered_signals = []

            for signal in signals:
                si_stats['total_signals'] += 1
                ticker = signal['ticker']
                signal_strength = signal.get('signal_strength', 70)

                # Get short interest data
                short_interest = get_short_interest(ticker, date_str, fetcher)
                si_level = classify_short_interest(short_interest)

                # Track statistics
                si_stats['short_interest_levels'][si_level] += 1
                if short_interest is not None:
                    si_stats['avg_short_interest'].append(short_interest)

                # Calculate squeeze score
                volume = signal.get('volume', 1000000)
                avg_volume = signal.get('avg_volume', 1000000)
                volume_surge = volume / avg_volume if avg_volume > 0 else 1.0

                squeeze_score = calculate_squeeze_score(short_interest, signal_strength, volume_surge)
                si_stats['squeeze_scores'].append(squeeze_score)

                # Apply strategy filters
                accepted = False

                if strategy == 'baseline':
                    # No filtering
                    accepted = True

                elif strategy == 'high_si':
                    # Only trade high/extreme short interest (>15%)
                    if short_interest is not None and short_interest >= 15:
                        accepted = True

                elif strategy == 'moderate_si':
                    # Only trade moderate+ short interest (>10%)
                    if short_interest is not None and short_interest >= 10:
                        accepted = True

                elif strategy == 'low_si_avoid':
                    # Avoid low short interest (<5%)
                    if short_interest is None or short_interest >= 5:
                        accepted = True

                if accepted:
                    si_stats['signals_accepted'] += 1
                    # Add short interest metadata to signal
                    signal['short_interest_pct'] = short_interest
                    signal['short_interest_level'] = si_level
                    signal['squeeze_score'] = squeeze_score
                    filtered_signals.append(signal)

                    si_display = f"{short_interest:.1f}%" if short_interest is not None else "N/A"
                    print(f"  {ticker}: ACCEPTED (SI: {si_display}, Level: {si_level}, Squeeze: {squeeze_score:.0f})")
                else:
                    si_stats['signals_rejected'] += 1
                    si_display = f"{short_interest:.1f}%" if short_interest is not None else "N/A"
                    print(f"  {ticker}: REJECTED (SI: {si_display}, Level: {si_level})")

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

    # Add short interest statistics
    if si_stats['avg_short_interest']:
        si_stats['avg_short_interest'] = np.mean(si_stats['avg_short_interest'])
    else:
        si_stats['avg_short_interest'] = 0.0

    if si_stats['squeeze_scores']:
        si_stats['avg_squeeze_score'] = np.mean(si_stats['squeeze_scores'])
    else:
        si_stats['avg_squeeze_score'] = 0.0

    if si_stats['total_signals'] > 0:
        si_stats['acceptance_rate'] = (si_stats['signals_accepted'] / si_stats['total_signals']) * 100
    else:
        si_stats['acceptance_rate'] = 0.0

    perf['si_stats'] = si_stats
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

    # Print short interest statistics
    if 'si_stats' in perf:
        stats = perf['si_stats']
        print(f"\n{'='*70}")
        print("SHORT INTEREST ANALYSIS:")
        print(f"{'='*70}")
        print(f"Total raw signals: {stats['total_signals']}")
        print(f"Signals accepted: {stats['signals_accepted']}")
        print(f"Signals rejected: {stats['signals_rejected']}")
        print(f"Acceptance rate: {stats['acceptance_rate']:.1f}%")
        print(f"Avg short interest: {stats['avg_short_interest']:.2f}%")
        print(f"Avg squeeze score: {stats['avg_squeeze_score']:.1f}/100")

        print(f"\nShort interest distribution:")
        levels = stats['short_interest_levels']
        total = sum(levels.values())
        if total > 0:
            for level, count in levels.items():
                pct = (count / total) * 100
                print(f"  {level.title()}: {count} ({pct:.1f}%)")

    print(f"{'='*70}\n")


def compare_results(results: Dict[str, dict]):
    """Print comparison table of all results."""
    print(f"\n{'='*70}")
    print("COMPARISON: SHORT INTEREST SQUEEZE STRATEGIES")
    print(f"{'='*70}\n")

    # Create comparison table
    headers = ["Strategy", "Trades", "WR%", "Return%", "$/Trade", "Sharpe", "MaxDD%", "PF", "AvgSI%"]
    rows = []

    for strategy, perf in results.items():
        avg_si = perf['si_stats']['avg_short_interest']

        row = [
            strategy.replace('_', ' ').title(),
            perf['total_trades'],
            f"{perf['win_rate']:.1f}",
            f"{perf['total_return']:.2f}",
            f"{perf.get('avg_pnl', 0):.2f}",
            f"{perf.get('sharpe_ratio', 0):.2f}",
            f"{perf.get('max_drawdown', 0):.2f}",
            f"{perf.get('profit_factor', 0):.2f}",
            f"{avg_si:.1f}",
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
    """Run EXP-132 backtest."""
    print("="*70)
    print("EXP-132: SHORT INTEREST SQUEEZE DETECTION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("OBJECTIVE: Exploit short squeeze dynamics for better returns")
    print("THEORY: High short interest + mean reversion = squeeze amplification\n")

    # Test period: 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~2 years

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test period: {start_str} to {end_str}\n")

    # Run all strategies
    results = {}

    print("\n" + "="*70)
    print("BASELINE: All Mean Reversion Signals (No SI Filtering)")
    print("="*70)
    results['baseline'] = backtest_short_interest_strategy(start_str, end_str, strategy='baseline')
    print_performance(results['baseline'], "BASELINE PERFORMANCE (NO SI FILTER)")

    print("\n" + "="*70)
    print("TEST 1: High Short Interest Only (>15%)")
    print("="*70)
    results['high_si'] = backtest_short_interest_strategy(start_str, end_str, strategy='high_si')
    print_performance(results['high_si'], "HIGH SHORT INTEREST STRATEGY")

    print("\n" + "="*70)
    print("TEST 2: Moderate Short Interest (>10%)")
    print("="*70)
    results['moderate_si'] = backtest_short_interest_strategy(start_str, end_str, strategy='moderate_si')
    print_performance(results['moderate_si'], "MODERATE SHORT INTEREST STRATEGY")

    print("\n" + "="*70)
    print("TEST 3: Avoid Low Short Interest (<5%)")
    print("="*70)
    results['low_si_avoid'] = backtest_short_interest_strategy(start_str, end_str, strategy='low_si_avoid')
    print_performance(results['low_si_avoid'], "AVOID LOW SHORT INTEREST STRATEGY")

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
    print(f"Target: +5pp win rate AND +15% Sharpe")
    print(f"Actual: {win_rate_improvement:+.1f}pp win rate, {sharpe_improvement_pct:+.1f}% Sharpe")

    if win_rate_improvement >= 5 and sharpe_improvement_pct >= 15:
        print("SUCCESS: Targets achieved!")
        print(f"\nRECOMMENDATION: Deploy {best_strategy.replace('_', ' ').title()} to production")
    elif win_rate_improvement >= 5 or sharpe_improvement_pct >= 15:
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

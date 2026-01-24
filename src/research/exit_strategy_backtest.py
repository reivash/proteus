"""
Backtest the ExitOptimizer strategies against baseline (2-day hold).

This validates whether the tier-based exit strategies actually improve returns.
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trading.exit_optimizer import ExitOptimizer, StockTier


TICKERS = [
    'NVDA', 'AVGO', 'MSFT', 'ORCL', 'INTU', 'ADBE', 'CRM', 'NOW', 'AMAT', 'KLAC',
    'MRVL', 'QCOM', 'TXN', 'ADI', 'JPM', 'MS', 'SCHW', 'AXP', 'AIG', 'USB',
    'PNC', 'V', 'MA', 'ABBV', 'SYK', 'GILD', 'PFE', 'JNJ', 'CVS', 'HCA',
    'IDXX', 'INSM', 'COP', 'SLB', 'XOM', 'MPC', 'EOG', 'CAT', 'ETN', 'LMT',
    'ROAD', 'HD', 'LOW', 'TGT', 'WMT', 'TMUS', 'CMCSA', 'META', 'APD', 'ECL',
    'MLM', 'SHW', 'EXR', 'NEE'
]


def calculate_rsi(prices, period=14):
    """Calculate RSI."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


@dataclass
class BacktestTrade:
    """A single backtest trade result."""
    ticker: str
    entry_date: str
    entry_price: float
    signal_strength: float
    tier: str

    # Baseline results (2d hold)
    baseline_return: float

    # Optimized exit results
    optimized_return: float
    exit_reason: str
    exit_day: int


def fetch_stock_data(days_back: int = 400) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for all tickers."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    stock_data = {}
    print(f"Fetching data for {len(TICKERS)} stocks...")

    for i, ticker in enumerate(TICKERS):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if len(df) >= 50:
                stock_data[ticker] = df
        except:
            pass

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(TICKERS)} fetched")

    print(f"  Got data for {len(stock_data)} stocks")
    return stock_data


def simulate_optimized_exit(optimizer: ExitOptimizer,
                            ticker: str,
                            entry_price: float,
                            signal_strength: float,
                            daily_data: List[Tuple[float, float, float]]) -> Tuple[float, str, int]:
    """
    Simulate the optimized exit strategy on historical daily data.

    Args:
        daily_data: List of (close, high, low) for each day after entry

    Returns:
        (total_return, exit_reason, exit_day)
    """
    if not daily_data:
        return 0.0, "NO DATA", 0

    portion_remaining = 1.0
    total_return = 0.0
    high_since_entry = entry_price
    exit_reason = ""
    exit_day = 0

    for day, (close, high, low) in enumerate(daily_data, 1):
        high_since_entry = max(high_since_entry, high)

        should_exit, exit_portion, reason = optimizer.should_exit(
            ticker=ticker,
            entry_price=entry_price,
            current_price=close,
            high_since_entry=high_since_entry,
            days_held=day,
            portion_remaining=portion_remaining,
            signal_strength=signal_strength
        )

        if should_exit:
            current_return = ((close / entry_price) - 1) * 100
            total_return += exit_portion * current_return
            portion_remaining -= exit_portion
            exit_reason = reason
            exit_day = day

            if portion_remaining <= 0.001:
                break

    # If position still open after 5 days, exit at day 5 close
    if portion_remaining > 0.001 and len(daily_data) >= 5:
        close_5d = daily_data[4][0]
        final_return = ((close_5d / entry_price) - 1) * 100
        total_return += portion_remaining * final_return
        exit_reason = "MAX HOLD (5d)"
        exit_day = 5

    return total_return, exit_reason, exit_day


def run_backtest(stock_data: Dict[str, pd.DataFrame],
                 optimizer: ExitOptimizer,
                 days_back: int = 365) -> List[BacktestTrade]:
    """Run backtest comparing baseline to optimized exits."""

    results = []
    cutoff_date = datetime.now() - timedelta(days=days_back)
    cutoff_str = cutoff_date.strftime('%Y-%m-%d')

    print(f"\nRunning backtest from {cutoff_str}...")

    for ticker, df in stock_data.items():
        try:
            df = df.copy()
            df['RSI'] = calculate_rsi(df['Close'])

            for i in range(20, len(df) - 6):
                if df['RSI'].iloc[i] < 35:
                    signal_date = df.index[i]
                    signal_str = signal_date.strftime('%Y-%m-%d')

                    if signal_str < cutoff_str:
                        continue

                    entry_price = df['Close'].iloc[i]
                    rsi = df['RSI'].iloc[i]
                    signal_strength = max(0, min(100, (35 - rsi) * 3 + 50))

                    # Get tier
                    tier = optimizer.get_stock_tier(ticker)

                    # Baseline: 2-day hold
                    baseline_return = ((df['Close'].iloc[i+2] / entry_price) - 1) * 100

                    # Optimized exit: simulate 5 days
                    daily_data = []
                    for d in range(1, 6):
                        if i + d < len(df):
                            daily_data.append((
                                df['Close'].iloc[i+d],
                                df['High'].iloc[i+d],
                                df['Low'].iloc[i+d]
                            ))

                    opt_return, exit_reason, exit_day = simulate_optimized_exit(
                        optimizer, ticker, entry_price, signal_strength, daily_data
                    )

                    results.append(BacktestTrade(
                        ticker=ticker,
                        entry_date=signal_str,
                        entry_price=entry_price,
                        signal_strength=signal_strength,
                        tier=tier.value,
                        baseline_return=baseline_return,
                        optimized_return=opt_return,
                        exit_reason=exit_reason,
                        exit_day=exit_day
                    ))

        except Exception as e:
            continue

    print(f"  Completed {len(results)} trades")
    return results


def analyze_results(trades: List[BacktestTrade]) -> Dict:
    """Analyze backtest results."""

    print("\n" + "=" * 70)
    print("EXIT STRATEGY BACKTEST RESULTS")
    print("=" * 70)

    if not trades:
        print("No trades to analyze!")
        return {}

    # Overall results
    baseline_returns = [t.baseline_return for t in trades]
    optimized_returns = [t.optimized_return for t in trades]

    print(f"\nOVERALL COMPARISON ({len(trades)} trades):")
    print("-" * 50)
    print(f"  Baseline (2d hold):")
    print(f"    Avg return: {np.mean(baseline_returns):+.3f}%")
    print(f"    Win rate: {sum(1 for r in baseline_returns if r > 0)/len(baseline_returns)*100:.1f}%")
    print(f"    Std dev: {np.std(baseline_returns):.3f}%")

    print(f"\n  Optimized Exit:")
    print(f"    Avg return: {np.mean(optimized_returns):+.3f}%")
    print(f"    Win rate: {sum(1 for r in optimized_returns if r > 0)/len(optimized_returns)*100:.1f}%")
    print(f"    Std dev: {np.std(optimized_returns):.3f}%")

    improvement = np.mean(optimized_returns) - np.mean(baseline_returns)
    print(f"\n  >>> IMPROVEMENT: {improvement:+.3f}% per trade <<<")

    # By tier
    print("\n\nBY STOCK TIER:")
    print("-" * 50)

    tier_results = {}
    for tier in ['elite', 'strong', 'average', 'avoid']:
        tier_trades = [t for t in trades if t.tier == tier]
        if not tier_trades:
            continue

        baseline = [t.baseline_return for t in tier_trades]
        optimized = [t.optimized_return for t in tier_trades]

        tier_results[tier] = {
            'count': len(tier_trades),
            'baseline_return': np.mean(baseline),
            'optimized_return': np.mean(optimized),
            'improvement': np.mean(optimized) - np.mean(baseline)
        }

        print(f"\n  {tier.upper()} ({len(tier_trades)} trades):")
        print(f"    Baseline: {np.mean(baseline):+.3f}%")
        print(f"    Optimized: {np.mean(optimized):+.3f}%")
        print(f"    Improvement: {tier_results[tier]['improvement']:+.3f}%")

    # Exit reason breakdown
    print("\n\nEXIT REASONS:")
    print("-" * 50)

    exit_reasons = {}
    for t in trades:
        reason = t.exit_reason.split(' at ')[0]  # Get reason type
        if reason not in exit_reasons:
            exit_reasons[reason] = {'count': 0, 'returns': []}
        exit_reasons[reason]['count'] += 1
        exit_reasons[reason]['returns'].append(t.optimized_return)

    for reason, data in sorted(exit_reasons.items(), key=lambda x: -x[1]['count']):
        print(f"  {reason}: {data['count']} trades, avg return: {np.mean(data['returns']):+.3f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total trades: {len(trades)}")
    print(f"Baseline avg return: {np.mean(baseline_returns):+.3f}%")
    print(f"Optimized avg return: {np.mean(optimized_returns):+.3f}%")
    print(f"Improvement: {improvement:+.3f}%")

    if improvement > 0:
        print("\n>>> OPTIMIZED EXIT STRATEGY IS BETTER <<<")
    else:
        print("\n>>> WARNING: Baseline outperforms optimized strategy <<<")
        print("    Consider adjusting parameters")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'total_trades': len(trades),
        'baseline_avg_return': round(np.mean(baseline_returns), 3),
        'optimized_avg_return': round(np.mean(optimized_returns), 3),
        'improvement': round(improvement, 3),
        'by_tier': tier_results
    }

    output_path = Path('data/research/exit_strategy_backtest.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return output


def main():
    """Run the exit strategy backtest."""
    print("=" * 70)
    print("EXIT STRATEGY BACKTEST")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Initialize optimizer
    optimizer = ExitOptimizer()

    # Fetch data
    stock_data = fetch_stock_data(days_back=400)

    # Run backtest
    trades = run_backtest(stock_data, optimizer, days_back=365)

    # Analyze results
    results = analyze_results(trades)

    return results


if __name__ == '__main__':
    main()

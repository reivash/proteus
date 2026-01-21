"""
Test mean reversion strategy on multiple stocks.
Individual stocks are more volatile than indices and may show better mean reversion.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from exp008_mean_reversion import run_mean_reversion_experiment

# Test on multiple stocks
stocks = [
    ("NVDA", "NVIDIA"),
    ("AAPL", "Apple"),
    ("TSLA", "Tesla"),
    ("MSFT", "Microsoft"),
    ("GOOGL", "Google"),
    ("META", "Meta"),
    ("AMZN", "Amazon")
]

print("=" * 70)
print(" " * 15 + "MULTI-STOCK MEAN REVERSION TEST")
print("=" * 70)
print("\nTesting mean reversion on 7 volatile tech stocks...")
print("Hypothesis: Individual stocks show more overcorrections than indices")
print()

results_summary = []

for ticker, name in stocks:
    print(f"\n{'=' * 70}")
    print(f"Testing: {name} ({ticker})")
    print(f"{'=' * 70}\n")

    result = run_mean_reversion_experiment(ticker, name)

    if result:
        results_summary.append({
            'ticker': ticker,
            'name': name,
            'trades': result['backtest_results']['total_trades'],
            'win_rate': result['backtest_results']['win_rate'],
            'total_return': result['backtest_results']['total_return'],
            'sharpe': result['backtest_results']['sharpe_ratio']
        })
    else:
        results_summary.append({
            'ticker': ticker,
            'name': name,
            'trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'sharpe': 0
        })

# Print summary
print("\n" + "=" * 70)
print("SUMMARY ACROSS ALL STOCKS:")
print("=" * 70)
print(f"{'Stock':<12} {'Trades':>8} {'Win Rate':>10} {'Return':>10} {'Sharpe':>10}")
print("-" * 70)

for r in results_summary:
    print(f"{r['ticker']:<12} {r['trades']:>8} {r['win_rate']:>9.1f}% {r['total_return']:>9.2f}% {r['sharpe']:>10.2f}")

# Overall stats
total_trades = sum(r['trades'] for r in results_summary)
avg_win_rate = sum(r['win_rate'] for r in results_summary if r['trades'] > 0) / len([r for r in results_summary if r['trades'] > 0]) if any(r['trades'] > 0 for r in results_summary) else 0
avg_return = sum(r['total_return'] for r in results_summary) / len(results_summary)

print("-" * 70)
print(f"{'AVERAGE':<12} {total_trades/len(results_summary):>8.1f} {avg_win_rate:>9.1f}% {avg_return:>9.2f}%")
print("=" * 70)

if avg_win_rate >= 60:
    print("\n[SUCCESS] Mean reversion shows promise on volatile stocks!")
elif avg_win_rate >= 50:
    print("\n[MIXED] Some stocks show mean reversion, needs refinement.")
else:
    print("\n[CAUTION] Mean reversion not strongly evident in this data.")

"""
EXPERIMENT: EXP-021
Date: 2025-11-16
Objective: Expand Tier A stock universe from 3 to 8-10 stocks

RESEARCH MOTIVATION:
EXP-014 was the biggest win: +15.99pp improvement from stock selection.
Currently monitoring only 3 Tier A stocks (NVDA, V, MA).

More high-quality stocks = More trading opportunities = Higher returns.

Current state:
- 3 Tier A stocks × ~5 signals/year = ~15 trades/year
- Target: 8-10 Tier A stocks × ~5 signals/year = ~40-50 trades/year
- Expected improvement: +5-10pp from increased opportunities

HYPOTHESIS:
Testing 20 additional S&P 100 stocks will reveal 3-5 new Tier A candidates.
EXP-014 showed 27% hit rate (3 Tier A in 11 tested).
Expected: 20 stocks × 27% = 5-6 new Tier A stocks.

SYMBOLS TO TEST (20 stocks):
Finance: WFC, C, MS, AXP (JPM was 55.6% - borderline)
Tech: MSFT, GOOGL, META, AAPL (AAPL was 50%, retest)
Semiconductors: AMD, INTC, MU, QCOM, AVGO
Payments: PYPL, SQ
Consumer: AMZN, COST, WMT
Healthcare: UNH, JNJ

TIER A CRITERIA (>70% win rate):
- Win rate: >70% (CRITICAL)
- Total return: >5% over 3 years (minimum)
- Sharpe ratio: >5.0 (preferred)
- Avg gain: >2% per trade

EXPECTED OUTCOMES:
- Find 3-6 new Tier A stocks (>70% win rate)
- Expand monitoring from 3 to 8-10 stocks
- Increase trade frequency by 2-3x
- Total return improvement: +5-10pp from more opportunities
- Diversification: Reduced single-stock risk
"""

import sys
import os

# Reuse EXP-014 infrastructure
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.experiments.exp014_stock_selection_optimization import (
    run_exp014_stock_selection,
    StockSelectionBacktester
)
import pandas as pd
import json
from datetime import datetime


def run_exp021_expand_tier_a(period: str = "3y"):
    """
    Test 20 additional stocks to find more Tier A candidates.

    Args:
        period: Backtest period
    """
    print("=" * 70)
    print("EXP-021: EXPAND TIER A STOCK UNIVERSE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Find 3-6 new Tier A stocks (>70% win rate)")
    print("Current Tier A: NVDA (87.5%), V (75.0%), MA (71.4%)")
    print()

    # Test 20 new symbols from S&P 100
    new_symbols = [
        # Finance (strong sector historically)
        'WFC',   # Wells Fargo
        'C',     # Citigroup
        'MS',    # Morgan Stanley
        'AXP',   # American Express

        # Tech giants
        'MSFT',  # Microsoft
        'GOOGL', # Google
        'META',  # Meta/Facebook
        'AAPL',  # Apple (retest - was 50% in EXP-014)

        # Semiconductors (NVDA sector)
        'AMD',   # AMD
        'INTC',  # Intel (66.7% in old params)
        'MU',    # Micron
        'QCOM',  # Qualcomm
        'AVGO',  # Broadcom

        # Payments (V/MA sector)
        'PYPL',  # PayPal
        'SQ',    # Block (Square)

        # Consumer
        'AMZN',  # Amazon
        'COST',  # Costco
        'WMT',   # Walmart

        # Healthcare
        'UNH',   # UnitedHealth
        'JNJ',   # Johnson & Johnson
    ]

    print(f"Testing {len(new_symbols)} new symbols:")
    print(f"  Finance: WFC, C, MS, AXP")
    print(f"  Tech: MSFT, GOOGL, META, AAPL")
    print(f"  Semiconductors: AMD, INTC, MU, QCOM, AVGO")
    print(f"  Payments: PYPL, SQ")
    print(f"  Consumer: AMZN, COST, WMT")
    print(f"  Healthcare: UNH, JNJ")
    print()

    # Run stock selection backtest (reuse EXP-014 code)
    results = run_exp014_stock_selection(new_symbols, period=period)

    # Additional analysis: Compare with existing Tier A
    print("\n" + "=" * 70)
    print("COMPARISON WITH EXISTING TIER A")
    print("=" * 70)
    print()

    existing_tier_a = {
        'NVDA': {'win_rate': 87.5, 'return': 49.70, 'sharpe': 19.58},
        'V': {'win_rate': 75.0, 'return': 7.33, 'sharpe': 7.03},
        'MA': {'win_rate': 71.4, 'return': 5.99, 'sharpe': 6.92}
    }

    print("EXISTING TIER A (Confirmed):")
    for ticker, stats in existing_tier_a.items():
        print(f"  [{ticker}] {stats['win_rate']:.1f}% win rate, {stats['return']:+.2f}% return, {stats['sharpe']:.2f} Sharpe")

    print()
    print("NEW TIER A CANDIDATES (from today's test):")
    print("  (See tier classification above)")

    # Portfolio construction
    print()
    print("=" * 70)
    print("RECOMMENDED PORTFOLIO")
    print("=" * 70)
    print()

    print("Based on EXP-021 results:")
    print("1. Keep all existing Tier A: NVDA, V, MA")
    print("2. Add new Tier A candidates (>70% win rate) from today's test")
    print("3. Monitor Tier B candidates (55-70%) for potential promotion")
    print()
    print("Expected impact:")
    print("  - Trade frequency: +2-3x (more opportunities)")
    print("  - Diversification: Reduced single-stock risk")
    print("  - Return improvement: +5-10pp from increased activity")

    print()
    print("=" * 70)

    # Save combined results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    combined_results = {
        'experiment_id': 'EXP-021',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': period,
        'symbols_tested': len(new_symbols),
        'existing_tier_a': existing_tier_a,
        'new_test_results': results,
        'recommendation': 'Update mean_reversion_params.py with new Tier A stocks'
    }

    results_file = os.path.join(results_dir, 'exp021_expand_tier_a.json')
    with open(results_file, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)

    print(f"Results saved to: {results_file}")

    return combined_results


if __name__ == '__main__':
    """Run EXP-021 to expand Tier A stock universe."""

    results = run_exp021_expand_tier_a(period='3y')

    print("\n\nNext steps:")
    print("1. Add new Tier A stocks to mean_reversion_params.py")
    print("2. Update dashboard to monitor expanded universe")
    print("3. Deploy with 8-10 Tier A stocks for increased opportunities")
    print("4. Monitor performance vs 3-stock baseline")

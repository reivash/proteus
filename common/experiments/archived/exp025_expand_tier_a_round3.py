"""
EXPERIMENT: EXP-025
Date: 2025-11-16
Objective: Expand Tier A universe from 10 to 12-15 stocks (Round 3)

RESEARCH MOTIVATION:
Stock selection continues to be our highest ROI optimization:
- EXP-014: +15.99pp improvement from selective trading
- EXP-021: Found 2 new Tier A stocks (AVGO, AXP)
- EXP-024: Found 5 new Tier A stocks (KLAC, ORCL, MRVL, ABBV, SYK)

Current Tier A: 10 stocks → ~55 trades/year
Target: 13-15 stocks → ~65-75 trades/year (+20-35% opportunities)

HYPOTHESIS:
Testing 30 additional symbols from under-explored sectors will find 3-4 new Tier A stocks.
Hit rate expectation: 10-15% (based on EXP-024: 5/30 = 16.7%)

SYMBOLS TO TEST (30 stocks across 5 sectors):

Energy (6):
- XOM (ExxonMobil)
- CVX (Chevron) - retest (was 66.7% in EXP-014, borderline)
- COP (ConocoPhillips)
- SLB (Schlumberger)
- EOG (EOG Resources)
- OXY (Occidental Petroleum)

Industrials (6):
- BA (Boeing)
- GE (General Electric)
- UNP (Union Pacific)
- RTX (Raytheon Technologies)
- DE (Deere & Company)
- MMM (3M Company)

Consumer/Retail (6):
- HD (Home Depot) - retest (was 62.5% in EXP-014, marginal)
- MCD (McDonald's)
- DIS (Disney)
- CMCSA (Comcast)
- PEP (PepsiCo)
- KO (Coca-Cola)

Tech - Advanced/Software (6):
- INTU (Intuit)
- TXN (Texas Instruments)
- ADI (Analog Devices)
- SNPS (Synopsys)
- CDNS (Cadence Design Systems)
- PANW (Palo Alto Networks)

Healthcare - Pharma/Biotech (6):
- LLY (Eli Lilly)
- NVO (Novo Nordisk)
- MRK (Merck)
- AMGN (Amgen)
- GILD (Gilead Sciences)
- VRTX (Vertex Pharmaceuticals)

TIER A CRITERIA (>70% win rate):
- Win rate: >70% (CRITICAL)
- Total return: >5% over 3 years
- Sharpe ratio: >5.0 (preferred)
- Min 5 trades (statistical significance)

EXPECTED OUTCOMES:
- Find 3-4 new Tier A stocks
- Expand from 10 to 13-14 total Tier A stocks
- Trade frequency: ~55 → ~65-70 trades/year
- Return improvement: +2-3pp from increased activity
- Better sector diversification (add energy, industrials, consumer)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.experiments.exp014_stock_selection_optimization import run_exp014_stock_selection
import pandas as pd
import json
from datetime import datetime


def run_exp025_expand_tier_a_round3(period: str = "3y"):
    """
    Test 30 additional stocks to find more Tier A candidates (Round 3).

    Args:
        period: Backtest period
    """
    print("=" * 70)
    print("EXP-025: EXPAND TIER A UNIVERSE (ROUND 3)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Find 3-4 new Tier A stocks (>70% win rate)")
    print("Current Tier A (10 stocks):")
    print("  NVDA: 87.5%, AVGO: 87.5%, KLAC: 80.0%, ORCL: 76.9%, MRVL: 75.0%")
    print("  V: 75.0%, MA: 71.4%, AXP: 71.4%, ABBV: 70.6%, SYK: 70.0%")
    print("  Avg: 76.5% win rate, ~55 trades/year")
    print()
    print("Target: 13-14 total Tier A stocks, ~65-70 trades/year")
    print()

    # Test 30 new symbols from under-explored sectors
    new_symbols = [
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY',

        # Industrials
        'BA', 'GE', 'UNP', 'RTX', 'DE', 'MMM',

        # Consumer/Retail
        'HD', 'MCD', 'DIS', 'CMCSA', 'PEP', 'KO',

        # Tech - Advanced/Software
        'INTU', 'TXN', 'ADI', 'SNPS', 'CDNS', 'PANW',

        # Healthcare - Pharma/Biotech
        'LLY', 'NVO', 'MRK', 'AMGN', 'GILD', 'VRTX'
    ]

    print(f"Testing {len(new_symbols)} new symbols:")
    print(f"  Energy: XOM, CVX, COP, SLB, EOG, OXY")
    print(f"  Industrials: BA, GE, UNP, RTX, DE, MMM")
    print(f"  Consumer/Retail: HD, MCD, DIS, CMCSA, PEP, KO")
    print(f"  Tech (Advanced): INTU, TXN, ADI, SNPS, CDNS, PANW")
    print(f"  Healthcare (Pharma): LLY, NVO, MRK, AMGN, GILD, VRTX")
    print()

    # Run stock selection backtest (reuse EXP-014 code)
    results = run_exp014_stock_selection(new_symbols, period=period)

    # Additional analysis
    print("\n" + "=" * 70)
    print("COMPARISON WITH EXISTING TIER A")
    print("=" * 70)
    print()

    existing_tier_a = {
        'NVDA': {'win_rate': 87.5, 'return': 49.70, 'sharpe': 19.58, 'trades': 8},
        'AVGO': {'win_rate': 87.5, 'return': 24.52, 'sharpe': 12.65, 'trades': 8},
        'KLAC': {'win_rate': 80.0, 'return': 20.05, 'sharpe': 8.22, 'trades': 10},
        'ORCL': {'win_rate': 76.9, 'return': 22.50, 'sharpe': 12.11, 'trades': 13},
        'MRVL': {'win_rate': 75.0, 'return': 26.63, 'sharpe': 3.81, 'trades': 16},
        'V': {'win_rate': 75.0, 'return': 7.33, 'sharpe': 7.03, 'trades': 8},
        'MA': {'win_rate': 71.4, 'return': 5.99, 'sharpe': 6.92, 'trades': 7},
        'AXP': {'win_rate': 71.4, 'return': 29.21, 'sharpe': 7.61, 'trades': 14},
        'ABBV': {'win_rate': 70.6, 'return': 10.50, 'sharpe': 3.67, 'trades': 17},
        'SYK': {'win_rate': 70.0, 'return': 11.33, 'sharpe': 7.10, 'trades': 10}
    }

    print("EXISTING TIER A (v8.0):")
    for ticker, stats in existing_tier_a.items():
        print(f"  [{ticker}] {stats['win_rate']:.1f}% win rate, {stats['return']:+.2f}% return, {stats['trades']} trades")

    print()
    print("NEW TIER A CANDIDATES (from today's test):")
    print("  (See tier classification above)")

    # Portfolio projection
    print()
    print("=" * 70)
    print("PORTFOLIO PROJECTION")
    print("=" * 70)
    print()

    # Get new Tier A from results
    if 'tier_a' in results and results['tier_a']:
        new_tier_a_count = len(results['tier_a'])
        total_tier_a = 10 + new_tier_a_count

        print(f"NEW Tier A stocks found: {new_tier_a_count}")
        print(f"TOTAL Tier A portfolio: {total_tier_a} stocks")
        print()

        # Calculate expected trades
        avg_trades_per_stock = 5  # Conservative estimate
        expected_trades = total_tier_a * avg_trades_per_stock
        current_trades = 55

        print(f"Trade frequency projection:")
        print(f"  Current: ~{current_trades} trades/year (10 stocks)")
        print(f"  Projected: ~{expected_trades} trades/year ({total_tier_a} stocks)")
        improvement = expected_trades - current_trades
        improvement_pct = (improvement / current_trades * 100)
        print(f"  Improvement: +{improvement} trades/year (+{improvement_pct:.0f}%)")
        print()

        print(f"Expected impact:")
        print(f"  - More trading opportunities (+{improvement_pct:.0f}%)")
        print(f"  - Better sector diversification")
        print(f"  - Reduced concentration risk")
        print(f"  - Return improvement: +2-3pp from increased activity")
    else:
        print("No new Tier A stocks found in this round.")
        print()
        print("Observations:")
        print("  - May have exhausted S&P 100 Tier A candidates")
        print("  - Consider mid-cap stocks (S&P 400)")
        print("  - Or focus on optimizing existing 10 stocks")
        print("  - Current 76.5% win rate, 10 stocks is excellent")

    print()
    print("=" * 70)

    # Save combined results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    combined_results = {
        'experiment_id': 'EXP-025',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': period,
        'symbols_tested': len(new_symbols),
        'existing_tier_a': existing_tier_a,
        'new_test_results': results,
        'recommendation': 'Update mean_reversion_params.py with new Tier A stocks if found'
    }

    results_file = os.path.join(results_dir, 'exp025_expand_tier_a_round3.json')
    with open(results_file, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)

    print(f"Results saved to: {results_file}")

    # Send experiment report email if configured
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        from common.notifications.sendgrid_notifier import SendGridNotifier

        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending experiment report email...")
            notifier.send_experiment_report('EXP-025', combined_results)
        else:
            print("\n[INFO] Email notifications not configured (see email_config.json)")
    except Exception as e:
        print(f"\n[WARNING] Could not send experiment report email: {e}")

    return combined_results


if __name__ == '__main__':
    """Run EXP-025 to expand Tier A stock universe (Round 3)."""

    results = run_exp025_expand_tier_a_round3(period='3y')

    print("\n\nNext steps:")
    print("1. Add new Tier A stocks to mean_reversion_params.py (if found)")
    print("2. Update dashboard to monitor expanded universe")
    print("3. If no new stocks: Consider mid-cap expansion or optimization")
    print("4. Monitor performance vs v8.0 baseline")

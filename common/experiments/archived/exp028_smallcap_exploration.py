"""
EXPERIMENT: EXP-028
Date: 2025-11-16
Objective: Test small-cap stocks ($1B-$5B) for Tier A candidates

RESEARCH MOTIVATION:
Mid-cap expansion (EXP-026, EXP-027) delivered mixed results:
- Batch 1: 22% hit rate (excellent)
- Batch 2: 8.3% hit rate (declining)
- Cumulative: 14.3% hit rate across 42 stocks

Hypothesis: Small-caps may offer better mean reversion characteristics.

SMALL-CAP ADVANTAGES:
1. Higher volatility = stronger panic sells
2. Less institutional ownership = more emotional retail selling
3. Less algorithmic trading = market inefficiencies
4. Wider bid-ask spreads create larger overshoots

SMALL-CAP RISKS:
1. Lower liquidity (need >500K avg daily volume)
2. Higher spreads (execution costs)
3. More susceptible to manipulation
4. Higher delisting risk

SELECTION CRITERIA:
- Market cap: $1B-$5B
- CRITICAL: Avg daily volume >500K shares (liquidity threshold)
- Minimum 5 years public (stability)
- Profitability required (avoid speculative/unprofitable)
- Diverse sectors

SYMBOLS TO TEST (20 liquid small-caps):

Healthcare/Biotech:
- INSM (Insmed) - Rare disease treatment
- UTHR (United Therapeutics) - Pulmonary hypertension
- EXAS (Exact Sciences) - Cancer diagnostics
- ARVN (Arvinas) - Protein degradation
- SGEN (Seagen) - Cancer therapeutics

Tech/Software:
- PCTY (Paylocity) - Payroll/HR software
- HUBS (HubSpot) - Marketing automation
- ZS (Zscaler) - Cloud security
- DDOG (Datadog) - Monitoring (may be mid-cap now)
- BILL (Bill.com) - AP/AR automation

Consumer:
- CHEF (The Chefs' Warehouse) - Food distribution
- PLNT (Planet Fitness) - Gyms
- W (Wayfair) - E-commerce furniture
- ETSY (Etsy) - E-commerce marketplace

Industrials:
- ROAD (Construction Partners) - Infrastructure
- MTZ (MasTec) - Infrastructure construction
- PRIM (Primoris Services) - Engineering/construction

Energy/Materials:
- SM (SM Energy) - Oil & gas E&P
- CEIX (CONSOL Energy) - Coal/gas
- CLF (Cleveland-Cliffs) - Steel

TIER A CRITERIA (same as always):
- Win rate: >70%
- Total return: >5% over 3 years
- Sharpe ratio: >5.0
- Min 5 trades

EXPECTED OUTCOMES:
- If hypothesis correct: 15-20% hit rate (3-4 new Tier A stocks)
- If hypothesis wrong: <10% hit rate (abandon small-caps, focus on optimization)
- Liquidity issues may disqualify many candidates
- Higher execution costs may reduce net returns
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.experiments.exp014_stock_selection_optimization import run_exp014_stock_selection
import pandas as pd
import json
from datetime import datetime


def run_exp028_smallcap_exploration(period: str = "3y"):
    """
    Test small-cap stocks ($1B-$5B) for Tier A candidates.

    Args:
        period: Backtest period
    """
    print("=" * 70)
    print("EXP-028: SMALL-CAP EXPLORATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Test if small-caps have better mean reversion than mid-caps")
    print("Current portfolio (v11.0): 20 Tier A stocks")
    print("Mid-cap hit rate: 14.3% (6 found from 42 tested)")
    print()
    print("Hypothesis: Higher volatility + less institutional ownership = stronger signals")
    print()

    # Test 20 liquid small-cap stocks
    smallcap_symbols = [
        # Healthcare/Biotech
        'INSM', 'UTHR', 'EXAS', 'ARVN', 'SGEN',

        # Tech/Software
        'PCTY', 'HUBS', 'ZS', 'DDOG', 'BILL',

        # Consumer
        'CHEF', 'PLNT', 'W', 'ETSY',

        # Industrials
        'ROAD', 'MTZ', 'PRIM',

        # Energy/Materials
        'SM', 'CEIX', 'CLF'
    ]

    print(f"Testing {len(smallcap_symbols)} small-cap symbols:")
    print(f"  Healthcare: INSM, UTHR, EXAS, ARVN, SGEN")
    print(f"  Tech: PCTY, HUBS, ZS, DDOG, BILL")
    print(f"  Consumer: CHEF, PLNT, W, ETSY")
    print(f"  Industrials: ROAD, MTZ, PRIM")
    print(f"  Energy/Materials: SM, CEIX, CLF")
    print()
    print("NOTE: Liquidity is CRITICAL - need >500K avg daily volume")
    print()

    # Run stock selection backtest (reuse EXP-014 code)
    results = run_exp014_stock_selection(smallcap_symbols, period=period)

    # Additional analysis
    print("\n" + "=" * 70)
    print("SMALL-CAP VS MID-CAP COMPARISON")
    print("=" * 70)
    print()

    print("MID-CAP RESULTS (EXP-026 + EXP-027):")
    print("  Total tested: 42 stocks")
    print("  Tier A found: 6 stocks")
    print("  Hit rate: 14.3%")
    print("  Best: EXR (85.7% win rate, +18.95% return)")
    print()

    print("SMALL-CAP RESULTS (EXP-028):")
    print("  (See tier classification above)")
    print()

    # Portfolio projection
    print("=" * 70)
    print("PORTFOLIO PROJECTION")
    print("=" * 70)
    print()

    if 'tier_a' in results and results['tier_a']:
        new_tier_a_count = len(results['tier_a'])
        total_tier_a = 20 + new_tier_a_count  # 20 from v11.0

        print(f"NEW Tier A small-caps found: {new_tier_a_count}")
        print(f"TOTAL portfolio (v12.0): {total_tier_a} stocks")
        print()

        expected_trades = total_tier_a * 5
        current_trades = 100  # From v11.0 with 20 stocks

        print(f"Trade frequency projection:")
        print(f"  v11.0: ~{current_trades} trades/year (20 stocks)")
        print(f"  v12.0: ~{expected_trades} trades/year ({total_tier_a} stocks)")
        print(f"  Improvement: +{expected_trades - current_trades} trades/year")
        print()

        hit_rate = (new_tier_a_count / len(smallcap_symbols)) * 100

        print(f"Small-cap hit rate: {hit_rate:.1f}%")
        print()

        if hit_rate >= 15:
            print(f"HYPOTHESIS VALIDATED! Small-caps show good mean reversion.")
            print(f"Recommendation: Continue small-cap testing (Batch 2)")
        elif hit_rate >= 10:
            print(f"HYPOTHESIS PARTIALLY VALIDATED. Decent hit rate.")
            print(f"Recommendation: Update to v12.0, monitor liquidity in production")
        else:
            print(f"HYPOTHESIS REJECTED. Hit rate below mid-caps.")
            print(f"Recommendation: Focus on optimizing existing 20 stocks instead")

        print()
        print(f"Expected impact:")
        print(f"  - More opportunities (+{((expected_trades - current_trades) / current_trades * 100):.0f}%)")
        print(f"  - Small-cap volatility exposure")
        print(f"  - WARNING: Watch for liquidity issues in production")
    else:
        print("No Tier A small-caps found.")
        print()
        print("Observations:")
        print("  - Small-caps may be too illiquid for our strategy")
        print("  - Volatility may be noise rather than signal")
        print("  - Execution costs may eat into returns")
        print("  - Focus on mid-cap and large-cap optimization instead")
        print()
        print("RECOMMENDATION: Abandon small-cap expansion.")
        print("Next steps:")
        print("  1. Parameter optimization for existing 20 stocks")
        print("  2. Portfolio management (position sizing, correlation)")
        print("  3. Signal enhancement (earnings filter, regime detection)")

    print()
    print("=" * 70)

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    combined_results = {
        'experiment_id': 'EXP-028',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': period,
        'symbols_tested': len(smallcap_symbols),
        'market_cap_range': '$1B-$5B (small-cap)',
        'test_results': results,
        'recommendation': 'Update to v12.0 if hit rate >10%, otherwise optimize existing portfolio'
    }

    results_file = os.path.join(results_dir, 'exp028_smallcap_exploration.json')
    with open(results_file, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)

    print(f"Results saved to: {results_file}")

    # Send email report
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending experiment report email...")
            notifier.send_experiment_report('EXP-028', combined_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return combined_results


if __name__ == '__main__':
    """Run EXP-028 small-cap exploration."""

    results = run_exp028_smallcap_exploration(period='3y')

    print("\n\nNext steps:")
    print("1. If successful: Update to v12.0 with new small-caps")
    print("2. If unsuccessful: Focus on parameter optimization")
    print("3. Monitor liquidity closely in production")

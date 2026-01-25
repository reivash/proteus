"""
EXPERIMENT: EXP-026
Date: 2025-11-16
Objective: Test mid-cap stocks (S&P 400) for Tier A candidates

RESEARCH MOTIVATION:
We've tested ~90 large-cap stocks (S&P 100) across 3 experiments.
Found 14 Tier A stocks (hit rate: ~15-16%).

Mid-cap stocks may have:
- Higher volatility = stronger panic sells
- Better mean reversion (less institutional ownership)
- Underexplored opportunity set

HYPOTHESIS:
Testing 20 liquid mid-cap stocks will find 2-4 new Tier A candidates.
Expected hit rate: 10-20% (similar to large-caps).

Mid-caps selected criteria:
- Market cap $5B-$20B
- High liquidity (avg volume >1M shares/day)
- Established companies (>10 years public)
- Diverse sectors

SYMBOLS TO TEST (20 liquid mid-caps):

Tech/Software:
- SNPS (Synopsys) - retest, was 57.9% in EXP-025
- CDNS (Cadence) - retest, was 36.4% in EXP-025
- FTNT (Fortinet)
- DDOG (Datadog)

Healthcare:
- IDXX (IDEXX Laboratories)
- ALGN (Align Technology)
- DXCM (DexCom)
- PODD (Insulet)

Finance:
- FITB (Fifth Third Bancorp)
- HBAN (Huntington Bancshares)
- KEY (KeyCorp)
- RF (Regions Financial)

Industrials/Materials:
- FAST (Fastenal)
- POOL (Pool Corporation)
- VMC (Vulcan Materials)
- MLM (Martin Marietta)

Consumer:
- ULTA (Ulta Beauty)
- LULU (Lululemon)
- ORLY (O'Reilly Automotive)
- AZO (AutoZone)

TIER A CRITERIA:
- Win rate: >70%
- Total return: >5% over 3 years
- Sharpe ratio: >5.0
- Min 5 trades

EXPECTED OUTCOMES:
- Find 2-4 new Tier A stocks
- Expand from 14 to 16-18 total
- Trade frequency: ~70 → ~80-90 trades/year
- Better volatility exposure (mid-caps more volatile)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.experiments.exp014_stock_selection_optimization import run_exp014_stock_selection
import pandas as pd
import json
from datetime import datetime


def run_exp026_midcap_expansion(period: str = "3y"):
    """
    Test 20 mid-cap stocks for Tier A candidates.

    Args:
        period: Backtest period
    """
    print("=" * 70)
    print("EXP-026: MID-CAP EXPANSION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Find 2-4 new Tier A stocks from S&P 400 mid-caps")
    print("Current Tier A (14 large-cap stocks): 77.6% avg win rate")
    print()
    print("Hypothesis: Mid-caps have stronger volatility = better mean reversion")
    print()

    # Test 20 liquid mid-cap stocks
    midcap_symbols = [
        # Tech/Software
        'FTNT', 'DDOG',

        # Healthcare
        'IDXX', 'ALGN', 'DXCM', 'PODD',

        # Finance
        'FITB', 'HBAN', 'KEY', 'RF',

        # Industrials/Materials
        'FAST', 'POOL', 'VMC', 'MLM',

        # Consumer
        'ULTA', 'LULU', 'ORLY', 'AZO'
    ]

    print(f"Testing {len(midcap_symbols)} mid-cap symbols:")
    print(f"  Tech: FTNT, DDOG")
    print(f"  Healthcare: IDXX, ALGN, DXCM, PODD")
    print(f"  Finance: FITB, HBAN, KEY, RF")
    print(f"  Industrials: FAST, POOL, VMC, MLM")
    print(f"  Consumer: ULTA, LULU, ORLY, AZO")
    print()

    # Run stock selection backtest (reuse EXP-014 code)
    results = run_exp014_stock_selection(midcap_symbols, period=period)

    # Additional analysis
    print("\n" + "=" * 70)
    print("MID-CAP VS LARGE-CAP COMPARISON")
    print("=" * 70)
    print()

    print("LARGE-CAP TIER A (v9.0 - 14 stocks):")
    print("  Average win rate: 77.6%")
    print("  Average return: +22.07%")
    print("  Trade frequency: ~70/year")
    print()

    print("MID-CAP RESULTS:")
    print("  (See tier classification above)")
    print()

    # Portfolio projection
    print("=" * 70)
    print("PORTFOLIO PROJECTION")
    print("=" * 70)
    print()

    if 'tier_a' in results and results['tier_a']:
        new_tier_a_count = len(results['tier_a'])
        total_tier_a = 14 + new_tier_a_count

        print(f"NEW Tier A mid-caps found: {new_tier_a_count}")
        print(f"TOTAL Tier A portfolio: {total_tier_a} stocks")
        print()

        expected_trades = total_tier_a * 5
        current_trades = 70

        print(f"Trade frequency projection:")
        print(f"  Current: ~{current_trades} trades/year (14 stocks)")
        print(f"  Projected: ~{expected_trades} trades/year ({total_tier_a} stocks)")
        print(f"  Improvement: +{expected_trades - current_trades} trades/year")
        print()

        print(f"Expected impact:")
        print(f"  - More opportunities (+{((expected_trades - current_trades) / current_trades * 100):.0f}%)")
        print(f"  - Mid-cap volatility exposure")
        print(f"  - Potential for higher returns (mid-caps more volatile)")
    else:
        print("No new Tier A mid-caps found.")
        print()
        print("Observations:")
        print("  - Mid-caps may not have better mean reversion than large-caps")
        print("  - Liquidity differences might affect signal quality")
        print("  - Focus on optimizing existing 14 Tier A stocks")

    print()
    print("=" * 70)

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    combined_results = {
        'experiment_id': 'EXP-026',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': period,
        'symbols_tested': len(midcap_symbols),
        'market_cap_range': '$5B-$20B (S&P 400 mid-caps)',
        'test_results': results,
        'recommendation': 'Add new Tier A mid-caps if found, otherwise focus on optimization'
    }

    results_file = os.path.join(results_dir, 'exp026_midcap_expansion.json')
    with open(results_file, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)

    print(f"Results saved to: {results_file}")

    # Send email report
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending experiment report email...")
            notifier.send_experiment_report('EXP-026', combined_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return combined_results


if __name__ == '__main__':
    """Run EXP-026 mid-cap expansion."""

    results = run_exp026_midcap_expansion(period='3y')

    print("\n\nNext steps:")
    print("1. Add new Tier A mid-caps to configuration (if found)")
    print("2. Compare mid-cap vs large-cap performance")
    print("3. Deploy if improvement justifies complexity")

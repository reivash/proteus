"""
EXPERIMENT: EXP-027
Date: 2025-11-16
Objective: Test second batch of mid-cap stocks for additional Tier A candidates

RESEARCH MOTIVATION:
EXP-026 was highly successful:
- Tested 18 mid-caps → found 4 Tier A stocks
- Hit rate: 22% (BETTER than large-caps at 13-16%)
- FTNT delivers +48.74% return (rivals NVDA!)
- Hypothesis VALIDATED: Mid-caps have stronger mean reversion

CONTINUATION STRATEGY:
Given 22% hit rate, testing 25 more mid-caps should find 5-6 new Tier A stocks.
This would bring portfolio from 18 → 23-24 stocks (~115-120 trades/year).

Mid-cap selection criteria (same as EXP-026):
- Market cap $5B-$20B
- High liquidity (avg volume >1M shares/day)
- Established companies (>5 years public)
- Diverse sectors

SYMBOLS TO TEST (25 liquid mid-caps - Batch 2):

Tech/Software:
- SNPS (Synopsys) - EDA software, was 57.9% in EXP-025 but worth retest
- CDNS (Cadence) - EDA software, was 36.4% in EXP-025 but worth retest
- ANSS (Ansys) - Engineering simulation
- TYL (Tyler Technologies) - Government software
- GDDY (GoDaddy) - Web services

Healthcare/Biotech:
- TECH (Bio-Techne) - Life sciences tools
- VRTX (Vertex Pharma) - CF/gene therapy (may be large-cap now, worth testing)
- HOLX (Hologic) - Medical devices
- ZBRA (Zebra Technologies) - Healthcare tech

Finance:
- NTRS (Northern Trust) - Wealth management
- CFG (Citizens Financial) - Regional bank
- SIVB (SVB Financial) - Note: May be delisted, will handle error
- CMA (Comerica) - Regional bank

Industrials/Materials/Energy:
- LDOS (Leidos) - Defense/IT
- J (Jacobs Engineering) - Engineering consulting
- SWK (Stanley Black & Decker) - Tools (may be large-cap)
- EME (EMCOR Group) - Construction
- FLS (Flowserve) - Industrial equipment

Consumer:
- DPZ (Domino's Pizza) - Food delivery/retail
- YUM (Yum! Brands) - Fast food (may be large-cap)
- BURL (Burlington Stores) - Discount retail
- FIVE (Five Below) - Discount retail

REITs/Real Estate:
- EXR (Extra Space Storage) - Self-storage REIT
- INVH (Invitation Homes) - Residential REIT
- AVB (AvalonBay) - Residential REIT (may be large-cap)

TIER A CRITERIA (same as always):
- Win rate: >70%
- Total return: >5% over 3 years
- Sharpe ratio: >5.0
- Min 5 trades

EXPECTED OUTCOMES:
- Find 5-6 new Tier A stocks (22% hit rate)
- Expand from 18 to 23-24 total
- Trade frequency: ~90 → ~115-120 trades/year
- Better sector diversification (REITs, more healthcare)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.experiments.exp014_stock_selection_optimization import run_exp014_stock_selection
import pandas as pd
import json
from datetime import datetime


def run_exp027_midcap_batch2(period: str = "3y"):
    """
    Test second batch of 25 mid-cap stocks for Tier A candidates.

    Args:
        period: Backtest period
    """
    print("=" * 70)
    print("EXP-027: MID-CAP EXPANSION - BATCH 2")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Find 5-6 new Tier A stocks from second mid-cap batch")
    print("Current portfolio (v10.0): 18 Tier A stocks (14 large + 4 mid)")
    print("EXP-026 hit rate: 22% (4 found from 18 tested)")
    print()
    print("Hypothesis: Continue high mid-cap hit rate (20-25%)")
    print()

    # Test 25 liquid mid-cap stocks (Batch 2)
    midcap_batch2_symbols = [
        # Tech/Software
        'SNPS', 'CDNS', 'ANSS', 'TYL', 'GDDY',

        # Healthcare/Biotech
        'TECH', 'VRTX', 'HOLX', 'ZBRA',

        # Finance
        'NTRS', 'CFG', 'CMA',

        # Industrials/Materials/Energy
        'LDOS', 'J', 'SWK', 'EME', 'FLS',

        # Consumer
        'DPZ', 'YUM', 'BURL', 'FIVE',

        # REITs/Real Estate
        'EXR', 'INVH', 'AVB'
    ]

    print(f"Testing {len(midcap_batch2_symbols)} mid-cap symbols (Batch 2):")
    print(f"  Tech: SNPS, CDNS, ANSS, TYL, GDDY")
    print(f"  Healthcare: TECH, VRTX, HOLX, ZBRA")
    print(f"  Finance: NTRS, CFG, CMA")
    print(f"  Industrials: LDOS, J, SWK, EME, FLS")
    print(f"  Consumer: DPZ, YUM, BURL, FIVE")
    print(f"  REITs: EXR, INVH, AVB")
    print()

    # Run stock selection backtest (reuse EXP-014 code)
    results = run_exp014_stock_selection(midcap_batch2_symbols, period=period)

    # Additional analysis
    print("\n" + "=" * 70)
    print("CUMULATIVE MID-CAP RESULTS")
    print("=" * 70)
    print()

    print("BATCH 1 (EXP-026 - 18 stocks):")
    print("  Tier A found: 4 stocks (FTNT, MLM, IDXX, DXCM)")
    print("  Hit rate: 22.2%")
    print("  Best performer: FTNT (+48.74% return, 77.8% win rate)")
    print()

    print("BATCH 2 (EXP-027 - 24 stocks):")
    print("  (See tier classification above)")
    print()

    # Portfolio projection
    print("=" * 70)
    print("PORTFOLIO PROJECTION")
    print("=" * 70)
    print()

    if 'tier_a' in results and results['tier_a']:
        new_tier_a_count = len(results['tier_a'])
        total_tier_a = 18 + new_tier_a_count  # 18 from v10.0
        batch1_tier_a = 4  # From EXP-026
        cumulative_midcap_tier_a = batch1_tier_a + new_tier_a_count

        print(f"NEW Tier A from Batch 2: {new_tier_a_count}")
        print(f"TOTAL mid-cap Tier A: {cumulative_midcap_tier_a} stocks (Batch 1 + Batch 2)")
        print(f"TOTAL portfolio (v11.0): {total_tier_a} stocks")
        print()

        expected_trades = total_tier_a * 5
        current_trades = 90  # From v10.0 with 18 stocks

        print(f"Trade frequency projection:")
        print(f"  v10.0: ~{current_trades} trades/year (18 stocks)")
        print(f"  v11.0: ~{expected_trades} trades/year ({total_tier_a} stocks)")
        print(f"  Improvement: +{expected_trades - current_trades} trades/year")
        print()

        cumulative_tested = 18 + len(midcap_batch2_symbols)
        cumulative_hit_rate = (cumulative_midcap_tier_a / cumulative_tested) * 100

        print(f"Cumulative mid-cap statistics:")
        print(f"  Total tested: {cumulative_tested} stocks")
        print(f"  Total Tier A: {cumulative_midcap_tier_a} stocks")
        print(f"  Hit rate: {cumulative_hit_rate:.1f}%")
        print()

        print(f"Expected impact:")
        print(f"  - More opportunities (+{((expected_trades - current_trades) / current_trades * 100):.0f}%)")
        print(f"  - Sector diversification (REITs, more healthcare)")
        print(f"  - Mid-cap volatility exposure continues to deliver")
    else:
        print("No new Tier A mid-caps found in Batch 2.")
        print()
        print("Observations:")
        print("  - Batch 1 was exceptional (22% hit rate)")
        print("  - Batch 2 stocks may be lower quality or less liquid")
        print("  - Current 18 stocks may be sufficient")
        print("  - Consider optimization over expansion")

    print()
    print("=" * 70)

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    combined_results = {
        'experiment_id': 'EXP-027',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': period,
        'symbols_tested': len(midcap_batch2_symbols),
        'batch_number': 2,
        'market_cap_range': '$5B-$20B (S&P 400 mid-caps)',
        'test_results': results,
        'cumulative_stats': {
            'batch1_tested': 18,
            'batch2_tested': len(midcap_batch2_symbols),
            'batch1_tier_a': 4,
            'batch2_tier_a': len(results.get('tier_a', [])),
        },
        'recommendation': 'Update to v11.0 if new Tier A stocks found'
    }

    results_file = os.path.join(results_dir, 'exp027_midcap_batch2.json')
    with open(results_file, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)

    print(f"Results saved to: {results_file}")

    # Send email report
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending experiment report email...")
            notifier.send_experiment_report('EXP-027', combined_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return combined_results


if __name__ == '__main__':
    """Run EXP-027 mid-cap batch 2."""

    results = run_exp027_midcap_batch2(period='3y')

    print("\n\nNext steps:")
    print("1. Add new Tier A mid-caps to configuration (if found)")
    print("2. Update to v11.0")
    print("3. Consider Batch 3 if hit rate remains high (>20%)")

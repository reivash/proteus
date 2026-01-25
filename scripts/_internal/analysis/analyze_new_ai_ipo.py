"""
Quick Analysis of Newly Discovered AI/ML IPOs

Analyzes recently found AI/ML companies to add to watchlist.

Usage:
    python analyze_new_ai_ipo.py
"""

import sys
sys.path.insert(0, 'src')

from common.trading.ipo_advanced_screener import AdvancedIPOScreener
from datetime import datetime


def main():
    print("=" * 80)
    print("NEW AI/ML IPO DISCOVERY & ANALYSIS")
    print("=" * 80)
    print()

    # Newly discovered AI/ML IPOs
    new_companies = [
        'TEM',   # Tempus AI (healthcare AI, IPO: June 2024)
        'ARM',   # ARM Holdings (AI chip architecture, IPO: Sept 2023)
    ]

    print("Newly Discovered Companies:")
    print("  TEM - Tempus AI (Healthcare AI, June 2024 IPO)")
    print("  ARM - ARM Holdings (AI chip architecture, Sept 2023 IPO)")
    print()

    # Initialize screener
    print("Initializing screener...")
    screener = AdvancedIPOScreener(
        max_ipo_age_months=24,
        min_yc_score=20.0,
        min_market_cap_b=1.0,
        max_market_cap_b=500.0
    )
    print("[OK] Screener initialized")
    print()

    # Run analysis
    print("Running deep analysis...")
    print()
    results = screener.deep_screen_candidates(new_companies)

    if results.empty:
        print("\n[WARNING] No results - Companies may not have passed filters")
        return

    # Sort by score
    results = results.sort_values('advanced_yc_score', ascending=False)

    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    print()

    for idx, row in results.iterrows():
        print(f"{row['ticker']} - {row['company_name']}")
        print("-" * 80)
        print(f"Advanced YC Score: {row['advanced_yc_score']:.1f}/100")
        print(f"  - Base YC Score: {row['base_yc_score']:.1f}/100")
        print(f"  - Competitive Score: {row['competitive_score']:.1f}/100")
        print(f"  - Trajectory Score: {row['trajectory_score']:.1f}/100")
        print()

        tier = "Exceptional" if row['advanced_yc_score'] >= 75 else \
               "Very Strong" if row['advanced_yc_score'] >= 65 else \
               "Strong" if row['advanced_yc_score'] >= 50 else \
               "Moderate" if row['advanced_yc_score'] >= 35 else "Weak"

        print(f"Tier: {tier}")
        print(f"Category: {row['category']}")
        print(f"IPO: {row['ipo_date'].strftime('%Y-%m-%d')} ({row['months_since_ipo']:.0f} months ago)")
        print(f"Market Cap: ${row['market_cap_b']:.2f}B")
        print()

        print("KEY METRICS:")
        print(f"  Revenue Growth: {(row.get('revenue_growth_yoy', 0) or 0)*100:.1f}% YoY")
        print(f"  Gross Margin: {(row.get('gross_margin', 0) or 0)*100:.1f}%")
        print(f"  Operating Margin: {(row.get('operating_margin', 0) or 0)*100:.1f}%")

        traj = row.get('trajectory', {})
        if traj and traj.get('has_data'):
            print()
            print("PERFORMANCE SINCE IPO:")
            print(f"  Total Return: {traj.get('total_return_pct', 0):+.1f}%")
            print(f"  Annualized Return: {traj.get('annualized_return_pct', 0):+.1f}%")
            print(f"  Scaling Score: {traj.get('scaling_score', 0):.0f}/100")
            print(f"  TAM Expansion: {traj.get('tam_expansion_potential_x', 1):.1f}x")

        print()
        print()

    # Summary
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()

    for _, row in results.iterrows():
        tier = "Exceptional" if row['advanced_yc_score'] >= 75 else \
               "Very Strong" if row['advanced_yc_score'] >= 65 else \
               "Strong" if row['advanced_yc_score'] >= 50 else \
               "Moderate" if row['advanced_yc_score'] >= 35 else "Weak"

        if row['advanced_yc_score'] >= 50:
            print(f"ADD TO WATCHLIST: {row['ticker']} - {tier} ({row['advanced_yc_score']:.1f}/100)")
        elif row['advanced_yc_score'] >= 35:
            print(f"CONSIDER: {row['ticker']} - {tier} ({row['advanced_yc_score']:.1f}/100)")
        else:
            print(f"SKIP: {row['ticker']} - {tier} ({row['advanced_yc_score']:.1f}/100)")

    print()
    print("Next: Update analyze_ai_ml_ipos_expanded.py to include promising companies")
    print("=" * 80)


if __name__ == "__main__":
    main()

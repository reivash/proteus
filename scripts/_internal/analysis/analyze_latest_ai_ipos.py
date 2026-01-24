"""
Analyze Latest AI/ML IPOs (2024-2025)

Quick analysis of the newest AI/ML companies that have gone public.

Usage:
    python analyze_latest_ai_ipos.py
"""

import sys
sys.path.insert(0, 'src')

from src.trading.ipo_advanced_screener import AdvancedIPOScreener
from datetime import datetime


def main():
    print("=" * 80)
    print("LATEST AI/ML IPO ANALYSIS (2024-2025)")
    print("=" * 80)
    print()

    # Latest AI/ML IPOs
    latest_ipos = [
        'CRWV',  # CoreWeave - AI cloud infrastructure (IPO: Mar 2025)
        'ALAB',  # Astera Labs - AI/cloud semiconductor (IPO: Mar 2024)
    ]

    print("Latest AI/ML IPOs:")
    print("  CRWV - CoreWeave (AI cloud infrastructure, IPO: Mar 2025)")
    print("  ALAB - Astera Labs (AI/cloud semiconductor, IPO: Mar 2024)")
    print()

    # Initialize screener with very permissive filters
    print("Initializing screener...")
    screener = AdvancedIPOScreener(
        max_ipo_age_months=24,
        min_yc_score=10.0,
        min_market_cap_b=0.1,
        max_market_cap_b=500.0
    )
    print("[OK] Screener initialized")
    print()

    # Run analysis
    results = screener.deep_screen_candidates(latest_ipos)

    if results.empty:
        print("\n[WARNING] No results - Companies may not have passed filters or data unavailable")
        return

    # Sort by score
    results = results.sort_values('advanced_yc_score', ascending=False)

    print("\n" + "=" * 80)
    print("RESULTS - LATEST AI/ML IPOs")
    print("=" * 80)
    print(f"Companies Analyzed: {len(results)}")
    print()

    # Display results
    for idx, row in results.iterrows():
        print(f"\n{row['ticker']} - {row['company_name']}")
        print("-" * 80)
        print(f"Advanced YC Score: {row['advanced_yc_score']:.1f}/100")
        print(f"  - Base YC Score: {row['base_yc_score']:.1f}/100")
        print(f"  - Competitive Score: {row['competitive_score']:.1f}/100")
        print(f"  - Trajectory Score: {row['trajectory_score']:.1f}/100")
        print()
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
        print()

    # Generate report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"results/latest_ai_ipos_{timestamp}.txt"
    screener.generate_advanced_report(results, filename=report_filename)

    print("=" * 80)
    print(f"Report saved: {report_filename}")
    print("=" * 80)


if __name__ == "__main__":
    main()

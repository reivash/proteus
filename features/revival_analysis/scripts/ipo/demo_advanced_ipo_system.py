"""
Advanced IPO System - Complete Demonstration

Showcases all capabilities:
- Y Combinator scoring
- Competitive positioning
- Growth trajectory analysis
- Advanced composite scoring
- Comprehensive reporting

Usage:
    python demo_advanced_ipo_system.py
"""

import sys
sys.path.insert(0, 'src')

from common.trading.ipo_advanced_screener import AdvancedIPOScreener
from datetime import datetime


def main():
    print("=" * 80)
    print("ADVANCED IPO NICHE DETECTION SYSTEM - DEMO")
    print("Complete Y Combinator-Style Investment Analysis")
    print("=" * 80)
    print()

    # Initialize advanced screener
    print("Initializing advanced screener...")
    screener = AdvancedIPOScreener(
        max_ipo_age_months=36,     # Last 3 years
        min_yc_score=50.0,         # Moderate threshold
        min_market_cap_b=1.0,      # $1B+
        max_market_cap_b=100.0     # <$100B (room to grow)
    )
    print("[OK] Screener initialized\n")

    # Demo candidates (known strong performers)
    demo_candidates = [
        'RDDT',  # Reddit - Social Platform
        'ASND',  # Ascendis Pharma - Biotech
    ]

    print(f"Demo Candidates ({len(demo_candidates)}):")
    for ticker in demo_candidates:
        print(f"  - {ticker}")
    print()

    print("=" * 80)
    print("STARTING DEEP ANALYSIS")
    print("=" * 80)
    print("This will analyze:")
    print("  1. Y Combinator Scoring (niche dominance, emergence, moat, scaling)")
    print("  2. Competitive Positioning (market share, moat, 10x better)")
    print("  3. Growth Trajectory (stock performance, inflection points)")
    print()
    print("Estimated time: 1-2 minutes\n")

    # Run deep screening
    results = screener.deep_screen_candidates(demo_candidates)

    if results.empty:
        print("\n[INFO] No candidates passed filters")
        return

    # Display results summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Candidates Analyzed: {len(results)}")
    print(f"Average Advanced Score: {results['advanced_yc_score'].mean():.1f}/100")
    print(f"Top Candidate: {results['ticker'].iloc[0]} ({results['advanced_yc_score'].iloc[0]:.1f}/100)")
    print()

    # Display detailed table
    print("=" * 80)
    print("DETAILED SCORES")
    print("=" * 80)

    display_cols = [
        'ticker',
        'company_name',
        'advanced_yc_score',
        'base_yc_score',
        'competitive_score',
        'trajectory_score'
    ]

    print(results[display_cols].to_string(index=False))
    print()

    # Investment insights for top candidate
    print("=" * 80)
    print("TOP CANDIDATE DEEP DIVE")
    print("=" * 80)

    top = results.iloc[0]

    print(f"\n{top['ticker']} - {top['company_name']}")
    print("-" * 80)
    print(f"Advanced YC Score: {top['advanced_yc_score']:.1f}/100")
    print(f"Category: {top['category']}")
    print(f"IPO Date: {top['ipo_date'].strftime('%Y-%m-%d')} ({top['months_since_ipo']:.0f}mo ago)")
    print(f"Market Cap: ${top['market_cap_b']:.2f}B")
    print()

    # Competitive position
    comp = top.get('competitive', {})
    if comp.get('has_peers'):
        print("COMPETITIVE ANALYSIS:")
        print(f"  Market Share: {comp.get('market_share_pct', 0):.1f}%")
        print(f"  Rank: #{comp.get('revenue_rank', '?')} of {comp.get('total_peers', '?')}")
        print(f"  Competitive Moat: {comp.get('moat_score', 0):.1f}/100")
        print(f"  10x Better: {comp.get('is_10x_better', False)}")
        if comp.get('10x_signals'):
            print("  Signals:")
            for signal in comp.get('10x_signals', []):
                print(f"    - {signal}")
        print()

    # Growth trajectory
    traj = top.get('trajectory', {})
    if traj.get('has_data'):
        print("GROWTH TRAJECTORY:")
        print(f"  Total Return: {traj.get('total_return_pct', 0):+.1f}%")
        print(f"  Annualized Return: {traj.get('annualized_return_pct', 0):+.1f}%")
        print(f"  ATH Return: {traj.get('ath_return_pct', 0):+.1f}%")
        print(f"  Scaling: {traj.get('is_scaling', False)} ({traj.get('scaling_score', 0):.0f}/100)")
        print(f"  TAM Expansion: {traj.get('tam_expansion_potential_x', 1):.1f}x")
        print(f"  Outpacing Market: {traj.get('outpacing_market_growth', False)}")
        print()

    # Key metrics
    print("KEY FUNDAMENTALS:")
    print(f"  Revenue Growth: {(top.get('revenue_growth_yoy', 0) or 0)*100:.1f}% YoY")
    print(f"  Gross Margin: {(top.get('gross_margin', 0) or 0)*100:.1f}%")
    print(f"  Operating Margin: {(top.get('operating_margin', 0) or 0)*100:.1f}%")
    print()

    # Investment thesis
    print("=" * 80)
    print("INVESTMENT THESIS")
    print("=" * 80)

    thesis = []

    # Niche dominance
    if top.get('is_niche'):
        thesis.append(f"Dominates niche {top['category']} category")

    # Growth
    growth_rate = (top.get('revenue_growth_yoy', 0) or 0) * 100
    if growth_rate > 50:
        thesis.append(f"Explosive revenue growth ({growth_rate:.0f}% YoY)")
    elif growth_rate > 30:
        thesis.append(f"Strong revenue growth ({growth_rate:.0f}% YoY)")

    # Margins
    gross_margin = (top.get('gross_margin', 0) or 0) * 100
    if gross_margin > 70:
        thesis.append(f"Excellent unit economics ({gross_margin:.0f}% gross margin)")

    # Scaling
    if traj.get('is_scaling'):
        thesis.append("Successfully scaling operations")

    # Stock performance
    total_return = traj.get('total_return_pct', 0)
    if total_return > 100:
        thesis.append(f"Strong market validation ({total_return:+.0f}% return since IPO)")

    # TAM expansion
    if traj.get('outpacing_market_growth'):
        thesis.append("Outpacing overall market growth (gaining share)")

    for i, point in enumerate(thesis, 1):
        print(f"{i}. {point}")

    print()

    # Generate full report
    print("=" * 80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("=" * 80)

    report_filename = f"results/demo_advanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    screener.generate_advanced_report(results, filename=report_filename)

    print(f"\nFull report saved: {report_filename}")
    print()

    print("=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review the generated report for full details")
    print("2. Run on full universe: python common/trading/ipo_advanced_screener.py")
    print("3. Customize criteria for your investment strategy")
    print("4. Monitor candidates over time for score changes")
    print()
    print("Happy investing! Find the next Snowflake, Airbnb, or Reddit!")
    print("=" * 80)


if __name__ == "__main__":
    main()

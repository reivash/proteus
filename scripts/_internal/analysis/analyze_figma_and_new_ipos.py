"""
Analyze Figma and Other New 2025 IPOs

Analyzes recently discovered 2025 AI/ML IPO companies.

Usage:
    python analyze_figma_and_new_ipos.py
"""

import sys
sys.path.insert(0, 'src')

from common.trading.ipo_advanced_screener import AdvancedIPOScreener
from datetime import datetime


def main():
    print("=" * 80)
    print("NEW 2025 AI/ML IPO ANALYSIS - FIGMA & MORE")
    print("=" * 80)
    print()

    # New 2025 IPOs to analyze
    new_companies = [
        'FIG',   # Figma (AI-powered design, IPO: July 2025)
    ]

    print("Companies to Analyze:")
    print("  FIG - Figma (AI-powered design platform, July 2025 IPO)")
    print("        - Figma Make: AI generates designs from text")
    print("        - Figma Weave: AI video/animation creation")
    print("        - $1B+ annual revenue run rate")
    print("        - 38-40% YoY growth")
    print("        - Stock down ~50% from peak ($142 -> ~$70)")
    print()

    # Initialize screener with appropriate filters for new IPOs
    print("Initializing screener...")
    screener = AdvancedIPOScreener(
        max_ipo_age_months=12,  # 2025 IPOs only
        min_yc_score=20.0,
        min_market_cap_b=0.5,   # Lower threshold for newer IPOs
        max_market_cap_b=500.0
    )
    print("[OK] Screener initialized")
    print()

    # Run analysis
    print("Running deep analysis...")
    print("- Fetching financial data from Yahoo Finance")
    print("- Calculating YC scores (base + competitive + trajectory)")
    print("- Analyzing growth trajectory since IPO")
    print()

    results = screener.deep_screen_candidates(new_companies)

    if results.empty:
        print("\n[WARNING] No results - Companies may not have passed filters")
        print("This could mean:")
        print("  - Data not available yet (too recent)")
        print("  - Company filtered out due to thresholds")
        print("  - Ticker symbol incorrect")
        return

    # Sort by score
    results = results.sort_values('advanced_yc_score', ascending=False)

    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS RESULTS")
    print("=" * 80)
    print()

    for idx, row in results.iterrows():
        print(f"{row['ticker']} - {row['company_name']}")
        print("=" * 80)

        # Score breakdown
        print(f"\nADVANCED YC SCORE: {row['advanced_yc_score']:.1f}/100")
        print(f"  Base YC Score:        {row['base_yc_score']:.1f}/100 (40% weight)")
        print(f"  Competitive Score:    {row['competitive_score']:.1f}/100 (30% weight)")
        print(f"  Trajectory Score:     {row['trajectory_score']:.1f}/100 (30% weight)")
        print()

        # Determine tier
        score = row['advanced_yc_score']
        if score >= 75:
            tier = "EXCEPTIONAL"
            emoji = "⭐"
            rec = "STRONG BUY"
        elif score >= 65:
            tier = "VERY STRONG"
            emoji = "🔥"
            rec = "BUY"
        elif score >= 50:
            tier = "STRONG"
            emoji = "✅"
            rec = "BUY"
        elif score >= 35:
            tier = "MODERATE"
            emoji = "⚠️"
            rec = "HOLD"
        else:
            tier = "WEAK"
            emoji = "❌"
            rec = "AVOID"

        print(f"TIER: {tier} {emoji}")
        print(f"RECOMMENDATION: {rec}")
        print(f"Category: {row['category']}")
        print(f"IPO Date: {row['ipo_date'].strftime('%Y-%m-%d')} ({row['months_since_ipo']:.0f} months ago)")
        print(f"Market Cap: ${row['market_cap_b']:.2f}B")
        print()

        # Key financial metrics
        print("FINANCIAL METRICS:")
        rev_growth = (row.get('revenue_growth_yoy', 0) or 0) * 100
        gross_margin = (row.get('gross_margin', 0) or 0) * 100
        op_margin = (row.get('operating_margin', 0) or 0) * 100

        print(f"  Revenue Growth YoY: {rev_growth:.1f}%")
        print(f"  Gross Margin:       {gross_margin:.1f}%")
        print(f"  Operating Margin:   {op_margin:.1f}%")

        # Profitability check
        if op_margin > 0:
            print(f"  Status: PROFITABLE ✅")
        else:
            print(f"  Status: Not yet profitable")
        print()

        # Trajectory analysis
        traj = row.get('trajectory', {})
        if traj and traj.get('has_data'):
            print("PERFORMANCE SINCE IPO:")
            total_return = traj.get('total_return_pct', 0)
            annual_return = traj.get('annualized_return_pct', 0)

            print(f"  Total Return:       {total_return:+.1f}%")
            print(f"  Annualized Return:  {annual_return:+.1f}%")
            print(f"  Scaling Score:      {traj.get('scaling_score', 0):.0f}/100")
            print(f"  TAM Expansion:      {traj.get('tam_expansion_potential_x', 1):.1f}x potential")

            if total_return > 0:
                print(f"  Trend: UP ⬆️")
            else:
                print(f"  Trend: DOWN ⬇️ (potential opportunity?)")
        print()

        # Competitive positioning
        print("COMPETITIVE ANALYSIS:")
        comp_score = row['competitive_score']
        if comp_score >= 70:
            print(f"  Moat Strength: STRONG ({comp_score:.1f}/100)")
            print(f"  - Dominant market position")
            print(f"  - High switching costs likely")
        elif comp_score >= 60:
            print(f"  Moat Strength: MODERATE ({comp_score:.1f}/100)")
            print(f"  - Good positioning")
            print(f"  - Some competitive advantages")
        else:
            print(f"  Moat Strength: WEAK ({comp_score:.1f}/100)")
            print(f"  - Competitive market")
            print(f"  - Need to watch competition")
        print()

        print("-" * 80)
        print()

    # Investment recommendations
    print("\n" + "=" * 80)
    print("INVESTMENT RECOMMENDATIONS")
    print("=" * 80)
    print()

    for _, row in results.iterrows():
        score = row['advanced_yc_score']
        ticker = row['ticker']

        if score >= 75:
            print(f"✅ STRONG BUY: {ticker} - Exceptional tier ({score:.1f}/100)")
            print(f"   Action: Core portfolio holding (15-25% allocation)")
            print(f"   Entry: Dollar-cost average over 2-4 weeks")
            print()
        elif score >= 65:
            print(f"✅ BUY: {ticker} - Very Strong tier ({score:.1f}/100)")
            print(f"   Action: Strong portfolio holding (10-20% allocation)")
            print(f"   Entry: Dollar-cost average over 2-4 weeks")
            print()
        elif score >= 50:
            print(f"✅ BUY: {ticker} - Strong tier ({score:.1f}/100)")
            print(f"   Action: Diversification holding (5-15% allocation)")
            print(f"   Entry: Start with 5%, scale if performs well")
            print()
        elif score >= 35:
            print(f"⚠️ HOLD: {ticker} - Moderate tier ({score:.1f}/100)")
            print(f"   Action: Monitor only, don't buy yet")
            print(f"   Watch: If score improves to 50+, reconsider")
            print()
        else:
            print(f"❌ AVOID: {ticker} - Weak tier ({score:.1f}/100)")
            print(f"   Action: Skip, better opportunities exist")
            print()

    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Add promising companies (50+ score) to analyze_ai_ml_ipos_expanded.py")
    print("2. Update CURRENT_STATE.md with new findings")
    print("3. Add to portfolio tracker for ongoing monitoring")
    print("4. Run compare_weekly_results.py to track score changes")
    print()


if __name__ == "__main__":
    main()

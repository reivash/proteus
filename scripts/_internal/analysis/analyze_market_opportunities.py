"""
Analyze Additional Market Opportunities Beyond Current Watchlist

Focuses on high-growth AI/ML companies we haven't analyzed yet.

Usage:
    python analyze_market_opportunities.py
"""

import sys
sys.path.insert(0, 'src')

from src.trading.ipo_advanced_screener import AdvancedIPOScreener
from datetime import datetime


def main():
    print("=" * 80)
    print("MARKET OPPORTUNITIES ANALYSIS - BEYOND CURRENT WATCHLIST")
    print("=" * 80)
    print()

    # High-potential companies to analyze
    candidates = [
        'APP',   # AppLovin (AI advertising platform - 77% revenue growth)
        'MRVL',  # Marvell Technology (AI chips/ASICs - 63% revenue growth)
        'MU',    # Micron Technology (AI memory/HBM - strong AI tailwind)
        'QCOM',  # Qualcomm (AI edge devices - 10% revenue growth, 16x P/E)
    ]

    print("CANDIDATES TO ANALYZE:")
    print()
    print("1. APP (AppLovin)")
    print("   - AI-powered advertising platform")
    print("   - Q2 2025: 77% revenue growth, 156% earnings growth")
    print("   - Axon 2.0: 2M ad auctions/sec, learns from 1B devices")
    print("   - Expanding into e-commerce vertical")
    print()
    print("2. MRVL (Marvell Technology)")
    print("   - AI infrastructure semiconductors")
    print("   - Q1 FY2026: 63% revenue growth, 158% earnings growth")
    print("   - Custom ASICs for data centers")
    print("   - Valuation: 22x P/E (reasonable)")
    print()
    print("3. MU (Micron Technology)")
    print("   - AI memory (HBM4) for data centers")
    print("   - +56% YTD, shipping HBM4 36GB to customers")
    print("   - Valuation: 9.98x forward P/E (very cheap!)")
    print("   - PEG ratio: 0.14 (extremely undervalued)")
    print()
    print("4. QCOM (Qualcomm)")
    print("   - AI edge devices (smartphones, IoT, automotive)")
    print("   - Q3 FY2025: 10% revenue growth, 25% earnings growth")
    print("   - Valuation: 16x P/E (attractive)")
    print("   - Acquired MovianAI for expanded AI capabilities")
    print()

    # Initialize screener with NO IPO age filter (these are older companies)
    print("Initializing screener (NO IPO age limit for established companies)...")
    screener = AdvancedIPOScreener(
        max_ipo_age_months=240,  # 20 years (essentially no limit)
        min_yc_score=20.0,
        min_market_cap_b=1.0,
        max_market_cap_b=1000.0  # Higher cap for established companies
    )
    print("[OK] Screener initialized")
    print()

    # Run analysis
    print("Running deep analysis...")
    print("This may take 2-3 minutes...")
    print()

    results = screener.deep_screen_candidates(candidates)

    if results.empty:
        print("\n[WARNING] No results - Companies may not have passed filters")
        return

    # Sort by score
    results = results.sort_values('advanced_yc_score', ascending=False)

    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS - RANKED BY SCORE")
    print("=" * 80)
    print()

    for idx, row in results.iterrows():
        ticker = row['ticker']
        score = row['advanced_yc_score']

        print(f"\n{'='*80}")
        print(f"{ticker} - {row['company_name']}")
        print(f"{'='*80}")

        # Score
        print(f"\nADVANCED YC SCORE: {score:.1f}/100")
        print(f"  Base YC:        {row['base_yc_score']:.1f}/100")
        print(f"  Competitive:    {row['competitive_score']:.1f}/100")
        print(f"  Trajectory:     {row['trajectory_score']:.1f}/100")
        print()

        # Tier
        if score >= 75:
            tier, emoji, rec = "EXCEPTIONAL", "⭐", "STRONG BUY"
        elif score >= 65:
            tier, emoji, rec = "VERY STRONG", "🔥", "BUY"
        elif score >= 50:
            tier, emoji, rec = "STRONG", "✅", "BUY"
        elif score >= 35:
            tier, emoji, rec = "MODERATE", "⚠️", "HOLD"
        else:
            tier, emoji, rec = "WEAK", "❌", "AVOID"

        print(f"TIER: {tier}")
        print(f"RECOMMENDATION: {rec}")
        print()

        # Metrics
        rev_growth = (row.get('revenue_growth_yoy', 0) or 0) * 100
        gross_margin = (row.get('gross_margin', 0) or 0) * 100
        op_margin = (row.get('operating_margin', 0) or 0) * 100

        print("KEY METRICS:")
        print(f"  Market Cap:       ${row['market_cap_b']:.1f}B")
        print(f"  Revenue Growth:   {rev_growth:.1f}% YoY")
        print(f"  Gross Margin:     {gross_margin:.1f}%")
        print(f"  Operating Margin: {op_margin:.1f}%")
        profitable = " (PROFITABLE)" if op_margin > 0 else " (Not profitable)"
        print(f"  Status:{profitable}")
        print()

        # Trajectory
        traj = row.get('trajectory', {})
        if traj and traj.get('has_data'):
            total_ret = traj.get('total_return_pct', 0)
            annual_ret = traj.get('annualized_return_pct', 0)
            scaling = traj.get('scaling_score', 0)

            print("PERFORMANCE:")
            print(f"  Total Return:     {total_ret:+.1f}%")
            print(f"  Annual Return:    {annual_ret:+.1f}%")
            print(f"  Scaling Score:    {scaling:.0f}/100")
            print()

        print("-" * 80)

    # Rankings
    print("\n" + "=" * 80)
    print("INVESTMENT RANKINGS")
    print("=" * 80)
    print()

    rankings = []
    for _, row in results.iterrows():
        score = row['advanced_yc_score']
        ticker = row['ticker']

        if score >= 65:
            tier = "STRONG BUY"
        elif score >= 50:
            tier = "BUY"
        elif score >= 35:
            tier = "HOLD"
        else:
            tier = "AVOID"

        rankings.append({
            'ticker': ticker,
            'score': score,
            'tier': tier,
            'company': row['company_name']
        })

    for i, r in enumerate(rankings, 1):
        print(f"{i}. {r['ticker']} - {r['score']:.1f}/100 ({r['tier']})")
        print(f"   {r['company']}")
        print()

    # Comparison to current holdings
    print("=" * 80)
    print("COMPARISON TO CURRENT TOP PICKS")
    print("=" * 80)
    print()
    print("CURRENT TOP 3:")
    print("  1. PLTR - 81.8/100 (EXCEPTIONAL)")
    print("  2. ALAB - 70.8/100 (VERY STRONG)")
    print("  3. CRWV - 65.2/100 (VERY STRONG)")
    print()
    print("NEW CANDIDATES:")
    for i, r in enumerate(rankings, 1):
        print(f"  {i}. {r['ticker']} - {r['score']:.1f}/100 ({r['tier']})")
    print()

    # Investment recommendations
    print("=" * 80)
    print("INVESTMENT RECOMMENDATIONS")
    print("=" * 80)
    print()

    for r in rankings:
        score = r['score']
        ticker = r['ticker']

        if score >= 75:
            print(f"STRONG BUY: {ticker} ({score:.1f})")
            print(f"  - Core portfolio holding (15-25%)")
            print(f"  - Comparable to PLTR (81.8)")
        elif score >= 65:
            print(f"BUY: {ticker} ({score:.1f})")
            print(f"  - Strong portfolio addition (10-20%)")
            print(f"  - Comparable to ALAB/CRWV (65-71)")
        elif score >= 50:
            print(f"BUY: {ticker} ({score:.1f})")
            print(f"  - Diversification holding (5-15%)")
            print(f"  - Comparable to ARM/TEM/IONQ (55-61)")
        elif score >= 35:
            print(f"HOLD/WATCH: {ticker} ({score:.1f})")
            print(f"  - Monitor, consider if score improves")
        else:
            print(f"AVOID: {ticker} ({score:.1f})")
            print(f"  - Better opportunities available")
        print()

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()

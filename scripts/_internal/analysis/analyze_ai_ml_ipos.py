"""
AI/ML IPO Company Analysis - Unicorn Potential Detection

Focuses specifically on AI/ML/data companies from recent IPOs to identify
high-growth potential "unicorn" investment opportunities.

Usage:
    python analyze_ai_ml_ipos.py
"""

import sys
sys.path.insert(0, 'src')

from common.trading.ipo_advanced_screener import AdvancedIPOScreener
from datetime import datetime


# AI/ML/Data companies from IPO database
# Categorized by AI/ML focus strength
AI_ML_COMPANIES = {
    'core_ai': [
        'AI',      # C3.ai - Enterprise AI
        'PLTR',    # Palantir - Big data analytics/AI
        'PATH',    # UiPath - RPA/AI automation
        'IONQ',    # IonQ - Quantum computing/AI
    ],
    'ai_enabled_platforms': [
        'SNOW',    # Snowflake - Data platform (enables AI/ML)
        'DDOG',    # Datadog - Monitoring with ML
        'U',       # Unity - Game engine with ML capabilities
    ],
    'ai_security': [
        'CRWD',    # CrowdStrike - AI-powered cybersecurity
        'ZS',      # Zscaler - Cloud security with AI
    ],
    'data_infrastructure': [
        'GTLB',    # GitLab - DevOps with AI features
        'BILL',    # Bill.com - FinTech with ML
    ],
}


def main():
    print("=" * 80)
    print("AI/ML IPO COMPANY ANALYSIS - UNICORN POTENTIAL DETECTION")
    print("=" * 80)
    print()
    print("Mission: Identify high-growth AI/ML companies from recent IPOs")
    print("Focus: '10x better' companies with unicorn potential")
    print()

    # Flatten all AI/ML companies
    all_ai_ml_companies = []
    for category, tickers in AI_ML_COMPANIES.items():
        all_ai_ml_companies.extend(tickers)

    print(f"AI/ML Companies to Analyze: {len(all_ai_ml_companies)}")
    print()

    # Show breakdown by category
    for category, tickers in AI_ML_COMPANIES.items():
        print(f"  {category.replace('_', ' ').title()}: {', '.join(tickers)}")
    print()

    # Initialize advanced screener
    print("Initializing advanced screener...")
    screener = AdvancedIPOScreener(
        max_ipo_age_months=72,      # Last 6 years (capture more AI/ML companies)
        min_yc_score=30.0,          # Lower threshold for exploratory
        min_market_cap_b=0.1,       # Include smaller companies
        max_market_cap_b=500.0      # Include larger successful ones
    )
    print("[OK] Screener initialized")
    print()

    print("=" * 80)
    print("STARTING DEEP ANALYSIS OF AI/ML COMPANIES")
    print("=" * 80)
    print("This will analyze:")
    print("  1. Y Combinator Scoring (niche dominance, emergence, moat, scaling)")
    print("  2. Competitive Positioning (market share, moat, 10x better)")
    print("  3. Growth Trajectory (stock performance, inflection points)")
    print()
    print(f"Estimated time: {len(all_ai_ml_companies) * 30} seconds")
    print()

    # Run deep screening
    results = screener.deep_screen_candidates(all_ai_ml_companies)

    if results.empty:
        print("\n[WARNING] No AI/ML candidates passed filters")
        print("This might indicate:")
        print("  1. API rate limits reached")
        print("  2. Data not available for some companies")
        print("  3. Need to adjust filter criteria")
        return

    # Sort by advanced score
    results = results.sort_values('advanced_yc_score', ascending=False)

    # Display results summary
    print("\n" + "=" * 80)
    print("AI/ML COMPANIES - RESULTS SUMMARY")
    print("=" * 80)
    print(f"Companies Successfully Analyzed: {len(results)}")
    print(f"Average Advanced Score: {results['advanced_yc_score'].mean():.1f}/100")
    print(f"Top Candidate: {results['ticker'].iloc[0]} ({results['advanced_yc_score'].iloc[0]:.1f}/100)")
    print()

    # Categorize by score
    exceptional = results[results['advanced_yc_score'] >= 75]
    very_strong = results[(results['advanced_yc_score'] >= 65) & (results['advanced_yc_score'] < 75)]
    strong = results[(results['advanced_yc_score'] >= 50) & (results['advanced_yc_score'] < 65)]
    moderate = results[results['advanced_yc_score'] < 50]

    print("SCORE DISTRIBUTION:")
    print(f"  Exceptional (75-100): {len(exceptional)} companies")
    if len(exceptional) > 0:
        print(f"    -> {', '.join(exceptional['ticker'].tolist())}")
    print(f"  Very Strong (65-74): {len(very_strong)} companies")
    if len(very_strong) > 0:
        print(f"    -> {', '.join(very_strong['ticker'].tolist())}")
    print(f"  Strong (50-64): {len(strong)} companies")
    if len(strong) > 0:
        print(f"    -> {', '.join(strong['ticker'].tolist())}")
    print(f"  Moderate (<50): {len(moderate)} companies")
    if len(moderate) > 0:
        print(f"    -> {', '.join(moderate['ticker'].tolist())}")
    print()

    # Display top 10 AI/ML companies
    print("=" * 80)
    print("TOP 10 AI/ML COMPANIES BY ADVANCED SCORE")
    print("=" * 80)

    display_cols = [
        'ticker',
        'company_name',
        'advanced_yc_score',
        'market_cap_b',
        'revenue_growth_yoy',
        'category'
    ]

    top_10 = results.head(10)[display_cols].copy()
    if 'revenue_growth_yoy' in top_10.columns:
        top_10['revenue_growth_yoy'] = top_10['revenue_growth_yoy'].apply(
            lambda x: f"{x*100:.1f}%" if x is not None else "N/A"
        )

    print(top_10.to_string(index=False))
    print()

    # Investment insights for top 3
    print("=" * 80)
    print("TOP 3 AI/ML UNICORN CANDIDATES - DETAILED ANALYSIS")
    print("=" * 80)

    for rank, (idx, row) in enumerate(results.head(3).iterrows(), 1):
        print(f"\n#{rank} {row['ticker']} - {row['company_name']}")
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

        # Key metrics
        print("KEY METRICS:")
        print(f"  Revenue Growth: {(row.get('revenue_growth_yoy', 0) or 0)*100:.1f}% YoY")
        print(f"  Gross Margin: {(row.get('gross_margin', 0) or 0)*100:.1f}%")
        print(f"  Operating Margin: {(row.get('operating_margin', 0) or 0)*100:.1f}%")

        # Trajectory info
        traj = row.get('trajectory', {})
        if traj and traj.get('has_data'):
            print()
            print("PERFORMANCE:")
            print(f"  Total Return: {traj.get('total_return_pct', 0):+.1f}%")
            print(f"  Scaling Score: {traj.get('scaling_score', 0):.0f}/100")
            print(f"  TAM Expansion: {traj.get('tam_expansion_potential_x', 1):.1f}x")
        print()

    # Generate comprehensive report
    print("=" * 80)
    print("GENERATING COMPREHENSIVE AI/ML REPORT")
    print("=" * 80)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"results/ai_ml_unicorn_analysis_{timestamp}.txt"
    screener.generate_advanced_report(results, filename=report_filename)

    print(f"\nFull report saved: {report_filename}")
    print()

    # Key findings summary
    print("=" * 80)
    print("KEY FINDINGS - AI/ML UNICORN POTENTIAL")
    print("=" * 80)
    print()

    if len(exceptional) > 0:
        print(f"EXCEPTIONAL OPPORTUNITIES ({len(exceptional)}):")
        for _, row in exceptional.iterrows():
            return_pct = row.get('trajectory', {}).get('total_return_pct', 0)
            print(f"  * {row['ticker']}: {row['advanced_yc_score']:.1f}/100 score, {return_pct:+.0f}% since IPO")
    else:
        print("WARNING: No exceptional (75+) candidates found in current AI/ML set")

    print()

    if len(very_strong) > 0:
        print(f"VERY STRONG CANDIDATES ({len(very_strong)}):")
        for _, row in very_strong.iterrows():
            return_pct = row.get('trajectory', {}).get('total_return_pct', 0)
            print(f"  * {row['ticker']}: {row['advanced_yc_score']:.1f}/100 score, {return_pct:+.0f}% since IPO")

    print()

    # Identify highest growth
    if not results.empty and 'revenue_growth_yoy' in results.columns:
        growth_data = results[results['revenue_growth_yoy'].notna()].copy()
        if not growth_data.empty:
            fastest_grower = growth_data.loc[growth_data['revenue_growth_yoy'].idxmax()]
            print(f"FASTEST REVENUE GROWTH:")
            print(f"  {fastest_grower['ticker']}: {fastest_grower['revenue_growth_yoy']*100:.1f}% YoY")
            print()

    # Best stock performance
    results_with_return = results[
        results['trajectory'].apply(lambda x: x.get('has_data', False) if isinstance(x, dict) else False)
    ].copy()

    if not results_with_return.empty:
        returns = results_with_return['trajectory'].apply(lambda x: x.get('total_return_pct', 0))
        best_performer_idx = returns.idxmax()
        best_performer = results_with_return.loc[best_performer_idx]
        best_return = returns.max()

        print(f"BEST STOCK PERFORMANCE:")
        print(f"  {best_performer['ticker']}: {best_return:+.1f}% since IPO")
        print()

    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("1. Review detailed report for full investment thesis")
    print("2. Monitor top candidates for entry points")
    print("3. Track score changes over time")
    print("4. Consider adding newly public AI/ML companies to database")
    print()
    print("Focus: Look for companies with:")
    print("  • High competitive moat scores")
    print("  • Strong revenue acceleration")
    print("  • TAM expansion signals")
    print("  • Scaling inflection points")
    print()
    print("Remember: Find '10x better' companies, not just '10% better'!")
    print("=" * 80)


if __name__ == "__main__":
    main()

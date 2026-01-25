"""
AI/ML IPO Company Analysis - EXPANDED VERSION

Includes slightly older IPOs (up to 8 years) to capture more mature AI/ML companies.
Focuses on AI/ML/data/security companies from IPOs to identify unicorn potential.

Usage:
    python analyze_ai_ml_ipos_expanded.py
"""

import sys
sys.path.insert(0, 'src')

from common.trading.ipo_advanced_screener import AdvancedIPOScreener
from datetime import datetime
import sys
from pathlib import Path

# Import score tracker
sys.path.insert(0, '.')
try:
    from track_score_changes import ScoreTracker
    TRACKING_ENABLED = True
except ImportError:
    TRACKING_ENABLED = False
    print("[WARNING] Score tracking not available")


# Expanded AI/ML/Data companies from IPO database
# Including slightly older but highly relevant companies
AI_ML_COMPANIES_EXPANDED = {
    'core_ai': [
        'AI',      # C3.ai - Enterprise AI (lower score but worth tracking)
        'PLTR',    # Palantir - Big data analytics/AI
        'PATH',    # UiPath - RPA/AI automation
        'IONQ',    # IonQ - Quantum computing/AI
        'TEM',     # Tempus AI - Healthcare AI/precision medicine
    ],
    'ai_enabled_platforms': [
        'SNOW',    # Snowflake - Data platform (enables AI/ML)
        'DDOG',    # Datadog - Monitoring with ML
        'U',       # Unity - Game engine with ML capabilities
    ],
    'ai_infrastructure': [
        'ALAB',    # Astera Labs - AI/cloud semiconductor connectivity
        'CRWV',    # CoreWeave - AI cloud infrastructure
        'ARM',     # ARM Holdings - AI chip architecture
    ],
    'ai_security': [
        'CRWD',    # CrowdStrike - AI-powered cybersecurity
        'ZS',      # Zscaler - Cloud security with AI
        'OKTA',    # Okta - Identity/security
        'RBRK',    # Rubrik - AI-powered cloud data security
    ],
    'data_infrastructure': [
        'GTLB',    # GitLab - DevOps with AI features
        'BILL',    # Bill.com - FinTech with ML
        'TWLO',    # Twilio - Communications API
    ],
    'ai_saas_platforms': [
        'KVYO',    # Klaviyo - AI-powered marketing automation
        'OS',      # OneStream - AI-powered finance/CPM software
        'TTAN',    # ServiceTitan - AI-powered service business platform
    ],
}


def main():
    print("=" * 80)
    print("AI/ML IPO COMPANY ANALYSIS - EXPANDED (8-YEAR LOOKBACK)")
    print("=" * 80)
    print()
    print("Mission: Comprehensive analysis of AI/ML companies from IPOs")
    print("Focus: Includes mature companies (up to 8 years post-IPO)")
    print()

    # Flatten all AI/ML companies
    all_ai_ml_companies = []
    for category, tickers in AI_ML_COMPANIES_EXPANDED.items():
        all_ai_ml_companies.extend(tickers)

    print(f"AI/ML Companies to Analyze: {len(all_ai_ml_companies)}")
    print()

    # Show breakdown by category
    for category, tickers in AI_ML_COMPANIES_EXPANDED.items():
        print(f"  {category.replace('_', ' ').title()}: {', '.join(tickers)}")
    print()

    # Initialize advanced screener with EXPANDED criteria
    print("Initializing expanded screener...")
    screener = AdvancedIPOScreener(
        max_ipo_age_months=96,      # 8 years (capture more mature companies)
        min_yc_score=15.0,          # Very low threshold (include AI even with low score)
        min_market_cap_b=0.1,       # Include smaller companies
        max_market_cap_b=500.0      # Include larger successful ones
    )
    print("[OK] Screener initialized")
    print()

    print("=" * 80)
    print("STARTING COMPREHENSIVE AI/ML ANALYSIS")
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
        return

    # Sort by advanced score
    results = results.sort_values('advanced_yc_score', ascending=False)

    # Display results summary
    print("\n" + "=" * 80)
    print("AI/ML COMPANIES - COMPREHENSIVE RESULTS")
    print("=" * 80)
    print(f"Companies Successfully Analyzed: {len(results)}")
    print(f"Average Advanced Score: {results['advanced_yc_score'].mean():.1f}/100")
    print(f"Top Candidate: {results['ticker'].iloc[0]} ({results['advanced_yc_score'].iloc[0]:.1f}/100)")
    print()

    # Categorize by score
    exceptional = results[results['advanced_yc_score'] >= 75]
    very_strong = results[(results['advanced_yc_score'] >= 65) & (results['advanced_yc_score'] < 75)]
    strong = results[(results['advanced_yc_score'] >= 50) & (results['advanced_yc_score'] < 65)]
    moderate = results[(results['advanced_yc_score'] >= 35) & (results['advanced_yc_score'] < 50)]
    weak = results[results['advanced_yc_score'] < 35]

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
    print(f"  Moderate (35-49): {len(moderate)} companies")
    if len(moderate) > 0:
        print(f"    -> {', '.join(moderate['ticker'].tolist())}")
    print(f"  Weak (<35): {len(weak)} companies")
    if len(weak) > 0:
        print(f"    -> {', '.join(weak['ticker'].tolist())}")
    print()

    # Display ALL AI/ML companies
    print("=" * 80)
    print("ALL AI/ML COMPANIES RANKED BY ADVANCED SCORE")
    print("=" * 80)

    display_cols = [
        'ticker',
        'company_name',
        'advanced_yc_score',
        'market_cap_b',
        'revenue_growth_yoy',
        'category'
    ]

    all_results = results[display_cols].copy()
    if 'revenue_growth_yoy' in all_results.columns:
        all_results['revenue_growth_yoy'] = all_results['revenue_growth_yoy'].apply(
            lambda x: f"{x*100:.1f}%" if x is not None else "N/A"
        )

    print(all_results.to_string(index=False))
    print()

    # Investment insights for top 5
    print("=" * 80)
    print("TOP 5 AI/ML UNICORN CANDIDATES - DETAILED ANALYSIS")
    print("=" * 80)

    for rank, (idx, row) in enumerate(results.head(5).iterrows(), 1):
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
            print("PERFORMANCE SINCE IPO:")
            print(f"  Total Return: {traj.get('total_return_pct', 0):+.1f}%")
            print(f"  Annualized Return: {traj.get('annualized_return_pct', 0):+.1f}%")
            print(f"  Scaling Score: {traj.get('scaling_score', 0):.0f}/100")
            print(f"  TAM Expansion: {traj.get('tam_expansion_potential_x', 1):.1f}x")
        print()

    # Generate comprehensive report
    print("=" * 80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("=" * 80)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"results/ai_ml_expanded_analysis_{timestamp}.txt"
    screener.generate_advanced_report(results, filename=report_filename)

    print(f"\nFull report saved: {report_filename}")
    print()

    # Auto-save scores for tracking
    if TRACKING_ENABLED:
        print("=" * 80)
        print("UPDATING SCORE HISTORY")
        print("=" * 80)
        try:
            tracker = ScoreTracker()
            tracker.record_scores(results)
            print("Score history updated successfully!")

            # Check for significant changes
            changes = tracker.get_score_changes()
            tier_changes = tracker.get_tier_changes()

            if tier_changes:
                print()
                print("TIER CHANGES DETECTED:")
                for change in tier_changes:
                    direction = "UPGRADED" if change['score_change'] > 0 else "DOWNGRADED"
                    print(f"  * {change['ticker']}: {change['previous_tier']} -> {change['current_tier']} ({direction})")

            if changes:
                print()
                print("SIGNIFICANT SCORE CHANGES:")
                for change in changes[:3]:  # Top 3
                    print(f"  * {change['ticker']}: {change['previous_score']:.1f} -> {change['current_score']:.1f} ({change['score_change']:+.1f})")

            print()
        except Exception as e:
            print(f"[WARNING] Could not update score tracking: {e}")
            print()
    else:
        print("[INFO] Score tracking not enabled")
        print()

    # Key findings summary
    print("=" * 80)
    print("KEY FINDINGS - COMPREHENSIVE AI/ML ANALYSIS")
    print("=" * 80)
    print()

    if len(exceptional) > 0:
        print(f"EXCEPTIONAL OPPORTUNITIES ({len(exceptional)}):")
        for _, row in exceptional.iterrows():
            return_pct = row.get('trajectory', {}).get('total_return_pct', 0)
            print(f"  * {row['ticker']}: {row['advanced_yc_score']:.1f}/100 score, {return_pct:+.0f}% since IPO")
    else:
        print("WARNING: No exceptional (75+) candidates found")

    print()

    if len(very_strong) > 0:
        print(f"VERY STRONG CANDIDATES ({len(very_strong)}):")
        for _, row in very_strong.iterrows():
            return_pct = row.get('trajectory', {}).get('total_return_pct', 0)
            print(f"  * {row['ticker']}: {row['advanced_yc_score']:.1f}/100 score, {return_pct:+.0f}% since IPO")

    print()

    # Highest growth
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
    print("Summary:")
    print(f"  - Analyzed {len(results)} AI/ML companies")
    print(f"  - {len(exceptional)} exceptional (75+)")
    print(f"  - {len(very_strong)} very strong (65-74)")
    print(f"  - {len(strong)} strong (50-64)")
    print(f"  - Report: {report_filename}")
    print()
    print("Next: Research recent 2024-2025 IPOs to add to the database!")
    print("=" * 80)


if __name__ == "__main__":
    main()

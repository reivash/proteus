"""
Weekly IPO Stock Recommender

Analyzes all IPO candidates and generates top weekly recommendation.
Uses advanced scoring to identify highest-potential opportunities.

Criteria for weekly pick:
1. Highest Advanced YC Score (75+ exceptional)
2. Recent positive momentum (stock performing well)
3. Strong fundamentals (revenue growth, margins)
4. Scaling inflection (transitioning to profitability)
5. Market opportunity (TAM expansion)

Output:
- Top recommendation with detailed analysis
- Email-ready format
- Investment thesis and entry strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.trading.ipo_advanced_screener import AdvancedIPOScreener


class WeeklyIPORecommender:
    """
    Generate weekly IPO stock recommendations.
    """

    def __init__(self):
        """Initialize weekly recommender."""
        # Initialize screener with production criteria
        self.screener = AdvancedIPOScreener(
            max_ipo_age_months=36,     # 3 years
            min_yc_score=60.0,         # Strong candidates only
            min_market_cap_b=1.0,      # $1B+ (established)
            max_market_cap_b=100.0     # <$100B (room to grow)
        )

    def calculate_momentum_score(self, trajectory: Dict) -> float:
        """
        Calculate momentum score based on recent performance.

        Args:
            trajectory: Trajectory analysis dict

        Returns:
            Momentum score 0-100
        """
        score = 0.0

        if not trajectory.get('has_data'):
            return 50.0  # Default/neutral

        # Recent performance (50 points)
        total_return = trajectory.get('total_return_pct', 0)
        if total_return > 200:  # >200% since IPO
            score += 50
        elif total_return > 100:  # >100%
            score += 40
        elif total_return > 50:   # >50%
            score += 30
        elif total_return > 0:    # Positive
            score += 20

        # Near ATH (25 points) - showing strength
        current_vs_ath = trajectory.get('current_vs_ath_pct', -50)
        if current_vs_ath > -5:    # Within 5% of ATH
            score += 25
        elif current_vs_ath > -10:  # Within 10%
            score += 20
        elif current_vs_ath > -20:  # Within 20%
            score += 10

        # Growth acceleration (25 points)
        if trajectory.get('is_accelerating'):
            score += 15
        if trajectory.get('inflection_detected'):
            score += 10

        return min(100, score)

    def calculate_recommendation_score(self, candidate: Dict) -> Tuple[float, Dict]:
        """
        Calculate overall recommendation score.

        Combines:
        - Advanced YC Score (50%)
        - Momentum (25%)
        - Fundamentals (25%)

        Args:
            candidate: Candidate analysis dict

        Returns:
            Tuple of (recommendation_score, breakdown_dict)
        """
        # Advanced YC score
        advanced_score = candidate.get('advanced_yc_score', 0)

        # Momentum score
        trajectory = candidate.get('trajectory', {})
        momentum_score = self.calculate_momentum_score(trajectory)

        # Fundamentals score
        fundamentals_score = 0.0

        # Revenue growth (40 points)
        revenue_growth = (candidate.get('revenue_growth_yoy', 0) or 0) * 100
        if revenue_growth > 100:    # >100% growth
            fundamentals_score += 40
        elif revenue_growth > 50:   # >50%
            fundamentals_score += 35
        elif revenue_growth > 30:   # >30%
            fundamentals_score += 25

        # Margins (40 points)
        gross_margin = (candidate.get('gross_margin', 0) or 0) * 100
        operating_margin = (candidate.get('operating_margin', 0) or 0) * 100

        if gross_margin > 70:
            fundamentals_score += 25
        elif gross_margin > 50:
            fundamentals_score += 15

        if operating_margin > 15:
            fundamentals_score += 15
        elif operating_margin > 0:
            fundamentals_score += 10

        # Scaling (20 points)
        if trajectory.get('is_scaling'):
            fundamentals_score += 20

        # Calculate weighted score
        recommendation_score = (
            advanced_score * 0.50 +
            momentum_score * 0.25 +
            fundamentals_score * 0.25
        )

        breakdown = {
            'recommendation_score': round(recommendation_score, 2),
            'advanced_yc_score': round(advanced_score, 2),
            'momentum_score': round(momentum_score, 2),
            'fundamentals_score': round(fundamentals_score, 2),
        }

        return recommendation_score, breakdown

    def generate_weekly_pick(self, custom_candidates: Optional[List[str]] = None) -> Dict:
        """
        Generate top weekly recommendation.

        Args:
            custom_candidates: Optional custom candidate list

        Returns:
            Top recommendation dictionary
        """
        print("=" * 80)
        print("WEEKLY IPO RECOMMENDATION GENERATOR")
        print("=" * 80)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Recommendation Period: Next Week")
        print()

        # Get candidates (always use deep analysis for weekly picks)
        if custom_candidates:
            candidates = self.screener.deep_screen_candidates(custom_candidates)
        else:
            # Use default universe with deep analysis
            candidates = self.screener.deep_screen_candidates()

        if candidates.empty:
            print("[INFO] No qualifying candidates found")
            return None

        print(f"Candidates Analyzed: {len(candidates)}")
        print()

        # Calculate recommendation scores
        rec_scores = []
        for idx, candidate in candidates.iterrows():
            score, breakdown = self.calculate_recommendation_score(candidate)
            rec_scores.append({
                'ticker': candidate['ticker'],
                'recommendation_score': score,
                **breakdown
            })

        # Add to dataframe
        rec_df = pd.DataFrame(rec_scores)
        candidates = candidates.merge(rec_df, on='ticker')

        # Sort by recommendation score
        candidates = candidates.sort_values('recommendation_score', ascending=False)

        # Top pick
        top_pick = candidates.iloc[0]

        print("=" * 80)
        print(f"TOP WEEKLY PICK: {top_pick['ticker']} - {top_pick.get('company_name', 'Unknown')}")
        print("=" * 80)
        print(f"Recommendation Score: {top_pick.get('recommendation_score', 0):.1f}/100")
        print(f"  - Advanced YC Score: {top_pick.get('advanced_yc_score', top_pick.get('base_yc_score', 0)):.1f}/100")
        print(f"  - Momentum Score: {top_pick.get('momentum_score', 0):.1f}/100")
        print(f"  - Fundamentals Score: {top_pick.get('fundamentals_score', 0):.1f}/100")
        print()

        return top_pick.to_dict()

    def generate_email_report(self, recommendation: Dict) -> str:
        """
        Generate email-ready recommendation report.

        Args:
            recommendation: Recommendation dictionary

        Returns:
            Email body text
        """
        if not recommendation:
            return "No recommendation available this week."

        ticker = recommendation['ticker']
        company_name = recommendation['company_name']
        category = recommendation.get('category', 'Unknown')

        # Build email
        lines = []
        lines.append("=" * 80)
        lines.append("WEEKLY IPO STOCK RECOMMENDATION")
        lines.append(f"Week of {datetime.now().strftime('%B %d, %Y')}")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"TOP PICK: {ticker} - {company_name}")
        lines.append("=" * 80)
        lines.append("")

        # Summary scores
        lines.append("RECOMMENDATION SCORE: {:.1f}/100".format(recommendation.get('recommendation_score', 0)))
        lines.append("")
        lines.append("Score Breakdown:")
        lines.append("  - Advanced YC Score: {:.1f}/100".format(
            recommendation.get('advanced_yc_score', recommendation.get('base_yc_score', 0))))
        lines.append("  - Momentum Score: {:.1f}/100".format(recommendation.get('momentum_score', 0)))
        lines.append("  - Fundamentals Score: {:.1f}/100".format(recommendation.get('fundamentals_score', 0)))
        lines.append("")

        # Company basics
        lines.append("COMPANY PROFILE:")
        lines.append(f"  Category: {category}")
        lines.append(f"  IPO Date: {recommendation['ipo_date'].strftime('%B %d, %Y')}")
        lines.append(f"  Months Since IPO: {recommendation['months_since_ipo']:.0f}")
        lines.append(f"  Market Cap: ${recommendation['market_cap_b']:.2f}B")
        lines.append("")

        # Performance
        traj = recommendation.get('trajectory', {})
        if traj.get('has_data'):
            lines.append("STOCK PERFORMANCE:")
            lines.append(f"  IPO Price: ${traj.get('ipo_price', 0):.2f}")
            lines.append(f"  Current Price: ${traj.get('current_price', 0):.2f}")
            lines.append(f"  Total Return: {traj.get('total_return_pct', 0):+.1f}%")
            lines.append(f"  Annualized Return: {traj.get('annualized_return_pct', 0):+.1f}%")
            lines.append("")

        # Fundamentals
        lines.append("KEY FUNDAMENTALS:")
        lines.append(f"  Revenue Growth: {(recommendation.get('revenue_growth_yoy', 0) or 0)*100:.1f}% YoY")
        lines.append(f"  Gross Margin: {(recommendation.get('gross_margin', 0) or 0)*100:.1f}%")
        lines.append(f"  Operating Margin: {(recommendation.get('operating_margin', 0) or 0)*100:.1f}%")
        lines.append("")

        # Competitive position
        comp = recommendation.get('competitive', {})
        if comp.get('has_peers'):
            lines.append("COMPETITIVE POSITION:")
            lines.append(f"  Market Share: {comp.get('market_share_pct', 0):.1f}%")
            lines.append(f"  Competitive Moat: {comp.get('moat_score', 0):.1f}/100")
            if comp.get('10x_signals'):
                lines.append("  10x Signals:")
                for signal in comp.get('10x_signals', [])[:3]:  # Top 3
                    lines.append(f"    - {signal}")
            lines.append("")

        # Growth trajectory
        if traj.get('has_data'):
            lines.append("GROWTH TRAJECTORY:")
            lines.append(f"  Scaling: {traj.get('is_scaling', False)}")
            lines.append(f"  TAM Expansion Potential: {traj.get('tam_expansion_potential_x', 1):.1f}x")
            lines.append(f"  Outpacing Market Growth: {traj.get('outpacing_market_growth', False)}")
            lines.append("")

        # Investment thesis
        lines.append("=" * 80)
        lines.append("INVESTMENT THESIS")
        lines.append("=" * 80)

        thesis_points = []

        # Build thesis
        if recommendation.get('is_niche'):
            thesis_points.append(f"Dominates {category} niche market")

        growth_rate = (recommendation.get('revenue_growth_yoy', 0) or 0) * 100
        if growth_rate > 50:
            thesis_points.append(f"Explosive {growth_rate:.0f}% YoY revenue growth")

        gross_margin = (recommendation.get('gross_margin', 0) or 0) * 100
        if gross_margin > 70:
            thesis_points.append(f"Exceptional unit economics ({gross_margin:.0f}% gross margin)")

        if traj.get('is_scaling'):
            thesis_points.append("Successfully scaling operations")

        if traj.get('total_return_pct', 0) > 100:
            thesis_points.append(f"Strong market validation (+{traj['total_return_pct']:.0f}% since IPO)")

        if traj.get('outpacing_market_growth'):
            thesis_points.append("Outpacing overall market growth rate")

        for i, point in enumerate(thesis_points, 1):
            lines.append(f"{i}. {point}")

        lines.append("")

        # Recommendation
        lines.append("=" * 80)
        lines.append("RECOMMENDATION")
        lines.append("=" * 80)
        lines.append(f"BUY {ticker} for next week")
        lines.append("")
        lines.append("Entry Strategy:")
        lines.append(f"  - Current Price: ${traj.get('current_price', 0):.2f}")
        lines.append(f"  - Suggested Entry: Market order or limit near current price")
        lines.append(f"  - Position Size: Based on your portfolio allocation")
        lines.append("")
        lines.append("Risk Factors:")
        lines.append(f"  - Volatility: {traj.get('annualized_volatility_pct', 0):.1f}% annualized")
        lines.append(f"  - Max Drawdown: {traj.get('max_drawdown_pct', 0):.1f}%")
        lines.append("")

        lines.append("=" * 80)
        lines.append("Disclaimer: This is an algorithmic recommendation based on Y Combinator")
        lines.append("investment principles. Past performance does not guarantee future results.")
        lines.append("Always conduct your own research before investing.")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Generated by: Proteus Advanced IPO Analysis System")
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(lines)


if __name__ == "__main__":
    # Generate this week's recommendation
    recommender = WeeklyIPORecommender()

    # Focus on proven candidates for demo
    demo_candidates = ['RDDT', 'ASND', 'COIN', 'SNOW', 'RBLX']

    # Generate pick
    recommendation = recommender.generate_weekly_pick(demo_candidates)

    if recommendation:
        # Generate email report
        email_body = recommender.generate_email_report(recommendation)

        # Save to file
        filename = f"results/weekly_recommendation_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(email_body)

        print(f"\nEmail report saved: {filename}")
        print("\n" + "=" * 80)
        print("EMAIL PREVIEW:")
        print("=" * 80)
        print(email_body)

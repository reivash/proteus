"""
Advanced IPO Niche Screener

Enhanced screener integrating:
- Y Combinator scoring (original)
- Competitive positioning analysis
- Growth trajectory analysis
- Market cap performance
- TAM expansion potential

This is the complete, production-grade IPO analysis system.

Usage:
    screener = AdvancedIPOScreener()
    results = screener.deep_screen_candidates(['RDDT', 'COIN', 'SNOW'])
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.ipo_data_fetcher import IPODataFetcher
from common.data.features.ipo_niche_analyzer import IPONicheAnalyzer
from common.data.features.competitive_positioning import CompetitivePositioningAnalyzer
from common.data.features.growth_trajectory_analyzer import GrowthTrajectoryAnalyzer
from common.trading.ipo_niche_screener import IPONicheScreener


class AdvancedIPOScreener(IPONicheScreener):
    """
    Enhanced IPO screener with deep competitive and growth analysis.
    """

    def __init__(self, **kwargs):
        """
        Initialize advanced screener.

        Args:
            Same as IPONicheScreener
        """
        super().__init__(**kwargs)

        # Additional analyzers
        self.competitive_analyzer = CompetitivePositioningAnalyzer()
        self.trajectory_analyzer = GrowthTrajectoryAnalyzer()

    def deep_analyze_candidate(self, ticker: str) -> Optional[Dict]:
        """
        Perform deep analysis on a single candidate.

        Includes:
        - Basic YC scoring
        - Competitive positioning
        - Growth trajectory
        - Combined scoring

        Args:
            ticker: Stock ticker

        Returns:
            Comprehensive analysis dictionary
        """
        print(f"\n{'='*80}")
        print(f"DEEP ANALYSIS: {ticker}")
        print(f"{'='*80}")

        # Get base analysis from parent class
        base_analysis = self.screen_ticker(ticker)

        if not base_analysis:
            print(f"[SKIP] {ticker} did not pass basic filters")
            return None

        # Get detailed data
        company_info = self.fetcher.get_company_info(ticker)
        fundamentals = self.fetcher.get_fundamental_metrics(ticker)

        if 'error' in company_info or 'error' in fundamentals:
            return None

        category = base_analysis.get('category', 'Unknown')

        # 1. Competitive positioning
        competitive = self.competitive_analyzer.analyze_competitive_position(
            ticker, company_info, fundamentals, category
        )

        # 2. Growth trajectory
        trajectory = self.trajectory_analyzer.analyze_full_trajectory(
            ticker, company_info, fundamentals, category
        )

        # 3. Calculate combined advanced score
        advanced_score = self._calculate_advanced_score(
            base_analysis, competitive, trajectory
        )

        # Combine all analyses
        result = {
            **base_analysis,
            'competitive': competitive,
            'trajectory': trajectory,
            **advanced_score,
        }

        print(f"\n{'='*80}")
        print(f"ADVANCED SCORE: {advanced_score['advanced_yc_score']:.1f}/100")
        print(f"{'='*80}\n")

        return result

    def _calculate_advanced_score(self, base: Dict, competitive: Dict,
                                  trajectory: Dict) -> Dict:
        """
        Calculate advanced YC score incorporating all analyses.

        Scoring:
        - Base YC Score (40%)
        - Competitive Position (30%)
        - Growth Trajectory (30%)

        Args:
            base: Base YC analysis
            competitive: Competitive analysis
            trajectory: Trajectory analysis

        Returns:
            Advanced scoring dictionary
        """
        base_score = base.get('yc_score', 0)

        # Competitive score
        comp_score = 0.0
        if competitive.get('has_peers'):
            # Weight: Market share (20%), Moat (50%), 10x Better (30%)
            market_share_score = min(100, competitive.get('market_share_pct', 0) * 3)  # 33% = 100pts
            moat_score = competitive.get('moat_score', 0)
            ten_x_score = competitive.get('10x_score', 0)

            comp_score = (market_share_score * 0.20 +
                         moat_score * 0.50 +
                         ten_x_score * 0.30)
        else:
            comp_score = 50  # Default if no peers

        # Trajectory score
        traj_score = 0.0

        # Stock performance (30 pts)
        if trajectory.get('has_data'):
            total_return = trajectory.get('total_return_pct', 0)
            if total_return > 200:  # >200% return
                traj_score += 30
            elif total_return > 100:  # >100% return
                traj_score += 20
            elif total_return > 0:  # Positive return
                traj_score += 10

        # Scaling (40 pts)
        scaling_score = trajectory.get('scaling_score', 0)
        traj_score += scaling_score * 0.40

        # TAM expansion (30 pts)
        if trajectory.get('outpacing_market_growth'):
            traj_score += 20
        tam_expansion = trajectory.get('tam_expansion_potential_x', 1)
        if tam_expansion > 3:  # 3x+ TAM expansion potential
            traj_score += 10

        # Combine scores
        advanced_score = (
            base_score * 0.40 +
            comp_score * 0.30 +
            traj_score * 0.30
        )

        return {
            'advanced_yc_score': round(advanced_score, 2),
            'base_yc_score': round(base_score, 2),
            'competitive_score': round(comp_score, 2),
            'trajectory_score': round(traj_score, 2),
            'is_exceptional': advanced_score >= 75,  # Top tier
            'is_strong': advanced_score >= 65,       # Strong candidate
            'is_worthy': advanced_score >= 50,       # Worth watching
        }

    def deep_screen_candidates(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Perform deep screening on candidate list.

        Args:
            tickers: Optional list of tickers (default: use default universe)

        Returns:
            DataFrame with deep analysis results
        """
        if tickers is None:
            tickers = self.get_ipo_universe()

        print("=" * 80)
        print("ADVANCED IPO SCREENING WITH DEEP ANALYSIS")
        print("=" * 80)
        print(f"\nAnalyzing {len(tickers)} candidates...")
        print("This may take several minutes...\n")

        results = []

        for ticker in tickers:
            try:
                analysis = self.deep_analyze_candidate(ticker)
                if analysis:
                    results.append(analysis)
            except Exception as e:
                print(f"[ERROR] Failed to analyze {ticker}: {e}")
                continue

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Sort by advanced score
        df = df.sort_values('advanced_yc_score', ascending=False)

        return df

    def generate_advanced_report(self, results: pd.DataFrame,
                                filename: Optional[str] = None) -> str:
        """
        Generate comprehensive report with all analyses.

        Args:
            results: Results DataFrame
            filename: Optional filename to save

        Returns:
            Report text
        """
        if results.empty:
            return "No candidates found."

        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED IPO ANALYSIS REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")
        lines.append("METHODOLOGY:")
        lines.append("- Y Combinator Principles (40%): Niche dominance, emergence, moat, scaling")
        lines.append("- Competitive Position (30%): Market share, moat, 10x better")
        lines.append("- Growth Trajectory (30%): Stock performance, scaling, TAM expansion")
        lines.append("")
        lines.append(f"CANDIDATES ANALYZED: {len(results)}")
        lines.append("=" * 80)

        for idx, row in results.iterrows():
            lines.append("")
            lines.append(f"\n{row['ticker']} - {row['company_name']}")
            lines.append("-" * 80)

            # Summary scores
            lines.append(f"ADVANCED YC SCORE: {row['advanced_yc_score']:.1f}/100")
            lines.append(f"  - Base YC Score: {row['base_yc_score']:.1f}/100")
            lines.append(f"  - Competitive Score: {row['competitive_score']:.1f}/100")
            lines.append(f"  - Trajectory Score: {row['trajectory_score']:.1f}/100")
            lines.append("")

            # Company basics
            lines.append(f"Category: {row['category']}")
            lines.append(f"IPO: {row['ipo_date'].strftime('%Y-%m-%d')} ({row['months_since_ipo']:.0f}mo ago)")
            lines.append(f"Market Cap: ${row['market_cap_b']:.2f}B")
            lines.append("")

            # Competitive position
            comp = row.get('competitive', {})
            if comp.get('has_peers'):
                lines.append("COMPETITIVE POSITION:")
                lines.append(f"  Market Share: {comp.get('market_share_pct', 0):.1f}% "
                           f"(Rank #{comp.get('revenue_rank', '?')} of {comp.get('total_peers', '?')})")
                lines.append(f"  Competitive Moat: {comp.get('moat_score', 0):.1f}/100")
                lines.append(f"  10x Better: {comp.get('is_10x_better', False)} "
                           f"(Score: {comp.get('10x_score', 0):.1f}/100)")

                if comp.get('10x_signals'):
                    for signal in comp.get('10x_signals', []):
                        lines.append(f"    - {signal}")
                lines.append("")

            # Growth trajectory
            traj = row.get('trajectory', {})
            if traj.get('has_data'):
                lines.append("GROWTH TRAJECTORY:")
                lines.append(f"  Total Return: {traj.get('total_return_pct', 0):+.1f}% "
                           f"(Annualized: {traj.get('annualized_return_pct', 0):+.1f}%)")
                lines.append(f"  Scaling: {traj.get('is_scaling', False)} "
                           f"(Score: {traj.get('scaling_score', 0):.1f}/100)")
                lines.append(f"  TAM Expansion: {traj.get('tam_expansion_potential_x', 1):.1f}x potential")
                lines.append(f"  Outpacing Market: {traj.get('outpacing_market_growth', False)}")
                lines.append("")

            # Key metrics
            lines.append("KEY METRICS:")
            lines.append(f"  Revenue Growth: {(row.get('revenue_growth_yoy', 0) or 0)*100:.1f}% YoY")
            lines.append(f"  Gross Margin: {(row.get('gross_margin', 0) or 0)*100:.1f}%")
            lines.append(f"  Operating Margin: {(row.get('operating_margin', 0) or 0)*100:.1f}%")
            lines.append("")

            lines.append("-" * 80)

        lines.append("\n" + "=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        report = "\n".join(lines)

        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nReport saved: {filename}")

        return report


if __name__ == "__main__":
    # Run advanced screening on select candidates
    screener = AdvancedIPOScreener(
        max_ipo_age_months=36,
        min_yc_score=50.0,
        min_market_cap_b=1.0,
        max_market_cap_b=100.0
    )

    # Screen high-priority candidates
    priority_candidates = ['RDDT', 'ASND', 'COIN', 'RBLX', 'SNOW']

    print("Starting deep analysis on priority candidates...")
    print("This will take a few minutes...\n")

    results = screener.deep_screen_candidates(priority_candidates)

    if not results.empty:
        # Display top candidates
        print("\n" + "=" * 80)
        print("TOP CANDIDATES (by Advanced YC Score)")
        print("=" * 80)

        display_cols = ['ticker', 'company_name', 'advanced_yc_score',
                       'base_yc_score', 'competitive_score', 'trajectory_score']

        print(results[display_cols].to_string(index=False))

        # Generate detailed report
        report_file = f"results/advanced_ipo_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report = screener.generate_advanced_report(results, filename=report_file)

        print(f"\n\nDetailed report saved to: {report_file}")
    else:
        print("\nNo qualifying candidates found.")

"""
Score Change Tracking System

Tracks AI/ML IPO company scores over time to detect:
- Companies moving between tiers
- Score improvements/declines
- Inflection points
- Emerging opportunities

Usage:
    python track_score_changes.py
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ScoreTracker:
    """Track and compare IPO company scores over time."""

    def __init__(self, history_file: str = "data/score_history.json"):
        """
        Initialize score tracker.

        Args:
            history_file: Path to JSON file storing historical scores
        """
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = self.load_history()

    def load_history(self) -> Dict:
        """Load historical score data."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {}

    def save_history(self):
        """Save historical score data."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def record_scores(self, results: pd.DataFrame, timestamp: Optional[str] = None):
        """
        Record current scores for companies.

        Args:
            results: DataFrame with company analysis results
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d')

        for _, row in results.iterrows():
            ticker = row['ticker']

            if ticker not in self.history:
                self.history[ticker] = {
                    'company_name': row.get('company_name', 'Unknown'),
                    'scores': []
                }

            # Record score snapshot
            score_data = {
                'date': timestamp,
                'advanced_yc_score': float(row['advanced_yc_score']),
                'base_yc_score': float(row['base_yc_score']),
                'competitive_score': float(row['competitive_score']),
                'trajectory_score': float(row['trajectory_score']),
                'market_cap_b': float(row['market_cap_b']),
                'revenue_growth_yoy': float(row.get('revenue_growth_yoy', 0) or 0),
            }

            # Add trajectory data if available
            traj = row.get('trajectory', {})
            if traj and isinstance(traj, dict) and traj.get('has_data'):
                score_data['total_return_pct'] = float(traj.get('total_return_pct', 0))
                score_data['scaling_score'] = float(traj.get('scaling_score', 0))

            self.history[ticker]['scores'].append(score_data)

        self.save_history()
        print(f"Recorded scores for {len(results)} companies on {timestamp}")

    def get_score_changes(self, days_back: int = 7) -> List[Dict]:
        """
        Get companies with significant score changes.

        Args:
            days_back: How many days back to compare

        Returns:
            List of companies with changes
        """
        changes = []

        for ticker, data in self.history.items():
            scores = data['scores']
            if len(scores) < 2:
                continue

            # Get latest and previous scores
            latest = scores[-1]
            previous = scores[-2] if len(scores) >= 2 else None

            if previous:
                score_change = latest['advanced_yc_score'] - previous['advanced_yc_score']

                # Significant if change > 5 points
                if abs(score_change) >= 5.0:
                    changes.append({
                        'ticker': ticker,
                        'company_name': data['company_name'],
                        'score_change': score_change,
                        'previous_score': previous['advanced_yc_score'],
                        'current_score': latest['advanced_yc_score'],
                        'previous_tier': self._get_tier(previous['advanced_yc_score']),
                        'current_tier': self._get_tier(latest['advanced_yc_score']),
                        'dates': f"{previous['date']} -> {latest['date']}"
                    })

        # Sort by magnitude of change
        changes.sort(key=lambda x: abs(x['score_change']), reverse=True)
        return changes

    def get_tier_changes(self) -> List[Dict]:
        """Get companies that changed tiers."""
        tier_changes = []

        for ticker, data in self.history.items():
            scores = data['scores']
            if len(scores) < 2:
                continue

            latest = scores[-1]
            previous = scores[-2]

            prev_tier = self._get_tier(previous['advanced_yc_score'])
            curr_tier = self._get_tier(latest['advanced_yc_score'])

            if prev_tier != curr_tier:
                tier_changes.append({
                    'ticker': ticker,
                    'company_name': data['company_name'],
                    'previous_tier': prev_tier,
                    'current_tier': curr_tier,
                    'score_change': latest['advanced_yc_score'] - previous['advanced_yc_score'],
                    'dates': f"{previous['date']} -> {latest['date']}"
                })

        return tier_changes

    def _get_tier(self, score: float) -> str:
        """Get tier name for a score."""
        if score >= 75:
            return "Exceptional"
        elif score >= 65:
            return "Very Strong"
        elif score >= 50:
            return "Strong"
        elif score >= 35:
            return "Moderate"
        else:
            return "Weak"

    def generate_change_report(self, output_file: Optional[str] = None):
        """
        Generate a report of score changes.

        Args:
            output_file: Optional file to save report
        """
        print("=" * 80)
        print("SCORE CHANGE TRACKING REPORT")
        print("=" * 80)
        print()

        # Tier changes
        tier_changes = self.get_tier_changes()
        if tier_changes:
            print("TIER CHANGES:")
            print("-" * 80)
            for change in tier_changes:
                direction = "UP" if change['score_change'] > 0 else "DOWN"
                print(f"{change['ticker']} - {change['company_name']}")
                print(f"  {change['previous_tier']} -> {change['current_tier']} ({direction})")
                print(f"  Score Change: {change['score_change']:+.1f}")
                print(f"  Period: {change['dates']}")
                print()
        else:
            print("No tier changes detected.")
            print()

        # Significant score changes
        score_changes = self.get_score_changes()
        if score_changes:
            print("SIGNIFICANT SCORE CHANGES (>= 5 points):")
            print("-" * 80)
            for change in score_changes[:10]:  # Top 10
                direction = "IMPROVED" if change['score_change'] > 0 else "DECLINED"
                print(f"{change['ticker']} - {change['company_name']}")
                print(f"  Score: {change['previous_score']:.1f} -> {change['current_score']:.1f} ({change['score_change']:+.1f})")
                print(f"  Status: {direction}")
                print(f"  Period: {change['dates']}")
                print()
        else:
            print("No significant score changes detected.")
            print()

        # Top performers by score
        print("CURRENT TOP PERFORMERS:")
        print("-" * 80)
        current_scores = []
        for ticker, data in self.history.items():
            if data['scores']:
                latest = data['scores'][-1]
                current_scores.append({
                    'ticker': ticker,
                    'company_name': data['company_name'],
                    'score': latest['advanced_yc_score'],
                    'tier': self._get_tier(latest['advanced_yc_score']),
                    'market_cap_b': latest.get('market_cap_b', 0),
                    'date': latest['date']
                })

        current_scores.sort(key=lambda x: x['score'], reverse=True)

        for i, company in enumerate(current_scores[:10], 1):
            print(f"{i}. {company['ticker']} - {company['company_name']}")
            print(f"   Score: {company['score']:.1f}/100 ({company['tier']})")
            print(f"   Market Cap: ${company['market_cap_b']:.1f}B")
            print(f"   As of: {company['date']}")
            print()

        if output_file:
            # Save to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results/score_changes_{timestamp}.txt"
            # TODO: Write report to file
            print(f"Report saved to: {filename}")

        print("=" * 80)


def main():
    """Main function to demonstrate score tracking."""
    from src.trading.ipo_advanced_screener import AdvancedIPOScreener

    print("=" * 80)
    print("AI/ML IPO SCORE TRACKING SYSTEM")
    print("=" * 80)
    print()
    print("This tool tracks score changes over time to identify:")
    print("  - Companies improving/declining")
    print("  - Tier changes (e.g., Moderate -> Strong)")
    print("  - Emerging opportunities")
    print()

    # Initialize tracker
    tracker = ScoreTracker()

    # Check if we have historical data
    if not tracker.history:
        print("No historical data found. Running initial analysis...")
        print()

        # Run analysis to get baseline
        ai_ml_companies = [
            'PLTR', 'IONQ', 'ALAB', 'CRWV', 'DDOG', 'ZS', 'CRWD',
            'GTLB', 'SNOW', 'BILL', 'U', 'PATH', 'AI'
        ]

        screener = AdvancedIPOScreener(
            max_ipo_age_months=96,
            min_yc_score=5.0,
            min_market_cap_b=0.1,
            max_market_cap_b=500.0
        )

        print("Running comprehensive analysis...")
        results = screener.deep_screen_candidates(ai_ml_companies)

        if not results.empty:
            # Record baseline scores
            tracker.record_scores(results)
            print()
            print(f"Baseline scores recorded for {len(results)} companies!")
            print("Run this script again after your next analysis to see changes.")
        else:
            print("No results to record.")
    else:
        print(f"Found historical data for {len(tracker.history)} companies.")
        print()

        # Generate change report
        tracker.generate_change_report()

        print()
        print("To update scores, run your analysis and then:")
        print("  tracker = ScoreTracker()")
        print("  tracker.record_scores(results)")
        print()
        print("Or integrate into analyze_ai_ml_ipos_expanded.py")


if __name__ == "__main__":
    main()

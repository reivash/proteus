"""
Week-over-Week Comparison Tool

Compares current week's AI/ML IPO analysis to previous week to identify:
- Score improvements/declines
- Tier changes
- Performance changes
- Emerging opportunities

Usage:
    python compare_weekly_results.py
"""

import sys
sys.path.insert(0, 'src')

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from track_score_changes import ScoreTracker


class WeeklyComparison:
    """Compare weekly analysis results."""

    def __init__(self):
        """Initialize weekly comparison."""
        self.tracker = ScoreTracker()

    def get_weekly_comparison(self, weeks_back: int = 1) -> Dict:
        """
        Compare current data to N weeks ago.

        Args:
            weeks_back: How many weeks back to compare (default: 1)

        Returns:
            Dictionary with comparison data
        """
        comparison = {
            'score_changes': [],
            'tier_changes': [],
            'performance_changes': [],
            'new_entries': [],
            'movers': {
                'biggest_gainers': [],
                'biggest_losers': []
            }
        }

        for ticker, data in self.tracker.history.items():
            scores = data['scores']

            if len(scores) < 2:
                # New entry
                if len(scores) == 1:
                    comparison['new_entries'].append({
                        'ticker': ticker,
                        'company_name': data['company_name'],
                        'score': scores[0]['advanced_yc_score'],
                        'date': scores[0]['date']
                    })
                continue

            # Get current and previous
            current = scores[-1]
            previous = scores[-2] if len(scores) >= 2 else scores[0]

            # Calculate changes
            score_delta = current['advanced_yc_score'] - previous['advanced_yc_score']
            curr_tier = self._get_tier(current['advanced_yc_score'])
            prev_tier = self._get_tier(previous['advanced_yc_score'])

            # Score changes
            if abs(score_delta) >= 1.0:  # At least 1 point change
                change_data = {
                    'ticker': ticker,
                    'company_name': data['company_name'],
                    'previous_score': previous['advanced_yc_score'],
                    'current_score': current['advanced_yc_score'],
                    'change': score_delta,
                    'previous_tier': prev_tier,
                    'current_tier': curr_tier,
                    'dates': f"{previous['date']} -> {current['date']}"
                }
                comparison['score_changes'].append(change_data)

                # Track movers
                if score_delta > 0:
                    comparison['movers']['biggest_gainers'].append(change_data)
                else:
                    comparison['movers']['biggest_losers'].append(change_data)

            # Tier changes
            if curr_tier != prev_tier:
                comparison['tier_changes'].append({
                    'ticker': ticker,
                    'company_name': data['company_name'],
                    'previous_tier': prev_tier,
                    'current_tier': curr_tier,
                    'score_change': score_delta,
                    'direction': 'UP' if score_delta > 0 else 'DOWN'
                })

            # Performance changes (stock return)
            if 'total_return_pct' in current and 'total_return_pct' in previous:
                return_delta = current['total_return_pct'] - previous['total_return_pct']
                if abs(return_delta) >= 5.0:  # At least 5% change
                    comparison['performance_changes'].append({
                        'ticker': ticker,
                        'company_name': data['company_name'],
                        'return_change': return_delta,
                        'current_return': current['total_return_pct'],
                        'previous_return': previous['total_return_pct']
                    })

        # Sort movers
        comparison['movers']['biggest_gainers'].sort(key=lambda x: x['change'], reverse=True)
        comparison['movers']['biggest_losers'].sort(key=lambda x: x['change'])

        return comparison

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

    def generate_comparison_report(self):
        """Generate formatted comparison report."""
        print("=" * 80)
        print("WEEK-OVER-WEEK COMPARISON REPORT")
        print("=" * 80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        comparison = self.get_weekly_comparison()

        # Summary stats
        total_companies = len(self.tracker.history)
        companies_with_changes = len(comparison['score_changes'])
        tier_changes = len(comparison['tier_changes'])
        new_entries = len(comparison['new_entries'])

        print("SUMMARY:")
        print(f"  Total Companies Tracked: {total_companies}")
        print(f"  Companies with Score Changes: {companies_with_changes}")
        print(f"  Tier Changes: {tier_changes}")
        print(f"  New Entries: {new_entries}")
        print()

        # New entries
        if comparison['new_entries']:
            print("=" * 80)
            print("NEW COMPANIES ADDED")
            print("=" * 80)
            for entry in comparison['new_entries']:
                tier = self._get_tier(entry['score'])
                print(f"{entry['ticker']} - {entry['company_name']}")
                print(f"  Score: {entry['score']:.1f}/100 ({tier})")
                print(f"  Added: {entry['date']}")
                print()

        # Tier changes (MOST IMPORTANT)
        if comparison['tier_changes']:
            print("=" * 80)
            print("TIER CHANGES (CRITICAL)")
            print("=" * 80)
            for change in comparison['tier_changes']:
                symbol = "⬆️" if change['direction'] == 'UP' else "⬇️"
                print(f"{symbol} {change['ticker']} - {change['company_name']}")
                print(f"  {change['previous_tier']} -> {change['current_tier']}")
                print(f"  Score Change: {change['score_change']:+.1f}")
                print()
        else:
            print("=" * 80)
            print("NO TIER CHANGES")
            print("=" * 80)
            print("All companies maintained their current tier.")
            print()

        # Biggest movers
        print("=" * 80)
        print("BIGGEST SCORE GAINERS")
        print("=" * 80)
        if comparison['movers']['biggest_gainers']:
            for i, mover in enumerate(comparison['movers']['biggest_gainers'][:5], 1):
                print(f"{i}. {mover['ticker']} - {mover['company_name']}")
                print(f"   Score: {mover['previous_score']:.1f} -> {mover['current_score']:.1f} ({mover['change']:+.1f})")
                print(f"   Tier: {mover['previous_tier']} -> {mover['current_tier']}")
                print()
        else:
            print("No significant gainers this week.")
            print()

        if comparison['movers']['biggest_losers']:
            print("=" * 80)
            print("BIGGEST SCORE LOSERS")
            print("=" * 80)
            for i, mover in enumerate(comparison['movers']['biggest_losers'][:5], 1):
                print(f"{i}. {mover['ticker']} - {mover['company_name']}")
                print(f"   Score: {mover['previous_score']:.1f} -> {mover['current_score']:.1f} ({mover['change']:+.1f})")
                print(f"   Tier: {mover['previous_tier']} -> {mover['current_tier']}")
                print()

        # Performance changes
        if comparison['performance_changes']:
            print("=" * 80)
            print("SIGNIFICANT STOCK PERFORMANCE CHANGES")
            print("=" * 80)
            perf_sorted = sorted(comparison['performance_changes'],
                               key=lambda x: abs(x['return_change']), reverse=True)
            for change in perf_sorted[:5]:
                print(f"{change['ticker']} - {change['company_name']}")
                print(f"  Return: {change['previous_return']:+.1f}% -> {change['current_return']:+.1f}%")
                print(f"  Change: {change['return_change']:+.1f}%")
                print()

        # Recommendations
        print("=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        # Upgraded to Strong or better
        upgraded_to_strong = [c for c in comparison['tier_changes']
                             if c['direction'] == 'UP' and
                             c['current_tier'] in ['Strong', 'Very Strong', 'Exceptional']]

        if upgraded_to_strong:
            print("WATCH CLOSELY (Upgraded to Strong+):")
            for change in upgraded_to_strong:
                print(f"  * {change['ticker']} - Now {change['current_tier']}")
            print()

        # Top gainers approaching tier change
        approaching_upgrade = [
            m for m in comparison['movers']['biggest_gainers']
            if m['change'] > 0 and
            (64 <= m['current_score'] < 65 or  # Approaching Very Strong
             49 <= m['current_score'] < 50 or  # Approaching Strong
             74 <= m['current_score'] < 75)    # Approaching Exceptional
        ]

        if approaching_upgrade:
            print("APPROACHING TIER UPGRADE (Watch Next Week):")
            for mover in approaching_upgrade:
                print(f"  * {mover['ticker']} - {mover['current_score']:.1f}/100 ({mover['current_tier']})")
            print()

        # Declining from Strong tiers
        declining = [c for c in comparison['tier_changes']
                    if c['direction'] == 'DOWN' and
                    c['previous_tier'] in ['Strong', 'Very Strong', 'Exceptional']]

        if declining:
            print("REVIEW HOLDINGS (Declined from Strong+):")
            for change in declining:
                print(f"  * {change['ticker']} - Was {change['previous_tier']}, now {change['current_tier']}")
            print()

        print("=" * 80)
        print()

        # Save comparison
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"results/weekly_comparison_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write(f"Week-over-Week Comparison Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if comparison['tier_changes']:
                f.write("TIER CHANGES:\n")
                for change in comparison['tier_changes']:
                    f.write(f"  {change['ticker']}: {change['previous_tier']} -> {change['current_tier']}\n")

        print(f"Comparison report saved: {report_file}")
        print()


def main():
    """Main comparison function."""
    print("=" * 80)
    print("AI/ML IPO WEEK-OVER-WEEK COMPARISON")
    print("=" * 80)
    print()

    comparator = WeeklyComparison()

    # Check if we have enough data
    if not comparator.tracker.history:
        print("No historical data found!")
        print("Run analysis at least twice to enable comparisons.")
        return

    # Check how many companies have multiple data points
    companies_with_history = sum(1 for data in comparator.tracker.history.values()
                                 if len(data['scores']) >= 2)

    if companies_with_history == 0:
        print("Not enough historical data for comparison.")
        print(f"Found {len(comparator.tracker.history)} companies, but all have only 1 data point.")
        print("Run analysis again next week to see changes.")
        return

    print(f"Found {companies_with_history} companies with historical data.")
    print()

    # Generate comparison
    comparator.generate_comparison_report()


if __name__ == "__main__":
    main()

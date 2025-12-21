"""
AI/ML IPO Dashboard - Quick Summary View

Generates a concise dashboard of top AI/ML IPO performers.

Usage:
    python dashboard_summary.py
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
from datetime import datetime
from pathlib import Path


def load_latest_results():
    """Load the most recent analysis results."""
    results_dir = Path('results')

    # Find most recent AI/ML analysis file
    ai_ml_files = list(results_dir.glob('ai_ml_expanded_analysis_*.txt'))

    if not ai_ml_files:
        return None

    # Get most recent
    latest_file = max(ai_ml_files, key=lambda p: p.stat().st_mtime)

    print(f"Loading data from: {latest_file.name}")
    return latest_file


def parse_report(report_file):
    """Parse the analysis report to extract key data."""
    companies = []

    with open(report_file, 'r') as f:
        content = f.read()

    # Simple parsing - look for company sections
    sections = content.split('\n\n')

    current_company = {}
    for section in sections:
        lines = section.strip().split('\n')
        if not lines:
            continue

        # Check if this is a company header
        if ' - ' in lines[0] and 'ADVANCED YC SCORE' not in lines[0]:
            # Save previous company if exists
            if current_company:
                companies.append(current_company)

            # Start new company
            parts = lines[0].split(' - ')
            if len(parts) == 2:
                current_company = {
                    'ticker': parts[0].strip(),
                    'name': parts[1].strip()
                }

        # Extract score
        for line in lines:
            if 'ADVANCED YC SCORE:' in line:
                score = line.split(':')[1].split('/')[0].strip()
                current_company['score'] = float(score)

            elif 'Market Cap:' in line and '$' in line:
                mcap = line.split('$')[1].split('B')[0].strip()
                current_company['market_cap'] = float(mcap)

            elif 'Total Return:' in line and '%' in line:
                ret = line.split(':')[1].split('%')[0].strip().replace('+', '')
                current_company['return'] = float(ret)

            elif 'Revenue Growth:' in line and 'YoY' in line:
                growth = line.split(':')[1].split('%')[0].strip()
                current_company['revenue_growth'] = float(growth)

    # Add last company
    if current_company:
        companies.append(current_company)

    return companies


def print_dashboard(companies):
    """Print formatted dashboard."""
    print("\n" + "=" * 80)
    print("AI/ML IPO UNICORN DASHBOARD")
    print("=" * 80)
    print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Sort by score
    companies.sort(key=lambda x: x.get('score', 0), reverse=True)

    # Tier breakdown
    exceptional = [c for c in companies if c.get('score', 0) >= 75]
    very_strong = [c for c in companies if 65 <= c.get('score', 0) < 75]
    strong = [c for c in companies if 50 <= c.get('score', 0) < 65]
    moderate = [c for c in companies if 35 <= c.get('score', 0) < 50]

    print(f"Total Companies: {len(companies)}")
    print(f"  Exceptional (75-100): {len(exceptional)}")
    print(f"  Very Strong (65-74):  {len(very_strong)}")
    print(f"  Strong (50-64):       {len(strong)}")
    print(f"  Moderate (35-49):     {len(moderate)}")
    print()

    # Top performers
    print("=" * 80)
    print("TOP PERFORMERS")
    print("=" * 80)
    print()

    for i, company in enumerate(companies[:7], 1):
        score = company.get('score', 0)
        tier = "EXCEPTIONAL" if score >= 75 else \
               "VERY STRONG" if score >= 65 else \
               "STRONG" if score >= 50 else "MODERATE"

        print(f"#{i} {company['ticker']:6} - {tier:12} - {score:.1f}/100")
        print(f"    {company.get('name', 'Unknown')}")
        print(f"    Market Cap: ${company.get('market_cap', 0):.1f}B | " +
              f"Return: {company.get('return', 0):+.0f}% | " +
              f"Revenue Growth: {company.get('revenue_growth', 0):.0f}%")
        print()

    # Quick stats
    print("=" * 80)
    print("QUICK STATS")
    print("=" * 80)
    print()

    if companies:
        avg_score = sum(c.get('score', 0) for c in companies) / len(companies)
        avg_return = sum(c.get('return', 0) for c in companies if 'return' in c) / max(len([c for c in companies if 'return' in c]), 1)
        best_return = max((c.get('return', 0) for c in companies), default=0)
        best_growth = max((c.get('revenue_growth', 0) for c in companies), default=0)

        print(f"Average Score: {avg_score:.1f}/100")
        print(f"Average Return Since IPO: {avg_return:+.1f}%")
        print(f"Best Return: {best_return:+.1f}%")
        print(f"Highest Revenue Growth: {best_growth:.1f}% YoY")

    print()

    # Investment focus
    print("=" * 80)
    print("INVESTMENT FOCUS")
    print("=" * 80)
    print()

    if exceptional:
        print("EXCEPTIONAL OPPORTUNITIES:")
        for c in exceptional:
            print(f"  * {c['ticker']} - {c.get('score', 0):.1f}/100")
    else:
        print("EXCEPTIONAL: None (watch for companies crossing 75)")

    print()

    if very_strong:
        print("VERY STRONG OPPORTUNITIES:")
        for c in very_strong:
            print(f"  * {c['ticker']} - {c.get('score', 0):.1f}/100")

    print()

    # Growth leaders
    growth_leaders = sorted(companies, key=lambda x: x.get('revenue_growth', 0), reverse=True)[:3]
    if growth_leaders:
        print("FASTEST GROWING:")
        for c in growth_leaders:
            print(f"  * {c['ticker']} - {c.get('revenue_growth', 0):.1f}% YoY")

    print()
    print("=" * 80)


def main():
    """Main dashboard function."""
    print("=" * 80)
    print("AI/ML IPO DASHBOARD GENERATOR")
    print("=" * 80)
    print()

    # Load latest results
    latest_file = load_latest_results()

    if not latest_file:
        print("No analysis results found!")
        print("Run: python analyze_ai_ml_ipos_expanded.py")
        return

    print()

    # Parse report
    companies = parse_report(latest_file)

    if not companies:
        print("Could not parse company data from report")
        return

    print(f"Found {len(companies)} companies in report")
    print()

    # Display dashboard
    print_dashboard(companies)

    # Save dashboard snapshot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot_file = f"results/dashboard_snapshot_{timestamp}.txt"

    with open(snapshot_file, 'w') as f:
        f.write(f"AI/ML IPO Dashboard Snapshot\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nTop Performers:\n")
        for i, c in enumerate(companies[:10], 1):
            f.write(f"{i}. {c['ticker']} - {c.get('score', 0):.1f}/100\n")

    print(f"\nDashboard snapshot saved: {snapshot_file}")
    print()


if __name__ == "__main__":
    main()

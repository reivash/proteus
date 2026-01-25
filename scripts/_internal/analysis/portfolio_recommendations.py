"""
AI/ML IPO Portfolio Recommendation Engine

Generates personalized portfolio recommendations based on:
- Risk tolerance
- Investment horizon
- Diversification preferences
- Score tiers

Usage:
    python portfolio_recommendations.py
"""

import sys
sys.path.insert(0, 'src')

from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List


class PortfolioRecommender:
    """Generate AI/ML IPO portfolio recommendations."""

    def __init__(self, history_file: str = "data/score_history.json"):
        """Initialize recommender with score history."""
        self.history_file = Path(history_file)
        self.companies = self._load_latest_scores()

    def _load_latest_scores(self) -> List[Dict]:
        """Load latest scores for all companies."""
        if not self.history_file.exists():
            return []

        with open(self.history_file, 'r') as f:
            history = json.load(f)

        companies = []
        for ticker, data in history.items():
            if data['scores']:
                latest = data['scores'][-1]
                companies.append({
                    'ticker': ticker,
                    'company_name': data['company_name'],
                    **latest
                })

        return companies

    def get_tier(self, score: float) -> str:
        """Get tier for a score."""
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

    def recommend_conservative(self, budget: float = 100000) -> Dict:
        """
        Conservative portfolio for risk-averse investors.

        Focus: Exceptional and Very Strong tiers only
        Diversification: Balanced across 3-5 companies
        Risk: Low to moderate

        Args:
            budget: Total investment amount

        Returns:
            Portfolio recommendation
        """
        # Filter for top tiers
        top_companies = [c for c in self.companies
                        if c['advanced_yc_score'] >= 65]

        if not top_companies:
            return {'error': 'No Exceptional or Very Strong companies available'}

        # Sort by score
        top_companies.sort(key=lambda x: x['advanced_yc_score'], reverse=True)

        # Allocate budget
        portfolio = []
        num_holdings = min(5, len(top_companies))

        # Weighted allocation based on score
        total_score = sum(c['advanced_yc_score'] for c in top_companies[:num_holdings])

        for company in top_companies[:num_holdings]:
            weight = company['advanced_yc_score'] / total_score
            allocation = budget * weight

            portfolio.append({
                'ticker': company['ticker'],
                'company_name': company['company_name'],
                'tier': self.get_tier(company['advanced_yc_score']),
                'score': company['advanced_yc_score'],
                'allocation': allocation,
                'weight': weight * 100
            })

        return {
            'strategy': 'Conservative',
            'risk_level': 'Low-Moderate',
            'portfolio': portfolio,
            'total_allocation': sum(p['allocation'] for p in portfolio),
            'num_holdings': num_holdings
        }

    def recommend_balanced(self, budget: float = 100000) -> Dict:
        """
        Balanced portfolio for moderate risk tolerance.

        Focus: Mix of Very Strong, Strong, and select Moderate
        Diversification: 5-8 companies
        Risk: Moderate

        Args:
            budget: Total investment amount

        Returns:
            Portfolio recommendation
        """
        # Core holdings (70%): Very Strong + Strong
        core = [c for c in self.companies if c['advanced_yc_score'] >= 50]
        core.sort(key=lambda x: x['advanced_yc_score'], reverse=True)
        core = core[:4]  # Top 4

        # Growth holdings (30%): Top Moderate
        growth = [c for c in self.companies
                 if 35 <= c['advanced_yc_score'] < 50]
        growth.sort(key=lambda x: x.get('revenue_growth_yoy', 0), reverse=True)
        growth = growth[:2]  # Top 2 by growth

        portfolio = []

        # Allocate core (70%)
        core_budget = budget * 0.70
        for company in core:
            allocation = core_budget / len(core)
            portfolio.append({
                'ticker': company['ticker'],
                'company_name': company['company_name'],
                'tier': self.get_tier(company['advanced_yc_score']),
                'score': company['advanced_yc_score'],
                'allocation': allocation,
                'weight': (allocation / budget) * 100,
                'category': 'Core'
            })

        # Allocate growth (30%)
        if growth:
            growth_budget = budget * 0.30
            for company in growth:
                allocation = growth_budget / len(growth)
                portfolio.append({
                    'ticker': company['ticker'],
                    'company_name': company['company_name'],
                    'tier': self.get_tier(company['advanced_yc_score']),
                    'score': company['advanced_yc_score'],
                    'revenue_growth': company.get('revenue_growth_yoy', 0) * 100,
                    'allocation': allocation,
                    'weight': (allocation / budget) * 100,
                    'category': 'Growth'
                })

        return {
            'strategy': 'Balanced',
            'risk_level': 'Moderate',
            'portfolio': portfolio,
            'total_allocation': sum(p['allocation'] for p in portfolio),
            'num_holdings': len(portfolio)
        }

    def recommend_aggressive(self, budget: float = 100000) -> Dict:
        """
        Aggressive portfolio for high risk tolerance.

        Focus: High growth potential, including Strong and Moderate
        Diversification: 6-10 companies
        Risk: High

        Args:
            budget: Total investment amount

        Returns:
            Portfolio recommendation
        """
        # Core (40%): Top performers
        core = [c for c in self.companies if c['advanced_yc_score'] >= 55]
        core.sort(key=lambda x: x['advanced_yc_score'], reverse=True)
        core = core[:3]

        # Growth (40%): Highest revenue growth
        growth = [c for c in self.companies
                 if c['advanced_yc_score'] >= 35]
        growth.sort(key=lambda x: x.get('revenue_growth_yoy', 0), reverse=True)
        growth = [g for g in growth if g not in core][:3]

        # Speculative (20%): Undervalued or turnaround candidates
        speculative = [c for c in self.companies
                      if c['advanced_yc_score'] >= 25]
        speculative.sort(key=lambda x: x.get('total_return_pct', -999), reverse=True)
        speculative = [s for s in speculative if s not in core and s not in growth][:2]

        portfolio = []

        # Allocate core (40%)
        core_budget = budget * 0.40
        for company in core:
            portfolio.append({
                'ticker': company['ticker'],
                'company_name': company['company_name'],
                'tier': self.get_tier(company['advanced_yc_score']),
                'score': company['advanced_yc_score'],
                'allocation': core_budget / len(core),
                'weight': 40 / len(core),
                'category': 'Core'
            })

        # Allocate growth (40%)
        if growth:
            growth_budget = budget * 0.40
            for company in growth:
                portfolio.append({
                    'ticker': company['ticker'],
                    'company_name': company['company_name'],
                    'tier': self.get_tier(company['advanced_yc_score']),
                    'score': company['advanced_yc_score'],
                    'revenue_growth': company.get('revenue_growth_yoy', 0) * 100,
                    'allocation': growth_budget / len(growth),
                    'weight': 40 / len(growth),
                    'category': 'Growth'
                })

        # Allocate speculative (20%)
        if speculative:
            spec_budget = budget * 0.20
            for company in speculative:
                portfolio.append({
                    'ticker': company['ticker'],
                    'company_name': company['company_name'],
                    'tier': self.get_tier(company['advanced_yc_score']),
                    'score': company['advanced_yc_score'],
                    'allocation': spec_budget / len(speculative),
                    'weight': 20 / len(speculative),
                    'category': 'Speculative'
                })

        return {
            'strategy': 'Aggressive',
            'risk_level': 'High',
            'portfolio': portfolio,
            'total_allocation': sum(p['allocation'] for p in portfolio),
            'num_holdings': len(portfolio)
        }

    def print_portfolio(self, recommendation: Dict):
        """Print formatted portfolio recommendation."""
        print("=" * 80)
        print(f"{recommendation['strategy'].upper()} PORTFOLIO RECOMMENDATION")
        print("=" * 80)
        print(f"Risk Level: {recommendation['risk_level']}")
        print(f"Total Allocation: ${recommendation['total_allocation']:,.2f}")
        print(f"Number of Holdings: {recommendation['num_holdings']}")
        print()

        if 'error' in recommendation:
            print(f"ERROR: {recommendation['error']}")
            return

        # Group by category if exists
        categories = {}
        for holding in recommendation['portfolio']:
            category = holding.get('category', 'Holdings')
            if category not in categories:
                categories[category] = []
            categories[category].append(holding)

        for category, holdings in categories.items():
            if len(categories) > 1:
                print(f"\n{category.upper()} HOLDINGS:")
                print("-" * 80)

            for i, holding in enumerate(holdings, 1):
                print(f"{i}. {holding['ticker']} - {holding['company_name']}")
                print(f"   Tier: {holding['tier']} ({holding['score']:.1f}/100)")
                print(f"   Allocation: ${holding['allocation']:,.2f} ({holding['weight']:.1f}%)")

                if 'revenue_growth' in holding:
                    print(f"   Revenue Growth: {holding['revenue_growth']:.1f}% YoY")

                print()

        print("=" * 80)


def main():
    """Main portfolio recommendation function."""
    print("=" * 80)
    print("AI/ML IPO PORTFOLIO RECOMMENDATION ENGINE")
    print("=" * 80)
    print()

    recommender = PortfolioRecommender()

    if not recommender.companies:
        print("No company data available!")
        print("Run analysis first: python analyze_ai_ml_ipos_expanded.py")
        return

    print(f"Loaded {len(recommender.companies)} companies from database")
    print()

    # Get budget
    print("Enter your investment budget (default: $100,000):")
    budget_input = input("Budget ($): ").strip()

    if budget_input:
        try:
            budget = float(budget_input.replace(',', '').replace('$', ''))
        except:
            budget = 100000
    else:
        budget = 100000

    print()

    # Generate all three portfolios
    print("\n")
    conservative = recommender.recommend_conservative(budget)
    recommender.print_portfolio(conservative)

    print("\n")
    balanced = recommender.recommend_balanced(budget)
    recommender.print_portfolio(balanced)

    print("\n")
    aggressive = recommender.recommend_aggressive(budget)
    recommender.print_portfolio(aggressive)

    # Save recommendations
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"results/portfolio_recommendations_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump({
            'date': datetime.now().isoformat(),
            'budget': budget,
            'conservative': conservative,
            'balanced': balanced,
            'aggressive': aggressive
        }, f, indent=2)

    print()
    print(f"Recommendations saved: {filename}")
    print()
    print("=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Review each portfolio based on your risk tolerance")
    print("2. Research individual companies in detail")
    print("3. Consider your investment timeline (long-term recommended)")
    print("4. Dollar-cost average into positions over time")
    print("5. Rebalance quarterly based on score changes")
    print()


if __name__ == "__main__":
    main()

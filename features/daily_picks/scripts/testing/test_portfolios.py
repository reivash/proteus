"""
Quick test of portfolio recommendation engine.
"""

import sys
sys.path.insert(0, 'src')

from portfolio_recommendations import PortfolioRecommender

def main():
    print("=" * 80)
    print("PORTFOLIO RECOMMENDATION TEST")
    print("=" * 80)
    print()

    recommender = PortfolioRecommender()

    if not recommender.companies:
        print("No company data available!")
        return

    print(f"Loaded {len(recommender.companies)} companies")
    print()

    # Test with $100,000 budget
    budget = 100000

    print(f"Testing portfolios with ${budget:,.0f} budget")
    print()

    # Conservative
    conservative = recommender.recommend_conservative(budget)
    recommender.print_portfolio(conservative)

    print("\n")

    # Balanced
    balanced = recommender.recommend_balanced(budget)
    recommender.print_portfolio(balanced)

    print("\n")

    # Aggressive
    aggressive = recommender.recommend_aggressive(budget)
    recommender.print_portfolio(aggressive)

    print()
    print("=" * 80)
    print("PORTFOLIO COMPARISON")
    print("=" * 80)
    print()
    print(f"Conservative: {conservative.get('num_holdings', 0)} holdings, Risk: {conservative.get('risk_level', 'N/A')}")
    print(f"Balanced:     {balanced.get('num_holdings', 0)} holdings, Risk: {balanced.get('risk_level', 'N/A')}")
    print(f"Aggressive:   {aggressive.get('num_holdings', 0)} holdings, Risk: {aggressive.get('risk_level', 'N/A')}")
    print()


if __name__ == "__main__":
    main()

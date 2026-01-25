"""
Competitive Positioning Analyzer

Deep competitive analysis for IPO companies vs. peers.
Identifies market leaders, niche dominators, and "10x better" candidates.

Y Combinator Principle: "10x better, not 10% better"
- How does this company compare to alternatives?
- Is it monopolizing a small market?
- What makes it defensible?

Analysis Framework:
1. Peer Comparison (same industry/category)
2. Market Share Estimation
3. Competitive Moat Assessment
4. "10x Better" Validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from datetime import datetime, timedelta


class CompetitivePositioningAnalyzer:
    """
    Analyzes competitive positioning using peer comparison.
    """

    def __init__(self):
        """Initialize competitive positioning analyzer."""
        # Industry peer groups for comparison
        self.peer_groups = {
            'Social Media': ['META', 'SNAP', 'PINS', 'RDDT'],
            'Cloud Infrastructure': ['SNOW', 'DDOG', 'MDB', 'NET'],
            'Cybersecurity': ['CRWD', 'ZS', 'PANW', 'OKTA'],
            'Fintech': ['SQ', 'COIN', 'HOOD', 'BILL', 'PYPL'],
            'E-commerce': ['SHOP', 'AMZN', 'ETSY'],
            'Gaming': ['RBLX', 'U', 'EA', 'TTWO'],
            'Food Delivery': ['DASH', 'UBER'],
            'EV': ['RIVN', 'TSLA', 'LCID'],
            'Biotech': ['ASND', 'MRNA', 'BNTX', 'NVAX'],
            'Developer Tools': ['GTLB', 'PATH', 'TEAM', 'ATLASSIAN'],
            'Data Analytics': ['PLTR', 'SNOW', 'DDOG'],
            'AI/ML': ['AI', 'PLTR', 'SNOW'],
            'Payments': ['SQ', 'PYPL', 'MA', 'V'],
            'Cloud Security': ['CRWD', 'ZS', 'PANW'],
        }

    def identify_peer_group(self, ticker: str, category: str) -> List[str]:
        """
        Identify peer group for comparison.

        Args:
            ticker: Stock ticker
            category: Company category

        Returns:
            List of peer tickers
        """
        # Check predefined peer groups
        for group_name, peers in self.peer_groups.items():
            if ticker in peers:
                return [p for p in peers if p != ticker]  # Exclude self

        # Fallback: extract from category
        category_lower = category.lower()
        for group_name, peers in self.peer_groups.items():
            if group_name.lower() in category_lower or category_lower in group_name.lower():
                return peers

        return []

    def get_peer_metrics(self, peers: List[str]) -> pd.DataFrame:
        """
        Fetch comparable metrics for peer group.

        Args:
            peers: List of peer tickers

        Returns:
            DataFrame with peer metrics
        """
        peer_data = []

        for peer in peers:
            try:
                stock = yf.Ticker(peer)
                info = stock.info

                peer_data.append({
                    'ticker': peer,
                    'name': info.get('longName', peer),
                    'market_cap': info.get('marketCap', 0),
                    'market_cap_b': info.get('marketCap', 0) / 1e9 if info.get('marketCap') else 0,
                    'revenue': info.get('totalRevenue', 0),
                    'revenue_b': info.get('totalRevenue', 0) / 1e9 if info.get('totalRevenue') else 0,
                    'revenue_growth': info.get('revenueGrowth', 0) or 0,
                    'gross_margin': info.get('grossMargins', 0) or 0,
                    'operating_margin': info.get('operatingMargins', 0) or 0,
                    'profit_margin': info.get('profitMargins', 0) or 0,
                    'pe_ratio': info.get('trailingPE', 0) or 0,
                    'price_to_sales': info.get('priceToSalesTrailing12Months', 0) or 0,
                    'employees': info.get('fullTimeEmployees', 1) or 1,
                })

            except Exception as e:
                print(f"[WARN] Could not fetch peer data for {peer}: {e}")
                continue

        return pd.DataFrame(peer_data)

    def calculate_market_share(self, ticker: str, company_metrics: Dict,
                              peer_df: pd.DataFrame) -> Dict:
        """
        Estimate market share within peer group.

        Market share proxy = company revenue / total peer group revenue

        Args:
            ticker: Company ticker
            company_metrics: Company's metrics
            peer_df: Peer group metrics

        Returns:
            Dictionary with market share metrics
        """
        company_revenue = company_metrics.get('revenue', 0)

        # Calculate total addressable market (peer group sum)
        total_peer_revenue = peer_df['revenue'].sum()
        total_market = total_peer_revenue + company_revenue

        # Market share
        market_share = company_revenue / total_market if total_market > 0 else 0

        # Rank by revenue
        all_revenues = list(peer_df['revenue']) + [company_revenue]
        rank = sorted(all_revenues, reverse=True).index(company_revenue) + 1

        return {
            'market_share_pct': market_share * 100,
            'revenue_rank': rank,
            'total_peers': len(peer_df) + 1,
            'is_market_leader': rank == 1,
            'peer_group_revenue_b': total_peer_revenue / 1e9,
            'total_market_b': total_market / 1e9,
        }

    def calculate_competitive_moat(self, company_metrics: Dict,
                                   peer_df: pd.DataFrame) -> Dict:
        """
        Calculate competitive moat vs. peers.

        Moat indicators:
        - Margin advantage (higher margins = pricing power)
        - Growth premium (faster growth = winning market share)
        - Efficiency advantage (revenue per employee)
        - Valuation premium (market expects dominance)

        Args:
            company_metrics: Company's metrics
            peer_df: Peer group metrics

        Returns:
            Moat analysis dictionary
        """
        # Get peer averages
        peer_avg = {
            'gross_margin': peer_df['gross_margin'].mean(),
            'operating_margin': peer_df['operating_margin'].mean(),
            'revenue_growth': peer_df['revenue_growth'].mean(),
            'price_to_sales': peer_df['price_to_sales'].mean(),
        }

        # Calculate advantages
        gross_margin_advantage = (company_metrics.get('gross_margin', 0) -
                                 peer_avg['gross_margin']) * 100

        operating_margin_advantage = (company_metrics.get('operating_margin', 0) -
                                     peer_avg['operating_margin']) * 100

        growth_premium = (company_metrics.get('revenue_growth', 0) -
                         peer_avg['revenue_growth']) * 100

        valuation_premium = (company_metrics.get('price_to_sales', 0) -
                           peer_avg['price_to_sales'])

        # Calculate moat score (0-100)
        moat_score = 0.0

        # Margin advantage (40 points)
        if gross_margin_advantage > 20:  # >20pp advantage
            moat_score += 25
        elif gross_margin_advantage > 10:  # >10pp advantage
            moat_score += 15
        elif gross_margin_advantage > 0:  # Any advantage
            moat_score += 5

        if operating_margin_advantage > 10:
            moat_score += 15
        elif operating_margin_advantage > 5:
            moat_score += 10
        elif operating_margin_advantage > 0:
            moat_score += 5

        # Growth premium (30 points)
        if growth_premium > 30:  # Growing 30pp faster than peers
            moat_score += 30
        elif growth_premium > 15:
            moat_score += 20
        elif growth_premium > 0:
            moat_score += 10

        # Valuation premium (30 points) - market expects dominance
        if valuation_premium > 5:  # Much higher P/S = market expects dominance
            moat_score += 30
        elif valuation_premium > 2:
            moat_score += 20
        elif valuation_premium > 0:
            moat_score += 10

        return {
            'moat_score': min(100, moat_score),
            'gross_margin_advantage_pp': gross_margin_advantage,
            'operating_margin_advantage_pp': operating_margin_advantage,
            'growth_premium_pp': growth_premium,
            'valuation_premium_x': valuation_premium,
            'peer_avg_gross_margin': peer_avg['gross_margin'],
            'peer_avg_revenue_growth': peer_avg['revenue_growth'],
            'has_margin_advantage': gross_margin_advantage > 0,
            'has_growth_advantage': growth_premium > 0,
            'has_valuation_premium': valuation_premium > 0,
        }

    def assess_10x_better(self, company_metrics: Dict, peer_df: pd.DataFrame) -> Dict:
        """
        Assess if company is "10x better" than alternatives.

        Y Combinator principle: Build something 10x better, not 10% better.

        Indicators:
        - 10x faster growth than peers
        - 10x better margins
        - 10x higher efficiency
        - Clear differentiation

        Args:
            company_metrics: Company's metrics
            peer_df: Peer group metrics

        Returns:
            10x assessment dictionary
        """
        signals = []
        score = 0.0

        # Growth differential
        company_growth = company_metrics.get('revenue_growth', 0)
        peer_median_growth = peer_df['revenue_growth'].median()

        if peer_median_growth > 0:
            growth_multiple = company_growth / peer_median_growth
            if growth_multiple >= 2.0:  # 2x faster growth
                signals.append(f"Growing {growth_multiple:.1f}x faster than peers")
                score += 30

        # Margin superiority
        company_gross_margin = company_metrics.get('gross_margin', 0)
        peer_median_gross_margin = peer_df['gross_margin'].median()

        if peer_median_gross_margin > 0:
            margin_ratio = company_gross_margin / peer_median_gross_margin
            if margin_ratio >= 1.3:  # 30%+ better margins
                signals.append(f"Margins {margin_ratio:.1%} of peers (pricing power)")
                score += 25

        # Efficiency
        company_revenue = company_metrics.get('revenue', 0)
        company_employees = company_metrics.get('employees', 1) or 1
        company_rev_per_emp = company_revenue / company_employees

        peer_df['rev_per_employee'] = peer_df['revenue'] / peer_df['employees'].replace(0, 1)
        peer_median_rev_per_emp = peer_df['rev_per_employee'].median()

        if peer_median_rev_per_emp > 0:
            efficiency_ratio = company_rev_per_emp / peer_median_rev_per_emp
            if efficiency_ratio >= 1.5:  # 50%+ more efficient
                signals.append(f"Revenue per employee {efficiency_ratio:.1f}x peers (leverage)")
                score += 20

        # Market cap efficiency (if outperforming with smaller market cap)
        company_market_cap = company_metrics.get('market_cap', 0)
        peer_median_market_cap = peer_df['market_cap'].median()

        if company_market_cap > 0 and peer_median_market_cap > 0:
            if company_growth > peer_median_growth and company_market_cap < peer_median_market_cap:
                signals.append("Outperforming with smaller market cap (undervalued)")
                score += 25

        is_10x_better = score >= 60

        return {
            'is_10x_better': is_10x_better,
            '10x_score': min(100, score),
            '10x_signals': signals,
            'growth_vs_peers': f"{company_growth:.1%} vs {peer_median_growth:.1%}",
            'margin_vs_peers': f"{company_gross_margin:.1%} vs {peer_median_gross_margin:.1%}",
        }

    def analyze_competitive_position(self, ticker: str, company_info: Dict,
                                    fundamentals: Dict, category: str) -> Dict:
        """
        Complete competitive positioning analysis.

        Args:
            ticker: Stock ticker
            company_info: Company information
            fundamentals: Fundamental metrics
            category: Company category

        Returns:
            Comprehensive competitive analysis
        """
        print(f"\n[COMPETITIVE ANALYSIS] {ticker} in {category}")
        print("-" * 70)

        # Identify peers
        peers = self.identify_peer_group(ticker, category)

        if not peers:
            print(f"  No peer group identified for {ticker}")
            return {
                'has_peers': False,
                'ticker': ticker,
                'category': category,
            }

        print(f"  Peer Group ({len(peers)}): {', '.join(peers)}")

        # Get peer metrics
        peer_df = self.get_peer_metrics(peers)

        if peer_df.empty:
            print(f"  Could not fetch peer data")
            return {
                'has_peers': False,
                'ticker': ticker,
                'category': category,
            }

        # Combine company metrics
        company_metrics = {
            **company_info,
            **fundamentals,
        }

        # Market share analysis
        market_share = self.calculate_market_share(ticker, company_metrics, peer_df)
        print(f"  Market Share: {market_share['market_share_pct']:.1f}% "
              f"(Rank #{market_share['revenue_rank']} of {market_share['total_peers']})")

        # Competitive moat
        moat = self.calculate_competitive_moat(company_metrics, peer_df)
        print(f"  Competitive Moat Score: {moat['moat_score']:.1f}/100")
        print(f"    - Gross Margin Advantage: {moat['gross_margin_advantage_pp']:+.1f}pp")
        print(f"    - Growth Premium: {moat['growth_premium_pp']:+.1f}pp")

        # 10x better assessment
        ten_x = self.assess_10x_better(company_metrics, peer_df)
        print(f"  10x Better: {ten_x['is_10x_better']} (Score: {ten_x['10x_score']:.1f}/100)")
        if ten_x['10x_signals']:
            for signal in ten_x['10x_signals']:
                print(f"    - {signal}")

        return {
            'has_peers': True,
            'ticker': ticker,
            'category': category,
            'peer_count': len(peers),
            'peers': peers,
            **market_share,
            **moat,
            **ten_x,
        }


if __name__ == "__main__":
    # Test competitive positioning analyzer
    import sys
    sys.path.insert(0, '../../..')
    from common.data.fetchers.ipo_data_fetcher import IPODataFetcher

    analyzer = CompetitivePositioningAnalyzer()
    fetcher = IPODataFetcher()

    # Test with RDDT (Reddit)
    ticker = 'RDDT'
    category = 'Social Media'

    print("=" * 70)
    print("COMPETITIVE POSITIONING ANALYSIS")
    print("=" * 70)

    # Get company data
    company_info = fetcher.get_company_info(ticker)
    fundamentals = fetcher.get_fundamental_metrics(ticker)

    # Analyze
    analysis = analyzer.analyze_competitive_position(
        ticker, company_info, fundamentals, category
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Ticker: {ticker}")
    print(f"Market Share: {analysis.get('market_share_pct', 0):.1f}%")
    print(f"Market Leader: {analysis.get('is_market_leader', False)}")
    print(f"Competitive Moat: {analysis.get('moat_score', 0):.1f}/100")
    print(f"10x Better: {analysis.get('is_10x_better', False)}")

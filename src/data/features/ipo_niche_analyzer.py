"""
IPO Niche Market Analyzer

Identifies companies dominating small, fast-growing niche markets.
Applies Y Combinator principles to find breakout IPO candidates.

Y Combinator Principles:
1. Start with a small niche market you can dominate (monopoly power)
2. Find markets that are small today but growing explosively (category emergence)
3. Build something 10x better than alternatives (competitive moat)
4. "Do things that don't scale" graduating to scale (inflection points)

Detection Signals:
- Market dominance in small categories (>30% market share in <$10B TAM)
- Explosive category growth (>50% annual growth rate)
- Strong competitive moats (high margins, network effects, switching costs)
- Scaling inflection (accelerating revenue while maintaining margins)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf


class IPONicheAnalyzer:
    """
    Analyzes IPO companies for niche market dominance signals.
    """

    def __init__(self):
        """Initialize the niche market analyzer."""
        # Industry size estimates (rough TAM in billions USD)
        # Small markets (<$50B) with high growth potential
        self.niche_industries = {
            # Cloud/SaaS niches
            'Data Infrastructure': 15.0,
            'Cybersecurity Services': 30.0,
            'Developer Tools': 10.0,
            'Marketing Automation': 8.0,
            'Business Intelligence': 25.0,

            # Consumer niches
            'Social Media Platforms': 40.0,
            'Gaming Platforms': 35.0,
            'Food Delivery': 30.0,
            'Ride Sharing': 25.0,

            # Fintech niches
            'Cryptocurrency Exchanges': 15.0,
            'Digital Payments': 45.0,
            'Buy Now Pay Later': 12.0,
            'Neo Banking': 20.0,

            # Emerging tech
            'Electric Vehicles': 40.0,
            'Renewable Energy': 35.0,
            'Biotechnology': 30.0,
            'AI/ML Platforms': 20.0,
            'Quantum Computing': 5.0,

            # Other emerging categories
            'Space Technology': 10.0,
            'EdTech Platforms': 15.0,
            'HealthTech': 25.0,
            'PropTech': 18.0,
        }

        # High-growth industry indicators
        self.high_growth_keywords = [
            'cloud', 'saas', 'ai', 'machine learning', 'platform',
            'marketplace', 'data', 'cybersecurity', 'fintech',
            'crypto', 'blockchain', 'electric', 'renewable',
            'biotech', 'gene', 'space', 'quantum', 'automation'
        ]

    def identify_niche_category(self, company_info: Dict) -> Tuple[str, float, bool]:
        """
        Identify if company operates in a niche category.

        Returns:
            Tuple of (category_name, estimated_tam_billions, is_niche)
        """
        industry = company_info.get('industry', '').lower()
        description = company_info.get('description', '').lower()

        # Check against known niche industries
        for niche, tam in self.niche_industries.items():
            if niche.lower() in industry or niche.lower() in description:
                return (niche, tam, True)

        # Check for high-growth keywords
        for keyword in self.high_growth_keywords:
            if keyword in industry or keyword in description:
                # Estimate TAM based on sector
                estimated_tam = 25.0  # Default estimate for emerging categories
                return (f"Emerging: {keyword.title()}", estimated_tam, True)

        # Not a clear niche
        return (company_info.get('industry', 'General'), 100.0, False)

    def calculate_market_dominance_score(self, company_info: Dict, fundamentals: Dict) -> float:
        """
        Calculate market dominance score (0-100).

        Factors:
        - Relative market cap vs. category size (market share proxy)
        - Revenue growth rate (winning market share)
        - Margins (pricing power/monopoly indicator)
        - Market position (is this a clear leader?)

        Args:
            company_info: Company information dict
            fundamentals: Fundamental metrics dict

        Returns:
            Dominance score 0-100 (higher = more dominant)
        """
        score = 0.0

        # Get market cap and estimate market share
        market_cap_b = company_info.get('market_cap_b', 0)
        category, tam, is_niche = self.identify_niche_category(company_info)

        # 1. Market share indicator (40 points)
        # If company market cap is significant portion of TAM, shows dominance
        if tam > 0:
            market_share_proxy = min(market_cap_b / tam, 1.0)  # Cap at 100%
            score += market_share_proxy * 40

        # 2. Revenue growth (30 points)
        # Fast growth = winning market share
        revenue_growth = fundamentals.get('revenue_growth_yoy', 0) or 0
        if revenue_growth > 0:
            # Scale: 0% = 0 pts, 50%+ = 30 pts
            growth_score = min(revenue_growth / 0.5, 1.0) * 30
            score += growth_score

        # 3. Profitability/Margins (20 points)
        # High margins indicate pricing power (monopoly-like)
        gross_margin = fundamentals.get('gross_margin', 0) or 0
        operating_margin = fundamentals.get('operating_margin', 0) or 0

        # Strong gross margin (>60%) = high moat
        if gross_margin > 0.6:
            score += 10
        elif gross_margin > 0.4:
            score += 5

        # Positive operating margin = profitable scaling
        if operating_margin > 0.15:
            score += 10
        elif operating_margin > 0:
            score += 5

        # 4. Niche focus bonus (10 points)
        # Extra points for clear niche positioning
        if is_niche and tam < 30:  # Small, focused market
            score += 10

        return min(100, max(0, score))

    def detect_category_emergence(self, company_info: Dict, fundamentals: Dict) -> Dict:
        """
        Detect if company is in an emerging/exploding category.

        Signals:
        - High revenue growth (category expanding)
        - Young company (early in category lifecycle)
        - High valuation multiples (market expects explosive growth)
        - Recent IPO (category reaching mainstream)

        Returns:
            Dictionary with emergence signals
        """
        category, tam, is_niche = self.identify_niche_category(company_info)

        # Revenue growth rate
        revenue_growth = fundamentals.get('revenue_growth_yoy', 0) or 0

        # Company age (months since IPO)
        months_since_ipo = company_info.get('months_since_ipo', 999)

        # Valuation multiples (high P/S = high growth expectations)
        price_to_sales = fundamentals.get('price_to_sales', 0) or 0

        # Emergence score
        emergence_score = 0.0

        # Very high revenue growth (>50%) = emerging category
        if revenue_growth > 0.5:
            emergence_score += 40

        # Recent IPO (<18 months) = category going mainstream
        if months_since_ipo < 18:
            emergence_score += 30

        # High valuation (P/S > 10) = market expects explosive growth
        if price_to_sales > 10:
            emergence_score += 20

        # Niche focus
        if is_niche:
            emergence_score += 10

        return {
            'category': category,
            'estimated_tam_billions': tam,
            'is_niche': is_niche,
            'revenue_growth_yoy': revenue_growth,
            'months_since_ipo': months_since_ipo,
            'price_to_sales': price_to_sales,
            'emergence_score': min(100, emergence_score),
            'is_emerging': emergence_score >= 60,
        }

    def calculate_10x_moat_score(self, company_info: Dict, fundamentals: Dict) -> float:
        """
        Calculate "10x better" moat score.

        Y Combinator principle: Build something 10x better, not 10% better.

        Indicators:
        - High gross margins (unique value prop = pricing power)
        - Strong revenue per employee (efficiency/leverage)
        - High customer retention implied by stable margins
        - Network effects (marketplace/platform businesses)

        Returns:
            Moat score 0-100
        """
        score = 0.0

        # 1. Gross margin (40 points) - pricing power
        gross_margin = fundamentals.get('gross_margin', 0) or 0
        if gross_margin > 0.8:  # 80%+ = software/network business
            score += 40
        elif gross_margin > 0.6:  # 60%+ = strong moat
            score += 30
        elif gross_margin > 0.4:  # 40%+ = decent moat
            score += 15

        # 2. Operating leverage (30 points) - scaling efficiency
        operating_margin = fundamentals.get('operating_margin', 0) or 0
        if operating_margin > 0.2:  # 20%+ operating margin = strong leverage
            score += 30
        elif operating_margin > 0.1:  # 10%+ = developing leverage
            score += 20
        elif operating_margin > 0:  # Profitable = some leverage
            score += 10

        # 3. Revenue per employee (20 points) - efficiency
        revenue = company_info.get('revenue', 0) or 0
        employees = company_info.get('employees', 1) or 1
        revenue_per_employee = revenue / employees if employees > 0 else 0

        # Software/platform companies can have $500K-$2M+ per employee
        if revenue_per_employee > 1_000_000:  # $1M+ per employee
            score += 20
        elif revenue_per_employee > 500_000:  # $500K+ per employee
            score += 15
        elif revenue_per_employee > 250_000:  # $250K+ per employee
            score += 10

        # 4. Platform/network effects (10 points) - check description
        description = company_info.get('description', '').lower()
        network_keywords = ['platform', 'marketplace', 'network', 'community', 'ecosystem']
        if any(keyword in description for keyword in network_keywords):
            score += 10

        return min(100, max(0, score))

    def detect_scaling_inflection(self, ticker: str, company_info: Dict, fundamentals: Dict) -> Dict:
        """
        Detect if company is at a "doing things that don't scale" → scaling inflection.

        Signals:
        - Revenue growth acceleration
        - Improving margins (economics improving at scale)
        - Recent profitability inflection
        - Operating leverage kicking in

        Returns:
            Dictionary with scaling signals
        """
        signals = {
            'is_at_inflection': False,
            'growth_accelerating': False,
            'margins_improving': False,
            'recently_profitable': False,
            'scaling_score': 0.0,
        }

        # Get revenue growth curve
        try:
            # Import IPO data fetcher to get growth curve
            import sys
            import os
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
            from src.data.fetchers.ipo_data_fetcher import IPODataFetcher

            fetcher = IPODataFetcher()
            growth_curve = fetcher.calculate_revenue_growth_curve(ticker)

            if not growth_curve.empty and len(growth_curve) >= 3:
                # Check if growth is accelerating
                recent_growth = growth_curve['yoy_growth_pct'].iloc[-1]
                previous_growth = growth_curve['yoy_growth_pct'].iloc[-2]

                if pd.notna(recent_growth) and pd.notna(previous_growth):
                    if recent_growth > previous_growth:
                        signals['growth_accelerating'] = True
                        signals['scaling_score'] += 30

        except Exception as e:
            print(f"[WARN] Could not analyze growth curve for {ticker}: {e}")

        # Check margin improvement
        operating_margin = fundamentals.get('operating_margin', -1) or -1
        gross_margin = fundamentals.get('gross_margin', 0) or 0

        if operating_margin > 0:
            signals['recently_profitable'] = True
            signals['scaling_score'] += 25

        if gross_margin > 0.6:  # High gross margins = good unit economics
            signals['margins_improving'] = True
            signals['scaling_score'] += 25

        # Check revenue per employee (efficiency at scale)
        revenue = company_info.get('revenue', 0) or 0
        employees = company_info.get('employees', 1) or 1
        revenue_per_employee = revenue / employees if employees > 0 else 0

        if revenue_per_employee > 500_000:
            signals['scaling_score'] += 20

        # Determine if at inflection point
        signals['scaling_score'] = min(100, signals['scaling_score'])
        signals['is_at_inflection'] = signals['scaling_score'] >= 60

        return signals

    def calculate_yc_score(self, ticker: str, company_info: Dict, fundamentals: Dict) -> Dict:
        """
        Calculate comprehensive Y Combinator-style score.

        Combines all signals:
        1. Niche market dominance
        2. Category emergence
        3. 10x moat
        4. Scaling inflection

        Returns:
            Dictionary with all scores and final YC score
        """
        # Calculate individual scores
        dominance_score = self.calculate_market_dominance_score(company_info, fundamentals)
        emergence_signals = self.detect_category_emergence(company_info, fundamentals)
        moat_score = self.calculate_10x_moat_score(company_info, fundamentals)
        scaling_signals = self.detect_scaling_inflection(ticker, company_info, fundamentals)

        # Weighted combination
        # Weights: Dominance (30%), Emergence (30%), Moat (25%), Scaling (15%)
        yc_score = (
            dominance_score * 0.30 +
            emergence_signals['emergence_score'] * 0.30 +
            moat_score * 0.25 +
            scaling_signals['scaling_score'] * 0.15
        )

        return {
            'ticker': ticker,
            'company_name': company_info.get('name', ticker),
            'yc_score': round(yc_score, 2),
            'dominance_score': round(dominance_score, 2),
            'emergence_score': round(emergence_signals['emergence_score'], 2),
            'moat_score': round(moat_score, 2),
            'scaling_score': round(scaling_signals['scaling_score'], 2),
            'category': emergence_signals['category'],
            'estimated_tam_billions': emergence_signals['estimated_tam_billions'],
            'is_niche': emergence_signals['is_niche'],
            'is_emerging': emergence_signals['is_emerging'],
            'is_at_inflection': scaling_signals['is_at_inflection'],
            'is_strong_candidate': yc_score >= 60,  # Score 60+ = strong YC-style candidate
        }


if __name__ == "__main__":
    # Test the niche analyzer
    from src.data.fetchers.ipo_data_fetcher import IPODataFetcher

    analyzer = IPONicheAnalyzer()
    fetcher = IPODataFetcher()

    # Test with some IPO stocks
    test_tickers = ['RDDT', 'ARM', 'SNOW', 'PLTR', 'COIN']

    print("IPO Niche Market Analyzer - Y Combinator Style")
    print("=" * 70)

    for ticker in test_tickers:
        print(f"\nAnalyzing {ticker}...")

        # Get company data
        company_info = fetcher.get_company_info(ticker)
        fundamentals = fetcher.get_fundamental_metrics(ticker)

        if 'error' in company_info or 'error' in fundamentals:
            print(f"  [ERROR] Could not analyze {ticker}")
            continue

        # Calculate YC score
        yc_analysis = analyzer.calculate_yc_score(ticker, company_info, fundamentals)

        print(f"\n{ticker} - {yc_analysis['company_name']}")
        print(f"Category: {yc_analysis['category']} (TAM: ${yc_analysis['estimated_tam_billions']:.1f}B)")
        print(f"\nY Combinator Score: {yc_analysis['yc_score']:.1f}/100")
        print(f"  - Market Dominance: {yc_analysis['dominance_score']:.1f}/100")
        print(f"  - Category Emergence: {yc_analysis['emergence_score']:.1f}/100")
        print(f"  - 10x Moat: {yc_analysis['moat_score']:.1f}/100")
        print(f"  - Scaling Inflection: {yc_analysis['scaling_score']:.1f}/100")
        print(f"\nSignals:")
        print(f"  - Niche Market: {yc_analysis['is_niche']}")
        print(f"  - Emerging Category: {yc_analysis['is_emerging']}")
        print(f"  - At Inflection: {yc_analysis['is_at_inflection']}")
        print(f"  - Strong Candidate: {yc_analysis['is_strong_candidate']}")
        print("-" * 70)

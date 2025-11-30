"""
IPO Niche Market Screener

Scans for recent IPOs dominating small, fast-growing niche markets.
Uses Y Combinator principles to identify breakout candidates.

Y Combinator Investment Thesis:
1. Small market TODAY that will be BIG tomorrow (category emergence)
2. Company can dominate this small market (monopoly power)
3. 10x better product (not 10% better) - creates moat
4. Transitioning from "doing things that don't scale" to scale (inflection point)

Success Examples:
- Airbnb: Dominated "home sharing" niche → massive market
- Uber: Dominated "ride sharing" niche → transformed transportation
- Stripe: Dominated "developer payment APIs" → fintech giant
- Snowflake: Dominated "cloud data warehouse" → data infrastructure leader

This screener finds the NEXT generation of these companies right after IPO.

Usage:
    screener = IPONicheScreener()
    candidates = screener.screen_ipo_universe()
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.ipo_data_fetcher import IPODataFetcher
from src.data.features.ipo_niche_analyzer import IPONicheAnalyzer


class IPONicheScreener:
    """
    Screen for IPO stocks with Y Combinator-style growth potential.
    """

    def __init__(self,
                 max_ipo_age_months: int = 24,
                 min_yc_score: float = 60.0,
                 min_market_cap_b: float = 0.5,
                 max_market_cap_b: float = 15.0):
        """
        Initialize IPO niche screener.

        Args:
            max_ipo_age_months: Maximum months since IPO (default: 24)
            min_yc_score: Minimum YC score to qualify (default: 60)
            min_market_cap_b: Minimum market cap in billions (default: 0.5)
            max_market_cap_b: Maximum market cap in billions (default: 15)
        """
        self.max_ipo_age_months = max_ipo_age_months
        self.min_yc_score = min_yc_score
        self.min_market_cap_b = min_market_cap_b
        self.max_market_cap_b = max_market_cap_b

        self.fetcher = IPODataFetcher(max_ipo_age_months=max_ipo_age_months)
        self.analyzer = IPONicheAnalyzer()

    def get_ipo_universe(self) -> List[str]:
        """
        Get list of IPO tickers to screen.

        This is a curated list of notable recent IPOs.
        In production, this could be:
        - Fetched from IPO calendar APIs
        - Scraped from IPO tracking websites
        - Maintained as a dynamic list

        Returns:
            List of ticker symbols
        """
        # Recent IPOs (2020-2024) - update this list regularly
        # Focus on tech, fintech, and emerging category IPOs
        ipo_universe = [
            # 2024 IPOs
            'RDDT',   # Reddit (Social Media)
            'ASND',   # Ascendis Pharma (Biotech)
            'RELY',   # Remitly (Fintech)

            # 2023 IPOs
            'ARM',    # ARM Holdings (Semiconductors)
            'KVUE',   # Kenvue (Consumer Health)
            'FYBR',   # Frontier Communications (Telecom)

            # 2022 IPOs
            'TPG',    # TPG (Private Equity)
            'GTLB',   # GitLab (DevOps)

            # 2021 IPOs
            'COIN',   # Coinbase (Crypto)
            'HOOD',   # Robinhood (Fintech)
            'RIVN',   # Rivian (EV)
            'RBLX',   # Roblox (Gaming)
            'BIRD',   # Allbirds (Consumer)
            'BROS',   # Dutch Bros (Coffee)
            'NU',     # Nu Holdings (Fintech)
            'IONQ',   # IonQ (Quantum Computing)
            'UPST',   # Upstart (AI Lending)
            'PATH',   # UiPath (RPA)
            'DDOG',   # Datadog (Monitoring)

            # 2020 IPOs
            'SNOW',   # Snowflake (Data Cloud)
            'PLTR',   # Palantir (Data Analytics)
            'DASH',   # DoorDash (Food Delivery)
            'ABNB',   # Airbnb (Travel)
            'U',      # Unity (Gaming Engine)
            'CRWD',   # CrowdStrike (Cybersecurity)
            'ZM',     # Zoom (Video Conferencing)
            'PTON',   # Peloton (Fitness)
            'BILL',   # Bill.com (B2B Payments)
            'C3AI',   # C3.ai (Enterprise AI)

            # 2019 IPOs
            'UBER',   # Uber (Ride Sharing)
            'LYFT',   # Lyft (Ride Sharing)
            'PINS',   # Pinterest (Social Media)
            'WORK',   # Slack (Collaboration) - acquired
            'ZS',     # Zscaler (Cloud Security)
            'PDD',    # Pinduoduo (E-commerce)

            # 2018 IPOs
            'SPOT',   # Spotify (Music Streaming)
            'DBX',    # Dropbox (Cloud Storage)
            'DOCU',   # DocuSign (E-signature)

            # Earlier but still relevant
            'SQ',     # Block/Square (Fintech)
            'SHOP',   # Shopify (E-commerce)
            'TWLO',   # Twilio (Communications API)
            'TEAM',   # Atlassian (Dev Tools)
            'OKTA',   # Okta (Identity)
            'ZUO',    # Zuora (Subscription Mgmt)
        ]

        return ipo_universe

    def screen_ticker(self, ticker: str) -> Optional[Dict]:
        """
        Screen a single ticker for IPO niche opportunity.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Analysis dict if qualifies, None otherwise
        """
        try:
            # Get company info
            company_info = self.fetcher.get_company_info(ticker)

            if 'error' in company_info:
                return None

            # Check basic filters
            months_since_ipo = company_info.get('months_since_ipo')
            if months_since_ipo is None or months_since_ipo > self.max_ipo_age_months:
                return None

            market_cap_b = company_info.get('market_cap_b', 0)
            if market_cap_b < self.min_market_cap_b or market_cap_b > self.max_market_cap_b:
                return None

            # Get fundamentals
            fundamentals = self.fetcher.get_fundamental_metrics(ticker)

            if 'error' in fundamentals:
                return None

            # Calculate YC score
            yc_analysis = self.analyzer.calculate_yc_score(ticker, company_info, fundamentals)

            # Check if meets threshold
            if yc_analysis['yc_score'] < self.min_yc_score:
                return None

            # Combine all data
            result = {
                **company_info,
                **fundamentals,
                **yc_analysis
            }

            return result

        except Exception as e:
            print(f"[ERROR] Failed to screen {ticker}: {e}")
            return None

    def screen_ipo_universe(self, custom_tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Screen all IPOs in the universe.

        Args:
            custom_tickers: Optional custom list of tickers to screen

        Returns:
            DataFrame with qualifying IPO candidates
        """
        tickers = custom_tickers if custom_tickers else self.get_ipo_universe()

        print("=" * 80)
        print("IPO NICHE MARKET SCREENER")
        print("Identifying breakout IPOs using Y Combinator principles")
        print("=" * 80)
        print(f"\nCriteria:")
        print(f"  - IPO within {self.max_ipo_age_months} months")
        print(f"  - Market cap: ${self.min_market_cap_b}B - ${self.max_market_cap_b}B")
        print(f"  - YC Score: {self.min_yc_score}+ / 100")
        print(f"\nScanning {len(tickers)} tickers...")
        print("-" * 80)

        results = []

        for ticker in tickers:
            analysis = self.screen_ticker(ticker)

            if analysis:
                results.append(analysis)
                print(f"\n[MATCH] {ticker} - {analysis['company_name']}")
                print(f"  YC Score: {analysis['yc_score']:.1f}/100 | "
                      f"Category: {analysis['category']} | "
                      f"IPO: {analysis['months_since_ipo']:.0f}mo ago")
                print(f"  Market Cap: ${analysis['market_cap_b']:.2f}B | "
                      f"Revenue Growth: {(analysis.get('revenue_growth_yoy', 0) or 0)*100:.1f}%")
            else:
                print(f"[SKIP] {ticker}")

        print("\n" + "=" * 80)
        print(f"RESULTS: Found {len(results)} qualifying IPO candidates")
        print("=" * 80)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Sort by YC score
        df = df.sort_values('yc_score', ascending=False)

        return df

    def generate_report(self, results: pd.DataFrame, filename: Optional[str] = None) -> str:
        """
        Generate detailed report of IPO screening results.

        Args:
            results: DataFrame from screen_ipo_universe()
            filename: Optional filename to save report

        Returns:
            Report text
        """
        if results.empty:
            return "No qualifying IPO candidates found."

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("IPO NICHE MARKET SCREENING REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("Y COMBINATOR PRINCIPLES:")
        report_lines.append("1. Dominate small niche markets (monopoly power)")
        report_lines.append("2. Emerging categories (small today, massive tomorrow)")
        report_lines.append("3. Build 10x better products (strong moats)")
        report_lines.append("4. Inflection from 'unscalable' to scale")
        report_lines.append("")
        report_lines.append(f"FOUND {len(results)} QUALIFYING CANDIDATES:")
        report_lines.append("=" * 80)
        report_lines.append("")

        for idx, row in results.iterrows():
            report_lines.append(f"\n{row['ticker']} - {row['company_name']}")
            report_lines.append("-" * 80)
            report_lines.append(f"Category: {row['category']} (Est. TAM: ${row['estimated_tam_billions']:.1f}B)")
            report_lines.append(f"IPO Date: {row['ipo_date'].strftime('%Y-%m-%d') if row['ipo_date'] else 'Unknown'} "
                              f"({row['months_since_ipo']:.0f} months ago)")
            report_lines.append(f"Market Cap: ${row['market_cap_b']:.2f}B")
            report_lines.append("")
            report_lines.append(f"Y COMBINATOR SCORE: {row['yc_score']:.1f}/100")
            report_lines.append(f"  - Market Dominance:   {row['dominance_score']:.1f}/100")
            report_lines.append(f"  - Category Emergence: {row['emergence_score']:.1f}/100")
            report_lines.append(f"  - 10x Moat:           {row['moat_score']:.1f}/100")
            report_lines.append(f"  - Scaling Inflection: {row['scaling_score']:.1f}/100")
            report_lines.append("")
            report_lines.append("KEY METRICS:")
            report_lines.append(f"  Revenue Growth: {(row.get('revenue_growth_yoy', 0) or 0)*100:.1f}% YoY")
            report_lines.append(f"  Gross Margin: {(row.get('gross_margin', 0) or 0)*100:.1f}%")
            report_lines.append(f"  Operating Margin: {(row.get('operating_margin', 0) or 0)*100:.1f}%")
            report_lines.append(f"  Price/Sales: {row.get('price_to_sales', 0):.1f}x")
            report_lines.append("")
            report_lines.append("SIGNALS:")
            report_lines.append(f"  [X] Niche Market: {row['is_niche']}")
            report_lines.append(f"  [X] Emerging Category: {row['is_emerging']}")
            report_lines.append(f"  [X] At Inflection: {row['is_at_inflection']}")
            report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        # Save to file if requested
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nReport saved to: {filename}")

        return report


if __name__ == "__main__":
    # Run the IPO niche screener
    screener = IPONicheScreener(
        max_ipo_age_months=36,  # Look back 3 years
        min_yc_score=50.0,      # Lower threshold for exploration
        min_market_cap_b=1.0,   # $1B+ minimum
        max_market_cap_b=20.0   # <$20B (room to grow)
    )

    # Screen the universe
    candidates = screener.screen_ipo_universe()

    # Generate and display report
    if not candidates.empty:
        print("\n\nDETAILED REPORT:")
        print("=" * 80)
        report = screener.generate_report(candidates, filename="results/ipo_niche_screening_report.txt")
        print(report)

        # Display top candidates
        print("\n\nTOP 10 CANDIDATES (by YC Score):")
        print("=" * 80)
        top_candidates = candidates.head(10)
        print(top_candidates[['ticker', 'company_name', 'yc_score', 'category',
                            'market_cap_b', 'months_since_ipo']].to_string(index=False))
    else:
        print("\n[INFO] No qualifying candidates found with current criteria")
        print("Try adjusting min_yc_score or max_ipo_age_months parameters")

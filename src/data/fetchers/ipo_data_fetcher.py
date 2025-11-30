"""
IPO Data Fetcher

Fetches IPO dates, company information, and fundamental data for recent IPOs.
Uses multiple data sources to identify recent listings and track their performance.

Data Sources:
- yfinance: Company info, IPO date (from info dict)
- SEC Edgar: S-1 filings for IPO dates
- Manual tracking: High-quality IPO data

Y Combinator Principles Applied:
- Find companies <24 months post-IPO (early movers advantage)
- Small market caps ($500M - $10B) with room to grow
- Rapid revenue growth (>40% YoY)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import requests
import json
import sys
import os

# Import IPO database
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.fetchers.ipo_database import IPODatabase


class IPODataFetcher:
    """Fetches IPO data and fundamental information for recent public companies."""

    def __init__(self, max_ipo_age_months: int = 24):
        """
        Initialize IPO data fetcher.

        Args:
            max_ipo_age_months: Maximum age of IPO to consider (default: 24 months)
        """
        self.max_ipo_age_months = max_ipo_age_months
        self.ipo_cache = {}  # Cache IPO data to reduce API calls
        self.ipo_db = IPODatabase()  # Curated IPO database

    def get_ipo_date(self, ticker: str) -> Optional[datetime]:
        """
        Get IPO date for a ticker.

        Tries curated database first, then falls back to yfinance.

        Args:
            ticker: Stock ticker symbol

        Returns:
            IPO date as datetime or None if not available
        """
        # Try curated database first (most reliable)
        ipo_date = self.ipo_db.get_ipo_date(ticker)
        if ipo_date:
            return ipo_date

        # Fallback to yfinance
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Try multiple fields that might contain IPO date
            if 'firstTradeDateEpochUtc' in info and info['firstTradeDateEpochUtc']:
                return datetime.fromtimestamp(info['firstTradeDateEpochUtc'])

            # Some tickers have ipoDate field
            if 'ipoDate' in info and info['ipoDate']:
                return pd.to_datetime(info['ipoDate'])

            return None

        except Exception as e:
            print(f"[WARN] Could not fetch IPO date for {ticker}: {e}")
            return None

    def is_recent_ipo(self, ticker: str, max_months: Optional[int] = None) -> bool:
        """
        Check if ticker is a recent IPO.

        Args:
            ticker: Stock ticker symbol
            max_months: Maximum months since IPO (default: self.max_ipo_age_months)

        Returns:
            True if IPO within max_months, False otherwise
        """
        if max_months is None:
            max_months = self.max_ipo_age_months

        ipo_date = self.get_ipo_date(ticker)
        if ipo_date is None:
            return False

        months_since_ipo = (datetime.now() - ipo_date).days / 30.44  # Average days per month
        return months_since_ipo <= max_months

    def get_company_info(self, ticker: str) -> Dict:
        """
        Get comprehensive company information.

        Includes:
        - IPO date and age
        - Market cap
        - Sector/industry
        - Description
        - Fundamental metrics

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with company information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            ipo_date = self.get_ipo_date(ticker)
            months_since_ipo = None
            if ipo_date:
                months_since_ipo = (datetime.now() - ipo_date).days / 30.44

            return {
                'ticker': ticker,
                'name': info.get('longName', info.get('shortName', ticker)),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'description': info.get('longBusinessSummary', ''),
                'ipo_date': ipo_date,
                'months_since_ipo': months_since_ipo,
                'market_cap': info.get('marketCap', 0),
                'market_cap_b': info.get('marketCap', 0) / 1e9 if info.get('marketCap') else 0,
                'enterprise_value': info.get('enterpriseValue', 0),
                'revenue': info.get('totalRevenue', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'employees': info.get('fullTimeEmployees', 0),
                'website': info.get('website', ''),
                'country': info.get('country', ''),
            }

        except Exception as e:
            print(f"[ERROR] Failed to fetch company info for {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}

    def get_fundamental_metrics(self, ticker: str) -> Dict:
        """
        Get key fundamental metrics for growth analysis.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with fundamental metrics
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get financial data
            financials = stock.financials
            quarterly_financials = stock.quarterly_financials

            return {
                'ticker': ticker,
                'revenue': info.get('totalRevenue', 0),
                'revenue_growth_yoy': info.get('revenueGrowth', 0),
                'revenue_per_employee': info.get('revenuePerShare', 0) if info.get('revenuePerShare') else 0,
                'profit_margin': info.get('profitMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'gross_margin': info.get('grossMargins', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'price_to_book': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'return_on_assets': info.get('returnOnAssets', 0),
            }

        except Exception as e:
            print(f"[ERROR] Failed to fetch fundamental metrics for {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}

    def calculate_revenue_growth_curve(self, ticker: str) -> pd.DataFrame:
        """
        Calculate revenue growth curve from historical financials.

        Analyzes quarterly and annual revenue to identify acceleration patterns.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with revenue growth metrics over time
        """
        try:
            stock = yf.Ticker(ticker)

            # Get quarterly financials for more granular growth analysis
            quarterly_financials = stock.quarterly_financials

            if quarterly_financials.empty:
                return pd.DataFrame()

            # Extract revenue (Total Revenue row)
            if 'Total Revenue' in quarterly_financials.index:
                revenue_series = quarterly_financials.loc['Total Revenue']
            else:
                print(f"[WARN] No 'Total Revenue' found for {ticker}")
                return pd.DataFrame()

            # Sort by date (oldest to newest)
            revenue_series = revenue_series.sort_index()

            # Calculate quarter-over-quarter growth
            qoq_growth = revenue_series.pct_change() * 100

            # Calculate year-over-year growth (compare to 4 quarters ago)
            yoy_growth = revenue_series.pct_change(periods=4) * 100

            # Build growth curve dataframe
            growth_df = pd.DataFrame({
                'date': revenue_series.index,
                'revenue': revenue_series.values,
                'qoq_growth_pct': qoq_growth.values,
                'yoy_growth_pct': yoy_growth.values
            })

            # Calculate growth acceleration (is growth rate increasing?)
            growth_df['growth_acceleration'] = growth_df['yoy_growth_pct'].diff()

            return growth_df

        except Exception as e:
            print(f"[ERROR] Failed to calculate revenue growth curve for {ticker}: {e}")
            return pd.DataFrame()

    def screen_recent_ipos(self,
                          tickers: List[str],
                          max_months: int = 24,
                          min_market_cap_b: float = 0.5,
                          max_market_cap_b: float = 10.0,
                          min_revenue_growth: float = 0.30) -> pd.DataFrame:
        """
        Screen for recent IPOs matching Y Combinator-style criteria.

        Criteria:
        - Recent IPO (within max_months)
        - Small to mid-cap ($500M - $10B market cap)
        - High revenue growth (>30% YoY)

        Args:
            tickers: List of tickers to screen
            max_months: Maximum months since IPO
            min_market_cap_b: Minimum market cap in billions
            max_market_cap_b: Maximum market cap in billions
            min_revenue_growth: Minimum revenue growth rate (0.30 = 30%)

        Returns:
            DataFrame with qualifying recent IPOs
        """
        results = []

        print(f"Screening {len(tickers)} tickers for recent IPOs...")
        print(f"Criteria: IPO within {max_months} months, "
              f"${min_market_cap_b}B-${max_market_cap_b}B market cap, "
              f">{min_revenue_growth*100:.0f}% revenue growth")
        print("=" * 70)

        for ticker in tickers:
            # Get company info
            company_info = self.get_company_info(ticker)

            # Check if error
            if 'error' in company_info:
                continue

            # Filter criteria
            if company_info.get('months_since_ipo') is None:
                continue  # No IPO date available

            if company_info['months_since_ipo'] > max_months:
                continue  # Too old

            market_cap_b = company_info.get('market_cap_b', 0)
            if market_cap_b < min_market_cap_b or market_cap_b > max_market_cap_b:
                continue  # Outside market cap range

            revenue_growth = company_info.get('revenue_growth', 0) or 0
            if revenue_growth < min_revenue_growth:
                continue  # Insufficient growth

            # Get fundamentals
            fundamentals = self.get_fundamental_metrics(ticker)

            # Combine data
            result = {**company_info, **fundamentals}
            results.append(result)

            print(f"[MATCH] {ticker}: {company_info['name']}")
            print(f"  IPO: {company_info['ipo_date'].strftime('%Y-%m-%d') if company_info['ipo_date'] else 'Unknown'} "
                  f"({company_info['months_since_ipo']:.1f} months ago)")
            print(f"  Market Cap: ${market_cap_b:.2f}B")
            print(f"  Revenue Growth: {revenue_growth*100:.1f}%")

        print("=" * 70)
        print(f"Found {len(results)} qualifying recent IPOs")

        return pd.DataFrame(results)


if __name__ == "__main__":
    # Test the IPO data fetcher
    fetcher = IPODataFetcher(max_ipo_age_months=24)

    # Test with some known recent IPOs (examples - update with actual recent IPOs)
    test_tickers = [
        'SNOW',  # Snowflake (2020 IPO)
        'PLTR',  # Palantir (2020 IPO)
        'DASH',  # DoorDash (2020 IPO)
        'ABNB',  # Airbnb (2020 IPO)
        'COIN',  # Coinbase (2021 IPO)
        'RBLX',  # Roblox (2021 IPO)
        'HOOD',  # Robinhood (2021 IPO)
        'RIVN',  # Rivian (2021 IPO)
        'ARM',   # ARM Holdings (2023 IPO)
        'RDDT',  # Reddit (2024 IPO)
    ]

    print("Testing IPO Data Fetcher")
    print("=" * 70)

    # Test individual company
    print("\nTesting individual company info:")
    info = fetcher.get_company_info('RDDT')
    print(f"Company: {info.get('name')}")
    print(f"IPO Date: {info.get('ipo_date')}")
    print(f"Months Since IPO: {info.get('months_since_ipo'):.1f}")
    print(f"Market Cap: ${info.get('market_cap_b'):.2f}B")
    print(f"Revenue Growth: {info.get('revenue_growth', 0)*100:.1f}%")

    # Screen for recent IPOs
    print("\n" + "=" * 70)
    print("Screening for recent IPOs...")
    recent_ipos = fetcher.screen_recent_ipos(
        test_tickers,
        max_months=36,  # Expand to 36 months for testing
        min_market_cap_b=1.0,
        max_market_cap_b=50.0,
        min_revenue_growth=0.20  # 20%+ growth
    )

    if not recent_ipos.empty:
        print("\nRecent IPO Candidates:")
        print(recent_ipos[['ticker', 'name', 'months_since_ipo', 'market_cap_b', 'revenue_growth']].to_string())

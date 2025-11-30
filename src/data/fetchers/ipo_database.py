"""
Curated IPO Database

Manual database of notable IPO dates for Y Combinator-style screening.
This is more reliable than API lookups which often don't have IPO dates.

Sources:
- NASDAQ IPO calendar
- Renaissance Capital IPO data
- Public company filings

Note: In production, this could be:
- Fetched from IPO calendar APIs
- Scraped from IPO tracking websites
- Integrated with SEC Edgar filings
"""

from datetime import datetime
from typing import Dict, Optional


# Curated IPO database: ticker -> IPO date
IPO_DATABASE = {
    # 2024 IPOs
    'RDDT': '2024-03-21',   # Reddit
    'ASND': '2024-09-13',   # Ascendis Pharma
    'RELY': '2024-09-21',   # Remitly (relisted)

    # 2023 IPOs
    'ARM': '2023-09-14',    # ARM Holdings
    'KVUE': '2023-05-04',   # Kenvue
    'FYBR': '2023-05-02',   # Frontier Communications
    'GTLB': '2021-10-14',   # GitLab (fixing date)

    # 2022 IPOs
    'TPG': '2022-01-13',    # TPG Inc

    # 2021 IPOs
    'COIN': '2021-04-14',   # Coinbase
    'HOOD': '2021-07-29',   # Robinhood
    'RIVN': '2021-11-10',   # Rivian
    'RBLX': '2021-03-10',   # Roblox
    'BIRD': '2021-11-03',   # Allbirds
    'BROS': '2021-09-15',   # Dutch Bros
    'NU': '2021-12-09',     # Nu Holdings
    'IONQ': '2021-10-01',   # IonQ
    'UPST': '2020-12-16',   # Upstart (fixing date)
    'PATH': '2021-04-21',   # UiPath
    'DDOG': '2019-09-19',   # Datadog (fixing date)

    # 2020 IPOs
    'SNOW': '2020-09-16',   # Snowflake
    'PLTR': '2020-09-30',   # Palantir
    'DASH': '2020-12-09',   # DoorDash
    'ABNB': '2020-12-10',   # Airbnb
    'U': '2020-09-18',      # Unity
    'CRWD': '2019-06-12',   # CrowdStrike (fixing date)
    'ZM': '2019-04-18',     # Zoom (fixing date)
    'PTON': '2019-09-26',   # Peloton (fixing date)
    'BILL': '2019-12-12',   # Bill.com (fixing date)
    'AI': '2020-12-09',     # C3.ai (correct ticker)

    # 2019 IPOs
    'UBER': '2019-05-10',   # Uber
    'LYFT': '2019-03-29',   # Lyft
    'PINS': '2019-04-18',   # Pinterest
    'ZS': '2018-03-16',     # Zscaler (fixing date)
    'PDD': '2018-07-26',    # Pinduoduo (fixing date)

    # 2018 IPOs
    'SPOT': '2018-04-03',   # Spotify
    'DBX': '2018-03-23',    # Dropbox
    'DOCU': '2018-04-27',   # DocuSign

    # Earlier notable IPOs
    'SQ': '2015-11-19',     # Block/Square
    'SHOP': '2015-05-21',   # Shopify
    'TWLO': '2016-06-23',   # Twilio
    'TEAM': '2015-12-10',   # Atlassian
    'OKTA': '2017-04-07',   # Okta
    'ZUO': '2018-04-12',    # Zuora
}


class IPODatabase:
    """Access curated IPO date database."""

    def __init__(self):
        """Initialize IPO database."""
        self.database = IPO_DATABASE

    def get_ipo_date(self, ticker: str) -> Optional[datetime]:
        """
        Get IPO date for ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            IPO date as datetime or None if not in database
        """
        ipo_date_str = self.database.get(ticker.upper())
        if ipo_date_str:
            return datetime.strptime(ipo_date_str, '%Y-%m-%d')
        return None

    def has_ipo_date(self, ticker: str) -> bool:
        """
        Check if ticker has IPO date in database.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if IPO date is available
        """
        return ticker.upper() in self.database

    def get_all_tickers(self) -> list:
        """
        Get all tickers in database.

        Returns:
            List of ticker symbols
        """
        return list(self.database.keys())

    def get_recent_ipos(self, months: int = 24) -> Dict[str, datetime]:
        """
        Get IPOs within last N months.

        Args:
            months: Number of months to look back

        Returns:
            Dictionary of ticker -> IPO date for recent IPOs
        """
        cutoff_date = datetime.now() - timedelta(days=months * 30.44)

        recent = {}
        for ticker, date_str in self.database.items():
            ipo_date = datetime.strptime(date_str, '%Y-%m-%d')
            if ipo_date >= cutoff_date:
                recent[ticker] = ipo_date

        return recent

    def add_ipo(self, ticker: str, ipo_date: str):
        """
        Add IPO to database (for dynamic updates).

        Args:
            ticker: Stock ticker symbol
            ipo_date: IPO date in 'YYYY-MM-DD' format
        """
        self.database[ticker.upper()] = ipo_date


if __name__ == "__main__":
    from datetime import timedelta

    # Test the IPO database
    db = IPODatabase()

    print("IPO Database Test")
    print("=" * 70)
    print(f"Total IPOs in database: {len(db.get_all_tickers())}")
    print()

    # Test recent IPOs
    print("Recent IPOs (last 24 months):")
    print("-" * 70)
    recent = db.get_recent_ipos(months=24)
    for ticker, date in sorted(recent.items(), key=lambda x: x[1], reverse=True):
        months_ago = (datetime.now() - date).days / 30.44
        print(f"{ticker:8} - {date.strftime('%Y-%m-%d')} ({months_ago:.0f} months ago)")

    print()
    print(f"Found {len(recent)} IPOs in last 24 months")

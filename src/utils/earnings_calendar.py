"""
Earnings Calendar Utility

Fetches and caches earnings dates for stocks to enable earnings-aware trading.
Avoids entering trades near earnings announcements (unpredictable volatility).

Based on EXP-031 findings:
- Filtering ±3 days around earnings improves win rate by ~2.3pp
- Reduces unpredictable volatility from fundamental changes
- Industry-standard practice for mean reversion strategies
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os


class EarningsCalendar:
    """
    Manages earnings dates for stocks.

    Fetches earnings dates from Yahoo Finance and caches them locally.
    Provides methods to check if a date is near earnings.
    """

    def __init__(self, cache_dir: str = 'data/earnings_cache'):
        """
        Initialize earnings calendar.

        Args:
            cache_dir: Directory to cache earnings dates
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_earnings_dates(self, symbol: str, force_refresh: bool = False) -> List[datetime]:
        """
        Fetch earnings dates for a symbol.

        Args:
            symbol: Stock ticker
            force_refresh: Force refresh from API (ignore cache)

        Returns:
            List of earnings dates
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_earnings.json")

        # Check cache first (unless force refresh)
        if not force_refresh and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)

                # Check if cache is recent (less than 7 days old)
                cache_date = datetime.fromisoformat(data['cached_at'])
                if datetime.now() - cache_date < timedelta(days=7):
                    # Cache is fresh
                    earnings_dates = [datetime.fromisoformat(d) for d in data['earnings_dates']]
                    return earnings_dates
            except Exception as e:
                print(f"[WARNING] Error reading cache for {symbol}: {e}")

        # Fetch from Yahoo Finance
        try:
            stock = yf.Ticker(symbol)

            # Get earnings calendar (returns dict with 'Earnings Date' key)
            calendar = stock.calendar

            if calendar is not None:
                earnings_dates = []

                # yfinance returns a dict with calendar info
                if isinstance(calendar, dict):
                    if 'Earnings Date' in calendar:
                        date_val = calendar['Earnings Date']

                        # Can be a list of dates or single date
                        if isinstance(date_val, list):
                            for d in date_val:
                                try:
                                    earnings_dates.append(pd.to_datetime(d))
                                except:
                                    pass
                        else:
                            try:
                                earnings_dates.append(pd.to_datetime(date_val))
                            except:
                                pass

                # Also try legacy DataFrame/Series format
                elif isinstance(calendar, (pd.DataFrame, pd.Series)):
                    if 'Earnings Date' in calendar:
                        date_val = calendar['Earnings Date']
                        if isinstance(date_val, list):
                            earnings_dates = [pd.to_datetime(d) for d in date_val if pd.notna(d)]
                        elif pd.notna(date_val):
                            earnings_dates = [pd.to_datetime(date_val)]

                # Convert to datetime
                earnings_dates = [d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d
                                 for d in earnings_dates]

                # Cache the results
                cache_data = {
                    'symbol': symbol,
                    'cached_at': datetime.now().isoformat(),
                    'earnings_dates': [d.isoformat() for d in earnings_dates]
                }

                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)

                return earnings_dates
            else:
                print(f"[WARNING] No earnings calendar found for {symbol}")
                return []

        except Exception as e:
            print(f"[ERROR] Failed to fetch earnings for {symbol}: {e}")
            return []

    def is_near_earnings(self, symbol: str, check_date: datetime,
                        window_days: int = 3) -> bool:
        """
        Check if a date is within ±window_days of any earnings announcement.

        Args:
            symbol: Stock ticker
            check_date: Date to check
            window_days: Days before/after earnings to avoid (default: 3)

        Returns:
            True if near earnings, False otherwise
        """
        earnings_dates = self.fetch_earnings_dates(symbol)

        if not earnings_dates:
            # No earnings data available - allow trade (conservative)
            return False

        # Check if check_date is within window of any earnings date
        for earnings_date in earnings_dates:
            days_diff = abs((check_date - earnings_date).days)
            if days_diff <= window_days:
                return True

        return False

    def get_next_earnings_date(self, symbol: str) -> Optional[datetime]:
        """
        Get the next upcoming earnings date for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            Next earnings date, or None if not available
        """
        earnings_dates = self.fetch_earnings_dates(symbol)

        if not earnings_dates:
            return None

        # Find next earnings date after today
        now = datetime.now()
        future_earnings = [d for d in earnings_dates if d > now]

        if future_earnings:
            return min(future_earnings)
        else:
            return None

    def refresh_all_cache(self, symbols: List[str]) -> None:
        """
        Refresh earnings cache for all symbols.

        Args:
            symbols: List of stock tickers to refresh
        """
        print(f"Refreshing earnings cache for {len(symbols)} symbols...")

        for i, symbol in enumerate(symbols, 1):
            print(f"  [{i}/{len(symbols)}] {symbol}...", end=' ')
            try:
                earnings_dates = self.fetch_earnings_dates(symbol, force_refresh=True)
                print(f"OK ({len(earnings_dates)} dates)")
            except Exception as e:
                print(f"ERROR: {e}")

        print("Cache refresh complete!")


def test_earnings_calendar():
    """Test the earnings calendar functionality."""
    print("=" * 70)
    print("EARNINGS CALENDAR TEST")
    print("=" * 70)
    print()

    calendar = EarningsCalendar()

    # Test with a few stocks
    test_symbols = ['NVDA', 'AAPL', 'TSLA']

    for symbol in test_symbols:
        print(f"Testing {symbol}:")
        print("-" * 70)

        # Fetch earnings dates
        earnings_dates = calendar.fetch_earnings_dates(symbol)
        print(f"  Earnings dates found: {len(earnings_dates)}")

        for date in earnings_dates:
            print(f"    {date.strftime('%Y-%m-%d')}")

        # Get next earnings
        next_earnings = calendar.get_next_earnings_date(symbol)
        if next_earnings:
            print(f"  Next earnings: {next_earnings.strftime('%Y-%m-%d')}")
            days_until = (next_earnings - datetime.now()).days
            print(f"  Days until earnings: {days_until}")
        else:
            print(f"  Next earnings: Not available")

        # Test if today is near earnings
        today = datetime.now()
        is_near = calendar.is_near_earnings(symbol, today, window_days=3)
        print(f"  Is today (±3 days) near earnings? {is_near}")

        print()

    print("=" * 70)


if __name__ == '__main__':
    """Run earnings calendar test."""
    test_earnings_calendar()

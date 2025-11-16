"""
Earnings Calendar Fetcher

Retrieves earnings announcement dates to avoid trading near earnings.

PROBLEM:
"Panic sells" near earnings are often justified by poor results or guidance.
Mean reversion fails because the drop is fundamental, not technical.

SOLUTION:
Exclude trades within 3 days before and after earnings announcements.
"""

import pandas as pd
import yfinance as yf
import json
import os
from datetime import datetime, timedelta
from typing import List, Set, Optional


class EarningsCalendarFetcher:
    """
    Fetch earnings dates for stocks and create exclusion windows.
    """

    def __init__(self, exclusion_days_before=3, exclusion_days_after=3,
                 cache_dir='data/earnings_cache', cache_ttl_days=7):
        """
        Initialize earnings calendar fetcher.

        Args:
            exclusion_days_before: Days before earnings to stop trading
            exclusion_days_after: Days after earnings to stop trading
            cache_dir: Directory for persistent cache (default: data/earnings_cache)
            cache_ttl_days: Cache time-to-live in days (default: 7)
        """
        self.exclusion_days_before = exclusion_days_before
        self.exclusion_days_after = exclusion_days_after
        self.cache_dir = cache_dir
        self.cache_ttl_days = cache_ttl_days
        self._earnings_cache = {}  # In-memory cache

        # Create cache directory
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def fetch_earnings_dates(self, ticker: str) -> pd.DataFrame:
        """
        Fetch historical and upcoming earnings dates.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with earnings dates
        """
        # Check in-memory cache first
        if ticker in self._earnings_cache:
            return self._earnings_cache[ticker]

        # Check persistent disk cache
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{ticker}_earnings.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)

                    # Check if cache is still fresh
                    cache_date = datetime.fromisoformat(cache_data['cached_at'])
                    if datetime.now() - cache_date < timedelta(days=self.cache_ttl_days):
                        # Cache is fresh, load it
                        earnings_dates = [pd.to_datetime(d).tz_localize(None) if hasattr(pd.to_datetime(d), 'tz_localize')
                                         else pd.to_datetime(d) for d in cache_data['earnings_dates']]
                        df = pd.DataFrame({'earnings_date': earnings_dates})
                        df['earnings_date'] = pd.to_datetime(df['earnings_date'])
                        # Ensure timezone-naive
                        if df['earnings_date'].dt.tz is not None:
                            df['earnings_date'] = df['earnings_date'].dt.tz_localize(None)
                        self._earnings_cache[ticker] = df  # Store in memory too
                        return df
                except Exception as e:
                    print(f"[WARN] Error reading cache for {ticker}: {e}")

        try:
            stock = yf.Ticker(ticker)

            # Get earnings dates from calendar
            earnings_dates = []

            # Try to get earnings calendar (handles both dict and DataFrame formats)
            try:
                calendar = stock.calendar
                if calendar is not None:
                    # Handle new dict format (yfinance >= 0.2.0)
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

                    # Handle legacy DataFrame/Series format
                    elif isinstance(calendar, (pd.DataFrame, pd.Series)):
                        if 'Earnings Date' in calendar.index:
                            upcoming = calendar.loc['Earnings Date']
                            if isinstance(upcoming, pd.Series):
                                for date in upcoming:
                                    if pd.notna(date):
                                        earnings_dates.append(pd.to_datetime(date))
                            elif pd.notna(upcoming):
                                earnings_dates.append(pd.to_datetime(upcoming))
            except Exception as e:
                pass  # Calendar might not be available

            # Get historical earnings from earnings history
            try:
                earnings_history = stock.earnings_dates
                if earnings_history is not None and len(earnings_history) > 0:
                    # Add all historical earnings dates
                    for date in earnings_history.index:
                        if pd.notna(date):
                            dt = pd.to_datetime(date)
                            # Ensure timezone-naive
                            if hasattr(dt, 'tz_localize') and dt.tz is not None:
                                dt = dt.tz_localize(None)
                            earnings_dates.append(dt)
            except Exception as e:
                pass  # Earnings history might not be available

            # Create DataFrame
            if earnings_dates:
                df = pd.DataFrame({
                    'earnings_date': sorted(set(earnings_dates))
                })
                # Strip timezone for consistent comparison
                df['earnings_date'] = pd.to_datetime(df['earnings_date']).dt.tz_localize(None)
                self._earnings_cache[ticker] = df

                # Save to persistent cache
                if self.cache_dir:
                    try:
                        cache_file = os.path.join(self.cache_dir, f"{ticker}_earnings.json")
                        cache_data = {
                            'ticker': ticker,
                            'cached_at': datetime.now().isoformat(),
                            'earnings_dates': [d.isoformat() for d in df['earnings_date']]
                        }
                        with open(cache_file, 'w') as f:
                            json.dump(cache_data, f, indent=2)
                    except Exception as e:
                        print(f"[WARN] Could not save cache for {ticker}: {e}")

                return df
            else:
                # Return empty DataFrame if no earnings data
                df = pd.DataFrame(columns=['earnings_date'])
                self._earnings_cache[ticker] = df
                return df

        except Exception as e:
            print(f"[WARN] Could not fetch earnings for {ticker}: {e}")
            return pd.DataFrame(columns=['earnings_date'])

    def get_exclusion_dates(self, ticker: str) -> Set[pd.Timestamp]:
        """
        Get all dates that should be excluded from trading.

        Includes earnings date +/- exclusion window.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Set of dates to exclude from trading
        """
        earnings_df = self.fetch_earnings_dates(ticker)

        exclusion_dates = set()

        for earnings_date in earnings_df['earnings_date']:
            # Add earnings date itself
            exclusion_dates.add(earnings_date)

            # Add days before
            for i in range(1, self.exclusion_days_before + 1):
                exclusion_dates.add(earnings_date - timedelta(days=i))

            # Add days after
            for i in range(1, self.exclusion_days_after + 1):
                exclusion_dates.add(earnings_date + timedelta(days=i))

        return exclusion_dates

    def should_trade_on_date(self, ticker: str, trade_date: pd.Timestamp) -> bool:
        """
        Check if trading should be allowed on a given date.

        Args:
            ticker: Stock ticker symbol
            trade_date: Date to check

        Returns:
            True if trading is allowed, False if in earnings exclusion window
        """
        exclusion_dates = self.get_exclusion_dates(ticker)

        # Normalize dates to just date (no time)
        trade_date_normalized = pd.Timestamp(trade_date.date())
        exclusion_dates_normalized = {pd.Timestamp(d.date()) for d in exclusion_dates}

        return trade_date_normalized not in exclusion_dates_normalized

    def add_earnings_filter_to_signals(
        self,
        df: pd.DataFrame,
        ticker: str,
        signal_column: str = 'panic_sell'
    ) -> pd.DataFrame:
        """
        Filter out signals that occur near earnings dates.

        Args:
            df: DataFrame with trading signals and 'Date' column
            ticker: Stock ticker symbol
            signal_column: Name of signal column to filter

        Returns:
            DataFrame with filtered signals
        """
        data = df.copy()

        # Ensure Date column exists
        if 'Date' not in data.columns:
            print("[WARN] No 'Date' column found, cannot apply earnings filter")
            return data

        # Get exclusion dates
        exclusion_dates = self.get_exclusion_dates(ticker)
        exclusion_dates_normalized = {pd.Timestamp(d.date()) for d in exclusion_dates}

        # Store original signals
        if signal_column in data.columns:
            data[f'{signal_column}_before_earnings_filter'] = data[signal_column].copy()

            # Disable signals on exclusion dates
            for idx, row in data.iterrows():
                trade_date = pd.Timestamp(row['Date'].date())
                if trade_date in exclusion_dates_normalized:
                    data.loc[idx, signal_column] = 0

        return data

    def get_earnings_stats(self, ticker: str, start_date: str = None, end_date: str = None) -> dict:
        """
        Get statistics about earnings dates in a period.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)

        Returns:
            Dictionary with earnings statistics
        """
        earnings_df = self.fetch_earnings_dates(ticker)

        if len(earnings_df) == 0:
            return {
                'total_earnings': 0,
                'earnings_per_year': 0,
                'data_available': False
            }

        # Filter by date range if provided
        if start_date:
            start = pd.to_datetime(start_date)
            earnings_df = earnings_df[earnings_df['earnings_date'] >= start]

        if end_date:
            end = pd.to_datetime(end_date)
            earnings_df = earnings_df[earnings_df['earnings_date'] <= end]

        total_earnings = len(earnings_df)

        # Calculate earnings per year
        if total_earnings > 0:
            date_range_days = (earnings_df['earnings_date'].max() - earnings_df['earnings_date'].min()).days
            years = date_range_days / 365.25
            earnings_per_year = total_earnings / max(years, 1)
        else:
            earnings_per_year = 0

        return {
            'total_earnings': total_earnings,
            'earnings_per_year': earnings_per_year,
            'data_available': True,
            'exclusion_days_before': self.exclusion_days_before,
            'exclusion_days_after': self.exclusion_days_after,
            'total_exclusion_days': total_earnings * (self.exclusion_days_before + self.exclusion_days_after + 1)
        }


# Convenience function for quick filtering
def filter_signals_near_earnings(
    df: pd.DataFrame,
    ticker: str,
    signal_column: str = 'panic_sell',
    exclusion_days_before: int = 3,
    exclusion_days_after: int = 3
) -> pd.DataFrame:
    """
    Quick function to filter signals near earnings.

    Args:
        df: DataFrame with signals
        ticker: Stock ticker
        signal_column: Signal column name
        exclusion_days_before: Days before earnings to exclude
        exclusion_days_after: Days after earnings to exclude

    Returns:
        Filtered DataFrame
    """
    fetcher = EarningsCalendarFetcher(
        exclusion_days_before=exclusion_days_before,
        exclusion_days_after=exclusion_days_after
    )
    return fetcher.add_earnings_filter_to_signals(df, ticker, signal_column)


if __name__ == "__main__":
    print("Earnings Calendar Fetcher module loaded")
    print()
    print("Purpose: Exclude trades near earnings announcements")
    print("Default exclusion: 3 days before + 3 days after earnings")
    print()
    print("Usage:")
    print("  fetcher = EarningsCalendarFetcher()")
    print("  exclusion_dates = fetcher.get_exclusion_dates('NVDA')")
    print("  filtered_data = fetcher.add_earnings_filter_to_signals(data, 'NVDA')")

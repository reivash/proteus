"""
Yahoo Finance data fetcher for stock market data.
Uses yfinance library to download historical price data.
"""

import time
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


# Rate limiting configuration
_last_request_time = 0
_min_request_interval = 0.2  # 200ms between requests


def _rate_limit():
    """Simple rate limiter for yfinance requests."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _min_request_interval:
        time.sleep(_min_request_interval - elapsed)
    _last_request_time = time.time()


class YahooFinanceFetcher:
    """Fetches historical stock data from Yahoo Finance."""

    def __init__(self):
        """Initialize the Yahoo Finance fetcher."""
        pass

    def fetch_stock_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y"
    ) -> pd.DataFrame:
        """
        Fetch historical stock data for a given ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', '^GSPC' for S&P 500)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch if dates not specified (e.g., '1mo', '3mo', '1y', '2y', '5y', 'max')

        Returns:
            DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
        """
        try:
            _rate_limit()  # Avoid hitting rate limits
            stock = yf.Ticker(ticker)

            if start_date and end_date:
                df = stock.history(start=start_date, end=end_date)
            else:
                df = stock.history(period=period)

            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")

            # Reset index to make Date a column
            df.reset_index(inplace=True)

            return df

        except Exception as e:
            raise Exception(f"Error fetching data for {ticker}: {str(e)}")

    def fetch_multiple_stocks(
        self,
        tickers: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y"
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.

        Args:
            tickers: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch if dates not specified

        Returns:
            Dictionary mapping ticker -> DataFrame
        """
        results = {}

        for ticker in tickers:
            try:
                results[ticker] = self.fetch_stock_data(ticker, start_date, end_date, period)
                print(f"✓ Fetched data for {ticker}: {len(results[ticker])} rows")
            except Exception as e:
                print(f"✗ Failed to fetch {ticker}: {str(e)}")
                results[ticker] = None

        return results


if __name__ == "__main__":
    # Test the fetcher
    fetcher = YahooFinanceFetcher()

    # Fetch S&P 500 data
    print("Testing Yahoo Finance Fetcher...")
    print("-" * 50)

    sp500_data = fetcher.fetch_stock_data("^GSPC", period="1y")
    print(f"\nS&P 500 Data Shape: {sp500_data.shape}")
    print(f"Date Range: {sp500_data['Date'].min()} to {sp500_data['Date'].max()}")
    print("\nFirst few rows:")
    print(sp500_data.head())
    print("\nColumns:", sp500_data.columns.tolist())

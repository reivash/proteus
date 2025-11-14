"""
Fetch market context data (VIX, sector ETFs) to enhance predictions.
"""

import pandas as pd
from typing import List, Dict
from .yahoo_finance import YahooFinanceFetcher


class MarketContextFetcher:
    """
    Fetches market context data including VIX and sector ETFs.
    """

    def __init__(self):
        """Initialize market context fetcher."""
        self.fetcher = YahooFinanceFetcher()

        # Market context tickers
        self.context_tickers = {
            'VIX': '^VIX',          # Volatility Index
            'SPY': 'SPY',           # S&P 500 ETF
            'QQQ': 'QQQ',           # NASDAQ-100 ETF (Tech-heavy)
            'IWM': 'IWM',           # Russell 2000 (Small-cap)
            'DIA': 'DIA'            # Dow Jones Industrial Average
        }

    def fetch_market_context(
        self,
        start_date=None,
        end_date=None,
        period="2y"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all market context data.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch if dates not specified

        Returns:
            Dictionary mapping context name to DataFrame
        """
        context_data = {}

        print("Fetching market context data...")
        for name, ticker in self.context_tickers.items():
            try:
                data = self.fetcher.fetch_stock_data(
                    ticker,
                    start_date=start_date,
                    end_date=end_date,
                    period=period
                )
                context_data[name] = data
                print(f"  [OK] {name} ({ticker}): {len(data)} rows")
            except Exception as e:
                print(f"  [FAIL] {name} ({ticker}): {str(e)}")
                context_data[name] = None

        return context_data

    def merge_context_with_stock(
        self,
        stock_data: pd.DataFrame,
        context_data: Dict[str, pd.DataFrame],
        features_to_include: List[str] = None
    ) -> pd.DataFrame:
        """
        Merge market context features with stock data.

        Args:
            stock_data: Stock DataFrame with Date column
            context_data: Dictionary of context DataFrames
            features_to_include: List of context sources to include
                                (default: all available)

        Returns:
            DataFrame with merged context features
        """
        if features_to_include is None:
            features_to_include = list(context_data.keys())

        merged = stock_data.copy()

        for name in features_to_include:
            if name not in context_data or context_data[name] is None:
                print(f"  [WARN] Skipping {name} - data not available")
                continue

            ctx = context_data[name][['Date', 'Close', 'Volume']].copy()
            ctx.columns = ['Date', f'{name}_Close', f'{name}_Volume']

            # Merge on Date
            merged = pd.merge(merged, ctx, on='Date', how='left')

            # Add derived features
            if name == 'VIX':
                # VIX change rate
                merged[f'{name}_Change'] = merged[f'{name}_Close'].pct_change()

            # Relative performance vs stock
            merged[f'{name}_Relative'] = merged['Close'] / merged[f'{name}_Close']

        # Forward fill missing values (e.g., VIX not available on weekends)
        merged = merged.fillna(method='ffill').fillna(method='bfill')

        return merged


# Tech stock tickers for testing
TECH_STOCKS = {
    'NVDA': 'NVIDIA Corporation',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc. (Google)',
    'AAPL': 'Apple Inc.',
    'TSLA': 'Tesla Inc.',
    'META': 'Meta Platforms Inc.',
    'AMZN': 'Amazon.com Inc.',
    'AMD': 'Advanced Micro Devices'
}


if __name__ == "__main__":
    # Test market context fetcher
    print("=" * 70)
    print("TESTING MARKET CONTEXT FETCHER")
    print("=" * 70)

    fetcher = MarketContextFetcher()

    # Fetch context
    context = fetcher.fetch_market_context(period="1y")

    print(f"\nFetched {len([k for k, v in context.items() if v is not None])} context sources")

    # Test merging
    if context.get('VIX') is not None:
        # Fetch sample stock
        stock_fetcher = YahooFinanceFetcher()
        stock_data = stock_fetcher.fetch_stock_data("^GSPC", period="1y")

        print(f"\nOriginal stock data: {stock_data.shape}")

        merged = fetcher.merge_context_with_stock(stock_data, context)

        print(f"Merged data: {merged.shape}")
        print(f"New columns added: {merged.shape[1] - stock_data.shape[1]}")

        print("\nSample of merged data:")
        print(merged[['Date', 'Close', 'VIX_Close', 'SPY_Close', 'QQQ_Close']].tail())

    print("\n" + "=" * 70)
    print("Market context test PASSED!")
    print("=" * 70)

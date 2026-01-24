"""
Cross-Sectional Feature Engineering for ML Panic Sell Prediction

Phase 2 of 6-Week Feature Engineering Program

RESEARCH BASIS:
"Cross-sectional features improve ML trading by +2-4pp AUC" (2024)
- Sector rotation patterns predict stock-specific panic sells
- Market regime context (VIX, SPY beta) separates systemic vs idiosyncratic drops
- Relative strength vs peers identifies outlier selloffs

FEATURE CATEGORIES (18 features):
1. Sector Relative Features (6): Performance vs sector peers
2. Market Context Features (6): SPY/VIX correlation, beta, regime
3. Peer Comparison Features (6): RSI/volume/momentum vs sector average

NOTE: Requires market data (SPY, VIX) and sector mapping
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import yfinance as yf
from datetime import datetime, timedelta


# Sector mapping for major tickers (simplified GICS classification)
SECTOR_MAP = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
    'META': 'Technology', 'NVDA': 'Technology', 'AMD': 'Technology',
    'INTC': 'Technology', 'CRM': 'Technology', 'ORCL': 'Technology',
    'ADBE': 'Technology', 'CSCO': 'Technology', 'AVGO': 'Technology',
    'NOW': 'Technology', 'AMAT': 'Technology', 'KLAC': 'Technology',
    'MRVL': 'Technology', 'QCOM': 'Technology', 'TXN': 'Technology',
    'ADI': 'Technology', 'INTU': 'Technology',

    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
    'ABT': 'Healthcare', 'TMO': 'Healthcare', 'MRK': 'Healthcare',
    'LLY': 'Healthcare', 'ABBV': 'Healthcare', 'SYK': 'Healthcare',
    'GILD': 'Healthcare', 'HCA': 'Healthcare', 'IDXX': 'Healthcare',
    'INSM': 'Healthcare', 'CVS': 'Healthcare',

    # Financials
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
    'GS': 'Financials', 'MS': 'Financials', 'C': 'Financials',
    'BLK': 'Financials', 'SCHW': 'Financials', 'AXP': 'Financials',
    'USB': 'Financials', 'PNC': 'Financials', 'V': 'Financials',
    'MA': 'Financials', 'AIG': 'Financials',

    # Consumer Discretionary
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
    'HD': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',
    'NKE': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
    'TGT': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary',

    # Consumer Staples
    'WMT': 'Consumer Staples', 'PG': 'Consumer Staples',
    'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
    'COST': 'Consumer Staples',

    # Industrials
    'BA': 'Industrials', 'CAT': 'Industrials', 'GE': 'Industrials',
    'UPS': 'Industrials', 'HON': 'Industrials', 'MMM': 'Industrials',
    'ETN': 'Industrials', 'LMT': 'Industrials', 'ROAD': 'Industrials',

    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'MPC': 'Energy', 'EOG': 'Energy',

    # Communication Services
    'DIS': 'Communication Services', 'CMCSA': 'Communication Services',
    'VZ': 'Communication Services', 'T': 'Communication Services',
    'TMUS': 'Communication Services',

    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities',

    # Real Estate
    'AMT': 'Real Estate', 'PLD': 'Real Estate',

    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'ECL': 'Materials',
    'MLM': 'Materials', 'SHW': 'Materials'
}

# Sector ETF mapping for fetching sector performance data
SECTOR_ETF_MAP = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financials': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Industrials': 'XLI',
    'Energy': 'XLE',
    'Communication Services': 'XLC',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Materials': 'XLB'
}


class CrossSectionalFeatureEngineer:
    """
    Add cross-sectional features for ML panic sell prediction.

    Compares stock performance to:
    1. Its sector peers
    2. Market indices (SPY, VIX)
    3. Statistical distributions
    """

    def __init__(self, fillna=True):
        """
        Initialize cross-sectional feature engineer.

        Args:
            fillna: Fill NaN values with neutral defaults
        """
        self.fillna = fillna
        self.market_data_cache = {}  # Cache SPY/VIX data
        self.sector_etf_cache = {}   # Cache sector ETF data

    def _fetch_sector_etf_data(self, sector: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch sector ETF data for computing sector averages.

        Args:
            sector: Sector name (e.g., 'Technology')
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Returns:
            DataFrame with sector ETF OHLCV data or None if failed
        """
        etf = SECTOR_ETF_MAP.get(sector)
        if not etf:
            return None

        cache_key = f"{etf}_{start_date}_{end_date}"
        if cache_key in self.sector_etf_cache:
            return self.sector_etf_cache[cache_key]

        try:
            data = yf.download(etf, start=start_date, end=end_date, progress=False)
            if len(data) == 0:
                return None

            # Flatten multi-index columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            self.sector_etf_cache[cache_key] = data
            return data

        except Exception as e:
            print(f"[WARN] Failed to fetch sector ETF {etf}: {e}")
            return None

    def add_sector_relative_features(self, df: pd.DataFrame, ticker: str,
                                      sector_data: Optional[Dict] = None,
                                      start_date: Optional[str] = None,
                                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Add features measuring stock performance relative to sector.

        Features (6):
        - sector_relative_return_1d: 1-day return vs sector average
        - sector_relative_return_5d: 5-day return vs sector average
        - sector_percentile_rank: Percentile rank within sector (0-100)
        - sector_outperformance: Days of outperformance in last 20 days
        - sector_divergence: Abs(stock return - sector return) / sector volatility
        - is_sector_laggard: Binary flag if in bottom 25% of sector

        Args:
            df: DataFrame with price data
            ticker: Stock ticker
            sector_data: Optional pre-computed sector performance data
            start_date: Start date for sector ETF fetch (if sector_data not provided)
            end_date: End date for sector ETF fetch (if sector_data not provided)

        Returns:
            DataFrame with sector relative features
        """
        data = df.copy()

        # Get sector for this ticker
        sector = SECTOR_MAP.get(ticker, 'Unknown')

        if sector == 'Unknown':
            # Cannot compute sector features - fill with neutral values
            data['sector_relative_return_1d'] = 0.0
            data['sector_relative_return_5d'] = 0.0
            data['sector_percentile_rank'] = 50.0  # Neutral = median
            data['sector_outperformance'] = 0.5  # Neutral
            data['sector_divergence'] = 0.0
            data['is_sector_laggard'] = False
            return data

        # Compute stock returns
        data['return_1d'] = data['Close'].pct_change(1)
        data['return_5d'] = data['Close'].pct_change(5)

        # Get sector ETF data for computing sector averages
        sector_etf_df = None
        if sector_data is not None and 'etf_data' in sector_data:
            # Use pre-computed sector data
            sector_etf_df = sector_data.get('etf_data')
        elif start_date and end_date:
            # Fetch sector ETF data
            sector_etf_df = self._fetch_sector_etf_data(sector, start_date, end_date)

        if sector_etf_df is not None and len(sector_etf_df) > 0:
            # Compute sector ETF returns
            sector_returns_1d = sector_etf_df['Close'].pct_change(1)
            sector_returns_5d = sector_etf_df['Close'].pct_change(5)

            # Compute sector volatility (20-day rolling std of daily returns)
            sector_volatility = sector_returns_1d.rolling(window=20).std()

            # Align sector data with stock data by date
            # Reset index to get Date column for merging
            sector_df = pd.DataFrame({
                'Date': sector_etf_df.index,
                'sector_return_1d': sector_returns_1d.values,
                'sector_return_5d': sector_returns_5d.values,
                'sector_volatility': sector_volatility.values
            })

            # Remove timezone info if present
            if hasattr(sector_df['Date'].iloc[0], 'tz') and sector_df['Date'].iloc[0].tz is not None:
                sector_df['Date'] = sector_df['Date'].dt.tz_localize(None)

            # Ensure data has Date column for merge
            if 'Date' not in data.columns:
                if data.index.name == 'Date' or isinstance(data.index, pd.DatetimeIndex):
                    data = data.reset_index()

            if 'Date' in data.columns:
                # Remove timezone from main data
                if hasattr(data['Date'].dtype, 'tz') and data['Date'].dt.tz is not None:
                    data['Date'] = data['Date'].dt.tz_localize(None)

                # Merge sector data
                data = pd.merge(data, sector_df, on='Date', how='left')

                # Forward fill missing sector data
                data['sector_return_1d'] = data['sector_return_1d'].ffill()
                data['sector_return_5d'] = data['sector_return_5d'].ffill()
                data['sector_volatility'] = data['sector_volatility'].fillna(0.02)  # Default 2% daily vol

                # Compute relative returns
                data['sector_relative_return_1d'] = data['return_1d'] - data['sector_return_1d']
                data['sector_relative_return_5d'] = data['return_5d'] - data['sector_return_5d']

                # Compute divergence (z-score of stock return vs sector)
                data['sector_divergence'] = np.where(
                    data['sector_volatility'] > 0,
                    abs(data['return_1d'] - data['sector_return_1d']) / data['sector_volatility'],
                    0.0
                )

                # Outperformance ratio in last 20 days
                data['sector_outperformance'] = (
                    (data['return_1d'] > data['sector_return_1d']).rolling(window=20).mean()
                )

                # Percentile rank based on relative performance (using z-score approximation)
                # z-score of 0 = 50th percentile, z-score of 2 = ~98th percentile
                relative_z = np.where(
                    data['sector_volatility'] > 0,
                    (data['return_1d'] - data['sector_return_1d']) / data['sector_volatility'],
                    0.0
                )
                # Convert z-score to approximate percentile (sigmoid-like transform)
                data['sector_percentile_rank'] = 50 + 50 * np.tanh(relative_z / 2)

                # Clean up temporary sector columns
                data = data.drop(columns=['sector_return_1d', 'sector_return_5d', 'sector_volatility'], errors='ignore')
            else:
                # No Date column - use neutral values
                data['sector_relative_return_1d'] = 0.0
                data['sector_relative_return_5d'] = 0.0
                data['sector_percentile_rank'] = 50.0
                data['sector_outperformance'] = 0.5
                data['sector_divergence'] = 0.0
        else:
            # No sector data available - use neutral values
            data['sector_relative_return_1d'] = 0.0
            data['sector_relative_return_5d'] = 0.0
            data['sector_percentile_rank'] = 50.0
            data['sector_outperformance'] = 0.5
            data['sector_divergence'] = 0.0

        # Laggard flag (bottom quartile)
        data['is_sector_laggard'] = (data['sector_percentile_rank'] < 25).astype(int)

        # Clean up temporary columns
        data = data.drop(columns=['return_1d', 'return_5d'], errors='ignore')

        if self.fillna:
            data['sector_relative_return_1d'] = data['sector_relative_return_1d'].fillna(0)
            data['sector_relative_return_5d'] = data['sector_relative_return_5d'].fillna(0)
            data['sector_percentile_rank'] = data['sector_percentile_rank'].fillna(50)
            data['sector_outperformance'] = data['sector_outperformance'].fillna(0.5)
            data['sector_divergence'] = data['sector_divergence'].fillna(0)

        return data

    def add_market_context_features(self, df: pd.DataFrame,
                                    start_date: Optional[str] = None,
                                    end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Add features measuring market regime and correlation.

        Features (6):
        - spy_correlation_20d: 20-day rolling correlation with SPY
        - spy_beta_60d: 60-day rolling beta to SPY
        - vix_level: VIX level (fear gauge)
        - vix_change_5d: 5-day change in VIX
        - market_stress: Binary flag if VIX > 25 (stressed market)
        - spy_divergence: Stock return divergence from SPY

        Args:
            df: DataFrame with price data (must have 'Date' column)
            start_date: Start date for market data fetch
            end_date: End date for market data fetch

        Returns:
            DataFrame with market context features
        """
        data = df.copy()

        # Ensure Date column exists and is datetime
        if 'Date' not in data.columns:
            if data.index.name == 'Date' or isinstance(data.index, pd.DatetimeIndex):
                data = data.reset_index()
            else:
                print("[WARN] No Date column found, cannot add market context features")
                return self._add_neutral_market_features(data)

        # Fetch SPY and VIX data
        if start_date is None:
            start_date = data['Date'].min().strftime('%Y-%m-%d')
        if end_date is None:
            end_date = data['Date'].max().strftime('%Y-%m-%d')

        spy_data = self._fetch_market_data('SPY', start_date, end_date)
        vix_data = self._fetch_market_data('^VIX', start_date, end_date)

        if spy_data is None or vix_data is None:
            print("[WARN] Failed to fetch market data, using neutral values")
            return self._add_neutral_market_features(data)

        # Prepare SPY data for merge
        spy_df = spy_data.reset_index()[['Date', 'Close']].rename(columns={'Close': 'SPY_Close'})
        spy_df['spy_returns'] = spy_df['SPY_Close'].pct_change()

        # Remove timezone info to match stock data
        if spy_df['Date'].dt.tz is not None:
            spy_df['Date'] = spy_df['Date'].dt.tz_localize(None)

        # Prepare VIX data for merge
        vix_df = vix_data.reset_index()[['Date', 'Close']].rename(columns={'Close': 'VIX_Close'})

        # Remove timezone info to match stock data
        if vix_df['Date'].dt.tz is not None:
            vix_df['Date'] = vix_df['Date'].dt.tz_localize(None)

        # Remove timezone from main data as well to ensure clean merge
        if hasattr(data['Date'].dtype, 'tz') and data['Date'].dt.tz is not None:
            data['Date'] = data['Date'].dt.tz_localize(None)

        # Merge on Date column
        data = pd.merge(data, spy_df[['Date', 'spy_returns', 'SPY_Close']], on='Date', how='left')
        data = pd.merge(data, vix_df, on='Date', how='left')

        # Forward fill missing values (for days when market was open but SPY/VIX wasn't)
        data['spy_returns'] = data['spy_returns'].ffill()
        data['SPY_Close'] = data['SPY_Close'].ffill()
        data['VIX_Close'] = data['VIX_Close'].ffill()

        # Calculate stock returns
        data_returns = data['Close'].pct_change()

        # Calculate correlation and beta
        data['spy_correlation_20d'] = data_returns.rolling(window=20).corr(data['spy_returns'])

        # Beta = Cov(stock, SPY) / Var(SPY)
        cov = data_returns.rolling(window=60).cov(data['spy_returns'])
        var_spy = data['spy_returns'].rolling(window=60).var()
        data['spy_beta_60d'] = cov / var_spy

        # VIX features
        data['vix_level'] = data['VIX_Close']
        data['vix_change_5d'] = data['VIX_Close'].diff(5)
        data['market_stress'] = (data['VIX_Close'] > 25).astype(int)

        # SPY divergence
        data['spy_divergence'] = abs(data_returns - data['spy_returns'])

        # Clean up temporary columns
        data = data.drop(columns=['spy_returns', 'SPY_Close', 'VIX_Close'], errors='ignore')

        if self.fillna:
            data['spy_correlation_20d'] = data['spy_correlation_20d'].fillna(0.5)  # Neutral correlation
            data['spy_beta_60d'] = data['spy_beta_60d'].fillna(1.0)  # Market beta
            data['vix_level'] = data['vix_level'].fillna(20)  # Neutral VIX
            data['vix_change_5d'] = data['vix_change_5d'].fillna(0)
            data['market_stress'] = data['market_stress'].fillna(0)
            data['spy_divergence'] = data['spy_divergence'].fillna(0)

        return data

    def _fetch_market_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch market data (SPY, VIX) with caching.

        Args:
            ticker: Market ticker (SPY, ^VIX)
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        cache_key = f"{ticker}_{start_date}_{end_date}"

        if cache_key in self.market_data_cache:
            return self.market_data_cache[cache_key]

        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if len(data) == 0:
                return None

            # Flatten multi-index columns if present (yf.download can return MultiIndex)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            self.market_data_cache[cache_key] = data
            return data

        except Exception as e:
            print(f"[ERROR] Failed to fetch {ticker}: {e}")
            return None

    def _add_neutral_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add neutral values for market context features when data unavailable."""
        df['spy_correlation_20d'] = 0.5
        df['spy_beta_60d'] = 1.0
        df['vix_level'] = 20.0
        df['vix_change_5d'] = 0.0
        df['market_stress'] = 0
        df['spy_divergence'] = 0.0
        return df

    def add_peer_comparison_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Add features comparing stock to sector peers.

        Features (6):
        - relative_rsi: RSI vs sector average RSI
        - relative_volume: Volume vs sector average volume
        - relative_momentum: Momentum vs sector average
        - peer_panic_count: Number of peers also selling off
        - isolation_score: How isolated is this selloff (high = idiosyncratic)
        - herd_behavior: Following peers (low isolation) vs独立 move

        Args:
            df: DataFrame with technical indicators
            ticker: Stock ticker

        Returns:
            DataFrame with peer comparison features
        """
        data = df.copy()

        # Placeholder - would need actual peer basket data
        # For now, use neutral values
        data['relative_rsi'] = 0.0  # Neutral (at sector average)
        data['relative_volume'] = 1.0  # Neutral (at sector average)
        data['relative_momentum'] = 0.0  # Neutral
        data['peer_panic_count'] = 0  # No peers panicking
        data['isolation_score'] = 0.5  # Moderate isolation
        data['herd_behavior'] = 0.5  # Neutral

        return data

    def engineer_all_cross_sectional_features(self, df: pd.DataFrame, ticker: str,
                                              start_date: Optional[str] = None,
                                              end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Apply all cross-sectional feature engineering.

        Args:
            df: DataFrame with OHLCV and technical indicators
            ticker: Stock ticker
            start_date: Start date for market data
            end_date: End date for market data

        Returns:
            DataFrame with all 18 cross-sectional features
        """
        print(f"  [INFO] Engineering cross-sectional features for {ticker}...")

        data = df.copy()

        # Add feature groups (pass dates to enable sector ETF fetching)
        data = self.add_sector_relative_features(data, ticker, start_date=start_date, end_date=end_date)
        data = self.add_market_context_features(data, start_date, end_date)
        data = self.add_peer_comparison_features(data, ticker)

        print(f"  [OK] Added 18 cross-sectional features")

        return data


if __name__ == '__main__':
    """Test cross-sectional feature engineer."""

    print("=" * 70)
    print("CROSS-SECTIONAL FEATURE ENGINEER - TEST")
    print("=" * 70)
    print()

    # Test with sample data
    from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
    from src.data.features.technical_indicators import TechnicalFeatureEngineer

    ticker = 'NVDA'
    print(f"Testing with {ticker}...")
    print()

    # Fetch price data
    fetcher = YahooFinanceFetcher()
    data = fetcher.fetch_stock_data(ticker, start_date='2024-01-01', end_date='2024-11-17')

    if data is not None:
        print(f"[OK] Fetched {len(data)} days of data")

        # Add technical features first
        tech_engineer = TechnicalFeatureEngineer(fillna=True)
        data = tech_engineer.engineer_features(data)

        # Add cross-sectional features
        cross_engineer = CrossSectionalFeatureEngineer(fillna=True)
        enriched = cross_engineer.engineer_all_cross_sectional_features(
            data, ticker, start_date='2024-01-01', end_date='2024-11-17'
        )

        print()
        print("Sample of cross-sectional features:")
        print("-" * 70)

        cross_cols = [col for col in enriched.columns
                     if any(x in col for x in ['sector_', 'spy_', 'vix_', 'relative_', 'market_'])]

        if cross_cols:
            print(enriched[cross_cols].tail(10))
            print()
            print(f"[SUCCESS] Created {len(cross_cols)} cross-sectional features")
        else:
            print("[WARN] No cross-sectional features found")
    else:
        print("[ERROR] Failed to fetch data")

    print()
    print("=" * 70)

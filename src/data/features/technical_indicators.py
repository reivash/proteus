"""
Technical indicator features for stock prediction.
Uses the 'ta' library to compute common technical indicators.
"""

import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


class TechnicalFeatureEngineer:
    """
    Creates technical indicator features from OHLCV data.
    """

    def __init__(self, fillna=True):
        """
        Initialize the feature engineer.

        Args:
            fillna: Whether to fill NaN values with 0
        """
        self.fillna = fillna

    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic price and volume features.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with additional basic features
        """
        data = df.copy()

        # Returns
        data['returns_1d'] = data['Close'].pct_change(1)
        data['returns_5d'] = data['Close'].pct_change(5)
        data['returns_20d'] = data['Close'].pct_change(20)

        # Price momentum
        data['momentum_1d'] = data['Close'] - data['Close'].shift(1)
        data['momentum_5d'] = data['Close'] - data['Close'].shift(5)

        # Volatility
        data['volatility_5d'] = data['returns_1d'].rolling(window=5).std()
        data['volatility_20d'] = data['returns_1d'].rolling(window=20).std()

        # High-Low spread
        data['hl_spread'] = (data['High'] - data['Low']) / data['Close']

        # Volume features
        data['volume_change'] = data['Volume'].pct_change(1)
        data['volume_ma_ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()

        return data

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend-following indicators.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with trend indicators
        """
        data = df.copy()

        # Moving Averages
        data['sma_10'] = SMAIndicator(close=data['Close'], window=10, fillna=self.fillna).sma_indicator()
        data['sma_20'] = SMAIndicator(close=data['Close'], window=20, fillna=self.fillna).sma_indicator()
        data['sma_50'] = SMAIndicator(close=data['Close'], window=50, fillna=self.fillna).sma_indicator()
        data['sma_200'] = SMAIndicator(close=data['Close'], window=200, fillna=self.fillna).sma_indicator()

        # Exponential Moving Averages
        data['ema_12'] = EMAIndicator(close=data['Close'], window=12, fillna=self.fillna).ema_indicator()
        data['ema_26'] = EMAIndicator(close=data['Close'], window=26, fillna=self.fillna).ema_indicator()

        # MACD
        macd = MACD(close=data['Close'], fillna=self.fillna)
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_diff'] = macd.macd_diff()

        # MA Ratios (price relative to MAs)
        data['price_sma20_ratio'] = data['Close'] / data['sma_20']
        data['price_sma50_ratio'] = data['Close'] / data['sma_50']

        return data

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with momentum indicators
        """
        data = df.copy()

        # RSI
        data['rsi_14'] = RSIIndicator(close=data['Close'], window=14, fillna=self.fillna).rsi()

        # Stochastic Oscillator
        stoch = StochasticOscillator(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            window=14,
            smooth_window=3,
            fillna=self.fillna
        )
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()

        # Williams %R
        data['williams_r'] = WilliamsRIndicator(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            lbp=14,
            fillna=self.fillna
        ).williams_r()

        return data

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with volatility indicators
        """
        data = df.copy()

        # Bollinger Bands
        bb = BollingerBands(close=data['Close'], window=20, window_dev=2, fillna=self.fillna)
        data['bb_upper'] = bb.bollinger_hband()
        data['bb_middle'] = bb.bollinger_mavg()
        data['bb_lower'] = bb.bollinger_lband()
        data['bb_width'] = bb.bollinger_wband()
        data['bb_pct'] = bb.bollinger_pband()

        # Average True Range
        data['atr'] = AverageTrueRange(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            window=14,
            fillna=self.fillna
        ).average_true_range()

        # ATR percentage
        data['atr_pct'] = data['atr'] / data['Close'] * 100

        return data

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume indicators.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with volume indicators
        """
        data = df.copy()

        # On-Balance Volume
        data['obv'] = OnBalanceVolumeIndicator(
            close=data['Close'],
            volume=data['Volume'],
            fillna=self.fillna
        ).on_balance_volume()

        # OBV normalized
        data['obv_norm'] = (data['obv'] - data['obv'].rolling(window=20).mean()) / data['obv'].rolling(window=20).std()

        # Volume Weighted Average Price (if High, Low available)
        if 'High' in data.columns and 'Low' in data.columns:
            data['vwap'] = VolumeWeightedAveragePrice(
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                volume=data['Volume'],
                window=14,
                fillna=self.fillna
            ).volume_weighted_average_price()

            # Price to VWAP ratio
            data['price_vwap_ratio'] = data['Close'] / data['vwap']

        return data

    def engineer_features(self, df: pd.DataFrame, include_all=True) -> pd.DataFrame:
        """
        Apply all feature engineering steps.

        Args:
            df: DataFrame with OHLCV columns (Date, Open, High, Low, Close, Volume)
            include_all: If True, add all indicators. If False, add only essential ones.

        Returns:
            DataFrame with all engineered features
        """
        data = df.copy()

        # Add all feature groups
        data = self.add_basic_features(data)
        data = self.add_trend_indicators(data)
        data = self.add_momentum_indicators(data)
        data = self.add_volatility_indicators(data)
        data = self.add_volume_indicators(data)

        # Fill remaining NaN values if requested
        if self.fillna:
            data = data.fillna(0)

        return data

    def get_feature_names(self, df: pd.DataFrame) -> list:
        """
        Get list of feature column names (excluding Date and target).

        Args:
            df: DataFrame with engineered features

        Returns:
            List of feature column names
        """
        exclude_cols = ['Date', 'Actual_Direction', 'Price_Change', 'Prediction']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        return feature_cols


if __name__ == "__main__":
    # Test feature engineering
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

    from src.data.fetchers.yahoo_finance import YahooFinanceFetcher

    print("=" * 70)
    print("TESTING TECHNICAL FEATURE ENGINEER")
    print("=" * 70)

    # Fetch data
    fetcher = YahooFinanceFetcher()
    data = fetcher.fetch_stock_data("^GSPC", period="1y")
    print(f"\nOriginal data shape: {data.shape}")
    print(f"Original columns: {data.columns.tolist()}")

    # Engineer features
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)

    print(f"\nEnriched data shape: {enriched_data.shape}")
    print(f"Number of features: {len(engineer.get_feature_names(enriched_data))}")
    print(f"\nNew features added: {enriched_data.shape[1] - data.shape[1]}")

    # Show sample
    print("\nSample of enriched data:")
    print(enriched_data[['Date', 'Close', 'rsi_14', 'macd', 'bb_pct', 'atr_pct']].tail())

    # Check for NaN values
    nan_count = enriched_data.isna().sum().sum()
    print(f"\nTotal NaN values: {nan_count}")

    print("\n" + "=" * 70)
    print("Feature engineering test PASSED!")
    print("=" * 70)

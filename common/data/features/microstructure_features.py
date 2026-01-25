"""
Market Microstructure Features for ML Enhancement
Part of 6-Week Feature Engineering Program - PHASE 1

RESEARCH BASIS:
"Smart design of features, including appropriate preprocessing and denoising,
typically leads to an effective strategy" - 2024 ML Trading Research

FEATURES ADDED:
1. Intraday Volatility Patterns - Gap behavior, range analysis
2. Volume Profile - Time-weighted, block trades, distribution
3. Price Action Quality - Reversals, trend strength, consolidation

EXPECTED IMPACT: +1-3pp AUC improvement
"""

import pandas as pd
import numpy as np
from typing import Optional


class MicrostructureFeatureEngineer:
    """
    Adds market microstructure features that capture intraday trading dynamics.

    These features complement traditional technical indicators by focusing on:
    - How prices move during the day (not just close-to-close)
    - Volume distribution and quality
    - Price action patterns and reversals
    """

    def __init__(self, fillna: bool = True):
        """
        Initialize the microstructure feature engineer.

        Args:
            fillna: Whether to fill NaN values with 0
        """
        self.fillna = fillna

    def add_intraday_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features capturing intraday volatility patterns.

        Research shows intraday range relative to historical volatility
        predicts future returns and mean reversion probability.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with intraday volatility features
        """
        data = df.copy()

        # 1. Gap Analysis
        # Morning gap as % of previous close
        data['gap_pct'] = ((data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)) * 100

        # Gap fill rate - how much of the gap was filled by close
        gap_size = data['Open'] - data['Close'].shift(1)
        gap_fill = data['Close'] - data['Open']
        data['gap_fill_ratio'] = np.where(
            gap_size != 0,
            gap_fill / gap_size,
            0
        )

        # Is this a gap-down day (potential panic sell signal)
        data['is_gap_down'] = (data['gap_pct'] < -1.0).astype(int)

        # 2. Intraday Range Analysis
        # True range as % of close (accounts for gaps)
        data['true_range_pct'] = (
            (np.maximum(
                data['High'] - data['Low'],
                np.maximum(
                    abs(data['High'] - data['Close'].shift(1)),
                    abs(data['Low'] - data['Close'].shift(1))
                )
            ) / data['Close']) * 100
        )

        # Intraday range vs 20-day average (volatility spike detector)
        avg_range = data['true_range_pct'].rolling(window=20).mean()
        data['range_vs_avg'] = data['true_range_pct'] / avg_range

        # High-Low range position (where did close finish in the day's range?)
        # 0 = closed at low, 1 = closed at high
        day_range = data['High'] - data['Low']
        data['close_position_in_range'] = np.where(
            day_range != 0,
            (data['Close'] - data['Low']) / day_range,
            0.5  # Default to middle if no range
        )

        # 3. Volatility Regime
        # Compare recent volatility to longer-term baseline
        vol_5d = data['true_range_pct'].rolling(window=5).mean()
        vol_60d = data['true_range_pct'].rolling(window=60).mean()
        data['volatility_regime'] = vol_5d / vol_60d

        # Volatility acceleration (is volatility increasing?)
        data['vol_acceleration'] = vol_5d.pct_change(5)

        return data

    def add_volume_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features analyzing volume distribution and quality.

        Research shows volume patterns (not just volume magnitude)
        predict price continuation vs reversal.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with volume profile features
        """
        data = df.copy()

        # 1. Volume Quality
        # Average volume per price movement (efficiency)
        price_change = abs(data['Close'] - data['Open'])
        data['volume_efficiency'] = np.where(
            price_change != 0,
            data['Volume'] / price_change,
            0
        )

        # Volume concentration - is volume clustered or distributed?
        # High concentration = institutional activity
        avg_vol = data['Volume'].rolling(window=20).mean()
        std_vol = data['Volume'].rolling(window=20).std()
        data['volume_concentration'] = np.where(
            avg_vol != 0,
            std_vol / avg_vol,
            0
        )

        # 2. Block Trade Detection
        # Unusual volume spike could indicate institutional activity
        data['volume_zscore'] = (
            (data['Volume'] - data['Volume'].rolling(window=20).mean()) /
            data['Volume'].rolling(window=20).std()
        )

        # Is this an abnormally high volume day? (z-score > 2)
        data['high_volume_day'] = (data['volume_zscore'] > 2).astype(int)

        # 3. Volume Trend
        # Is volume increasing or decreasing over time?
        vol_ma_5 = data['Volume'].rolling(window=5).mean()
        vol_ma_20 = data['Volume'].rolling(window=20).mean()
        data['volume_trend'] = vol_ma_5 / vol_ma_20

        # Volume momentum (rate of change)
        data['volume_momentum'] = data['Volume'].pct_change(5)

        # 4. Volume-Price Confirmation
        # Does volume confirm price movement? (both up/both down = strong signal)
        price_direction = np.sign(data['Close'] - data['Open'])
        volume_direction = np.sign(data['Volume'] - data['Volume'].shift(1))
        data['volume_price_alignment'] = (price_direction == volume_direction).astype(int)

        return data

    def add_price_action_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features measuring price action quality and patterns.

        Research shows price action structure (not just direction)
        predicts reversal probability and trend strength.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with price action quality features
        """
        data = df.copy()

        # 1. Reversal Detection
        # Count of intraday reversals (high>open>close>low or opposite)
        # More reversals = choppy action, fewer = trending
        up_reversal = (data['Open'] > data['Low']) & (data['Close'] < data['Open'])
        down_reversal = (data['Open'] < data['High']) & (data['Close'] > data['Open'])
        data['reversal_count_5d'] = (
            (up_reversal.astype(int) + down_reversal.astype(int))
            .rolling(window=5)
            .sum()
        )

        # Tail ratio - measure of rejection (long tails indicate rejection of extreme prices)
        upper_tail = data['High'] - np.maximum(data['Open'], data['Close'])
        lower_tail = np.minimum(data['Open'], data['Close']) - data['Low']
        body = abs(data['Close'] - data['Open'])

        data['upper_tail_ratio'] = np.where(
            (body + upper_tail) != 0,
            upper_tail / (body + upper_tail),
            0
        )
        data['lower_tail_ratio'] = np.where(
            (body + lower_tail) != 0,
            lower_tail / (body + lower_tail),
            0
        )

        # 2. Trend Quality
        # Consistency of closes above/below open (trending vs choppy)
        close_above_open = (data['Close'] > data['Open']).astype(int)
        data['trend_consistency_5d'] = abs(
            close_above_open.rolling(window=5).mean() - 0.5
        ) * 2  # Scale to 0-1, where 1 = strong trend, 0 = choppy

        # Directional movement efficiency
        # Measures how much net movement vs total movement
        net_movement = abs(data['Close'] - data['Close'].shift(5))
        total_movement = data['true_range_pct'].rolling(window=5).sum()
        data['movement_efficiency_5d'] = np.where(
            total_movement != 0,
            net_movement / total_movement,
            0
        )

        # 3. Consolidation Detection
        # Is price consolidating (low volatility, narrow range)?
        # Bollinger Band width as proxy
        if 'bb_width' not in data.columns:
            # Calculate BB width if not already present
            sma_20 = data['Close'].rolling(window=20).mean()
            std_20 = data['Close'].rolling(window=20).std()
            bb_width = (4 * std_20) / sma_20
        else:
            bb_width = data['bb_width']

        bb_width_avg = bb_width.rolling(window=60).mean() if isinstance(bb_width, pd.Series) else bb_width
        data['is_consolidating'] = (bb_width < bb_width_avg * 0.7).astype(int) if isinstance(bb_width, pd.Series) else 0

        # 4. Momentum Strength
        # Rate of change adjusted for volatility (risk-adjusted momentum)
        roc_5d = data['Close'].pct_change(5)
        vol_5d = data['true_range_pct'].rolling(window=5).mean()
        data['risk_adjusted_momentum'] = np.where(
            vol_5d != 0,
            roc_5d / vol_5d,
            0
        )

        # 5. Support/Resistance Touch
        # Is price near recent high/low? (potential reversal zone)
        high_20 = data['High'].rolling(window=20).max()
        low_20 = data['Low'].rolling(window=20).min()
        price_range_20 = high_20 - low_20

        # Distance to highs/lows as % of range
        data['distance_to_high_pct'] = np.where(
            price_range_20 != 0,
            ((high_20 - data['Close']) / price_range_20) * 100,
            0
        )
        data['distance_to_low_pct'] = np.where(
            price_range_20 != 0,
            ((data['Close'] - low_20) / price_range_20) * 100,
            0
        )

        # Is price in extreme zone? (top/bottom 10% of 20-day range)
        data['in_extreme_zone'] = (
            (data['distance_to_high_pct'] < 10) |
            (data['distance_to_low_pct'] < 10)
        ).astype(int)

        return data

    def engineer_all_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all microstructure feature engineering.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with all microstructure features added
        """
        data = df.copy()

        # Add all feature groups
        data = self.add_intraday_volatility_features(data)
        data = self.add_volume_profile_features(data)
        data = self.add_price_action_quality_features(data)

        # Fill NaN values if requested
        if self.fillna:
            # Fill with 0 for most features
            data = data.fillna(0)

        return data

    def get_feature_names(self) -> list:
        """
        Get list of feature names that will be added.

        Returns:
            List of feature column names
        """
        return [
            # Intraday Volatility (9 features)
            'gap_pct', 'gap_fill_ratio', 'is_gap_down',
            'true_range_pct', 'range_vs_avg', 'close_position_in_range',
            'volatility_regime', 'vol_acceleration',

            # Volume Profile (8 features)
            'volume_efficiency', 'volume_concentration',
            'volume_zscore', 'high_volume_day',
            'volume_trend', 'volume_momentum', 'volume_price_alignment',

            # Price Action Quality (12 features)
            'reversal_count_5d', 'upper_tail_ratio', 'lower_tail_ratio',
            'trend_consistency_5d', 'movement_efficiency_5d',
            'is_consolidating', 'risk_adjusted_momentum',
            'distance_to_high_pct', 'distance_to_low_pct', 'in_extreme_zone'
        ]

    def get_feature_count(self) -> int:
        """Get total number of features added."""
        return len(self.get_feature_names())

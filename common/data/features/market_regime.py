"""
Market Regime Detection

Classifies market conditions into:
- BULL: Uptrend, mean reversion works well
- BEAR: Downtrend, mean reversion fails (panic sells continue down)
- SIDEWAYS: Range-bound, moderate performance

Critical for mean reversion strategy:
DO NOT TRADE in bear markets - "panic sells" are often justified.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, Tuple


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class MarketRegimeDetector:
    """
    Detect current market regime using multiple signals.
    """

    def __init__(
        self,
        sma_fast=50,
        sma_slow=200,
        lookback_periods=60,
        bull_threshold=0.05,  # 5% gain over lookback = bull
        bear_threshold=-0.05  # -5% loss over lookback = bear
    ):
        """
        Initialize regime detector.

        Args:
            sma_fast: Fast SMA period for trend detection
            sma_slow: Slow SMA period for long-term trend
            lookback_periods: Days to look back for return calculation
            bull_threshold: % gain threshold to classify as bull
            bear_threshold: % loss threshold to classify as bear
        """
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.lookback_periods = lookback_periods
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold

    def detect_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime classification to dataframe.

        Args:
            df: DataFrame with 'Close' prices

        Returns:
            DataFrame with 'regime' column
        """
        data = df.copy()

        # Calculate SMAs
        data['sma_fast'] = data['Close'].rolling(window=self.sma_fast).mean()
        data['sma_slow'] = data['Close'].rolling(window=self.sma_slow).mean()

        # Trend: Fast SMA above slow SMA = uptrend
        data['trend_up'] = (data['sma_fast'] > data['sma_slow']).astype(int)

        # Momentum: Return over lookback period
        data['momentum'] = data['Close'].pct_change(self.lookback_periods)

        # Price position relative to 200-day SMA
        data['price_vs_sma200'] = (data['Close'] / data['sma_slow'] - 1)

        # Classify regime
        def classify_regime(row):
            """Classify single row into regime."""
            if pd.isna(row['momentum']) or pd.isna(row['sma_slow']):
                return MarketRegime.UNKNOWN.value

            # BULL: Uptrend + positive momentum + price above 200 SMA
            if (row['trend_up'] == 1 and
                row['momentum'] > self.bull_threshold and
                row['price_vs_sma200'] > 0):
                return MarketRegime.BULL.value

            # BEAR: Downtrend + negative momentum + price below 200 SMA
            elif (row['trend_up'] == 0 and
                  row['momentum'] < self.bear_threshold and
                  row['price_vs_sma200'] < -0.05):  # At least 5% below 200 SMA
                return MarketRegime.BEAR.value

            # SIDEWAYS: Everything else
            else:
                return MarketRegime.SIDEWAYS.value

        data['regime'] = data.apply(classify_regime, axis=1)

        return data

    def get_current_regime(self, df: pd.DataFrame) -> MarketRegime:
        """
        Get the most recent market regime.

        Args:
            df: DataFrame with regime column (from detect_regime)

        Returns:
            Current regime enum
        """
        if 'regime' not in df.columns:
            df = self.detect_regime(df)

        latest_regime = df['regime'].iloc[-1]
        return MarketRegime(latest_regime)

    def should_trade_mean_reversion(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Determine if mean reversion trading should be enabled.

        Args:
            df: DataFrame with price data

        Returns:
            Tuple of (should_trade, reason)
        """
        regime = self.get_current_regime(df)

        if regime == MarketRegime.BULL:
            return True, "Bull market: Mean reversion safe to trade"

        elif regime == MarketRegime.BEAR:
            return False, "Bear market: AVOID mean reversion (panic sells often justified)"

        elif regime == MarketRegime.SIDEWAYS:
            return True, "Sideways market: Mean reversion may work with caution"

        else:
            return False, "Unknown regime: Insufficient data"

    def analyze_regime_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Analyze regime distribution and characteristics.

        Args:
            df: DataFrame with regime column

        Returns:
            Dictionary with regime statistics
        """
        if 'regime' not in df.columns:
            df = self.detect_regime(df)

        total_days = len(df[df['regime'] != MarketRegime.UNKNOWN.value])

        regime_counts = df['regime'].value_counts()

        stats = {
            'total_days': total_days,
            'bull_days': regime_counts.get(MarketRegime.BULL.value, 0),
            'bear_days': regime_counts.get(MarketRegime.BEAR.value, 0),
            'sideways_days': regime_counts.get(MarketRegime.SIDEWAYS.value, 0),
            'bull_pct': (regime_counts.get(MarketRegime.BULL.value, 0) / total_days * 100) if total_days > 0 else 0,
            'bear_pct': (regime_counts.get(MarketRegime.BEAR.value, 0) / total_days * 100) if total_days > 0 else 0,
            'sideways_pct': (regime_counts.get(MarketRegime.SIDEWAYS.value, 0) / total_days * 100) if total_days > 0 else 0,
        }

        # Calculate returns by regime
        for regime_value in [MarketRegime.BULL.value, MarketRegime.BEAR.value, MarketRegime.SIDEWAYS.value]:
            regime_data = df[df['regime'] == regime_value]
            if len(regime_data) > 1:
                regime_return = (regime_data['Close'].iloc[-1] / regime_data['Close'].iloc[0] - 1) * 100
                stats[f'{regime_value}_return'] = regime_return
            else:
                stats[f'{regime_value}_return'] = 0

        return stats


def add_regime_filter_to_signals(df: pd.DataFrame, detector: MarketRegimeDetector = None) -> pd.DataFrame:
    """
    Add regime filter to trading signals.

    Disables signals in bear market.

    Args:
        df: DataFrame with trading signals
        detector: MarketRegimeDetector instance (creates default if None)

    Returns:
        DataFrame with regime-filtered signals
    """
    if detector is None:
        detector = MarketRegimeDetector()

    data = df.copy()

    # Detect regime
    data = detector.detect_regime(data)

    # Disable signals in bear market
    if 'panic_sell' in data.columns:
        data['panic_sell_original'] = data['panic_sell'].copy()
        data.loc[data['regime'] == MarketRegime.BEAR.value, 'panic_sell'] = 0

    if 'panic_buy' in data.columns:
        data['panic_buy_original'] = data['panic_buy'].copy()
        data.loc[data['regime'] == MarketRegime.BEAR.value, 'panic_buy'] = 0

    return data


if __name__ == "__main__":
    print("Market Regime Detector module loaded")
    print()
    print("Regimes:")
    print("  - BULL: Uptrend, mean reversion works")
    print("  - BEAR: Downtrend, avoid mean reversion")
    print("  - SIDEWAYS: Range-bound, trade with caution")
    print()
    print("Usage:")
    print("  detector = MarketRegimeDetector()")
    print("  data = detector.detect_regime(df)")
    print("  should_trade, reason = detector.should_trade_mean_reversion(data)")

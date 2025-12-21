"""
Market Regime Detector

Classifies market conditions to adjust signal filtering:
- BULL: Strong uptrend - mean reversion less reliable, reduce position sizes
- BEAR: Strong downtrend - mean reversion can catch falling knives, be selective
- CHOPPY: Range-bound - ideal for mean reversion, full position sizes
- VOLATILE: High VIX/uncertainty - widen stops, reduce positions

Uses SPY as market proxy with multiple timeframe analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import yfinance as yf


class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    CHOPPY = "choppy"
    VOLATILE = "volatile"


@dataclass
class RegimeAnalysis:
    """Complete regime analysis output."""
    regime: MarketRegime
    confidence: float  # 0-1
    spy_trend_20d: float  # % change
    spy_trend_50d: float
    vix_level: float
    vix_percentile: float  # vs last year
    adv_decline_ratio: float
    breadth_score: float  # % stocks above 20d MA
    recommendation: str  # Trading recommendation


class MarketRegimeDetector:
    """
    Detects current market regime using multiple indicators.
    """

    def __init__(self):
        self.spy_data: Optional[pd.DataFrame] = None
        self.vix_data: Optional[pd.DataFrame] = None

    def fetch_data(self, days_back: int = 252) -> bool:
        """Fetch SPY and VIX data."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back + 50)

            # SPY for trend
            spy = yf.Ticker("SPY")
            self.spy_data = spy.history(start=start_date, end=end_date)

            # VIX for volatility
            vix = yf.Ticker("^VIX")
            self.vix_data = vix.history(start=start_date, end=end_date)

            return len(self.spy_data) > 50 and len(self.vix_data) > 50

        except Exception as e:
            print(f"[ERROR] Failed to fetch market data: {e}")
            return False

    def calculate_trend(self, prices: pd.Series, period: int) -> float:
        """Calculate trend as % change over period."""
        if len(prices) < period:
            return 0.0
        return ((prices.iloc[-1] / prices.iloc[-period]) - 1) * 100

    def calculate_volatility_regime(self) -> Tuple[float, float]:
        """
        Calculate VIX level and percentile.
        Returns (current_vix, percentile_rank).
        """
        if self.vix_data is None or len(self.vix_data) < 20:
            return 20.0, 50.0

        current_vix = self.vix_data['Close'].iloc[-1]
        vix_1y = self.vix_data['Close'].tail(252)
        percentile = (vix_1y < current_vix).sum() / len(vix_1y) * 100

        return current_vix, percentile

    def calculate_trend_strength(self) -> Tuple[float, float, float]:
        """
        Calculate trend strength using multiple MAs.
        Returns (trend_20d, trend_50d, ma_alignment_score).
        """
        if self.spy_data is None or len(self.spy_data) < 50:
            return 0.0, 0.0, 0.0

        close = self.spy_data['Close']

        # Trend calculations
        trend_20d = self.calculate_trend(close, 20)
        trend_50d = self.calculate_trend(close, 50)

        # MA alignment: price vs 20 MA vs 50 MA
        ma_20 = close.rolling(20).mean().iloc[-1]
        ma_50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]

        # Score: +1 for each bullish alignment, -1 for bearish
        alignment = 0
        if current > ma_20:
            alignment += 1
        else:
            alignment -= 1
        if ma_20 > ma_50:
            alignment += 1
        else:
            alignment -= 1
        if current > ma_50:
            alignment += 1
        else:
            alignment -= 1

        return trend_20d, trend_50d, alignment / 3.0

    def calculate_momentum_breadth(self) -> float:
        """
        Calculate internal momentum using RSI.
        Returns RSI deviation from 50 (neutral).
        """
        if self.spy_data is None or len(self.spy_data) < 20:
            return 0.0

        close = self.spy_data['Close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]
        return (current_rsi - 50) / 50  # -1 to +1

    def detect_regime(self) -> RegimeAnalysis:
        """
        Detect current market regime.
        """
        # Fetch fresh data
        if not self.fetch_data():
            return RegimeAnalysis(
                regime=MarketRegime.CHOPPY,
                confidence=0.0,
                spy_trend_20d=0.0,
                spy_trend_50d=0.0,
                vix_level=20.0,
                vix_percentile=50.0,
                adv_decline_ratio=1.0,
                breadth_score=0.5,
                recommendation="Unable to fetch data - using defaults"
            )

        # Calculate indicators
        trend_20d, trend_50d, ma_alignment = self.calculate_trend_strength()
        vix_level, vix_percentile = self.calculate_volatility_regime()
        momentum = self.calculate_momentum_breadth()

        # Regime classification logic
        regime = MarketRegime.CHOPPY
        confidence = 0.5

        # High volatility overrides other regimes
        if vix_level > 30 or vix_percentile > 85:
            regime = MarketRegime.VOLATILE
            confidence = min(0.95, 0.6 + (vix_level - 25) / 50)

        # Strong uptrend
        elif trend_20d > 3 and trend_50d > 5 and ma_alignment > 0.5:
            regime = MarketRegime.BULL
            confidence = min(0.95, 0.5 + trend_20d / 20 + ma_alignment * 0.3)

        # Strong downtrend
        elif trend_20d < -3 and trend_50d < -5 and ma_alignment < -0.5:
            regime = MarketRegime.BEAR
            confidence = min(0.95, 0.5 + abs(trend_20d) / 20 + abs(ma_alignment) * 0.3)

        # Choppy/range-bound
        else:
            regime = MarketRegime.CHOPPY
            # Higher confidence if truly range-bound (small trends, low VIX)
            trend_magnitude = abs(trend_20d) + abs(trend_50d)
            if trend_magnitude < 5 and vix_level < 20:
                confidence = 0.8
            else:
                confidence = 0.6

        # Generate recommendation
        recommendations = {
            MarketRegime.BULL: "Reduce mean reversion size. Trends persist - consider momentum instead.",
            MarketRegime.BEAR: "Be very selective. Only take signals with strength > 70. Widen stops.",
            MarketRegime.CHOPPY: "Ideal for mean reversion. Use full position sizes.",
            MarketRegime.VOLATILE: "Reduce all positions 50%. Widen stops to 3.5%. Cash is a position."
        }

        return RegimeAnalysis(
            regime=regime,
            confidence=round(confidence, 2),
            spy_trend_20d=round(trend_20d, 2),
            spy_trend_50d=round(trend_50d, 2),
            vix_level=round(vix_level, 1),
            vix_percentile=round(vix_percentile, 1),
            adv_decline_ratio=1.0,  # Placeholder - could add breadth data
            breadth_score=round((momentum + 1) / 2, 2),  # Normalize to 0-1
            recommendation=recommendations[regime]
        )

    def get_position_multiplier(self, regime: MarketRegime) -> float:
        """
        Get position size multiplier based on regime.
        """
        multipliers = {
            MarketRegime.BULL: 0.7,      # Reduce - trends persist
            MarketRegime.BEAR: 0.5,      # Reduce - falling knives
            MarketRegime.CHOPPY: 1.0,    # Full size - ideal conditions
            MarketRegime.VOLATILE: 0.5   # Reduce - high uncertainty
        }
        return multipliers.get(regime, 1.0)

    def get_stop_adjustment(self, regime: MarketRegime) -> float:
        """
        Get stop loss adjustment factor based on regime.
        > 1.0 means widen stops.
        """
        adjustments = {
            MarketRegime.BULL: 1.0,      # Normal stops
            MarketRegime.BEAR: 1.25,     # Widen 25%
            MarketRegime.CHOPPY: 1.0,    # Normal stops
            MarketRegime.VOLATILE: 1.4   # Widen 40%
        }
        return adjustments.get(regime, 1.0)

    def get_min_signal_threshold(self, regime: MarketRegime) -> float:
        """
        Get minimum signal strength threshold based on regime.
        """
        thresholds = {
            MarketRegime.BULL: 60.0,     # Higher bar
            MarketRegime.BEAR: 70.0,     # Much higher bar
            MarketRegime.CHOPPY: 50.0,   # Normal
            MarketRegime.VOLATILE: 65.0  # Higher bar
        }
        return thresholds.get(regime, 50.0)

    def get_dynamic_threshold(self, base_threshold: float = 50.0) -> Dict[str, float]:
        """
        Get dynamically adjusted thresholds based on current market conditions.

        Returns dict with:
        - threshold: Adjusted signal strength threshold
        - position_mult: Position size multiplier
        - stop_mult: Stop loss multiplier
        - vix_adjustment: VIX-based adjustment factor
        - confidence: Confidence in the adjustment
        """
        if self.vix_data is None:
            self.fetch_data()

        # Get current VIX
        vix_level, vix_percentile = self.calculate_volatility_regime()

        # VIX-based threshold adjustment
        # Low VIX (<15): Reduce threshold by up to 10%
        # Normal VIX (15-25): No change
        # High VIX (25-35): Increase threshold by up to 20%
        # Extreme VIX (>35): Increase threshold by up to 40%

        if vix_level < 15:
            vix_adj = 1.0 - (15 - vix_level) / 50  # Up to -10%
        elif vix_level <= 25:
            vix_adj = 1.0
        elif vix_level <= 35:
            vix_adj = 1.0 + (vix_level - 25) / 50  # Up to +20%
        else:
            vix_adj = 1.2 + (vix_level - 35) / 50  # +20% to +40%

        # Trend-based adjustment
        trend_20d, trend_50d, ma_alignment = self.calculate_trend_strength()

        # In strong trends (>5%), increase threshold
        trend_magnitude = abs(trend_20d)
        if trend_magnitude > 5:
            trend_adj = 1.0 + (trend_magnitude - 5) / 20  # Up to +25%
        else:
            trend_adj = 1.0

        # Combined adjustment (cap at 1.5x)
        combined_adj = min(1.5, vix_adj * trend_adj)

        # Calculate final values
        adjusted_threshold = base_threshold * combined_adj

        # Position multiplier (inverse of threshold adjustment)
        position_mult = max(0.3, 1.0 / combined_adj)

        # Stop multiplier (wider in volatile markets)
        stop_mult = 1.0 + max(0, (vix_level - 20) / 40)  # Up to 1.5x at VIX 40

        return {
            'threshold': round(adjusted_threshold, 1),
            'base_threshold': base_threshold,
            'position_mult': round(position_mult, 2),
            'stop_mult': round(stop_mult, 2),
            'vix_adjustment': round(vix_adj, 3),
            'trend_adjustment': round(trend_adj, 3),
            'combined_adjustment': round(combined_adj, 3),
            'vix_level': round(vix_level, 1),
            'vix_percentile': round(vix_percentile, 1),
            'trend_20d': round(trend_20d, 2),
            'confidence': round(min(0.95, 0.5 + (1 - abs(1 - combined_adj))), 2)
        }

    def get_volatility_adjusted_exits(self, base_profit: float = 2.0,
                                       base_stop: float = -2.0) -> Dict[str, float]:
        """
        Get volatility-adjusted exit parameters.

        In high volatility:
        - Widen profit targets (more room to run)
        - Widen stop losses (avoid whipsaws)
        - Tighten trailing stops trigger (lock in gains faster)
        """
        if self.vix_data is None:
            self.fetch_data()

        vix_level, _ = self.calculate_volatility_regime()

        # Volatility multiplier (1.0 at VIX 20, scales up/down)
        vol_mult = vix_level / 20.0
        vol_mult = max(0.7, min(2.0, vol_mult))  # Cap between 0.7x and 2.0x

        adjusted_profit = base_profit * vol_mult
        adjusted_stop = base_stop * vol_mult

        # Trailing stop adjustments
        # Lower VIX = tighter trailing (less whipsaw risk)
        # Higher VIX = wider trailing (more whipsaw risk)
        trailing_trigger = 1.0 + (vix_level - 20) / 40  # 0.5% to 1.5%
        trailing_distance = 0.5 + (vix_level - 20) / 60  # 0.3% to 0.8%

        trailing_trigger = max(0.5, min(2.0, trailing_trigger))
        trailing_distance = max(0.3, min(1.0, trailing_distance))

        return {
            'profit_target': round(adjusted_profit, 2),
            'stop_loss': round(adjusted_stop, 2),
            'trailing_trigger': round(trailing_trigger, 2),
            'trailing_distance': round(trailing_distance, 2),
            'vix_level': round(vix_level, 1),
            'volatility_multiplier': round(vol_mult, 2)
        }


def get_current_regime() -> RegimeAnalysis:
    """Quick function to get current regime."""
    detector = MarketRegimeDetector()
    return detector.detect_regime()


def print_regime_report():
    """Print detailed regime analysis."""
    print("=" * 70)
    print("MARKET REGIME ANALYSIS")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    detector = MarketRegimeDetector()
    analysis = detector.detect_regime()

    print(f"REGIME: {analysis.regime.value.upper()}")
    print(f"Confidence: {analysis.confidence * 100:.0f}%")
    print()

    print("--- INDICATORS ---")
    print(f"SPY 20-day trend: {analysis.spy_trend_20d:+.2f}%")
    print(f"SPY 50-day trend: {analysis.spy_trend_50d:+.2f}%")
    print(f"VIX Level: {analysis.vix_level:.1f}")
    print(f"VIX Percentile (1Y): {analysis.vix_percentile:.0f}%")
    print(f"Breadth Score: {analysis.breadth_score:.2f}")
    print()

    print("--- TRADING ADJUSTMENTS ---")
    print(f"Position Multiplier: {detector.get_position_multiplier(analysis.regime):.1f}x")
    print(f"Stop Adjustment: {detector.get_stop_adjustment(analysis.regime):.2f}x")
    print(f"Min Signal Threshold: {detector.get_min_signal_threshold(analysis.regime):.0f}")
    print()

    print("--- RECOMMENDATION ---")
    print(analysis.recommendation)
    print()

    return analysis


if __name__ == "__main__":
    print_regime_report()

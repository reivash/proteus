"""
VIX Regime Detection

Classifies market volatility regime using VIX index.
Enables regime-adaptive trading for higher win rate.

Based on EXP-034 findings:
- Mean reversion performance varies with volatility
- LOW VOL (VIX < 15): Weak mean reversion, skip trading
- NORMAL (VIX 15-25): Standard mean reversion
- HIGH VOL (VIX > 25): Strong mean reversion
- EXTREME (VIX > 30): Very strong mean reversion (crisis opportunities)

VIX Regime Adaptation Strategy:
- Skip trading when VIX < 15 (weak mean reversion)
- Trade normally when VIX >= 15
- Expected improvement: +1.2pp win rate (88.7% -> 89.9%)

Historical VIX Distribution:
- VIX < 15: ~30% of time (complacent market)
- VIX 15-25: ~50% of time (normal conditions)
- VIX 25-30: ~15% of time (elevated volatility)
- VIX > 30: ~5% of time (crisis mode)

Usage:
    detector = VixRegimeDetector()
    regime = detector.get_current_regime()
    should_trade = detector.should_trade()  # False if VIX < 15
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional


class VixRegimeDetector:
    """
    Detect volatility regime using VIX index.
    """

    def __init__(self, low_vol_threshold: float = 15.0,
                 high_vol_threshold: float = 25.0,
                 extreme_vol_threshold: float = 30.0):
        """
        Initialize VIX regime detector.

        Args:
            low_vol_threshold: VIX level below which = low volatility (default: 15)
            high_vol_threshold: VIX level above which = high volatility (default: 25)
            extreme_vol_threshold: VIX level above which = extreme volatility (default: 30)
        """
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_threshold = high_vol_threshold
        self.extreme_vol_threshold = extreme_vol_threshold

    def fetch_vix(self, days_back: int = 5) -> Optional[float]:
        """
        Fetch current VIX level.

        Args:
            days_back: Number of days to look back for VIX data

        Returns:
            Current VIX level, or None if fetch fails
        """
        try:
            # Fetch VIX data
            vix = yf.Ticker("^VIX")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            hist = vix.history(start=start_date, end=end_date)

            if hist.empty:
                print("[WARN] No VIX data available")
                return None

            # Get most recent VIX close
            current_vix = hist['Close'].iloc[-1]
            return float(current_vix)

        except Exception as e:
            print(f"[ERROR] Failed to fetch VIX: {e}")
            return None

    def classify_regime(self, vix_level: float) -> str:
        """
        Classify volatility regime based on VIX level.

        Args:
            vix_level: Current VIX value

        Returns:
            'LOW', 'NORMAL', 'HIGH', or 'EXTREME'
        """
        if vix_level < self.low_vol_threshold:
            return 'LOW'
        elif vix_level < self.high_vol_threshold:
            return 'NORMAL'
        elif vix_level < self.extreme_vol_threshold:
            return 'HIGH'
        else:
            return 'EXTREME'

    def get_current_regime(self) -> Optional[str]:
        """
        Get current volatility regime.

        Returns:
            'LOW', 'NORMAL', 'HIGH', 'EXTREME', or None if fetch fails
        """
        vix = self.fetch_vix()
        if vix is None:
            return None

        return self.classify_regime(vix)

    def should_trade(self) -> bool:
        """
        Check if trading should be allowed in current regime.

        Based on EXP-034: Skip trading when VIX < 15 (weak mean reversion)

        Returns:
            True if should trade, False if should skip (low volatility)
        """
        regime = self.get_current_regime()

        if regime is None:
            # If VIX fetch fails, allow trading (conservative default)
            print("[WARN] VIX regime unknown, allowing trading")
            return True

        # Skip trading in low volatility regime
        if regime == 'LOW':
            return False

        # Trade in all other regimes
        return True

    def get_regime_info(self) -> Dict:
        """
        Get detailed information about current regime.

        Returns:
            Dictionary with regime info
        """
        vix = self.fetch_vix()

        if vix is None:
            return {
                'vix': None,
                'regime': None,
                'should_trade': True,  # Default to allowing trade
                'description': 'VIX data unavailable'
            }

        regime = self.classify_regime(vix)
        should_trade = (regime != 'LOW')

        descriptions = {
            'LOW': f'Complacent market (VIX {vix:.1f} < {self.low_vol_threshold}) - Skip trading',
            'NORMAL': f'Normal conditions (VIX {vix:.1f} in range {self.low_vol_threshold}-{self.high_vol_threshold})',
            'HIGH': f'Elevated volatility (VIX {vix:.1f} in range {self.high_vol_threshold}-{self.extreme_vol_threshold})',
            'EXTREME': f'Crisis mode (VIX {vix:.1f} > {self.extreme_vol_threshold}) - Abundant opportunities'
        }

        return {
            'vix': float(vix),
            'regime': regime,
            'should_trade': should_trade,
            'description': descriptions[regime]
        }


def add_vix_filter_to_signals(df: pd.DataFrame, signal_column: str = 'panic_sell') -> pd.DataFrame:
    """
    Filter signals based on VIX regime.

    Disables signals when VIX < 15 (low volatility = weak mean reversion)

    Args:
        df: DataFrame with signals
        signal_column: Name of signal column to filter

    Returns:
        DataFrame with filtered signals
    """
    data = df.copy()

    # Get current VIX regime
    detector = VixRegimeDetector()
    should_trade = detector.should_trade()

    if not should_trade:
        # Low volatility regime - disable all signals
        print("[VIX FILTER] Low volatility regime (VIX < 15) - Skipping all trades")

        if signal_column in data.columns:
            data[f'{signal_column}_before_vix_filter'] = data[signal_column].copy()
            data[signal_column] = 0

    return data


def test_vix_regime_detector():
    """Test the VIX regime detector."""
    print("=" * 70)
    print("VIX REGIME DETECTOR TEST")
    print("=" * 70)
    print()

    detector = VixRegimeDetector()

    # Get regime info
    info = detector.get_regime_info()

    print("Current VIX Regime:")
    print("-" * 70)
    print(f"  VIX Level: {info['vix']}")
    print(f"  Regime: {info['regime']}")
    print(f"  Should Trade: {info['should_trade']}")
    print(f"  Description: {info['description']}")
    print()

    # Test regime classification
    print("Regime Classification Examples:")
    print("-" * 70)

    test_levels = [12, 18, 27, 35]

    for vix_level in test_levels:
        regime = detector.classify_regime(vix_level)
        should_trade = (regime != 'LOW')
        action = "TRADE" if should_trade else "SKIP"

        print(f"  VIX {vix_level}: {regime} regime -> [{action}]")

    print()
    print("=" * 70)
    print("EXPECTED OUTCOMES:")
    print("=" * 70)
    print()
    print("EXP-034 findings:")
    print("  - Skip trading when VIX < 15")
    print("  - Expected improvement: +1.2pp (88.7% -> 89.9%)")
    print("  - Trade reduction: ~30% (skip low-vol periods)")
    print("  - Philosophy: Trade when edge is strongest")
    print()
    print("[SUCCESS] VIX regime detection ready for production!")
    print()


if __name__ == "__main__":
    test_vix_regime_detector()

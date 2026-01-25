"""
Enhanced Regime Detector
========================

Wraps UnifiedRegimeDetector with FastBearDetector for early warning support.

This module provides a single entry point for complete regime detection:
- HMM + Rule-based ensemble detection
- FastBear early warning for rapid market declines
- Combined recommendations

Usage:
    from analysis.enhanced_regime_detector import get_enhanced_regime
    result = get_enhanced_regime()
    print(f"Regime: {result['regime']}")
    if result['early_warning']:
        print(f"WARNING: Bear score {result['bear_score']}")
"""

import os
import sys
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.unified_regime_detector import (
    UnifiedRegimeDetector,
    UnifiedRegimeResult,
    DetectionMethod,
    get_current_regime
)
from analysis.fast_bear_detector import (
    FastBearDetector,
    FastBearSignal,
    get_fast_bear_signal
)


@dataclass
class EnhancedRegimeResult:
    """Complete regime detection with early warning."""
    # Core regime info
    regime: str
    confidence: float
    hmm_regime: str
    rule_regime: str
    agreement: bool
    vix_level: float

    # Early warning from FastBear
    bear_score: float
    bear_alert_level: str
    bear_triggers: list
    early_warning: bool

    # Combined recommendation
    recommendation: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'regime': self.regime,
            'confidence': self.confidence,
            'hmm_regime': self.hmm_regime,
            'rule_regime': self.rule_regime,
            'agreement': self.agreement,
            'vix_level': self.vix_level,
            'bear_score': self.bear_score,
            'bear_alert_level': self.bear_alert_level,
            'bear_triggers': self.bear_triggers,
            'early_warning': self.early_warning,
            'recommendation': self.recommendation
        }


class EnhancedRegimeDetector:
    """
    Enhanced regime detector combining:
    - UnifiedRegimeDetector (HMM + Rule-based)
    - FastBearDetector (early warning)

    Provides complete market assessment with early warning support.
    """

    def __init__(self, method: str = 'ensemble'):
        method_map = {
            'hmm': DetectionMethod.HMM,
            'rule': DetectionMethod.RULE_BASED,
            'ensemble': DetectionMethod.ENSEMBLE
        }
        self.unified = UnifiedRegimeDetector(method=method_map.get(method, DetectionMethod.ENSEMBLE))
        self.fast_bear = FastBearDetector()

    def detect(self) -> EnhancedRegimeResult:
        """
        Run complete regime detection.

        Returns:
            EnhancedRegimeResult with regime and early warning info
        """
        # Get unified regime
        regime_result = self.unified.detect_regime()

        # Get fast bear signal
        bear_signal = self.fast_bear.detect()

        # Determine early warning status
        early_warning = bear_signal.bear_score >= 30 or bear_signal.alert_level in ['WARNING', 'CRITICAL']

        # Build recommendation
        recommendation = regime_result.recommendation

        if early_warning:
            if bear_signal.alert_level == 'CRITICAL':
                recommendation = (
                    f"CRITICAL BEAR WARNING (Score: {bear_signal.bear_score}/100). "
                    f"Triggers: {len(bear_signal.triggers)}. "
                    f"{recommendation}"
                )
            elif bear_signal.alert_level == 'WARNING':
                recommendation = (
                    f"BEAR WARNING (Score: {bear_signal.bear_score}/100). "
                    f"Monitor closely. {recommendation}"
                )
            elif bear_signal.alert_level == 'WATCH':
                recommendation = (
                    f"Bear Watch (Score: {bear_signal.bear_score}/100). "
                    f"{recommendation}"
                )

        return EnhancedRegimeResult(
            regime=regime_result.regime,
            confidence=regime_result.confidence,
            hmm_regime=regime_result.hmm_regime,
            rule_regime=regime_result.rule_regime,
            agreement=regime_result.agreement,
            vix_level=regime_result.vix_level,
            bear_score=bear_signal.bear_score,
            bear_alert_level=bear_signal.alert_level,
            bear_triggers=bear_signal.triggers,
            early_warning=early_warning,
            recommendation=recommendation
        )


def get_enhanced_regime(method: str = 'ensemble') -> EnhancedRegimeResult:
    """
    Quick function to get enhanced regime with early warning.

    Args:
        method: 'hmm', 'rule', or 'ensemble' (default)

    Returns:
        EnhancedRegimeResult
    """
    detector = EnhancedRegimeDetector(method=method)
    return detector.detect()


def print_enhanced_regime():
    """Print complete regime analysis with early warning."""
    print("=" * 70)
    print("ENHANCED REGIME DETECTION")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    result = get_enhanced_regime()

    print("--- REGIME DETECTION ---")
    print(f"Regime: {result.regime.upper()}")
    print(f"Confidence: {result.confidence * 100:.1f}%")
    print(f"HMM says: {result.hmm_regime.upper()}")
    print(f"Rule-based says: {result.rule_regime.upper()}")
    print(f"Agreement: {'YES' if result.agreement else 'NO'}")
    print(f"VIX: {result.vix_level:.1f}")
    print()

    print("--- EARLY WARNING (FastBear) ---")
    print(f"Bear Score: {result.bear_score}/100")
    print(f"Alert Level: {result.bear_alert_level}")

    if result.bear_triggers:
        print(f"Active Triggers:")
        for trigger in result.bear_triggers:
            print(f"  - {trigger}")

    if result.early_warning:
        print()
        print("*** EARLY WARNING ACTIVE ***")
    print()

    print("--- RECOMMENDATION ---")
    print(result.recommendation)
    print()

    return result


if __name__ == "__main__":
    print_enhanced_regime()

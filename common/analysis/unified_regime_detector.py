"""
Unified Regime Detector
=======================

Combines HMM-based and rule-based regime detection for more robust classification.

Strategy:
1. Use HMM as primary detector (learned patterns, probabilistic)
2. Use rule-based as validation (interpretable, VIX-aware)
3. When they disagree, use consensus logic

Benefits:
- HMM catches regime shifts earlier
- Rule-based provides explainability
- Combined approach reduces false signals

Jan 2026 - Created for Proteus trading system.
"""

import os
import sys
from datetime import datetime
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.hmm_regime_detector import HMMRegimeDetector, HMMRegimeResult
from analysis.market_regime import MarketRegimeDetector, MarketRegime, RegimeAnalysis
from analysis.fast_bear_detector import FastBearDetector, FastBearSignal
from analysis.hierarchical_hmm import HierarchicalHMM, HierarchicalRegimeResult
from data.fetchers.fred_macro import FREDMacroFetcher, MacroIndicators
from analysis.correlation_regime import CorrelationRegimeTracker, CorrelationRegimeResult


class DetectionMethod(Enum):
    """Regime detection method."""
    HMM = "hmm"              # HMM-based (primary)
    RULE_BASED = "rule"      # Rule-based (VIX + trends)
    ENSEMBLE = "ensemble"    # Combined approach


@dataclass
class UnifiedRegimeResult:
    """Combined regime detection result."""
    regime: str               # Final regime classification
    confidence: float         # Combined confidence
    hmm_regime: str           # HMM classification
    hmm_confidence: float
    rule_regime: str          # Rule-based classification
    rule_confidence: float
    agreement: bool           # Do methods agree?
    vix_level: float
    transition_signal: str
    recommendation: str
    method_used: str          # Which method determined final result
    # Quick Win #1: HMM posterior probabilities (Jan 2026)
    hmm_probabilities: dict = None  # Full posterior: {'volatile': 0.1, 'bear': 0.2, ...}
    days_in_regime: int = 0         # Days since last regime change
    # Quick Win #2: VIX term structure (Jan 2026)
    vix_term_structure: float = 1.0  # VIX/VIX3M ratio (>1 = backwardation/fear, <1 = contango/complacent)
    vix_3m: float = 0.0              # 3-month VIX futures proxy
    # Fast bear early warning (Jan 2026)
    early_warning: bool = False        # True if bear score >= 30
    early_warning_score: float = 0.0   # Bear score 0-100
    early_warning_level: str = "NORMAL"  # NORMAL, WATCH, WARNING, CRITICAL
    early_warning_triggers: list = None  # Active triggers
    data_error: bool = False           # True if data fetch failed
    # Hierarchical HMM (Jan 2026 - based on MDPI 2025 research)
    meta_regime: str = "low_uncertainty"  # HIGH/LOW uncertainty (structural)
    meta_confidence: float = 0.5
    hierarchical_risk_mult: float = 1.0   # Position size multiplier from HHMM
    # FRED Macroeconomic Indicators (Jan 2026)
    recession_probability: float = 0.0    # 0-100%
    recession_signal: str = "LOW"         # LOW, MODERATE, ELEVATED, HIGH
    macro_risk_mult: float = 1.0          # Position size multiplier from macro
    # Model Disagreement Signal (Jan 2026)
    model_disagreement: float = 0.0       # 0.0-1.0 (0=agree, 1=max disagreement)
    disagreement_risk_mult: float = 1.0   # Position size multiplier
    disagreement_note: str = ""           # Details on disagreement
    # Correlation Regime (Jan 2026)
    correlation_regime: str = "NORMAL"    # NORMAL, ELEVATED, CRISIS
    avg_correlation: float = 0.5          # Average SPY correlation
    correlation_risk_mult: float = 1.0    # Position size multiplier


class UnifiedRegimeDetector:
    """
    Unified regime detector combining HMM and rule-based approaches.

    Usage:
        detector = UnifiedRegimeDetector(method=DetectionMethod.ENSEMBLE)
        result = detector.detect_regime()
        print(f"Regime: {result.regime}, Confidence: {result.confidence}")
    """

    def __init__(self, method: DetectionMethod = DetectionMethod.ENSEMBLE):
        self.method = method
        self.hmm_detector = HMMRegimeDetector()
        self.rule_detector = MarketRegimeDetector()
        self.bear_detector = FastBearDetector()
        self.hierarchical_hmm = HierarchicalHMM()  # MDPI 2025 research-based
        self.macro_fetcher = FREDMacroFetcher()    # Chen 2009 research-based
        self.correlation_tracker = CorrelationRegimeTracker()  # Crisis detection

    def detect_regime(self) -> UnifiedRegimeResult:
        """
        Detect current market regime using configured method.

        Returns:
            UnifiedRegimeResult with regime classification and metadata
        """
        # Get HMM result
        hmm_result = self._get_hmm_result()

        # Get rule-based result
        rule_result = self._get_rule_result()

        # Get fast bear early warning
        bear_signal = self._get_bear_signal()

        # Get hierarchical HMM result (MDPI 2025 research)
        hierarchical_result = self._get_hierarchical_result()

        # Combine based on method
        if self.method == DetectionMethod.HMM:
            result = self._use_hmm(hmm_result, rule_result)
        elif self.method == DetectionMethod.RULE_BASED:
            result = self._use_rule_based(hmm_result, rule_result)
        else:
            result = self._use_ensemble(hmm_result, rule_result)

        # Add early warning data
        result.early_warning = bear_signal.bear_score >= 30
        result.early_warning_score = bear_signal.bear_score
        result.early_warning_level = bear_signal.alert_level
        result.early_warning_triggers = bear_signal.triggers

        # Add hierarchical HMM data (structural regime awareness)
        result.meta_regime = hierarchical_result.meta_regime
        result.meta_confidence = hierarchical_result.meta_confidence
        result.hierarchical_risk_mult = hierarchical_result.risk_multiplier

        # Get FRED macro indicators (Chen 2009 research)
        macro_indicators = self._get_macro_indicators()
        result.recession_probability = macro_indicators.recession_probability
        result.recession_signal = macro_indicators.recession_signal
        # Map recession signal to risk multiplier
        macro_mult_map = {'LOW': 1.0, 'MODERATE': 0.85, 'ELEVATED': 0.65, 'HIGH': 0.40}
        result.macro_risk_mult = macro_mult_map.get(macro_indicators.recession_signal, 0.85)

        # Calculate model disagreement signal
        disagreement, disagree_mult, disagree_note = self._calculate_model_disagreement(
            hmm_result, rule_result, hierarchical_result
        )
        result.model_disagreement = disagreement
        result.disagreement_risk_mult = disagree_mult
        result.disagreement_note = disagree_note

        # Get correlation regime
        corr_result = self._get_correlation_regime()
        result.correlation_regime = corr_result.correlation_regime
        result.avg_correlation = corr_result.avg_correlation
        result.correlation_risk_mult = corr_result.risk_multiplier

        # Get VIX term structure (Quick Win #2)
        vix_term, vix_3m = self._get_vix_term_structure(result.vix_level)
        result.vix_term_structure = vix_term
        result.vix_3m = vix_3m

        return result

    def _get_bear_signal(self) -> FastBearSignal:
        """Get fast bear early warning signal."""
        try:
            return self.bear_detector.detect()
        except Exception as e:
            print(f"[REGIME] Bear detection failed: {e}")
            return FastBearSignal(
                timestamp=datetime.now().isoformat(),
                bear_score=0.0,
                alert_level='NORMAL',
                confidence=0.0,
                triggers=[],
                vix_level=20.0,
                vix_spike_pct=0.0,
                market_breadth_pct=50.0,
                spy_roc_3d=0.0,
                sectors_declining=0,
                sectors_total=11,
                volume_confirmation=False,
                yield_curve_spread=0.5,
                credit_spread_change=0.0,
                momentum_divergence=False,
                recommendation='Bear detection unavailable'
            )

    def _get_hmm_result(self) -> HMMRegimeResult:
        """Get HMM regime detection."""
        try:
            return self.hmm_detector.detect_regime()
        except Exception as e:
            print(f"[REGIME] HMM detection failed: {e}")
            return HMMRegimeResult(
                regime='choppy',
                regime_id=2,
                confidence=0.0,
                probabilities={'choppy': 1.0},
                transition_signal='unknown',
                days_in_regime=0,
                features={},
                recommendation='HMM failed - using default'
            )

    def _get_rule_result(self) -> RegimeAnalysis:
        """Get rule-based regime detection."""
        try:
            return self.rule_detector.detect_regime()
        except Exception as e:
            print(f"[REGIME] Rule-based detection failed: {e}")
            return RegimeAnalysis(
                regime=MarketRegime.CHOPPY,
                confidence=0.0,
                spy_trend_20d=0.0,
                spy_trend_50d=0.0,
                vix_level=20.0,
                vix_percentile=50.0,
                adv_decline_ratio=1.0,
                breadth_score=0.5,
                recommendation='Rule-based failed - using default',
                data_error=True
            )

    def _get_hierarchical_result(self) -> HierarchicalRegimeResult:
        """Get hierarchical HMM regime detection (MDPI 2025 research)."""
        try:
            return self.hierarchical_hmm.detect_regime()
        except Exception as e:
            print(f"[REGIME] Hierarchical HMM detection failed: {e}")
            return HierarchicalRegimeResult(
                meta_regime='low_uncertainty',
                meta_confidence=0.5,
                meta_days_in_regime=0,
                market_state='choppy',
                state_confidence=0.5,
                state_days=0,
                regime_label='LOW_UNCERTAINTY:CHOPPY',
                transition_probability={'to_high_uncertainty': 0.05, 'state_change': 0.2},
                risk_multiplier=1.0,
                recommendation='Hierarchical HMM unavailable - using defaults'
            )

    def _get_macro_indicators(self) -> MacroIndicators:
        """Get FRED macroeconomic indicators (Chen 2009 research)."""
        try:
            return self.macro_fetcher.get_indicators()
        except Exception as e:
            print(f"[REGIME] Macro indicators failed: {e}")
            return MacroIndicators(
                timestamp=datetime.now().isoformat(),
                m2_growth_yoy=0.0,
                m2_growth_3m=0.0,
                m2_signal='unknown',
                initial_claims=200.0,
                claims_4wk_avg=200.0,
                claims_trend='unknown',
                yield_spread=0.5,
                yield_inverted=False,
                inversion_days=0,
                pmi_composite=50.0,
                pmi_trend='neutral',
                recession_probability=10.0,
                recession_signal='LOW',
                recommendation='Macro data unavailable'
            )

    def _get_vix_term_structure(self, vix_level: float) -> tuple:
        """
        Get VIX term structure (VIX/VIX3M ratio).

        Returns:
            (vix_term_structure, vix_3m)

        Interpretation:
        - Ratio > 1.0: Backwardation (near-term fear exceeds long-term)
          → Warning sign, potential crash ahead
        - Ratio < 1.0: Contango (normal, complacent markets)
          → Normal conditions
        - Ratio >> 1.1: Severe backwardation → High crash probability
        """
        try:
            import yfinance as yf

            # Fetch VIX3M (3-month VIX)
            vix3m = yf.Ticker("^VIX3M")
            vix3m_data = vix3m.history(period="5d")

            if len(vix3m_data) > 0:
                vix_3m = vix3m_data['Close'].iloc[-1]
                if vix_3m > 0:
                    term_structure = vix_level / vix_3m
                    return round(term_structure, 3), round(vix_3m, 2)

            return 1.0, vix_level  # Default if VIX3M unavailable

        except Exception as e:
            print(f"[REGIME] VIX term structure fetch failed: {e}")
            return 1.0, vix_level

    def _get_correlation_regime(self) -> CorrelationRegimeResult:
        """Get correlation regime analysis."""
        try:
            return self.correlation_tracker.analyze()
        except Exception as e:
            print(f"[REGIME] Correlation analysis failed: {e}")
            return CorrelationRegimeResult(
                timestamp=datetime.now().isoformat(),
                avg_correlation=0.5,
                correlation_percentile=50.0,
                correlation_dispersion=0.1,
                dispersion_percentile=50.0,
                correlation_regime='NORMAL',
                risk_multiplier=1.0,
                sector_correlations={},
                correlation_change_5d=0.0,
                is_correlation_spike=False,
                recommendation='Correlation analysis unavailable'
            )

    def _calculate_model_disagreement(
        self,
        hmm: HMMRegimeResult,
        rule: RegimeAnalysis,
        hierarchical: HierarchicalRegimeResult
    ) -> Tuple[float, float, str]:
        """
        Calculate disagreement level between models.

        High disagreement suggests regime transition uncertainty.

        Returns:
            (disagreement 0-1, risk_multiplier, note)
        """
        disagreement = 0.0
        notes = []

        # HMM vs Rule-based disagreement
        if hmm.regime != rule.regime.value:
            disagreement += 0.3
            notes.append(f"HMM({hmm.regime}) vs Rule({rule.regime.value})")

        # HMM vs Hierarchical state disagreement
        if hmm.regime != hierarchical.market_state:
            disagreement += 0.2
            notes.append(f"HMM({hmm.regime}) vs HHMM({hierarchical.market_state})")

        # Confidence disagreement (both low confidence)
        avg_conf = (hmm.confidence + rule.confidence) / 2
        if avg_conf < 0.6:
            disagreement += 0.2
            notes.append(f"Low confidence ({avg_conf:.0%})")

        # Transition signal suggests instability
        if 'entering' in hmm.transition_signal or 'exiting' in hmm.transition_signal:
            disagreement += 0.15
            notes.append(f"Transition: {hmm.transition_signal}")

        # Meta-regime is high uncertainty but market state is bull
        if hierarchical.meta_regime == 'high_uncertainty' and hierarchical.market_state == 'bull':
            disagreement += 0.15
            notes.append("High uncertainty but bull state")

        disagreement = min(1.0, disagreement)

        # Map disagreement to risk multiplier
        if disagreement > 0.6:
            risk_mult = 0.5
        elif disagreement > 0.4:
            risk_mult = 0.7
        elif disagreement > 0.2:
            risk_mult = 0.85
        else:
            risk_mult = 1.0

        note = "; ".join(notes) if notes else "Models agree"

        return disagreement, risk_mult, note

    def _use_hmm(self, hmm: HMMRegimeResult, rule: RegimeAnalysis) -> UnifiedRegimeResult:
        """Use HMM as primary detector."""
        agreement = hmm.regime == rule.regime.value
        has_data_error = getattr(rule, 'data_error', False) or getattr(hmm, 'data_error', False)

        return UnifiedRegimeResult(
            regime=hmm.regime,
            confidence=hmm.confidence,
            hmm_regime=hmm.regime,
            hmm_confidence=hmm.confidence,
            rule_regime=rule.regime.value,
            rule_confidence=rule.confidence,
            agreement=agreement,
            vix_level=rule.vix_level,
            transition_signal=hmm.transition_signal,
            recommendation=hmm.recommendation,
            method_used='hmm',
            hmm_probabilities=hmm.probabilities,
            days_in_regime=hmm.days_in_regime,
            data_error=has_data_error
        )

    def _use_rule_based(self, hmm: HMMRegimeResult, rule: RegimeAnalysis) -> UnifiedRegimeResult:
        """Use rule-based as primary detector."""
        agreement = hmm.regime == rule.regime.value
        has_data_error = getattr(rule, 'data_error', False) or getattr(hmm, 'data_error', False)

        return UnifiedRegimeResult(
            regime=rule.regime.value,
            confidence=rule.confidence,
            hmm_regime=hmm.regime,
            hmm_confidence=hmm.confidence,
            rule_regime=rule.regime.value,
            rule_confidence=rule.confidence,
            agreement=agreement,
            vix_level=rule.vix_level,
            transition_signal=hmm.transition_signal,
            recommendation=rule.recommendation,
            method_used='rule_based',
            hmm_probabilities=hmm.probabilities,
            days_in_regime=hmm.days_in_regime,
            data_error=has_data_error
        )

    def _use_ensemble(self, hmm: HMMRegimeResult, rule: RegimeAnalysis) -> UnifiedRegimeResult:
        """
        Combine HMM and rule-based detection using ensemble logic.

        Decision rules:
        1. If both agree -> use that regime with boosted confidence
        2. If VIX is extreme (>30) -> prefer rule-based (VIX-aware)
        3. If HMM shows transition -> weight HMM higher (catches shifts early)
        4. Otherwise -> use weighted average of confidences
        """
        rule_regime = rule.regime.value
        hmm_regime = hmm.regime
        has_data_error = getattr(rule, 'data_error', False) or getattr(hmm, 'data_error', False)

        agreement = hmm_regime == rule_regime

        # Both agree - high confidence
        if agreement:
            combined_conf = min(0.95, (hmm.confidence + rule.confidence) / 2 + 0.1)
            return UnifiedRegimeResult(
                regime=hmm_regime,
                confidence=combined_conf,
                hmm_regime=hmm_regime,
                hmm_confidence=hmm.confidence,
                rule_regime=rule_regime,
                rule_confidence=rule.confidence,
                agreement=True,
                vix_level=rule.vix_level,
                transition_signal=hmm.transition_signal,
                recommendation=self._combine_recommendations(hmm, rule),
                method_used='ensemble_agree',
                hmm_probabilities=hmm.probabilities,
                days_in_regime=hmm.days_in_regime,
                data_error=has_data_error
            )

        # Disagreement - use decision rules

        # Rule 1: Extreme VIX -> trust rule-based more
        if rule.vix_level > 30:
            # In extreme volatility, VIX-based detection is more reliable
            final_regime = rule_regime
            method = 'ensemble_vix_override'
            recommendation = f"High VIX ({rule.vix_level:.1f}) - using rule-based. " + rule.recommendation

        # Rule 2: HMM shows transition -> trust HMM (catches shifts early)
        elif 'entering' in hmm.transition_signal or 'exiting' in hmm.transition_signal:
            final_regime = hmm_regime
            method = 'ensemble_hmm_transition'
            recommendation = f"HMM detects regime shift ({hmm.transition_signal}). " + hmm.recommendation

        # Rule 3: HMM has much higher confidence
        elif hmm.confidence > rule.confidence + 0.3:
            final_regime = hmm_regime
            method = 'ensemble_hmm_confidence'
            recommendation = f"HMM high confidence. " + hmm.recommendation

        # Rule 4: Rule-based has much higher confidence
        elif rule.confidence > hmm.confidence + 0.3:
            final_regime = rule_regime
            method = 'ensemble_rule_confidence'
            recommendation = rule.recommendation

        # Rule 5: Default to more conservative (higher threshold regime)
        else:
            # Regime order by mean reversion performance (worst to best)
            regime_priority = {'bull': 0, 'choppy': 1, 'bear': 2, 'volatile': 3}

            # Pick the one that's more conservative for mean reversion
            if regime_priority.get(hmm_regime, 1) < regime_priority.get(rule_regime, 1):
                final_regime = hmm_regime  # More conservative
            else:
                final_regime = rule_regime

            method = 'ensemble_conservative'
            recommendation = f"Methods disagree - using conservative ({final_regime}). Monitor closely."

        # Calculate combined confidence (lower when disagreeing)
        combined_conf = min(hmm.confidence, rule.confidence) * 0.8

        return UnifiedRegimeResult(
            regime=final_regime,
            confidence=combined_conf,
            hmm_regime=hmm_regime,
            hmm_confidence=hmm.confidence,
            rule_regime=rule_regime,
            rule_confidence=rule.confidence,
            agreement=False,
            vix_level=rule.vix_level,
            transition_signal=hmm.transition_signal,
            recommendation=recommendation,
            method_used=method,
            hmm_probabilities=hmm.probabilities,
            days_in_regime=hmm.days_in_regime,
            data_error=has_data_error
        )

    def _combine_recommendations(self, hmm: HMMRegimeResult, rule: RegimeAnalysis) -> str:
        """Combine recommendations when methods agree."""
        # Use HMM recommendation but add VIX context
        return f"{hmm.recommendation} VIX: {rule.vix_level:.1f}"


def log_regime_state(result: UnifiedRegimeResult, history_file: str = None):
    """
    Log regime state to history file for future analysis (Quick Win #4).

    Logs every regime detection with full context for:
    - Transition analysis
    - Pattern discovery
    - Performance attribution

    Args:
        result: UnifiedRegimeResult from detect_regime()
        history_file: Path to history JSON file
    """
    import json
    from pathlib import Path

    if history_file is None:
        history_file = Path(__file__).resolve().parent.parent.parent / 'features' / 'market_conditions' / 'data' / 'regime_history.json'

    history_file = Path(history_file)
    history_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing history
    history = []
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []

    # Create enhanced log entry
    entry = {
        'timestamp': datetime.now().isoformat(),
        'regime': result.regime,
        'confidence': round(result.confidence, 3),
        'method_used': result.method_used,
        'agreement': result.agreement,
        # HMM details
        'hmm_regime': result.hmm_regime,
        'hmm_confidence': round(result.hmm_confidence, 3),
        'hmm_probabilities': {k: round(v, 3) for k, v in (result.hmm_probabilities or {}).items()},
        'days_in_regime': result.days_in_regime,
        'transition_signal': result.transition_signal,
        # VIX data
        'vix': round(result.vix_level, 2),
        'vix_3m': round(result.vix_3m, 2),
        'vix_term_structure': round(result.vix_term_structure, 3),
        # Rule-based details
        'rule_regime': result.rule_regime,
        'rule_confidence': round(result.rule_confidence, 3),
        # Risk multipliers
        'hierarchical_risk_mult': round(result.hierarchical_risk_mult, 3),
        'macro_risk_mult': round(result.macro_risk_mult, 3),
        'disagreement_risk_mult': round(result.disagreement_risk_mult, 3),
        'correlation_risk_mult': round(result.correlation_risk_mult, 3),
        # Early warning
        'early_warning_score': round(result.early_warning_score, 1),
        'early_warning_level': result.early_warning_level,
        # Meta regime
        'meta_regime': result.meta_regime,
        'recession_signal': result.recession_signal
    }

    history.append(entry)

    # Keep last 1000 entries (about 3 years of daily scans)
    if len(history) > 1000:
        history = history[-1000:]

    # Save
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    # Log transition detection
    if len(history) >= 2:
        prev = history[-2]
        if prev['regime'] != entry['regime']:
            print(f"[REGIME] TRANSITION: {prev['regime'].upper()} -> {entry['regime'].upper()} "
                  f"(after {prev.get('days_in_regime', '?')} days)")


def get_current_regime(method: str = 'ensemble') -> UnifiedRegimeResult:
    """
    Quick function to get current regime.

    Args:
        method: 'hmm', 'rule', or 'ensemble' (default)

    Returns:
        UnifiedRegimeResult
    """
    method_map = {
        'hmm': DetectionMethod.HMM,
        'rule': DetectionMethod.RULE_BASED,
        'ensemble': DetectionMethod.ENSEMBLE
    }

    detector = UnifiedRegimeDetector(method=method_map.get(method, DetectionMethod.ENSEMBLE))
    return detector.detect_regime()


def print_regime_comparison():
    """Print comparison of HMM vs rule-based detection."""
    print("=" * 70)
    print("UNIFIED REGIME DETECTION COMPARISON")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    detector = UnifiedRegimeDetector(method=DetectionMethod.ENSEMBLE)
    result = detector.detect_regime()

    print(f"{'Method':<15} {'Regime':<12} {'Confidence':<12}")
    print("-" * 40)
    print(f"{'HMM':<15} {result.hmm_regime.upper():<12} {result.hmm_confidence*100:.1f}%")
    print(f"{'Rule-based':<15} {result.rule_regime.upper():<12} {result.rule_confidence*100:.1f}%")
    print("-" * 40)
    print(f"{'ENSEMBLE':<15} {result.regime.upper():<12} {result.confidence*100:.1f}%")
    print()

    print(f"Agreement: {'YES' if result.agreement else 'NO'}")
    print(f"Method used: {result.method_used}")
    print(f"VIX Level: {result.vix_level:.1f}")
    print(f"VIX 3M: {result.vix_3m:.1f}")
    term_indicator = "BACKWARDATION (fear)" if result.vix_term_structure > 1.05 else "CONTANGO (normal)" if result.vix_term_structure < 0.95 else "FLAT"
    print(f"VIX Term Structure: {result.vix_term_structure:.3f} ({term_indicator})")
    print(f"Days in Regime: {result.days_in_regime}")
    print(f"Transition: {result.transition_signal}")
    print()

    # HMM Probabilities section
    print("--- HMM PROBABILITIES ---")
    if result.hmm_probabilities:
        for regime, prob in sorted(result.hmm_probabilities.items(), key=lambda x: -x[1]):
            bar = "#" * int(prob * 20)
            print(f"  {regime.upper():<10} {prob*100:5.1f}% {bar}")
    print()

    # Hierarchical HMM section (MDPI 2025 research)
    print("--- HIERARCHICAL HMM (Meta-Regime) ---")
    meta_indicator = "[!]" if result.meta_regime == "high_uncertainty" else ""
    print(f"Meta-Regime: {result.meta_regime.upper()} {meta_indicator}")
    print(f"Meta Confidence: {result.meta_confidence*100:.1f}%")
    print(f"Risk Multiplier: {result.hierarchical_risk_mult:.2f}")
    print()

    # Macro indicators section (Chen 2009 research)
    print("--- FRED MACRO (Recession Risk) ---")
    recession_indicator = {
        'HIGH': '!!!',
        'ELEVATED': '!!',
        'MODERATE': '!',
        'LOW': ''
    }.get(result.recession_signal, '')
    print(f"Recession Probability: {result.recession_probability:.0f}% {recession_indicator}")
    print(f"Recession Signal: {result.recession_signal}")
    print(f"Macro Risk Mult: {result.macro_risk_mult:.2f}")
    print()

    # Model disagreement section
    print("--- MODEL DISAGREEMENT ---")
    disagree_bar = "#" * int(result.model_disagreement * 10) + "-" * (10 - int(result.model_disagreement * 10))
    print(f"Disagreement: [{disagree_bar}] {result.model_disagreement*100:.0f}%")
    print(f"Risk Mult: {result.disagreement_risk_mult:.2f}")
    if result.disagreement_note:
        print(f"Note: {result.disagreement_note}")
    print()

    # Correlation regime section
    print("--- CORRELATION REGIME ---")
    corr_indicator = {
        'CRISIS': '!!!',
        'ELEVATED': '!!',
        'NORMAL': ''
    }.get(result.correlation_regime, '')
    corr_bar = "#" * int(result.avg_correlation * 10) + "-" * (10 - int(result.avg_correlation * 10))
    print(f"Avg Correlation: [{corr_bar}] {result.avg_correlation:.2f}")
    print(f"Regime: {result.correlation_regime} {corr_indicator}")
    print(f"Risk Mult: {result.correlation_risk_mult:.2f}")
    print()

    # Early warning section
    print("--- EARLY WARNING ---")
    warning_indicator = {
        'CRITICAL': '!!!',
        'WARNING': '!!',
        'WATCH': '!',
        'NORMAL': ''
    }.get(result.early_warning_level, '')
    print(f"Bear Score: {result.early_warning_score:.0f}/100 {warning_indicator}")
    print(f"Alert Level: {result.early_warning_level}")
    if result.early_warning_triggers:
        print(f"Triggers: {', '.join(result.early_warning_triggers[:3])}")
    print()

    print("--- RECOMMENDATION ---")
    print(result.recommendation)
    print()

    return result


if __name__ == "__main__":
    print_regime_comparison()

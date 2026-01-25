"""
Regime Alert System
====================

Monitors market regime transitions and sends alerts when high-alpha
conditions are detected (particularly BEAR markets).

Based on Jan 2026 overnight backtest findings:
- BEAR markets: 100% win rate, Sharpe 1.80 (at threshold 70)
- BULL markets: 69.4% win rate, Sharpe 0.43
- CHOPPY markets: ~53% win rate, Sharpe ~0

Key insight: BEAR markets are rare (6.6% of time) but highly profitable.
We need to catch them early for maximum alpha capture.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from analysis.market_regime import (
    MarketRegimeDetector, MarketRegime, RegimeAnalysis,
    RegimeAwareFilter, TradingDecision
)


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RegimeAlert:
    """A regime change alert."""
    timestamp: str
    alert_type: str
    priority: AlertPriority
    current_regime: str
    previous_regime: Optional[str]
    message: str
    recommendation: str
    metrics: Dict


class RegimeTransitionDetector:
    """
    Detects regime transitions and generates alerts.

    Key transitions to watch:
    - ANY -> BEAR: Critical opportunity (rare but highly profitable)
    - BEAR -> ANY: Exit high-alpha conditions
    - CHOPPY -> BULL: Potential trend starting
    - BULL -> CHOPPY: Trend ending, reduce exposure
    """

    HISTORY_FILE = "features/market_conditions/data/regime_history.json"
    ALERTS_FILE = "data/regime_alerts.json"

    def __init__(self):
        self.detector = MarketRegimeDetector()
        self.history: List[Dict] = self._load_history()
        self.alerts: List[RegimeAlert] = []

    def _load_history(self) -> List[Dict]:
        """Load regime history from file."""
        path = Path(self.HISTORY_FILE)
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except:
                pass
        return []

    def _save_history(self):
        """Save regime history to file."""
        path = Path(self.HISTORY_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.history[-100:], f, indent=2)  # Keep last 100

    def _save_alerts(self):
        """Save alerts to file."""
        path = Path(self.ALERTS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        alerts_data = [asdict(a) for a in self.alerts[-50:]]  # Keep last 50
        # Convert enum to string
        for a in alerts_data:
            a['priority'] = a['priority'].value if isinstance(a['priority'], AlertPriority) else a['priority']
        with open(path, 'w') as f:
            json.dump(alerts_data, f, indent=2)

    def get_last_regime(self) -> Optional[str]:
        """Get the last recorded regime."""
        if self.history:
            return self.history[-1].get('regime')
        return None

    def check_for_transition(self) -> Tuple[RegimeAnalysis, Optional[RegimeAlert]]:
        """
        Check current regime and detect any transitions.

        Returns:
            Tuple of (current_analysis, alert_if_any)
        """
        # Get current regime
        analysis = self.detector.detect_regime()
        current_regime = analysis.regime.value
        previous_regime = self.get_last_regime()

        # Record history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'regime': current_regime,
            'confidence': analysis.confidence,
            'vix': analysis.vix_level,
            'trend_20d': analysis.spy_trend_20d,
            'trend_50d': analysis.spy_trend_50d
        }
        self.history.append(history_entry)
        self._save_history()

        # Check for transition
        alert = None
        if previous_regime and previous_regime != current_regime:
            alert = self._create_transition_alert(
                previous_regime, current_regime, analysis
            )
            if alert:
                self.alerts.append(alert)
                self._save_alerts()

        # Also alert on first BEAR detection even without transition
        elif current_regime == 'bear' and analysis.confidence > 0.7:
            # Check if we've alerted about this BEAR recently
            recent_bear_alert = any(
                a.current_regime == 'bear' and
                datetime.fromisoformat(a.timestamp) > datetime.now() - timedelta(hours=4)
                for a in self.alerts
            )
            if not recent_bear_alert:
                alert = self._create_bear_opportunity_alert(analysis)
                self.alerts.append(alert)
                self._save_alerts()

        return analysis, alert

    def _create_transition_alert(
        self,
        from_regime: str,
        to_regime: str,
        analysis: RegimeAnalysis
    ) -> RegimeAlert:
        """Create an alert for a regime transition."""

        # Determine priority based on transition type
        if to_regime == 'bear':
            priority = AlertPriority.CRITICAL
            alert_type = "BEAR_MARKET_ENTRY"
            message = "BEAR MARKET DETECTED - High alpha opportunity!"
            recommendation = (
                "IMMEDIATE ACTION RECOMMENDED:\n"
                "- Use threshold 70 for signal filtering\n"
                "- Full position sizes (100%)\n"
                "- Historical: 100% win rate, Sharpe 1.80\n"
                "- This is RARE (6.6% of time) - maximize exposure"
            )
        elif from_regime == 'bear':
            priority = AlertPriority.HIGH
            alert_type = "BEAR_MARKET_EXIT"
            message = f"Exiting BEAR market - transitioning to {to_regime.upper()}"
            recommendation = (
                "Adjust strategy for new regime:\n"
                f"- New regime: {to_regime.upper()}\n"
                "- Review open positions\n"
                "- Adjust thresholds accordingly"
            )
        elif to_regime == 'volatile':
            priority = AlertPriority.HIGH
            alert_type = "VOLATILITY_SPIKE"
            message = "High volatility detected - VIX elevated"
            recommendation = (
                "- Use threshold 70\n"
                "- Reduce position sizes to 50%\n"
                "- Widen stop losses\n"
                "- Consider reducing exposure"
            )
        elif from_regime == 'choppy' and to_regime == 'bull':
            priority = AlertPriority.MEDIUM
            alert_type = "TREND_STARTING"
            message = "Potential uptrend starting - exiting choppy conditions"
            recommendation = (
                "- Threshold 50 optimal for BULL\n"
                "- Full position sizes\n"
                "- Mean reversion may be less reliable\n"
                "- Watch for trend continuation"
            )
        elif to_regime == 'choppy':
            priority = AlertPriority.MEDIUM
            alert_type = "CHOPPY_ENTRY"
            message = f"Entering CHOPPY market from {from_regime.upper()}"
            recommendation = (
                "CAUTION - Poor conditions:\n"
                "- Historical Sharpe ~0 in CHOPPY\n"
                "- Use threshold 65+ to filter low quality\n"
                "- Reduce position sizes to 30%\n"
                "- Consider skipping trades entirely"
            )
        else:
            priority = AlertPriority.LOW
            alert_type = "REGIME_CHANGE"
            message = f"Regime changed: {from_regime.upper()} -> {to_regime.upper()}"
            recommendation = "Review current strategy settings"

        return RegimeAlert(
            timestamp=datetime.now().isoformat(),
            alert_type=alert_type,
            priority=priority,
            current_regime=to_regime,
            previous_regime=from_regime,
            message=message,
            recommendation=recommendation,
            metrics={
                'vix': analysis.vix_level,
                'vix_percentile': analysis.vix_percentile,
                'trend_20d': analysis.spy_trend_20d,
                'trend_50d': analysis.spy_trend_50d,
                'confidence': analysis.confidence
            }
        )

    def _create_bear_opportunity_alert(self, analysis: RegimeAnalysis) -> RegimeAlert:
        """Create alert for BEAR market opportunity."""
        return RegimeAlert(
            timestamp=datetime.now().isoformat(),
            alert_type="BEAR_OPPORTUNITY",
            priority=AlertPriority.CRITICAL,
            current_regime='bear',
            previous_regime=None,
            message="BEAR MARKET ACTIVE - Maximum alpha conditions!",
            recommendation=(
                "BEAR MARKET STRATEGY:\n"
                "- Threshold: 70 (filters to highest quality)\n"
                "- Position size: 100% (full exposure)\n"
                "- Historical performance: 100% win rate\n"
                "- Sharpe: 1.80 (exceptional)\n"
                "- Duration: Typically short (avg 29 days)\n"
                "- ACTION: Actively scan for signals!"
            ),
            metrics={
                'vix': analysis.vix_level,
                'vix_percentile': analysis.vix_percentile,
                'trend_20d': analysis.spy_trend_20d,
                'trend_50d': analysis.spy_trend_50d,
                'confidence': analysis.confidence
            }
        )

    def get_regime_summary(self) -> Dict:
        """Get a summary of regime history and current state."""
        if not self.history:
            analysis = self.detector.detect_regime()
            return {
                'current_regime': analysis.regime.value,
                'confidence': analysis.confidence,
                'history_days': 0,
                'regime_counts': {},
                'last_bear': None
            }

        # Count regimes in history
        regime_counts = {}
        last_bear = None
        for entry in self.history:
            regime = entry['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            if regime == 'bear':
                last_bear = entry['timestamp']

        current = self.history[-1]
        return {
            'current_regime': current['regime'],
            'confidence': current['confidence'],
            'history_days': len(self.history),
            'regime_counts': regime_counts,
            'last_bear': last_bear,
            'vix': current.get('vix'),
            'trend_20d': current.get('trend_20d')
        }


def check_regime_and_alert() -> Tuple[RegimeAnalysis, Optional[RegimeAlert]]:
    """
    Quick function to check regime and get any alerts.

    Usage:
        analysis, alert = check_regime_and_alert()
        if alert and alert.priority == AlertPriority.CRITICAL:
            print(f"ALERT: {alert.message}")
    """
    detector = RegimeTransitionDetector()
    return detector.check_for_transition()


def print_regime_status():
    """Print current regime status with alerts."""
    detector = RegimeTransitionDetector()
    analysis, alert = detector.check_for_transition()
    summary = detector.get_regime_summary()

    print("=" * 70)
    print("REGIME STATUS")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print(f"Current Regime: {analysis.regime.value.upper()}")
    print(f"Confidence: {analysis.confidence*100:.0f}%")
    print(f"VIX: {analysis.vix_level:.1f} ({analysis.vix_percentile:.0f}th percentile)")
    print(f"SPY 20d Trend: {analysis.spy_trend_20d:+.2f}%")
    print(f"SPY 50d Trend: {analysis.spy_trend_50d:+.2f}%")
    print()

    # Regime-specific recommendations
    recommendations = {
        MarketRegime.BULL: ("threshold=50", "position=100%", "Sharpe=0.43"),
        MarketRegime.BEAR: ("threshold=70", "position=100%", "Sharpe=1.80 [BEST]"),
        MarketRegime.CHOPPY: ("threshold=65", "position=30%", "Sharpe=~0 [AVOID]"),
        MarketRegime.VOLATILE: ("threshold=70", "position=50%", "High uncertainty")
    }
    rec = recommendations.get(analysis.regime, ("threshold=50", "position=100%", ""))
    print(f"Recommended: {rec[0]}, {rec[1]}")
    print(f"Expected: {rec[2]}")
    print()

    if alert:
        print("=" * 70)
        print(f"*** ALERT: {alert.alert_type} ***")
        print(f"Priority: {alert.priority.value.upper()}")
        print(f"Message: {alert.message}")
        print()
        print("Recommendation:")
        print(alert.recommendation)
        print("=" * 70)

    # History summary
    if summary['last_bear']:
        print(f"\nLast BEAR market: {summary['last_bear'][:10]}")

    return analysis, alert


if __name__ == "__main__":
    print_regime_status()

"""
Rolling Correlation Regime Tracker
==================================

Detects correlation regime changes in the market.

Key insight: During market stress, correlations spike ("all correlations go to 1").
This is a useful leading indicator for:
- Risk-off events
- Regime transitions
- Diversification breakdown

Methodology:
1. Track rolling correlation between SPY and key sectors
2. Monitor correlation dispersion (low dispersion = crisis)
3. Detect correlation breakouts vs historical norms

Research basis:
- "All Correlations Go to 1" phenomenon during crises
- Correlation clustering in risk-off environments

Jan 2026 - Created for Proteus trading system.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yfinance as yf


@dataclass
class CorrelationRegimeResult:
    """Container for correlation regime analysis."""
    timestamp: str
    # Average correlation levels
    avg_correlation: float         # Average correlation with SPY
    correlation_percentile: float  # Percentile vs 1-year history
    # Correlation dispersion (diversity)
    correlation_dispersion: float  # Std dev of correlations (lower = risk-off)
    dispersion_percentile: float   # Percentile vs history
    # Regime classification
    correlation_regime: str        # 'NORMAL', 'ELEVATED', 'CRISIS'
    risk_multiplier: float         # Position sizing multiplier
    # Details
    sector_correlations: Dict[str, float]  # Per-sector correlations
    correlation_change_5d: float   # 5-day change in avg correlation
    is_correlation_spike: bool     # Sudden correlation increase
    recommendation: str


class CorrelationRegimeTracker:
    """
    Tracks correlation regimes for risk management.

    Usage:
        tracker = CorrelationRegimeTracker()
        result = tracker.analyze()
        print(f"Correlation regime: {result.correlation_regime}")
    """

    # Sector ETFs to track
    SECTOR_ETFS = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLI': 'Industrials',
        'XLY': 'Consumer Disc',
        'XLP': 'Consumer Staples',
        'XLU': 'Utilities',
        'XLRE': 'Real Estate',
        'XLB': 'Materials'
    }

    def __init__(self, lookback_days: int = 252, correlation_window: int = 20):
        """
        Initialize correlation tracker.

        Args:
            lookback_days: Days of history for percentile calculation
            correlation_window: Rolling window for correlation calculation
        """
        self.lookback_days = lookback_days
        self.correlation_window = correlation_window

    def _fetch_data(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Fetch price data for SPY and sector ETFs.

        Returns:
            Dict mapping ticker to returns array
        """
        tickers = ['SPY'] + list(self.SECTOR_ETFS.keys())

        try:
            end = datetime.now()
            start = end - timedelta(days=self.lookback_days + 50)  # Extra buffer

            data = yf.download(
                tickers,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True
            )['Close']

            if data.empty:
                return None

            # Calculate returns
            returns = data.pct_change().dropna()

            min_days = max(60, self.correlation_window + 20)  # Need at least 60 days
            if len(returns) < min_days:
                print(f"[CORR] Insufficient data: {len(returns)} days (need {min_days})")
                return None

            return {col: returns[col].values for col in returns.columns}

        except Exception as e:
            print(f"[CORR] Error fetching data: {e}")
            return None

    def _calculate_rolling_correlations(
        self,
        returns: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Calculate rolling correlations with SPY.

        Returns:
            (avg_correlations, per_sector_rolling_correlations)
        """
        spy_returns = returns.get('SPY')
        if spy_returns is None:
            return np.array([0.5]), {}

        n = len(spy_returns)
        sector_corrs = {}

        for ticker in self.SECTOR_ETFS.keys():
            if ticker not in returns:
                continue

            sector_returns = returns[ticker]
            rolling_corr = np.zeros(n)

            for i in range(self.correlation_window, n):
                spy_window = spy_returns[i-self.correlation_window:i]
                sector_window = sector_returns[i-self.correlation_window:i]

                if np.std(spy_window) > 0 and np.std(sector_window) > 0:
                    corr = np.corrcoef(spy_window, sector_window)[0, 1]
                    rolling_corr[i] = corr if not np.isnan(corr) else 0.5
                else:
                    rolling_corr[i] = 0.5

            sector_corrs[ticker] = rolling_corr

        # Calculate average correlation across sectors
        if sector_corrs:
            all_corrs = np.array(list(sector_corrs.values()))
            avg_corrs = np.mean(all_corrs, axis=0)
        else:
            avg_corrs = np.full(n, 0.5)

        return avg_corrs, sector_corrs

    def _calculate_correlation_dispersion(
        self,
        sector_corrs: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Calculate correlation dispersion (std dev of correlations).

        Lower dispersion = more uniform correlations = risk-off.

        Returns:
            Rolling dispersion array
        """
        if not sector_corrs:
            return np.array([0.1])

        all_corrs = np.array(list(sector_corrs.values()))
        # Standard deviation across sectors at each time point
        dispersion = np.std(all_corrs, axis=0)

        return dispersion

    def _detect_correlation_spike(
        self,
        avg_corrs: np.ndarray,
        lookback: int = 10
    ) -> Tuple[bool, float]:
        """
        Detect sudden correlation spike.

        Returns:
            (is_spike, 5d_change)
        """
        if len(avg_corrs) < lookback:
            return False, 0.0

        current = avg_corrs[-1]
        avg_recent = np.mean(avg_corrs[-lookback:-5]) if len(avg_corrs) > lookback else avg_corrs[-5]

        change_5d = current - avg_corrs[-5] if len(avg_corrs) >= 5 else 0.0

        # Spike = correlation increased by more than 0.15 in 5 days
        is_spike = change_5d > 0.15

        return is_spike, change_5d

    def _classify_regime(
        self,
        avg_corr: float,
        corr_percentile: float,
        dispersion: float,
        dispersion_percentile: float,
        is_spike: bool
    ) -> Tuple[str, float]:
        """
        Classify correlation regime.

        Returns:
            (regime, risk_multiplier)
        """
        # Crisis: Very high correlation + low dispersion (everything moving together)
        if avg_corr > 0.85 or (corr_percentile > 90 and dispersion_percentile < 20):
            return 'CRISIS', 0.50

        # Spike: Sudden increase even if not at crisis level yet
        if is_spike and corr_percentile > 75:
            return 'CRISIS', 0.60

        # Elevated: High correlation or low dispersion
        if avg_corr > 0.7 or corr_percentile > 80 or dispersion_percentile < 25:
            return 'ELEVATED', 0.75

        # Moderate: Above normal but not concerning
        if avg_corr > 0.55 or corr_percentile > 60:
            return 'NORMAL', 0.90

        # Normal: Healthy diversification
        return 'NORMAL', 1.0

    def analyze(self) -> CorrelationRegimeResult:
        """
        Analyze current correlation regime.

        Returns:
            CorrelationRegimeResult with analysis
        """
        print("[CORR] Analyzing correlation regime...")

        returns = self._fetch_data()
        if returns is None:
            return self._default_result("Data fetch failed")

        # Calculate rolling correlations
        avg_corrs, sector_corrs = self._calculate_rolling_correlations(returns)

        # Calculate dispersion
        dispersion = self._calculate_correlation_dispersion(sector_corrs)

        # Current values
        current_avg_corr = avg_corrs[-1]
        current_dispersion = dispersion[-1]

        # Calculate percentiles vs history
        valid_avg = avg_corrs[self.correlation_window:]  # Skip warmup
        valid_disp = dispersion[self.correlation_window:]

        corr_percentile = (np.sum(valid_avg < current_avg_corr) / len(valid_avg)) * 100
        disp_percentile = (np.sum(valid_disp < current_dispersion) / len(valid_disp)) * 100

        # Detect spike
        is_spike, change_5d = self._detect_correlation_spike(avg_corrs)

        # Classify regime
        regime, risk_mult = self._classify_regime(
            current_avg_corr, corr_percentile,
            current_dispersion, disp_percentile,
            is_spike
        )

        # Get current sector correlations
        current_sector_corrs = {
            self.SECTOR_ETFS[t]: round(sector_corrs[t][-1], 3)
            for t in sector_corrs
        }

        # Generate recommendation
        if regime == 'CRISIS':
            recommendation = "CRISIS correlation regime - reduce all positions, high contagion risk"
        elif regime == 'ELEVATED':
            recommendation = "Elevated correlations - reduce sizes, diversification less effective"
        else:
            recommendation = "Normal correlation regime - standard positioning"

        return CorrelationRegimeResult(
            timestamp=datetime.now().isoformat(),
            avg_correlation=round(current_avg_corr, 3),
            correlation_percentile=round(corr_percentile, 1),
            correlation_dispersion=round(current_dispersion, 3),
            dispersion_percentile=round(disp_percentile, 1),
            correlation_regime=regime,
            risk_multiplier=risk_mult,
            sector_correlations=current_sector_corrs,
            correlation_change_5d=round(change_5d, 3),
            is_correlation_spike=is_spike,
            recommendation=recommendation
        )

    def _default_result(self, reason: str) -> CorrelationRegimeResult:
        """Return default result on error."""
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
            recommendation=f"Correlation analysis unavailable: {reason}"
        )


def print_correlation_regime():
    """Print current correlation regime analysis."""
    print("=" * 70)
    print("CORRELATION REGIME ANALYSIS")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    tracker = CorrelationRegimeTracker()
    result = tracker.analyze()

    print("--- CORRELATION LEVELS ---")
    print(f"Average SPY Correlation: {result.avg_correlation:.2f}")
    print(f"Percentile (1Y): {result.correlation_percentile:.0f}%")
    print(f"5-Day Change: {result.correlation_change_5d:+.3f}")
    if result.is_correlation_spike:
        print("[!] CORRELATION SPIKE DETECTED")
    print()

    print("--- DISPERSION (Diversity) ---")
    print(f"Correlation Dispersion: {result.correlation_dispersion:.3f}")
    print(f"Percentile (1Y): {result.dispersion_percentile:.0f}%")
    print("(Lower dispersion = less diversification benefit)")
    print()

    print("--- REGIME ---")
    regime_indicator = {
        'CRISIS': '!!!',
        'ELEVATED': '!!',
        'NORMAL': ''
    }.get(result.correlation_regime, '')
    print(f"Regime: {result.correlation_regime} {regime_indicator}")
    print(f"Risk Multiplier: {result.risk_multiplier:.2f}")
    print()

    print("--- SECTOR CORRELATIONS ---")
    for sector, corr in sorted(result.sector_correlations.items(), key=lambda x: -x[1]):
        bar = "#" * int(corr * 10) + "-" * (10 - int(corr * 10))
        print(f"  {sector:<18} [{bar}] {corr:.2f}")
    print()

    print("--- RECOMMENDATION ---")
    print(result.recommendation)
    print()

    return result


if __name__ == "__main__":
    print_correlation_regime()

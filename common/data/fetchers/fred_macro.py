"""
FRED Macroeconomic Data Fetcher
===============================

Fetches leading economic indicators from FRED (Federal Reserve Economic Data).
These indicators help predict recession risk 3-12 months ahead.

Key indicators:
- M2 Money Supply Growth (M2SL) - monetary conditions
- Initial Jobless Claims (ICSA) - labor market weakness
- ISM Manufacturing PMI (MANEMP) - economic activity
- Yield Curve Spread (T10Y2Y) - recession predictor
- Consumer Confidence (UMCSENT) - sentiment

Research basis:
- Chen et al. 2009: Macroeconomic variables as leading indicators
- NBER Business Cycle Dating

Jan 2026 - Created for Proteus trading system.
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# FRED API base URL (no key needed for public data via alternative method)
FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


@dataclass
class MacroIndicators:
    """Container for macroeconomic indicators."""
    timestamp: str
    # M2 Money Supply
    m2_growth_yoy: float          # Year-over-year growth rate
    m2_growth_3m: float           # 3-month annualized growth
    m2_signal: str                # 'expansionary', 'neutral', 'contractionary'
    # Jobless Claims
    initial_claims: float         # Weekly initial claims (thousands)
    claims_4wk_avg: float         # 4-week moving average
    claims_trend: str             # 'rising', 'stable', 'falling'
    # Yield Curve
    yield_spread: float           # 10Y - 2Y spread
    yield_inverted: bool          # True if inverted
    inversion_days: int           # Days of continuous inversion
    # PMI / Economic Activity
    pmi_composite: float          # ISM Manufacturing index
    pmi_trend: str                # 'expanding', 'neutral', 'contracting'
    # Composite Recession Indicator
    recession_probability: float  # 0-100%
    recession_signal: str         # 'LOW', 'MODERATE', 'ELEVATED', 'HIGH'
    recommendation: str


class FREDMacroFetcher:
    """
    Fetches and analyzes FRED macroeconomic data.

    Usage:
        fetcher = FREDMacroFetcher()
        indicators = fetcher.get_indicators()
        print(f"Recession probability: {indicators.recession_probability}%")
    """

    # FRED series IDs
    SERIES = {
        'M2': 'M2SL',           # M2 Money Stock (billions, seasonally adjusted)
        'CLAIMS': 'ICSA',       # Initial Claims (thousands)
        'T10Y2Y': 'T10Y2Y',     # 10-Year Treasury Minus 2-Year
        'T10Y3M': 'T10Y3M',     # 10-Year Treasury Minus 3-Month
        'UMCSENT': 'UMCSENT',   # Consumer Sentiment
        'UNRATE': 'UNRATE',     # Unemployment Rate
        'INDPRO': 'INDPRO',     # Industrial Production
    }

    # Cache directory
    CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'features', 'market_conditions', 'data', 'cache', 'fred')
    CACHE_HOURS = 12  # Cache validity in hours

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED fetcher.

        Args:
            api_key: Optional FRED API key for higher rate limits
        """
        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def _fetch_series_csv(self, series_id: str, years: int = 3) -> Optional[Dict]:
        """
        Fetch series data from FRED using CSV endpoint (no API key needed).

        Args:
            series_id: FRED series ID
            years: Years of history to fetch

        Returns:
            Dict with 'dates' and 'values' lists, or None on error
        """
        cache_file = os.path.join(self.CACHE_DIR, f"{series_id}.json")

        # Check cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                cache_time = datetime.fromisoformat(cached['timestamp'])
                if datetime.now() - cache_time < timedelta(hours=self.CACHE_HOURS):
                    return cached['data']
            except Exception:
                pass

        # Fetch from FRED
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)

            url = (
                f"{FRED_BASE_URL}?id={series_id}"
                f"&cosd={start_date.strftime('%Y-%m-%d')}"
                f"&coed={end_date.strftime('%Y-%m-%d')}"
            )

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Parse CSV
            lines = response.text.strip().split('\n')
            dates = []
            values = []

            for line in lines[1:]:  # Skip header
                if ',' in line:
                    parts = line.split(',')
                    date_str = parts[0]
                    value_str = parts[1] if len(parts) > 1 else ''

                    if value_str and value_str != '.':
                        try:
                            dates.append(date_str)
                            values.append(float(value_str))
                        except ValueError:
                            continue

            if not dates:
                print(f"[FRED] No data for {series_id}")
                return None

            data = {'dates': dates, 'values': values}

            # Cache result
            with open(cache_file, 'w') as f:
                json.dump({'timestamp': datetime.now().isoformat(), 'data': data}, f)

            return data

        except Exception as e:
            print(f"[FRED] Error fetching {series_id}: {e}")
            return None

    def _calculate_m2_metrics(self) -> Tuple[float, float, str]:
        """
        Calculate M2 money supply growth metrics.

        Returns:
            (yoy_growth, 3m_growth, signal)
        """
        data = self._fetch_series_csv('M2SL')
        if not data or len(data['values']) < 13:
            return 0.0, 0.0, 'unknown'

        values = np.array(data['values'])

        # Year-over-year growth (12 months)
        if len(values) >= 13:
            yoy_growth = ((values[-1] / values[-13]) - 1) * 100
        else:
            yoy_growth = 0.0

        # 3-month annualized growth
        if len(values) >= 4:
            growth_3m = ((values[-1] / values[-4]) ** 4 - 1) * 100
        else:
            growth_3m = 0.0

        # Signal interpretation
        if yoy_growth > 8:
            signal = 'expansionary'
        elif yoy_growth < 2:
            signal = 'contractionary'
        else:
            signal = 'neutral'

        return round(yoy_growth, 2), round(growth_3m, 2), signal

    def _calculate_claims_metrics(self) -> Tuple[float, float, str]:
        """
        Calculate initial jobless claims metrics.

        Returns:
            (latest_claims, 4wk_avg, trend)
        """
        data = self._fetch_series_csv('ICSA')
        if not data or len(data['values']) < 8:
            return 0.0, 0.0, 'unknown'

        values = np.array(data['values'])

        latest = values[-1]
        avg_4wk = np.mean(values[-4:])
        avg_8wk = np.mean(values[-8:])

        # Trend detection
        if avg_4wk > avg_8wk * 1.05:
            trend = 'rising'
        elif avg_4wk < avg_8wk * 0.95:
            trend = 'falling'
        else:
            trend = 'stable'

        return round(latest, 0), round(avg_4wk, 0), trend

    def _calculate_yield_curve_metrics(self) -> Tuple[float, bool, int]:
        """
        Calculate yield curve metrics.

        Returns:
            (spread, is_inverted, inversion_days)
        """
        data = self._fetch_series_csv('T10Y2Y')
        if not data or len(data['values']) < 1:
            return 0.0, False, 0

        values = np.array(data['values'])
        dates = data['dates']

        latest_spread = values[-1]
        is_inverted = latest_spread < 0

        # Count consecutive inversion days
        inversion_days = 0
        for i in range(len(values) - 1, -1, -1):
            if values[i] < 0:
                inversion_days += 1
            else:
                break

        return round(latest_spread, 2), is_inverted, inversion_days

    def _calculate_pmi_proxy(self) -> Tuple[float, str]:
        """
        Calculate PMI proxy using Industrial Production.

        Note: ISM PMI not directly available; using IP growth as proxy.

        Returns:
            (pmi_proxy, trend)
        """
        data = self._fetch_series_csv('INDPRO')
        if not data or len(data['values']) < 13:
            return 50.0, 'neutral'

        values = np.array(data['values'])

        # Year-over-year IP growth
        yoy_growth = ((values[-1] / values[-13]) - 1) * 100

        # Convert to PMI-like scale (50 = neutral)
        # Rough conversion: 0% growth = 50, +/-5% = +/-10 points
        pmi_proxy = 50 + (yoy_growth * 2)
        pmi_proxy = max(30, min(70, pmi_proxy))  # Clip to reasonable range

        if pmi_proxy > 52:
            trend = 'expanding'
        elif pmi_proxy < 48:
            trend = 'contracting'
        else:
            trend = 'neutral'

        return round(pmi_proxy, 1), trend

    def _calculate_recession_probability(
        self,
        m2_signal: str,
        claims_trend: str,
        yield_inverted: bool,
        inversion_days: int,
        pmi_trend: str
    ) -> Tuple[float, str]:
        """
        Calculate composite recession probability.

        Based on historical recession indicators.

        Returns:
            (probability 0-100, signal)
        """
        probability = 0.0

        # Yield curve inversion is strongest predictor
        if yield_inverted:
            probability += 25
            # Longer inversions are more predictive
            if inversion_days > 90:
                probability += 15
            elif inversion_days > 30:
                probability += 10

        # Rising jobless claims
        if claims_trend == 'rising':
            probability += 20
        elif claims_trend == 'stable':
            probability += 5

        # Money supply contraction
        if m2_signal == 'contractionary':
            probability += 20
        elif m2_signal == 'neutral':
            probability += 5

        # PMI/Industrial contraction
        if pmi_trend == 'contracting':
            probability += 20
        elif pmi_trend == 'neutral':
            probability += 5

        # Classify signal level
        if probability >= 60:
            signal = 'HIGH'
        elif probability >= 40:
            signal = 'ELEVATED'
        elif probability >= 20:
            signal = 'MODERATE'
        else:
            signal = 'LOW'

        return min(100, probability), signal

    def get_indicators(self) -> MacroIndicators:
        """
        Get current macroeconomic indicators.

        Returns:
            MacroIndicators with all metrics and signals
        """
        print("[FRED] Fetching macroeconomic indicators...")

        # Fetch all metrics
        m2_yoy, m2_3m, m2_signal = self._calculate_m2_metrics()
        claims, claims_avg, claims_trend = self._calculate_claims_metrics()
        yield_spread, yield_inverted, inversion_days = self._calculate_yield_curve_metrics()
        pmi, pmi_trend = self._calculate_pmi_proxy()

        # Calculate recession probability
        recession_prob, recession_signal = self._calculate_recession_probability(
            m2_signal, claims_trend, yield_inverted, inversion_days, pmi_trend
        )

        # Generate recommendation
        if recession_signal == 'HIGH':
            recommendation = "HIGH recession risk - minimize exposure, prioritize capital preservation"
        elif recession_signal == 'ELEVATED':
            recommendation = "Elevated recession risk - reduce position sizes, focus on quality"
        elif recession_signal == 'MODERATE':
            recommendation = "Moderate recession risk - normal positioning with defensive tilt"
        else:
            recommendation = "Low recession risk - favorable conditions for equity exposure"

        return MacroIndicators(
            timestamp=datetime.now().isoformat(),
            m2_growth_yoy=m2_yoy,
            m2_growth_3m=m2_3m,
            m2_signal=m2_signal,
            initial_claims=claims,
            claims_4wk_avg=claims_avg,
            claims_trend=claims_trend,
            yield_spread=yield_spread,
            yield_inverted=yield_inverted,
            inversion_days=inversion_days,
            pmi_composite=pmi,
            pmi_trend=pmi_trend,
            recession_probability=recession_prob,
            recession_signal=recession_signal,
            recommendation=recommendation
        )

    def get_recession_risk_multiplier(self) -> float:
        """
        Get position sizing multiplier based on recession risk.

        Returns:
            Float 0.0-1.0 where lower values indicate higher risk
        """
        indicators = self.get_indicators()

        # Map recession signal to multiplier
        multiplier_map = {
            'LOW': 1.0,
            'MODERATE': 0.85,
            'ELEVATED': 0.65,
            'HIGH': 0.40
        }

        return multiplier_map.get(indicators.recession_signal, 0.85)


def print_macro_indicators():
    """Print current macroeconomic indicators."""
    print("=" * 70)
    print("FRED MACROECONOMIC INDICATORS")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    fetcher = FREDMacroFetcher()
    indicators = fetcher.get_indicators()

    print("--- MONEY SUPPLY (M2) ---")
    print(f"YoY Growth: {indicators.m2_growth_yoy:.1f}%")
    print(f"3M Annualized: {indicators.m2_growth_3m:.1f}%")
    print(f"Signal: {indicators.m2_signal.upper()}")
    print()

    print("--- LABOR MARKET ---")
    print(f"Initial Claims: {indicators.initial_claims:.0f}K")
    print(f"4-Week Avg: {indicators.claims_4wk_avg:.0f}K")
    print(f"Trend: {indicators.claims_trend.upper()}")
    print()

    print("--- YIELD CURVE ---")
    inv_marker = "[INVERTED]" if indicators.yield_inverted else ""
    print(f"10Y-2Y Spread: {indicators.yield_spread:.2f}% {inv_marker}")
    if indicators.yield_inverted:
        print(f"Inversion Days: {indicators.inversion_days}")
    print()

    print("--- ECONOMIC ACTIVITY ---")
    print(f"PMI Proxy: {indicators.pmi_composite:.1f}")
    print(f"Trend: {indicators.pmi_trend.upper()}")
    print()

    print("--- RECESSION RISK ---")
    risk_marker = {
        'HIGH': '!!!',
        'ELEVATED': '!!',
        'MODERATE': '!',
        'LOW': ''
    }.get(indicators.recession_signal, '')
    print(f"Probability: {indicators.recession_probability:.0f}% {risk_marker}")
    print(f"Signal: {indicators.recession_signal}")
    print()

    print("--- RECOMMENDATION ---")
    print(indicators.recommendation)
    print()

    return indicators


if __name__ == "__main__":
    print_macro_indicators()

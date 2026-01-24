"""
UnifiedPositionSizer - Single source of truth for position sizing.

Consolidates logic from:
- position_sizer.py (Kelly criterion, signal quality)
- volatility_sizing.py (ATR adjustment, beta)
- config/position_sizing.py (replaced by unified_config.json)
- Bear score awareness (Jan 2026: reduce size when bear warnings elevated)

All configuration read from config/unified_config.json.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np


class SignalQuality(Enum):
    """Signal quality classification."""
    SKIP = "skip"          # Don't trade
    WEAK = "weak"          # Minimal size
    MODERATE = "moderate"  # Standard size
    STRONG = "strong"      # Increased size
    EXCELLENT = "excellent"  # Maximum size


@dataclass
class PositionRecommendation:
    """Complete position sizing recommendation."""
    ticker: str
    tier: str
    size_pct: float           # Position size as % of portfolio
    dollar_size: float        # Dollar amount
    shares: int               # Number of shares
    risk_dollars: float       # Dollar risk (position * stop loss %)
    quality: SignalQuality
    skip_trade: bool
    rationale: str


class UnifiedPositionSizer:
    """
    Single position sizer using unified_config.json.

    Sizing based on:
    1. Stock tier (from config)
    2. Market regime adjustments
    3. Signal strength
    4. ATR-based volatility adjustment
    """

    CONFIG_PATH = Path('config/unified_config.json')
    POSITIONS_FILE = Path('data/portfolio/active_positions.json')

    def __init__(self, portfolio_value: float = 100000.0, use_bear_adjustment: bool = True):
        self.portfolio_value = portfolio_value
        self.use_bear_adjustment = use_bear_adjustment
        self.config = self._load_config()
        self.current_positions: Dict[str, dict] = {}
        self._load_positions()

        # Cache for bear score
        self._bear_score_cache: Optional[float] = None
        self._bear_score_time: Optional[datetime] = None

        # Build ticker -> tier lookup
        self.ticker_tiers: Dict[str, str] = {}
        for tier_name, tier_data in self.config.get('stock_tiers', {}).items():
            if tier_name.startswith('_'):
                continue
            for ticker in tier_data.get('tickers', []):
                self.ticker_tiers[ticker] = tier_name

    def _load_config(self) -> dict:
        """Load unified configuration."""
        if self.CONFIG_PATH.exists():
            with open(self.CONFIG_PATH) as f:
                return json.load(f)
        return self._default_config()

    def _default_config(self) -> dict:
        """Fallback config if file missing."""
        return {
            'stock_tiers': {
                'elite': {'position_size_pct': 12, 'signal_multiplier': 1.10},
                'strong': {'position_size_pct': 10, 'signal_multiplier': 1.05},
                'average': {'position_size_pct': 7, 'signal_multiplier': 1.00},
                'weak': {'position_size_pct': 5, 'signal_multiplier': 0.90},
                'avoid': {'position_size_pct': 0, 'signal_multiplier': 0.80}
            },
            'regime_adjustments': {
                'volatile': {'position_multiplier': 1.3},
                'bear': {'position_multiplier': 1.2},
                'choppy': {'position_multiplier': 1.0},
                'bull': {'position_multiplier': 0.8}
            },
            'portfolio_constraints': {
                'max_positions': 6,
                'max_portfolio_heat_pct': 15,
                'min_position_size_pct': 2,
                'max_position_size_pct': 15,
                'max_per_sector': 2,
                'risk_per_trade_pct': 2
            },
            'exit_strategy': {
                'average': {'stop_loss': -2.5}
            }
        }

    def _load_positions(self):
        """Load persisted positions."""
        if self.POSITIONS_FILE.exists():
            try:
                with open(self.POSITIONS_FILE) as f:
                    data = json.load(f)
                self.current_positions = data.get('positions', {})
            except Exception:
                self.current_positions = {}

    def _save_positions(self):
        """Save positions to disk."""
        self.POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'last_updated': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_value,
            'positions': self.current_positions
        }
        with open(self.POSITIONS_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def get_tier(self, ticker: str) -> str:
        """Get stock tier from config."""
        return self.ticker_tiers.get(ticker, 'average')

    def get_tier_config(self, tier: str) -> dict:
        """Get configuration for a tier."""
        tiers = self.config.get('stock_tiers', {})
        return tiers.get(tier, tiers.get('average', {}))

    def get_regime_config(self, regime: str) -> dict:
        """Get regime adjustments."""
        regimes = self.config.get('regime_adjustments', {})
        return regimes.get(regime.lower(), regimes.get('choppy', {}))

    def get_constraints(self) -> dict:
        """Get portfolio constraints."""
        return self.config.get('portfolio_constraints', {})

    def get_exit_config(self, tier: str) -> dict:
        """Get exit strategy for tier."""
        exits = self.config.get('exit_strategy', {})
        return exits.get(tier, exits.get('average', {}))

    def get_bear_score(self, force_refresh: bool = False) -> Tuple[float, str]:
        """
        Get current bear score for position size adjustment.

        Returns (bear_score, alert_level).
        Caches for 5 minutes to avoid repeated API calls.
        """
        from datetime import timedelta

        # Check cache (5 minute expiry)
        if (not force_refresh and
            self._bear_score_cache is not None and
            self._bear_score_time is not None and
            datetime.now() - self._bear_score_time < timedelta(minutes=5)):
            return self._bear_score_cache, getattr(self, '_bear_alert_cache', 'NORMAL')

        try:
            from analysis.fast_bear_detector import FastBearDetector
            detector = FastBearDetector()
            signal = detector.detect()
            self._bear_score_cache = signal.bear_score
            self._bear_alert_cache = signal.alert_level
            self._bear_score_time = datetime.now()
            return signal.bear_score, signal.alert_level
        except Exception as e:
            # If bear detector unavailable, assume normal
            return 0.0, 'NORMAL'

    def get_bear_adjustment(self) -> Tuple[float, str]:
        """
        Get position size multiplier based on bear score.

        Returns (multiplier, reason).

        Bear Score Adjustments:
        - 0-29 (NORMAL): 1.0x (no adjustment)
        - 30-49 (WATCH): 0.85x (reduce 15%)
        - 50-69 (WARNING): 0.65x (reduce 35%)
        - 70+ (CRITICAL): 0.40x (reduce 60%)
        """
        if not self.use_bear_adjustment:
            return 1.0, "Bear adjustment disabled"

        bear_score, alert_level = self.get_bear_score()

        if bear_score >= 70:
            return 0.40, f"CRITICAL bear ({bear_score:.0f}) -60%"
        elif bear_score >= 50:
            return 0.65, f"WARNING bear ({bear_score:.0f}) -35%"
        elif bear_score >= 30:
            return 0.85, f"WATCH bear ({bear_score:.0f}) -15%"
        else:
            return 1.0, f"Normal ({bear_score:.0f})"

    def assess_quality(self, signal_strength: float, tier: str) -> SignalQuality:
        """
        Assess signal quality based on strength and tier.

        Simplified scoring:
        - AVOID tier always SKIP
        - Signal >= 80 with elite/strong = EXCELLENT
        - Signal >= 70 with elite/strong = STRONG
        - Signal >= 60 = MODERATE
        - Signal < 60 = WEAK
        """
        if tier == 'avoid':
            return SignalQuality.SKIP

        if signal_strength >= 80:
            if tier in ['elite', 'strong']:
                return SignalQuality.EXCELLENT
            return SignalQuality.STRONG
        elif signal_strength >= 70:
            if tier in ['elite', 'strong']:
                return SignalQuality.STRONG
            return SignalQuality.MODERATE
        elif signal_strength >= 60:
            return SignalQuality.MODERATE
        else:
            return SignalQuality.WEAK

    def get_current_heat(self) -> float:
        """Get total portfolio heat (% allocated)."""
        total = 0.0
        for info in self.current_positions.values():
            if isinstance(info, dict):
                total += info.get('size_pct', 0)
            else:
                total += info
        return total

    def can_add_position(self) -> Tuple[bool, str]:
        """Check if we can add a position."""
        constraints = self.get_constraints()
        max_positions = constraints.get('max_positions', 6)
        max_heat = constraints.get('max_portfolio_heat_pct', 15) / 100

        if len(self.current_positions) >= max_positions:
            return False, f"Max positions reached ({max_positions})"

        current_heat = self.get_current_heat()
        if current_heat >= max_heat:
            return False, f"Heat limit reached ({current_heat*100:.1f}%)"

        return True, "OK"

    def is_holding(self, ticker: str) -> bool:
        """Check if already holding this ticker."""
        return ticker in self.current_positions

    def calculate_size(self,
                       ticker: str,
                       current_price: float,
                       signal_strength: float = 70.0,
                       regime: str = 'choppy',
                       atr_pct: float = 2.5) -> PositionRecommendation:
        """
        Calculate position size using unified config.

        Args:
            ticker: Stock ticker
            current_price: Current price
            signal_strength: Signal strength (0-100)
            regime: Market regime (volatile/bear/choppy/bull)
            atr_pct: ATR as % of price (for vol adjustment)

        Returns:
            PositionRecommendation
        """
        tier = self.get_tier(ticker)
        tier_config = self.get_tier_config(tier)
        regime_config = self.get_regime_config(regime)
        constraints = self.get_constraints()
        exit_config = self.get_exit_config(tier)

        # Assess quality
        quality = self.assess_quality(signal_strength, tier)

        # Skip if poor quality
        if quality == SignalQuality.SKIP:
            return PositionRecommendation(
                ticker=ticker,
                tier=tier,
                size_pct=0.0,
                dollar_size=0.0,
                shares=0,
                risk_dollars=0.0,
                quality=quality,
                skip_trade=True,
                rationale=f"SKIP: {tier.upper()} tier"
            )

        # Base size from tier
        base_size_pct = tier_config.get('position_size_pct', 7) / 100

        # Regime adjustment
        regime_mult = regime_config.get('position_multiplier', 1.0)
        adjusted_size = base_size_pct * regime_mult

        # Signal strength adjustment
        quality_multipliers = {
            SignalQuality.EXCELLENT: 1.2,
            SignalQuality.STRONG: 1.1,
            SignalQuality.MODERATE: 1.0,
            SignalQuality.WEAK: 0.7,
            SignalQuality.SKIP: 0.0
        }
        adjusted_size *= quality_multipliers[quality]

        # Volatility adjustment (Jan 4, 2026: INCREASE for high vol - they outperform!)
        # Research: High Vol 57.4% win +0.905%, Low Vol 49.6% win -0.101%
        # Mean reversion NEEDS volatility to generate returns
        if atr_pct > 0:
            if atr_pct > 3.34:      # High Vol - best performers
                vol_adjustment = 1.25
            elif atr_pct > 2.50:    # Med-High Vol - good
                vol_adjustment = 1.10
            elif atr_pct > 2.05:    # Med-Low Vol - neutral
                vol_adjustment = 0.95
            else:                    # Low Vol - worst performers
                vol_adjustment = 0.70
            adjusted_size *= vol_adjustment

        # Bear score adjustment (Jan 2026: reduce exposure in bearish conditions)
        bear_mult, bear_reason = self.get_bear_adjustment()
        adjusted_size *= bear_mult

        # Apply constraints
        min_size = constraints.get('min_position_size_pct', 2) / 100
        max_size = constraints.get('max_position_size_pct', 15) / 100
        adjusted_size = max(min_size, min(adjusted_size, max_size))

        # Check heat constraint
        current_heat = self.get_current_heat()
        max_heat = constraints.get('max_portfolio_heat_pct', 15) / 100
        available_heat = max_heat - current_heat

        if adjusted_size > available_heat:
            if available_heat < min_size:
                return PositionRecommendation(
                    ticker=ticker,
                    tier=tier,
                    size_pct=0.0,
                    dollar_size=0.0,
                    shares=0,
                    risk_dollars=0.0,
                    quality=quality,
                    skip_trade=True,
                    rationale=f"SKIP: Heat limit ({current_heat*100:.1f}%)"
                )
            adjusted_size = available_heat

        # Calculate dollars and shares
        dollar_size = self.portfolio_value * adjusted_size
        shares = int(dollar_size / current_price) if current_price > 0 else 0
        actual_dollars = shares * current_price

        # Risk calculation
        stop_loss_pct = abs(exit_config.get('stop_loss', -2.5)) / 100
        risk_dollars = actual_dollars * stop_loss_pct

        # Build rationale
        parts = [
            f"{tier.upper()}",
            f"sig={signal_strength:.0f}",
            f"{regime}",
            f"ATR={atr_pct:.1f}%"
        ]
        if bear_mult < 1.0:
            parts.append(f"Bear: {bear_mult:.0%}")
        rationale = " | ".join(parts)

        return PositionRecommendation(
            ticker=ticker,
            tier=tier,
            size_pct=round(adjusted_size * 100, 2),
            dollar_size=round(actual_dollars, 2),
            shares=shares,
            risk_dollars=round(risk_dollars, 2),
            quality=quality,
            skip_trade=False,
            rationale=rationale
        )

    def add_position(self, ticker: str, size_pct: float, entry_price: float = None):
        """Track a new position."""
        self.current_positions[ticker] = {
            'size_pct': size_pct / 100,
            'entry_date': datetime.now().isoformat(),
            'entry_price': entry_price
        }
        self._save_positions()

    def remove_position(self, ticker: str):
        """Remove a closed position."""
        if ticker in self.current_positions:
            del self.current_positions[ticker]
            self._save_positions()

    def get_status(self) -> dict:
        """Get portfolio status."""
        constraints = self.get_constraints()
        current_heat = self.get_current_heat()
        can_add, reason = self.can_add_position()

        return {
            'positions': len(self.current_positions),
            'max_positions': constraints.get('max_positions', 6),
            'can_add': can_add,
            'reason': reason,
            'heat_pct': round(current_heat * 100, 2),
            'max_heat_pct': constraints.get('max_portfolio_heat_pct', 15),
            'available_heat_pct': round((constraints.get('max_portfolio_heat_pct', 15)/100 - current_heat) * 100, 2),
            'holdings': list(self.current_positions.keys())
        }

    def rank_opportunities(self, opportunities: List[dict],
                           regime: str = 'choppy') -> List[dict]:
        """
        Rank opportunities by quality and size.

        Args:
            opportunities: List of dicts with ticker, price, signal_strength
            regime: Current market regime

        Returns:
            Sorted list with position recommendations
        """
        results = []

        for opp in opportunities:
            rec = self.calculate_size(
                ticker=opp['ticker'],
                current_price=opp.get('price', opp.get('current_price', 100)),
                signal_strength=opp.get('signal_strength', 70),
                regime=regime,
                atr_pct=opp.get('atr_pct', 2.5)
            )

            if not rec.skip_trade:
                results.append({
                    **opp,
                    'recommendation': rec
                })

        # Sort by quality then size
        quality_order = {
            SignalQuality.EXCELLENT: 4,
            SignalQuality.STRONG: 3,
            SignalQuality.MODERATE: 2,
            SignalQuality.WEAK: 1,
            SignalQuality.SKIP: 0
        }

        return sorted(
            results,
            key=lambda x: (quality_order[x['recommendation'].quality], x['recommendation'].size_pct),
            reverse=True
        )


def demo():
    """Demonstrate unified position sizer."""
    sizer = UnifiedPositionSizer(portfolio_value=100000)

    print("=" * 70)
    print("UNIFIED POSITION SIZER")
    print("=" * 70)
    print(f"Portfolio: ${sizer.portfolio_value:,.0f}")
    print(f"Config loaded from: {sizer.CONFIG_PATH}")
    print()

    # Test cases spanning tiers
    test_cases = [
        {'ticker': 'COP', 'price': 105.0, 'signal_strength': 85, 'atr_pct': 2.8},   # Elite
        {'ticker': 'QCOM', 'price': 180.0, 'signal_strength': 75, 'atr_pct': 3.2},  # Strong
        {'ticker': 'MSFT', 'price': 420.0, 'signal_strength': 65, 'atr_pct': 2.0},  # Average
        {'ticker': 'NVDA', 'price': 140.0, 'signal_strength': 70, 'atr_pct': 4.0},  # Weak
        {'ticker': 'NOW', 'price': 900.0, 'signal_strength': 80, 'atr_pct': 3.5},   # Avoid
    ]

    print("Position Sizing (VOLATILE regime):")
    print("-" * 70)

    for case in test_cases:
        rec = sizer.calculate_size(
            ticker=case['ticker'],
            current_price=case['price'],
            signal_strength=case['signal_strength'],
            regime='volatile',
            atr_pct=case['atr_pct']
        )

        if rec.skip_trade:
            print(f"{rec.ticker}: SKIP ({rec.rationale})")
        else:
            print(f"{rec.ticker}: {rec.size_pct:.1f}% (${rec.dollar_size:,.0f}, "
                  f"{rec.shares} shares) | {rec.quality.value.upper()} | {rec.rationale}")

    print()
    print("Portfolio Status:")
    print("-" * 70)
    status = sizer.get_status()
    print(f"  Positions: {status['positions']}/{status['max_positions']}")
    print(f"  Heat: {status['heat_pct']}% (max {status['max_heat_pct']}%)")
    print(f"  Available: {status['available_heat_pct']}%")


if __name__ == '__main__':
    demo()

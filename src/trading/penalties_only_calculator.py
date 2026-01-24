"""
Penalties-Only Signal Calculator

Research-backed approach (Jan 5, 2026):
- Boosts inflate trade count without improving quality
- Penalties filter bad trades while preserving quality
- Result: Same win rate, +30% better avg return per trade

This calculator ONLY applies penalties - no positive boosts allowed.
Signals must pass threshold on their own merit.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path


@dataclass
class PenaltyBreakdown:
    """Breakdown of signal calculation with penalties only."""
    ticker: str
    base_signal: float
    tier: str
    tier_multiplier: float
    penalties_applied: Dict[str, float] = field(default_factory=dict)
    final_signal: float = 0.0

    @property
    def total_penalty(self) -> float:
        return sum(self.penalties_applied.values())

    @property
    def boosts_applied(self) -> Dict[str, float]:
        """Alias for compatibility with SmartScannerV2."""
        return self.penalties_applied


class PenaltiesOnlyCalculator:
    """
    Signal calculator using ONLY penalties - no positive boosts.

    Research showed:
    - Full (115 modifiers): 184 trades, 59.2% win, 0.76% avg return
    - Penalties only (12): 8 trades, 62.5% win, 1.72% avg return

    Penalties-only maintains quality while filtering bad setups.
    """

    def __init__(self, config_path: str = None):
        if config_path is None:
            # Navigate from src/trading/ to project root/config/
            config_path = Path(__file__).resolve().parent.parent.parent / 'config' / 'penalties_only_config.json'

        with open(config_path) as f:
            self.config = json.load(f)

        self.penalties = self.config.get('signal_boosts', {})
        self.tiers = self.config.get('stock_tiers', {})

        # For compatibility with unified_config.json structure
        # Load stock_tiers from unified config if not in penalties config
        if not self.tiers:
            unified_path = Path(__file__).resolve().parent.parent.parent / 'config' / 'unified_config.json'
            if unified_path.exists():
                with open(unified_path) as f:
                    unified = json.load(f)
                    self.tiers = unified.get('stock_tiers', {})

        # Count penalties for logging
        self._count_penalties()

    def _count_penalties(self):
        """Count total penalties for logging."""
        count = 0
        for category in self.penalties.values():
            if isinstance(category, dict):
                for penalty in category.values():
                    if isinstance(penalty, dict) and penalty.get('enabled'):
                        count += 1
        self.penalty_count = count

    def get_tier(self, ticker: str) -> str:
        """Get stock tier."""
        for tier_name, tier_data in self.tiers.items():
            if tier_name.startswith('_'):
                continue
            if isinstance(tier_data, dict) and ticker in tier_data.get('tickers', []):
                return tier_name
        return 'average'

    def get_tier_multiplier(self, tier: str) -> float:
        """Get tier multiplier (reduced from full config - less aggressive)."""
        tier_data = self.tiers.get(tier, {})
        if isinstance(tier_data, dict):
            return tier_data.get('signal_multiplier', 1.0)
        return 1.0

    def calculate(self,
                  ticker: str,
                  base_signal: float,
                  regime: str = 'choppy',
                  is_monday: bool = False,
                  is_tuesday: bool = False,
                  is_wednesday: bool = False,
                  is_thursday: bool = False,
                  is_friday: bool = False,
                  consecutive_down_days: int = 0,
                  has_rsi_divergence: bool = False,  # Ignored in penalties-only
                  rsi_level: float = 50.0,
                  volume_ratio: float = 1.0,
                  is_down_day: bool = False,
                  sector: str = 'Technology',
                  sector_momentum: float = 0.0,
                  close_position: float = 0.5,
                  gap_pct: float = 0.0,
                  sma200_distance: float = 0.0,
                  day_range_pct: float = 2.0,
                  drawdown_pct: float = -5.0,
                  atr_pct: float = 2.0) -> PenaltyBreakdown:
        """
        Calculate final signal using ONLY penalties.

        No positive boosts - signals must pass threshold naturally.
        Penalties filter out bad setups.
        """

        penalties = {}

        # =================================================================
        # 1. BASE SIGNAL QUALITY (CRITICAL - strongest penalties)
        # =================================================================
        bsq = self.penalties.get('base_signal_quality', {})

        # Weak base signal - strong penalty
        if base_signal < 55:
            weak_penalty = bsq.get('weak_base_penalty', {})
            if weak_penalty.get('enabled', True):
                penalties['weak_base'] = weak_penalty.get('penalty', -15)

        # Marginal base signal - moderate penalty
        elif base_signal < 60:
            marginal_penalty = bsq.get('marginal_base_penalty', {})
            if marginal_penalty.get('enabled', True):
                penalties['marginal_base'] = marginal_penalty.get('penalty', -8)

        # =================================================================
        # 2. DAY OF WEEK PENALTIES
        # =================================================================
        dow = self.penalties.get('day_of_week_penalties', {})

        # Friday - worst day for mean reversion
        if is_friday:
            friday_pen = dow.get('friday_penalty', {})
            if friday_pen.get('enabled', True):
                penalties['friday'] = friday_pen.get('penalty', -5)

        # Wednesday - mid-week trap
        if is_wednesday:
            wed_pen = dow.get('wednesday_penalty', {})
            if wed_pen.get('enabled', True):
                penalties['wednesday'] = wed_pen.get('penalty', -3)

        # =================================================================
        # 3. FALLING KNIFE DETECTION
        # =================================================================
        fk = self.penalties.get('falling_knife', {})

        # 7+ consecutive down days - likely fundamental issues
        if consecutive_down_days >= 7:
            seven_pen = fk.get('seven_plus_down', {})
            if seven_pen.get('enabled', True):
                penalties['falling_knife_7+'] = seven_pen.get('penalty', -10)

        # 5-6 down days - caution
        elif consecutive_down_days >= 5:
            five_pen = fk.get('five_six_down', {})
            if five_pen.get('enabled', True):
                penalties['falling_knife_5-6'] = five_pen.get('penalty', -5)

        # =================================================================
        # 4. REGIME PENALTIES
        # =================================================================
        reg = self.penalties.get('regime_penalties', {})

        # Bull regime - mean reversion underperforms
        if regime.lower() == 'bull':
            bull_pen = reg.get('bull_regime', {})
            if bull_pen.get('enabled', True):
                penalties['bull_regime'] = bull_pen.get('penalty', -6)

        # =================================================================
        # 5. VOLUME PENALTIES
        # =================================================================
        vol = self.penalties.get('volume_penalties', {})

        # Low volume - weak conviction
        if volume_ratio < 0.6:
            low_vol = vol.get('low_volume', {})
            if low_vol.get('enabled', True):
                penalties['low_volume'] = low_vol.get('penalty', -5)

        # =================================================================
        # 6. TECHNICAL PENALTIES
        # =================================================================
        tech = self.penalties.get('technical_penalties', {})

        # Far from SMA200 - no technical anchor
        if abs(sma200_distance) > 15:
            far_sma = tech.get('far_from_sma200', {})
            if far_sma.get('enabled', True):
                penalties['far_from_sma200'] = far_sma.get('penalty', -4)

        # Minimal drawdown - not really oversold
        if drawdown_pct > -3:
            min_dd = tech.get('minimal_drawdown', {})
            if min_dd.get('enabled', True):
                penalties['minimal_drawdown'] = min_dd.get('penalty', -5)

        # =================================================================
        # 7. SECTOR TRAPS
        # =================================================================
        sector_traps = self.penalties.get('sector_traps', {})

        # Consumer discretionary underperforms
        if sector.lower() in ['consumer discretionary', 'consumer']:
            consumer_trap = sector_traps.get('consumer_trap', {})
            if consumer_trap.get('enabled', True):
                penalties['consumer_sector'] = consumer_trap.get('penalty', -6)

        # =================================================================
        # CALCULATE FINAL SIGNAL
        # =================================================================

        tier = self.get_tier(ticker)
        tier_mult = self.get_tier_multiplier(tier)

        # Apply tier multiplier to base signal
        adjusted_base = base_signal * tier_mult

        # Apply penalties (all negative)
        total_penalty = sum(penalties.values())
        final_signal = adjusted_base + total_penalty

        return PenaltyBreakdown(
            ticker=ticker,
            base_signal=base_signal,
            tier=tier,
            tier_multiplier=tier_mult,
            penalties_applied=penalties,
            final_signal=round(final_signal, 1)
        )

    def format_breakdown(self, breakdown: PenaltyBreakdown) -> str:
        """Format breakdown for display."""
        lines = [
            f"Signal Breakdown for {breakdown.ticker}:",
            f"  Base signal: {breakdown.base_signal:.1f}",
            f"  Tier: {breakdown.tier} (x{breakdown.tier_multiplier:.2f})",
            f"  Penalties applied: {len(breakdown.penalties_applied)}"
        ]

        if breakdown.penalties_applied:
            for name, value in sorted(breakdown.penalties_applied.items(), key=lambda x: x[1]):
                lines.append(f"    {name}: {value:+.0f}")

        lines.append(f"  Total penalty: {breakdown.total_penalty:+.0f}")
        lines.append(f"  Final signal: {breakdown.final_signal:.1f}")

        return '\n'.join(lines)


# Quick test
if __name__ == '__main__':
    calc = PenaltiesOnlyCalculator()
    print(f"Penalties-Only Calculator loaded with {calc.penalty_count} penalties")
    print()

    # Test cases
    test_cases = [
        # Strong signal, good conditions - should pass
        {'ticker': 'MPC', 'base_signal': 72, 'regime': 'volatile',
         'is_monday': True, 'consecutive_down_days': 2, 'drawdown_pct': -8},

        # Weak signal - should be heavily penalized
        {'ticker': 'NVDA', 'base_signal': 52, 'regime': 'choppy',
         'is_friday': True, 'consecutive_down_days': 1, 'drawdown_pct': -2},

        # Falling knife - should be blocked
        {'ticker': 'MRVL', 'base_signal': 65, 'regime': 'choppy',
         'is_monday': True, 'consecutive_down_days': 8, 'drawdown_pct': -20},
    ]

    for tc in test_cases:
        result = calc.calculate(**tc)
        print(calc.format_breakdown(result))
        print()

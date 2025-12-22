"""
Enhanced Signal Strength Calculator

Combines all research findings into a unified signal strength formula:
1. Base signal strength from GPU model
2. Regime adjustments (VOLATILE/BEAR better for mean reversion)
3. Sector momentum (weak sectors outperform by +0.27%)
4. Day-of-week effects (Monday has +0.86% edge)
5. Consecutive down-day patterns (2 days optimal, +0.95% avg)
6. Volume profile (high volume = capitulation, +1.73% edge)
7. Stock tier adjustments (elite vs weak performers)

Deployed: 2025-12-22
Sources: regime_analysis.json, sector_momentum_analysis.json, day_of_week_analysis.json,
         consecutive_down_days_analysis.json, volume_profile_analysis.json
"""

import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pandas as pd


@dataclass
class SignalAdjustments:
    """Container for all signal adjustments."""
    base_strength: float
    regime_multiplier: float
    sector_momentum_multiplier: float
    day_of_week_multiplier: float
    consecutive_down_multiplier: float
    volume_multiplier: float
    stock_tier_multiplier: float
    final_strength: float

    # Debug info
    regime: str
    sector_name: str
    sector_momentum: float
    day_name: str
    consecutive_down_days: int
    volume_ratio: float
    stock_tier: str


class EnhancedSignalCalculator:
    """
    Calculates enhanced signal strength using all research findings.
    """

    # Day-of-week multipliers (based on 365-day analysis)
    # Monday: +0.86% edge, best day for mean reversion
    DAY_MULTIPLIERS = {
        0: 1.05,  # Monday - best day (+0.86% edge)
        1: 1.00,  # Tuesday - baseline
        2: 1.00,  # Wednesday - baseline
        3: 1.02,  # Thursday - slight edge
        4: 1.00   # Friday - baseline
    }

    # Consecutive down-day multipliers (based on 365-day analysis)
    # 2 days optimal (+0.95% avg return)
    DOWN_DAY_MULTIPLIERS = {
        0: 1.10,  # No down days - actually good (fresh oversold)
        1: 1.00,  # 1 day - baseline
        2: 1.10,  # 2 days - optimal (+0.95% avg)
        3: 1.10,  # 3-4 days - still good
        4: 1.10,
        'extended': 1.00  # 5+ days - getting exhausted
    }

    # Volume ratio multipliers (based on 365-day analysis)
    # High volume = capitulation, +1.73% edge
    VOLUME_THRESHOLDS = {
        'low': (0, 0.8, 0.90),       # Low volume: penalty
        'normal': (0.8, 1.5, 1.00),  # Normal: baseline
        'high': (1.5, 2.5, 1.10),    # High: capitulation potential
        'very_high': (2.5, float('inf'), 1.15)  # Very high: strong capitulation
    }

    # Regime multipliers (based on 1-year regime analysis)
    REGIME_MULTIPLIERS = {
        'volatile': 1.15,  # 62% win rate, +1.94% avg
        'bear': 1.10,      # 69% win rate, +0.89% avg
        'choppy': 1.00,    # 56% win rate, +0.17% avg - baseline
        'bull': 0.85       # 51% win rate, +0.15% avg - worst
    }

    # Sector momentum multipliers (based on sector momentum analysis)
    SECTOR_MOMENTUM_MULTIPLIERS = {
        'weak': 1.10,          # <-3%: 56% win, +0.62% avg
        'slightly_weak': 1.05, # -3% to -1%: 57.6% win
        'neutral': 1.00,       # -1% to +1%: baseline
        'slightly_strong': 0.95,  # +1% to +3%
        'strong': 0.90         # >+3%: 49.7% win
    }

    # Stock tier multipliers (based on per-stock backtest)
    STOCK_TIER_MULTIPLIERS = {
        'elite': 1.10,    # COP, CVS, SLB, XOM, ADI, GILD, JPM, EOG, IDXX, TXN
        'strong': 1.05,   # QCOM, JNJ, V, MPC, SHW, KLAC, MS, AMAT
        'average': 1.00,  # Baseline
        'weak': 0.90,     # NVDA, AVGO, AXP, WMT, etc.
        'avoid': 0.80     # NOW, CAT, CRM, HCA, TGT, ETN, HD, ORCL, ADBE, INTU
    }

    # Consumer sector consistently underperforms - additional penalty
    AVOID_SECTORS = ['Consumer', 'XLY']

    def __init__(self):
        pass

    def get_day_of_week_multiplier(self, date: datetime) -> Tuple[float, str]:
        """Get multiplier based on day of week."""
        dow = date.weekday()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_name = day_names[dow] if dow < 5 else 'Weekend'
        mult = self.DAY_MULTIPLIERS.get(dow, 1.0)
        return mult, day_name

    def get_consecutive_down_multiplier(self, consecutive_down_days: int) -> float:
        """Get multiplier based on consecutive down days."""
        if consecutive_down_days >= 5:
            return self.DOWN_DAY_MULTIPLIERS['extended']
        return self.DOWN_DAY_MULTIPLIERS.get(consecutive_down_days, 1.0)

    def get_volume_multiplier(self, volume_ratio: float) -> float:
        """Get multiplier based on volume ratio."""
        for level, (low, high, mult) in self.VOLUME_THRESHOLDS.items():
            if low <= volume_ratio < high:
                return mult
        return 1.0

    def get_regime_multiplier(self, regime: str) -> float:
        """Get multiplier based on market regime."""
        return self.REGIME_MULTIPLIERS.get(regime.lower(), 1.0)

    def get_sector_momentum_multiplier(self, momentum: float, sector_name: str) -> float:
        """Get multiplier based on sector momentum."""
        # Additional penalty for Consumer sector
        sector_penalty = 0.90 if sector_name in self.AVOID_SECTORS else 1.0

        # Momentum category
        if momentum < -3:
            momentum_mult = self.SECTOR_MOMENTUM_MULTIPLIERS['weak']
        elif momentum < -1:
            momentum_mult = self.SECTOR_MOMENTUM_MULTIPLIERS['slightly_weak']
        elif momentum < 1:
            momentum_mult = self.SECTOR_MOMENTUM_MULTIPLIERS['neutral']
        elif momentum < 3:
            momentum_mult = self.SECTOR_MOMENTUM_MULTIPLIERS['slightly_strong']
        else:
            momentum_mult = self.SECTOR_MOMENTUM_MULTIPLIERS['strong']

        return momentum_mult * sector_penalty

    def get_stock_tier_multiplier(self, ticker: str) -> Tuple[float, str]:
        """Get multiplier based on stock performance tier."""
        elite_stocks = ['COP', 'CVS', 'SLB', 'XOM', 'ADI', 'GILD', 'JPM', 'EOG', 'IDXX', 'TXN']
        strong_stocks = ['QCOM', 'JNJ', 'V', 'MPC', 'SHW', 'KLAC', 'MS', 'AMAT']
        weak_stocks = ['NVDA', 'AVGO', 'AXP', 'WMT', 'CMCSA', 'META', 'INSM', 'ROAD', 'MRVL', 'MLM', 'LOW', 'PNC', 'ECL']
        avoid_stocks = ['NOW', 'CAT', 'CRM', 'HCA', 'TGT', 'ETN', 'HD', 'ORCL', 'ADBE', 'INTU']

        if ticker in elite_stocks:
            return self.STOCK_TIER_MULTIPLIERS['elite'], 'elite'
        elif ticker in strong_stocks:
            return self.STOCK_TIER_MULTIPLIERS['strong'], 'strong'
        elif ticker in avoid_stocks:
            return self.STOCK_TIER_MULTIPLIERS['avoid'], 'avoid'
        elif ticker in weak_stocks:
            return self.STOCK_TIER_MULTIPLIERS['weak'], 'weak'
        else:
            return self.STOCK_TIER_MULTIPLIERS['average'], 'average'

    def calculate_enhanced_strength(
        self,
        ticker: str,
        base_strength: float,
        regime: str = 'choppy',
        sector_name: str = '',
        sector_momentum: float = 0.0,
        signal_date: Optional[datetime] = None,
        consecutive_down_days: int = 1,
        volume_ratio: float = 1.0
    ) -> SignalAdjustments:
        """
        Calculate enhanced signal strength using all research findings.

        Args:
            ticker: Stock ticker
            base_strength: Raw signal strength from GPU model (0-100)
            regime: Market regime (volatile, bear, choppy, bull)
            sector_name: Sector name or ETF
            sector_momentum: 5-day sector momentum in percent
            signal_date: Date of the signal
            consecutive_down_days: Number of consecutive down days before signal
            volume_ratio: Current volume / 20-day average volume

        Returns:
            SignalAdjustments with all multipliers and final strength
        """
        if signal_date is None:
            signal_date = datetime.now()

        # Get all multipliers
        regime_mult = self.get_regime_multiplier(regime)
        sector_mult = self.get_sector_momentum_multiplier(sector_momentum, sector_name)
        dow_mult, day_name = self.get_day_of_week_multiplier(signal_date)
        down_mult = self.get_consecutive_down_multiplier(consecutive_down_days)
        vol_mult = self.get_volume_multiplier(volume_ratio)
        tier_mult, tier_name = self.get_stock_tier_multiplier(ticker)

        # Calculate final strength
        # Apply multipliers in order, cap at 100
        adjusted_strength = base_strength
        adjusted_strength *= regime_mult
        adjusted_strength *= sector_mult
        adjusted_strength *= dow_mult
        adjusted_strength *= down_mult
        adjusted_strength *= vol_mult
        adjusted_strength *= tier_mult

        # Cap at 100
        final_strength = min(100.0, adjusted_strength)

        return SignalAdjustments(
            base_strength=base_strength,
            regime_multiplier=regime_mult,
            sector_momentum_multiplier=sector_mult,
            day_of_week_multiplier=dow_mult,
            consecutive_down_multiplier=down_mult,
            volume_multiplier=vol_mult,
            stock_tier_multiplier=tier_mult,
            final_strength=round(final_strength, 1),
            regime=regime,
            sector_name=sector_name,
            sector_momentum=sector_momentum,
            day_name=day_name,
            consecutive_down_days=consecutive_down_days,
            volume_ratio=volume_ratio,
            stock_tier=tier_name
        )

    def print_adjustment_breakdown(self, adj: SignalAdjustments):
        """Print detailed breakdown of signal adjustments."""
        print(f"\n{'='*50}")
        print("SIGNAL STRENGTH BREAKDOWN")
        print(f"{'='*50}")
        print(f"Base strength:                {adj.base_strength:.1f}")
        print(f"  x Regime ({adj.regime}):    {adj.regime_multiplier:.2f}x")
        print(f"  x Sector ({adj.sector_name}, {adj.sector_momentum:+.1f}%): {adj.sector_momentum_multiplier:.2f}x")
        print(f"  x Day ({adj.day_name}):     {adj.day_of_week_multiplier:.2f}x")
        print(f"  x Down days ({adj.consecutive_down_days}):   {adj.consecutive_down_multiplier:.2f}x")
        print(f"  x Volume ({adj.volume_ratio:.1f}x):   {adj.volume_multiplier:.2f}x")
        print(f"  x Tier ({adj.stock_tier}):  {adj.stock_tier_multiplier:.2f}x")
        print(f"{'='*50}")
        print(f"FINAL STRENGTH:               {adj.final_strength:.1f}")
        print()


# Singleton instance
_calculator = None


def get_enhanced_calculator() -> EnhancedSignalCalculator:
    """Get singleton calculator instance."""
    global _calculator
    if _calculator is None:
        _calculator = EnhancedSignalCalculator()
    return _calculator


if __name__ == '__main__':
    # Test the calculator
    calc = EnhancedSignalCalculator()

    # Test case 1: Optimal conditions
    print("\nTEST 1: OPTIMAL CONDITIONS")
    print("(Monday, volatile regime, weak sector, 2 down days, high volume, elite stock)")
    adj = calc.calculate_enhanced_strength(
        ticker='COP',
        base_strength=65.0,
        regime='volatile',
        sector_name='Energy',
        sector_momentum=-4.5,  # Weak sector
        signal_date=datetime(2025, 12, 23),  # Monday
        consecutive_down_days=2,
        volume_ratio=2.0  # High volume
    )
    calc.print_adjustment_breakdown(adj)

    # Test case 2: Poor conditions
    print("\nTEST 2: POOR CONDITIONS")
    print("(Wednesday, bull regime, strong sector, 5+ down days, low volume, avoid stock)")
    adj = calc.calculate_enhanced_strength(
        ticker='NOW',
        base_strength=65.0,
        regime='bull',
        sector_name='Technology',
        sector_momentum=4.0,  # Strong sector
        signal_date=datetime(2025, 12, 24),  # Wednesday
        consecutive_down_days=6,
        volume_ratio=0.5  # Low volume
    )
    calc.print_adjustment_breakdown(adj)

    # Test case 3: Mixed conditions
    print("\nTEST 3: MIXED CONDITIONS")
    print("(Friday, choppy regime, neutral sector, 1 down day, normal volume, average stock)")
    adj = calc.calculate_enhanced_strength(
        ticker='MSFT',
        base_strength=70.0,
        regime='choppy',
        sector_name='Technology',
        sector_momentum=0.5,
        signal_date=datetime(2025, 12, 26),  # Friday
        consecutive_down_days=1,
        volume_ratio=1.1
    )
    calc.print_adjustment_breakdown(adj)

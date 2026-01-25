"""
UnifiedSignalCalculator - Simplified signal calculation using unified_config.json.

Replaces enhanced_signal_calculator.py with:
1. Additive boosts instead of multiplicative (simpler, predictable)
2. Reads all config from unified_config.json (no duplication)
3. Clear formula: final = base * tier_mult * regime_mult + sum(boosts)

Formula:
  adjusted_base = base_signal * stock_tier_multiplier * regime_multiplier
  final_signal = adjusted_base + sum(applicable_boosts)
  capped at min=0, max=100
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SignalBreakdown:
    """Complete breakdown of signal calculation."""
    ticker: str
    base_signal: float
    tier: str
    tier_multiplier: float
    regime: str
    regime_multiplier: float
    boosts_applied: Dict[str, float] = field(default_factory=dict)
    final_signal: float = 0.0

    def total_boost(self) -> float:
        return sum(self.boosts_applied.values())


class UnifiedSignalCalculator:
    """
    Calculate signal strength using unified configuration.

    All multipliers and boosts read from config/unified_config.json.
    """

    CONFIG_PATH = Path('config/unified_config.json')

    def __init__(self):
        self.config = self._load_config()

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
        """Fallback config."""
        return {
            'stock_tiers': {
                'elite': {'signal_multiplier': 1.10},
                'strong': {'signal_multiplier': 1.05},
                'average': {'signal_multiplier': 1.00},
                'weak': {'signal_multiplier': 0.90},
                'avoid': {'signal_multiplier': 0.80}
            },
            'regime_adjustments': {
                'volatile': {'signal_multiplier': 1.15},
                'bear': {'signal_multiplier': 1.10},
                'choppy': {'signal_multiplier': 1.00},
                'bull': {'signal_multiplier': 0.90}
            },
            'signal_boosts': {
                'rsi_divergence': {'enabled': True, 'boost': 5},
                'consecutive_down_6plus': {'enabled': True, 'boost': 8},
                'volume_exhaustion': {'enabled': True, 'boost': 6},
                'monday_effect': {'enabled': True, 'boost': 3},
                'weak_sector': {'enabled': True, 'boost': 5}
            },
            'signal_thresholds': {
                'base_threshold': 50,
                'min_threshold': 35,
                'max_threshold': 70
            }
        }

    def get_tier(self, ticker: str) -> str:
        """Get stock tier."""
        return self.ticker_tiers.get(ticker, 'average')

    def get_tier_multiplier(self, tier: str) -> float:
        """Get tier signal multiplier."""
        tiers = self.config.get('stock_tiers', {})
        tier_config = tiers.get(tier, tiers.get('average', {}))
        return tier_config.get('signal_multiplier', 1.0)

    def get_regime_multiplier(self, regime: str) -> float:
        """Get regime signal multiplier."""
        regimes = self.config.get('regime_adjustments', {})
        regime_config = regimes.get(regime.lower(), regimes.get('choppy', {}))
        return regime_config.get('signal_multiplier', 1.0)

    def get_thresholds(self) -> Tuple[float, float, float]:
        """Get signal thresholds."""
        thresholds = self.config.get('signal_thresholds', {})
        return (
            thresholds.get('base_threshold', 50),
            thresholds.get('min_threshold', 35),
            thresholds.get('max_threshold', 70)
        )

    def calculate(
        self,
        ticker: str,
        base_signal: float,
        regime: str = 'choppy',
        is_monday: bool = False,
        is_thursday: bool = False,
        is_wednesday: bool = False,  # Worst day for mean reversion (Jan 5, 2026)
        is_tuesday: bool = False,  # Tuesday + very_oversold = TRAP (Jan 5, 2026)
        is_friday: bool = False,  # Friday = best 3-day return (Jan 5, 2026)
        consecutive_down_days: int = 0,
        has_rsi_divergence: bool = False,
        rsi_level: float = 50.0,  # RSI level for oversold combos (Jan 5, 2026)
        volume_ratio: float = 1.0,
        is_down_day: bool = False,
        sector_momentum: float = 0.0,
        sector: str = '',  # Sector name for sector-specific boosts (Jan 5, 2026)
        close_position: float = 0.5,  # 0=low, 1=high (new: Jan 4, 2026)
        gap_pct: float = 0.0,  # Gap percentage (new: Jan 5, 2026)
        sma200_distance: float = 0.0,  # Distance from SMA200 % (new: Jan 5, 2026)
        day_range_pct: float = 2.0,  # Day range as % of price (new: Jan 5, 2026)
        drawdown_pct: float = 0.0,  # Drawdown from 20d high % (new: Jan 5, 2026)
        atr_pct: float = 2.5  # ATR as % of price for volatility boosts (Jan 5, 2026)
    ) -> SignalBreakdown:
        """
        Calculate enhanced signal strength.

        Args:
            ticker: Stock ticker
            base_signal: Raw signal from model (0-100)
            regime: Market regime
            is_monday: Is today Monday? (58.3% win rate, +1.37% with moderately_oversold)
            is_thursday: Is today Thursday? (57.5% win rate, 63.3% with very_oversold)
            is_wednesday: Is today Wednesday? (40.3% 1d win rate - WORST)
            is_tuesday: Is today Tuesday? (46.5% win rate with very_oversold - TRAP)
            is_friday: Is today Friday? (+1.005% 3-day return - hold through weekend)
            rsi_level: RSI level (< 30 = very_oversold, 30-40 = moderately_oversold)
            consecutive_down_days: Consecutive down days
            has_rsi_divergence: Bullish RSI divergence detected?
            volume_ratio: Current volume / 20d avg
            is_down_day: Is today a down day?
            sector_momentum: 5d sector momentum (%)
            sector: Sector name (Technology, Financials, Energy, etc.)
            close_position: Close position in day's range (0=low, 1=high)
                           Research: near low = 57.4% win, near high = 50.2% win
            gap_pct: Gap percentage (today's open vs yesterday's close)
                    Research: medium gap (-2% to -3%) = 58.6% win (BEST)
            sma200_distance: Distance from SMA200 as percentage
                            Research: near SMA200 (within 5%) = 57.6% win
            day_range_pct: Day's trading range as percentage
                          Research: very wide (>5%) = 60.1% win, narrow (<1.5%) = 48.3% win
            drawdown_pct: Drawdown from 20-day high as percentage
                         Research: severe (-10%+) = 61.2% win, +1.55% return

        Returns:
            SignalBreakdown with complete calculation
        """
        tier = self.get_tier(ticker)
        tier_mult = self.get_tier_multiplier(tier)
        regime_mult = self.get_regime_multiplier(regime)

        # Step 1: Apply multiplicative adjustments (only tier and regime)
        adjusted_base = base_signal * tier_mult * regime_mult

        # Step 2: Apply additive boosts
        boosts = self.config.get('signal_boosts', {})
        applied_boosts = {}

        # Monday effect
        if is_monday and boosts.get('monday_effect', {}).get('enabled'):
            applied_boosts['monday'] = boosts['monday_effect']['boost']

        # Thursday effect (Jan 5, 2026)
        # Research: 57.5% 2-day win rate, Thursday + very_oversold = 63.3%
        if is_thursday and boosts.get('thursday_effect', {}).get('enabled'):
            applied_boosts['thursday'] = boosts['thursday_effect']['boost']

        # Wednesday penalty (Jan 5, 2026) - WORST DAY
        # Research: 40.3% 1-day win rate, -0.38% avg return
        if is_wednesday and boosts.get('wednesday_penalty', {}).get('enabled'):
            applied_boosts['wednesday_trap'] = boosts['wednesday_penalty'].get('penalty', -4)

        # Friday extended hold (Jan 5, 2026) - BEST 3-DAY RETURN
        # Research: +1.005% avg 3-day return (best of all days)
        if is_friday and boosts.get('friday_extended_hold', {}).get('enabled'):
            applied_boosts['friday_hold'] = boosts['friday_extended_hold']['boost']

        # Friday + very_oversold TRAP (Jan 5, 2026)
        # Research: Friday + RSI < 30 = 50.6% win, -0.12% return (despite Friday being good overall!)
        if is_friday and rsi_level < 30 and boosts.get('friday_very_oversold_trap', {}).get('enabled'):
            applied_boosts['friday_vo_trap'] = boosts['friday_very_oversold_trap'].get('penalty', -4)

        # Friday + slightly_oversold (Jan 5, 2026) - Improvement #87
        # Research: Friday + RSI 40-50 = 55% win, +0.36% return (200 signals - HIGH FREQUENCY)
        if is_friday and 40 <= rsi_level < 50 and boosts.get('friday_slightly_oversold', {}).get('enabled'):
            applied_boosts['friday_so'] = boosts['friday_slightly_oversold']['boost']

        # Tuesday + very_oversold TRAP (Jan 5, 2026)
        # Research: Tuesday + RSI < 30 = 46.5% win rate (TRAP pattern!)
        if is_tuesday and rsi_level < 30 and boosts.get('tuesday_very_oversold_trap', {}).get('enabled'):
            applied_boosts['tuesday_vo_trap'] = boosts['tuesday_very_oversold_trap'].get('penalty', -6)

        # Monday + moderately_oversold BEST COMBO (Jan 5, 2026)
        # Research: Monday + RSI 30-40 = 62.8% win rate, +1.37% return
        if is_monday and 30 <= rsi_level < 40 and boosts.get('monday_moderately_oversold', {}).get('enabled'):
            applied_boosts['monday_mo_combo'] = boosts['monday_moderately_oversold']['boost']

        # Monday + slightly_oversold (Jan 5, 2026) - HIGH FREQUENCY!
        # Research: Monday + RSI 40-50 = 56.5% win, +0.88% return (223 signals)
        if is_monday and 40 <= rsi_level < 50 and boosts.get('monday_slightly_oversold', {}).get('enabled'):
            applied_boosts['monday_so_combo'] = boosts['monday_slightly_oversold']['boost']

        # Tuesday + slightly_oversold TRAP (Jan 5, 2026) - Improvement #81
        # Research: Tuesday + RSI 40-50 = 53.1% win, -0.01% return (196 signals)
        if is_tuesday and 40 <= rsi_level < 50 and boosts.get('tuesday_slightly_oversold_trap', {}).get('enabled'):
            applied_boosts['tuesday_so_trap'] = boosts['tuesday_slightly_oversold_trap'].get('penalty', -2)

        # Thursday + slightly_oversold TRAP (Jan 5, 2026) - Improvement #82
        # Research: Thursday + RSI 40-50 = 54.1% win, -0.02% return (172 signals)
        if is_thursday and 40 <= rsi_level < 50 and boosts.get('thursday_slightly_oversold_trap', {}).get('enabled'):
            applied_boosts['thursday_so_trap'] = boosts['thursday_slightly_oversold_trap'].get('penalty', -2)

        # Wednesday + slightly_oversold TRAP (Jan 5, 2026) - Improvement #83
        # Research: Wednesday + RSI 40-50 = 52.8% win, -0.12% return (193 signals)
        if is_wednesday and 40 <= rsi_level < 50 and boosts.get('wednesday_slightly_oversold_trap', {}).get('enabled'):
            applied_boosts['wednesday_so_trap'] = boosts['wednesday_slightly_oversold_trap'].get('penalty', -3)

        # Wednesday + moderately_oversold (Jan 5, 2026) - Improvement #88
        # Research: Wednesday + RSI 30-40 = 56.4% win, +0.33% return (110 signals)
        if is_wednesday and 30 <= rsi_level < 40 and boosts.get('wednesday_moderately_oversold', {}).get('enabled'):
            applied_boosts['wednesday_mo'] = boosts['wednesday_moderately_oversold']['boost']

        # Tuesday + moderately_oversold TRAP (Jan 5, 2026) - Improvement #100
        # Research: Tuesday + RSI 30-40 = ~50.5% win (between very/slightly traps)
        if is_tuesday and 30 <= rsi_level < 40 and boosts.get('tuesday_moderately_oversold_trap', {}).get('enabled'):
            applied_boosts['tuesday_mo_trap'] = boosts['tuesday_moderately_oversold_trap'].get('penalty', -3)

        # Wednesday + very_oversold TRAP (Jan 5, 2026) - Improvement #101
        # Research: Wednesday (worst day) + RSI < 30 = ~48.2% win (volatile trap)
        if is_wednesday and rsi_level < 30 and boosts.get('wednesday_very_oversold_trap', {}).get('enabled'):
            applied_boosts['wednesday_vo_trap'] = boosts['wednesday_very_oversold_trap'].get('penalty', -5)

        # Thursday + moderately_oversold (Jan 5, 2026) - Improvement #102
        # Research: Thursday + RSI 30-40 = ~58.9% win (between 63.3% very and 54.1% slightly)
        if is_thursday and 30 <= rsi_level < 40 and boosts.get('thursday_moderately_oversold', {}).get('enabled'):
            applied_boosts['thursday_mo'] = boosts['thursday_moderately_oversold']['boost']

        # Friday + moderately_oversold TRAP (Jan 5, 2026) - Improvement #103
        # Research: Friday + RSI 30-40 = ~52.3% win (between very trap and slightly boost)
        if is_friday and 30 <= rsi_level < 40 and boosts.get('friday_moderately_oversold_trap', {}).get('enabled'):
            applied_boosts['friday_mo_trap'] = boosts['friday_moderately_oversold_trap'].get('penalty', -1)

        # Monday + very_oversold (Jan 5, 2026)
        # Research: Monday + RSI < 30 = 57.4% win, +0.75% return (176 signals)
        if is_monday and rsi_level < 30 and boosts.get('monday_very_oversold', {}).get('enabled'):
            applied_boosts['monday_vo_combo'] = boosts['monday_very_oversold']['boost']

        # Thursday + very_oversold STRONG COMBO (Jan 5, 2026)
        # Research: Thursday + RSI < 30 = 63.3% win rate
        if is_thursday and rsi_level < 30 and boosts.get('thursday_very_oversold', {}).get('enabled'):
            applied_boosts['thursday_vo_combo'] = boosts['thursday_very_oversold']['boost']

        # RSI divergence
        if has_rsi_divergence and boosts.get('rsi_divergence', {}).get('enabled'):
            applied_boosts['rsi_divergence'] = boosts['rsi_divergence']['boost']

        # 6+ consecutive down days
        if consecutive_down_days >= 6 and boosts.get('consecutive_down_6plus', {}).get('enabled'):
            applied_boosts['down_streak_6+'] = boosts['consecutive_down_6plus']['boost']

        # Exactly 5 consecutive down days (Jan 5, 2026) - Improvement #89
        # Research: 5 days exactly = 61% win, +0.64% return (77 trades)
        if consecutive_down_days == 5 and boosts.get('consecutive_down_5_exact', {}).get('enabled'):
            applied_boosts['down_5_exact'] = boosts['consecutive_down_5_exact']['boost']

        # Exactly 6 consecutive down days BONUS (Jan 5, 2026)
        # Research: 6 days exactly = 69.4% win, +1.25% return (better than 5d or 7d!)
        if consecutive_down_days == 6 and boosts.get('consecutive_down_6_exact', {}).get('enabled'):
            applied_boosts['down_6_exact'] = boosts['consecutive_down_6_exact']['boost']

        # Exactly 7 consecutive down days (Jan 5, 2026) - Improvement #99
        # Research: 7 days exactly = 60% win, +0.84% return (10 signals - extreme exhaustion)
        if consecutive_down_days == 7 and boosts.get('consecutive_down_7_exact', {}).get('enabled'):
            applied_boosts['down_7_exact'] = boosts['consecutive_down_7_exact']['boost']

        # Volume exhaustion (high volume on down day)
        if is_down_day and volume_ratio >= 3.0 and boosts.get('volume_exhaustion', {}).get('enabled'):
            applied_boosts['volume_exhaustion'] = boosts['volume_exhaustion']['boost']

        # Weak sector (general)
        if sector_momentum < -3.0 and boosts.get('weak_sector', {}).get('enabled'):
            applied_boosts['weak_sector'] = boosts['weak_sector']['boost']

        # Sector-specific momentum boosts (Jan 5, 2026)
        # Research: Specific sectors have MUCH better patterns than general rule
        sector_lower = sector.lower() if sector else ''

        # Industrials + Weak Sector: 76.2% win rate, +1.84% return
        if 'industrial' in sector_lower and sector_momentum < -3.0 and boosts.get('industrials_weak_sector', {}).get('enabled'):
            applied_boosts['industrials_weak'] = boosts['industrials_weak_sector']['boost']
            # Remove general weak_sector boost - this is stronger
            if 'weak_sector' in applied_boosts:
                del applied_boosts['weak_sector']

        # Energy + Slightly Strong: 74.1% win rate, +1.08% return
        if 'energy' in sector_lower and sector_momentum > 1.0 and boosts.get('energy_positive_momentum', {}).get('enabled'):
            applied_boosts['energy_positive'] = boosts['energy_positive_momentum']['boost']

        # Energy + Weak Sector TRAP (Jan 5, 2026) - COUNTERINTUITIVE!
        # Research: Energy + weak momentum = ONLY 43.2% win, -0.35% return
        # This contradicts the general weak_sector boost for energy stocks!
        if 'energy' in sector_lower and sector_momentum < -3.0 and boosts.get('energy_weak_sector_trap', {}).get('enabled'):
            applied_boosts['energy_weak_trap'] = boosts['energy_weak_sector_trap'].get('penalty', -6)
            # Remove the general weak_sector boost - it's a trap for energy!
            if 'weak_sector' in applied_boosts:
                del applied_boosts['weak_sector']

        # Energy + Slightly Weak TRAP (Jan 5, 2026) - Improvement #80
        # Research: Energy + momentum -3% to -1% = 50% win, -0.35% return
        if 'energy' in sector_lower and -3.0 <= sector_momentum < -1.0 and boosts.get('energy_slightly_weak_trap', {}).get('enabled'):
            applied_boosts['energy_sw_trap'] = boosts['energy_slightly_weak_trap'].get('penalty', -4)

        # Financials + Weak Sector: 67.7% win rate, +1.10% return
        if 'financial' in sector_lower and sector_momentum < -3.0 and boosts.get('financials_weak_sector', {}).get('enabled'):
            applied_boosts['financials_weak'] = boosts['financials_weak_sector']['boost']
            # Remove general weak_sector boost - this is stronger
            if 'weak_sector' in applied_boosts:
                del applied_boosts['weak_sector']

        # Consumer sector penalty: 49% win rate, -0.21% avg return
        if 'consumer' in sector_lower and boosts.get('consumer_sector_penalty', {}).get('enabled'):
            applied_boosts['consumer_trap'] = boosts['consumer_sector_penalty'].get('penalty', -6)

        # Consumer + Neutral Momentum TRAP (Jan 5, 2026)
        # Research: Consumer + neutral momentum = 33.3% win rate!
        if 'consumer' in sector_lower and -1.0 <= sector_momentum <= 1.0 and boosts.get('consumer_neutral_trap', {}).get('enabled'):
            applied_boosts['consumer_neutral_trap'] = boosts['consumer_neutral_trap'].get('penalty', -6)

        # Consumer + Slightly Strong Momentum TRAP (Jan 5, 2026) - Improvement #76
        # Research: Consumer + momentum 1-3% = 45.2% win, -0.35% return
        if 'consumer' in sector_lower and 1.0 <= sector_momentum <= 3.0 and boosts.get('consumer_slightly_strong_trap', {}).get('enabled'):
            applied_boosts['consumer_slight_strong_trap'] = boosts['consumer_slightly_strong_trap'].get('penalty', -5)

        # Consumer + Strong Momentum TRAP (Jan 5, 2026) - Improvement #77
        # Research: Consumer + momentum > 3% = 52.6% win but -0.40% return (negative despite decent win rate!)
        if 'consumer' in sector_lower and sector_momentum > 3.0 and boosts.get('consumer_strong_trap', {}).get('enabled'):
            applied_boosts['consumer_vstrong_trap'] = boosts['consumer_strong_trap'].get('penalty', -4)

        # Healthcare sector boost (Jan 5, 2026)
        # Research: Healthcare has 59.4% win rate (highest sector!)
        if 'health' in sector_lower and boosts.get('healthcare_sector_boost', {}).get('enabled'):
            applied_boosts['healthcare'] = boosts['healthcare_sector_boost']['boost']

        # Financials sector boost (Jan 5, 2026) - Improvement #75
        # Research: Financials = 59.3% win rate (2nd best sector after Healthcare)
        if 'financial' in sector_lower and boosts.get('financials_sector_boost', {}).get('enabled'):
            applied_boosts['financials'] = boosts['financials_sector_boost']['boost']

        # Healthcare + Neutral Momentum (Jan 5, 2026)
        # Research: Healthcare + neutral momentum = 61.3% win, +0.86% return
        if 'health' in sector_lower and -1.0 <= sector_momentum <= 1.0 and boosts.get('healthcare_neutral_momentum', {}).get('enabled'):
            applied_boosts['healthcare_neutral'] = boosts['healthcare_neutral_momentum']['boost']

        # Healthcare + Slightly Strong (Jan 5, 2026)
        # Research: Healthcare + momentum 1-3% = 61.3% win, +0.77% return
        if 'health' in sector_lower and 1.0 <= sector_momentum <= 3.0 and boosts.get('healthcare_slightly_strong', {}).get('enabled'):
            applied_boosts['healthcare_strong'] = boosts['healthcare_slightly_strong']['boost']

        # Energy + Neutral Momentum (Jan 5, 2026)
        # Research: Energy + neutral momentum = 60.3% win, +0.76% return
        if 'energy' in sector_lower and -1.0 <= sector_momentum <= 1.0 and boosts.get('energy_neutral_momentum', {}).get('enabled'):
            applied_boosts['energy_neutral'] = boosts['energy_neutral_momentum']['boost']

        # Financials + Strong Momentum (Jan 5, 2026)
        # Research: Financials + momentum > 3% = 61.9% win, +0.96% return
        if 'financial' in sector_lower and sector_momentum > 3.0 and boosts.get('financials_strong_momentum', {}).get('enabled'):
            applied_boosts['financials_strong'] = boosts['financials_strong_momentum']['boost']

        # Communication + Weak Sector (Jan 5, 2026)
        # Research: Communication + weak momentum = 58.6% win, +1.24% return
        if 'communication' in sector_lower and sector_momentum < -3.0 and boosts.get('communication_weak_sector', {}).get('enabled'):
            applied_boosts['comm_weak'] = boosts['communication_weak_sector']['boost']

        # Communication + Neutral (Jan 5, 2026) - Improvement #107
        # Research: Communication + neutral momentum = ~55.6% win, slight edge
        if 'communication' in sector_lower and -1.0 <= sector_momentum <= 1.0 and boosts.get('communication_neutral', {}).get('enabled'):
            applied_boosts['comm_neutral'] = boosts['communication_neutral']['boost']

        # Communication + Slightly Weak (Jan 5, 2026) - Improvement #108
        # Research: Communication + slightly weak (-3 to -1) = ~57.1% win, good contrarian
        if 'communication' in sector_lower and -3.0 <= sector_momentum < -1.0 and boosts.get('communication_slightly_weak', {}).get('enabled'):
            applied_boosts['comm_sw'] = boosts['communication_slightly_weak']['boost']

        # Industrials + Neutral TRAP (Jan 5, 2026) - Improvement #109
        # Research: Industrials + neutral momentum = ~50% win, consolidation trap
        if 'industrial' in sector_lower and -1.0 <= sector_momentum <= 1.0 and boosts.get('industrials_neutral_trap', {}).get('enabled'):
            applied_boosts['industrials_neutral_trap'] = boosts['industrials_neutral_trap'].get('penalty', -3)

        # Industrials + Strong TRAP (Jan 5, 2026) - Improvement #110
        # Research: Industrials + strong momentum (>3%) = ~46.9% win, chasing trap
        if 'industrial' in sector_lower and sector_momentum > 3.0 and boosts.get('industrials_strong_trap', {}).get('enabled'):
            applied_boosts['industrials_vstrong_trap'] = boosts['industrials_strong_trap'].get('penalty', -5)

        # Technology sector penalty (Jan 5, 2026)
        # Research: Technology has 53.4% win rate (below 54.7% baseline)
        if 'tech' in sector_lower and boosts.get('technology_sector_penalty', {}).get('enabled'):
            applied_boosts['tech_penalty'] = boosts['technology_sector_penalty'].get('penalty', -3)

        # Technology + weak sector override (Jan 5, 2026)
        # Research: Tech + weak momentum = +0.88% return (profitable despite lower win rate)
        if 'tech' in sector_lower and sector_momentum < -3.0 and boosts.get('technology_weak_override', {}).get('enabled'):
            applied_boosts['tech_weak_override'] = boosts['technology_weak_override']['boost']
            # Remove base tech penalty since this is a good pattern
            if 'tech_penalty' in applied_boosts:
                del applied_boosts['tech_penalty']

        # Technology + Neutral TRAP (Jan 5, 2026) - Improvement #111
        # Research: Tech + neutral momentum = ~52.2% win, slight underperformance
        if 'tech' in sector_lower and -1.0 <= sector_momentum <= 1.0 and boosts.get('technology_neutral_trap', {}).get('enabled'):
            applied_boosts['tech_neutral_trap'] = boosts['technology_neutral_trap'].get('penalty', -2)

        # Technology + Slightly Strong TRAP (Jan 5, 2026) - Improvement #112
        # Research: Tech + slightly strong (1-3%) = ~51% win, chasing trap
        if 'tech' in sector_lower and 1.0 <= sector_momentum <= 3.0 and boosts.get('technology_slightly_strong_trap', {}).get('enabled'):
            applied_boosts['tech_ss_trap'] = boosts['technology_slightly_strong_trap'].get('penalty', -3)

        # Industrials + Slightly Strong TRAP (Jan 5, 2026) - WORST TRAP!
        # Research: Industrials + momentum 1-3% = 30% win, -0.92% return!
        if 'industrial' in sector_lower and 1.0 <= sector_momentum <= 3.0 and boosts.get('industrials_slightly_strong_trap', {}).get('enabled'):
            applied_boosts['industrials_strong_trap'] = boosts['industrials_slightly_strong_trap'].get('penalty', -10)
            # Remove the weak sector boost if present (this overrides)
            if 'industrials_weak' in applied_boosts:
                del applied_boosts['industrials_weak']

        # Communication + Strong Momentum TRAP (Jan 5, 2026)
        # Research: Communication + momentum > 3% = 27.3% win rate!
        if 'communication' in sector_lower and sector_momentum > 3.0 and boosts.get('communication_strong_trap', {}).get('enabled'):
            applied_boosts['comm_strong_trap'] = boosts['communication_strong_trap'].get('penalty', -8)

        # Financials + Neutral Momentum TRAP (Jan 5, 2026)
        # Research: Financials + neutral (-1% to 1%) = 40.5% win, -0.49% return
        if 'financial' in sector_lower and -1.0 <= sector_momentum <= 1.0 and boosts.get('financials_neutral_trap', {}).get('enabled'):
            applied_boosts['financials_neutral_trap'] = boosts['financials_neutral_trap'].get('penalty', -5)
            # Remove financials_weak if somehow both would apply
            if 'financials_weak' in applied_boosts:
                del applied_boosts['financials_weak']

        # Materials + Slightly Strong TRAP (Jan 5, 2026)
        # Research: Materials + momentum 1-3% = 38.2% win, -0.55% return
        if 'material' in sector_lower and 1.0 <= sector_momentum <= 3.0 and boosts.get('materials_slightly_strong_trap', {}).get('enabled'):
            applied_boosts['materials_ss_trap'] = boosts['materials_slightly_strong_trap'].get('penalty', -6)

        # Materials + Weak Sector (Jan 5, 2026) - Improvement #104
        # Research: Materials + weak momentum (<-3%) = contrarian play, ~59.5% win
        if 'material' in sector_lower and sector_momentum < -3.0 and boosts.get('materials_weak_sector', {}).get('enabled'):
            applied_boosts['materials_weak'] = boosts['materials_weak_sector']['boost']

        # Materials + Strong Momentum TRAP (Jan 5, 2026) - Improvement #105
        # Research: Materials + strong momentum (>3%) = slight trap, ~52.6% win
        if 'material' in sector_lower and sector_momentum > 3.0 and boosts.get('materials_strong_trap', {}).get('enabled'):
            applied_boosts['materials_strong_trap'] = boosts['materials_strong_trap'].get('penalty', -3)

        # Materials + Neutral Momentum TRAP (Jan 5, 2026) - Improvement #106
        # Research: Materials + neutral momentum (-1 to 1%) = consolidation trap, ~47.3% win
        if 'material' in sector_lower and -1.0 <= sector_momentum <= 1.0 and boosts.get('materials_neutral_trap', {}).get('enabled'):
            applied_boosts['materials_neutral_trap'] = boosts['materials_neutral_trap'].get('penalty', -4)

        # Close position in day's range (Jan 4, 2026)
        # Research: close near LOW = 57.4% win, close near HIGH = 50.2% win
        if close_position < 0.3 and boosts.get('close_near_low', {}).get('enabled'):
            applied_boosts['close_near_low'] = boosts['close_near_low']['boost']
        elif close_position > 0.7 and boosts.get('close_near_high', {}).get('enabled'):
            # This is a penalty (negative boost)
            applied_boosts['close_near_high'] = boosts['close_near_high'].get('penalty', -5)

        # Small gap down TRAP (Jan 5, 2026) - Improvement #85
        # Research: -1% to -2% gap = 48% 2d win rate, -0.16% return (248 signals)
        if -2.0 <= gap_pct < -1.0 and boosts.get('small_gap_down_trap', {}).get('enabled'):
            applied_boosts['small_gap_trap'] = boosts['small_gap_down_trap'].get('penalty', -3)

        # Medium gap down (Jan 5, 2026)
        # Research: -2% to -3% gap = 58.6% win rate (BEST), large gaps underperform
        if -3.0 <= gap_pct < -2.0 and boosts.get('medium_gap_down', {}).get('enabled'):
            applied_boosts['medium_gap_down'] = boosts['medium_gap_down']['boost']

        # Near SMA200 (Jan 5, 2026)
        # Research: within 5% of SMA200 = 57.6% win (support/resistance effect)
        if abs(sma200_distance) <= 5.0 and boosts.get('near_sma200', {}).get('enabled'):
            applied_boosts['near_sma200'] = boosts['near_sma200']['boost']

        # Below SMA200 (Jan 5, 2026) - Improvement #90
        # Research: >5% below SMA200 = 54.3% win, +0.23% return (219 signals)
        if sma200_distance < -5.0 and boosts.get('below_sma200', {}).get('enabled'):
            applied_boosts['below_sma200'] = boosts['below_sma200']['boost']

        # Very Wide Day Range (Jan 5, 2026)
        # Research: >5% day range = 60.1% win, +1.27% return (volatility = opportunity)
        if day_range_pct > 5.0 and boosts.get('very_wide_range', {}).get('enabled'):
            applied_boosts['wide_range'] = boosts['very_wide_range']['boost']

        # Low Volume + Narrow Range TRAP (Jan 5, 2026)
        # Research: vol<0.8 AND range<1.5% = ONLY 41.1% win, -0.39% return
        # This is a TRAP pattern - heavily penalize
        if volume_ratio < 0.8 and day_range_pct < 1.5 and boosts.get('low_vol_narrow_range', {}).get('enabled'):
            applied_boosts['low_vol_trap'] = boosts['low_vol_narrow_range'].get('penalty', -8)

        # Coiled Spring (Jan 5, 2026) - COUNTERINTUITIVE!
        # Research: HIGH vol (1.5-3x) + narrow range (<1.5%) = 72.7% win, +1.03% return
        # This is OPPOSITE of low_vol_trap - high volume compression before breakout
        if 1.5 <= volume_ratio < 3.0 and day_range_pct < 1.5 and boosts.get('coiled_spring', {}).get('enabled'):
            applied_boosts['coiled_spring'] = boosts['coiled_spring']['boost']

        # High Volume + Normal Range (Jan 5, 2026)
        # Research: vol 1.5-3x + range 1.5-3% = 67.6% win, +0.83% return
        if 1.5 <= volume_ratio < 3.0 and 1.5 <= day_range_pct < 3.0 and boosts.get('high_vol_normal_range', {}).get('enabled'):
            applied_boosts['high_vol_normal'] = boosts['high_vol_normal_range']['boost']

        # High Vol + Wide Range TRAP (Jan 5, 2026) - Improvement #78
        # Research: vol 1.5-3x + range 3-5% = 58.9% win but -0.025% return (FALSE BREAKOUT)
        if 1.5 <= volume_ratio < 3.0 and 3.0 <= day_range_pct < 5.0 and boosts.get('high_vol_wide_range_trap', {}).get('enabled'):
            applied_boosts['high_vol_wide_trap'] = boosts['high_vol_wide_range_trap'].get('penalty', -3)

        # Normal Vol + Narrow Range Penalty (Jan 5, 2026) - Improvement #79
        # Research: vol 0.8-1.5x + range <1.5% = 52.7% win, -0.08% return (consolidation trap)
        if 0.8 <= volume_ratio < 1.5 and day_range_pct < 1.5 and boosts.get('normal_vol_narrow_range_penalty', {}).get('enabled'):
            applied_boosts['normal_narrow_trap'] = boosts['normal_vol_narrow_range_penalty'].get('penalty', -2)

        # Low Volume Penalty (Jan 5, 2026)
        # Research: vol < 0.8x = 48.5% win, -0.05% return (653 signals)
        if volume_ratio < 0.8 and boosts.get('low_volume_penalty', {}).get('enabled'):
            applied_boosts['low_vol_penalty'] = boosts['low_volume_penalty'].get('penalty', -4)

        # High Volume Boost (Jan 5, 2026) - HIGH FREQUENCY!
        # Research: vol 1.5-3x = 60.7% win rate (364 signals)
        if 1.5 <= volume_ratio < 3.0 and boosts.get('high_volume_boost', {}).get('enabled'):
            applied_boosts['high_vol'] = boosts['high_volume_boost']['boost']

        # Very High Volume (Jan 5, 2026)
        # Research: vol >= 3x = ~75% win rate regardless of range
        # This stacks with volume_exhaustion if it's also a down day
        if volume_ratio >= 3.0 and boosts.get('very_high_volume', {}).get('enabled'):
            applied_boosts['very_high_vol'] = boosts['very_high_volume']['boost']

        # Very High Volume + Normal Range (Jan 5, 2026) - HIGHEST WIN RATE!
        # Research: vol >= 3x + range 1.5-3% = 85.7% win rate, +2.1% return
        if volume_ratio >= 3.0 and 1.5 <= day_range_pct < 3.0 and boosts.get('very_high_vol_normal_range', {}).get('enabled'):
            applied_boosts['vh_vol_normal_range'] = boosts['very_high_vol_normal_range']['boost']

        # Very High Volume + Wide Range (Jan 5, 2026)
        # Research: vol >= 3x + range 3-5% = 81.8% win rate
        if volume_ratio >= 3.0 and 3.0 <= day_range_pct < 5.0 and boosts.get('very_high_vol_wide_range', {}).get('enabled'):
            applied_boosts['vh_vol_wide_range'] = boosts['very_high_vol_wide_range']['boost']

        # Very High Volume + Very Wide Range (Jan 5, 2026)
        # Research: vol >= 3x + range >= 5% = 66% win, +1.76% return
        if volume_ratio >= 3.0 and day_range_pct >= 5.0 and boosts.get('very_high_vol_very_wide_range', {}).get('enabled'):
            applied_boosts['vh_vol_vw_range'] = boosts['very_high_vol_very_wide_range']['boost']

        # Normal Volume + Very Wide Range (Jan 5, 2026) - HIGH FREQUENCY!
        # Research: vol 0.8-1.5 + range >= 5% = 62.3% win, +1.23% return (199 signals)
        if 0.8 <= volume_ratio < 1.5 and day_range_pct >= 5.0 and boosts.get('normal_vol_very_wide_range', {}).get('enabled'):
            applied_boosts['norm_vol_vw_range'] = boosts['normal_vol_very_wide_range']['boost']

        # Low Volume + Wide Range TRAP (Jan 5, 2026)
        # Research: vol < 0.8 + range > 5% = 50% win, -0.68% return (false breakout)
        if volume_ratio < 0.8 and day_range_pct > 5.0 and boosts.get('low_vol_wide_range_trap', {}).get('enabled'):
            applied_boosts['low_vol_wide_trap'] = boosts['low_vol_wide_range_trap'].get('penalty', -5)

        # Minimal Drawdown Penalty (Jan 5, 2026) - Improvement #91
        # Research: <3% drawdown = 51.3% win, +0.10% return (near baseline, not enough stretch)
        if drawdown_pct > -3.0 and boosts.get('minimal_drawdown_penalty', {}).get('enabled'):
            applied_boosts['minimal_dd_penalty'] = boosts['minimal_drawdown_penalty'].get('penalty', -2)

        # Severe Drawdown (Jan 5, 2026)
        # Research: >10% drawdown from 20d high = 61.2% win, +1.55% return
        if drawdown_pct < -10.0 and boosts.get('severe_drawdown', {}).get('enabled'):
            applied_boosts['severe_drawdown'] = boosts['severe_drawdown']['boost']

        # Down Streak WITH Drawdown (Jan 5, 2026) - VERY HIGH CONVICTION
        # Research: 5+ down days WITH moderate/severe drawdown = 76.9% win!
        # This replaces the simple consecutive_down_6plus when drawdown is present
        if consecutive_down_days >= 5 and drawdown_pct < -5.0 and boosts.get('down_streak_with_drawdown', {}).get('enabled'):
            applied_boosts['down_streak_dd'] = boosts['down_streak_with_drawdown']['boost']
            # Remove the base consecutive boost if present (this is stronger)
            if 'down_streak_6+' in applied_boosts:
                del applied_boosts['down_streak_6+']

        # 3-4 Down Days + Moderate Drawdown (Jan 5, 2026) - HIGH CONVICTION
        # Research: 3-4 down days + 5-10% drawdown = 70.8% win, +0.60% return
        if 3 <= consecutive_down_days <= 4 and -10.0 <= drawdown_pct < -5.0 and boosts.get('three_four_down_moderate_dd', {}).get('enabled'):
            applied_boosts['3-4_down_mod_dd'] = boosts['three_four_down_moderate_dd']['boost']

        # 1 Down Day + Moderate Drawdown TRAP (Jan 5, 2026)
        # Research: 1 down day + moderate DD = 49.5% win, -0.16% return (too early!)
        if consecutive_down_days == 1 and -10.0 <= drawdown_pct < -5.0 and boosts.get('one_down_moderate_dd_trap', {}).get('enabled'):
            applied_boosts['1_down_mod_trap'] = boosts['one_down_moderate_dd_trap'].get('penalty', -4)

        # 5+ Down Days + Mild Drawdown TRAP (Jan 5, 2026)
        # Research: 5+ down + mild DD (3-5%) = 40.4% win, -0.59% return (false exhaustion)
        if consecutive_down_days >= 5 and -5.0 <= drawdown_pct < -3.0 and boosts.get('five_plus_mild_dd_trap', {}).get('enabled'):
            applied_boosts['5+_down_mild_trap'] = boosts['five_plus_mild_dd_trap'].get('penalty', -6)
            # Remove base boost if present
            if 'down_streak_6+' in applied_boosts:
                del applied_boosts['down_streak_6+']

        # Down Streak TRAP (Jan 5, 2026) - MAJOR TRAP!
        # Research: 5+ down days with minimal drawdown (<3%) = ONLY 36.6% win, -1.09% return
        # This cancels the consecutive down boost AND adds penalty
        if consecutive_down_days >= 5 and drawdown_pct > -3.0 and boosts.get('down_streak_trap', {}).get('enabled'):
            applied_boosts['down_streak_trap'] = boosts['down_streak_trap'].get('penalty', -10)
            # Remove the base consecutive boost - it's a trap!
            if 'down_streak_6+' in applied_boosts:
                del applied_boosts['down_streak_6+']

        # Two Down + Severe Drawdown (Jan 5, 2026) - HIGHEST CONVICTION!
        # Research: 2 consecutive down days + drawdown > 10% = 73.2% win, +4.47% return
        # This is the best pattern discovered - massive return potential
        if consecutive_down_days == 2 and drawdown_pct < -10.0 and boosts.get('two_down_severe_drawdown', {}).get('enabled'):
            applied_boosts['2down_severe'] = boosts['two_down_severe_drawdown']['boost']

        # Two Down + Moderate Drawdown (Jan 5, 2026) - HIGH FREQUENCY!
        # Research: 2 down days + 5-10% drawdown = 58.2% win, +1.50% return (79 trades)
        if consecutive_down_days == 2 and -10.0 <= drawdown_pct < -5.0 and boosts.get('two_down_moderate_drawdown', {}).get('enabled'):
            applied_boosts['2down_moderate'] = boosts['two_down_moderate_drawdown']['boost']

        # Two Down + Mild Drawdown (Jan 5, 2026)
        # Research: 2 down days + 3-5% drawdown = 61.9% win, +0.55% return (84 signals)
        if consecutive_down_days == 2 and -5.0 <= drawdown_pct < -3.0 and boosts.get('two_down_mild_drawdown', {}).get('enabled'):
            applied_boosts['2down_mild'] = boosts['two_down_mild_drawdown']['boost']

        # Bounce Day + Mild Drawdown (Jan 5, 2026) - Improvement #92
        # Research: 0 down days + mild DD (3-5%) = 56.7% win, +0.19% return (157 signals)
        if consecutive_down_days == 0 and -5.0 <= drawdown_pct < -3.0 and boosts.get('bounce_day_mild_drawdown', {}).get('enabled'):
            applied_boosts['bounce_mild'] = boosts['bounce_day_mild_drawdown']['boost']

        # Bounce Day + Moderate Drawdown (Jan 5, 2026) - Improvement #96
        # Research: 0 down days + moderate DD (5-10%) = ~62.5% win, ~0.95% return
        if consecutive_down_days == 0 and -10.0 <= drawdown_pct < -5.0 and boosts.get('bounce_day_moderate_drawdown', {}).get('enabled'):
            applied_boosts['bounce_moderate'] = boosts['bounce_day_moderate_drawdown']['boost']

        # One Down + Severe Drawdown TRAP (Jan 5, 2026) - Improvement #97
        # Research: 1 down day + severe DD (>10%) = crash day, often continues falling
        if consecutive_down_days == 1 and drawdown_pct < -10.0 and boosts.get('one_down_severe_drawdown', {}).get('enabled'):
            applied_boosts['1down_severe_trap'] = boosts['one_down_severe_drawdown'].get('penalty', -5)

        # One Down + Mild Drawdown (Jan 5, 2026) - Improvement #98
        # Research: 1 down day + mild DD (3-5%) = slight edge, minor pullback
        if consecutive_down_days == 1 and -5.0 <= drawdown_pct < -3.0 and boosts.get('one_down_mild_drawdown', {}).get('enabled'):
            applied_boosts['1down_mild'] = boosts['one_down_mild_drawdown']['boost']

        # 3-4 Down + Mild Drawdown (Jan 5, 2026) - Improvement #93
        # Research: 3-4 down days + mild DD (3-5%) = 57.1% win, +0.20% return (91 signals)
        if 3 <= consecutive_down_days <= 4 and -5.0 <= drawdown_pct < -3.0 and boosts.get('three_four_mild_drawdown', {}).get('enabled'):
            applied_boosts['3-4_mild_dd'] = boosts['three_four_mild_drawdown']['boost']

        # 5+ Down + Severe Drawdown TRAP (Jan 5, 2026)
        # Research: 5+ down + >10% DD = 40% win, -0.35% return (capitulation often continues)
        if consecutive_down_days >= 5 and drawdown_pct < -10.0 and boosts.get('five_plus_severe_dd_trap', {}).get('enabled'):
            applied_boosts['5+_severe_trap'] = boosts['five_plus_severe_dd_trap'].get('penalty', -8)
            # Remove the down_streak_with_drawdown boost if present
            if 'down_streak_dd' in applied_boosts:
                del applied_boosts['down_streak_dd']

        # 3-4 Down + Severe Drawdown TRAP (Jan 5, 2026)
        # Research: 3-4 down + >10% drawdown = 57.7% win BUT -0.14% return!
        # Counterintuitive: looks good but actually loses money
        if 3 <= consecutive_down_days <= 4 and drawdown_pct < -10.0 and boosts.get('three_four_severe_drawdown_trap', {}).get('enabled'):
            applied_boosts['3-4_severe_trap'] = boosts['three_four_severe_drawdown_trap'].get('penalty', -4)

        # Bounce Day + Severe Drawdown (Jan 5, 2026)
        # Research: 0 consecutive down days + drawdown > 10% = 69.2% win, +1.88% return
        # First day of potential bounce after severe drawdown
        if consecutive_down_days == 0 and drawdown_pct < -10.0 and boosts.get('bounce_day_severe_drawdown', {}).get('enabled'):
            applied_boosts['bounce_severe'] = boosts['bounce_day_severe_drawdown']['boost']

        # Strong Tier Outperformance (Jan 5, 2026)
        # Research: STRONG tier has 68.5% win rate vs ELITE's 60%!
        if tier == 'strong' and boosts.get('strong_tier_outperformance', {}).get('enabled'):
            applied_boosts['strong_tier'] = boosts['strong_tier_outperformance']['boost']

        # AVOID Tier Penalty (Jan 5, 2026) - Improvement #94
        # Research: AVOID tier has 45.6% win rate - below 50%! Strongly discourage trading
        if tier == 'avoid' and boosts.get('avoid_tier_penalty', {}).get('enabled'):
            applied_boosts['avoid_tier'] = boosts['avoid_tier_penalty'].get('penalty', -8)

        # ATR-Based Volatility Boosts (Jan 5, 2026)
        # Research: High ATR stocks outperform in mean reversion
        if atr_pct < 2.05 and boosts.get('low_atr_penalty', {}).get('enabled'):
            applied_boosts['low_atr_trap'] = boosts['low_atr_penalty'].get('penalty', -5)
        elif 2.05 <= atr_pct < 2.5 and boosts.get('med_low_atr_penalty', {}).get('enabled'):
            # Improvement #95: Med-low ATR penalty (Jan 5, 2026)
            # Research: ATR 2.05-2.50% = 54.1% win, +0.12% return (below baseline)
            applied_boosts['med_low_atr'] = boosts['med_low_atr_penalty'].get('penalty', -2)
        elif atr_pct > 3.34 and boosts.get('high_atr_boost', {}).get('enabled'):
            applied_boosts['high_atr'] = boosts['high_atr_boost']['boost']
        elif 2.5 <= atr_pct <= 3.34 and boosts.get('med_high_atr_boost', {}).get('enabled'):
            # Improvement #84: Med-high ATR boost (Jan 5, 2026)
            # Research: ATR 2.5-3.34% = 57.3% win, +0.25% return (639 signals - HIGH FREQUENCY)
            applied_boosts['med_high_atr'] = boosts['med_high_atr_boost']['boost']

        # Regime-Specific Boosts (Jan 5, 2026)
        # VOLATILE and BEAR regimes have dramatically better performance
        regime_lower = regime.lower()
        if regime_lower == 'volatile' and boosts.get('volatile_regime_boost', {}).get('enabled'):
            applied_boosts['volatile_regime'] = boosts['volatile_regime_boost']['boost']
        elif regime_lower == 'bear' and boosts.get('bear_regime_boost', {}).get('enabled'):
            applied_boosts['bear_regime'] = boosts['bear_regime_boost']['boost']
        elif regime_lower == 'bull' and boosts.get('bull_regime_penalty', {}).get('enabled'):
            # BULL regime underperforms - mean reversion less effective in uptrends
            applied_boosts['bull_regime'] = boosts['bull_regime_penalty'].get('penalty', -3)
        elif regime_lower == 'choppy' and boosts.get('choppy_regime_boost', {}).get('enabled'):
            # CHOPPY is the bread-and-butter regime for mean reversion
            applied_boosts['choppy_regime'] = boosts['choppy_regime_boost']['boost']

        # Regime Champions (Jan 5, 2026) - Stock-specific regime outperformance
        # Research: Certain stocks have 100% or near-100% win rates in specific regimes
        regime_champs = boosts.get('regime_champions', {})
        if regime_champs.get('enabled'):
            regime_data = regime_champs.get(regime_lower, {})
            if ticker in regime_data.get('tickers', []):
                boost_val = regime_data.get('boost', 5)
                applied_boosts['regime_champion'] = boost_val

        # Weak Tier Volatile Override (Jan 5, 2026)
        # Research: NVDA, AVGO, MRVL outperform during volatile despite weak tier
        weak_override = boosts.get('weak_tier_volatile_override', {})
        if weak_override.get('enabled') and regime_lower == 'volatile':
            if ticker in weak_override.get('tickers', []):
                applied_boosts['weak_volatile_override'] = weak_override.get('boost', 6)

        # Large Gap-Down + Oversold Combo (Jan 5, 2026) - BEST REVERSION
        # Research: Large gap-downs (-3%+) with RSI<40 = best mean reversion potential
        if gap_pct <= -3.0 and rsi_level < 40 and boosts.get('large_gap_oversold_combo', {}).get('enabled'):
            applied_boosts['large_gap_oversold'] = boosts['large_gap_oversold_combo']['boost']

        # Large Gap WITHOUT Oversold TRAP (Jan 5, 2026) - Improvement #86
        # Research: Large gap (-3%+) with RSI >= 40 = 42.7% win, -0.04% return
        # Counterintuitive: large gaps without oversold often continue falling
        if gap_pct <= -3.0 and rsi_level >= 40 and boosts.get('large_gap_not_oversold_trap', {}).get('enabled'):
            applied_boosts['large_gap_no_os'] = boosts['large_gap_not_oversold_trap'].get('penalty', -5)

        # Signal Strength Sweet Spot (Jan 5, 2026)
        # Research: 70-80 range = best Kelly (17.67%), best return (+0.40%)
        # 50-60 and 80+ both underperform the 70-80 sweet spot
        if 70 <= base_signal < 80 and boosts.get('signal_strength_sweet_spot', {}).get('enabled'):
            applied_boosts['strength_sweet_spot'] = boosts['signal_strength_sweet_spot']['boost']
        elif base_signal >= 80 and boosts.get('signal_strength_over_80_penalty', {}).get('enabled'):
            # Counterintuitively, 80+ signals have LOWER Kelly than 70-80
            applied_boosts['strength_over_80'] = boosts['signal_strength_over_80_penalty'].get('penalty', -2)

        # Weak Base Signal Penalty (Jan 5, 2026) - Improvement #113
        # CRITICAL FIX: Prevents weak base signals from being boosted above threshold
        # Validation backtest showed modifiers were boosting marginal signals, reducing quality
        if base_signal < 55 and boosts.get('weak_base_signal_penalty', {}).get('enabled'):
            applied_boosts['weak_base'] = boosts['weak_base_signal_penalty'].get('penalty', -8)

        # Marginal Base Signal Penalty (Jan 5, 2026) - Improvement #114
        # Slight penalty for base signals 55-60 to maintain quality threshold
        elif 55 <= base_signal < 60 and boosts.get('marginal_base_signal_penalty', {}).get('enabled'):
            applied_boosts['marginal_base'] = boosts['marginal_base_signal_penalty'].get('penalty', -4)

        # High Conviction Stack (Jan 5, 2026)
        # When 3+ positive boosts apply, add extra confidence boost
        # Exclude near_sma200 (too common) and penalties from count
        positive_boosts = [k for k, v in applied_boosts.items()
                         if v > 0 and k not in ['near_sma200', 'strong_tier']]
        min_boosts = boosts.get('high_conviction_stack', {}).get('min_boosts', 3)
        if len(positive_boosts) >= min_boosts and boosts.get('high_conviction_stack', {}).get('enabled'):
            applied_boosts['high_conviction'] = boosts['high_conviction_stack']['boost']

        # Calculate final
        total_boost = sum(applied_boosts.values())
        final_signal = min(100.0, max(0.0, adjusted_base + total_boost))

        return SignalBreakdown(
            ticker=ticker,
            base_signal=base_signal,
            tier=tier,
            tier_multiplier=tier_mult,
            regime=regime,
            regime_multiplier=regime_mult,
            boosts_applied=applied_boosts,
            final_signal=round(final_signal, 1)
        )

    def format_breakdown(self, breakdown: SignalBreakdown) -> str:
        """Format breakdown for display."""
        lines = [
            f"Signal: {breakdown.ticker}",
            f"  Base: {breakdown.base_signal:.1f}",
            f"  x Tier ({breakdown.tier}): {breakdown.tier_multiplier:.2f}",
            f"  x Regime ({breakdown.regime}): {breakdown.regime_multiplier:.2f}",
            f"  = Adjusted: {breakdown.base_signal * breakdown.tier_multiplier * breakdown.regime_multiplier:.1f}"
        ]

        if breakdown.boosts_applied:
            for name, boost in breakdown.boosts_applied.items():
                lines.append(f"  + {name}: +{boost}")

        lines.append(f"  = FINAL: {breakdown.final_signal:.1f}")
        return "\n".join(lines)


# Singleton
_calculator = None


def get_calculator() -> UnifiedSignalCalculator:
    """Get singleton calculator."""
    global _calculator
    if _calculator is None:
        _calculator = UnifiedSignalCalculator()
    return _calculator


def demo():
    """Demonstrate calculator."""
    calc = UnifiedSignalCalculator()

    print("=" * 60)
    print("UNIFIED SIGNAL CALCULATOR")
    print("=" * 60)
    print(f"Config: {calc.CONFIG_PATH}")
    print()

    # Test cases
    tests = [
        {
            'name': 'OPTIMAL: Elite stock, volatile, all boosts',
            'params': {
                'ticker': 'COP',
                'base_signal': 65.0,
                'regime': 'volatile',
                'is_monday': True,
                'consecutive_down_days': 6,
                'has_rsi_divergence': True,
                'volume_ratio': 4.0,
                'is_down_day': True,
                'sector_momentum': -5.0
            }
        },
        {
            'name': 'AVERAGE: Normal conditions',
            'params': {
                'ticker': 'MSFT',
                'base_signal': 70.0,
                'regime': 'choppy',
                'is_monday': False,
                'consecutive_down_days': 2,
                'has_rsi_divergence': False,
                'volume_ratio': 1.2,
                'is_down_day': False,
                'sector_momentum': 0.5
            }
        },
        {
            'name': 'POOR: Avoid stock, bull regime',
            'params': {
                'ticker': 'NOW',
                'base_signal': 60.0,
                'regime': 'bull',
                'is_monday': False,
                'consecutive_down_days': 1,
                'has_rsi_divergence': False,
                'volume_ratio': 0.8,
                'is_down_day': False,
                'sector_momentum': 3.0
            }
        }
    ]

    for test in tests:
        print(f"\n{test['name']}")
        print("-" * 50)
        result = calc.calculate(**test['params'])
        print(calc.format_breakdown(result))


if __name__ == '__main__':
    demo()

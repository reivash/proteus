"""
Trading Constants for Proteus

Centralizes all trading-related magic numbers with documentation
explaining the research basis for each value.

Usage:
    from src.config.trading_constants import (
        SIGNAL_THRESHOLDS,
        POSITION_SIZING,
        REGIME_SETTINGS,
        BEAR_DETECTION
    )

All values are based on backtesting research conducted 2024-2026.
See data/research/ for detailed analysis results.
"""

from dataclasses import dataclass
from typing import Dict


# =============================================================================
# SIGNAL THRESHOLDS
# =============================================================================
# Research basis: Backtest analysis of 1-year data (2024-2025)
# Higher thresholds = fewer signals but higher win rate

@dataclass(frozen=True)
class SignalThresholds:
    """Signal strength thresholds by regime."""

    # Minimum signal strength to consider a trade
    BASE_MINIMUM: float = 50.0

    # Regime-specific thresholds (optimized via backtest)
    BULL: float = 50.0      # Good conditions, standard threshold
    BEAR: float = 70.0      # Best conditions! Higher bar = 100% win rate
    CHOPPY: float = 65.0    # Poor conditions, high bar to filter noise
    VOLATILE: float = 70.0  # Uncertain, require high conviction

    # Signal quality tiers
    TIER_EXCELLENT: float = 85.0   # Top tier signals
    TIER_GOOD: float = 70.0        # Good quality signals
    TIER_ACCEPTABLE: float = 55.0  # Minimum for trading
    TIER_SKIP: float = 50.0        # Below this = skip


SIGNAL_THRESHOLDS = SignalThresholds()


# =============================================================================
# POSITION SIZING
# =============================================================================
# Research basis: Kelly Criterion analysis + practical constraints

@dataclass(frozen=True)
class PositionSizing:
    """Position sizing parameters."""

    # Portfolio constraints
    MAX_POSITIONS: int = 6           # Max concurrent positions
    MAX_PORTFOLIO_HEAT: float = 0.50  # Max 50% of portfolio at risk
    MAX_SINGLE_POSITION: float = 0.15 # Max 15% in single position
    MIN_POSITION_SIZE: float = 0.02   # Min 2% position (worth the fees)

    # Tier-based sizing (from Kelly analysis)
    # Format: (kelly_fraction, half_kelly, win_rate, recommended_size)
    ELITE_KELLY: float = 0.24
    ELITE_HALF_KELLY: float = 0.12
    ELITE_WIN_RATE: float = 0.60
    ELITE_SIZE: float = 0.12  # 12% max for elite stocks

    STRONG_KELLY: float = 0.37
    STRONG_HALF_KELLY: float = 0.18
    STRONG_WIN_RATE: float = 0.63
    STRONG_SIZE: float = 0.10  # 10% for strong stocks

    AVERAGE_KELLY: float = 0.16
    AVERAGE_HALF_KELLY: float = 0.08
    AVERAGE_WIN_RATE: float = 0.56
    AVERAGE_SIZE: float = 0.07  # 7% for average stocks

    AVOID_SIZE: float = 0.0  # Skip avoid-tier stocks

    # Signal strength multipliers
    STRENGTH_VERY_STRONG: float = 1.3   # 85+ signal strength
    STRENGTH_STRONG: float = 1.1        # 70-84 signal strength
    STRENGTH_MODERATE: float = 1.0      # 55-69 signal strength
    STRENGTH_WEAK: float = 0.7          # Below 55


POSITION_SIZING = PositionSizing()


# =============================================================================
# REGIME SETTINGS
# =============================================================================
# Research basis: Regime analysis of SPY/VIX patterns + backtest results

@dataclass(frozen=True)
class RegimeSettings:
    """Market regime detection and trading parameters."""

    # Position multipliers by regime
    # Higher = larger positions, lower = reduce exposure
    MULT_BULL: float = 1.0       # Standard size in bull
    MULT_BEAR: float = 1.0       # Full size - best regime for mean reversion!
    MULT_CHOPPY: float = 0.3     # 70% reduction - poor conditions
    MULT_VOLATILE: float = 0.5   # 50% reduction - high uncertainty

    # Stop-loss adjustments by regime
    STOP_BULL: float = 1.0       # Normal stops
    STOP_BEAR: float = 1.25      # Widen 25% for higher volatility
    STOP_CHOPPY: float = 1.0     # Normal stops
    STOP_VOLATILE: float = 1.4   # Widen 40% for extreme volatility

    # VIX thresholds for regime classification
    VIX_HIGH: float = 30.0       # Above = VOLATILE regime
    VIX_ELEVATED: float = 25.0   # Elevated caution
    VIX_NORMAL: float = 20.0     # Normal market conditions
    VIX_LOW: float = 15.0        # Low volatility / complacency

    # Trend thresholds (% change for regime detection)
    TREND_BULL_20D: float = 3.0   # 20-day SPY trend for bull
    TREND_BEAR_20D: float = -3.0  # 20-day SPY trend for bear
    TREND_BULL_50D: float = 5.0   # 50-day SPY trend for bull
    TREND_BEAR_50D: float = -5.0  # 50-day SPY trend for bear

    # Backtest performance by regime (for reference)
    SHARPE_BULL: float = 0.43
    SHARPE_BEAR: float = 1.80    # Excellent!
    SHARPE_CHOPPY: float = 0.0   # Poor
    SHARPE_VOLATILE: float = 0.5  # Moderate


REGIME_SETTINGS = RegimeSettings()


# =============================================================================
# BEAR DETECTION
# =============================================================================
# Research basis: 5-year backtest with genetic algorithm optimization

@dataclass(frozen=True)
class BearDetection:
    """Fast bear detection parameters."""

    # Alert level thresholds (score 0-100)
    LEVEL_NORMAL: int = 0
    LEVEL_WATCH: int = 30
    LEVEL_WARNING: int = 50
    LEVEL_CRITICAL: int = 70

    # Indicator weights (optimized via genetic algorithm)
    # Must sum to 1.0
    WEIGHT_BREADTH: float = 0.158        # Market breadth (KEY)
    WEIGHT_SECTOR_BREADTH: float = 0.141 # Sector breadth (KEY)
    WEIGHT_HIGH_YIELD: float = 0.132     # HYG vs LQD spread
    WEIGHT_CREDIT_SPREAD: float = 0.126  # LQD vs TLT spread
    WEIGHT_PUT_CALL: float = 0.124       # Options sentiment
    WEIGHT_VOLUME: float = 0.108         # Volume confirmation
    WEIGHT_DIVERGENCE: float = 0.096     # Price/breadth divergence
    WEIGHT_SPY_ROC: float = 0.053        # Short-term momentum
    WEIGHT_YIELD_CURVE: float = 0.040    # Recession indicator
    WEIGHT_VIX: float = 0.023            # Fear index

    # Position sizing adjustments by alert level
    SIZE_MULT_NORMAL: float = 1.0     # Full size
    SIZE_MULT_WATCH: float = 0.85     # 15% reduction
    SIZE_MULT_WARNING: float = 0.65   # 35% reduction
    SIZE_MULT_CRITICAL: float = 0.40  # 60% reduction

    # Alert cooldowns (hours)
    COOLDOWN_WATCH: int = 24
    COOLDOWN_WARNING: int = 4
    COOLDOWN_CRITICAL: int = 1

    # Validation metrics (5-year backtest)
    VALIDATION_HIT_RATE: float = 100.0  # % of drawdowns warned
    VALIDATION_AVG_LEAD: float = 5.2    # Days warning before drop
    VALIDATION_FALSE_POSITIVES: int = 0


BEAR_DETECTION = BearDetection()


# =============================================================================
# SECTOR MOMENTUM
# =============================================================================
# Research basis: 1-year sector rotation analysis

@dataclass(frozen=True)
class SectorMomentum:
    """Sector momentum parameters."""

    # Momentum category thresholds (5-day %)
    WEAK_THRESHOLD: float = -3.0        # Very weak sector
    SLIGHTLY_WEAK: float = -1.0         # Mildly weak
    NEUTRAL_LOW: float = -1.0           # Neutral range
    NEUTRAL_HIGH: float = 1.0
    SLIGHTLY_STRONG: float = 3.0        # Mildly strong
    STRONG_THRESHOLD: float = 3.0       # Very strong sector

    # Signal boost/penalty by sector momentum
    # Weak sectors outperform for mean reversion
    BOOST_WEAK: float = 1.10            # +10% boost for weak sectors
    BOOST_SLIGHTLY_WEAK: float = 1.05   # +5% boost
    MULT_NEUTRAL: float = 1.0           # No adjustment
    PENALTY_SLIGHTLY_STRONG: float = 0.95  # -5% penalty
    PENALTY_STRONG: float = 0.90        # -10% penalty for strong sectors


SECTOR_MOMENTUM = SectorMomentum()


# =============================================================================
# EXIT STRATEGY
# =============================================================================
# Research basis: Exit timing and stop-loss optimization

@dataclass(frozen=True)
class ExitStrategy:
    """Exit and stop-loss parameters."""

    # Default stop-loss percentages
    STOP_LOSS_ELITE: float = 0.08      # 8% for elite stocks
    STOP_LOSS_STRONG: float = 0.07     # 7% for strong stocks
    STOP_LOSS_AVERAGE: float = 0.06    # 6% for average stocks

    # Profit targets
    TARGET_ELITE: float = 0.10         # 10% target for elite
    TARGET_STRONG: float = 0.08        # 8% for strong
    TARGET_AVERAGE: float = 0.06       # 6% for average

    # Trailing stop parameters
    TRAILING_TRIGGER: float = 0.03     # Activate at 3% profit
    TRAILING_DISTANCE: float = 0.02    # Trail by 2%

    # Maximum hold periods (trading days)
    MAX_HOLD_ELITE: int = 10
    MAX_HOLD_STRONG: int = 8
    MAX_HOLD_AVERAGE: int = 5

    # Time-based exit adjustments
    DAY_1_TARGET_MULT: float = 0.5     # Take half at day 1 target
    DAY_3_STOP_TIGHTEN: float = 0.8    # Tighten stop 20% at day 3


EXIT_STRATEGY = ExitStrategy()


# =============================================================================
# DATA QUALITY
# =============================================================================

@dataclass(frozen=True)
class DataQuality:
    """Data quality thresholds."""

    # Price validation
    MIN_PRICE: float = 1.0             # Skip penny stocks
    MAX_PRICE: float = 10000.0         # Sanity check
    MAX_STALE_DAYS: int = 3            # Data older than 3 days = stale

    # Volume requirements
    MIN_AVG_VOLUME: int = 100000       # Min 100K daily volume
    MIN_DOLLAR_VOLUME: int = 5000000   # Min $5M daily dollar volume

    # Data coverage
    MIN_HISTORY_DAYS: int = 60         # Need 60+ days for indicators
    MIN_BARS_FOR_SCAN: int = 50        # Min bars for scanning


DATA_QUALITY = DataQuality()


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

# Stock tier classifications (for reference)
STOCK_TIERS = {
    'elite': [
        'COP', 'ABBV', 'SLB', 'CVS', 'GILD', 'EOG', 'AIG',
        'IDXX', 'USB', 'TMUS', 'SCHW', 'SYK', 'LMT', 'ADI', 'TXN'
    ],
    'strong': [
        'QCOM', 'JNJ', 'V', 'MPC', 'SHW', 'KLAC', 'MS', 'AMAT'
    ],
    'weak': [
        'NVDA', 'AVGO', 'AXP', 'WMT', 'CMCSA', 'META', 'INSM',
        'ROAD', 'MRVL', 'MLM', 'LOW', 'PNC', 'ECL'
    ],
    'avoid': [
        'NOW', 'CAT', 'CRM', 'HCA', 'TGT', 'ETN', 'HD',
        'ORCL', 'ADBE', 'INTU'
    ]
}


if __name__ == '__main__':
    # Print all constants for verification
    print("Proteus Trading Constants")
    print("=" * 60)

    print("\nSignal Thresholds:")
    print(f"  Base minimum: {SIGNAL_THRESHOLDS.BASE_MINIMUM}")
    print(f"  Bull: {SIGNAL_THRESHOLDS.BULL}")
    print(f"  Bear: {SIGNAL_THRESHOLDS.BEAR}")
    print(f"  Choppy: {SIGNAL_THRESHOLDS.CHOPPY}")

    print("\nPosition Sizing:")
    print(f"  Max positions: {POSITION_SIZING.MAX_POSITIONS}")
    print(f"  Max portfolio heat: {POSITION_SIZING.MAX_PORTFOLIO_HEAT:.0%}")
    print(f"  Elite size: {POSITION_SIZING.ELITE_SIZE:.0%}")

    print("\nRegime Settings:")
    print(f"  VIX high threshold: {REGIME_SETTINGS.VIX_HIGH}")
    print(f"  Choppy multiplier: {REGIME_SETTINGS.MULT_CHOPPY}")

    print("\nBear Detection:")
    print(f"  Warning threshold: {BEAR_DETECTION.LEVEL_WARNING}")
    print(f"  Critical threshold: {BEAR_DETECTION.LEVEL_CRITICAL}")
    print(f"  Hit rate: {BEAR_DETECTION.VALIDATION_HIT_RATE}%")

    print("\nStock Tiers:")
    for tier, stocks in STOCK_TIERS.items():
        print(f"  {tier}: {len(stocks)} stocks")

"""
PositionSizer - Risk-adjusted position sizing based on research findings.

Research from position_sizing_research.py (2,555 trades, 400 days):
- Kelly fractions vary by tier: STRONG 36.7%, ELITE 23.8%, AVERAGE 15.9%, AVOID 0%
- Half-Kelly recommended for safety
- Volatility-based sizing: higher vol = bigger moves but more risk
- Max 6 positions, max 15% portfolio heat

Additional research insights:
- Day of week: Monday +0.86% edge (best for mean reversion)
- Consecutive down days: 2 days optimal (+1.02% edge)
- Volume: Very high volume = 72.6% win rate (+1.73% edge)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum
import json
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.trading.exit_optimizer import ExitOptimizer
from common.analysis.fast_bear_detector import FastBearDetector


# Bear adjustment multipliers by alert level
BEAR_POSITION_MULTIPLIERS = {
    'NORMAL': 1.0,
    'WATCH': 0.85,
    'WARNING': 0.65,
    'CRITICAL': 0.40
}


class StockTier(Enum):
    """Stock tier classification based on performance."""
    ELITE = "elite"      # Top performers, widest stops
    STRONG = "strong"    # Good performers
    AVERAGE = "average"  # Typical stocks
    WEAK = "weak"        # Underperformers
    AVOID = "avoid"      # Skip these


class SignalQuality(Enum):
    """Signal quality classification."""
    POOR = "poor"          # Skip trade
    WEAK = "weak"          # Minimal size
    MODERATE = "moderate"  # Standard size
    STRONG = "strong"      # Increased size
    VERY_STRONG = "very_strong"  # Maximum size


@dataclass
class PositionSize:
    """Complete position sizing recommendation."""
    ticker: str
    size_pct: float           # Position size as % of portfolio (0-15%)
    dollar_size: float        # Dollar amount for given portfolio
    shares: int               # Number of shares at current price
    risk_dollars: float       # Dollar risk (position * stop loss %)
    rationale: str            # Explanation of sizing decision
    quality: SignalQuality    # Signal quality assessment
    skip_trade: bool          # Whether to skip this trade entirely


class PositionSizer:
    """
    Determines optimal position size based on:
    1. Stock tier (from ExitOptimizer)
    2. Signal strength
    3. Current volatility (ATR)
    4. Portfolio constraints (max heat, max positions)

    Persists positions to disk for tracking across scans.
    """

    POSITIONS_FILE = Path('data/portfolio/active_positions.json')

    def __init__(self, portfolio_value: float = 100000.0, load_positions: bool = True,
                 use_bear_detection: bool = True):
        self.portfolio_value = portfolio_value
        self.exit_optimizer = ExitOptimizer()
        self.use_bear_detection = use_bear_detection

        # Initialize bear detector for market-aware sizing
        self.bear_detector = None
        self._bear_signal = None
        self._bear_adjustment = 1.0
        if use_bear_detection:
            try:
                self.bear_detector = FastBearDetector()
            except Exception as e:
                print(f"Warning: Could not initialize bear detector: {e}")
                self.use_bear_detection = False

        # Load research-based parameters
        self._load_research_params()

        # Portfolio constraints
        self.max_positions = 6
        self.max_portfolio_heat = 0.15  # 15% max at risk
        self.min_position_size = 0.02   # 2% minimum
        self.risk_per_trade = 0.02      # 2% risk per trade

        # Current portfolio state
        self.current_positions = {}  # ticker -> position_info dict

        # Load persisted positions
        if load_positions:
            self._load_positions()

    def _load_research_params(self):
        """Load parameters from position sizing research."""
        research_path = Path('data/research/position_sizing_analysis.json')

        if research_path.exists():
            with open(research_path) as f:
                data = json.load(f)

            # Tier-based sizing from research
            self.tier_params = {}
            for tier, metrics in data.get('tier_metrics', {}).items():
                self.tier_params[tier] = {
                    'kelly': metrics['kelly_fraction'] / 100,
                    'half_kelly': metrics['half_kelly'] / 100,
                    'win_rate': metrics['win_rate'] / 100,
                    'avg_win': metrics['avg_win'],
                    'avg_loss': metrics['avg_loss'],
                    'recommended': metrics['recommended_size'] / 100
                }

            # Portfolio recommendations
            recs = data.get('portfolio_recommendations', {})
            self.tier_limits = recs.get('tier_limits', {})
            self.strength_multipliers = recs.get('strength_multipliers', {})
        else:
            # Default parameters if research not available
            self._set_default_params()

    def _set_default_params(self):
        """Set default parameters if research file not found."""
        self.tier_params = {
            'elite': {'kelly': 0.24, 'half_kelly': 0.12, 'win_rate': 0.60, 'recommended': 0.12},
            'strong': {'kelly': 0.37, 'half_kelly': 0.18, 'win_rate': 0.63, 'recommended': 0.10},
            'average': {'kelly': 0.16, 'half_kelly': 0.08, 'win_rate': 0.56, 'recommended': 0.07},
            'avoid': {'kelly': 0.0, 'half_kelly': 0.0, 'win_rate': 0.45, 'recommended': 0.0}
        }
        self.tier_limits = {
            'elite': {'min_size': 0.02, 'max_size': 0.12, 'default_size': 0.12},
            'strong': {'min_size': 0.02, 'max_size': 0.10, 'default_size': 0.10},
            'average': {'min_size': 0.02, 'max_size': 0.07, 'default_size': 0.07},
            'avoid': {'min_size': 0.02, 'max_size': 0.04, 'default_size': 0.0}
        }
        self.strength_multipliers = {
            'very_strong': 1.3,
            'strong': 1.1,
            'moderate': 1.0,
            'weak': 0.7
        }

    def get_bear_adjustment(self, refresh: bool = False) -> Tuple[float, str]:
        """
        Get bear market adjustment factor for position sizing.

        Returns:
            Tuple of (multiplier, alert_level)
            - multiplier: 1.0 for NORMAL, down to 0.40 for CRITICAL
            - alert_level: Current bear alert level string
        """
        if not self.use_bear_detection or self.bear_detector is None:
            return 1.0, 'DISABLED'

        # Refresh bear signal if needed or not cached
        if refresh or self._bear_signal is None:
            try:
                self._bear_signal = self.bear_detector.detect()
                alert_level = self._bear_signal.alert_level.upper()
                self._bear_adjustment = BEAR_POSITION_MULTIPLIERS.get(alert_level, 1.0)
            except Exception as e:
                print(f"Warning: Bear detection failed: {e}")
                return 1.0, 'ERROR'

        alert_level = self._bear_signal.alert_level.upper() if self._bear_signal else 'UNKNOWN'
        return self._bear_adjustment, alert_level

    def _load_positions(self):
        """Load persisted positions from disk."""
        if self.POSITIONS_FILE.exists():
            try:
                with open(self.POSITIONS_FILE) as f:
                    data = json.load(f)
                self.current_positions = data.get('positions', {})
                # Convert size_pct back to decimal if stored as percentage
                for ticker, info in self.current_positions.items():
                    if isinstance(info, (int, float)):
                        # Old format: just size_pct as float
                        self.current_positions[ticker] = {
                            'size_pct': info,
                            'entry_date': None,
                            'entry_price': None
                        }
            except Exception as e:
                print(f"Warning: Could not load positions: {e}")
                self.current_positions = {}

    def _save_positions(self):
        """Save current positions to disk."""
        self.POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'last_updated': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_value,
            'positions': self.current_positions,
            'summary': {
                'position_count': len(self.current_positions),
                'total_heat': sum(
                    p.get('size_pct', p) if isinstance(p, dict) else p
                    for p in self.current_positions.values()
                )
            }
        }
        with open(self.POSITIONS_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def can_add_position(self) -> Tuple[bool, str]:
        """Check if we can add a new position based on constraints."""
        # Check position count
        if len(self.current_positions) >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"

        # Check portfolio heat
        current_heat = self._get_current_heat()
        if current_heat >= self.max_portfolio_heat:
            return False, f"Portfolio heat limit reached ({current_heat*100:.1f}%)"

        return True, "OK"

    def _get_current_heat(self) -> float:
        """Get total portfolio heat (sum of all position sizes)."""
        total = 0.0
        for info in self.current_positions.values():
            if isinstance(info, dict):
                total += info.get('size_pct', 0)
            else:
                total += info
        return total

    def get_signal_quality(self, signal_strength: float, tier: StockTier,
                          volume_ratio: float = 1.0,
                          consecutive_down_days: int = 0) -> SignalQuality:
        """
        Assess overall signal quality based on multiple factors.

        Args:
            signal_strength: Base signal strength (0-100)
            tier: Stock tier from ExitOptimizer
            volume_ratio: Current volume vs average (1.0 = normal)
            consecutive_down_days: Number of consecutive down days
        """
        # AVOID tier stocks are always poor quality
        if tier == StockTier.AVOID:
            return SignalQuality.POOR

        score = 0

        # Signal strength contribution (0-40 points)
        if signal_strength >= 80:
            score += 40
        elif signal_strength >= 70:
            score += 30
        elif signal_strength >= 60:
            score += 20
        else:
            score += 10

        # Tier contribution (0-30 points)
        tier_scores = {
            StockTier.ELITE: 30,
            StockTier.STRONG: 25,
            StockTier.AVERAGE: 15,
            StockTier.AVOID: 0
        }
        score += tier_scores.get(tier, 15)

        # Volume contribution (0-20 points) - high volume = capitulation
        if volume_ratio >= 2.5:
            score += 20
        elif volume_ratio >= 1.5:
            score += 15
        elif volume_ratio >= 1.0:
            score += 10
        else:
            score += 5

        # Consecutive down days (0-10 points) - 2 days optimal
        if consecutive_down_days == 2:
            score += 10
        elif consecutive_down_days in [1, 3]:
            score += 7
        elif consecutive_down_days == 0:
            score += 5
        else:  # 4+ days
            score += 3

        # Map score to quality
        if score >= 80:
            return SignalQuality.VERY_STRONG
        elif score >= 65:
            return SignalQuality.STRONG
        elif score >= 50:
            return SignalQuality.MODERATE
        elif score >= 35:
            return SignalQuality.WEAK
        else:
            return SignalQuality.POOR

    def calculate_position_size(self,
                               ticker: str,
                               current_price: float,
                               signal_strength: float = 70,
                               atr_pct: float = 2.5,
                               volume_ratio: float = 1.0,
                               consecutive_down_days: int = 0) -> PositionSize:
        """
        Calculate optimal position size for a trade.

        Args:
            ticker: Stock ticker
            current_price: Current stock price
            signal_strength: Signal strength (0-100)
            atr_pct: Average True Range as % of price
            volume_ratio: Current volume vs 20-day average
            consecutive_down_days: Number of consecutive down days

        Returns:
            PositionSize with complete sizing recommendation
        """
        # Get tier (returns string from exit_optimizer)
        tier_str = self.exit_optimizer.get_stock_tier(ticker)
        # Convert to enum if possible, otherwise use string directly
        try:
            tier = StockTier(tier_str.lower())
            tier_value = tier.value
        except (ValueError, AttributeError):
            tier_value = tier_str.lower() if isinstance(tier_str, str) else 'average'
            tier = StockTier.AVERAGE  # Default for signal quality check

        # Assess signal quality
        quality = self.get_signal_quality(
            signal_strength, tier, volume_ratio, consecutive_down_days
        )

        # Check if we should skip
        if quality == SignalQuality.POOR:
            return PositionSize(
                ticker=ticker,
                size_pct=0.0,
                dollar_size=0.0,
                shares=0,
                risk_dollars=0.0,
                rationale=f"SKIP: {tier_value.upper()} tier stock with poor signal quality",
                quality=quality,
                skip_trade=True
            )

        # Get base size from tier
        tier_params = self.tier_params.get(tier_value, self.tier_params['average'])
        tier_limit = self.tier_limits.get(tier_value, self.tier_limits['average'])

        base_size = tier_params['recommended']

        # Apply signal strength multiplier
        if signal_strength >= 80:
            strength_mult = self.strength_multipliers.get('very_strong', 1.3)
        elif signal_strength >= 70:
            strength_mult = self.strength_multipliers.get('strong', 1.1)
        elif signal_strength >= 60:
            strength_mult = self.strength_multipliers.get('moderate', 1.0)
        else:
            strength_mult = self.strength_multipliers.get('weak', 0.7)

        adjusted_size = base_size * strength_mult

        # Apply quality adjustment
        quality_multipliers = {
            SignalQuality.VERY_STRONG: 1.2,
            SignalQuality.STRONG: 1.1,
            SignalQuality.MODERATE: 1.0,
            SignalQuality.WEAK: 0.7,
            SignalQuality.POOR: 0.0
        }
        adjusted_size *= quality_multipliers[quality]

        # Volatility adjustment (reduce size for high vol to maintain risk parity)
        # Target 2% risk per trade
        if atr_pct > 0:
            vol_adjustment = min(2.0 / atr_pct, 1.5)  # Cap at 1.5x
            adjusted_size *= vol_adjustment

        # Bear market adjustment - reduce size when bear conditions elevated
        bear_mult, bear_level = self.get_bear_adjustment()
        adjusted_size *= bear_mult

        # Apply tier limits
        max_size = tier_limit.get('max_size', 0.10)
        min_size = tier_limit.get('min_size', 0.02)
        adjusted_size = max(min_size, min(adjusted_size, max_size))

        # Check portfolio heat constraint
        current_heat = sum(self.current_positions.values())
        available_heat = self.max_portfolio_heat - current_heat

        if adjusted_size > available_heat:
            if available_heat < min_size:
                return PositionSize(
                    ticker=ticker,
                    size_pct=0.0,
                    dollar_size=0.0,
                    shares=0,
                    risk_dollars=0.0,
                    rationale=f"SKIP: Portfolio heat limit reached ({current_heat*100:.1f}% used)",
                    quality=quality,
                    skip_trade=True
                )
            adjusted_size = available_heat

        # Calculate dollar amounts
        dollar_size = self.portfolio_value * adjusted_size
        shares = int(dollar_size / current_price) if current_price > 0 else 0
        actual_dollar_size = shares * current_price

        # Calculate risk in dollars (using stop loss from exit strategy)
        exit_strategy = self.exit_optimizer.get_exit_strategy(ticker, signal_strength)
        stop_loss_pct = abs(exit_strategy.stop_loss) / 100
        risk_dollars = actual_dollar_size * stop_loss_pct

        # Build rationale
        rationale_parts = [
            f"Tier: {tier_value.upper()} (base {tier_params['recommended']*100:.0f}%)",
            f"Signal: {signal_strength:.0f} ({quality.value})",
            f"Vol adj: {atr_pct:.1f}% ATR"
        ]

        if volume_ratio >= 1.5:
            rationale_parts.append(f"High vol: {volume_ratio:.1f}x")
        if consecutive_down_days >= 2:
            rationale_parts.append(f"{consecutive_down_days}d down")

        # Add bear adjustment info if not NORMAL
        if bear_level not in ('NORMAL', 'DISABLED'):
            rationale_parts.append(f"Bear: {bear_level} ({bear_mult:.0%})")

        rationale = " | ".join(rationale_parts)

        return PositionSize(
            ticker=ticker,
            size_pct=round(adjusted_size * 100, 2),
            dollar_size=round(actual_dollar_size, 2),
            shares=shares,
            risk_dollars=round(risk_dollars, 2),
            rationale=rationale,
            quality=quality,
            skip_trade=False
        )

    def add_position(self, ticker: str, size_pct: float, entry_price: float = None):
        """
        Track a new position and persist to disk.

        Args:
            ticker: Stock ticker
            size_pct: Position size as percentage (e.g., 10 for 10%)
            entry_price: Entry price (optional, for tracking)
        """
        self.current_positions[ticker] = {
            'size_pct': size_pct / 100,  # Store as decimal
            'entry_date': datetime.now().isoformat(),
            'entry_price': entry_price
        }
        self._save_positions()

    def remove_position(self, ticker: str):
        """Remove a closed position and persist to disk."""
        if ticker in self.current_positions:
            del self.current_positions[ticker]
            self._save_positions()

    def is_already_holding(self, ticker: str) -> bool:
        """Check if we already have a position in this ticker."""
        return ticker in self.current_positions

    def get_position_age_days(self, ticker: str) -> Optional[int]:
        """Get how many days we've held a position."""
        if ticker not in self.current_positions:
            return None
        info = self.current_positions[ticker]
        if isinstance(info, dict) and info.get('entry_date'):
            entry = datetime.fromisoformat(info['entry_date'])
            return (datetime.now() - entry).days
        return None

    def get_portfolio_status(self) -> Dict:
        """Get current portfolio allocation status."""
        total_heat = self._get_current_heat()
        can_add, reason = self.can_add_position()

        # Build position details
        position_details = {}
        for ticker, info in self.current_positions.items():
            if isinstance(info, dict):
                size = info.get('size_pct', 0)
                position_details[ticker] = {
                    'size_pct': round(size * 100, 2),
                    'entry_date': info.get('entry_date'),
                    'entry_price': info.get('entry_price'),
                    'days_held': self.get_position_age_days(ticker)
                }
            else:
                position_details[ticker] = {'size_pct': round(info * 100, 2)}

        return {
            'positions': len(self.current_positions),
            'max_positions': self.max_positions,
            'can_add': can_add,
            'can_add_reason': reason,
            'total_heat': round(total_heat * 100, 2),
            'max_heat': self.max_portfolio_heat * 100,
            'available_heat': round((self.max_portfolio_heat - total_heat) * 100, 2),
            'current_positions': position_details
        }

    def rank_opportunities(self, opportunities: list) -> list:
        """
        Rank multiple trading opportunities by quality.

        Args:
            opportunities: List of dicts with ticker, price, signal_strength, etc.

        Returns:
            Sorted list with position sizing added
        """
        sized_opportunities = []

        for opp in opportunities:
            size = self.calculate_position_size(
                ticker=opp['ticker'],
                current_price=opp.get('current_price', opp.get('price', 100)),
                signal_strength=opp.get('signal_strength', 70),
                atr_pct=opp.get('atr_pct', 2.5),
                volume_ratio=opp.get('volume_ratio', 1.0),
                consecutive_down_days=opp.get('consecutive_down_days', 0)
            )

            if not size.skip_trade:
                sized_opportunities.append({
                    **opp,
                    'position_size': size
                })

        # Sort by quality score (higher is better)
        quality_order = {
            SignalQuality.VERY_STRONG: 4,
            SignalQuality.STRONG: 3,
            SignalQuality.MODERATE: 2,
            SignalQuality.WEAK: 1,
            SignalQuality.POOR: 0
        }

        return sorted(
            sized_opportunities,
            key=lambda x: (
                quality_order[x['position_size'].quality],
                x['position_size'].size_pct
            ),
            reverse=True
        )

    def format_recommendation(self, size: PositionSize) -> str:
        """Format position size for display."""
        if size.skip_trade:
            return f"  {size.ticker}: SKIP - {size.rationale}"

        return (
            f"  {size.ticker}: {size.size_pct:.1f}% (${size.dollar_size:,.0f}, "
            f"{size.shares} shares) | Risk: ${size.risk_dollars:.0f} | "
            f"{size.quality.value.upper()}"
        )


def demo():
    """Demonstrate PositionSizer usage."""
    sizer = PositionSizer(portfolio_value=100000)

    print("=" * 70)
    print("POSITION SIZER DEMO")
    print("=" * 70)
    print(f"Portfolio: ${sizer.portfolio_value:,.0f}")
    print()

    # Test different scenarios
    test_cases = [
        {'ticker': 'COP', 'current_price': 105.0, 'signal_strength': 85, 'atr_pct': 2.8, 'volume_ratio': 2.0, 'consecutive_down_days': 2},
        {'ticker': 'QCOM', 'current_price': 180.0, 'signal_strength': 75, 'atr_pct': 3.2, 'volume_ratio': 1.3, 'consecutive_down_days': 1},
        {'ticker': 'MSFT', 'current_price': 420.0, 'signal_strength': 65, 'atr_pct': 2.0, 'volume_ratio': 0.8, 'consecutive_down_days': 0},
        {'ticker': 'NOW', 'current_price': 900.0, 'signal_strength': 70, 'atr_pct': 3.5, 'volume_ratio': 1.0, 'consecutive_down_days': 1},  # AVOID tier
    ]

    print("INDIVIDUAL SIZING:")
    print("-" * 70)

    for case in test_cases:
        size = sizer.calculate_position_size(**case)
        print(sizer.format_recommendation(size))
        print(f"    Rationale: {size.rationale}")
        print()

    print("\nRANKED OPPORTUNITIES:")
    print("-" * 70)

    ranked = sizer.rank_opportunities(test_cases)
    for i, opp in enumerate(ranked, 1):
        size = opp['position_size']
        print(f"  {i}. {sizer.format_recommendation(size)}")

    print("\nPORTFOLIO STATUS:")
    print("-" * 70)

    # Simulate adding positions
    sizer.add_position('COP', 10.0)
    sizer.add_position('QCOM', 8.0)

    status = sizer.get_portfolio_status()
    print(f"  Positions: {status['positions']}/{status['max_positions']}")
    print(f"  Portfolio heat: {status['total_heat']}% (max {status['max_heat']}%)")
    print(f"  Available: {status['available_heat']}%")
    print(f"  Holdings: {status['current_positions']}")


if __name__ == '__main__':
    demo()

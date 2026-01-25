"""
PositionRebalancer - Checks positions and recommends exits/rebalancing.

Uses unified_config.json exit_strategy rules per tier:
- profit_target_1: First profit target (partial exit)
- profit_target_2: Second profit target (full exit)
- stop_loss: Stop loss level
- max_hold_days: Maximum holding period
- trailing_distance: Trailing stop distance
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum


class ExitReason(Enum):
    """Reason for exit recommendation."""
    NONE = "hold"
    PROFIT_TARGET_1 = "profit_target_1"
    PROFIT_TARGET_2 = "profit_target_2"
    STOP_LOSS = "stop_loss"
    MAX_HOLD_DAYS = "max_hold_days"
    TRAILING_STOP = "trailing_stop"


@dataclass
class PositionStatus:
    """Status of a single position."""
    ticker: str
    tier: str
    entry_price: float
    current_price: float
    entry_date: datetime
    days_held: int
    pnl_pct: float
    high_since_entry: float
    exit_reason: ExitReason
    exit_pct: int  # 0, 50, or 100
    recommendation: str


class PositionRebalancer:
    """
    Check positions against exit rules and recommend actions.
    """

    CONFIG_PATH = Path('features/simulation/config.json')
    POSITIONS_FILE = Path('features/simulation/data/portfolio/active_positions.json')

    def __init__(self):
        self.config = self._load_config()
        self.positions = self._load_positions()

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
        return {}

    def _load_positions(self) -> dict:
        """Load active positions."""
        if self.POSITIONS_FILE.exists():
            try:
                with open(self.POSITIONS_FILE) as f:
                    data = json.load(f)
                return data.get('positions', {})
            except Exception:
                return {}
        return {}

    def get_tier(self, ticker: str) -> str:
        """Get stock tier."""
        return self.ticker_tiers.get(ticker, 'average')

    def get_exit_rules(self, tier: str) -> dict:
        """Get exit rules for tier."""
        exits = self.config.get('exit_strategy', {})
        return exits.get(tier, exits.get('average', {
            'profit_target_1': 2.0,
            'profit_target_2': 2.5,
            'partial_exit_pct': 50,
            'stop_loss': -2.5,
            'max_hold_days': 3,
            'use_trailing': True,
            'trailing_distance': 1.0
        }))

    def check_position(
        self,
        ticker: str,
        entry_price: float,
        current_price: float,
        entry_date: datetime,
        high_since_entry: float = None
    ) -> PositionStatus:
        """
        Check a position against exit rules.

        Args:
            ticker: Stock ticker
            entry_price: Entry price
            current_price: Current price
            entry_date: Entry date
            high_since_entry: Highest price since entry (for trailing stop)

        Returns:
            PositionStatus with exit recommendation
        """
        tier = self.get_tier(ticker)
        rules = self.get_exit_rules(tier)

        # Calculate metrics
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        days_held = (datetime.now() - entry_date).days
        if high_since_entry is None:
            high_since_entry = max(entry_price, current_price)

        # Check each exit condition (order matters: stop loss first, then profits)
        exit_reason = ExitReason.NONE
        exit_pct = 0

        # 1. Stop loss check (highest priority - always exit)
        stop_loss = rules.get('stop_loss', -2.5)
        if pnl_pct <= stop_loss:
            exit_reason = ExitReason.STOP_LOSS
            exit_pct = 100

        # 2. Profit target 2 (full exit)
        elif rules.get('profit_target_2') and pnl_pct >= rules['profit_target_2']:
            exit_reason = ExitReason.PROFIT_TARGET_2
            exit_pct = 100

        # 3. Trailing stop check (only when in profit and pullback from high)
        elif rules.get('use_trailing', False) and high_since_entry > entry_price:
            trailing_dist = rules.get('trailing_distance', 1.0)
            trail_from_high = ((current_price - high_since_entry) / high_since_entry) * 100
            if trail_from_high <= -trailing_dist and pnl_pct > 0:
                exit_reason = ExitReason.TRAILING_STOP
                exit_pct = 100

        # 4. Max hold days (only check if not exiting for other reasons)
        if exit_reason == ExitReason.NONE and days_held >= rules.get('max_hold_days', 5):
            exit_reason = ExitReason.MAX_HOLD_DAYS
            exit_pct = 100

        # 5. Profit target 1 (partial exit - only if no other exit triggered)
        if exit_reason == ExitReason.NONE:
            if rules.get('profit_target_1') and pnl_pct >= rules['profit_target_1']:
                exit_reason = ExitReason.PROFIT_TARGET_1
                exit_pct = rules.get('partial_exit_pct', 50)

        # Build recommendation
        if exit_reason == ExitReason.NONE:
            recommendation = f"HOLD (Day {days_held}, {pnl_pct:+.1f}%)"
        elif exit_pct == 100:
            recommendation = f"EXIT 100% - {exit_reason.value} ({pnl_pct:+.1f}%)"
        else:
            recommendation = f"EXIT {exit_pct}% - {exit_reason.value} ({pnl_pct:+.1f}%)"

        return PositionStatus(
            ticker=ticker,
            tier=tier,
            entry_price=entry_price,
            current_price=current_price,
            entry_date=entry_date,
            days_held=days_held,
            pnl_pct=round(pnl_pct, 2),
            high_since_entry=high_since_entry,
            exit_reason=exit_reason,
            exit_pct=exit_pct,
            recommendation=recommendation
        )

    def check_all_positions(self, current_prices: Dict[str, float]) -> List[PositionStatus]:
        """
        Check all active positions.

        Args:
            current_prices: Dict of ticker -> current price

        Returns:
            List of PositionStatus for all positions
        """
        results = []

        for ticker, info in self.positions.items():
            if ticker not in current_prices:
                continue

            entry_price = info.get('entry_price')
            entry_date_str = info.get('entry_date')

            if not entry_price or not entry_date_str:
                continue

            try:
                entry_date = datetime.fromisoformat(entry_date_str)
            except ValueError:
                continue

            status = self.check_position(
                ticker=ticker,
                entry_price=entry_price,
                current_price=current_prices[ticker],
                entry_date=entry_date
            )
            results.append(status)

        # Sort by urgency (exits first, then by P&L)
        results.sort(key=lambda x: (
            0 if x.exit_reason != ExitReason.NONE else 1,
            -abs(x.pnl_pct)
        ))

        return results

    def get_rebalancing_summary(self, positions: List[PositionStatus]) -> str:
        """Format rebalancing summary."""
        lines = ["=" * 60, "POSITION REBALANCING CHECK", "=" * 60]

        exits = [p for p in positions if p.exit_reason != ExitReason.NONE]
        holds = [p for p in positions if p.exit_reason == ExitReason.NONE]

        if exits:
            lines.append("\nACTION REQUIRED:")
            lines.append("-" * 40)
            for p in exits:
                lines.append(f"  {p.ticker} ({p.tier}): {p.recommendation}")

        if holds:
            lines.append("\nHOLDING:")
            lines.append("-" * 40)
            for p in holds:
                lines.append(f"  {p.ticker} ({p.tier}): {p.recommendation}")

        if not exits and not holds:
            lines.append("\nNo positions to check.")

        return "\n".join(lines)


def demo():
    """Demonstrate rebalancer."""
    rebalancer = PositionRebalancer()

    print("=" * 60)
    print("POSITION REBALANCER DEMO")
    print("=" * 60)

    # Test positions
    test_cases = [
        {
            'ticker': 'COP',
            'entry_price': 100.0,
            'current_price': 103.5,  # +3.5% -> profit target 2
            'entry_date': datetime(2026, 1, 2)
        },
        {
            'ticker': 'MSFT',
            'entry_price': 420.0,
            'current_price': 412.0,  # -1.9% -> approaching stop
            'entry_date': datetime(2026, 1, 1)
        },
        {
            'ticker': 'QCOM',
            'entry_price': 180.0,
            'current_price': 183.6,  # +2% -> profit target 1
            'entry_date': datetime(2026, 1, 3)
        },
        {
            'ticker': 'NVDA',
            'entry_price': 140.0,
            'current_price': 137.2,  # -2% -> stop loss (weak tier)
            'entry_date': datetime(2026, 1, 3)
        }
    ]

    for case in test_cases:
        status = rebalancer.check_position(**case)
        print(f"\n{status.ticker} ({status.tier}):")
        print(f"  Entry: ${status.entry_price:.2f} on {status.entry_date.strftime('%Y-%m-%d')}")
        print(f"  Current: ${status.current_price:.2f} ({status.pnl_pct:+.1f}%)")
        print(f"  Days held: {status.days_held}")
        print(f"  -> {status.recommendation}")


if __name__ == '__main__':
    demo()

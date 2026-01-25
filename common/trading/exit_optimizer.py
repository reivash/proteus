"""
ExitOptimizer - Dynamic exit strategy based on research findings.

Updated Jan 5, 2026: Now reads from unified_config.json (single source of truth)

Research from exit_strategy_analysis.py (2,226 signals, 365 days):
- Elite/Strong stocks: Hold longer (5d), use +3% target
- Avoid stocks: Exit fast (1d), use +1.5% target
- Best partial exit: 50% at +2%, 50% trailing = 1.267% return (4.4x improvement!)
- Stop-loss: -3% works best (wider is better)
"""

import json
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from pathlib import Path


@dataclass
class ExitStrategy:
    """Exit strategy parameters for a trade."""
    profit_target_1: float      # First profit target (%)
    profit_target_2: Optional[float]  # Second profit target (%) or None for no scale-out
    portion_at_target_1: float  # Portion to exit at first target (0-1)
    stop_loss: float            # Stop-loss level (negative %)
    max_hold_days: int          # Max days to hold
    use_trailing: bool          # Use trailing stop for remaining position
    trailing_distance: float    # Trailing stop distance (%) if use_trailing


class ExitOptimizer:
    """
    Determines optimal exit strategy based on stock tier and signal strength.

    Reads ALL configuration from features/simulation/config.json:
    - Stock tiers (elite/strong/average/weak/avoid)
    - Exit parameters per tier (targets, stops, hold days, trailing)

    Research findings applied:
    1. Elite/Strong stocks can hold longer (5d) with higher targets (+3%)
    2. Avoid tier stocks should exit fast (1d) with lower targets (+1.5%)
    3. Partial exits (50% at +2%, 50% trail) = 1.267% return (4.4x vs fixed!)
    4. Stop-losses at -3% preserve more capital than tight stops
    """

    CONFIG_PATH = Path('features/simulation/config.json')

    def __init__(self):
        self.config = self._load_config()

        # Build ticker -> tier lookup (same as other unified modules)
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
                'elite': {'tickers': []},
                'strong': {'tickers': []},
                'average': {'tickers': []},
                'weak': {'tickers': []},
                'avoid': {'tickers': []}
            },
            'exit_strategy': {
                'elite': {
                    'profit_target_1': 2.0,
                    'profit_target_2': 3.5,
                    'partial_exit_pct': 50,
                    'stop_loss': -3.0,
                    'max_hold_days': 5,
                    'use_trailing': True,
                    'trailing_distance': 1.0
                },
                'strong': {
                    'profit_target_1': 2.0,
                    'profit_target_2': 3.0,
                    'partial_exit_pct': 50,
                    'stop_loss': -3.0,
                    'max_hold_days': 5,
                    'use_trailing': True,
                    'trailing_distance': 1.0
                },
                'average': {
                    'profit_target_1': 2.0,
                    'profit_target_2': 2.5,
                    'partial_exit_pct': 50,
                    'stop_loss': -2.5,
                    'max_hold_days': 3,
                    'use_trailing': True,
                    'trailing_distance': 1.0
                },
                'weak': {
                    'profit_target_1': 1.5,
                    'profit_target_2': 2.0,
                    'partial_exit_pct': 50,
                    'stop_loss': -2.0,
                    'max_hold_days': 2,
                    'use_trailing': True,
                    'trailing_distance': 0.8
                },
                'avoid': {
                    'profit_target_1': 1.5,
                    'profit_target_2': None,
                    'partial_exit_pct': 100,
                    'stop_loss': -2.0,
                    'max_hold_days': 1,
                    'use_trailing': False,
                    'trailing_distance': 0
                }
            }
        }

    def get_stock_tier(self, ticker: str) -> str:
        """Determine stock tier from ticker."""
        return self.ticker_tiers.get(ticker, 'average')

    def get_signal_strength_level(self, strength: float) -> str:
        """Categorize signal strength."""
        if strength >= 80:
            return 'very_strong'
        elif strength >= 70:
            return 'strong'
        elif strength >= 60:
            return 'moderate'
        else:
            return 'weak'

    def _get_exit_config(self, tier: str) -> dict:
        """Get exit strategy config for a tier."""
        exits = self.config.get('exit_strategy', {})
        return exits.get(tier, exits.get('average', {}))

    def get_exit_strategy(self, ticker: str, signal_strength: float = 70) -> ExitStrategy:
        """
        Get optimal exit strategy for a given ticker and signal strength.

        Args:
            ticker: Stock ticker
            signal_strength: Signal strength (0-100)

        Returns:
            ExitStrategy with optimal parameters
        """
        tier = self.get_stock_tier(ticker)
        strength_level = self.get_signal_strength_level(signal_strength)
        exit_config = self._get_exit_config(tier)

        # Base parameters from config
        profit_target_1 = exit_config.get('profit_target_1', 2.0)
        profit_target_2 = exit_config.get('profit_target_2')
        partial_exit_pct = exit_config.get('partial_exit_pct', 50)
        stop_loss = exit_config.get('stop_loss', -2.5)
        max_hold_days = exit_config.get('max_hold_days', 3)
        use_trailing = exit_config.get('use_trailing', True)
        trailing_distance = exit_config.get('trailing_distance', 1.0)

        # Adjust based on signal strength
        if strength_level == 'very_strong':
            # Strong signals can hold longer, aim higher
            max_hold_days = min(max_hold_days + 1, 5)
            profit_target_1 += 0.5
            if profit_target_2:
                profit_target_2 += 0.5
        elif strength_level == 'weak':
            # Weak signals should exit faster
            max_hold_days = max(max_hold_days - 1, 1)
            profit_target_1 = max(profit_target_1 - 0.5, 1.0)

        return ExitStrategy(
            profit_target_1=profit_target_1,
            profit_target_2=profit_target_2,
            portion_at_target_1=partial_exit_pct / 100,
            stop_loss=stop_loss,
            max_hold_days=max_hold_days,
            use_trailing=use_trailing,
            trailing_distance=trailing_distance,
        )

    def should_exit(self,
                    ticker: str,
                    entry_price: float,
                    current_price: float,
                    high_since_entry: float,
                    days_held: int,
                    portion_remaining: float = 1.0,
                    signal_strength: float = 70) -> Tuple[bool, float, str]:
        """
        Determine if position should be exited.

        Args:
            ticker: Stock ticker
            entry_price: Entry price
            current_price: Current price
            high_since_entry: Highest price since entry
            days_held: Days position has been held
            portion_remaining: Remaining position (0-1)
            signal_strength: Original signal strength

        Returns:
            Tuple of (should_exit, portion_to_exit, reason)
        """
        strategy = self.get_exit_strategy(ticker, signal_strength)

        current_return = ((current_price / entry_price) - 1) * 100
        max_return = ((high_since_entry / entry_price) - 1) * 100

        # Check stop-loss (highest priority)
        if current_return <= strategy.stop_loss:
            return True, portion_remaining, f"STOP-LOSS at {current_return:.2f}%"

        # Check trailing stop (if applicable and already hit first target)
        if strategy.use_trailing and portion_remaining < 1.0:
            trailing_trigger = max_return - strategy.trailing_distance
            if current_return < trailing_trigger and max_return >= strategy.profit_target_1:
                return True, portion_remaining, f"TRAILING STOP at {current_return:.2f}% (max was {max_return:.2f}%)"

        # Check first profit target
        if current_return >= strategy.profit_target_1 and portion_remaining == 1.0:
            if strategy.profit_target_2 is None:
                # No second target, exit all
                return True, 1.0, f"TARGET HIT at {current_return:.2f}%"
            else:
                # Partial exit
                return True, strategy.portion_at_target_1, f"PARTIAL EXIT at {current_return:.2f}%"

        # Check second profit target
        if strategy.profit_target_2 and current_return >= strategy.profit_target_2:
            return True, portion_remaining, f"TARGET 2 HIT at {current_return:.2f}%"

        # Check max hold days
        if days_held >= strategy.max_hold_days:
            return True, portion_remaining, f"MAX HOLD ({strategy.max_hold_days}d) at {current_return:.2f}%"

        return False, 0, "HOLD"

    def get_expected_metrics(self, ticker: str, signal_strength: float = 70) -> Dict:
        """
        Get expected performance metrics for a trade.

        Returns expected return and win rate based on tier and config.
        """
        tier = self.get_stock_tier(ticker)
        tier_config = self.config.get('stock_tiers', {}).get(tier, {})

        # Get win rate and avg return from config if available
        win_rate = tier_config.get('win_rate', 55)
        avg_return = tier_config.get('avg_return', 0.5)

        # Adjust for signal strength
        if signal_strength >= 80:
            avg_return *= 1.1
            win_rate += 3
        elif signal_strength < 60:
            avg_return *= 0.9
            win_rate -= 2

        return {
            'expected_return': round(avg_return, 2),
            'win_rate': round(win_rate, 1),
            'tier': tier,
            'strategy': self.get_exit_strategy(ticker, signal_strength)
        }

    def format_strategy(self, strategy: ExitStrategy) -> str:
        """Format strategy for display."""
        if strategy.profit_target_2:
            return (f"Exit {int(strategy.portion_at_target_1*100)}% at +{strategy.profit_target_1:.1f}%, "
                   f"rest at +{strategy.profit_target_2:.1f}% or trail {strategy.trailing_distance:.1f}% | "
                   f"Stop: {strategy.stop_loss:.1f}% | Max: {strategy.max_hold_days}d")
        else:
            return (f"Exit all at +{strategy.profit_target_1:.1f}% | "
                   f"Stop: {strategy.stop_loss:.1f}% | Max: {strategy.max_hold_days}d")


def demo():
    """Demonstrate ExitOptimizer usage."""
    optimizer = ExitOptimizer()

    print("=" * 70)
    print("EXIT OPTIMIZER (Unified Config)")
    print("=" * 70)
    print(f"Config: {optimizer.CONFIG_PATH}")
    print()

    # Test tickers from different tiers
    test_tickers = ['COP', 'MS', 'MSFT', 'NVDA', 'NOW']  # Elite, Strong, Average, Weak, Avoid

    print("Exit Strategies by Tier:")
    print("-" * 70)

    for ticker in test_tickers:
        metrics = optimizer.get_expected_metrics(ticker, signal_strength=75)
        strategy = metrics['strategy']
        tier = metrics['tier']

        print(f"\n{ticker} ({tier.upper()}):")
        print(f"  Expected Return: +{metrics['expected_return']}%")
        print(f"  Win Rate: {metrics['win_rate']}%")
        print(f"  Strategy: {optimizer.format_strategy(strategy)}")

    # Simulate a trade showing trailing stop
    print("\n" + "=" * 70)
    print("SIMULATED TRADE: COP (ELITE)")
    print("=" * 70)

    ticker = 'COP'
    entry = 100.0
    days = [
        (100.5, 100.5),  # Day 1: Small gain
        (101.0, 101.5),  # Day 2: Building
        (102.0, 102.5),  # Day 3: Hit +2% target
        (102.3, 102.5),  # Day 4: Trailing begins
        (101.2, 102.5),  # Day 5: Trail triggered
    ]

    print(f"\nEntry: ${entry:.2f}")
    strategy = optimizer.get_exit_strategy(ticker, 75)
    print(f"Strategy: {optimizer.format_strategy(strategy)}")
    print()

    portion = 1.0
    for day, (current, high) in enumerate(days, 1):
        should_exit, exit_portion, reason = optimizer.should_exit(
            ticker=ticker,
            entry_price=entry,
            current_price=current,
            high_since_entry=high,
            days_held=day,
            portion_remaining=portion,
            signal_strength=75
        )

        ret = ((current / entry) - 1) * 100
        print(f"  Day {day}: ${current:.2f} ({ret:+.1f}%) - {reason}", end="")

        if should_exit:
            print(f" -> EXIT {int(exit_portion*100)}%")
            portion -= exit_portion
            if portion <= 0:
                break
        else:
            print()

    print("\n" + "=" * 70)


if __name__ == '__main__':
    demo()

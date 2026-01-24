"""
Regime-Adaptive Trading Strategy
================================

Updated Jan 7, 2026 based on overnight regime optimization (2-year backtest):

| Regime   | Win Rate | Avg Return | Sharpe | Threshold | Action      |
|----------|----------|------------|--------|-----------|-------------|
| BEAR@70  | 100.0%   | +8.71%     | 1.80   | 70        | TRADE       |
| BULL     | 69.4%    | +1.13%     | 0.43   | 50        | TRADE       |
| BEAR@50  | 76.4%    | +3.58%     | 0.74   | 50        | (balanced)  |
| CHOPPY   | 53.4%    | -0.01%     | 0.00   | any       | SKIP        |

KEY FINDING: CHOPPY regime has ZERO alpha. Skip entirely for +136% Sharpe improvement!

Strategy Modes:
1. AGGRESSIVE: Skip CHOPPY, BEAR@70, BULL@50 -> Expected Sharpe 0.66
2. BALANCED: All regimes with reduced CHOPPY size
3. CONSERVATIVE: Skip CHOPPY, reduce BULL

Expected improvement from AGGRESSIVE: 0.28 -> 0.66 Sharpe (+136%)
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, asdict
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TradingMode(Enum):
    """Trading aggressiveness modes."""
    AGGRESSIVE = "aggressive"   # VOLATILE/BEAR only (highest Sharpe)
    BALANCED = "balanced"       # All regimes, adaptive sizing
    CONSERVATIVE = "conservative"  # Skip BULL, reduce in CHOPPY


class RegimeCategory(Enum):
    """Regime performance categories."""
    HIGH_ALPHA = "high_alpha"     # VOLATILE, BEAR (Sharpe 3-5)
    MODERATE = "moderate"         # CHOPPY (Sharpe 1.25)
    LOW_ALPHA = "low_alpha"       # BULL (Sharpe 1.0)


@dataclass
class RegimeConfig:
    """Configuration for a specific regime."""
    regime: str
    category: RegimeCategory
    win_rate: float
    sharpe: float
    signal_threshold: int
    position_multiplier: float
    max_positions: int
    allow_trading: bool


@dataclass
class AdaptiveSignal:
    """Signal with regime-adaptive adjustments."""
    ticker: str
    raw_strength: float
    adjusted_strength: float
    regime: str
    regime_category: str
    position_multiplier: float
    recommended_size_pct: float
    trade_allowed: bool
    skip_reason: Optional[str] = None


# Regime configurations based on Jan 7, 2026 overnight analysis (2-year backtest)
# Key findings:
# - CHOPPY is actually break-even (Sharpe 0.00), NOT 1.25 as previously thought
# - BEAR at threshold 70 achieves 100% win rate and 1.80 Sharpe
# - Skip CHOPPY entirely: Sharpe improves from 0.28 -> 0.66 (+136%!)
REGIME_CONFIGS = {
    "volatile": RegimeConfig(
        regime="volatile",
        category=RegimeCategory.HIGH_ALPHA,
        win_rate=75.5,
        sharpe=4.86,
        signal_threshold=35,   # Very low - take more signals
        position_multiplier=1.5,  # Larger positions
        max_positions=8,        # More positions
        allow_trading=True
    ),
    "bear": RegimeConfig(
        regime="bear",
        category=RegimeCategory.HIGH_ALPHA,
        win_rate=76.4,  # Updated from Jan 7 analysis (threshold 50)
        sharpe=0.74,    # At threshold 50; at threshold 70 = 1.80 Sharpe!
        signal_threshold=50,  # Base; AGGRESSIVE mode uses 70 for max quality
        position_multiplier=1.3,
        max_positions=7,
        allow_trading=True
    ),
    "choppy": RegimeConfig(
        regime="choppy",
        category=RegimeCategory.LOW_ALPHA,  # Downgraded from MODERATE!
        win_rate=53.4,  # Jan 7 analysis: CHOPPY is near break-even
        sharpe=0.00,    # Zero Sharpe - no alpha in CHOPPY markets
        signal_threshold=80,  # Very high bar - effectively skip most
        position_multiplier=0.3,  # 70% reduction if trading at all
        max_positions=2,
        allow_trading=True  # Can be disabled in AGGRESSIVE mode
    ),
    "bull": RegimeConfig(
        regime="bull",
        category=RegimeCategory.MODERATE,  # Upgraded: BULL is better than CHOPPY
        win_rate=69.4,  # Jan 7 analysis: BULL actually performs okay
        sharpe=0.43,
        signal_threshold=50,   # Jan 7 optimal
        position_multiplier=1.0,  # Full size - BULL is decent
        max_positions=5,
        allow_trading=True
    )
}


class RegimeAdaptiveTrader:
    """
    Regime-adaptive trading strategy.

    Dynamically adjusts thresholds, position sizes, and trading activity
    based on current market regime performance characteristics.
    """

    def __init__(self, mode: TradingMode = TradingMode.BALANCED):
        self.mode = mode
        self.configs = REGIME_CONFIGS.copy()

        # Adjust configs based on mode
        if mode == TradingMode.AGGRESSIVE:
            # Jan 7, 2026: Based on overnight regime optimization:
            # - Skip CHOPPY entirely (0.00 Sharpe -> no alpha)
            # - BEAR at threshold 70 = 100% win rate, 1.80 Sharpe
            # - BULL at threshold 50 is fine (0.43 Sharpe, 69.4% win)
            # Result: Skip CHOPPY improves Sharpe from 0.28 -> 0.66 (+136%)!

            # CHOPPY: Skip entirely - zero alpha, just noise
            self.configs["choppy"] = RegimeConfig(
                regime="choppy",
                category=RegimeCategory.LOW_ALPHA,
                win_rate=53.4,
                sharpe=0.00,
                signal_threshold=100,  # Impossible threshold
                position_multiplier=0.0,
                max_positions=0,
                allow_trading=False  # Skip entirely - no alpha here
            )

            # BEAR: Use optimal threshold 70 for maximum quality
            # At threshold 70: 15 trades, 100% win rate, 8.7% avg return, 1.80 Sharpe!
            self.configs["bear"] = RegimeConfig(
                regime="bear",
                category=RegimeCategory.HIGH_ALPHA,
                win_rate=100.0,  # 100% at threshold 70!
                sharpe=1.80,
                signal_threshold=70,  # Optimal from Jan 7 analysis
                position_multiplier=1.5,  # Full size - highest conviction
                max_positions=5,
                allow_trading=True
            )

            # BULL: Keep at threshold 50 (good performance)
            self.configs["bull"] = RegimeConfig(
                regime="bull",
                category=RegimeCategory.MODERATE,
                win_rate=69.4,
                sharpe=0.43,
                signal_threshold=50,
                position_multiplier=1.0,
                max_positions=5,
                allow_trading=True
            )

        elif mode == TradingMode.CONSERVATIVE:
            # Jan 7, 2026: Skip CHOPPY (zero alpha), reduce BULL
            # Focus on BEAR and VOLATILE (high alpha)

            # CHOPPY: Skip entirely - zero alpha
            self.configs["choppy"] = RegimeConfig(
                regime="choppy",
                category=RegimeCategory.LOW_ALPHA,
                win_rate=53.4,
                sharpe=0.00,
                signal_threshold=100,
                position_multiplier=0.0,
                max_positions=0,
                allow_trading=False  # Skip - no alpha
            )

            # BULL: Reduce size but allow trading (it has 0.43 Sharpe)
            self.configs["bull"] = RegimeConfig(
                regime="bull",
                category=RegimeCategory.MODERATE,
                win_rate=69.4,
                sharpe=0.43,
                signal_threshold=60,  # Higher bar
                position_multiplier=0.5,  # Half size
                max_positions=3,
                allow_trading=True  # Allow but reduced
            )

    def get_regime_config(self, regime: str) -> RegimeConfig:
        """Get configuration for current regime."""
        return self.configs.get(regime.lower(), self.configs["choppy"])

    def should_trade(self, regime: str) -> tuple:
        """Check if trading is allowed in current regime."""
        config = self.get_regime_config(regime)

        if not config.allow_trading:
            return False, f"Trading disabled in {regime.upper()} regime (mode={self.mode.value})"

        return True, None

    def get_threshold(self, regime: str) -> int:
        """Get signal threshold for current regime."""
        return self.get_regime_config(regime).signal_threshold

    def get_position_multiplier(self, regime: str) -> float:
        """Get position size multiplier for current regime."""
        return self.get_regime_config(regime).position_multiplier

    def get_max_positions(self, regime: str) -> int:
        """Get max positions for current regime."""
        return self.get_regime_config(regime).max_positions

    def adjust_signal(self, ticker: str, raw_strength: float,
                     regime: str, base_size_pct: float = 10.0) -> AdaptiveSignal:
        """
        Apply regime-adaptive adjustments to a signal.

        Returns an AdaptiveSignal with adjusted parameters.
        """
        config = self.get_regime_config(regime)

        # Check if trading allowed
        trade_allowed, skip_reason = self.should_trade(regime)

        # Apply regime boost to signal
        if config.category == RegimeCategory.HIGH_ALPHA:
            adjusted_strength = raw_strength * 1.15  # Boost high-alpha signals
        elif config.category == RegimeCategory.LOW_ALPHA:
            adjusted_strength = raw_strength * 0.85  # Penalize low-alpha signals
        else:
            adjusted_strength = raw_strength

        # Check against threshold
        if adjusted_strength < config.signal_threshold:
            trade_allowed = False
            skip_reason = f"Below regime threshold ({adjusted_strength:.1f} < {config.signal_threshold})"

        # Calculate recommended size
        recommended_size = base_size_pct * config.position_multiplier

        return AdaptiveSignal(
            ticker=ticker,
            raw_strength=raw_strength,
            adjusted_strength=round(adjusted_strength, 1),
            regime=regime,
            regime_category=config.category.value,
            position_multiplier=config.position_multiplier,
            recommended_size_pct=round(recommended_size, 1),
            trade_allowed=trade_allowed,
            skip_reason=skip_reason
        )

    def filter_signals(self, signals: List[dict], regime: str) -> List[AdaptiveSignal]:
        """
        Filter and adjust a list of signals based on regime.

        Args:
            signals: List of signal dicts with 'ticker' and 'strength' keys
            regime: Current market regime

        Returns:
            List of AdaptiveSignals that pass filtering
        """
        config = self.get_regime_config(regime)

        # Check if any trading allowed
        if not config.allow_trading:
            print(f"[REGIME] Trading DISABLED in {regime.upper()} (mode={self.mode.value})")
            return []

        adjusted = []
        passed = 0
        failed = 0

        for sig in signals:
            ticker = sig.get('ticker', sig.get('Ticker', ''))
            strength = sig.get('strength', sig.get('signal_strength', 0))
            base_size = sig.get('size_pct', 10.0)

            adaptive = self.adjust_signal(ticker, strength, regime, base_size)

            if adaptive.trade_allowed:
                adjusted.append(adaptive)
                passed += 1
            else:
                failed += 1

        # Limit to max positions
        adjusted = adjusted[:config.max_positions]

        print(f"[REGIME] {regime.upper()}: {passed} passed, {failed} filtered, {len(adjusted)} positions allowed")

        return adjusted

    def get_regime_summary(self) -> str:
        """Get summary of current regime configurations."""
        lines = [
            f"\n{'='*60}",
            f"REGIME-ADAPTIVE TRADING (Mode: {self.mode.value.upper()})",
            f"{'='*60}",
            f"{'Regime':<12} {'Category':<12} {'Threshold':<10} {'Size Mult':<10} {'Trading':<10}",
            "-"*60
        ]

        for regime, config in self.configs.items():
            trading = "YES" if config.allow_trading else "NO"
            lines.append(
                f"{regime.upper():<12} {config.category.value:<12} {config.signal_threshold:<10} "
                f"{config.position_multiplier:<10.1f} {trading:<10}"
            )

        lines.append("="*60)
        return "\n".join(lines)


def backtest_regime_strategy(mode: TradingMode = TradingMode.BALANCED) -> Dict:
    """
    Backtest the regime-adaptive strategy.

    Returns performance metrics.
    """
    import json

    # Load extended backtest data
    data_file = "data/research/extended_backtest_results.json"
    if not os.path.exists(data_file):
        print(f"[ERROR] Backtest data not found: {data_file}")
        return {}

    with open(data_file, 'r') as f:
        data = json.load(f)

    trader = RegimeAdaptiveTrader(mode=mode)

    print(trader.get_regime_summary())

    # Simulate filtering based on regime
    regime_perf = data['optimized']['regime_performance']

    total_trades = 0
    weighted_win_rate = 0
    weighted_return = 0
    weighted_sharpe = 0

    print(f"\n{'Regime':<12} {'Trades':<10} {'Win Rate':<12} {'Avg Return':<12} {'Sharpe':<10} {'Included':<10}")
    print("-"*70)

    for rp in regime_perf:
        regime = rp['regime']
        config = trader.get_regime_config(regime)

        if not config.allow_trading:
            print(f"{regime.upper():<12} {rp['trade_count']:<10} {rp['win_rate']:.1f}%{'':<8} "
                  f"{rp['avg_return']:.2f}%{'':<8} {rp['sharpe_ratio']:.2f}{'':<8} NO")
            continue

        # Apply position multiplier to simulate sizing impact
        effective_trades = rp['trade_count'] * config.position_multiplier
        total_trades += effective_trades
        weighted_win_rate += rp['win_rate'] * effective_trades
        weighted_return += rp['avg_return'] * effective_trades
        weighted_sharpe += rp['sharpe_ratio'] * effective_trades

        print(f"{regime.upper():<12} {rp['trade_count']:<10} {rp['win_rate']:.1f}%{'':<8} "
              f"{rp['avg_return']:.2f}%{'':<8} {rp['sharpe_ratio']:.2f}{'':<8} YES (x{config.position_multiplier})")

    print("-"*70)

    if total_trades > 0:
        final_win_rate = weighted_win_rate / total_trades
        final_return = weighted_return / total_trades
        final_sharpe = weighted_sharpe / total_trades

        print(f"\n[REGIME-ADAPTIVE RESULTS ({mode.value.upper()})]")
        print(f"  Effective Trades: {total_trades:.0f}")
        print(f"  Weighted Win Rate: {final_win_rate:.1f}%")
        print(f"  Weighted Avg Return: {final_return:.2f}%")
        print(f"  Weighted Sharpe: {final_sharpe:.2f}")

        # Compare to baseline
        baseline = data['optimized']['metrics']
        print(f"\n[COMPARISON TO BASELINE]")
        print(f"  Win Rate: {baseline['win_rate']:.1f}% -> {final_win_rate:.1f}% ({final_win_rate - baseline['win_rate']:+.1f}%)")
        print(f"  Sharpe:   {baseline['sharpe_ratio']:.2f} -> {final_sharpe:.2f} ({final_sharpe - baseline['sharpe_ratio']:+.2f})")

        return {
            'mode': mode.value,
            'effective_trades': total_trades,
            'win_rate': final_win_rate,
            'avg_return': final_return,
            'sharpe': final_sharpe,
            'improvement_win_rate': final_win_rate - baseline['win_rate'],
            'improvement_sharpe': final_sharpe - baseline['sharpe_ratio']
        }

    return {}


def test_regime_adaptive():
    """Test all three trading modes."""
    print("\n" + "="*70)
    print("REGIME-ADAPTIVE TRADING BACKTEST")
    print("="*70)

    results = {}

    for mode in [TradingMode.BALANCED, TradingMode.AGGRESSIVE, TradingMode.CONSERVATIVE]:
        print(f"\n\n>>> Testing Mode: {mode.value.upper()}")
        result = backtest_regime_strategy(mode)
        results[mode.value] = result

    # Summary
    print("\n\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'Mode':<15} {'Win Rate':<12} {'Avg Return':<12} {'Sharpe':<10} {'vs Baseline':<12}")
    print("-"*70)

    for mode, result in results.items():
        if result:
            print(f"{mode.upper():<15} {result['win_rate']:.1f}%{'':<8} "
                  f"{result['avg_return']:.2f}%{'':<8} {result['sharpe']:.2f}{'':<8} "
                  f"{result['improvement_sharpe']:+.2f} Sharpe")

    return results


if __name__ == "__main__":
    test_regime_adaptive()

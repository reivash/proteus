"""
Vol-Targeted Strategy

Combines signal generation with risk management.
This is the main strategy class for live/backtest use.

Key formula: w[t+1] = clip((σ_target / σ_hat[t+1]) * g_τ(signal), w_min, w_max)
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from .risk_manager import VolTargetingRiskManager, RiskManagerConfig
from .signals import SignalGenerator, MomentumSignal


@dataclass
class StrategyState:
    """Current state of the strategy."""
    position: float = 0.0
    signal: float = 0.0
    filtered_signal: float = 0.0
    vol_estimate: float = 0.0
    vol_scalar: float = 1.0
    step: int = 0


@dataclass
class StrategyConfig:
    """Configuration for vol-targeted strategy."""
    # Risk manager config
    target_vol: float = 0.10
    vol_lookback: int = 20
    ewma_lambda: float = 0.94
    deadband_tau: float = 0.10
    max_leverage: float = 2.0
    min_leverage: float = -2.0
    min_vol_floor: float = 0.05

    # Strategy config
    warmup_period: int = 20  # Days before generating positions
    rebalance_threshold: float = 0.05  # Min position change to rebalance

    def to_risk_config(self) -> RiskManagerConfig:
        """Convert to risk manager config."""
        return RiskManagerConfig(
            target_vol=self.target_vol,
            vol_lookback=self.vol_lookback,
            ewma_lambda=self.ewma_lambda,
            deadband_tau=self.deadband_tau,
            max_leverage=self.max_leverage,
            min_leverage=self.min_leverage,
            min_vol_floor=self.min_vol_floor,
        )


class VolTargetedStrategy:
    """
    Vol-targeted strategy combining signal and risk management.

    This strategy:
    1. Uses any signal generator to get directional view
    2. Applies deadband filtering to reduce whipsaw
    3. Scales position by inverse volatility
    4. Clips to position limits

    The vol-targeting provides mechanical Sharpe improvement of 0.2-0.4.
    The signal provides (hopeful) additional alpha.
    """

    def __init__(
        self,
        signal_generator: Optional[SignalGenerator] = None,
        config: Optional[StrategyConfig] = None
    ):
        self.config = config or StrategyConfig()
        self.signal_generator = signal_generator or MomentumSignal()
        self.risk_manager = VolTargetingRiskManager(self.config.to_risk_config())

        self._state = StrategyState()
        self._returns_history: List[float] = []
        self._prices_history: List[float] = []

    def reset(self):
        """Reset strategy state for new backtest."""
        self._state = StrategyState()
        self._returns_history = []
        self._prices_history = []
        self.risk_manager.reset()
        self.signal_generator.reset()

    def update(
        self,
        price: float,
        return_: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Update strategy with new price data.

        Args:
            price: Current price
            return_: Daily return (computed from prices if None)

        Returns:
            Dictionary with position and diagnostics
        """
        # Track prices
        self._prices_history.append(price)

        # Compute return if not provided
        if return_ is None:
            if len(self._prices_history) >= 2:
                return_ = (price - self._prices_history[-2]) / self._prices_history[-2]
            else:
                return_ = 0.0

        self._returns_history.append(return_)
        self._state.step += 1

        # Update volatility estimate
        vol = self.risk_manager.update_volatility(return_)
        self._state.vol_estimate = vol

        # Warmup period - no positions
        if self._state.step < self.config.warmup_period:
            return self._make_result(position=0.0, reason="warmup")

        # Generate signal
        returns_array = np.array(self._returns_history)
        prices_array = np.array(self._prices_history)

        raw_signal = self.signal_generator.generate(returns_array, prices_array)
        self._state.signal = raw_signal

        # Apply risk management
        filtered_signal = self.risk_manager.apply_deadband(raw_signal)
        self._state.filtered_signal = filtered_signal

        # Compute position
        new_position = self.risk_manager.compute_position(raw_signal)

        # Check rebalance threshold
        position_change = abs(new_position - self._state.position)
        if position_change < self.config.rebalance_threshold:
            return self._make_result(
                position=self._state.position,
                reason="below_rebalance_threshold"
            )

        # Update position
        old_position = self._state.position
        self._state.position = new_position
        self._state.vol_scalar = self.config.target_vol / vol

        return self._make_result(
            position=new_position,
            reason="rebalanced",
            position_change=new_position - old_position
        )

    def _make_result(
        self,
        position: float,
        reason: str,
        position_change: float = 0.0
    ) -> Dict[str, Any]:
        """Create result dictionary."""
        return {
            'position': position,
            'reason': reason,
            'position_change': position_change,
            'signal': self._state.signal,
            'filtered_signal': self._state.filtered_signal,
            'vol_estimate': self._state.vol_estimate,
            'vol_scalar': self._state.vol_scalar,
            'step': self._state.step,
        }

    def get_state(self) -> StrategyState:
        """Get current strategy state."""
        return self._state

    def get_position(self) -> float:
        """Get current position."""
        return self._state.position

    def describe(self) -> str:
        """Describe strategy configuration."""
        return f"""
Vol-Targeted Strategy
=====================
Signal: {self.signal_generator.name()}

Risk Parameters:
  Target Vol: {self.config.target_vol:.1%}
  Vol Lookback: {self.config.vol_lookback} days
  EWMA Lambda: {self.config.ewma_lambda}
  Deadband Tau: {self.config.deadband_tau}
  Max Leverage: {self.config.max_leverage}
  Min Leverage: {self.config.min_leverage}

Strategy Parameters:
  Warmup Period: {self.config.warmup_period} days
  Rebalance Threshold: {self.config.rebalance_threshold}

Expected Performance:
  - Vol targeting adds Sharpe 0.2-0.4 mechanically
  - Signal alpha is additional (if any)
  - Realistic transaction costs will reduce both
"""


class StrategyFactory:
    """Factory for creating strategy variants."""

    @staticmethod
    def create_momentum_strategy(
        lookback: int = 20,
        target_vol: float = 0.10,
        max_leverage: float = 2.0
    ) -> VolTargetedStrategy:
        """Create momentum-based vol-targeted strategy."""
        from .signals import MomentumSignal

        config = StrategyConfig(
            target_vol=target_vol,
            max_leverage=max_leverage,
            min_leverage=-max_leverage,
        )

        signal = MomentumSignal(lookback=lookback)

        return VolTargetedStrategy(signal_generator=signal, config=config)

    @staticmethod
    def create_trend_following_strategy(
        fast_period: int = 10,
        slow_period: int = 50,
        target_vol: float = 0.10
    ) -> VolTargetedStrategy:
        """Create trend-following vol-targeted strategy."""
        from .signals import TrendFollowingSignal

        config = StrategyConfig(
            target_vol=target_vol,
            warmup_period=slow_period,
        )

        signal = TrendFollowingSignal(fast_period=fast_period, slow_period=slow_period)

        return VolTargetedStrategy(signal_generator=signal, config=config)

    @staticmethod
    def create_random_baseline(
        seed: int = 42,
        target_vol: float = 0.10
    ) -> VolTargetedStrategy:
        """Create random signal baseline for null hypothesis testing."""
        from .signals import RandomSignal

        config = StrategyConfig(target_vol=target_vol)
        signal = RandomSignal(seed=seed)

        return VolTargetedStrategy(signal_generator=signal, config=config)

    @staticmethod
    def create_buy_and_hold_baseline(
        target_vol: float = 0.10
    ) -> VolTargetedStrategy:
        """Create buy-and-hold baseline with vol targeting."""
        from .signals import BuyAndHoldSignal

        config = StrategyConfig(
            target_vol=target_vol,
            deadband_tau=0.0,  # No deadband for constant signal
        )
        signal = BuyAndHoldSignal()

        return VolTargetedStrategy(signal_generator=signal, config=config)

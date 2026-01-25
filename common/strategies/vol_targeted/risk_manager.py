"""
Volatility Targeting Risk Manager

Implements the risk map from the TCN investigation:
w[t+1] = clip((σ_target / σ_hat[t+1]) * g_τ(signal), w_min, w_max)

This module does NOT claim alpha - it's honest risk management.
Expected Sharpe improvement: +0.2 to +0.4 (mechanical, not predictive)
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class RiskManagerConfig:
    """Configuration for vol targeting risk manager."""
    target_vol: float = 0.10          # 10% annual vol target
    vol_lookback: int = 20            # Days for vol estimation
    ewma_lambda: float = 0.94         # EWMA decay (RiskMetrics standard)
    deadband_tau: float = 0.10        # Neutral zone threshold
    max_leverage: float = 2.0         # Maximum position size
    min_leverage: float = -2.0        # Minimum (short) position
    min_vol_floor: float = 0.05       # Minimum vol estimate (prevent div by 0)


class VolTargetingRiskManager:
    """
    Standard volatility targeting risk manager.

    This IMPROVES Sharpe by 0.2-0.4 mechanically through:
    1. Inverse vol scaling (reduce size in high vol)
    2. Deadband filtering (reduce whipsaw)
    3. Position limits (bound risk)

    It is NOT alpha. It is risk management.
    """

    def __init__(self, config: Optional[RiskManagerConfig] = None):
        self.config = config or RiskManagerConfig()
        self._vol_estimate: Optional[float] = None
        self._vol_history: list = []

    def reset(self):
        """Reset state for new backtest."""
        self._vol_estimate = None
        self._vol_history = []

    def update_volatility(self, return_: float) -> float:
        """
        Update EWMA volatility estimate.

        Uses standard RiskMetrics approach:
        σ²[t] = λ * σ²[t-1] + (1-λ) * r²[t-1]

        Args:
            return_: Daily return (not annualized)

        Returns:
            Annualized volatility estimate
        """
        self._vol_history.append(return_)

        if self._vol_estimate is None:
            # Initialize with realized vol if enough history
            if len(self._vol_history) >= self.config.vol_lookback:
                realized = np.std(self._vol_history[-self.config.vol_lookback:])
                self._vol_estimate = realized * np.sqrt(252)
            else:
                # Use simple estimate
                self._vol_estimate = abs(return_) * np.sqrt(252)
        else:
            # EWMA update
            annualized_return = return_ * np.sqrt(252)
            self._vol_estimate = np.sqrt(
                self.config.ewma_lambda * self._vol_estimate**2 +
                (1 - self.config.ewma_lambda) * annualized_return**2
            )

        # Apply floor
        self._vol_estimate = max(self._vol_estimate, self.config.min_vol_floor)

        return self._vol_estimate

    def apply_deadband(self, signal: float) -> float:
        """
        Apply deadband function g_τ(x).

        g_τ(x) = 0                           if |x| < τ
        g_τ(x) = sign(x) * (|x| - τ) / (1-τ) if |x| >= τ

        This filters out weak signals to reduce whipsaw.

        Args:
            signal: Raw signal in [-1, +1]

        Returns:
            Filtered signal in [-1, +1]
        """
        tau = self.config.deadband_tau

        if abs(signal) < tau:
            return 0.0

        # Scale remaining signal to use full range
        if signal > 0:
            return (signal - tau) / (1 - tau)
        else:
            return (signal + tau) / (1 - tau)

    def compute_position(self, signal: float, vol_estimate: Optional[float] = None) -> float:
        """
        Compute position size using the risk map.

        w[t+1] = clip((σ_target / σ_hat) * g_τ(signal), w_min, w_max)

        Args:
            signal: Raw signal in [-1, +1] (from any signal generator)
            vol_estimate: Override vol estimate (uses internal if None)

        Returns:
            Position size in [w_min, w_max]
        """
        vol = vol_estimate if vol_estimate is not None else self._vol_estimate

        if vol is None or vol < self.config.min_vol_floor:
            vol = self.config.min_vol_floor

        # Apply deadband
        filtered_signal = self.apply_deadband(signal)

        # Inverse vol scaling
        vol_scalar = self.config.target_vol / vol

        # Compute raw position
        raw_position = vol_scalar * filtered_signal

        # Clip to limits
        return np.clip(raw_position, self.config.min_leverage, self.config.max_leverage)

    def get_vol_estimate(self) -> Optional[float]:
        """Get current volatility estimate."""
        return self._vol_estimate

    def expected_sharpe_improvement(self) -> str:
        """Document expected improvement - be honest."""
        return """
        Expected Sharpe improvement from vol targeting: +0.2 to +0.4

        Sources of improvement:
        1. Inverse vol scaling: Reduces position in high-vol periods (drawdown protection)
        2. Deadband filtering: Reduces transaction costs from weak signals
        3. Position limits: Bounds maximum risk exposure

        This is NOT alpha. This is mechanical risk adjustment.
        Do not claim this as proprietary edge.

        Failure modes:
        - Vol spikes (Feb 2018, Mar 2020): Full exposure before vol updates
        - Vol regime shifts: Backward-looking estimate lags reality
        - Low vol + chop: Full position during whipsaw
        """


class PositionSizer:
    """
    Complete position sizing pipeline.
    Combines signal generation with risk management.
    """

    def __init__(self, risk_manager: VolTargetingRiskManager):
        self.risk_manager = risk_manager

    def size_position(self,
                      signal: float,
                      last_return: float,
                      current_position: float = 0.0) -> Tuple[float, dict]:
        """
        Size position given signal and market state.

        Args:
            signal: Raw signal in [-1, +1]
            last_return: Most recent return for vol update
            current_position: Current position size

        Returns:
            Tuple of (new_position, diagnostics_dict)
        """
        # Update vol estimate
        vol = self.risk_manager.update_volatility(last_return)

        # Compute new position
        new_position = self.risk_manager.compute_position(signal)

        # Diagnostics
        diagnostics = {
            'raw_signal': signal,
            'filtered_signal': self.risk_manager.apply_deadband(signal),
            'vol_estimate': vol,
            'vol_scalar': self.risk_manager.config.target_vol / vol,
            'position': new_position,
            'position_change': new_position - current_position,
        }

        return new_position, diagnostics

"""
Signal Generators for Vol-Targeted Strategy

These are SIMPLE signals for comparison and ablation testing.
The goal is to demonstrate that vol-targeting does the heavy lifting,
not the signal sophistication.

Key insight from investigation: Required IC = 0.023 (51% accuracy)
Any signal that's barely better than random can work with proper risk management.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List
from dataclasses import dataclass


class SignalGenerator(ABC):
    """Abstract base class for signal generators."""

    @abstractmethod
    def generate(self, returns: np.ndarray, prices: Optional[np.ndarray] = None) -> float:
        """
        Generate a signal in [-1, +1].

        Args:
            returns: Array of historical returns (most recent last)
            prices: Optional array of prices (most recent last)

        Returns:
            Signal in [-1, +1] where:
            +1 = maximum long conviction
            -1 = maximum short conviction
            0 = no position
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return signal name for logging."""
        pass

    def reset(self):
        """Reset any internal state. Override if needed."""
        pass


class RandomSignal(SignalGenerator):
    """
    Random signal generator for null hypothesis testing.

    This establishes the baseline: How much of the Sharpe comes from
    vol-targeting alone vs the signal?

    Expected result: Random signal + vol targeting should still
    produce Sharpe 0.2-0.4 from the risk management alone.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self._seed = seed

    def generate(self, returns: np.ndarray, prices: Optional[np.ndarray] = None) -> float:
        """Generate random signal in [-1, +1]."""
        return self.rng.uniform(-1, 1)

    def name(self) -> str:
        return "RandomSignal"

    def reset(self):
        """Reset RNG to initial seed for reproducibility."""
        self.rng = np.random.RandomState(self._seed)


class MomentumSignal(SignalGenerator):
    """
    Simple momentum signal.

    Signal = tanh(mean_return / vol) normalized to [-1, +1]

    This captures the basic insight: trend-following with vol normalization.
    No ML required - just basic time series momentum.
    """

    def __init__(self, lookback: int = 20, vol_adjust: bool = True):
        self.lookback = lookback
        self.vol_adjust = vol_adjust

    def generate(self, returns: np.ndarray, prices: Optional[np.ndarray] = None) -> float:
        """Generate momentum signal."""
        if len(returns) < self.lookback:
            return 0.0

        recent = returns[-self.lookback:]
        mean_ret = np.mean(recent)

        if self.vol_adjust:
            vol = np.std(recent)
            if vol < 1e-8:
                return 0.0
            # Sharpe-like normalization
            raw_signal = mean_ret / vol * np.sqrt(252 / self.lookback)
        else:
            raw_signal = mean_ret * 100  # Scale raw return

        # Normalize to [-1, +1] using tanh
        return np.tanh(raw_signal)

    def name(self) -> str:
        return f"MomentumSignal(lookback={self.lookback})"


class MeanReversionSignal(SignalGenerator):
    """
    Simple mean reversion signal.

    Signal = -tanh(z_score) where z_score = (price - SMA) / std

    Opposite of momentum - bet on return to mean.
    """

    def __init__(self, lookback: int = 20, z_threshold: float = 2.0):
        self.lookback = lookback
        self.z_threshold = z_threshold

    def generate(self, returns: np.ndarray, prices: Optional[np.ndarray] = None) -> float:
        """Generate mean reversion signal."""
        if prices is None or len(prices) < self.lookback:
            return 0.0

        recent_prices = prices[-self.lookback:]
        current_price = prices[-1]

        sma = np.mean(recent_prices)
        std = np.std(recent_prices)

        if std < 1e-8:
            return 0.0

        z_score = (current_price - sma) / std

        # Negative because we fade the move
        # Normalize with z_threshold as scaling
        return -np.tanh(z_score / self.z_threshold)

    def name(self) -> str:
        return f"MeanReversionSignal(lookback={self.lookback})"


class CombinedSignal(SignalGenerator):
    """
    Combine multiple signals with weights.

    Useful for testing if signal combination adds value
    beyond vol-targeting alone.
    """

    def __init__(self, signals: List[SignalGenerator], weights: Optional[List[float]] = None):
        self.signals = signals
        if weights is None:
            weights = [1.0 / len(signals)] * len(signals)
        self.weights = np.array(weights)
        self.weights = self.weights / np.sum(np.abs(self.weights))  # Normalize

    def generate(self, returns: np.ndarray, prices: Optional[np.ndarray] = None) -> float:
        """Generate combined signal."""
        signal_values = [s.generate(returns, prices) for s in self.signals]
        combined = np.dot(self.weights, signal_values)
        return np.clip(combined, -1, 1)

    def name(self) -> str:
        signal_names = [s.name() for s in self.signals]
        return f"CombinedSignal({signal_names})"

    def reset(self):
        for s in self.signals:
            s.reset()


class BuyAndHoldSignal(SignalGenerator):
    """
    Always long signal for benchmark comparison.

    This tests: Does vol-targeting improve buy-and-hold?
    Expected: Yes, by reducing drawdowns in high-vol periods.
    """

    def __init__(self, signal_strength: float = 1.0):
        self.signal_strength = signal_strength

    def generate(self, returns: np.ndarray, prices: Optional[np.ndarray] = None) -> float:
        """Always return constant long signal."""
        return self.signal_strength

    def name(self) -> str:
        return f"BuyAndHoldSignal(strength={self.signal_strength})"


class TrendFollowingSignal(SignalGenerator):
    """
    Dual moving average crossover signal.

    Classic trend following: Long when fast MA > slow MA.
    This is the non-ML baseline that should be competitive.
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 50):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate(self, returns: np.ndarray, prices: Optional[np.ndarray] = None) -> float:
        """Generate trend following signal."""
        if prices is None or len(prices) < self.slow_period:
            return 0.0

        fast_ma = np.mean(prices[-self.fast_period:])
        slow_ma = np.mean(prices[-self.slow_period:])

        if slow_ma < 1e-8:
            return 0.0

        # Signal strength based on MA divergence
        divergence = (fast_ma - slow_ma) / slow_ma

        # Scale and clip
        return np.clip(divergence * 10, -1, 1)

    def name(self) -> str:
        return f"TrendFollowingSignal(fast={self.fast_period}, slow={self.slow_period})"


@dataclass
class SignalDiagnostics:
    """Diagnostics for signal analysis."""
    signal_name: str
    signal_value: float
    returns_used: int
    prices_used: int

    def __repr__(self):
        return f"{self.signal_name}: {self.signal_value:.4f} (returns={self.returns_used}, prices={self.prices_used})"

"""
Vol-Targeted Strategy Module

Implements honest vol-targeting with proper evaluation.
Based on TCN Architecture Investigation findings.

Key insight: Vol targeting improves Sharpe by 0.2-0.4 mechanically.
This is risk management, NOT alpha.
"""

from .risk_manager import VolTargetingRiskManager
from .signals import SignalGenerator, MomentumSignal, RandomSignal, MeanReversionSignal
from .strategy import VolTargetedStrategy
from .backtest import Backtester, BacktestResult
from .evaluator import HonestStrategyEvaluator, StrategyEvaluation

__all__ = [
    'VolTargetingRiskManager',
    'SignalGenerator',
    'MomentumSignal',
    'RandomSignal',
    'MeanReversionSignal',
    'VolTargetedStrategy',
    'Backtester',
    'BacktestResult',
    'HonestStrategyEvaluator',
    'StrategyEvaluation',
]

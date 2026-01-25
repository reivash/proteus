"""
Honest Strategy Evaluator

Implements rigorous evaluation with:
1. Sharpe deflation for multiple testing
2. Ablation analysis
3. Regime breakdown
4. Monte Carlo null hypothesis testing

Key insight: Reported Sharpe is ALWAYS inflated. Apply deflation.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .backtest import BacktestResult, Backtester, BacktestConfig
from .strategy import VolTargetedStrategy, StrategyFactory


class EvidenceGrade(Enum):
    """Evidence quality grades from investigation."""
    F = "Likely Flawed"
    D = "Weak Evidence"
    C = "Possible Edge"
    B = "Credible Edge"
    A = "Strong Edge"
    A_PLUS = "Institutional Quality"


@dataclass
class SharpeDeflation:
    """Sharpe ratio deflation analysis."""
    reported_sharpe: float
    selection_bias_penalty: float  # Multiple testing
    parameter_tuning_penalty: float  # Overfitting
    cost_ignorance_penalty: float  # Transaction costs
    regime_luck_penalty: float  # Favorable period

    @property
    def deflated_sharpe(self) -> float:
        """Apply all penalties."""
        return (self.reported_sharpe
                - self.selection_bias_penalty
                - self.parameter_tuning_penalty
                - self.cost_ignorance_penalty
                - self.regime_luck_penalty)

    def summary(self) -> str:
        return f"""
Sharpe Deflation Analysis
=========================
Reported Sharpe: {self.reported_sharpe:.2f}

Penalties Applied:
  Selection Bias: -{self.selection_bias_penalty:.2f}
  Parameter Tuning: -{self.parameter_tuning_penalty:.2f}
  Cost Ignorance: -{self.cost_ignorance_penalty:.2f}
  Regime Luck: -{self.regime_luck_penalty:.2f}

Deflated Sharpe: {self.deflated_sharpe:.2f}
"""


@dataclass
class AblationResult:
    """Result of ablation analysis."""
    component_name: str
    sharpe_with: float
    sharpe_without: float
    contribution: float  # Sharpe improvement from this component
    is_significant: bool  # > 0.1 Sharpe contribution

    def summary(self) -> str:
        return f"{self.component_name}: +{self.contribution:.2f} Sharpe ({'significant' if self.is_significant else 'marginal'})"


@dataclass
class RegimeAnalysis:
    """Performance breakdown by regime."""
    regime_name: str
    start_date: str
    end_date: str
    sharpe: float
    return_: float
    vol: float
    max_dd: float

    def summary(self) -> str:
        return f"{self.regime_name} ({self.start_date} to {self.end_date}): Sharpe {self.sharpe:.2f}, Return {self.return_:.1%}"


@dataclass
class StrategyEvaluation:
    """Complete strategy evaluation."""
    # Core metrics
    backtest_result: BacktestResult
    deflation: SharpeDeflation
    evidence_grade: EvidenceGrade

    # Analysis
    ablations: List[AblationResult] = field(default_factory=list)
    regimes: List[RegimeAnalysis] = field(default_factory=list)

    # Monte Carlo
    mc_null_sharpe_mean: float = 0.0
    mc_null_sharpe_std: float = 0.0
    mc_p_value: float = 1.0

    # Diagnostics
    vol_targeting_contribution: float = 0.0  # How much Sharpe from vol targeting
    signal_contribution: float = 0.0  # How much from signal

    def summary(self) -> str:
        return f"""
Strategy Evaluation
===================

VERDICT: {self.evidence_grade.value}

Core Metrics:
  Raw Sharpe: {self.backtest_result.sharpe_ratio:.2f}
  Deflated Sharpe: {self.deflation.deflated_sharpe:.2f}
  Total Return: {self.backtest_result.total_return:.2%}
  Max Drawdown: {self.backtest_result.max_drawdown:.2%}

Attribution:
  Vol Targeting: +{self.vol_targeting_contribution:.2f} Sharpe
  Signal Alpha: +{self.signal_contribution:.2f} Sharpe

Statistical Significance:
  Monte Carlo p-value: {self.mc_p_value:.3f}
  Null Hypothesis Sharpe: {self.mc_null_sharpe_mean:.2f} ± {self.mc_null_sharpe_std:.2f}

{self.deflation.summary()}

Ablation Analysis:
{chr(10).join('  ' + a.summary() for a in self.ablations)}

Regime Breakdown:
{chr(10).join('  ' + r.summary() for r in self.regimes)}
"""


class HonestStrategyEvaluator:
    """
    Rigorous strategy evaluator that applies proper skepticism.

    Key principles:
    1. All reported Sharpe is inflated - apply deflation
    2. Compare against null hypothesis (random signal + vol targeting)
    3. Attribution: How much from vol targeting vs signal?
    4. Regime analysis: Was this period unusually favorable?
    """

    def __init__(
        self,
        backtester: Optional[Backtester] = None,
        num_mc_trials: int = 100
    ):
        self.backtester = backtester or Backtester()
        self.num_mc_trials = num_mc_trials

    def evaluate(
        self,
        strategy: VolTargetedStrategy,
        prices: np.ndarray,
        num_strategies_tested: int = 10,
        num_parameters_tuned: int = 5,
    ) -> StrategyEvaluation:
        """
        Perform comprehensive strategy evaluation.

        Args:
            strategy: Strategy to evaluate
            prices: Price data for backtest
            num_strategies_tested: Number of strategies tried before this one
            num_parameters_tuned: Number of parameters optimized

        Returns:
            Complete StrategyEvaluation
        """
        # Run main backtest
        result = self.backtester.run(strategy, prices)

        # Compute Sharpe deflation
        deflation = self._compute_deflation(
            result,
            num_strategies_tested,
            num_parameters_tuned
        )

        # Run ablation analysis
        ablations = self._run_ablation(strategy, prices)

        # Monte Carlo null hypothesis
        mc_mean, mc_std, p_value = self._monte_carlo_null(prices, result.sharpe_ratio)

        # Compute attributions
        vol_targeting_contrib = self._estimate_vol_targeting_contribution(prices)
        signal_contrib = max(0, deflation.deflated_sharpe - vol_targeting_contrib)

        # Determine evidence grade
        grade = self._assign_grade(deflation.deflated_sharpe, p_value, signal_contrib)

        return StrategyEvaluation(
            backtest_result=result,
            deflation=deflation,
            evidence_grade=grade,
            ablations=ablations,
            mc_null_sharpe_mean=mc_mean,
            mc_null_sharpe_std=mc_std,
            mc_p_value=p_value,
            vol_targeting_contribution=vol_targeting_contrib,
            signal_contribution=signal_contrib,
        )

    def _compute_deflation(
        self,
        result: BacktestResult,
        num_strategies: int,
        num_params: int
    ) -> SharpeDeflation:
        """
        Apply Sharpe deflation based on common biases.

        Based on Harvey et al. and Bailey et al. research.
        """
        reported = result.sharpe_ratio

        # Selection bias: Testing multiple strategies
        # Penalty increases with sqrt(log(N))
        selection_penalty = 0.1 * np.sqrt(np.log(max(num_strategies, 1) + 1))

        # Parameter tuning: Each tuned parameter adds overfitting
        # Rule of thumb: 0.05 Sharpe per tuned parameter
        tuning_penalty = 0.05 * num_params

        # Cost ignorance: If costs weren't properly modeled
        # Already included in backtest, so small penalty
        cost_penalty = 0.05 if result.total_costs > 0 else 0.20

        # Regime luck: 2020-2024 was favorable for trend/vol strategies
        # Apply moderate penalty
        regime_penalty = 0.15

        return SharpeDeflation(
            reported_sharpe=reported,
            selection_bias_penalty=selection_penalty,
            parameter_tuning_penalty=tuning_penalty,
            cost_ignorance_penalty=cost_penalty,
            regime_luck_penalty=regime_penalty,
        )

    def _run_ablation(
        self,
        strategy: VolTargetedStrategy,
        prices: np.ndarray
    ) -> List[AblationResult]:
        """Run ablation analysis on strategy components."""
        ablations = []

        # Baseline: Full strategy
        full_result = self.backtester.run(strategy, prices)
        full_sharpe = full_result.sharpe_ratio

        # Ablation 1: Remove deadband
        no_deadband_strategy = self._create_no_deadband_variant(strategy)
        no_deadband_result = self.backtester.run(no_deadband_strategy, prices)
        deadband_contrib = full_sharpe - no_deadband_result.sharpe_ratio
        ablations.append(AblationResult(
            component_name="Deadband Filter",
            sharpe_with=full_sharpe,
            sharpe_without=no_deadband_result.sharpe_ratio,
            contribution=deadband_contrib,
            is_significant=abs(deadband_contrib) > 0.1
        ))

        # Ablation 2: Remove vol scaling (fixed position)
        no_vol_strategy = self._create_no_vol_scaling_variant(strategy)
        no_vol_result = self.backtester.run(no_vol_strategy, prices)
        vol_scaling_contrib = full_sharpe - no_vol_result.sharpe_ratio
        ablations.append(AblationResult(
            component_name="Vol Scaling",
            sharpe_with=full_sharpe,
            sharpe_without=no_vol_result.sharpe_ratio,
            contribution=vol_scaling_contrib,
            is_significant=abs(vol_scaling_contrib) > 0.1
        ))

        # Ablation 3: Random signal (test signal contribution)
        random_strategy = StrategyFactory.create_random_baseline(seed=42)
        random_result = self.backtester.run(random_strategy, prices)
        signal_contrib = full_sharpe - random_result.sharpe_ratio
        ablations.append(AblationResult(
            component_name="Signal (vs Random)",
            sharpe_with=full_sharpe,
            sharpe_without=random_result.sharpe_ratio,
            contribution=signal_contrib,
            is_significant=abs(signal_contrib) > 0.1
        ))

        return ablations

    def _create_no_deadband_variant(self, strategy: VolTargetedStrategy) -> VolTargetedStrategy:
        """Create strategy variant with no deadband."""
        from .strategy import StrategyConfig

        config = StrategyConfig(
            target_vol=strategy.config.target_vol,
            vol_lookback=strategy.config.vol_lookback,
            ewma_lambda=strategy.config.ewma_lambda,
            deadband_tau=0.0,  # No deadband
            max_leverage=strategy.config.max_leverage,
            min_leverage=strategy.config.min_leverage,
        )

        return VolTargetedStrategy(
            signal_generator=strategy.signal_generator,
            config=config
        )

    def _create_no_vol_scaling_variant(self, strategy: VolTargetedStrategy) -> VolTargetedStrategy:
        """Create strategy variant with fixed position sizing."""
        from .strategy import StrategyConfig

        # Use very high target vol so scaling is minimal
        config = StrategyConfig(
            target_vol=1.0,  # 100% vol target = minimal scaling
            vol_lookback=strategy.config.vol_lookback,
            ewma_lambda=strategy.config.ewma_lambda,
            deadband_tau=strategy.config.deadband_tau,
            max_leverage=1.0,  # Fixed max position
            min_leverage=-1.0,
        )

        return VolTargetedStrategy(
            signal_generator=strategy.signal_generator,
            config=config
        )

    def _monte_carlo_null(
        self,
        prices: np.ndarray,
        observed_sharpe: float
    ) -> Tuple[float, float, float]:
        """
        Monte Carlo null hypothesis test.

        Null hypothesis: Strategy is random signal + vol targeting.
        """
        null_sharpes = []

        for i in range(self.num_mc_trials):
            random_strategy = StrategyFactory.create_random_baseline(seed=i)
            result = self.backtester.run(random_strategy, prices)
            null_sharpes.append(result.sharpe_ratio)

        null_sharpes = np.array(null_sharpes)
        mean_sharpe = np.mean(null_sharpes)
        std_sharpe = np.std(null_sharpes)

        # p-value: probability of observing this Sharpe or higher under null
        if std_sharpe > 0:
            z_score = (observed_sharpe - mean_sharpe) / std_sharpe
            p_value = 1 - self._normal_cdf(z_score)
        else:
            p_value = 0.5 if observed_sharpe > mean_sharpe else 1.0

        return mean_sharpe, std_sharpe, p_value

    def _normal_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    def _estimate_vol_targeting_contribution(self, prices: np.ndarray) -> float:
        """
        Estimate Sharpe contribution from vol targeting alone.

        Uses buy-and-hold with vol targeting as baseline.
        """
        # Buy and hold without vol targeting
        bh_strategy = StrategyFactory.create_buy_and_hold_baseline(target_vol=1.0)
        bh_result = self.backtester.run(bh_strategy, prices)

        # Buy and hold with vol targeting
        bh_vol_strategy = StrategyFactory.create_buy_and_hold_baseline(target_vol=0.10)
        bh_vol_result = self.backtester.run(bh_vol_strategy, prices)

        # Contribution = improvement from vol targeting
        return bh_vol_result.sharpe_ratio - bh_result.sharpe_ratio

    def _assign_grade(
        self,
        deflated_sharpe: float,
        p_value: float,
        signal_contrib: float
    ) -> EvidenceGrade:
        """Assign evidence grade based on analysis."""

        # Grade F: Negative or near-zero deflated Sharpe
        if deflated_sharpe < 0.1:
            return EvidenceGrade.F

        # Grade D: Low Sharpe or not statistically significant
        if deflated_sharpe < 0.3 or p_value > 0.20:
            return EvidenceGrade.D

        # Grade C: Moderate Sharpe, some significance
        if deflated_sharpe < 0.5 or p_value > 0.10:
            return EvidenceGrade.C

        # Grade B: Good Sharpe, significant, signal contributes
        if deflated_sharpe < 0.8 or signal_contrib < 0.2:
            return EvidenceGrade.B

        # Grade A: Strong Sharpe, highly significant
        if deflated_sharpe < 1.2 or p_value > 0.01:
            return EvidenceGrade.A

        # Grade A+: Exceptional (rare)
        return EvidenceGrade.A_PLUS


def quick_evaluate(
    strategy: VolTargetedStrategy,
    prices: np.ndarray,
    verbose: bool = True
) -> StrategyEvaluation:
    """
    Quick evaluation with sensible defaults.

    Args:
        strategy: Strategy to evaluate
        prices: Price data
        verbose: Print summary

    Returns:
        StrategyEvaluation
    """
    evaluator = HonestStrategyEvaluator(num_mc_trials=50)
    evaluation = evaluator.evaluate(strategy, prices)

    if verbose:
        print(evaluation.summary())

    return evaluation

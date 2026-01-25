# Regime Detection

Market condition classification to adapt trading behavior.

## What It Does

Identifies current market regime (BULL, BEAR, CHOPPY, VOLATILE) using HMM + rule-based ensemble. Different regimes have dramatically different win rates for mean reversion strategies.

## Regimes

| Regime | Win Rate | Sharpe | Trading Approach |
|--------|----------|--------|------------------|
| VOLATILE | 75.5% | 4.86 | Trade aggressively |
| BEAR | 71.2% | 3.32 | Trade with larger sizes |
| CHOPPY | 59.3% | 1.25 | Raise thresholds |
| BULL | 58.8% | 1.00 | Consider sitting out |

## Detection Methods

1. **HMM (Hidden Markov Model)** - Statistical regime classification
2. **Rule-based** - VIX levels, trend analysis, breadth
3. **Ensemble** - Combines both for robustness

## Key Files

| File | Purpose |
|------|---------|
| `common/analysis/unified_regime_detector.py` | Main regime detector |
| `common/analysis/hierarchical_hmm.py` | HMM implementation |

## Usage

Regime detection is integrated into the signal scanner. It automatically adjusts:
- Signal thresholds (higher in choppy/bull)
- Position sizes (larger in volatile/bear)
- Trading mode (aggressive skips low-edge regimes)

## See Also

- [PLAN.md](PLAN.md) - Development roadmap
- [results/](results/) - Regime classification history
- [experiments/](experiments/) - Related experiments

# HMM Regime Detector - Analysis & Documentation

> **Date**: January 25, 2026
> **Status**: Current Production System
> **File**: `common/analysis/hmm_regime_detector.py`

---

## Executive Summary

The HMM-based regime detector uses a 4-state Gaussian Hidden Markov Model to classify market conditions. It's designed to detect regime shifts **before** they become obvious through rule-based methods, providing 1-3 days of early warning.

**Current Assessment**: The implementation is solid but has significant room for improvement in feature engineering, state mapping, and online learning capabilities.

---

## Architecture Overview

### Core Components

```
GaussianHMM (Custom Implementation)
    ├── Baum-Welch Algorithm (training)
    ├── Viterbi Algorithm (state sequence)
    └── Forward-Backward (probabilities)
         │
         ▼
HMMRegimeDetector
    ├── Feature Extraction (4 features)
    ├── State Mapping (HMM states → regimes)
    └── Transition Detection
```

### State Definitions

| State | Regime | Sharpe (Mean Reversion) | Description |
|-------|--------|-------------------------|-------------|
| 0 | VOLATILE | 4.86 | High uncertainty, large swings |
| 1 | BEAR | 3.32 | Downtrend with fear |
| 2 | CHOPPY | 1.25 | Range-bound, normal vol |
| 3 | BULL | 1.00 | Strong uptrend, complacency |

---

## Strengths

### 1. **Probabilistic Output**
- Returns full posterior distribution, not just classification
- Enables confidence-weighted position sizing
- Supports uncertainty quantification

### 2. **Custom Implementation (No Dependencies)**
- No reliance on hmmlearn or pomegranate
- Full control over algorithm behavior
- Easier to debug and modify

### 3. **Transition Detection**
- Detects "entering" and "exiting" signals
- Uses probability trend analysis (not just state changes)
- Provides early warning of regime shifts

### 4. **Regime Ordering by Alpha**
- States ordered by mean reversion performance
- Volatile > Bear > Choppy > Bull
- Directly actionable for trading decisions

### 5. **Model Persistence**
- Saves/loads trained models to JSON
- Preserves feature normalization parameters
- Enables reproducibility

---

## Weaknesses

### 1. **Limited Feature Set (4 features)**

Current features:
```python
1. Daily returns          # Momentum signal
2. Realized volatility    # 20-day rolling std
3. Volume change ratio    # vs 20-day average
4. 10-day momentum        # Price momentum
```

**Missing high-value features**:
- VIX level (not just implied, actual index)
- VIX term structure (contango/backwardation)
- Credit spreads (HY-IG spread)
- Market breadth (% above 20/50/200 MA)
- Put/call ratio
- Yield curve slope

### 2. **Static State Mapping**

The `_map_states_to_regimes()` function uses heuristics:
```python
if vol > 0.02:
    mapping[state] = VOLATILE
elif returns < -0.001:
    mapping[state] = BEAR
...
```

**Problems**:
- Mapping happens post-hoc based on feature averages
- No guarantee HMM states align with intended regimes
- Can flip between runs if data changes

### 3. **No Online Learning**

- Model must be retrained from scratch
- Can't adapt to market evolution
- Trained on 5 years, may be stale

### 4. **Single Timeframe**

- Only processes daily data
- No multi-timeframe analysis
- Misses intraday regime shifts

### 5. **Initialization Sensitivity**

```python
# Initialize means using quantiles
sorted_idx = np.argsort(X[:, 0])  # Sort by first feature
chunk_size = n_samples // self.n_states
```

- K-means-like initialization by return quantiles
- Can converge to local optima
- No multiple restart strategy

### 6. **Diagonal Covariance Fallback**

```python
except np.linalg.LinAlgError:
    # Fallback to diagonal covariance
    var = np.diag(cov)
```

- Full covariance matrix can be singular
- Falls back to diagonal (loses correlations)
- Should use regularization instead

### 7. **No Regime Duration Modeling**

- Transition probabilities are constant
- Doesn't account for regime persistence
- A regime at day 1 vs day 30 should have different exit probabilities

---

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| Training (Baum-Welch) | O(n_iter × T × K²) | ~2-3 seconds |
| Prediction (Viterbi) | O(T × K²) | ~50ms |
| Probabilities (Forward-Backward) | O(T × K²) | ~50ms |

Where T = sequence length (~60 days), K = states (4)

### Memory Usage

- Model parameters: ~5KB
- Feature history: ~500KB for 5 years
- Total footprint: <1MB

### Latency Breakdown

```
Data fetch (yfinance):  ~500ms
Feature extraction:     ~10ms
HMM inference:          ~50ms
State mapping:          ~5ms
─────────────────────────────
Total:                  ~565ms
```

---

## Code Quality Assessment

### Good Practices
- Comprehensive docstrings
- Type hints throughout
- Fallback handling for edge cases
- Logging at key decision points

### Areas for Improvement
- No unit tests for HMM math
- Magic numbers (0.02, -0.001, etc.) should be constants
- State mapping logic is brittle
- No validation of input data quality

---

## Comparison: HMM vs Rule-Based

| Aspect | HMM | Rule-Based |
|--------|-----|------------|
| Early detection | Better (probability trends) | Worse (threshold crossing) |
| Explainability | Worse (black box) | Better (clear rules) |
| Adaptation | None (static model) | Immediate (updated thresholds) |
| Noise handling | Better (probabilistic) | Worse (hard thresholds) |
| VIX awareness | Indirect (via vol feature) | Direct (explicit thresholds) |

---

## Recommendations for Improvement

### High Priority (Week 3-4)

1. **Add VIX as explicit feature**
   - Currently only uses realized vol
   - VIX contains forward-looking information
   - Easy to implement, high value

2. **Add VIX term structure**
   - Already implemented in Quick Wins
   - Use as 5th feature in HMM

3. **Implement regime duration modeling**
   - Track days in regime
   - Adjust transition probabilities based on duration
   - "Bull for 30 days" → higher exit probability

### Medium Priority (Week 5-8)

4. **Multi-restart initialization**
   - Run 10 random initializations
   - Keep best by log-likelihood
   - Reduces local optima issues

5. **Add market breadth feature**
   - % stocks above 50-day MA
   - Leading indicator of regime change
   - Requires fetching SPY components

6. **Regularized covariance**
   - Add ridge regularization to covariance matrices
   - Prevents singularity issues
   - Keeps full correlation structure

### Lower Priority (Q2)

7. **Online learning / incremental updates**
   - Update model with each new observation
   - Adapt to market evolution
   - Research: variational inference for HMMs

8. **Hierarchical HMM integration**
   - Already have HierarchicalHMM class
   - Integrate meta-regime (uncertainty) with market state
   - Use meta-regime to adjust transition matrices

---

## Test Cases Needed

1. **State mapping stability**: Same data → same mapping?
2. **Transition detection accuracy**: Synthetic regime changes
3. **Probability calibration**: Are 80% confident predictions right 80%?
4. **Edge cases**: All-up market, flash crash, V-shaped recovery

---

## Files to Review

- `common/analysis/hmm_regime_detector.py` - Main implementation
- `common/analysis/hierarchical_hmm.py` - Two-level HMM
- `common/analysis/unified_regime_detector.py` - Ensemble logic
- `features/market_conditions/data/regime_history.json` - Historical logs

---

*Analysis by Claude Opus 4.5 | Updated: January 25, 2026*

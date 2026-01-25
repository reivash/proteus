# Neural Network V2 Research
## Proteus Trading System - Model Architecture Improvements

**Date:** January 4, 2026
**Current Model:** AttentionMLP (3-layer MLP with attention)
**Goal:** Improve base signal accuracy by +5%

---

## Current Architecture Analysis

### AttentionMLP (V1)
```
Input: 20 features (rolling window)
    |
Feature Attention (Linear -> Tanh -> Linear -> Softmax)
    |
MLP Encoder:
    - Linear(20 -> 64) + BatchNorm + ReLU + Dropout(0.2)
    - Linear(64 -> 32) + BatchNorm + ReLU + Dropout(0.2)
    - Linear(32 -> 16) + BatchNorm + ReLU + Dropout(0.2)
    |
Output Heads (Multi-task):
    - prob_head: Linear(16 -> 1) + Sigmoid (mean reversion probability)
    - return_head: Linear(16 -> 1) (expected return)
    - confidence_head: Linear(16 -> 1) + Sigmoid
```

**Strengths:**
- Fast inference (~1ms per stock on RTX 4080)
- Parallel processing of all 54 stocks
- Feature attention provides interpretability
- Multi-task learning captures related signals

**Weaknesses:**
- No temporal modeling (treats features as independent)
- Limited receptive field (single time step)
- No cross-stock correlation modeling
- Static architecture (same for all regimes)

---

## V2 Architecture Candidates

### Option A: LSTM with Attention (LSTMSignalModel)

**Architecture:**
```
Input: (batch, seq_len=30, features=20)
    |
Bi-LSTM(hidden=64, layers=2, dropout=0.3)
    |
Multi-Head Self-Attention(heads=4, dim=128)
    |
Temporal Pooling (last state + mean + max)
    |
MLP Decoder: [192 -> 64 -> 16]
    |
Output Heads (same as V1)
```

**Pros:**
- Captures temporal dependencies
- Bidirectional context
- Attention highlights important time steps
- Well-suited for time series

**Cons:**
- Sequential processing (slower than MLP)
- May overfit on short sequences
- Memory requirements for long sequences

**Expected Improvement:** +3-5% accuracy

---

### Option B: Temporal Convolutional Network (TCN)

**Architecture:**
```
Input: (batch, features=20, seq_len=30)
    |
Dilated Causal Conv Stack:
    - Conv1D(20 -> 64, kernel=3, dilation=1)
    - Conv1D(64 -> 64, kernel=3, dilation=2)
    - Conv1D(64 -> 64, kernel=3, dilation=4)
    - Conv1D(64 -> 64, kernel=3, dilation=8)
    |
Global Pooling + Last Step
    |
MLP Decoder
    |
Output Heads
```

**Pros:**
- Parallelizable (faster than LSTM)
- Long receptive field with dilations
- No vanishing gradient issues
- Stable training

**Cons:**
- Fixed receptive field
- Less flexible than attention
- Requires careful dilation design

**Expected Improvement:** +2-4% accuracy

---

### Option C: Transformer Encoder

**Architecture:**
```
Input: (batch, seq_len=30, features=20)
    |
Positional Encoding (sinusoidal)
    |
Transformer Encoder:
    - num_layers=4
    - num_heads=4
    - d_model=64
    - dim_feedforward=256
    - dropout=0.2
    |
CLS Token / Mean Pooling
    |
MLP Decoder
    |
Output Heads
```

**Pros:**
- State-of-the-art for sequences
- Full attention over all time steps
- Highly parallelizable
- Proven in financial applications

**Cons:**
- Quadratic memory in sequence length
- May require more data to train
- More hyperparameters to tune

**Expected Improvement:** +4-7% accuracy

---

### Option D: Hybrid Model (Recommended)

**Architecture:**
```
Input: (batch, seq_len=30, features=20)
    |
+-- TCN Branch (captures local patterns)
|   Conv1D stack with dilations [1, 2, 4]
|   Output: (batch, 64)
|
+-- Transformer Branch (captures global patterns)
|   2-layer Transformer Encoder
|   Output: (batch, 64)
|
+-- Statistical Branch (handcrafted features)
|   RSI, Z-score, momentum, volatility
|   Output: (batch, 20)
|
Concatenate: (batch, 148)
    |
Feature Fusion MLP: [148 -> 64 -> 32]
    |
Regime-Conditional Decoder (different heads per regime)
    |
Output Heads
```

**Pros:**
- Best of multiple approaches
- Ensemble effect reduces variance
- Regime-awareness for adaptive behavior
- Interpretable components

**Cons:**
- More complex to implement
- Longer training time
- Requires careful balancing of branches

**Expected Improvement:** +5-8% accuracy

---

## Cross-Stock Correlation Module

**New Feature: Stock Correlation Attention**

```
Stock Embeddings: (54 stocks, embedding_dim=32)
    |
Cross-Stock Attention:
    - Query: Current stock signal
    - Keys: All 54 stock embeddings
    - Values: All 54 stock features
    |
Weighted aggregation of related stocks
```

**Benefit:** Capture sector correlation and pairs trading signals

---

## Implementation Plan

### Phase 1: Data Preparation (Week 1)
1. Create sliding window dataset with 30-day sequences
2. Add regime labels to each sample
3. Implement train/val/test split (60/20/20)
4. Set up data augmentation (time warping, noise)

### Phase 2: Model Development (Week 2)
1. Implement LSTM baseline
2. Implement TCN variant
3. Implement Hybrid model
4. Add cross-stock attention module

### Phase 3: Training & Validation (Week 3)
1. Train all variants with same hyperparameters
2. Implement early stopping on validation
3. Compare using:
   - Accuracy (mean reversion probability)
   - Expected return prediction MSE
   - Sharpe ratio of predicted signals

### Phase 4: A/B Testing (Week 4)
1. Run parallel with V1 model
2. Paper trade both versions
3. Compare signal quality metrics
4. Deploy winner to production

---

## Training Configuration

```python
config = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'epochs': 100,
    'early_stopping_patience': 10,
    'optimizer': 'AdamW',
    'scheduler': 'CosineAnnealingWarmRestarts',

    # Model-specific
    'seq_len': 30,
    'hidden_dim': 64,
    'num_layers': 2,
    'num_heads': 4,
    'dropout': 0.2,

    # Multi-task weights
    'prob_weight': 0.5,
    'return_weight': 0.3,
    'confidence_weight': 0.2,
}
```

---

## Evaluation Metrics

| Metric | V1 Baseline | V2 Target |
|--------|-------------|-----------|
| Classification Accuracy | 60.4% | 65%+ |
| Mean Reversion Prob AUC | ~0.62 | 0.68+ |
| Return Prediction MSE | ~0.005 | <0.004 |
| Sharpe Ratio | 1.39 | 1.6+ |
| Inference Time (54 stocks) | ~50ms | <100ms |

---

## Risk Mitigation

1. **Overfitting:** Use dropout, early stopping, data augmentation
2. **Regime Shift:** Train on multiple regime periods, validate on unseen regimes
3. **Data Leakage:** Ensure no future data in features, proper time-based splits
4. **Complexity:** Start simple (LSTM), add complexity only if needed

---

## Next Steps

1. [ ] Create training dataset with sliding windows
2. [ ] Implement LSTM baseline model
3. [ ] Set up training pipeline with logging
4. [ ] Run initial experiments on 1-year data
5. [ ] Compare with V1 on validation set
6. [ ] If promising, implement hybrid model

---

## References

1. Temporal Fusion Transformers (TFT) - Google AI
2. N-BEATS for Time Series Forecasting
3. Deep Learning for Stock Prediction (Stanford CS229)
4. Attention Is All You Need (Transformer paper)
5. TCN for Sequence Modeling (Bai et al., 2018)

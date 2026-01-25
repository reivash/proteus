# Smart Scanner V2: Daily Signal Generation Pipeline

> **Project Status**: Production (Complete)
> **Version**: 2.0
> **Last Updated**: January 2026
> **Backtest Performance**: 60.4% win rate, +0.82% avg return

---

## Overview

### Problem Statement
Generating reliable trading signals requires synthesizing multiple data sources, ML models, market regime awareness, and risk management rules. A manual process is error-prone and inconsistent.

### Objectives
1. **Automated Daily Scanning**: Generate signals for 54 validated stocks every trading day
2. **ML-Powered Signals**: Use LSTM/Transformer ensemble for signal strength
3. **Regime-Aware**: Adjust thresholds and sizing based on market conditions
4. **Risk-Managed**: Enforce position limits, sector limits, and earnings avoidance

### Key Achievements
- 54-stock production universe with tier-based classification
- Hybrid ML ensemble (LSTM + Transformer + MLP) with 60.4% win rate
- Regime-adaptive thresholds (60-75 based on market conditions)
- 88 validated signal modifiers (55 boosts + 33 penalties)
- Full integration with VirtualWallet for paper trading

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SMART SCANNER V2 PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  DATA    │   │  REGIME  │   │   ML     │   │  SIGNAL  │   │ POSITION │ │
│  │ FETCHERS │──▶│ DETECTOR │──▶│  MODELS  │──▶│ ADJUSTOR │──▶│  SIZER   │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│       │              │              │              │              │         │
│       ▼              ▼              ▼              ▼              ▼         │
│  - Yahoo Fin    - HMM-based    - LSTM        - 55 boosts    - Kelly       │
│  - FRED Macro   - VIX rules    - Transformer - 33 penalties - Regime adj  │
│  - Earnings     - Bear early   - MLP         - Tier mults   - Sector lim  │
│  - Sentiment      warning      - Ensemble                   - Heat limit  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                              OUTPUT: SignalV2                                │
│  ticker, raw_strength, adjusted_strength, tier, regime, size_pct,          │
│  profit_targets, stop_loss, max_hold_days, near_earnings, skip_trade       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Signal Flow
1. **Data Collection**: Fetch OHLCV, fundamentals, sentiment for all tickers
2. **Regime Detection**: Determine BULL/BEAR/CHOPPY/VOLATILE
3. **ML Inference**: Run hybrid ensemble to get raw signal strength
4. **Signal Adjustment**: Apply 88 modifiers based on patterns
5. **Position Sizing**: Calculate position size with risk management
6. **Exit Strategy**: Set profit targets, stop loss, max hold days
7. **Output**: Generate SignalV2 objects for each stock

---

## Implementation Phases

> **Note**: This project is COMPLETE. All checkboxes below are checked to document what was built.

---

### Phase 1: Data Infrastructure
**Goal**: Reliable data fetching with caching

#### 1.1 Core Data Fetchers
- [x] **1.1.1 Yahoo Finance fetcher**
  **DoD**:
  - [x] Fetches OHLCV for any ticker
  - [x] Rate limiting (200ms between requests)
  - [x] Error handling for delisted symbols
  - [x] Returns clean DataFrame

- [x] **1.1.2 Market context fetcher**
  **DoD**:
  - [x] VIX current level
  - [x] SPY, QQQ, IWM performance
  - [x] Sector ETF performance
  - [x] Market breadth metrics

- [x] **1.1.3 FRED macro fetcher**
  **DoD**:
  - [x] M2 money supply
  - [x] Jobless claims
  - [x] Yield curve (10Y-2Y spread)
  - [x] PMI
  - [x] 12-hour cache TTL

- [x] **1.1.4 Earnings calendar fetcher**
  **DoD**:
  - [x] Upcoming earnings dates per ticker
  - [x] 3-day pre-earnings exclusion
  - [x] 1-day post-earnings exclusion
  - [x] 7-day cache TTL

#### 1.2 Feature Engineering
- [x] **1.2.1 Technical indicators**
  **DoD**:
  - [x] RSI (14)
  - [x] MACD (12, 26, 9)
  - [x] Bollinger Bands (20, 2)
  - [x] ATR (14)
  - [x] SMA (10, 20, 50, 200)
  - [x] Stochastic

- [x] **1.2.2 Cross-sectional features**
  **DoD**:
  - [x] Performance vs sector ETF
  - [x] Performance vs SPY
  - [x] Relative strength rank
  - [x] Sector momentum

- [x] **1.2.3 Temporal features**
  **DoD**:
  - [x] Day of week
  - [x] Days since earnings
  - [x] Days until earnings
  - [x] Month-end proximity

#### Phase 1 Quality Gate: PASSED
- [x] All data fetchers return valid data
- [x] Caching reduces API calls by >80%
- [x] Feature engineering produces consistent shapes
- [x] No NaN values in critical features

---

### Phase 2: ML Model Ensemble
**Goal**: Accurate signal prediction with multiple models

#### 2.1 Individual Models
- [x] **2.1.1 LSTM Signal Model**
  **DoD**:
  - [x] Bi-directional LSTM layers
  - [x] Temporal attention mechanism
  - [x] Multi-task output (probability, return, confidence)
  - [x] GPU inference support
  - [x] Trained model saved to disk

- [x] **2.1.2 Transformer Model**
  **DoD**:
  - [x] Multi-head self-attention
  - [x] Positional encoding
  - [x] Same output format as LSTM
  - [x] Handles long sequences

- [x] **2.1.3 MLP Fallback Model**
  **DoD**:
  - [x] Simple feedforward network
  - [x] Fast inference (<10ms)
  - [x] CPU-only fallback
  - [x] Baseline accuracy

#### 2.2 Ensemble Logic
- [x] **2.2.1 Hybrid ensemble voting**
  **DoD**:
  - [x] Configurable thresholds per model:
    - Transformer: 50%
    - LSTM: 40%
    - MLP: 40%
  - [x] Requires 2/3 model agreement
  - [x] Weighted average of signal strengths
  - [x] Consensus boost: +10% for 2-vote, +20% for 3-vote

- [x] **2.2.2 Model performance tracking**
  **DoD**:
  - [x] Per-model accuracy logged
  - [x] Ensemble vs individual comparison
  - [x] Automatic model selection if one fails

#### Phase 2 Quality Gate: PASSED
- [x] All models produce valid outputs
- [x] Ensemble accuracy > any individual model
- [x] Inference time <1 second per stock
- [x] GPU utilization >80% during inference

---

### Phase 3: Signal Adjustment System
**Goal**: Research-backed signal modifications

#### 3.1 Signal Modifiers
- [x] **3.1.1 Boost patterns (55 total)**
  **DoD**:
  - [x] Each boost has:
    - Pattern description
    - Boost value (+1 to +12)
    - Sample size from backtest
    - Win rate evidence
  - [x] Examples:
    - Two down days + severe drawdown: +12 (73.2% win)
    - Very high volatility + normal range: +10 (85.7% win)
    - Down streak with drawdown: +10 (76.9% win)

- [x] **3.1.2 Penalty patterns (33 total)**
  **DoD**:
  - [x] Each penalty has:
    - Pattern description
    - Penalty value (-1 to -10)
    - Sample size from backtest
    - Win rate evidence
  - [x] Examples:
    - Industrials + slight strength: -10 (30% win)
    - 5+ down days + minimal drawdown: -10 (36.6% win)
    - Wednesday trading: -4

- [x] **3.1.3 Tier multipliers**
  **DoD**:
  - [x] Elite: 1.1x signal multiplier
  - [x] Strong: 1.05x
  - [x] Average: 1.0x
  - [x] Weak: 0.9x
  - [x] Avoid: 0.8x

#### 3.2 Penalties-Only Mode
- [x] **3.2.1 Research finding implementation**
  **DoD**:
  - [x] Research showed penalties > boosts for filtering
  - [x] Full config: 59.2% win, +0.76% return
  - [x] Penalties only: 62.5% win, +1.72% return
  - [x] Implemented `penalties_only_calculator.py`

#### Phase 3 Quality Gate: PASSED
- [x] All 88 modifiers documented with evidence
- [x] Adjusted signals improve win rate
- [x] Configuration in unified_config.json
- [x] A/B test: adjusted vs raw signals

---

### Phase 4: Position Sizing & Risk Management
**Goal**: Optimal position sizing with risk controls

#### 4.1 Position Sizer
- [x] **4.1.1 Kelly Criterion base**
  **DoD**:
  - [x] Optimal sizing based on win rate and payoff
  - [x] Fractional Kelly (50%) for safety
  - [x] Signal strength scaling

- [x] **4.1.2 Regime adjustments**
  **DoD**:
  - [x] Volatile: 1.3x position multiplier (75.5% win)
  - [x] Bear: 1.2x multiplier (71.2% win)
  - [x] Choppy: 1.0x (baseline)
  - [x] Bull: 0.8x (counterintuitive but validated)

- [x] **4.1.3 Tier-based sizing**
  **DoD**:
  - [x] Elite: 15% max position
  - [x] Strong: 12% max
  - [x] Average: 7.5% max
  - [x] Weak: 4% max
  - [x] Avoid: 0% (skip)

#### 4.2 Portfolio Constraints
- [x] **4.2.1 Position limits**
  **DoD**:
  - [x] Max 6 concurrent positions
  - [x] Min 2% position size
  - [x] Max 15% position size

- [x] **4.2.2 Sector limits**
  **DoD**:
  - [x] Max 2 positions per sector
  - [x] Sector mapping for all 54 stocks
  - [x] Skip signal if sector full

- [x] **4.2.3 Correlation limits**
  **DoD**:
  - [x] Max 3 highly correlated positions
  - [x] Correlation threshold: 0.7

- [x] **4.2.4 Portfolio heat**
  **DoD**:
  - [x] Max 15% total portfolio at risk
  - [x] Heat = sum of (position size * stop distance)
  - [x] Skip signal if heat exceeded

#### Phase 4 Quality Gate: PASSED
- [x] Position sizing matches config
- [x] All constraints enforced
- [x] No portfolio heat violations
- [x] Sector diversification maintained

---

### Phase 5: Exit Strategy
**Goal**: Tier-based exit rules

#### 5.1 Exit Rules
- [x] **5.1.1 Profit targets**
  **DoD**:
  - [x] Tier-based targets:
    - Elite: +2.0% / +3.5%
    - Strong: +2.0% / +3.0%
    - Average: +2.0% / +2.5%
    - Weak: +1.5% / +2.0%
  - [x] Partial exit at target 1 (50%)
  - [x] Full exit at target 2

- [x] **5.1.2 Stop losses**
  **DoD**:
  - [x] Tier-based stops:
    - Elite/Strong: -3%
    - Average: -2.5%
    - Weak: -2%
  - [x] Triggered at close, not intraday

- [x] **5.1.3 Max hold days**
  **DoD**:
  - [x] Elite: 5 days
  - [x] Strong: 4 days (configurable as 5)
  - [x] Average: 3 days
  - [x] Weak: 2 days
  - [x] Exit at close of max day

- [x] **5.1.4 Trailing stops (optional)**
  **DoD**:
  - [x] Activated after target 1 hit
  - [x] Trail distance: 1.5% from high
  - [x] Configurable per tier

#### Phase 5 Quality Gate: PASSED
- [x] Exit rules match config
- [x] Partial exits calculated correctly
- [x] Max hold enforced
- [x] Backtest shows improved returns

---

### Phase 6: Integration & Output
**Goal**: Complete scanner with clean output

#### 6.1 SignalV2 Output
- [x] **6.1.1 SignalV2 dataclass**
  ```python
  @dataclass
  class SignalV2:
      ticker: str
      timestamp: datetime
      raw_strength: float
      adjusted_strength: float
      tier: str
      regime: str
      size_pct: float
      dollar_size: float
      shares: int
      risk_dollars: float
      quality: str
      skip_trade: bool
      skip_reason: Optional[str]
      profit_target_1: float
      profit_target_2: float
      stop_loss: float
      max_hold_days: int
      price: float
      sector: str
      sector_momentum: float
      boosts_applied: List[str]
      penalties_applied: List[str]
      near_earnings: bool
      earnings_warning: Optional[str]
  ```
  **DoD**:
  - [x] All fields populated
  - [x] JSON serializable
  - [x] Hashable for deduplication

- [x] **6.1.2 Scanner runner**
  **DoD**:
  - [x] Scans all 54 stocks
  - [x] Parallel execution where possible
  - [x] Progress reporting
  - [x] Error recovery (skip failed ticker, continue)

- [x] **6.1.3 Output formats**
  **DoD**:
  - [x] JSON file output
  - [x] Console summary
  - [x] CSV export option
  - [x] `latest_scan.json` symlink

#### 6.2 VirtualWallet Integration
- [x] **6.2.1 Signal to position**
  **DoD**:
  - [x] VirtualWallet receives SignalV2
  - [x] Opens position if not skipped
  - [x] Tracks entry price, time

- [x] **6.2.2 Exit monitoring**
  **DoD**:
  - [x] Daily check of open positions
  - [x] Triggers exits based on rules
  - [x] Logs exit reason

#### Phase 6 Quality Gate: PASSED
- [x] Full scan completes in <60 seconds
- [x] Output matches expected schema
- [x] VirtualWallet processes signals correctly
- [x] 30 days of paper trading verified

---

## Progress Summary

| Phase | Status | Checkboxes | Quality Gate |
|-------|--------|------------|--------------|
| 1. Data Infrastructure | ✅ Complete | 9/9 | ✅ |
| 2. ML Model Ensemble | ✅ Complete | 6/6 | ✅ |
| 3. Signal Adjustment | ✅ Complete | 4/4 | ✅ |
| 4. Position Sizing | ✅ Complete | 8/8 | ✅ |
| 5. Exit Strategy | ✅ Complete | 5/5 | ✅ |
| 6. Integration | ✅ Complete | 6/6 | ✅ |
| **TOTAL** | **✅ Production** | **38/38** | |

---

## Production Configuration

### Signal Thresholds by Regime
| Regime | Base Threshold | Adjusted Threshold |
|--------|----------------|-------------------|
| Bull | 60 | 65 (conservative) |
| Volatile | 60 | 45 (aggressive) |
| Choppy | 60 | 60 (standard) |
| Bear | 60 | 50 (selective aggressive) |

### Stock Universe (54 stocks)
| Tier | Count | Stocks |
|------|-------|--------|
| Elite | 9 | MPC, V, KLAC, SHW, QCOM, COP, SLB, EOG, GILD |
| Strong | 9 | JNJ, MS, AMAT, ADI, TXN, JPM, CVS, XOM, IDXX |
| Average | 14 | MSFT, ABBV, SYK, PFE, USB, PNC, APD, NEE, LMT, SCHW, AIG, MA, TMUS, EXR |
| Weak | 15 | NVDA, AVGO, AXP, WMT, CMCSA, META, INSM, ROAD, MRVL, MLM, LOW, ECL, NOW, CRM, INTU |
| Avoid | 7 | CAT, HCA, TGT, ETN, HD, ORCL, ADBE |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/trading/smart_scanner_v2.py` | Main scanner orchestration |
| `src/models/hybrid_signal_model.py` | ML ensemble |
| `src/models/lstm_signal_model.py` | LSTM model |
| `src/trading/penalties_only_calculator.py` | Signal adjustments |
| `src/trading/unified_position_sizer.py` | Position sizing |
| `src/trading/position_rebalancer.py` | Exit management |
| `src/analysis/unified_regime_detector.py` | Regime detection |
| `config/unified_config.json` | All configuration |

---

## Daily Workflow

```bash
# Full daily scan
python scripts/run_virtual_wallet_daily.py --full

# Check status
python scripts/run_virtual_wallet_daily.py --status

# Generate report
python scripts/generate_daily_report.py

# Run manual scan
python run_scan.py
```

---

## Performance Metrics

### Backtest Results (2-year validation)
| Metric | Value |
|--------|-------|
| Total Trades | 3,243 |
| Win Rate | 60.4% |
| Average Return | +0.82% |
| Sharpe Ratio | 1.39 |
| Max Drawdown | -8.2% |
| Profit Factor | 1.67 |

### By Regime
| Regime | Trades | Win Rate | Avg Return | Sharpe |
|--------|--------|----------|-----------|--------|
| Volatile | 412 | 75.5% | +1.82% | 4.86 |
| Bear | 287 | 71.2% | +1.45% | 3.32 |
| Choppy | 1,843 | 59.3% | +0.64% | 1.25 |
| Bull | 701 | 58.8% | +0.52% | 1.00 |

---

## References

- [SYSTEM_STATE.md](../../SYSTEM_STATE.md) - Full system documentation
- [unified_config.json](../../config/unified_config.json) - Production configuration
- [EXPERIMENT_CONCLUSIONS.md](../../EXPERIMENT_CONCLUSIONS.md) - Research backing

---

*Document Version: 2.0*
*Last Updated: January 2026*
*Status: Production Complete*

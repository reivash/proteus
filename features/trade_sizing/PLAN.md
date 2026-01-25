# Position Sizing & Risk Management System

> **Project Status**: Production (Complete)
> **Version**: 2.0
> **Last Updated**: January 2026
> **Improvement**: +24.6% avg return, +7.8% Sharpe vs baseline

---

## Overview

### Problem Statement
Position sizing dramatically impacts returns. Too large = excessive risk; too small = missed opportunity. A systematic approach incorporating signal strength, stock quality, and market regime is essential.

### Objectives
1. **Optimal Sizing**: Kelly Criterion-based position sizing
2. **Risk Control**: Portfolio heat limits, sector diversification
3. **Regime Adaptation**: Adjust sizing based on market conditions
4. **Tier Awareness**: Better stocks get larger positions

### Key Achievements
- Kelly Criterion with fractional sizing (50%)
- Regime-adaptive multipliers validated on 2-year backtest
- Tier-based position limits (4-15% range)
- Portfolio heat tracking (15% max)
- +24.6% improvement in average return vs baseline

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    POSITION SIZING & RISK MANAGEMENT                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   INPUTS     │    │   UNIFIED    │    │   OUTPUTS    │                  │
│  │              │───▶│   POSITION   │───▶│              │                  │
│  │ - Signal     │    │    SIZER     │    │ - Size %     │                  │
│  │ - Tier       │    │              │    │ - Shares     │                  │
│  │ - Regime     │    │ Kelly + Adj  │    │ - Risk $     │                  │
│  │ - Volatility │    │              │    │ - Skip?      │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                             │                                               │
│         ┌───────────────────┼───────────────────┐                          │
│         ▼                   ▼                   ▼                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                   │
│  │  PORTFOLIO   │   │   SECTOR     │   │    HEAT      │                   │
│  │  CONSTRAINTS │   │   LIMITS     │   │   TRACKING   │                   │
│  │              │   │              │   │              │                   │
│  │ - Max 6 pos  │   │ - Max 2/sect │   │ - Max 15%    │                   │
│  │ - 2-15% size │   │ - Skip if    │   │ - Per trade  │                   │
│  │ - Max 3 corr │   │   full       │   │   2% risk    │                   │
│  └──────────────┘   └──────────────┘   └──────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

> **Note**: This project is COMPLETE. All checkboxes document the production system.

---

### Phase 1: Kelly Criterion Foundation
**Goal**: Implement optimal sizing formula

#### 1.1 Kelly Formula
- [x] **1.1.1 Basic Kelly implementation**
  ```python
  kelly_fraction = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
  ```
  **DoD**:
  - [x] Calculates optimal fraction
  - [x] Uses historical win rate per tier
  - [x] Uses historical avg win/loss per tier
  - [x] Returns fraction 0-1

- [x] **1.1.2 Fractional Kelly**
  **DoD**:
  - [x] Apply 50% of Kelly (half-Kelly)
  - [x] Reduces volatility while maintaining edge
  - [x] Configurable fraction

- [x] **1.1.3 Signal strength scaling**
  **DoD**:
  - [x] Higher signal = closer to full Kelly
  - [x] Scale factor: signal_strength / 100
  - [x] Minimum 20% of Kelly if signal passes threshold

#### 1.2 Historical Data Integration
- [x] **1.2.1 Win rate by tier**
  **DoD**:
  - [x] Elite: 65% historical win rate
  - [x] Strong: 60%
  - [x] Average: 55%
  - [x] Weak: 50%
  - [x] Data from 2-year backtest

- [x] **1.2.2 Average win/loss by tier**
  **DoD**:
  - [x] Elite: +0.56% avg win
  - [x] Strong: +1.15% avg win
  - [x] Average: +0.57% avg win
  - [x] Weak: +0.30% avg win
  - [x] Stop loss defines avg loss

#### Phase 1 Quality Gate: PASSED
- [x] Kelly calculation matches manual verification
- [x] Fractional Kelly reduces position size appropriately
- [x] Signal scaling produces intuitive results

---

### Phase 2: Regime Adjustments
**Goal**: Adapt sizing to market conditions

#### 2.1 Regime Multipliers
- [x] **2.1.1 Research-backed multipliers**
  **DoD**:
  - [x] Volatile regime: 1.3x position multiplier
    - Rationale: 75.5% win rate, 4.86 Sharpe
  - [x] Bear regime: 1.2x multiplier
    - Rationale: 71.2% win rate, 3.32 Sharpe
  - [x] Choppy regime: 1.0x (baseline)
    - Rationale: 59.3% win rate, 1.25 Sharpe
  - [x] Bull regime: 0.8x multiplier
    - Rationale: 58.8% win rate, worst Sharpe

- [x] **2.1.2 Signal threshold adjustments**
  **DoD**:
  - [x] Volatile: Lower threshold (-15 from base)
  - [x] Bear: Lower threshold (-10 from base)
  - [x] Choppy: Base threshold
  - [x] Bull: Higher threshold (+5 from base)

- [x] **2.1.3 Regime detection integration**
  **DoD**:
  - [x] Receives regime from UnifiedRegimeDetector
  - [x] Applies correct multiplier
  - [x] Logs regime used

#### 2.2 Validation
- [x] **2.2.1 Backtest verification**
  **DoD**:
  - [x] 2-year backtest with regime sizing
  - [x] Compare to fixed sizing baseline
  - [x] Document improvement metrics

#### Phase 2 Quality Gate: PASSED
- [x] Regime multipliers applied correctly
- [x] Backtest shows improvement
- [x] Volatile/Bear regimes show best risk-adjusted returns

---

### Phase 3: Portfolio Constraints
**Goal**: Prevent over-concentration and excessive risk

#### 3.1 Position Limits
- [x] **3.1.1 Maximum positions**
  **DoD**:
  - [x] Max 6 concurrent positions
  - [x] Skip new signals if at max
  - [x] Log skip reason

- [x] **3.1.2 Position size bounds**
  **DoD**:
  - [x] Minimum: 2% of portfolio
  - [x] Maximum: 15% of portfolio
  - [x] Clamp calculated size to bounds

- [x] **3.1.3 Tier-based maximums**
  **DoD**:
  - [x] Elite: up to 15%
  - [x] Strong: up to 12%
  - [x] Average: up to 7.5%
  - [x] Weak: up to 4%
  - [x] Avoid: 0% (skip)

#### 3.2 Sector Diversification
- [x] **3.2.1 Sector mapping**
  **DoD**:
  - [x] All 54 stocks mapped to sectors
  - [x] 11 sectors (Tech, Healthcare, Finance, etc.)
  - [x] Mapping in unified_config.json

- [x] **3.2.2 Sector position limits**
  **DoD**:
  - [x] Max 2 positions per sector
  - [x] Skip signal if sector at limit
  - [x] Log sector skip reason

#### 3.3 Correlation Limits
- [x] **3.3.1 Correlation tracking**
  **DoD**:
  - [x] 90-day rolling correlation matrix
  - [x] Flag highly correlated pairs (>0.7)
  - [x] Max 3 highly correlated positions

- [x] **3.3.2 Correlation-based skipping**
  **DoD**:
  - [x] If new position correlated with 3+ existing
  - [x] Skip or reduce size
  - [x] Log correlation skip

#### Phase 3 Quality Gate: PASSED
- [x] Position limits enforced
- [x] Sector diversification maintained
- [x] Correlation limits prevent clustering

---

### Phase 4: Portfolio Heat Tracking
**Goal**: Real-time risk monitoring

#### 4.1 Heat Calculation
- [x] **4.1.1 Per-position heat**
  ```python
  position_heat = position_size_pct * stop_loss_distance
  ```
  **DoD**:
  - [x] Heat = potential loss at stop
  - [x] Example: 10% position with 3% stop = 0.3% heat
  - [x] Calculate for all open positions

- [x] **4.1.2 Total portfolio heat**
  **DoD**:
  - [x] Sum of all position heats
  - [x] Real-time tracking
  - [x] Displayed in status output

- [x] **4.1.3 Risk per trade**
  **DoD**:
  - [x] Max 2% portfolio risk per new trade
  - [x] Ensures no single trade can lose >2%

#### 4.2 Heat Limits
- [x] **4.2.1 Maximum heat enforcement**
  **DoD**:
  - [x] Max 15% total portfolio heat
  - [x] Skip new signals if heat exceeded
  - [x] Log heat-based skips

- [x] **4.2.2 Heat-based sizing reduction**
  **DoD**:
  - [x] If heat near limit (>12%)
  - [x] Reduce new position sizes
  - [x] Never exceed limit

#### Phase 4 Quality Gate: PASSED
- [x] Heat calculated correctly
- [x] Heat limit never exceeded
- [x] Risk per trade capped at 2%

---

### Phase 5: Exit Integration
**Goal**: Connect sizing to exit strategy

#### 5.1 Exit Rules by Tier
- [x] **5.1.1 Profit targets**
  **DoD**:
  - [x] Elite: +2.0% (partial), +3.5% (full)
  - [x] Strong: +2.0% / +3.0%
  - [x] Average: +2.0% / +2.5%
  - [x] Weak: +1.5% / +2.0%
  - [x] 50% exit at first target

- [x] **5.1.2 Stop losses**
  **DoD**:
  - [x] Elite/Strong: -3%
  - [x] Average: -2.5%
  - [x] Weak: -2%
  - [x] Used in heat calculation

- [x] **5.1.3 Max hold days**
  **DoD**:
  - [x] Elite: 5 days
  - [x] Strong: 5 days
  - [x] Average: 3 days
  - [x] Weak: 2 days

#### 5.2 Position Rebalancer
- [x] **5.2.1 Daily exit check**
  **DoD**:
  - [x] Check all positions against exit rules
  - [x] Trigger exits at targets/stops
  - [x] Track partial exits

- [x] **5.2.2 Heat update on exit**
  **DoD**:
  - [x] Reduce heat when position closed
  - [x] Real-time heat recalculation
  - [x] Free up capacity for new signals

#### Phase 5 Quality Gate: PASSED
- [x] Exit rules match config
- [x] Heat updates on exits
- [x] Partial exits tracked correctly

---

## Progress Summary

| Phase | Status | Checkboxes | Quality Gate |
|-------|--------|------------|--------------|
| 1. Kelly Foundation | ✅ Complete | 6/6 | ✅ |
| 2. Regime Adjustments | ✅ Complete | 5/5 | ✅ |
| 3. Portfolio Constraints | ✅ Complete | 8/8 | ✅ |
| 4. Heat Tracking | ✅ Complete | 5/5 | ✅ |
| 5. Exit Integration | ✅ Complete | 6/6 | ✅ |
| **TOTAL** | **✅ Production** | **30/30** | |

---

## Production Configuration

### Position Size Limits
```json
{
  "min_position_pct": 2.0,
  "max_position_pct": 15.0,
  "max_positions": 6,
  "max_sector_positions": 2,
  "max_correlated_positions": 3,
  "max_portfolio_heat": 15.0,
  "risk_per_trade": 2.0
}
```

### Tier-Based Sizing
| Tier | Max Size | Win Rate | Stop Loss |
|------|----------|----------|-----------|
| Elite | 15% | 65% | -3% |
| Strong | 12% | 60% | -3% |
| Average | 7.5% | 55% | -2.5% |
| Weak | 4% | 50% | -2% |
| Avoid | 0% | 45% | N/A |

### Regime Multipliers
| Regime | Size Mult | Threshold Adj | Win Rate |
|--------|-----------|---------------|----------|
| Volatile | 1.3x | -15 | 75.5% |
| Bear | 1.2x | -10 | 71.2% |
| Choppy | 1.0x | 0 | 59.3% |
| Bull | 0.8x | +5 | 58.8% |

---

## Key Files

| File | Purpose |
|------|---------|
| `common/trading/unified_position_sizer.py` | Main position sizer |
| `common/trading/position_rebalancer.py` | Exit monitoring |
| `common/trading/virtual_wallet.py` | Portfolio tracking |
| `config/unified_config.json` | All parameters |

---

## Performance Metrics

### Before vs After Position Sizing
| Metric | Baseline | With Sizing | Improvement |
|--------|----------|-------------|-------------|
| Sharpe Ratio | 1.29 | 1.39 | +7.8% |
| Win Rate | 58.3% | 60.8% | +2.5% |
| Avg Return | +0.533% | +0.664% | +24.6% |
| Max Drawdown | -12.1% | -8.2% | -32% |

### Heat Tracking Stats
- Average heat: 8.2%
- Max heat observed: 14.8%
- Heat limit breaches: 0

---

## References

- [ARCHITECTURE_SIMPLIFICATION.md](../../ARCHITECTURE_SIMPLIFICATION.md) - Unified sizer design
- [ROADMAP_2025_Q1.md](../../ROADMAP_2025_Q1.md) - Phase 2 results
- [unified_config.json](../../config/unified_config.json) - Configuration

---

*Document Version: 2.0*
*Last Updated: January 2026*
*Status: Production Complete*

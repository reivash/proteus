# Architecture Simplification Summary
## January 4, 2026

## Problem Identified

The Proteus system had accumulated complexity over time:

1. **Triple Position Sizing** - 3 competing modules:
   - `common/trading/position_sizer.py` (574 lines)
   - `common/trading/volatility_sizing.py` (411 lines)
   - `common/config/position_sizing.py` (287 lines) - outdated data

2. **12+ Signal Multipliers** - Multiplicative compounding:
   - 9 separate multipliers applied as `base * m1 * m2 * m3 * ...`
   - Unpredictable compound effects (0.8 * 0.9 * 1.15 * ... = ??)
   - Hard to debug and tune

3. **Stock Tiers Defined in 3 Places**:
   - `exit_optimizer.py`
   - `enhanced_signal_calculator.py`
   - `config/position_sizing.py`
   - Conflicting definitions (NVDA listed as both "87.5% win rate" and "weak")

4. **No Rebalancing Logic** - Missing exit monitoring

---

## Solution: Unified Architecture

### 1. Single Configuration Source

**File**: `config/unified_config.json`

```json
{
  "stock_tiers": {
    "elite": {"tickers": [...], "position_size_pct": 12, "signal_multiplier": 1.10},
    "strong": {...},
    "average": {...},
    "weak": {...},
    "avoid": {...}
  },
  "regime_adjustments": {...},
  "signal_boosts": {...},
  "exit_strategy": {...},
  "portfolio_constraints": {...}
}
```

**Benefits**:
- Single source of truth
- No conflicting definitions
- Easy to update and version control

---

### 2. Unified Position Sizer

**File**: `common/trading/unified_position_sizer.py`

Consolidates the best of all 3 modules:
- Kelly-based tier sizing
- Regime adjustments
- Volatility normalization
- Portfolio heat tracking
- Position persistence

**API**:
```python
sizer = UnifiedPositionSizer(portfolio_value=100000)
rec = sizer.calculate_size('COP', price=105.0, signal_strength=85, regime='volatile')
# Returns: PositionRecommendation with size_pct, shares, risk_dollars, quality
```

---

### 3. Simplified Signal Calculator

**File**: `common/trading/unified_signal_calculator.py`

**Before** (9 multiplicative layers):
```python
final = base * regime * sector * dow * down_days * volume * tier * rsi * vol_exh
# Example: 65 * 1.15 * 1.10 * 1.05 * 1.15 * 1.10 * 1.10 * 1.08 * 1.12 = 128.4 (capped at 100)
```

**After** (2 multipliers + additive boosts):
```python
adjusted = base * tier_mult * regime_mult  # Only 2 multiplicative
final = adjusted + monday_boost + rsi_boost + streak_boost + ...  # Additive
# Example: 65 * 1.10 * 1.15 = 82.2 + 3 + 5 + 8 + 6 + 5 = 100 (capped)
```

**Benefits**:
- Predictable behavior
- Each boost contributes fixed amount
- Easy to understand: "Monday adds +3 points"

---

### 4. Position Rebalancer

**File**: `common/trading/position_rebalancer.py`

Monitors active positions against exit rules:
- Stop loss triggers
- Profit target exits (partial and full)
- Max hold days enforcement
- Trailing stops

**API**:
```python
rebalancer = PositionRebalancer()
status = rebalancer.check_position('COP', entry_price=100, current_price=103.5, entry_date=...)
# Returns: PositionStatus with exit_reason, exit_pct, recommendation
```

---

## Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `config/unified_config.json` | Single config source | 175 |
| `common/trading/unified_position_sizer.py` | Position sizing | 260 |
| `common/trading/unified_signal_calculator.py` | Signal calculation | 195 |
| `common/trading/position_rebalancer.py` | Exit monitoring | 220 |
| `tests/test_unified_system.py` | Validation tests | 180 |

**Total**: ~1,030 lines of clean, modular code

---

## Test Results

```
============================================================
TOTAL: 14/14 tests passed
SUCCESS - All tests passed!
============================================================
```

- Position Sizer: 4/4 passed
- Signal Calculator: 5/5 passed
- Position Rebalancer: 5/5 passed

---

## Migration Path

The old modules are still in place and working. To migrate:

1. **Gradually replace imports**:
   ```python
   # Old
   from common.trading.position_sizer import PositionSizer

   # New
   from common.trading.unified_position_sizer import UnifiedPositionSizer
   ```

2. **Update SmartScanner** to use unified modules

3. **Deprecate old modules** after testing in production

---

## Performance Comparison

The simplified system maintains the same performance characteristics:

| Metric | Old System | Unified System |
|--------|-----------|----------------|
| Elite + volatile | ~15% size | 14.98% size |
| Avoid tier | Skip | Skip |
| Signal boost range | 0-35 pts | 0-27 pts |
| Exit triggers | Working | Working + tested |

---

## Next Steps

1. **Integrate unified modules with SmartScanner**
2. **Run parallel testing** (both old and new)
3. **Deprecate old modules** after validation
4. **Continue Phase 3** (Neural Network V2)

---

## Architecture Diagram

```
                    unified_config.json
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
   UnifiedPositionSizer  UnifiedSignalCalc  PositionRebalancer
          |                |                |
          +--------+-------+                |
                   |                        |
                   v                        v
              SmartScanner ---------> Exit Monitoring
                   |
                   v
              Daily Signals
```

---

**Result**: Cleaner, more maintainable architecture ready for continued development.

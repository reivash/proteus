# Regime Misclassification Analysis

> Generated: 2026-01-25 13:49

---

## Top 5 Worst Misclassifications

### #1: 2025-03-24

| Aspect | Value |
|--------|-------|
| HMM Regime | BULL |
| Actual Regime | VOLATILE |
| Severity | 4.86 |
| 5-day Return | -2.6% |
| 10-day Return | -12.1% |
| VIX | 17.5 |
| Root Cause | vol_underweight |

**Analysis**: HMM underweighted volatility. Realized vol: 20.5%

### #2: 2023-10-27

| Aspect | Value |
|--------|-------|
| HMM Regime | BEAR |
| Actual Regime | BULL |
| Severity | 3.64 |
| 5-day Return | +5.8% |
| 10-day Return | +7.3% |
| VIX | 21.3 |
| Root Cause | momentum_lag |

**Analysis**: HMM lagged momentum reversal. 10d return was -4.8% before rally.

### #3: 2024-09-06

| Aspect | Value |
|--------|-------|
| HMM Regime | BEAR |
| Actual Regime | BULL |
| Severity | 2.74 |
| 5-day Return | +4.0% |
| 10-day Return | +5.5% |
| VIX | 22.4 |
| Root Cause | momentum_lag |

**Analysis**: HMM lagged momentum reversal. 10d return was -2.9% before rally.

### #4: 2025-03-31

| Aspect | Value |
|--------|-------|
| HMM Regime | BULL |
| Actual Regime | VOLATILE |
| Severity | 1.45 |
| 5-day Return | -9.8% |
| 10-day Return | -3.6% |
| VIX | 22.3 |
| Root Cause | vol_underweight |

**Analysis**: HMM underweighted volatility. Realized vol: 20.8%

### #5: 2023-08-03

| Aspect | Value |
|--------|-------|
| HMM Regime | BEAR |
| Actual Regime | CHOPPY |
| Severity | 1.00 |
| 5-day Return | -0.7% |
| 10-day Return | -2.8% |
| VIX | 15.9 |
| Root Cause | unknown |

**Analysis**: Unable to determine root cause.

---

## Root Cause Summary

| Root Cause | Count | % |
|------------|-------|---|
| unknown | 64 | 94.1% |
| vol_underweight | 2 | 2.9% |
| momentum_lag | 2 | 2.9% |

---

## Recommendations

Based on root cause analysis:

1. **Add momentum divergence detection** - HMM is slow to recognize reversals

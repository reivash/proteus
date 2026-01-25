# EXPERIMENT EXP-008: FOMC Meeting Filter

**Date:** 2025-11-14
**Status:** COMPLETED - NEGATIVE RESULTS
**Priority:** MEDIUM
**Optimization:** FOMC filter to avoid Fed policy-driven moves

---

## EXECUTIVE SUMMARY

### Objective

Test FOMC meeting filter (1 day before + 1 day after meetings) to protect strategy from Federal Reserve policy-driven volatility.

### Results - NEGATIVE IMPACT

**Average Impact:**
- Signals blocked: 2.0 per stock
- Win rate change: **-1.2%** (negative)
- Return change: **-8.06%** (significant negative impact)

**Stock-by-stock results:**

| Stock | Signals Blocked | Win Rate Change | Return Change | Assessment |
|-------|-----------------|-----------------|---------------|------------|
| NVDA | 2 | -1.3% | **-17.25%** | **VERY NEGATIVE** |
| **TSLA** | **0** | **+0.0%** | **+0.00%** | **NEUTRAL** (no signals blocked) |
| JPM | 5 | -1.1% | -7.96% | NEGATIVE |
| JNJ | 1 | -2.3% | -7.02% | NEGATIVE |

### Conclusion

**FOMC filter is NOT recommended for production v4.0.**

**Reasons:**
1. Significant negative impact on returns (-8.06% average)
2. Blocks profitable opportunities (especially NVDA -17.25%)
3. Minimal signals blocked (2.0 average, only 6.6% of trading days)
4. Mean reversion works even around FOMC meetings
5. Regime + earnings filters already provide sufficient protection

**Decision:** DO NOT include FOMC filter in production strategy

---

## FOMC MEETING ANALYSIS

### Historical FOMC Meeting Frequency

**2020 (COVID + Emergency Cuts):**
- FOMC meetings: 9 (including 1 emergency meeting)
- Total days: 365
- Exclusion days: 27 (7.4% of year)
- Avg days between meetings: 41

**2022 (Aggressive Tightening):**
- FOMC meetings: 8
- Total days: 364
- Exclusion days: 24 (6.6% of year)
- Avg days between meetings: 46

**2023 (Rates Plateau):**
- FOMC meetings: 8
- Total days: 364
- Exclusion days: 24 (6.6% of year)
- Avg days between meetings: 46

**2024-2025 (Recent Period):**
- FOMC meetings: 15
- Total days: 683
- Exclusion days: 45 (6.6% of period)
- Avg days between meetings: 46

**Key Insight:** FOMC meetings exclude 6-7% of trading days consistently (~24 days/year).

---

## PERFORMANCE IMPACT ANALYSIS

### Test Period: 2020-2025 (Includes Major Fed Policy Shifts)

| Stock | Sector | Without FOMC Filter | With FOMC Filter | Impact |
|-------|--------|---------------------|------------------|--------|
| **NVDA** | Tech-GPU | 84.6% win, +51.92% | 83.3% win, +34.67% | 2 signals blocked, -1.3% win, **-17.25% return** |
| **TSLA** | Tech-EV | 66.7% win, +10.73% | 66.7% win, +10.73% | **0 signals blocked, +0.0% win, +0.00% return** |
| **JPM** | Finance | 61.1% win, +12.09% | 60.0% win, +4.13% | 5 signals blocked, -1.1% win, -7.96% return |
| **JNJ** | Healthcare | 75.0% win, +27.64% | 72.7% win, +20.62% | 1 signal blocked, -2.3% win, -7.02% return |

**Average:**
- Signals blocked: **2.0** (small number)
- Win rate change: **-1.2%**
- Return change: **-8.06%** (significant negative)

### Detailed Stock Analysis

#### 1. NVDA - VERY NEGATIVE IMPACT

**Performance:**
- Signals blocked: 2
- Win rate: 84.6% → 83.3% (-1.3%)
- Return: +51.92% → +34.67% (**-17.25% drop!**)

**Why FOMC filter hurts NVDA:**
- NVDA had 84.6% win rate before filter (excellent performance)
- Blocking 2 signals removed highly profitable trades
- Mean reversion works well for NVDA even during Fed volatility
- FOMC filter blocks opportunities without adding value

**Conclusion:** FOMC filter significantly hurts NVDA performance

#### 2. TSLA - NEUTRAL IMPACT

**Performance:**
- Signals blocked: **0**
- Win rate: 66.7% → 66.7% (no change)
- Return: +10.73% → +10.73% (no change)

**Why no impact:**
- No TSLA panic sell signals occurred near FOMC meetings
- FOMC filter had no opportunity to affect performance

**Conclusion:** No impact (no signals near FOMC dates)

#### 3. JPM - NEGATIVE IMPACT

**Performance:**
- Signals blocked: 5 (most in test)
- Win rate: 61.1% → 60.0% (-1.1%)
- Return: +12.09% → +4.13% (-7.96%)

**Why FOMC filter hurts JPM:**
- Finance stocks often sell off before/after Fed meetings (rate sensitivity)
- Many of these selloffs are technical overcorrections that revert
- Blocking 5 signals = lost profitable opportunities
- JPM mean reversion works even during Fed policy volatility

**Conclusion:** FOMC filter significantly reduces JPM returns

#### 4. JNJ - NEGATIVE IMPACT

**Performance:**
- Signals blocked: 1
- Win rate: 75.0% → 72.7% (-2.3%)
- Return: +27.64% → +20.62% (-7.02%)

**Why FOMC filter hurts JNJ:**
- Healthcare defensive stock, less affected by Fed policy
- Blocking 1 signal removed a profitable trade
- Mean reversion works for JNJ regardless of FOMC timing

**Conclusion:** FOMC filter reduces JNJ performance without benefit

---

## COMPARATIVE ANALYSIS

### With FOMC Filter vs Without

**Summary Table:**

| Stock | Blocked | Win Rate |  | Return |  |
|-------|---------|----------|----------|--------|--------|
|       | Signals | Without | With | Without | With |
| NVDA  | 2       | 84.6%   | 83.3% | +51.92% | +34.67% |
| TSLA  | 0       | 66.7%   | 66.7% | +10.73% | +10.73% |
| JPM   | 5       | 61.1%   | 60.0% | +12.09% | +4.13% |
| JNJ   | 1       | 75.0%   | 72.7% | +27.64% | +20.62% |
| **AVG** | **2.0** | **71.9%** | **70.7%** | **+25.60%** | **+17.54%** |

**Change:** -1.2% win rate, -8.06% return (significant negative impact)

### Filter Effectiveness by Period

**2020 (COVID + Emergency Cuts):**
- 9 FOMC meetings (including emergency rate cuts)
- 7.4% of trading days excluded
- Minimal signals blocked (most signals not near FOMC dates)

**2022 (Aggressive Tightening):**
- 8 FOMC meetings (four 75bp hikes)
- 6.6% of trading days excluded
- Some signals blocked, but often profitable trades lost

**2023-2025 (Rates Plateau):**
- 8 meetings/year (regular schedule)
- 6.6% of trading days excluded
- Minimal impact (few signals near meetings)

**Key Finding:** FOMC filter blocks few signals (2.0 average) but removes profitable opportunities, resulting in significant return reduction (-8.06%).

---

## TRADE-OFF ANALYSIS

### Benefits of FOMC Filter (Theoretical)

1. **Policy risk protection**: Avoids Fed policy-driven volatility
2. **Predictable exclusions**: FOMC schedule known in advance
3. **Minimal disruption**: Only 6.6% of trading days excluded
4. **Simple implementation**: Calendar-based, no complex calculations

### Costs of FOMC Filter (Actual)

1. **Significant return reduction**: -8.06% average impact
2. **Blocks profitable trades**: NVDA -17.25%, JPM -7.96%
3. **No win rate improvement**: -1.2% average change
4. **Mean reversion works anyway**: Technical patterns persist around FOMC
5. **Opportunity cost**: Lost trades that would have been winners

### Comparison to Other Filters

**Regime Filter (implemented):**
- Blocks bear market trades
- Prevented -21% TSLA disaster
- Significant positive impact ✅

**Earnings Filter (implemented):**
- Blocks fundamental-driven selloffs
- +5% TSLA win rate improvement
- Positive impact ✅

**VIX Filter (tested, rejected):**
- Blocks extreme volatility periods
- -3.91% return impact
- Negative impact ❌

**FOMC Filter (tested here, rejected):**
- Blocks Fed policy periods
- **-8.06% return impact**
- **Significant negative impact** ❌

**Conclusion:** FOMC filter has WORSE impact than VIX filter (-8.06% vs -3.91%). Both should be rejected.

---

## STRATEGIC INSIGHTS

### 1. Mean Reversion Works Around FOMC Meetings

**Observation:** Only 2.0 signals blocked on average, but -8.06% return impact

**Reason:** The signals that DO occur near FOMC are often highly profitable

**Examples:**
- NVDA: 2 blocked signals cost -17.25% return
- JPM: 5 blocked signals cost -7.96% return

**Insight:** Mean reversion technical patterns persist even during Fed policy events

### 2. Finance Stocks Particularly Affected

**Observation:** JPM most impacted (5 signals blocked, -7.96% return)

**Reason:** Finance stocks are rate-sensitive, often sell off before/after FOMC

**Reality:** These selloffs are often technical overcorrections that revert

**Implication:** FOMC filter especially harmful for finance sector

### 3. FOMC Filter Blocks Few But Profitable Signals

**Pattern observed:**
- Blocks only 2.0 signals per stock (small number)
- But causes -8.06% return impact (large negative effect)
- Win rate barely changes (-1.2%)

**Analysis:**
- The signals near FOMC meetings are HIGH QUALITY
- Blocking them removes best opportunities
- No improvement in win rate to justify cost

**Conclusion:** FOMC filter has poor benefit-to-cost ratio

### 4. Regime + Earnings Filters Sufficient

**Regime filter:**
- Blocks bear market trades (prevents sustained downtrends)
- Already prevented -21% TSLA disaster

**Earnings filter:**
- Blocks fundamental event risk
- Improved TSLA win rate +5%

**FOMC filter:**
- Adds no incremental value
- Reduces returns significantly
- Redundant protection

**Conclusion:** Existing filters provide sufficient risk management

---

## RECOMMENDATIONS

### Production Strategy: DO NOT Use FOMC Filter

**Reasons:**
1. **Significant negative impact**: -8.06% return (worst filter tested)
2. **Blocks profitable opportunities**: NVDA -17.25%, JPM -7.96%
3. **No win rate improvement**: -1.2% change (negative)
4. **Minimal signals affected**: Only 2.0 per stock (6.6% of days)
5. **Existing filters sufficient**: Regime + earnings provide protection

**Mean Reversion Strategy v4.0 (PRODUCTION) should use:**
1. Market regime filter (prevents bear market trades) ✅
2. Earnings date filter (avoids fundamental events) ✅
3. Stock-specific parameters (optimized per stock) ✅
4. **NO VIX filter** (negative impact) ❌
5. **NO FOMC filter** (significant negative impact) ❌

### Not Recommended for Any Use Case

Unlike VIX filter (which had optional use case for risk-averse traders), FOMC filter has:
- **Worse performance impact** (-8.06% vs -3.91%)
- **No scenarios where beneficial** (all stocks show negative or neutral impact)
- **No compensating win rate improvement** (-1.2% change)

**Recommendation:** FOMC filter should NOT be used even optionally

### Alternative Approaches (If Desired)

**Instead of FOMC filter, consider:**

1. **Wider exclusion window analysis**:
   - Test 2-3 days before/after (vs current 1 day)
   - Likely to be even more negative

2. **Post-FOMC only filter**:
   - Block only days AFTER meeting (policy known)
   - May have different impact profile

3. **FOMC-specific position sizing**:
   - Reduce position size near FOMC (not block trades)
   - Maintain opportunity, reduce risk

4. **No action** (recommended):
   - Trust regime + earnings filters
   - Allow mean reversion to work naturally

---

## IMPLEMENTATION STATUS

### Code Created

**Files:**
1. `common/data/fetchers/fomc_calendar.py` - FOMC meeting calendar (2020-2025)
2. `common/experiments/exp008_fomc_filter.py` - FOMC filter testing framework

**Functionality:**
- Historical FOMC meeting dates (2020-2025)
- Configurable exclusion window (default: 1 day before + 1 day after)
- Signal filtering around FOMC dates
- Performance comparison with/without filter

### Usage Example (Not Recommended)

```python
from common.data.fetchers.fomc_calendar import FOMCCalendarFetcher

# Initialize FOMC filter (NOT recommended for production)
fomc_fetcher = FOMCCalendarFetcher(
    exclusion_days_before=1,
    exclusion_days_after=1
)

# Filter signals (NOT recommended - reduces performance)
filtered_signals = fomc_fetcher.add_fomc_filter_to_signals(signals, 'panic_sell')
```

### Configuration

**NOT added to production config** (`mean_reversion_params.py`) due to significant negative test results.

**FOMC filter available in code** but should NOT be used.

---

## CONCLUSIONS

### Main Findings

1. **FOMC filter shows significant negative impact**: -8.06% return
2. **Blocks few but profitable signals**: 2.0 average, high quality trades lost
3. **Worst filter tested**: Worse than VIX filter (-8.06% vs -3.91%)
4. **No win rate improvement**: -1.2% change (negative)
5. **Not recommended for any use case**: Even worse than VIX filter

### Production Decision

**Mean Reversion Strategy v4.0 will NOT include FOMC filter.**

**Filters deployed:**
1. Market regime filter (prevents bear market trades) ✅
2. Earnings date filter (avoids fundamental events) ✅
3. Stock-specific parameters (optimized per stock) ✅

**Filters tested and rejected:**
1. VIX filter: -3.91% return impact ❌
2. **FOMC filter: -8.06% return impact** ❌ **WORST PERFORMER**

### Strategic Impact

**Before FOMC filter test:**
- Question: Should we avoid trading around Fed meetings?
- Hypothesis: FOMC filter protects against policy-driven volatility

**After FOMC filter test:**
- Answer: No, FOMC filter significantly reduces returns (-8.06%)
- Discovery: Blocks few signals (2.0 avg) but they are high-quality trades
- Conclusion: Regime + earnings filters sufficient, FOMC adds no value

### Comparison of All Filters Tested

| Filter | Impact | Win Rate Change | Return Change | Decision |
|--------|--------|-----------------|---------------|----------|
| **Regime Filter** | Positive | N/A | Prevented -21% | ✅ **DEPLOYED** |
| **Earnings Filter** | Positive | +5% (TSLA) | Modest | ✅ **DEPLOYED** |
| **VIX Filter** | Negative | -1.0% | -3.91% | ❌ Rejected |
| **FOMC Filter** | **Very Negative** | **-1.2%** | **-8.06%** | ❌ **Rejected (worst)** |

### Future Research

**FOMC filter alternatives** (low priority, current filters sufficient):
1. Post-FOMC only filter (block after, not before)
2. FOMC-specific position sizing (reduce, not eliminate)
3. Sector-specific FOMC treatment (exclude finance)

**Higher priorities:**
1. Dynamic position sizing (size by confidence/performance tier)
2. Paper trading setup (live validation)
3. Review AAPL/CVX (both marginal at 60% win rate)

---

**Experiment:** EXP-008-FOMC
**Date Completed:** 2025-11-14
**Status:** COMPLETED - SIGNIFICANT NEGATIVE RESULTS
**Recommendation:** DO NOT use FOMC filter (worst filter tested, -8.06% return impact)

**Key Finding:** FOMC filter blocks only 2.0 signals per stock but causes -8.06% return reduction. These signals near FOMC meetings are high-quality trades that should NOT be filtered. Regime + earnings filters provide sufficient protection.

---

**Report Prepared By:** Zeus (Proteus Coordinator)
**Testing:** Prometheus
**Next Priority:** Dynamic position sizing or paper trading setup

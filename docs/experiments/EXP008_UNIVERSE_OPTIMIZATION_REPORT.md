# EXPERIMENT EXP-008: Universe Optimization (Review AAPL & CVX)

**Date:** 2025-11-14
**Status:** COMPLETED - NUANCED TRADE-OFF
**Priority:** HIGH
**Optimization:** Review marginal performers (AAPL, CVX) for potential removal

---

## EXECUTIVE SUMMARY

### Objective

Review marginal performers (AAPL 60% win, CVX 60% win) and determine optimal universe size.

**Test:** Compare 10-stock universe (current) vs 8-stock universe (remove AAPL, CVX)

### Results - NUANCED TRADE-OFF

**Quality Metrics Favor 8-Stock:**
- Win rate: **+3.1%** improvement (80.4% vs 77.3%)
- Sharpe ratio: **+0.91** improvement (7.25 vs 6.35)
- Avg return per stock: **+2.64%** improvement (14.32% vs 11.68%)

**Absolute Returns Favor 10-Stock:**
- Portfolio return: **-2.25%** reduction (114.58% vs 116.83%)
- AAPL + CVX contribute **+2.25%** to portfolio

### Decision Framework

**Two valid approaches:**

1. **QUALITY-FOCUSED (8 stocks):**
   - Remove AAPL and CVX
   - Higher win rate (80.4%)
   - Better Sharpe ratio (7.25)
   - Focus on proven performers only

2. **RETURN-FOCUSED (10 stocks - CURRENT):**
   - Keep AAPL and CVX
   - Higher absolute portfolio return (+2.25%)
   - Better diversification
   - Accept marginal performers for their contribution

**Recommendation:** **KEEP 10-stock universe** (diversification + absolute returns)

**Rationale:** +2.25% portfolio return contribution outweighs quality metric improvements. Even marginal performers add value.

---

## PERFORMANCE COMPARISON

### Portfolio-Level Metrics (2022-2025)

| Metric | 10-Stock (Current) | 8-Stock (Optimized) | Change |
|--------|-------------------|---------------------|--------|
| Universe Size | 10 | 8 | -2 stocks |
| Total Trades | 66 | 56 | -10 trades |
| **Avg Win Rate** | **77.3%** | **80.4%** | **+3.1%** ✅ |
| **Portfolio Return (Sum)** | **116.83%** | **114.58%** | **-2.25%** ❌ |
| **Avg Return Per Stock** | **11.68%** | **14.32%** | **+2.64%** ✅ |
| **Avg Sharpe Ratio** | **6.35** | **7.25** | **+0.91** ✅ |
| Avg Max Drawdown | -3.20% | -3.26% | -0.07% |

**Analysis:**
- **8-stock wins on quality:** Higher win rate, better Sharpe, higher avg per stock
- **10-stock wins on absolute returns:** Higher total portfolio return (+2.25%)

### AAPL & CVX Contribution Analysis

**AAPL (60% win rate):**
- Trades: 5
- Return: **-0.25%** (NEGATIVE)
- Sharpe: **-0.13** (negative risk-adjusted)
- Assessment: **Marginal/negative performer**

**CVX (60% win rate):**
- Trades: 5
- Return: **+2.50%** (POSITIVE)
- Sharpe: **2.68** (acceptable)
- Assessment: **Marginal but positive contributor**

**Combined AAPL + CVX:**
- Total return contribution: **+2.25%**
- Trade-off: Remove for quality OR keep for diversification/returns

---

## 8-STOCK UNIVERSE COMPOSITION

### Remaining Stocks (Sorted by Win Rate)

| Stock | Tier | Trades | Win Rate | Return | Sharpe | Sector |
|-------|------|--------|----------|--------|--------|--------|
| **NVDA** | 1 | 8 | **87.5%** | **+45.75%** | **12.22** | Tech-GPU |
| **TSLA** | 1 | 8 | **87.5%** | **+12.81%** | **7.22** | Tech-EV |
| **UNH** | 1 | 7 | **85.7%** | **+5.45%** | **4.71** | Healthcare |
| **JNJ** | 1 | 6 | **83.3%** | **+13.21%** | **12.46** | Healthcare |
| **JPM** | 1 | 10 | **80.0%** | **+18.63%** | **8.19** | Finance |
| **AMZN** | 1 | 5 | **80.0%** | **+10.20%** | **4.06** | Tech-Cloud |
| **MSFT** | 2 | 6 | **66.7%** | **+2.06%** | **1.71** | Tech-Cloud |
| **INTC** | 2 | 6 | **66.7%** | **+6.46%** | **5.07** | Tech-Semiconductor |

**Characteristics:**
- **6 Tier 1 stocks** (80%+ win rate)
- **2 Tier 2 stocks** (65-75% win rate)
- **0 Tier 3 stocks** (removed)
- **All stocks 65%+ win rate** (quality threshold)

### Sector Distribution (8-Stock)

**Before (10-stock):**
- Tech: 60% (6 stocks)
- Healthcare: 20% (2 stocks)
- Finance: 10% (1 stock)
- Energy: 10% (1 stock)

**After (8-stock):**
- Tech: **62.5%** (5 stocks) - AAPL removed
- Healthcare: **25%** (2 stocks)
- Finance: **12.5%** (1 stock)
- Energy: **0%** (CVX removed)

**Impact:** Slight increase in tech concentration, loss of energy sector exposure

---

## TRADE-OFF ANALYSIS

### Benefits of 8-Stock Universe

**Quality improvements:**
1. **Win rate +3.1%** (80.4% vs 77.3%) - Better success rate
2. **Sharpe +0.91** (7.25 vs 6.35) - Superior risk-adjusted returns
3. **Avg return per stock +2.64%** - Higher individual performance
4. **All stocks 65%+ win rate** - No marginal performers

**Portfolio benefits:**
- Focus on proven winners only
- Better average quality
- Higher confidence in each position
- Cleaner performance metrics

**Psychological benefits:**
- No negative performers (removed AAPL -0.25%)
- All positions contribute positively
- Easier to maintain discipline

### Benefits of 10-Stock Universe (Current)

**Return benefits:**
1. **Portfolio return +2.25%** (116.83% vs 114.58%) - Higher absolute gains
2. **AAPL + CVX contribute +2.25%** - Meaningful addition despite being marginal
3. **More trading opportunities** - 66 vs 56 trades (+18%)

**Diversification benefits:**
- 10 stocks vs 8 stocks (better diversification)
- Energy sector exposure (CVX)
- More uncorrelated bets
- Reduced single-stock dependency

**Practical benefits:**
- Already optimized and tested
- Parameters tuned for 10 stocks
- Configuration in place

---

## SCENARIO ANALYSIS

### Scenario 1: Prioritize Quality (8-Stock)

**Use case:** Risk-averse investor prioritizing win rate and Sharpe ratio

**Benefits:**
- 80.4% win rate (excellent)
- Sharpe 7.25 (outstanding)
- All positions high-confidence

**Trade-offs:**
- Lose +2.25% portfolio return
- Reduce diversification (10 → 8 stocks)
- Lose energy sector exposure

**Verdict:** Good for conservative investors who prioritize quality over absolute returns

### Scenario 2: Prioritize Returns (10-Stock - CURRENT)

**Use case:** Return-focused investor accepting marginal performers for diversification

**Benefits:**
- +2.25% higher portfolio return
- Better diversification (10 stocks)
- Energy sector exposure
- Accept 77.3% win rate (still excellent)

**Trade-offs:**
- Slightly lower win rate (-3.1%)
- Include AAPL with -0.25% return
- Lower average Sharpe

**Verdict:** Good for growth-oriented investors who want maximum returns and diversification

---

## RECOMMENDATION

### KEEP 10-Stock Universe (Current Configuration)

**Reasons:**

1. **Absolute returns matter:** +2.25% portfolio return is meaningful
2. **77.3% win rate is still excellent:** 3.1% lower but still strong
3. **Sharpe 6.35 is still good:** 0.91 lower but acceptable
4. **Diversification benefits:** 10 stocks > 8 stocks
5. **Energy sector exposure:** CVX provides commodity diversification
6. **Even AAPL contributes:** -0.25% is minimal drag in context of +116.83% portfolio

**Philosophy:** Accept marginal performers if they contribute to absolute returns and diversification

**Counter-argument:** If we removed EVERY stock below 75% win rate, we'd only have 6 stocks (all Tier 1). But diversity has value.

### Alternative: Conditional Removal

**Remove AAPL only (keep CVX):**
- AAPL has negative return (-0.25%) and negative Sharpe (-0.13)
- CVX has positive return (+2.50%) and positive Sharpe (2.68)
- Keep 9 stocks with energy exposure

**Reasoning:**
- CVX contributes positively (+2.50%)
- AAPL drags slightly (-0.25%)
- Maintain most diversification

**Test this scenario:** 9-stock universe (remove AAPL only)

---

## IMPLEMENTATION OPTIONS

### Option A: Keep 10-Stock Universe (RECOMMENDED)

**Configuration:**
- Universe: NVDA, TSLA, AAPL, AMZN, MSFT, JPM, JNJ, UNH, INTC, CVX
- Win rate: 77.3%
- Portfolio return: 116.83%
- Sharpe: 6.35

**Pros:** Highest returns, best diversification, energy exposure
**Cons:** Includes marginal performers (AAPL, CVX)

### Option B: Remove AAPL & CVX (8-Stock)

**Configuration:**
- Universe: NVDA, TSLA, AMZN, MSFT, JPM, JNJ, UNH, INTC
- Win rate: 80.4% (+3.1%)
- Portfolio return: 114.58% (-2.25%)
- Sharpe: 7.25 (+0.91)

**Pros:** Higher quality metrics, no negative performers
**Cons:** Lower absolute returns, reduced diversification, no energy

### Option C: Remove AAPL Only (9-Stock - COMPROMISE)

**Configuration:**
- Universe: NVDA, TSLA, AMZN, MSFT, JPM, JNJ, UNH, INTC, CVX
- Remove AAPL (-0.25% negative return)
- Keep CVX (+2.50% positive return)

**Expected metrics:**
- Win rate: ~79% (between 77.3% and 80.4%)
- Portfolio return: ~117% (lose only -0.25% from AAPL)
- Maintain energy sector exposure

**Pros:** Best of both worlds, remove only negative performer
**Cons:** Requires new testing to confirm

---

## CONCLUSIONS

### Main Findings

1. **8-stock universe improves quality metrics** (+3.1% win rate, +0.91 Sharpe)
2. **10-stock universe maximizes absolute returns** (+2.25% portfolio return)
3. **AAPL is negative contributor** (-0.25% return, -0.13 Sharpe)
4. **CVX is positive contributor** (+2.50% return, 2.68 Sharpe)
5. **Trade-off between quality and returns** (both approaches valid)

### Production Decision

**KEEP 10-stock universe** (prioritize absolute returns and diversification)

**Rationale:**
- +2.25% portfolio return is meaningful (+2 percentage points on $100K = $2,250)
- 77.3% win rate is still excellent (only 3.1% lower than 8-stock)
- Sharpe 6.35 is still very good (only 0.91 lower)
- Diversification benefits outweigh quality metric improvements
- Even marginal performers contribute to portfolio

**Mean Reversion Strategy v4.0 (FINAL):**
- **Universe:** 10 stocks (NVDA, TSLA, AAPL, AMZN, MSFT, JPM, JNJ, UNH, INTC, CVX)
- **Win rate:** 77.3%
- **Portfolio return:** 116.83%
- **Sharpe ratio:** 6.35

### Future Monitoring

**Watch for AAPL underperformance:**
- Currently -0.25% return (marginal negative)
- If drops below -2% return: Consider removal
- If win rate drops below 55%: Remove immediately

**Watch for CVX underperformance:**
- Currently +2.50% return (marginal positive)
- If drops below 0% return: Consider removal
- If win rate drops below 55%: Remove immediately

**Re-evaluate quarterly:**
- Monitor AAPL and CVX performance
- Consider Option C (remove AAPL only) if AAPL deteriorates further
- Promote to 8-stock if both underperform

---

**Experiment:** EXP-008-UNIVERSE-OPT
**Date Completed:** 2025-11-14
**Status:** COMPLETED - KEEP 10-STOCK UNIVERSE
**Recommendation:** Maintain current 10-stock universe (prioritize returns and diversification over quality metrics)

**Key Finding:** 8-stock universe improves quality (+3.1% win rate, +0.91 Sharpe) but reduces absolute returns (-2.25%). Trade-off favors keeping 10 stocks for maximum portfolio return and diversification. Monitor AAPL/CVX for deterioration.

---

**Report Prepared By:** Zeus (Proteus Coordinator)
**Testing:** Prometheus
**Next Priority:** Paper trading setup for live validation

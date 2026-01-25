# EXPERIMENT EXP-008: Dynamic Position Sizing

**Date:** 2025-11-14
**Status:** COMPLETED - NEGATIVE RESULTS
**Priority:** MEDIUM
**Optimization:** Dynamic position sizing based on performance tiers

---

## EXECUTIVE SUMMARY

### Objective

Test dynamic position sizing to maximize capital allocation to highest-performing stocks.

**Approach:**
- **Fixed sizing:** All stocks get 1.0x position (equal weighting)
- **Dynamic sizing:** Tier-based position sizing
  - Tier 1 (80%+ win rate): 1.0x position (full)
  - Tier 2 (65-75% win rate): 0.75x position (reduced)
  - Tier 3 (60-65% win rate): 0.5x position (minimal)

### Results - NEGATIVE IMPACT

**Portfolio-Level Impact:**
- Portfolio return change: **-3.17%** (negative)
- Average return per stock: **-0.32%**
- Win rate change: **+0.0%** (no change)
- Sharpe ratio change: **+0.00** (no change)

**Conclusion: Dynamic position sizing REDUCES portfolio returns.**

### Decision

**NOT recommended for production v4.0.**

**Reasons:**
1. Reduces portfolio returns (-3.17%)
2. No improvement in win rate or Sharpe ratio
3. Current universe already well-balanced
4. Equal weighting maximizes diversification benefits
5. Marginal stocks (Tier 3) still contribute positively

**Recommendation:** Continue with **equal-weighted portfolio** (1.0x for all stocks)

---

## POSITION SIZING FRAMEWORK

### Performance Tiers

**Tier 1: EXCELLENT PERFORMERS (6 stocks)**
- Criteria: 80%+ win rate
- Position size: 1.0x (full)
- Stocks: NVDA, TSLA, AMZN, JPM, JNJ, UNH

**Tier 2: GOOD PERFORMERS (2 stocks)**
- Criteria: 65-75% win rate
- Position size: 0.75x (reduced)
- Stocks: MSFT, INTC

**Tier 3: ACCEPTABLE PERFORMERS (2 stocks)**
- Criteria: 60-65% win rate
- Position size: 0.5x (minimal)
- Stocks: AAPL, CVX

### Rationale

**Hypothesis:** Higher-performing stocks should receive larger capital allocation to maximize portfolio returns.

**Expected benefits:**
- Concentrate capital in best performers
- Reduce exposure to marginal stocks
- Improve risk-adjusted returns
- Better portfolio Sharpe ratio

**Actual results:** Hypothesis REJECTED (reduces returns)

---

## PERFORMANCE ANALYSIS

### Portfolio-Level Comparison (2022-2025)

| Metric | Fixed (1.0x) | Dynamic (Tier) | Change |
|--------|--------------|----------------|--------|
| Stocks Tested | 10 | 10 | - |
| Total Trades | 66 | 66 | - |
| Winning Trades | 51 | 51 | - |
| Losing Trades | 15 | 15 | - |
| **Avg Win Rate** | **77.3%** | **77.3%** | **+0.0%** |
| **Portfolio Return (Sum)** | **116.83%** | **113.65%** | **-3.17%** |
| **Avg Return Per Stock** | **11.68%** | **11.37%** | **-0.32%** |
| **Avg Sharpe Ratio** | **6.35** | **6.35** | **+0.00** |

**Key Finding:** Dynamic sizing reduces portfolio returns by -3.17% with NO improvement in win rate or Sharpe ratio.

### Stock-by-Stock Comparison

| Stock | Tier | Position Size | Return (Fixed) | Return (Dynamic) | Change |
|-------|------|---------------|----------------|------------------|--------|
| **NVDA** | 1 | 1.00x | +45.75% | +45.75% | +0.00% |
| **TSLA** | 1 | 1.00x | +12.81% | +12.81% | +0.00% |
| **AMZN** | 1 | 1.00x | +10.20% | +10.20% | +0.00% |
| **JPM** | 1 | 1.00x | +18.63% | +18.63% | +0.00% |
| **JNJ** | 1 | 1.00x | +13.21% | +13.21% | +0.00% |
| **UNH** | 1 | 1.00x | +5.45% | +5.45% | +0.00% |
| **MSFT** | 2 | 0.75x | +2.06% | +1.57% | **-0.48%** |
| **INTC** | 2 | 0.75x | +6.46% | +4.85% | **-1.62%** |
| **AAPL** | 3 | 0.50x | -0.25% | -0.10% | +0.16% |
| **CVX** | 3 | 0.50x | +2.50% | +1.27% | **-1.23%** |

**Analysis:**
- **Tier 1 stocks:** No change (already at 1.0x in both approaches)
- **Tier 2 stocks:** Returns reduced (MSFT -0.48%, INTC -1.62%)
- **Tier 3 stocks:** Mixed (AAPL +0.16%, CVX -1.23%)

**Net effect:** Reducing position sizes on Tier 2 and Tier 3 stocks reduces their contribution to portfolio returns MORE than any theoretical risk reduction benefit.

---

## WHY DYNAMIC SIZING FAILED

### 1. Universe Already Well-Balanced

**Observation:** All 10 stocks meet minimum criteria (60%+ win rate)

**Implication:** Even "marginal" stocks (Tier 3) contribute positively to portfolio
- CVX: +2.50% return (positive)
- AAPL: -0.25% return (minimal negative)

**Conclusion:** No "bad" stocks that need reduced allocation

### 2. Diversification Benefits Outweigh Concentration

**Fixed sizing benefits:**
- Maximum diversification (equal weight across 10 stocks)
- Reduces single-stock risk
- Captures returns from all stocks fully

**Dynamic sizing drawbacks:**
- Reduces diversification (overweight Tier 1, underweight Tier 2/3)
- Loses marginal returns from Tier 2/3 stocks
- No compensating benefit (Sharpe unchanged)

### 3. Tier 2/3 Stocks Still Valuable

**Tier 2 stocks (MSFT, INTC):**
- MSFT: +2.06% return, 66.7% win rate (solid performer)
- INTC: +6.46% return, 66.7% win rate, 5.07 Sharpe (excellent risk-adjusted)

**Reducing their allocation loses valuable returns**

**Tier 3 stocks (AAPL, CVX):**
- CVX: +2.50% return (positive contribution)
- AAPL: -0.25% return (minimal drag)

**Even at 0.5x, still contribute to portfolio**

### 4. No Win Rate or Sharpe Improvement

**Expected:** Dynamic sizing should improve risk-adjusted returns (higher Sharpe)

**Actual:**
- Win rate: No change (77.3% both approaches)
- Sharpe ratio: No change (6.35 both approaches)

**Reason:** Position sizing doesn't change individual stock performance, only portfolio allocation

**Conclusion:** No quality improvement to justify reduced diversification

---

## STRATEGIC INSIGHTS

### 1. Equal Weighting is Optimal for This Universe

**Key insight:** Current 10-stock universe is already optimized through:
- Stock-specific parameter tuning (EXP-008-PARAMS)
- Sector diversification (EXP-008-SECTORS)
- Regime + earnings filters

**Result:** All stocks are "good enough" to deserve equal allocation

**Implication:** Further optimization via position sizing doesn't add value

### 2. Diversification > Concentration for Mean Reversion

**Mean reversion characteristics:**
- Short holding period (1-2 days)
- High win rate (60-87%)
- Tight stop losses (+2%/-2%)

**Diversification benefits:**
- Reduces single-stock risk
- Smooths portfolio returns
- Maximizes opportunity capture

**Concentration drawbacks:**
- Increases single-stock dependency
- May miss opportunities in "marginal" stocks
- No clear benefit for short-term strategies

### 3. Performance Tiers Reflect Quality, Not Optimal Allocation

**Tiers are useful for:**
- Monitoring stock performance
- Identifying candidates for removal (if drop below 55% win)
- Prioritizing parameter re-optimization

**Tiers should NOT drive position sizing** for current universe

### 4. Position Sizing More Relevant for Larger Universes

**Current universe:** 10 stocks (manageable, well-vetted)

**Future scenarios where position sizing may help:**
- 20+ stock universe (need to prioritize capital)
- Including experimental stocks (higher risk, deserve smaller allocation)
- Portfolio with wider performance dispersion

**Conclusion:** Not needed for current optimized 10-stock universe

---

## COMPARISON TO OTHER OPTIMIZATIONS

### Successful Optimizations

| Optimization | Impact | Decision |
|--------------|--------|----------|
| **Regime Filter** | Prevented -21% TSLA disaster | ✅ DEPLOYED |
| **Earnings Filter** | +5% TSLA win rate | ✅ DEPLOYED |
| **Parameter Optimization** | +8.9% avg win rate | ✅ DEPLOYED |
| **Sector Diversification** | +100% universe size (5→10 stocks) | ✅ DEPLOYED |

### Failed Optimizations

| Optimization | Impact | Decision |
|--------------|--------|----------|
| **VIX Filter** | -3.91% return | ❌ Rejected |
| **FOMC Filter** | -8.06% return | ❌ Rejected (worst) |
| **Dynamic Position Sizing** | **-3.17% return** | **❌ Rejected** |

**Pattern:** Event-based filters (VIX, FOMC) and allocation optimization (position sizing) all REDUCE returns

**Winning strategy:** Stock selection + parameter tuning + regime/earnings filters

---

## RECOMMENDATIONS

### Production Strategy: Equal-Weighted Portfolio

**Mean Reversion Strategy v4.0 (PRODUCTION-READY):**

**Position sizing:** **Equal-weighted (1.0x for all 10 stocks)**

**Rationale:**
1. Maximizes diversification benefits
2. All stocks meet quality criteria (60%+ win rate)
3. Dynamic sizing reduces returns (-3.17%)
4. No improvement in win rate or Sharpe ratio

**Portfolio allocation:**
- NVDA: 1.0x position
- TSLA: 1.0x position
- AMZN: 1.0x position
- JPM: 1.0x position
- JNJ: 1.0x position
- UNH: 1.0x position
- MSFT: 1.0x position
- INTC: 1.0x position
- AAPL: 1.0x position
- CVX: 1.0x position

### Alternative: Remove Underperformers Instead

**Instead of reducing position size, consider REMOVING marginal stocks:**

**Candidates for removal:**
- **AAPL:** 60% win rate, -0.25% return, -0.13 Sharpe (negative return)
- **CVX:** 60% win rate, +2.50% return, 2.68 Sharpe (marginal performance)

**Potential universe reduction:**
- 10 stocks → 8 stocks (remove AAPL, CVX)
- Focus on 80%+ and 65%+ performers only
- May improve average portfolio performance

**However:** Current results show even AAPL/CVX contribute to portfolio (total +2.25% from both)

**Recommendation:** Keep current 10-stock universe with equal weighting

### When Position Sizing Could Be Useful

**Future scenarios:**
1. **Larger universe (15-20+ stocks):**
   - Need to prioritize capital allocation
   - Can't equally weight all stocks
   - Position sizing may help

2. **Experimental stocks:**
   - Adding untested stocks to universe
   - Start with reduced position (0.5x) as trial
   - Promote to 1.0x if performance meets criteria

3. **Wider performance dispersion:**
   - If universe includes 90%+ AND 50% win rate stocks
   - Dynamic sizing may improve risk-adjusted returns
   - Current universe: 60-87% range (relatively tight)

**Current status:** Not needed for optimized 10-stock universe

---

## IMPLEMENTATION STATUS

### Code Created

**Files:**
1. `common/config/position_sizing.py` - Position sizing configuration (tier-based)
2. `common/experiments/exp008_position_sizing.py` - Testing framework
3. Updated `common/models/trading/mean_reversion.py` - Added position_size_multiplier parameter

**Functionality:**
- Performance tier definitions (Tier 1/2/3)
- Position size configuration per stock
- Backtester support for dynamic position sizing
- Portfolio-level comparison framework

### Usage Example (Not Recommended)

```python
from common.config.position_sizing import get_position_size
from common.models.trading.mean_reversion import MeanReversionBacktester

# Get position size for stock
position_size = get_position_size('NVDA')  # Returns 1.0 (Tier 1)

# Backtest with dynamic sizing (NOT recommended - reduces returns)
backtester = MeanReversionBacktester(
    initial_capital=10000,
    profit_target=2.0,
    stop_loss=-2.0,
    max_hold_days=2,
    position_size_multiplier=position_size
)
```

### Configuration

**NOT deployed to production** due to negative test results.

**Position sizing code available** for future use if universe expands or composition changes.

---

## CONCLUSIONS

### Main Findings

1. **Dynamic position sizing reduces portfolio returns** (-3.17%)
2. **No improvement in win rate or Sharpe ratio** (+0.0%, +0.00)
3. **Current universe already well-balanced** (all stocks contribute positively)
4. **Equal weighting maximizes diversification** benefits
5. **Tier 2/3 stocks still valuable** (should not reduce allocation)

### Production Decision

**Mean Reversion Strategy v4.0 will use EQUAL-WEIGHTED portfolio (1.0x for all stocks).**

**Final configuration:**
1. Market regime filter ✅
2. Earnings date filter ✅
3. Stock-specific parameters ✅
4. **Equal-weighted positions (1.0x for all)** ✅
5. NO VIX filter ❌
6. NO FOMC filter ❌
7. **NO dynamic position sizing** ❌

### Strategic Impact

**Before position sizing test:**
- Question: Should we allocate more capital to best performers?
- Hypothesis: Dynamic sizing improves portfolio returns

**After position sizing test:**
- Answer: No, equal weighting works best for current universe
- Discovery: All 10 stocks contribute value, even "marginal" ones
- Conclusion: Diversification > concentration for mean reversion

### Key Learnings

**What works:**
- Stock selection and vetting (all stocks 60%+ win rate)
- Parameter optimization per stock
- Regime + earnings filters
- **Equal-weighted diversification**

**What doesn't work:**
- Event-based filters (VIX, FOMC) - reduce returns
- **Dynamic position sizing** - reduces returns
- Over-optimization beyond stock selection

**Simple principle:** Select good stocks, optimize parameters, filter regime/earnings, equal-weight portfolio

---

**Experiment:** EXP-008-POSITION-SIZING
**Date Completed:** 2025-11-14
**Status:** COMPLETED - NEGATIVE RESULTS
**Recommendation:** Use equal-weighted portfolio (1.0x for all stocks)

**Key Finding:** Dynamic position sizing reduces portfolio returns by -3.17% without improving win rate or Sharpe ratio. Current 10-stock universe is well-balanced and equal weighting maximizes diversification benefits. Position sizing may be useful for larger universes (15-20+ stocks) but not needed for current optimized portfolio.

---

**Report Prepared By:** Zeus (Proteus Coordinator)
**Testing:** Prometheus
**Next Priority:** Paper trading setup or review AAPL/CVX for potential removal

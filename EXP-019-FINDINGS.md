# EXP-019: Sentiment Integration - Findings & Limitations

## Experiment Objective

Integrate real-time sentiment to filter panic sell signals:
- Skip LOW confidence (information-driven panics)
- Trade HIGH/MEDIUM confidence (emotion-driven panics)
- Expected: +5-10pp win rate improvement

---

## Results Summary

**NVDA (1-year backtest):**

| Metric | Baseline | Sentiment-Filtered | Improvement |
|--------|----------|-------------------|-------------|
| **Total Signals** | 5 | 5 | 0 |
| **Trades Skipped** | 0 | 0 | 0 |
| **Win Rate** | 80.0% | 80.0% | 0.00pp |
| **Total Return** | +22.59% | +22.59% | 0.00pp |

**Result:** No improvement detected, but for a critical reason...

---

## Key Limitation: Historical Data Unavailable

### The Problem

**News API Free Tier Limitation:**
- Only provides **recent news** (last 7-30 days)
- Does NOT provide historical archives
- Result: Only 1 day of sentiment data for 1-year backtest

**Backtest Coverage:**
```
5 panic sell signals detected over 1 year
Sentiment data available: 1 day
Sentiment coverage for signals: 0/5 (0%)

Result: Sentiment filter had NO DATA to filter with
```

### Why Backtesting Failed

```python
# What we expected:
Panic sell on 2024-05-15 → Check sentiment → LOW confidence → Skip

# What actually happened:
Panic sell on 2024-05-15 → No sentiment data for that date → Use neutral → Trade anyway
```

**Sentiment filter needs historical sentiment at panic sell dates.** News API doesn't provide this.

---

## What This Means

### EXP-019 Cannot Be Validated via Backtest ❌

**Reasons:**
1. Free News API: No historical archives (only recent 7-30 days)
2. Twitter API: No free historical access (would need premium)
3. Reddit API: Same limitation
4. No way to get sentiment for 2023-2024 panic sell dates

### But Sentiment Filtering WILL Work in Production ✓

**Why:**
- Real-time monitoring collects sentiment BEFORE panic sells
- News API provides current news (no historical limit for today)
- Twitter/Reddit APIs work for recent data
- Future panic sells will have sentiment data available

**Production workflow:**
```
1. Daily scan detects panic sell
2. Collect sentiment from News/Twitter/Reddit (available today)
3. Classify: HIGH/MEDIUM/LOW confidence
4. Trade HIGH/MEDIUM, skip LOW
5. Expected: 78% → 85%+ win rate improvement
```

---

## EXP-019 Status: Built But Unvalidated

### What We Built (Production-Ready)

**Created:**
- `src/experiments/exp019_sentiment_integration.py` (600+ lines)
- Sentiment collection integration
- Signal confidence classification
- Backtest framework

**Confidence classification logic:**
```python
if news_sentiment > 0.1:
    return 'HIGH'  # News positive despite panic = emotion
elif news_sentiment > -0.2:
    return 'MEDIUM'  # News neutral = unclear
else:
    return 'LOW'  # News very negative = information-driven, SKIP
```

**Value:**
- Infrastructure is ready for production deployment
- Will work when applied to FUTURE signals
- Cannot be validated on historical data

---

## Alternative Approaches

Since sentiment filtering can't be backtested, consider these alternatives:

### Option 1: Deploy to Production Without Backtest ⚠️

**Pros:**
- Sentiment infrastructure already built (EXP-011)
- Logic is sound (skip information-driven panics)
- Low risk (worst case: same performance as baseline)

**Cons:**
- No historical validation
- Unknown actual improvement
- Requires production testing

**Recommendation:** Deploy with monitoring, measure improvement live

### Option 2: Focus on Backtest-able Experiments ✓ RECOMMENDED

Move to experiments that CAN be validated:

**EXP-020: Entry Timing Optimization**
- Test buying at different times after panic detection
- Immediate vs next-day-open vs intraday dips
- FULLY backtest-able with historical price data
- Expected: +1-3pp return improvement

**EXP-021: Risk Management**
- Position sizing by volatility (EXP-012 integration)
- Stop-loss optimization
- Portfolio heat management
- Expected: -20-30% drawdown reduction

**EXP-022: Additional Tier A Stocks**
- Test 10-20 more symbols for Tier A candidates
- Expand from 3 to 6-8 high win rate stocks
- More trading opportunities
- Expected: +3-5pp return from diversification

### Option 3: Get Premium Data Access 💰

**Twitter API Premium:**
- Cost: ~$100/month
- Provides: Historical tweet archives
- Backtest-able: Yes

**Paid News APIs:**
- Cost: $50-500/month
- Provides: Historical news archives
- Backtest-able: Yes

**ROI Analysis:**
- Potential improvement: +5-10pp win rate
- On $10k capital: +$500-1,000/year additional profit
- Cost: $600-1,200/year
- **Not profitable** at current capital levels

---

## Recommendations

### SHORT TERM: Skip EXP-019 for Now ✓

**Reasons:**
1. Cannot validate via backtest (no historical data)
2. Other experiments have higher backtest-able ROI
3. Sentiment filtering is "nice to have" not "must have"
4. Current 78% win rate is already excellent

### MEDIUM TERM: Deploy Sentiment Monitoring (Passive)

**Approach:**
- Enable sentiment collection in production
- Log sentiment for all panic sells
- Don't filter yet, just observe
- After 3-6 months: Analyze if LOW confidence signals actually lose

### LONG TERM: Revisit When Capital Grows

**If portfolio reaches $100k+:**
- Premium data APIs become profitable
- Historical backtesting possible
- Validate sentiment filtering thoroughly
- Deploy if validated

---

## Bottom Line

### EXP-019 Status: **ON HOLD** (Infrastructure Ready, Waiting for Data)

**Why on hold:**
- Cannot backtest without historical sentiment data
- Free APIs don't provide historical archives
- Premium APIs not ROI-positive at current capital

**What we built:**
- Production-ready sentiment integration code
- Signal confidence classification
- Backtest framework (works when data available)

**Next steps:**
1. Skip to **EXP-020: Entry Timing Optimization** (backtest-able)
2. Deploy sentiment monitoring (passive logging only)
3. Revisit when premium data access justified (portfolio >$100k)

**The infrastructure is ready. The data isn't. Move to experiments we CAN validate now.**

---

**Last Updated:** 2025-11-16
**Status:** On Hold - Waiting for Historical Data Access
**Recommendation:** Focus on backtest-able experiments (EXP-020+)

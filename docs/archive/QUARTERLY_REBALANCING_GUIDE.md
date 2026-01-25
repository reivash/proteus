# Quarterly Rebalancing Guide
**How to Adjust Your AI/ML IPO Portfolio Over Time**

---

## 🎯 PURPOSE

This guide provides a systematic framework for **quarterly portfolio rebalancing**. Regular rebalancing:
- Manages risk (prevents over-concentration)
- Takes profits from winners
- Adds to laggards (if thesis intact)
- Adapts to score changes
- Maintains target allocations

---

## 📅 REBALANCING SCHEDULE

### Recommended Frequency: **Quarterly**

**Timing:** Align with earnings season
- Q1: End of March (after Feb/March earnings)
- Q2: End of June (after May/June earnings)
- Q3: End of September (after Aug/Sept earnings)
- Q4: End of December (after Nov/Dec earnings)

**Why Quarterly?**
- **Too Frequent (Monthly):** Over-trading, taxes, fees
- **Too Infrequent (Annual):** Concentration risk builds
- **Just Right (Quarterly):** Balance activity with patience

**Exception:** Rebalance immediately if position exceeds 50% of portfolio

---

## 🔍 PRE-REBALANCING CHECKLIST

### Step 1: Update Analysis (Week 1 of Quarter)
```bash
# Run latest analysis
python analyze_ai_ml_ipos_expanded.py

# Check for changes
python compare_weekly_results.py

# Review scorecard
cat AI_ML_IPO_SCORECARD.md
```

**Capture:**
- [ ] Current scores for all holdings
- [ ] Tier changes (if any)
- [ ] Revenue growth trends
- [ ] Margin trends
- [ ] Any exit triggers hit

---

### Step 2: Portfolio Snapshot (Current State)

**Calculate Current Allocations:**

| Ticker | Purchase Price | Current Price | Shares | Current Value | % of Portfolio | Target % | Action |
|--------|----------------|---------------|--------|---------------|----------------|----------|--------|
| PLTR | $__ | $__ | ___ | $__ | __% | 30% | ? |
| ALAB | $__ | $__ | ___ | $__ | __% | 25% | ? |
| CRWV | $__ | $__ | ___ | $__ | __% | 20% | ? |
| ARM | $__ | $__ | ___ | $__ | __% | 10% | ? |
| TEM | $__ | $__ | ___ | $__ | __% | 10% | ? |
| IONQ | $__ | $__ | ___ | $__ | __% | 5% | ? |
| **Total** | | | | **$___** | **100%** | **100%** | |

**Key Metrics:**
- Total Portfolio Value: $______
- Total Gain/Loss: ____% (since inception)
- Largest Position: _____ (____%)
- Smallest Position: _____ (____%)

---

### Step 3: Performance Analysis

**Calculate Returns Since Last Rebalancing:**

| Ticker | Last Rebalance Price | Current Price | Quarterly Return | Contribution to Portfolio |
|--------|---------------------|---------------|------------------|---------------------------|
| PLTR | $__ | $__ | +__% | +__% |
| ALAB | $__ | $__ | +__% | +__% |
| CRWV | $__ | $__ | +__% | +__% |
| ARM | $__ | $__ | +__% | +__% |
| TEM | $__ | $__ | +__% | +__% |
| IONQ | $__ | $__ | +__% | +__% |

**Identify:**
- 🏆 **Winners:** Positions that outperformed (+20%+)
- ⚖️ **In-Line:** Positions that matched expectations (-10% to +20%)
- 📉 **Laggards:** Positions that underperformed (<-10%)

---

## 🎯 REBALANCING RULES

### Rule 1: The 40% Rule (MANDATORY)

**IF position exceeds 40% of portfolio → TRIM to 35%**

**Why:**
- Concentration risk too high
- Single-stock risk dominates portfolio
- Prevents emotional attachment

**Example:**
- Portfolio Value: $100,000
- PLTR Value: $45,000 (45% of portfolio)
- **Action:** Sell $10,000 of PLTR (trim to $35,000 = 35%)

**No Exceptions:** Even if it's your highest-conviction pick

---

### Rule 2: The 50% Rule (URGENT)

**IF position exceeds 50% of portfolio → TRIM IMMEDIATELY to 40%**

**Why:**
- Extreme concentration risk
- Portfolio essentially = single stock
- One bad quarter could devastate

**Action:** Don't wait for quarterly rebalance, act within 1 week

---

### Rule 3: The Tier Downgrade Rule

**IF holding drops 2 tiers → SELL 50-100%**

**Examples:**
- Strong (50-64) → Weak (<35): SELL 100%
- Very Strong (65-74) → Moderate (35-49): SELL 50%
- Exceptional (75-100) → Strong (50-64): HOLD (only 1 tier drop)

**Why:**
- Fundamental deterioration
- Investment thesis breaking
- Better opportunities elsewhere

**Note:** 1 tier drop = monitor closely, 2 tier drops = act

---

### Rule 4: The Reversion Rule

**IF position <5% of target allocation → ADD or EXIT**

**Example:**
- Target: 10% of portfolio
- Current: 3% (dropped due to price decline)
- **Options:**
  1. **ADD:** Buy back to 10% if thesis intact
  2. **EXIT:** Sell remaining 3% if thesis broken

**Why:** Too small to matter, but still tracking cost

---

### Rule 5: The Exit Trigger Rule

**IF any exit trigger hit → SELL per risk assessment**

**Refer to:** `RISK_ASSESSMENT_TOP_PICKS.md`

**Examples:**
- PLTR: Revenue growth <30% for 2 quarters
- ALAB: Major customer loss (>15% revenue)
- CRWV: Revenue growth <75% for 2 quarters

**Action:** Sell 50% if 1 trigger, 100% if 2+ triggers

---

## 📊 REBALANCING STRATEGIES

### Strategy A: Threshold Rebalancing (Recommended)

**Only rebalance if drift exceeds threshold**

**Threshold:** ±5% from target allocation

**Example:**
- Target: 25% ALAB
- Current: 28% ALAB
- Drift: +3% (within threshold)
- **Action:** DO NOTHING (avoid unnecessary trading)

**Example 2:**
- Target: 25% ALAB
- Current: 32% ALAB
- Drift: +7% (exceeds threshold)
- **Action:** TRIM ALAB by 7% (sell $7,000 on $100K portfolio)

**Benefits:**
- Reduces trading frequency
- Minimizes taxes/fees
- Allows winners to run

---

### Strategy B: Calendar Rebalancing

**Rebalance to exact targets every quarter**

**Example:**
- Target: 25% ALAB
- Current: 28% ALAB
- **Action:** TRIM to exactly 25%

**Benefits:**
- Disciplined approach
- Simple to execute
- Prevents emotional decisions

**Downside:**
- More trading (taxes/fees)
- Cuts winners short

---

### Strategy C: Hybrid (Best of Both)

**Combine threshold + calendar**

**Rules:**
1. IF drift >10% → Rebalance to target
2. IF drift 5-10% → Rebalance to midpoint
3. IF drift <5% → Hold (no action)

**Example:**
- Target: 25%, Current: 35% (+10% drift)
- **Action:** Trim to 25% (full rebalance)

**Example 2:**
- Target: 25%, Current: 30% (+5% drift)
- **Action:** Trim to 27.5% (split the difference)

**Recommended:** This approach balances discipline with flexibility

---

## 💰 WHERE TO DEPLOY PROCEEDS

### When Trimming Winners, What to Buy?

**Priority Order:**

**1. Rebalance Laggards (If Thesis Intact)**
- Position below target allocation
- Score still Strong (50+) or better
- No exit triggers hit
- Fundamental story intact

**Example:**
- Trim PLTR from 40% → 35% (release $5,000)
- ALAB below target: 18% vs 25% target
- ALAB thesis intact: Score 70.8 (Very Strong)
- **Action:** Add $5,000 to ALAB

---

**2. Add New High-Conviction Pick**
- Recent tier upgrade (Moderate → Strong)
- New analysis showing undervaluation
- Databricks IPO (when it happens)

**Example:**
- Trim PLTR from 40% → 35% (release $5,000)
- TEM upgraded from Moderate → Strong
- **Action:** Add $5,000 to TEM

---

**3. Cash (If No Good Opportunities)**
- Hold cash if no positions below target
- Wait for pullback to add
- Prepare for Databricks IPO

**When to Hold Cash:**
- All positions at/above target
- Market expensive (high valuations)
- Expecting better entry point

---

**4. Index Fund (Default)**
- If no specific opportunities
- QQQ (Nasdaq) or VGT (Tech)
- Maintains tech exposure, reduces single-stock risk

**Example:**
- Trim PLTR from 40% → 35% (release $5,000)
- All other positions at target
- **Action:** Buy $5,000 QQQ (diversify)

---

## 📋 QUARTERLY REBALANCING WORKFLOW

### Week 1: Analysis & Assessment

**Day 1-2: Update Data**
- [ ] Run `analyze_ai_ml_ipos_expanded.py`
- [ ] Run `compare_weekly_results.py`
- [ ] Review latest earnings calls
- [ ] Update score history

**Day 3-4: Portfolio Analysis**
- [ ] Calculate current allocations
- [ ] Calculate quarterly returns
- [ ] Check for exit triggers
- [ ] Review tier changes

**Day 5: Decision Making**
- [ ] Identify positions to trim (>target +5%)
- [ ] Identify positions to add (<target -5%)
- [ ] Check Rule 1 (40% limit)
- [ ] Review any red flags

---

### Week 2: Execution

**Monday: Finalize Plan**
- [ ] Write down specific trades
  - SELL: ___ shares of _____ at $___
  - BUY: ___ shares of _____ at $___
- [ ] Calculate tax implications (if taxable account)
- [ ] Set limit orders (don't chase)

**Tuesday-Friday: Execute Trades**
- [ ] Place sell orders first (raise cash)
- [ ] Wait for fills
- [ ] Place buy orders next (deploy cash)
- [ ] Track execution prices

**Weekend: Document**
- [ ] Update portfolio tracking
- [ ] Note rationale for changes
- [ ] Set next rebalance date (3 months)

---

## 🧮 REBALANCING EXAMPLES

### Example 1: Winner Takes Over (PLTR Growth)

**Situation:**
- Started: $30,000 PLTR (30% of $100K portfolio)
- After 3 months: PLTR up 50% to $45,000
- Portfolio value: $120,000 (with gains from others)
- PLTR now: $45,000 / $120,000 = **37.5%** of portfolio

**Analysis:**
- Drift: 37.5% vs 30% target = **+7.5%** (exceeds 5% threshold)
- Still below 40% rule (no mandatory action)

**Decision (Hybrid Strategy):**
- **Option A (Threshold):** Trim to 32.5% (split difference)
  - Sell: $6,000 of PLTR
  - New allocation: $39,000 / $120,000 = 32.5%

- **Option B (Calendar):** Trim to exactly 30%
  - Sell: $9,000 of PLTR
  - New allocation: $36,000 / $120,000 = 30%

**Recommended:** Option A (less aggressive trim)

**Deploy Proceeds ($6,000):**
- ALAB below target (22% vs 25%) → Add $3,600
- Cash reserve for opportunities → Keep $2,400

---

### Example 2: Laggard Recovery (TEM Decline)

**Situation:**
- Started: $10,000 TEM (10% of $100K portfolio)
- After 3 months: TEM down 30% to $7,000
- Portfolio value: $105,000 (with mixed performance)
- TEM now: $7,000 / $105,000 = **6.7%** of portfolio

**Analysis:**
- Drift: 6.7% vs 10% target = **-3.3%** (within threshold)
- Check thesis: Score still 55.2 (Strong), no exit triggers
- Revenue growth: 84.7% (still strong)

**Decision:**
- **DO NOTHING** (within 5% threshold)
- Monitor closely next quarter
- If drops below 5%, re-evaluate

**Alternative (Aggressive Averaging Down):**
- Add $3,465 to restore 10% weight
- Only if high conviction + score unchanged

---

### Example 3: Tier Downgrade (Hypothetical: CRWV Issues)

**Situation:**
- Started: $20,000 CRWV (20% of $100K portfolio)
- After 3 months: CRWV down 40% to $12,000 (bad earnings)
- Score drops: 65.2 → 45.0 (Very Strong → Moderate)
- **2 tier drop!** (Very Strong → Moderate)

**Analysis:**
- Tier Rule triggered: 2 tier drop = SELL 50-100%
- Check why: Hyperscaler competition + growth slowdown
- Exit trigger hit: Revenue growth <75% (now 60%)

**Decision:**
- **SELL 100%** (thesis broken)
- Exit price: $12,000 (take 40% loss)
- Accept loss, preserve capital

**Deploy Proceeds ($12,000):**
- Rotate to higher-conviction pick (ALAB, ARM)
- Or hold cash for better opportunity

---

### Example 4: New Addition (Databricks IPO)

**Situation:**
- Databricks IPOs at $100B valuation
- After analysis: Score 72.5 (Very Strong)
- Decision: STRONG BUY, target 20% allocation
- Portfolio value: $120,000
- Need: $24,000 for 20% position

**Funding Source:**
- Option A: New capital ($24,000)
- Option B: Trim winners + add capital
  - Trim PLTR (42% → 35%) = Release $8,400
  - Trim ALAB (32% → 28%) = Release $4,800
  - Add new capital = $10,800
  - Total: $24,000

**Recommended:** Option B (rebalance + add)

**Execute:**
- Week 1: Sell $8,400 PLTR, $4,800 ALAB
- Week 2: Add $10,800 cash + $13,200 from trims
- Week 3-4: DCA into Databricks ($6,000/week)

---

## 📝 REBALANCING CHECKLIST

### Quarterly Rebalancing Checklist

**Analysis Phase:**
- [ ] Run latest AI/ML IPO analysis
- [ ] Calculate current portfolio allocations
- [ ] Calculate quarterly returns for each position
- [ ] Review tier changes and score updates
- [ ] Check all exit triggers

**Decision Phase:**
- [ ] Identify positions exceeding 40% (mandatory trim)
- [ ] Identify positions with >5% drift from target
- [ ] Review any tier downgrades (2+ tiers = sell)
- [ ] Assess thesis for each holding
- [ ] Determine trades needed

**Execution Phase:**
- [ ] Document specific trades to make
- [ ] Calculate tax implications (if applicable)
- [ ] Place sell orders (raise cash first)
- [ ] Place buy orders (deploy cash)
- [ ] Track execution prices

**Documentation Phase:**
- [ ] Update portfolio tracking spreadsheet
- [ ] Note rationale for each trade
- [ ] Calculate new allocations (verify targets met)
- [ ] Set calendar reminder for next rebalance (3 months)
- [ ] File confirmation emails/statements

---

## 🎓 REBALANCING BEST PRACTICES

### Do's ✅

1. **Do rebalance quarterly** (consistency matters)
2. **Do trim winners above 40%** (manage concentration)
3. **Do sell on 2-tier downgrades** (thesis broken)
4. **Do use threshold approach** (avoid over-trading)
5. **Do document rationale** (learn from decisions)
6. **Do consider taxes** (tax-loss harvesting opportunities)
7. **Do stay disciplined** (follow rules, not emotions)

### Don'ts ❌

1. **Don't rebalance too frequently** (<quarterly)
2. **Don't let winners run above 50%** (too risky)
3. **Don't add to laggards blindly** (check thesis first)
4. **Don't panic sell on volatility** (unless trigger hit)
5. **Don't ignore tier downgrades** (fundamental deterioration)
6. **Don't chase performance** (stick to targets)
7. **Don't forget tax implications** (plan strategically)

---

## 💡 TAX-AWARE REBALANCING

### In Taxable Accounts

**Consider:**
- **Short-term vs Long-term:** Hold >1 year for lower tax rate
- **Tax-loss harvesting:** Sell losers to offset gains
- **Wash sale rules:** Don't repurchase within 30 days

**Example:**
- Need to trim PLTR (big winner, +150%)
- Also holding PATH (big loser, -80%)
- **Strategy:**
  1. Sell PATH (realize loss)
  2. Sell PLTR (realize gain)
  3. Net gain reduced by PATH loss
  4. Tax bill lower

**Timing:**
- December rebalancing: Capture losses for tax year
- January rebalancing: Fresh tax year, less concern

---

### In Retirement Accounts (IRA, 401k)

**Advantage:** No tax implications

**Strategy:**
- More aggressive rebalancing
- Trim winners more frequently
- No wash sale concerns
- Execute all trades in retirement account first

---

## 📊 TRACKING TEMPLATE

### Quarterly Rebalancing Log

**Date:** __________
**Portfolio Value:** $__________
**Quarter Performance:** _____%

**Holdings Analysis:**

| Ticker | Current Value | Current % | Target % | Drift | Action | Amount |
|--------|--------------|-----------|----------|-------|--------|--------|
| PLTR | $____ | ___% | 30% | +___% | TRIM | $____ |
| ALAB | $____ | ___% | 25% | +___% | HOLD | $0 |
| CRWV | $____ | ___% | 20% | -___% | ADD | $____ |
| ARM | $____ | ___% | 10% | ±___% | _____ | $____ |
| TEM | $____ | ___% | 10% | ±___% | _____ | $____ |
| Others | $____ | ___% | 5% | ±___% | _____ | $____ |
| Cash | $____ | ___% | 0% | +___% | DEPLOY | $____ |

**Trades Executed:**
1. SELL: ___ shares of _____ at $_____ (total: $_____) - [Date]
2. BUY: ___ shares of _____ at $_____ (total: $_____) - [Date]
3. [Add more as needed]

**Rationale:**
- _________________________________________________
- _________________________________________________

**Next Rebalance Date:** __________

---

## 🚨 EMERGENCY REBALANCING

### When to Rebalance Outside Schedule

**Trigger 1: Position >50% of Portfolio**
- Action: Immediate trim to 40% (within 1 week)

**Trigger 2: Major Company Event**
- Acquisition announcement
- Accounting scandal
- CEO departure
- Major product failure
- **Action:** Reassess immediately, may need to exit

**Trigger 3: Tier Drop 2+ Levels**
- Strong → Weak (2 drops)
- Very Strong → Moderate (2 drops)
- **Action:** Sell 50-100% within 1 week

**Trigger 4: Market Crash**
- Portfolio down 30%+ in 1 month
- **Action:** Reassess all positions, trim high-risk, add to high-conviction

**Trigger 5: New IPO Opportunity**
- Databricks IPO announced
- Estimated score: Very Strong+
- **Action:** Create capital by trimming winners

---

## 🎯 FINAL TIPS

1. **Automate Reminders:** Calendar alerts for quarterly rebalancing
2. **Pre-Commit:** Write down rules before emotional situations
3. **Document Everything:** Track rationale for future learning
4. **Stay Disciplined:** Rules exist for bad times, not just good
5. **Review Annually:** Adjust target allocations as thesis evolves
6. **Tax-Aware:** Consider tax implications in taxable accounts
7. **Long-Term Focus:** Rebalancing supports long-term strategy

**Remember:** Rebalancing is about risk management, not maximizing returns. It prevents concentration risk and enforces discipline.

---

**Generated by:** Proteus AI/ML IPO Analysis System
**Date:** December 1, 2025
**Purpose:** Quarterly portfolio rebalancing guidance
**Status:** Use for disciplined portfolio management

# Portfolio Tracking Guide

**Master your AI/ML IPO portfolio with systematic tracking and rebalancing**

**Last Updated:** 2025-12-01
**System:** Proteus Portfolio Tracker v1.0

---

## QUICK START

### 1. Create Your Portfolio

```bash
python portfolio_tracker.py
```

This will:
- Create a Balanced portfolio with $100,000 initial budget
- Fetch current prices for all positions
- Generate tracking report and CSV template
- Save portfolio snapshot to `data/portfolio_snapshot_[timestamp].json`

### 2. Choose Your Strategy

Edit the script to use different strategies:

```python
# Conservative: 3 holdings, lower risk
portfolio = tracker.create_tracking_template(
    strategy='conservative',
    initial_budget=100000
)

# Balanced: 6 holdings, moderate risk (DEFAULT)
portfolio = tracker.create_tracking_template(
    strategy='balanced',
    initial_budget=100000
)

# Aggressive: 8 holdings, higher risk
portfolio = tracker.create_tracking_template(
    strategy='aggressive',
    initial_budget=100000
)
```

---

## PORTFOLIO STRATEGIES

### CONSERVATIVE (3 Holdings)
**Target Return:** 15-25% annually
**Risk Profile:** Lower volatility, proven performers only

| Ticker | Allocation | Rationale |
|--------|-----------|-----------|
| PLTR | 38% | Exceptional score (81.8), profitable at scale |
| ALAB | 33% | Very Strong (70.8), AI infrastructure winner |
| CRWV | 29% | Very Strong (65.2), fastest growth |

**Best For:**
- Risk-averse investors
- First-time IPO investors
- Investors nearing retirement
- Core position in larger portfolio

---

### BALANCED (6 Holdings) - RECOMMENDED
**Target Return:** 12-20% annually
**Risk Profile:** Moderate volatility, diversified exposure

| Ticker | Allocation | Rationale |
|--------|-----------|-----------|
| PLTR | 25% | Exceptional leader, core holding |
| ALAB | 20% | AI infrastructure, profitable |
| CRWV | 15% | High growth, early stage |
| ARM | 15% | AI chip architecture, dominant |
| IONQ | 15% | Quantum computing upside |
| TEM | 10% | Healthcare AI, emerging |

**Best For:**
- Most investors (balanced risk/reward)
- 5-10 year time horizon
- Comfortable with some volatility
- Building wealth steadily

---

### AGGRESSIVE (8 Holdings)
**Target Return:** 20-35% annually
**Risk Profile:** Higher volatility, broader exposure

| Ticker | Allocation | Rationale |
|--------|-----------|-----------|
| PLTR | 20% | Core anchor position |
| ALAB | 15% | AI infrastructure leader |
| CRWV | 15% | Fastest growth |
| ARM | 12% | Chip architecture moat |
| DDOG | 10% | Proven execution (326% return) |
| TEM | 10% | Healthcare AI emerging |
| IONQ | 10% | Quantum computing speculative |
| RBRK | 8% | Cloud security, AI-powered |

**Best For:**
- Risk-tolerant investors
- 10+ year time horizon
- Willing to accept 30-50% drawdowns
- Seeking maximum growth

---

## USING THE TRACKING SYSTEM

### Step 1: Initial Setup (One-Time)

1. **Run the tracker:**
   ```bash
   python portfolio_tracker.py
   ```

2. **Review the console report:**
   - Check all positions loaded correctly
   - Verify target allocations match your strategy
   - Note any warnings about price fetching

3. **Open the CSV in Excel/Google Sheets:**
   ```
   results/portfolio_tracking_[timestamp].csv
   ```

4. **Save your JSON snapshot:**
   ```
   data/portfolio_snapshot_[timestamp].json
   ```
   This is your baseline for future comparisons.

---

### Step 2: Quarterly Updates (Every 3 Months)

**Timing:** Run during earnings season (Jan, Apr, Jul, Oct)

1. **Load your previous snapshot:**
   ```python
   # In portfolio_tracker.py, modify main():
   # Load existing portfolio instead of creating new
   with open('data/portfolio_snapshot_20251201_163641.json', 'r') as f:
       portfolio = json.load(f)

   # Update with current prices
   portfolio = tracker.update_portfolio(portfolio)

   # Generate new report
   report = tracker.generate_report(portfolio)
   print(report)
   ```

2. **Review the report sections:**
   - **REBALANCING ALERTS:** Check for required actions
   - **CURRENT HOLDINGS:** See performance of each position
   - **EXIT TRIGGER CHECKS:** Watch for warning signs

3. **Take action based on alerts:**
   - `URGENT TRIM`: Sell immediately (position >50%)
   - `TRIM`: Sell within 1 week (position >40% or drift >5%)
   - `ADD`: Buy within 1 week (position under target)
   - `ADD OR EXIT`: Position <5% - decide to increase or exit
   - `HOLD`: No action needed

---

### Step 3: Execute Rebalancing

When the tracker shows rebalancing is needed:

**Week 1: Analysis & Decision**
- Run `python analyze_ai_ml_ipos_expanded.py` for latest scores
- Check if any companies dropped 2 tiers (sell signal)
- Verify exit triggers haven't been hit
- Calculate exact trade amounts

**Week 2: Execution**
1. **Place SELL orders first** (raise cash)
   - Market orders for large positions
   - Limit orders for smaller positions

2. **Wait for settlements** (T+2 for US stocks)

3. **Place BUY orders** (deploy cash)
   - Dollar-cost average over 2-4 weeks for large adds
   - Limit orders at 2-3% below current price

4. **Update your tracking:**
   - Manually update purchase prices in CSV
   - Record new cost basis
   - Reset days held counter
   - Add notes about the rebalancing

---

## REBALANCING RULES (AUTOMATIC ALERTS)

The tracker automatically flags these situations:

### Rule 1: The 40% Rule (MANDATORY)
**Alert:** `TRIM`
**Trigger:** Position exceeds 40% of portfolio
**Action:** Sell enough to reduce to 35%

**Example:**
```
PLTR grew from 25% to 42% of portfolio
Alert: TRIM
Action: Sell $7,000 worth of PLTR to reduce to 35%
```

---

### Rule 2: The 50% Rule (URGENT)
**Alert:** `URGENT TRIM`
**Trigger:** Position exceeds 50% of portfolio
**Action:** Sell immediately to reduce to 40%

**Example:**
```
PLTR exploded to 55% of portfolio
Alert: URGENT TRIM
Action: Sell $15,000 immediately to reduce to 40%
```

---

### Rule 3: The Tier Downgrade Rule
**Alert:** Will show in EXIT TRIGGER CHECKS
**Trigger:** Company drops 2 tiers (e.g., Very Strong → Moderate)
**Action:** Sell 50-100% of position

**Example:**
```
CRWV dropped from Very Strong (65) to Moderate (45)
Alert: EXIT TRIGGER CHECK - Monitor closely
Action: Sell 50-100% depending on reason for drop
```

---

### Rule 4: The Reversion Rule
**Alert:** `ADD OR EXIT`
**Trigger:** Position is <5% of portfolio
**Action:** Either add to position or exit completely

**Example:**
```
TEM declined to 3% of portfolio (target: 10%)
Alert: ADD OR EXIT
Action: Either buy $7,000 more TEM or sell remaining $3,000
```

---

### Rule 5: Threshold Rebalancing
**Alert:** `TRIM` or `ADD`
**Trigger:** Position drift exceeds 5% from target
**Action:** Rebalance to target allocation

**Example:**
```
ALAB: Current 26%, Target 20%, Drift +6%
Alert: TRIM
Action: Sell $6,000 to return to 20%
```

---

## CSV TEMPLATE USAGE

### Opening in Excel

1. Open Excel
2. File > Open > Select `portfolio_tracking_[timestamp].csv`
3. Data should load into columns automatically

### Opening in Google Sheets

1. Go to Google Sheets
2. File > Import > Upload tab
3. Select `portfolio_tracking_[timestamp].csv`
4. Import Location: "Replace spreadsheet"
5. Separator Type: "Comma"
6. Click "Import Data"

---

### Key Columns Explained

| Column | Description | How to Use |
|--------|-------------|------------|
| **Ticker** | Stock symbol | Reference only |
| **Shares** | Number of shares owned | Update when you buy/sell |
| **Purchase Price** | Average cost per share | Update after new purchases |
| **Current Price** | Latest market price | Manual update or use formula |
| **Cost Basis** | Total amount invested | = Shares * Purchase Price |
| **Current Value** | Current worth | = Shares * Current Price |
| **Gain/Loss $** | Unrealized profit/loss | = Current Value - Cost Basis |
| **Gain/Loss %** | Return percentage | = (Gain/Loss $) / Cost Basis |
| **Current Alloc %** | Current portfolio weight | = Current Value / Total Value |
| **Target Alloc %** | Desired weight | From strategy |
| **Drift %** | Difference from target | = Current - Target |
| **Score** | Advanced YC Score | From analysis (update quarterly) |
| **Risk** | Risk score (1-10) | From risk assessment |
| **Action** | Rebalancing recommendation | HOLD, TRIM, ADD, etc. |
| **Notes** | Your observations | Free text |

---

### Adding Price Update Formulas (Google Sheets)

**Method 1: Google Finance Function**
```
=GOOGLEFINANCE("NASDAQ:PLTR", "price")
```

Replace cell in "Current Price" column with this formula.

**Method 2: Manual Update**
- Visit Yahoo Finance or your broker
- Copy current prices
- Paste into "Current Price" column
- All calculations update automatically

---

## PERFORMANCE TRACKING

### Key Metrics to Watch

**1. Total Return**
- Tracks overall portfolio performance
- Compare to S&P 500 (^GSPC) as benchmark
- Target: Beat S&P 500 by 5-15% annually

**2. Best/Worst Performers**
- Identifies winners and laggards
- Consider trimming mega-winners (concentration risk)
- Consider exiting persistent laggards

**3. Average Score**
- Portfolio quality indicator
- Target: Keep above 60/100
- If drops below 55, consider rebalancing to higher-quality names

**4. Weighted Risk**
- Portfolio risk level (1-10 scale)
- Conservative: 6.0-6.5
- Balanced: 7.0-7.5
- Aggressive: 7.5-8.5

---

## AUTOMATION OPTIONS

### Option 1: Manual Quarterly (RECOMMENDED)
**Frequency:** Every 3 months
**Time Required:** 30-60 minutes

**Process:**
1. Run portfolio tracker script
2. Review report
3. Execute trades if needed
4. Update tracking spreadsheet

**Best For:** Most investors who want control

---

### Option 2: Semi-Automated
**Frequency:** Monthly automatic price updates, quarterly rebalancing

**Setup:**
1. **Windows Task Scheduler:**
   ```
   Task: Update Portfolio Prices
   Trigger: First Monday of each month, 6:00 PM (after market close)
   Action: python C:\path\to\portfolio_tracker.py
   ```

2. **Email Report:**
   Add email functionality to send report automatically
   (Requires SMTP setup - see automation guide)

**Best For:** Active investors who want regular updates

---

### Option 3: Fully Automated
**Frequency:** Weekly monitoring, automatic rebalancing

**Components:**
1. Weekly price updates
2. Automatic trade execution (via broker API)
3. Email alerts for urgent actions

**Setup:** Advanced - requires broker API integration

**Best For:** Experienced investors with API access

---

## COMMON SCENARIOS

### Scenario 1: Winner Takes Over
**Situation:** PLTR grows from 25% → 42% of portfolio

**What Happens:**
```
Tracker Alert: TRIM
Reason: Position >40% (Rule 1)
Action: Sell $7,000 to reduce to 35%
```

**Your Action:**
1. Review PLTR's latest score (still 81.8? Good sign)
2. Check if growth is justified (new contracts, revenue beat?)
3. Sell $7,000 worth over 1-2 weeks
4. Redeploy proceeds to underweight positions or keep as cash

**Tax Consideration:** Prioritize selling shares held >1 year (long-term gains)

---

### Scenario 2: Laggard Declines
**Situation:** TEM drops 40%, now only 6% of portfolio (target: 10%)

**What Happens:**
```
Tracker Alert: ADD
Reason: Drift >5% below target
Action: Buy $4,000 to reach 10%
```

**Your Action:**
1. Run analysis: Did TEM's score drop significantly?
2. Check news: Temporary issue or fundamental problem?
3. **If score still 50+:** Add $4,000 (buy the dip)
4. **If score dropped to <45:** Consider exiting instead

---

### Scenario 3: Tier Downgrade
**Situation:** CRWV drops from Very Strong (65.2) to Moderate (44.8)

**What Happens:**
```
Tracker Alert: EXIT TRIGGER CHECK
Reason: Dropped 2 tiers (Rule 3)
Action: Sell 50-100% of position
```

**Your Action:**
1. Investigate: Why did score drop? (Revenue miss? Margin compression?)
2. Check if temporary or structural issue
3. **If temporary:** Hold and monitor closely next quarter
4. **If structural:** Sell 100% within 1 week
5. Redeploy proceeds to higher-quality names

---

### Scenario 4: New IPO Opportunity
**Situation:** Databricks IPOs at estimated score of 72 (Very Strong)

**What Happens:**
- No automatic alert (new position)
- Manual decision required

**Your Action:**
1. Run Databricks through IPO checklist
2. If score 65+, consider adding to portfolio
3. **Conservative:** Don't add (keep 3 holdings)
4. **Balanced:** Replace lowest-scoring position (TEM 55.2)
5. **Aggressive:** Add as 9th position at 5-8% allocation

---

### Scenario 5: Market Crash
**Situation:** Entire portfolio down 30% in 2 months

**What Happens:**
- Allocations stay roughly the same (all stocks down)
- No rebalancing alerts (no significant drift)
- Exit triggers NOT hit (scores based on fundamentals)

**Your Action:**
1. **DO NOT PANIC SELL**
2. Review each company's fundamentals (not just stock price)
3. If scores still strong (60+), HOLD or ADD
4. If you have cash, this is opportunity to dollar-cost average
5. Rebalance only if individual positions drift >10%

---

## INTEGRATION WITH ANALYSIS SYSTEM

### Weekly: Score Monitoring
```bash
# Check for score changes
python analyze_ai_ml_ipos_expanded.py
python compare_weekly_results.py
```

**Look For:**
- Tier changes (upgrade or downgrade)
- Significant score movements (>5 points)
- New companies entering Strong tier

---

### Quarterly: Full Rebalancing Review
```bash
# 1. Update scores
python analyze_ai_ml_ipos_expanded.py

# 2. Check week-over-week changes
python compare_weekly_results.py

# 3. Update portfolio tracking
python portfolio_tracker.py

# 4. Review portfolio recommendations
python portfolio_recommendations.py
```

---

### Ad-Hoc: New IPO Evaluation
```bash
# When new AI/ML IPO announced:
python analyze_new_ai_ipo.py

# Compare against current holdings
# Decide if worth adding to portfolio
```

---

## BEST PRACTICES

### DO's ✅

1. **Rebalance Quarterly** - Align with earnings season (Jan, Apr, Jul, Oct)
2. **Update Scores Quarterly** - Run analysis scripts before rebalancing
3. **Keep Cash Reserve** - 3-5% cash for opportunistic adds
4. **Dollar-Cost Average** - Spread large purchases over 2-4 weeks
5. **Document Trades** - Add notes in CSV about why you bought/sold
6. **Tax-Loss Harvest** - Sell losers in December for tax benefits
7. **Check Exit Triggers** - Review risk assessment quarterly
8. **Stay Disciplined** - Follow the rules, avoid emotional decisions

---

### DON'Ts ❌

1. **Don't Panic Sell** - Market crashes are opportunities if fundamentals strong
2. **Don't Chase Momentum** - Stick to target allocations, don't over-concentrate
3. **Don't Ignore Warnings** - If tracker says URGENT TRIM, act quickly
4. **Don't Overtrade** - Quarterly rebalancing is sufficient (not monthly)
5. **Don't Add Low Scores** - Avoid companies with scores <50 unless proven track record
6. **Don't Skip Updates** - Stale data leads to poor decisions
7. **Don't Forget Taxes** - Consider tax implications of all trades
8. **Don't Deviate from Strategy** - Changing strategies frequently reduces returns

---

## TROUBLESHOOTING

### Problem: "Could not fetch price for [TICKER]"

**Causes:**
- Yahoo Finance API temporarily down
- Ticker symbol incorrect
- Stock delisted or ticker changed

**Solutions:**
1. Wait 5 minutes and retry
2. Verify ticker on Yahoo Finance website
3. Check if company was acquired or delisted
4. Manually enter price in CSV if API fails

---

### Problem: Scores Not Updating

**Causes:**
- `data/score_history.json` not found
- Analysis scripts not run recently

**Solutions:**
1. Run: `python analyze_ai_ml_ipos_expanded.py`
2. Verify `data/score_history.json` exists and has recent dates
3. Check script output for any errors

---

### Problem: Allocations Don't Match Strategy

**Causes:**
- Manual trades executed outside tracker
- Prices changed significantly since last update
- Cash balance not reflected accurately

**Solutions:**
1. Update actual share counts in portfolio JSON
2. Recalculate cost basis based on actual purchases
3. Run `update_portfolio()` to refresh all values

---

### Problem: CSV Won't Open in Excel

**Causes:**
- File path has special characters
- Excel regional settings use semicolon separator

**Solutions:**
1. Move file to simple path (C:\Temp\portfolio.csv)
2. Open Excel > Data tab > From Text/CSV
3. Manually specify comma as delimiter

---

## ADVANCED: CUSTOM PORTFOLIO

Want to create your own allocation?

**Edit portfolio_tracker.py:**

```python
def _get_strategy_allocations(self, strategy: str) -> Dict[str, float]:
    """Get target allocations by strategy."""
    if strategy.lower() == 'custom':
        return {
            'PLTR': 0.30,   # 30%
            'ALAB': 0.25,   # 25%
            'CRWV': 0.20,   # 20%
            'DDOG': 0.15,   # 15%
            'IONQ': 0.10    # 10%
        }
    # ... existing strategies ...
```

**Then run:**
```python
portfolio = tracker.create_tracking_template(
    strategy='custom',
    initial_budget=100000
)
```

---

## FILES REFERENCE

**Generated by Tracker:**
- `data/portfolio_snapshot_[timestamp].json` - Portfolio state snapshot
- `results/portfolio_tracking_[timestamp].csv` - Excel/Sheets template
- Console output: Markdown report

**Related Documentation:**
- `INVESTMENT_RECOMMENDATIONS_20251201.md` - Strategy details
- `RISK_ASSESSMENT_TOP_PICKS.md` - Risk analysis for top holdings
- `QUARTERLY_REBALANCING_GUIDE.md` - Detailed rebalancing framework
- `AI_ML_IPO_SCORECARD.md` - Quick reference for all companies

---

## SUPPORT & UPDATES

**System Updates:**
- Scores update quarterly (run analysis scripts)
- New IPOs added as discovered (check `UPCOMING_AI_IPO_PIPELINE.md`)
- Rebalancing rules static (proven framework)

**Questions:**
- Review `CURRENT_STATE.md` for latest system status
- Check `DATABRICKS_IPO_CHECKLIST.md` for upcoming opportunities
- Read `TOP_PERFORMERS_DEEP_DIVE.md` for investment philosophy

---

## SUCCESS METRICS

**Track these over time:**

| Metric | Target | Excellent | World-Class |
|--------|--------|-----------|-------------|
| **Annual Return** | >12% | >20% | >30% |
| **vs S&P 500** | +5% | +10% | +15% |
| **Average Score** | >60 | >65 | >70 |
| **Weighted Risk** | <8.0 | <7.5 | <7.0 |
| **Rebalancing Frequency** | Quarterly | Quarterly | Quarterly |
| **Max Drawdown** | <40% | <30% | <25% |
| **Win Rate** | >60% | >70% | >80% |

---

## CONCLUSION

**The portfolio tracking system provides:**
- ✅ Systematic rebalancing framework
- ✅ Automatic risk monitoring
- ✅ Clear action alerts (TRIM, ADD, HOLD)
- ✅ Integration with scoring system
- ✅ Professional-grade tracking

**Remember:**
1. **Discipline beats timing** - Follow the rules consistently
2. **Quality over quantity** - Focus on high-scoring companies
3. **Rebalance systematically** - Quarterly cadence is optimal
4. **Monitor but don't obsess** - Check quarterly, not daily
5. **Long-term horizon wins** - Hold for 5-10 years minimum

**Your path to success:**
1. Choose your strategy (Conservative/Balanced/Aggressive)
2. Set up tracking template (one-time setup)
3. Review quarterly (15-30 minutes)
4. Rebalance when alerts trigger
5. Compound wealth over time

---

**Built on:** Proteus AI/ML IPO Analysis System
**Version:** 1.0
**Last Updated:** 2025-12-01

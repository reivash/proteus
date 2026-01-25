# Proteus Research Strategy & Roadmap

**Last Updated:** 2025-11-17
**Current Version:** v15.0-EXPANDED
**Status:** Optimization ceiling reached on mean reversion - EXPANDING TO NEW STRATEGIES

---

## Table of Contents
1. [Current State Verification](#current-state-verification)
2. [Research Pipeline Strategy](#research-pipeline-strategy)
3. [Unexplored Opportunities](#unexplored-opportunities)
4. [Experiment Queue](#experiment-queue)
5. [Success Metrics](#success-metrics)
6. [Verification Checklist](#verification-checklist)

---

## Current State Verification

### Mean Reversion System (v15.0-EXPANDED)
- **Win Rate:** 79.3%
- **Portfolio Size:** 45 stocks (all 70%+ win rate)
- **Avg Return per Stock:** ~19.2%
- **Trade Frequency:** ~251 trades/year
- **Status:** ✅ FULLY OPTIMIZED - Theoretical maximum reached

### Recent Optimization Attempts (ALL FAILED/NEGATIVE)
| Exp ID | Optimization Attempt | Result | Reason |
|--------|---------------------|--------|---------|
| EXP-052 | VIX volatility filtering | +0.9pp | Below deployment threshold |
| EXP-053 | Portfolio position limits | +0.3pp | Below deployment threshold |
| EXP-054 | Per-stock stop-loss optimization | +0.0pp | No benefit |
| EXP-055 | Per-stock holding periods | -2.0% | Negative impact |
| EXP-057 | Multi-timeframe confirmation | +0.0pp | No benefit |
| EXP-058 | Seasonality analysis | N/A | Data unavailable |
| EXP-059 | Market breadth filtering | -10.8pp | Significant negative |
| EXP-060 | Gap-enhanced entry | +7.6pp | False positive (2 stocks only) |
| EXP-061 | Gap validation (45 stocks) | -5.7pp | Failed validation |

**CONCLUSION:** Mean reversion parameter optimization has reached theoretical maximum. Further improvements require NEW STRATEGY DIRECTIONS, not parameter tuning.

---

## Research Pipeline Strategy

### Phase 1: Current State (COMPLETE ✅)
- [x] Mean reversion strategy optimized
- [x] 45-stock portfolio validated
- [x] Dynamic position sizing deployed
- [x] Daily scanner with email alerts operational
- [x] All parameter-based optimizations exhausted

### Phase 2: Strategy Diversification (CURRENT PHASE)

Based on comprehensive research from **ChatGPT Strategy Research Report** (Desktop PDF), three major unexplored directions:

#### A. Sentiment-Enhanced Mean Reversion (HIGHEST PRIORITY)
**Rationale:** Enhances existing mean reversion system rather than replacing it

**Hypothesis:** Social media/news sentiment can differentiate:
- **Panic-driven sell-offs** (buy opportunity) vs.
- **News-driven sell-offs** (avoid - legitimate bad news)

**Implementation:**
1. **Data Sources:**
   - Twitter API / snscrape for real-time sentiment
   - News API for financial news headlines
   - Reddit WSB/stocks/investing for retail sentiment

2. **Sentiment Analysis:**
   - **FinBERT** (finance-specific BERT model) for accurate classification
   - Sentiment score: -1.0 (bearish) to +1.0 (bullish)
   - Panic indicators: High volume + extreme negative sentiment

3. **Integration with Current System:**
   ```python
   # ENHANCED SIGNAL LOGIC
   if panic_sell_signal and sentiment_score < -0.6:
       # Strong negative sentiment confirms panic = BUY
       signal_quality = "HIGH"
   elif panic_sell_signal and sentiment_score > -0.3:
       # Weak negative sentiment = might be justified = SKIP
       signal_quality = "LOW"
   ```

4. **Expected Improvement:** +5-10pp win rate by filtering false positives

5. **Success Criteria:**
   - Win rate improvement >= +3pp
   - Trade reduction <= 40%
   - Maintains 70%+ overall win rate

**Next Steps:**
- EXP-062: Test sentiment data collection (Twitter/Reddit)
- EXP-063: Backtest FinBERT sentiment scoring
- EXP-064: Validate sentiment-enhanced signals on 10-stock subset
- EXP-065: Full portfolio deployment if validated

---

#### B. Machine Learning Stock Selection (MEDIUM PRIORITY)
**Rationale:** Cross-sectional prediction of which stocks will outperform

**Hypothesis:** ML models can identify stocks likely to experience mean reversion opportunities BEFORE they occur

**Implementation:**
1. **Feature Engineering (50-100 features):**
   - Technical: RSI, MACD, Bollinger Bands, volume ratios, price momentum
   - Fundamental: P/E ratio, debt-to-equity, revenue growth, earnings surprises
   - Alternative: Sentiment scores, option flow, insider trading
   - Market: Sector performance, correlation to SPY, beta

2. **Model Architecture:**
   - **Baseline:** XGBoost (fastest, proven effective)
   - **Advanced:** LightGBM, Random Forest ensemble
   - **Deep Learning:** Feed-forward neural network (if needed)

3. **Target Variable:**
   - Binary: Will stock have profitable mean reversion signal in next 7 days?
   - OR Regression: Expected return from mean reversion opportunity

4. **Validation Strategy:**
   - Walk-forward analysis (train on past, test on future)
   - Purged K-Fold cross-validation (avoid look-ahead bias)
   - Out-of-sample testing on 20% holdout set

5. **Expected Improvement:**
   - Better stock selection (focus scanning on high-probability stocks)
   - 10-20% increase in signal frequency
   - Potential for portfolio expansion beyond 45 stocks

**Next Steps:**
- EXP-066: Build feature engineering pipeline
- EXP-067: Train XGBoost baseline model
- EXP-068: Validate on held-out test set
- EXP-069: Deploy top 10 ML-selected stocks vs. random 10 comparison

---

#### C. Deep Reinforcement Learning Agent (EXPERIMENTAL)
**Rationale:** Learn optimal trading policy from experience (not rules)

**Hypothesis:** DRL agent can discover non-obvious patterns and optimal entry/exit timing beyond human-designed rules

**Implementation:**
1. **Environment Design:**
   - State: Price history, technical indicators, position status, P&L
   - Actions: BUY, SELL, HOLD (or continuous position size)
   - Reward: Risk-adjusted returns (Sharpe ratio maximization)

2. **Algorithm Selection:**
   - **PPO (Proximal Policy Optimization):** Most stable for trading
   - **DDPG (Deep Deterministic Policy Gradient):** For continuous actions
   - Alternative: A2C, SAC

3. **Framework:**
   - **FinRL library** (specialized for financial RL)
   - **stable-baselines3** (production-ready implementations)
   - **Gymnasium** (environment interface)

4. **Training Strategy:**
   - Train on historical data (3+ years)
   - Validate on walk-forward windows
   - Test on completely unseen period

5. **Expected Outcome:**
   - If successful: Autonomous trading agent
   - If unsuccessful: Insights for rule-based improvements

**Next Steps:**
- EXP-070: Setup FinRL environment
- EXP-071: Train PPO agent on single stock (NVDA)
- EXP-072: Compare agent vs. mean reversion on same stock
- EXP-073: Scale to multi-stock portfolio if validated

---

## Unexplored Opportunities

### From ChatGPT Research PDF

#### 1. Sentiment-Driven Trading ⭐ **TOP PRIORITY**
**Evidence:** "A 2025 analysis of ~3 million stock-related tweets found that tweet-based sentiment strongly predicts intraday market trends"

**Why This Works:**
- Social media reacts faster than news outlets
- Panic sells often driven by social media amplification
- FinBERT achieves 85%+ accuracy on financial sentiment

**Risk Level:** LOW (enhances existing system)

---

#### 2. Machine Learning Stock Selection ⭐ **HIGH VALUE**
**Evidence:** "Gu, Kelly, and Xiu (2020) showed ML models with 94 predictive signals achieved R² of 0.33-0.40% for monthly returns"

**Why This Works:**
- Nonlinear patterns that rules can't capture
- Feature importance reveals what actually drives returns
- Cross-sectional prediction (relative performance)

**Risk Level:** MEDIUM (requires significant development)

---

#### 3. Time-Series Forecasting with Deep Learning
**Evidence:** "Transformer architectures adapted for financial time-series show promise in recent studies"

**Why This Works:**
- LSTM/GRU can capture long-term dependencies
- Transformers handle multiple time horizons
- Attention mechanisms reveal important price patterns

**Risk Level:** HIGH (complex, prone to overfitting)

**Status:** Lower priority - only if sentiment + ML fail

---

#### 4. Deep Reinforcement Learning
**Evidence:** "DRL agents have shown ability to adapt to changing market conditions and discover non-obvious strategies"

**Why This Works:**
- No need to hand-craft rules
- Learns from experience
- Can optimize complex objectives (Sharpe, max drawdown)

**Risk Level:** VERY HIGH (experimental, unstable training)

**Status:** Experimental track only

---

## Experiment Queue

### Immediate Queue (Next 4-8 Weeks)

#### Week 1-2: Sentiment Data Collection & Integration
- **EXP-062:** Twitter/Reddit sentiment data pipeline
  - Test data collection APIs
  - Build sentiment database
  - Calculate sentiment scores for historical panic sells
  - **Success:** 80%+ data coverage for past 3 years

- **EXP-063:** FinBERT sentiment model integration
  - Load pre-trained FinBERT model
  - Test sentiment scoring accuracy
  - Compare vs. VADER baseline
  - **Success:** Sentiment correlation with price movements > 0.3

#### Week 3-4: Sentiment-Enhanced Backtesting
- **EXP-064:** Sentiment filter validation (10-stock subset)
  - Test: Baseline vs. Sentiment-filtered signals
  - Stocks: NVDA, MSFT, AAPL, JPM, V, MA, ORCL, AVGO, ABBV, TXN
  - **Success:** +3pp win rate improvement, trade reduction < 40%

- **EXP-065:** Full portfolio sentiment deployment
  - Deploy to all 45 stocks
  - Comprehensive validation report
  - **Success:** System-wide +3pp minimum, 75%+ win rate

#### Week 5-6: ML Feature Engineering & Baseline
- **EXP-066:** Feature engineering pipeline
  - Technical indicators (50+)
  - Fundamental data integration
  - Sentiment features
  - **Success:** Complete feature matrix for all stocks

- **EXP-067:** XGBoost baseline model
  - Train on 3-year history
  - Walk-forward validation
  - Feature importance analysis
  - **Success:** AUC > 0.60 for predicting profitable signals

#### Week 7-8: ML Model Validation & Deployment
- **EXP-068:** Out-of-sample testing
  - 20% holdout validation
  - Compare ML selection vs. random selection
  - **Success:** ML-selected stocks show 15%+ higher signal frequency

- **EXP-069:** Production ML deployment
  - Integrate with daily scanner
  - A/B test: ML-ranked vs. equal-weight scanning
  - **Success:** Improved scanning efficiency (fewer false positives)

---

### Future Queue (2-6 Months)

#### Advanced Sentiment Analysis
- **EXP-070:** Multi-source sentiment aggregation
  - Combine Twitter, Reddit, News, StockTwits
  - Weight sources by historical accuracy
  - Real-time sentiment dashboard

#### Deep Learning Experiments
- **EXP-071:** LSTM price forecasting
  - 1-day, 3-day, 7-day return prediction
  - Compare vs. mean reversion expected returns

- **EXP-072:** Transformer architecture
  - Multi-stock attention mechanism
  - Sector correlation modeling

#### Reinforcement Learning Track
- **EXP-073:** FinRL environment setup
- **EXP-074:** PPO agent training (single stock)
- **EXP-075:** Multi-stock DRL portfolio

#### Portfolio Optimization
- **EXP-076:** Correlation-based position sizing
  - Reduce positions in correlated stocks
  - Maximize portfolio diversification

- **EXP-077:** Kelly Criterion position sizing
  - Optimal bet sizing based on win rate
  - Compare vs. current LINEAR sizing

---

## Success Metrics

### System-Wide Performance Targets

#### Tier 1: Baseline Maintenance (REQUIRED)
- Win rate >= 75% (currently 79.3%)
- Avg return per stock >= 15% (currently 19.2%)
- Sharpe ratio >= 1.5
- Max drawdown <= 25%

#### Tier 2: Improvement Targets (GOAL)
- Win rate >= 82% (+3pp from current)
- Avg return per stock >= 22% (+3% from current)
- Sharpe ratio >= 2.0
- Trade frequency +20% (more opportunities)

#### Tier 3: Stretch Goals (ASPIRATIONAL)
- Win rate >= 85%
- Avg return per stock >= 25%
- Sharpe ratio >= 2.5
- 100+ stock universe

### Individual Experiment Success Criteria

**Minimum Deployment Threshold:**
- Win rate improvement >= +3pp OR
- Return improvement >= +3% AND
- Trade reduction <= 40% (unless intentional quality filter)

**Email Report Requirements (FIXED ✅):**
- ✅ Hypothesis clearly stated
- ✅ Stocks tested (full list prominently displayed)
- ✅ Test methodology (baseline vs experimental)
- ✅ Results with performance metrics
- ✅ Deployment recommendation

---

## Verification Checklist

### Before Starting New Experiment

- [ ] Review current system performance (check PRODUCTION_DEPLOYMENT.md)
- [ ] Review this RESEARCH_STRATEGY.md for priority direction
- [ ] Check if similar experiment already tested (avoid duplicates)
- [ ] Verify hypothesis is testable and falsifiable
- [ ] Define success criteria BEFORE running experiment
- [ ] Ensure email report will include all required info

### During Experiment

- [ ] Document all code changes with comments
- [ ] Save experiment code as `common/experiments/expXXX_description.py`
- [ ] Log all results to `logs/experiments/expXXX_description.json`
- [ ] Send comprehensive email report with hypothesis and stocks tested
- [ ] Git commit with clear message: "EXP-XXX: Description - Result"

### After Experiment

- [ ] Update PRODUCTION_DEPLOYMENT.md if experiment deployed
- [ ] Update this RESEARCH_STRATEGY.md with findings
- [ ] Document why experiment failed/succeeded
- [ ] Identify follow-up experiments or dead ends
- [ ] Archive experiment code for future reference

### Monthly Review

- [ ] Review last 30 days of experiments
- [ ] Calculate cumulative improvement from deployments
- [ ] Verify production system still meeting baseline targets
- [ ] Adjust research priorities based on findings
- [ ] Update roadmap with new insights

---

## Strategy Evolution Log

### 2025-11-17: Optimization Ceiling Recognition
**Finding:** 8 consecutive mean reversion optimizations showed no benefit

**Decision:** PIVOT from parameter optimization to strategy diversification

**Action Items:**
1. ✅ Fixed email reports (hypothesis + symbols + methodology)
2. ⏳ Begin sentiment integration (EXP-062)
3. ⏳ Start ML feature engineering (EXP-066)
4. ⏳ Research FinRL for DRL experiments

**Expected Timeline:**
- Sentiment deployment: 4 weeks
- ML deployment: 8 weeks
- DRL evaluation: 12 weeks

---

## Research Resources

### Key Files
- **This file:** `RESEARCH_STRATEGY.md` - Master research plan
- **Deployment:** `PRODUCTION_DEPLOYMENT.md` - Current system specs
- **Parameters:** `common/config/mean_reversion_params.py` - All stock configs
- **Scanner:** `common/experiments/exp056_production_scanner.py` - Daily signal detection
- **Email Reports:** `common/notifications/sendgrid_notifier.py` - Experiment reporting (FIXED)

### External Resources
- **ChatGPT Research PDF:** `C:\Users\javie\Desktop\Predicting and Profiting from the Stock Market_ A Comprehensive Strategy Research Report.pdf`
- **FinBERT Model:** https://huggingface.co/ProsusAI/finbert
- **FinRL Library:** https://github.com/AI4Finance-Foundation/FinRL
- **Research Papers:** See PDF bibliography

### Data APIs
- **Stock Data:** Yahoo Finance (yfinance)
- **Sentiment:** Twitter API, snscrape, Reddit PRAW
- **News:** News API, Alpha Vantage
- **Fundamentals:** Financial Modeling Prep, Alpha Vantage

---

## Critical Insights

### Why Mean Reversion Optimization Stopped Working
1. **Statistical Ceiling:** Win rate of 79.3% may be near theoretical maximum for panic sell detection
2. **Parameter Saturation:** All major parameters already grid-searched and optimized
3. **Market Efficiency:** Simple technical indicators becoming less effective as more traders use them
4. **Need New Alpha:** Additional alpha requires NEW INFORMATION SOURCES (sentiment, ML predictions)

### Why Sentiment Is The Best Next Step
1. **Complements Existing System:** Filters current signals, doesn't replace them
2. **New Information:** Social media not in current feature set
3. **Proven Effectiveness:** Multiple academic studies show predictive power
4. **Low Risk:** Can easily revert if doesn't work
5. **Fast Implementation:** APIs and pre-trained models available

### Why Machine Learning Is High Value
1. **Pattern Discovery:** Can find nonlinear relationships humans miss
2. **Feature Selection:** Shows what actually matters
3. **Scalability:** Once built, easily extends to 100+ stocks
4. **Cross-Sectional Edge:** Predicts relative performance, not absolute
5. **Continuous Learning:** Can retrain as markets evolve

### Why DRL Is Experimental
1. **Training Instability:** Very sensitive to hyperparameters
2. **Overfitting Risk:** Can memorize training data
3. **Black Box:** Hard to understand what it learned
4. **Computational Cost:** Requires GPU, long training times
5. **Unproven at Scale:** Few production deployments in finance

---

## Final Notes

### Mission Statement
**Make Proteus progressively better through rigorous experimentation, comprehensive validation, and strategic deployment of proven enhancements.**

### Core Principles
1. **Measure Everything:** Every experiment must have quantifiable success criteria
2. **Validate Thoroughly:** No deployment without multi-stock, multi-year validation
3. **Document Comprehensively:** Email reports, git commits, markdown logs
4. **Fail Fast:** If experiment shows no improvement after 10 stocks, abort
5. **Compound Gains:** Stack multiple small improvements into large advantage

### Success Definition
Proteus succeeds when it consistently generates profitable trading signals with:
- High win rate (75%+)
- Strong returns (15%+ per position)
- Controlled risk (Sharpe > 1.5)
- Scalable infrastructure (100+ stocks, fully automated)

---

**Remember:** The goal is not to optimize forever, but to DEPLOY working strategies and COMPOUND improvements over time.

**Next Action:** Run EXP-062 to begin sentiment data collection and integration.

---
*This document is a living strategy that should be updated after every significant finding or deployment.*

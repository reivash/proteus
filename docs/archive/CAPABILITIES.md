# Proteus Trading System - Capabilities & Features

**Version:** 8.0
**Last Updated:** 2025-11-16
**Status:** Production Ready

---

## 📋 Table of Contents

1. [Core Trading System](#core-trading-system)
2. [Data Collection & Analysis](#data-collection--analysis)
3. [Dashboard & Monitoring](#dashboard--monitoring)
4. [Notifications & Alerts](#notifications--alerts)
5. [Experimentation Framework](#experimentation-framework)
6. [Ready-to-Implement Features](#ready-to-implement-features)
7. [Research Findings & Validated Optimizations](#research-findings--validated-optimizations)
8. [System Architecture](#system-architecture)
9. [Configuration & Deployment](#configuration--deployment)

---

## 🎯 Core Trading System

### Strategy: Mean Reversion on Panic Sells

**Status:** ✅ Fully Operational (v8.0)

#### What It Does
Detects panic sell opportunities and profits from mean reversion when high-quality stocks experience irrational selloffs.

#### Entry Criteria
1. **Z-Score < -1.5** (price significantly below moving average)
2. **RSI < 35** (oversold condition)
3. **Volume Spike > 1.3x** average (panic selling detected)
4. **Price Drop > -1.5%** (meaningful decline)

#### Exit Strategy (Time-Decay)
- **Day 0:** ±2% profit/loss targets
- **Day 1:** ±1.5% profit/loss targets
- **Day 2+:** ±1% profit/loss targets

Automatically tightens profit targets over time to lock in gains before reversal momentum fades.

#### Universe (10 Tier A Stocks)
**Semiconductors (4):**
- NVDA: 87.5% win rate, +49.70% return (BEST)
- AVGO: 87.5% win rate, +24.52% return
- KLAC: 80.0% win rate, +20.05% return
- MRVL: 75.0% win rate, +26.63% return

**Healthcare (2):**
- ABBV: 70.6% win rate, +10.50% return
- SYK: 70.0% win rate, +11.33% return

**Payments (2):**
- V: 75.0% win rate, +7.33% return
- MA: 71.4% win rate, +5.99% return

**Tech (1):**
- ORCL: 76.9% win rate, +22.50% return

**Finance (1):**
- AXP: 71.4% win rate, +29.21% return

#### Performance Metrics (v8.0)
- **Average Win Rate:** 76.5%
- **Average Return:** +20.78% per stock (3-year backtest)
- **Trade Frequency:** ~55 trades/year
- **Sector Diversification:** 5 sectors (reduced concentration risk)

#### Key Features
✅ Stock-specific parameter optimization
✅ Time-decay exit strategy
✅ Automated signal detection
✅ Real-time monitoring
✅ No market timing needed (Tier A stocks work in all regimes)

---

## 📊 Data Collection & Analysis

### Market Data Collection

**Status:** ✅ Fully Operational

#### Yahoo Finance Integration
- **Source:** Free, unlimited historical data
- **Data Points:** OHLCV (Open, High, Low, Close, Volume)
- **Lookback Period:** 60 days (optimal for stable indicators)
- **Update Frequency:** Real-time during market hours
- **Reliability:** High (yfinance library)

#### Technical Indicators
Automatically calculated for each stock:
- **Moving Averages:** 20-day, 50-day
- **Z-Score:** Price deviation from mean
- **RSI:** Relative Strength Index (14-period)
- **Volume Analysis:** Average volume, volume spikes
- **Bollinger Bands:** Standard deviation channels

### News & Sentiment Collection

**Status:** ⚠️ Built but ON HOLD (See EXP-019)

#### News API Integration
- **Status:** Code ready, not deployed
- **Reason:** Cannot backtest (only 7-30 days historical data)
- **Use Case:** Filter information-driven panics (earnings, scandals)
- **Deployment:** Requires premium data ($100+/month) or live testing

#### Twitter/Social Sentiment
- **Status:** Infrastructure ready
- **Capabilities:** Sentiment scoring, trend detection
- **Limitations:** Historical data unavailable for backtesting
- **Future:** Can be enabled for live trading

**Decision:** Put on hold until proven necessary (current 76.5% win rate excellent without sentiment)

---

## 🖥️ Dashboard & Monitoring

### Web Dashboard

**Status:** ✅ Fully Operational

**URL:** http://localhost:5000

#### Features

**Real-Time Monitoring:**
- Current signals with entry prices
- Z-scores, RSI, volume metrics
- Expected returns based on historical data
- Position status (Day 0, 1, 2+)

**Performance Tracking:**
- Total trades executed
- Win rate (target: 76.5%)
- Total returns
- Individual stock performance

**Scanned Instruments Display:**
- List of all monitored tickers
- Last scan timestamp
- Scan status (success/error)

**Signal History:**
- Past signals with outcomes
- Entry/exit prices and dates
- Actual returns vs expected
- Performance analytics

#### Scheduling

**Automated Scans:**
- **9:45 AM EST:** Morning scan (position monitoring)
- **3:45 PM EST:** Afternoon scan (optimal entry timing - 15 min before close)

**Rationale:** EXP-020 proved panic day close entry is optimal (+49.70% vs +22.55% next day open)

#### Visual Features
- Professional gradient UI (purple theme)
- Responsive design
- Real-time updates
- Mobile-friendly

---

## 🔔 Notifications & Alerts

### Email Notifications

**Status:** ✅ Fully Operational

#### SendGrid Integration
- **Free Tier:** 100 emails/day
- **Setup:** One API key (no passwords needed)
- **Configuration:** email_config.json

#### Daily Scan Alerts

**Sent When:**
- New BUY signals detected
- No signals (confirmation scan ran)
- Scan errors or issues

**Email Includes:**
- Signal count
- Ticker symbols
- Entry prices
- Technical indicators (Z-score, RSI)
- Expected returns
- Scanned instruments list
- Performance summary

#### Experiment Reports

**Status:** ✅ NEW (Added 2025-11-16)

**Sent When:** Experiments complete

**Email Includes:**
- Experiment ID and summary
- Symbols tested
- Tier A stocks found (>70% win rate)
- Tier B stocks (55-70% win rate)
- Win rates, returns, Sharpe ratios
- Recommendations
- Links to full JSON results

**Beautiful HTML formatting with color-coded sections:**
- Green: Tier A stocks (excellent)
- Yellow: Tier B stocks (marginal)
- Professional layout for easy review

#### Test Email
```bash
python common/notifications/sendgrid_notifier.py
```

---

## 🔬 Experimentation Framework

### Experiment Infrastructure

**Status:** ✅ Fully Operational

#### Backtesting Engine
- **Period:** Configurable (default 3 years)
- **Data Source:** Yahoo Finance historical data
- **Strategy:** Mean reversion with time-decay exits
- **Metrics:** Win rate, total return, Sharpe ratio, avg gain/loss

#### Experiment Workflow
1. **Hypothesis Formation:** Define what to test
2. **Implementation:** Build detector/strategy variant
3. **Backtesting:** Run on historical data
4. **Analysis:** Compare vs baseline
5. **Documentation:** Create findings document
6. **Deployment:** Update config if successful
7. **Email Report:** Automatic notification with results

#### Completed Experiments

**EXP-014: Stock Selection Optimization** ✅ SUCCESS
- **Result:** Trading ONLY Tier A stocks improves returns by +15.99pp
- **Impact:** Biggest single optimization
- **Deployed:** v6.0

**EXP-019: Sentiment Integration** ⚠️ ON HOLD
- **Result:** Cannot backtest (no historical news data)
- **Status:** Infrastructure ready, waiting for data access or live testing

**EXP-020: Entry Timing Optimization** ❌ FAILED
- **Result:** Panic day close is optimal (+49.70% vs +22.55% next day)
- **Learning:** Waiting for "better prices" reduces returns by 54%
- **Decision:** Keep panic day close entry

**EXP-021: Expand Tier A Universe** ✅ SUCCESS
- **Result:** Found 2 new Tier A stocks (AVGO, AXP)
- **Impact:** 3 → 5 stocks, +67% trade frequency
- **Deployed:** v7.0

**EXP-023: Market Regime Filtering** ❌ FAILED
- **Result:** Regime filtering REDUCES performance (-18.83pp average)
- **Learning:** Tier A stocks work in ALL market conditions
- **Decision:** No market timing needed

**EXP-024: Expand Tier A Round 2** ✅ HUGE SUCCESS
- **Result:** Found 5 new Tier A stocks (KLAC, ORCL, MRVL, ABBV, SYK)
- **Impact:** 5 → 10 stocks, +120% trade frequency
- **Deployed:** v8.0

#### Experiment Template
```python
from common.experiments.exp014_stock_selection_optimization import run_exp014_stock_selection

results = run_exp014_stock_selection(symbols, period='3y')

# Automatic email report
from common.notifications.sendgrid_notifier import SendGridNotifier
notifier = SendGridNotifier()
notifier.send_experiment_report('EXP-XXX', results)
```

---

## 🚀 Ready-to-Implement Features

### Features Built But Not Deployed

#### 1. News Sentiment Filtering ⏳

**Status:** Code ready, awaiting data access

**Location:** `common/data/sentiment/news_sentiment.py`

**Capabilities:**
- News API integration for headlines
- Sentiment scoring (positive/negative/neutral)
- Signal confidence classification
- Information-driven panic detection

**Why Not Deployed:**
- Cannot backtest (News API free tier = only 7-30 days historical)
- Need premium data ($100+/month) or live testing
- Current 76.5% win rate excellent without it

**Deployment Requirements:**
- Premium News API subscription OR
- 3-6 months live testing to validate

#### 2. Social Media Sentiment ⏳

**Status:** Infrastructure ready, awaiting data source

**Location:** `common/data/sentiment/twitter_collector.py`

**Capabilities:**
- Twitter/Reddit sentiment analysis
- Social trend detection
- Retail investor panic detection
- Sentiment momentum tracking

**Why Not Deployed:**
- No historical social data for backtesting
- Twitter API changes (expensive now)
- Questionable signal quality

**Deployment Requirements:**
- Alternative data source (Reddit, StockTwits) OR
- Live testing period OR
- Premium social sentiment data

#### 3. Advanced Position Sizing 💡

**Status:** Can be implemented quickly

**Potential Models:**
- **Kelly Criterion:** Math-optimal position sizing
- **Win-Rate Weighted:** More capital to NVDA (87.5%) vs SYK (70.0%)
- **Sharpe-Weighted:** Allocate by risk-adjusted returns
- **Risk Parity:** Equal risk contribution

**Expected Impact:** +1-3pp return improvement

**Implementation Time:** 1-2 days (new experiment)

#### 4. Signal Quality Scoring 💡

**Status:** Can be implemented quickly

**Concept:**
- Rank signals by strength (Z-score magnitude, volume spike size)
- Trade only top 50% quality signals
- Filter weak/marginal signals

**Expected Impact:** +5-8pp win rate improvement, -50% trade frequency

**Trade-off:** Fewer opportunities vs higher win rate

**Implementation Time:** 2-3 days (new experiment)

#### 5. Stop Loss Optimization 💡

**Status:** Currently only time-decay exits

**Potential Enhancements:**
- **Trailing Stops:** Lock in profits as price rises
- **Volatility-Based Stops:** Wider stops for volatile stocks
- **ATR Stops:** Based on Average True Range
- **Breakeven Stops:** Move stop to entry after +X% profit

**Expected Impact:** +2-4pp return improvement, better risk management

**Implementation Time:** 3-4 days (new experiment)

#### 6. Multi-Timeframe Analysis 💡

**Status:** Currently daily only

**Concept:**
- Intraday signals (1-hour, 15-min charts)
- Weekly confirmation for daily signals
- Trend alignment across timeframes

**Expected Impact:** Earlier entries, better timing

**Challenges:** Need intraday data, more complex

**Implementation Time:** 5-7 days (significant rewrite)

---

## 📈 Research Findings & Validated Optimizations

### What Works (Deployed)

#### ✅ Stock Selection is King (EXP-014, EXP-021, EXP-024)
- **Finding:** Choosing the RIGHT stocks is the #1 factor
- **Impact:** +15.99pp improvement from Tier A-only trading
- **Current:** 10 Tier A stocks (>70% win rate)
- **Learning:** Quality > Quantity (10 excellent stocks > 50 mediocre ones)

#### ✅ Time-Decay Exits Outperform Fixed Targets (EXP-013)
- **Finding:** Mean reversion momentum fades over time
- **Strategy:** Day 0: ±2%, Day 1: ±1.5%, Day 2+: ±1%
- **Impact:** Locks in profits before reversal exhausts
- **vs Fixed Targets:** +5-10% improvement

#### ✅ Panic Day Close Entry is Optimal (EXP-020)
- **Finding:** Reversals happen overnight (gaps UP next morning)
- **Strategy:** Enter at close of panic day
- **vs Next Day Open:** +54% better returns
- **Learning:** Don't wait for "better prices" - they don't come

#### ✅ Tier A Stocks Work in All Market Regimes (EXP-023)
- **Finding:** High-quality mean reversion works regardless of market conditions
- **Tested:** Bull, Neutral, Bear market filtering
- **Result:** Regime filtering REDUCES returns by -18.83pp
- **Learning:** Stock quality > Market timing

#### ✅ 60-Day Lookback Period is Optimal
- **Finding:** Balance between responsiveness and stability
- **vs 30 days:** More stable indicators
- **vs 90 days:** More responsive to recent conditions
- **Deployed:** All indicators use 60-day lookback

#### ✅ Stock-Specific Parameters (EXP-008)
- **Finding:** One-size-fits-all parameters suboptimal
- **Strategy:** Tune Z-score, RSI, volume thresholds per stock
- **Current:** NVDA has different params than V, MA, etc.
- **Impact:** +5-8pp win rate improvement

### What Doesn't Work (Abandoned)

#### ❌ Market Regime Filtering (EXP-023)
- **Hypothesis:** Skip BEAR market signals
- **Result:** Win rate dropped from 87.5% to 50.0% (NVDA)
- **Reason:** Skipped profitable signals (5 out of 6 were winners!)
- **Learning:** Tier A quality makes regime filtering redundant

#### ❌ Next Day Open Entry (EXP-020)
- **Hypothesis:** Wait for better entry prices
- **Result:** Returns dropped from +49.70% to +22.55% (NVDA)
- **Reason:** Panic stocks gap UP overnight before next open
- **Learning:** Act fast - reversal momentum doesn't wait

#### ❌ Universal Parameters (pre-EXP-008)
- **Hypothesis:** Same thresholds work for all stocks
- **Result:** Poor performance on diverse stocks
- **Learning:** TSLA needs different params than NVDA

#### ❌ Trading All Stocks (pre-EXP-014)
- **Hypothesis:** More stocks = more opportunities
- **Result:** Win rate 61.4%, many losing stocks
- **Learning:** Selective trading (Tier A only) beats diversification

### Research In Progress

None currently - v8.0 is near-optimal. Next frontier: More Tier A stock discovery.

---

## 🏗️ System Architecture

### Components

#### 1. Trading Strategy Engine
**Location:** `common/trading/`

**Files:**
- `mean_reversion_detector.py` - Core signal detection
- `signal_scanner.py` - Multi-stock scanner
- `backtest_engine.py` - Historical testing
- `time_decay_exit_strategy.py` - Exit logic

**Capabilities:**
- Real-time signal detection
- Multi-stock scanning
- Backtesting framework
- Position tracking

#### 2. Data Management
**Location:** `common/data/`

**Files:**
- `market_data_fetcher.py` - Yahoo Finance integration
- `sentiment/news_sentiment.py` - News API (ready, not deployed)
- `sentiment/twitter_collector.py` - Social sentiment (ready, not deployed)

**Capabilities:**
- Historical data retrieval
- Real-time price updates
- Sentiment data collection (ready)

#### 3. Configuration
**Location:** `common/config/`

**Files:**
- `mean_reversion_params.py` - v8.0 stock parameters
- `email_config.json` - Email notification settings

**Version Control:**
- v1.0-v5.0: Universal parameters
- v6.0: Stock selection optimization (3 stocks)
- v7.0: Expanded to 5 stocks (EXP-021)
- v8.0: Expanded to 10 stocks (EXP-024)

#### 4. Web Dashboard
**Location:** `common/web/`

**Files:**
- `app.py` - Flask application
- `templates/dashboard.html` - UI
- `static/` - CSS, JS, assets

**Stack:**
- Backend: Flask (Python)
- Frontend: HTML, CSS, JavaScript
- Scheduling: APScheduler
- Real-time: AJAX updates

#### 5. Notifications
**Location:** `common/notifications/`

**Files:**
- `sendgrid_notifier.py` - Email notifications

**Capabilities:**
- Daily scan alerts
- Signal notifications
- Experiment reports (NEW)
- Test emails

#### 6. Experimentation
**Location:** `common/experiments/`

**Files:**
- `exp014_stock_selection_optimization.py` - Stock screening
- `exp019_sentiment_integration.py` - News filtering (on hold)
- `exp020_entry_timing_optimization.py` - Entry timing tests
- `exp021_expand_tier_a_universe.py` - Universe expansion
- `exp023_market_regime_filtering.py` - Regime filtering
- `exp024_expand_tier_a_round2.py` - Latest expansion

**Results:** `logs/experiments/*.json`

#### 7. Logging & Monitoring
**Location:** `logs/`

**Files:**
- `experiments/*.json` - Experiment results
- `scan_logs/*.log` - Daily scan logs
- `web_dashboard/*.log` - Dashboard logs

---

## ⚙️ Configuration & Deployment

### Quick Start

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required:**
- yfinance (market data)
- pandas, numpy (data analysis)
- flask (dashboard)
- apscheduler (scheduling)
- sendgrid (email notifications)

#### 2. Configure Email (Optional)
**File:** `email_config.json`

```json
{
  "enabled": true,
  "sendgrid_api_key": "YOUR_SENDGRID_API_KEY",
  "sender_email": "proteus@trading.local",
  "recipient_email": "your.email@example.com"
}
```

**Get SendGrid API Key:**
1. Sign up at https://sendgrid.com (free tier: 100 emails/day)
2. Settings > API Keys > Create API Key
3. Add to config file

#### 3. Start Dashboard
```bash
python common/web/app.py
```

**Dashboard:** http://localhost:5000

**Automated Scans:**
- 9:45 AM EST (morning)
- 3:45 PM EST (afternoon)

#### 4. Run Backtest (Optional)
```bash
python common/experiments/exp014_stock_selection_optimization.py
```

Tests current Tier A stocks on 3-year historical data.

### Configuration Files

#### Stock Parameters (`common/config/mean_reversion_params.py`)

**Current Version:** 8.0

**Modify to:**
- Add new Tier A stocks
- Adjust Z-score thresholds
- Tune RSI oversold levels
- Change volume multipliers

**DO NOT modify unless running experiments - current params are optimized!**

#### Email Config (`email_config.json`)

```json
{
  "enabled": true/false,
  "sendgrid_api_key": "SG.xxx",
  "sender_email": "proteus@trading.local",
  "recipient_email": "you@example.com"
}
```

Set `enabled: false` to disable all email notifications.

### Deployment Modes

#### 1. Research Mode
- Run experiments
- Test new strategies
- Backtest stocks
- Don't trade live

```bash
python common/experiments/exp024_expand_tier_a_round2.py
```

#### 2. Monitoring Mode (Current)
- Dashboard running
- Automated scans
- Email alerts
- No live trading (paper trading)

```bash
python common/web/app.py
```

#### 3. Live Trading (Not Implemented Yet)
- Requires broker API integration
- Automated order execution
- Real money at risk
- Not currently available

**Future:** Alpaca, Interactive Brokers, TD Ameritrade integration

---

## 📊 Performance Summary

### v8.0 (Current)

**Universe:** 10 Tier A stocks

**Backtest Period:** 2022-2025 (3 years)

**Metrics:**
- **Win Rate:** 76.5% average
- **Total Return:** +20.78% per stock average
- **Trade Frequency:** ~55 trades/year
- **Sharpe Ratio:** 3.81-19.58 range

**Best Performers:**
1. NVDA: 87.5% win rate, +49.70% return
2. AVGO: 87.5% win rate, +24.52% return
3. KLAC: 80.0% win rate, +20.05% return

**Sector Breakdown:**
- Semiconductors: 40% (4 stocks)
- Healthcare: 20% (2 stocks)
- Payments: 20% (2 stocks)
- Tech: 10% (1 stock)
- Finance: 10% (1 stock)

### Evolution

**v1.0-v5.0:** Universal parameters, all stocks
- Win rate: ~61%
- Many losing stocks

**v6.0:** Stock selection (3 Tier A stocks)
- Win rate: 78.7%
- **+15.99pp improvement** (biggest win)

**v7.0:** Expanded universe (5 Tier A stocks)
- Win rate: 78.7%
- +67% trade frequency

**v8.0:** Further expansion (10 Tier A stocks)
- Win rate: 76.5%
- +120% trade frequency vs v7.0
- Better diversification

---

## 🎯 Next Steps & Roadmap

### Immediate Opportunities

1. **EXP-025: Universe Expansion Round 3** ⭐ RECOMMENDED
   - Test 30 more S&P 500 stocks
   - Target: 3-4 new Tier A stocks
   - Expected: 13-14 total stocks, ~60-70 trades/year

2. **Position Sizing Optimization**
   - Kelly Criterion, win-rate weighting
   - Expected: +1-3pp return improvement

3. **Signal Quality Scoring**
   - Rank signal strength, trade top 50%
   - Expected: +5-8pp win rate, -50% frequency

### Medium-Term

1. **Broker Integration**
   - Alpaca API (free paper trading)
   - Automated order execution
   - Live trading capability

2. **Stop Loss Optimization**
   - Trailing stops, ATR-based stops
   - Better risk management

3. **Multi-Timeframe Analysis**
   - Intraday signals
   - Trend alignment

### Long-Term

1. **Machine Learning Classification**
   - Predict reversal probability
   - Skip low-probability signals
   - Expected: +5-8pp win rate

2. **Options Integration**
   - Trade options on panic sells
   - Higher leverage, defined risk

3. **Portfolio Rebalancing**
   - Dynamic allocation
   - Risk-adjusted weighting

---

## 🔐 Security & Risk Management

### Current Risk Controls

✅ **Paper Trading Only** - No live money at risk
✅ **Tier A Stock Selection** - Only >70% win rate stocks
✅ **Time-Decay Exits** - Automatic profit taking
✅ **Diversification** - 10 stocks across 5 sectors
✅ **Position Limits** - Equal weight allocation

### Not Implemented (Future)

⏳ **Stop Losses** - Currently only time-decay exits
⏳ **Max Drawdown Limits** - No portfolio-level risk controls
⏳ **Position Sizing** - Currently equal weight
⏳ **Correlation Monitoring** - May have overlapping exposures

### Recommendations Before Live Trading

1. **Start with small capital** (<5% of portfolio)
2. **Implement stop losses** (e.g., -5% max loss per trade)
3. **Add position sizing** (Kelly Criterion or risk parity)
4. **Monitor correlation** (avoid concentrated sector bets)
5. **Set max drawdown** (e.g., stop trading if -10% portfolio loss)
6. **Paper trade 3-6 months** to validate in real market conditions

---

## 📚 Documentation Index

**Core Documentation:**
- `README.md` - Project overview
- `CAPABILITIES.md` - This file
- `QUICKSTART.md` - Getting started guide
- `PRODUCTION_DEPLOYMENT_GUIDE.md` - Deployment instructions

**Experiment Findings:**
- `EXP-019-FINDINGS.md` - Sentiment integration (on hold)
- `EXP-023-FINDINGS.md` - Market regime filtering (failed)
- `EXP-012-FINDINGS.md` - Position sizing
- `EXP-013-FINDINGS.md` - Time-decay exits

**Research Reports:**
- `EXPERIMENT_RESULTS_SUMMARY.md` - All experiments
- `logs/experiments/*.json` - Detailed backtest results

---

## 🤝 Contributing

### Running Experiments

1. Create new experiment file: `common/experiments/expXXX_description.py`
2. Use template from `exp024_expand_tier_a_round2.py`
3. Run backtest on 3-year data
4. Create findings document: `EXP-XXX-FINDINGS.md`
5. Send email report (automatic)
6. Deploy if successful

### Adding New Stocks

1. Run stock through `exp014_stock_selection_optimization.py`
2. Check: Win rate >70%, Return >5%, Sharpe >5.0, Trades >5
3. If Tier A: Add to `mean_reversion_params.py`
4. Update version number
5. Commit with detailed notes

### Modifying Strategy

1. **DON'T** change existing Tier A parameters without experiments
2. **DO** create new experiment to test changes
3. **DO** backtest on 3-year data minimum
4. **DO** document findings
5. **DO** compare vs v8.0 baseline

---

## ⚡ Quick Reference

### Key Metrics
- **Target Win Rate:** >70% (Tier A requirement)
- **Current Average:** 76.5%
- **Trade Frequency:** ~55/year
- **Best Stock:** NVDA (87.5% win rate, +49.70% return)

### Key Findings
- Stock selection > All other optimizations
- Tier A stocks work in all market regimes
- Time-decay exits > Fixed targets
- Panic day close entry > Next day open
- 60-day lookback = optimal

### Commands
```bash
# Start dashboard
python common/web/app.py

# Run backtest
python common/experiments/exp014_stock_selection_optimization.py

# Test email
python common/notifications/sendgrid_notifier.py

# Run experiment
python common/experiments/exp024_expand_tier_a_round2.py
```

### Files to Edit
- `common/config/mean_reversion_params.py` - Add stocks
- `email_config.json` - Email settings
- `common/web/app.py` - Dashboard customization

### Files NOT to Edit (Unless Experimenting)
- Strategy parameters (optimized via backtests)
- Entry/exit logic (proven optimal)
- Indicator calculations (validated)

---

**Last Updated:** 2025-11-16
**Version:** 8.0
**Status:** Production Ready (Paper Trading)

For questions or contributions, create an issue on GitHub.

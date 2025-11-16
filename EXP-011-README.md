# EXP-011: Sentiment-Enhanced Panic Sell Detection

## Overview

**Objective:** Enhance mean reversion strategy with sentiment analysis to distinguish emotion-driven panics (mean revert quickly) from information-driven panics (justified drops to avoid).

**Research Basis:** "Predicting and Profiting from the Stock Market" comprehensive research report
**Status:** Phase 1 Complete - Data Collection Infrastructure Ready
**Date Started:** 2025-11-16

---

## Research Motivation

### Key Insights from Research Report

1. **Sentiment Distinguishes Panic Types**
   - Emotion-driven panics: Negative social media + neutral/positive news → Mean revert quickly (BUY)
   - Information-driven panics: Negative social + negative news → Justified drops (AVOID)

2. **Social Media Predictive Power**
   - Twitter sentiment predicts intraday trends: 68.7% accuracy
   - Social media outperforms news for daily returns: 67% vs 34%
   - Reddit r/wallstreetbets strong indicator of retail panic/euphoria

3. **Signal Quality Improvement**
   - Current v5.0 win rate: 77.3%
   - Expected with sentiment: 85%+ (filter false signals)
   - Better Sharpe ratio by avoiding information-driven drops

---

## Implementation Architecture

### Phase 1: Data Collection (COMPLETE ✓)

**Modules Created:**

1. **`src/data/sentiment/twitter_collector.py`**
   - Collects tweets mentioning stock tickers
   - Uses Twitter API v2 with tweepy
   - Focuses on original tweets (excludes retweets/replies)
   - Aggregates by day with engagement metrics

2. **`src/data/sentiment/reddit_collector.py`**
   - Monitors r/wallstreetbets, r/stocks, r/investing, r/StockMarket
   - Uses Reddit API (PRAW)
   - Detects hot/trending posts for real-time panic detection
   - Upvote ratio as consensus indicator

3. **`src/data/sentiment/news_collector.py`**
   - Collects financial news headlines
   - Uses Alpha Vantage News API (free tier: 25 calls/day)
   - Pre-scored sentiment from API
   - Filters by relevance score

4. **`src/models/sentiment/finbert_analyzer.py`**
   - FinBERT: Financial-specific BERT model
   - Pre-trained on earnings calls, news, social media
   - 87% accuracy on financial text vs 72% for general models
   - GPU-accelerated inference (~10ms per text)

5. **`src/data/sentiment/sentiment_features.py`**
   - Combines Twitter + Reddit + News
   - Calculates derived features:
     * Social sentiment (Twitter + Reddit combined)
     * News sentiment (professional sources)
     * Sentiment divergence (emotion vs information)
     * Sentiment velocity (rate of change)
   - Classifies signal confidence: HIGH/MEDIUM/LOW

### Phase 2: Enhanced Signal Detection (COMPLETE ✓)

**Enhanced Detector:** `src/experiments/exp011_sentiment_enhanced_panic.py`

**Signal Classification Logic:**

```python
HIGH CONFIDENCE (Emotion-driven panic - BUY):
  - Social sentiment < -0.3 (retail panic)
  - News sentiment > -0.1 (fundamentals OK)
  → Mean reversion opportunity

MEDIUM CONFIDENCE (Uncertain):
  - Mixed or divergent signals
  - Limited data availability
  → Trade with caution

LOW CONFIDENCE (Information-driven - AVOID):
  - Social sentiment < -0.3 (retail panic)
  - News sentiment < -0.3 (fundamentals bad)
  - Divergence < 0.3 (aligned negative)
  → Fundamental issue, avoid trade
```

### Phase 3: Backtesting (IN PROGRESS)

**Experiment File:** `src/experiments/exp011_sentiment_enhanced_panic.py`

**Comparison:**
- Baseline v5.0: No sentiment filtering
- Sentiment-Enhanced: Filters out LOW confidence signals

**Metrics to Compare:**
- Win rate: 77.3% → 85%+ (expected)
- Sharpe ratio improvement
- Total return
- Max drawdown
- Trade count (fewer but higher quality)

---

## Setup Instructions

### 1. Install Dependencies

```bash
# Core NLP and data collection
pip install transformers torch tweepy praw requests

# Optional: GPU support for faster FinBERT inference
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Configure API Keys

**Twitter API** (Optional - for Twitter sentiment)

1. Create developer account: https://developer.twitter.com
2. Create app and get bearer token
3. Create `twitter_config.json`:

```json
{
  "bearer_token": "YOUR_BEARER_TOKEN_HERE",
  "enabled": true
}
```

**Reddit API** (Optional - for Reddit sentiment)

1. Create app: https://www.reddit.com/prefs/apps
2. Choose "script" type
3. Create `reddit_config.json`:

```json
{
  "client_id": "YOUR_CLIENT_ID_HERE",
  "client_secret": "YOUR_CLIENT_SECRET_HERE",
  "user_agent": "proteus-trading-bot/1.0",
  "enabled": true
}
```

**Alpha Vantage API** (Optional - for News sentiment)

1. Get free key: https://www.alphavantage.co/support/#api-key
2. Create `news_config.json`:

```json
{
  "alphavantage_api_key": "YOUR_API_KEY_HERE",
  "enabled": true
}
```

**Note:** All APIs are optional. Strategy falls back to baseline v5.0 if sentiment data unavailable.

### 3. Test Individual Components

```bash
# Test Twitter collector
python src/data/sentiment/twitter_collector.py

# Test Reddit collector
python src/data/sentiment/reddit_collector.py

# Test News collector
python src/data/sentiment/news_collector.py

# Test FinBERT analyzer (downloads ~440MB model on first run)
python src/models/sentiment/finbert_analyzer.py

# Test full pipeline
python src/data/sentiment/sentiment_features.py
```

---

## Running the Experiment

### Baseline vs Sentiment Comparison

```bash
# Run full comparison (baseline vs sentiment-enhanced)
python src/experiments/exp011_sentiment_enhanced_panic.py
```

This will:
1. Run baseline v5.0 backtest (no sentiment)
2. Run sentiment-enhanced backtest (filters LOW confidence)
3. Compare performance metrics
4. Save results to `logs/experiments/`

### Single Mode Testing

```python
from src.experiments.exp011_sentiment_enhanced_panic import run_sentiment_enhanced_backtest

# Test baseline only
baseline_results = run_sentiment_enhanced_backtest(
    ticker='NVDA',
    period='3y',
    use_sentiment=False
)

# Test sentiment-enhanced
sentiment_results = run_sentiment_enhanced_backtest(
    ticker='NVDA',
    period='3y',
    use_sentiment=True
)
```

---

## Expected Outcomes

### Performance Targets

| Metric | Baseline v5.0 | Expected EXP-011 | Improvement |
|--------|---------------|------------------|-------------|
| Win Rate | 77.3% | 85%+ | +7.7pp |
| Sharpe Ratio | Varies by stock | Higher | Better risk-adj returns |
| Total Trades | Higher | Lower | Higher quality signals |
| False Positives | Higher | Lower | Filter info-driven drops |

### Success Criteria

✅ **SUCCESS if:**
- Win rate > baseline v5.0
- Sharpe ratio > baseline v5.0
- Max drawdown < baseline v5.0

⚠️ **PARTIAL SUCCESS if:**
- Win rate improved but other metrics mixed
- Need further optimization

❌ **NEEDS WORK if:**
- No improvement over baseline
- Review sentiment data quality and classification logic

---

## File Structure

```
proteus/
├── src/
│   ├── data/
│   │   └── sentiment/
│   │       ├── __init__.py
│   │       ├── twitter_collector.py      # Twitter API integration
│   │       ├── reddit_collector.py       # Reddit API integration
│   │       ├── news_collector.py         # Alpha Vantage News API
│   │       └── sentiment_features.py     # Feature engineering pipeline
│   ├── models/
│   │   └── sentiment/
│   │       ├── __init__.py
│   │       └── finbert_analyzer.py       # FinBERT NLP model
│   └── experiments/
│       └── exp011_sentiment_enhanced_panic.py  # Main experiment
│
├── twitter_config.json (optional)
├── reddit_config.json (optional)
├── news_config.json (optional)
└── EXP-011-README.md (this file)
```

---

## Next Steps

### Phase 3: Backtesting (Current)
- [x] Create experiment file
- [x] Implement baseline vs sentiment comparison
- [ ] Run on all 11 monitored stocks
- [ ] Analyze results and optimize thresholds

### Phase 4: Optimization
- [ ] Tune sentiment thresholds (-0.3, -0.1)
- [ ] Optimize divergence threshold (0.3)
- [ ] Test different weighting schemes (Twitter vs Reddit)
- [ ] A/B test on different time periods

### Phase 5: Production Integration
- [ ] Integrate into `src/trading/signal_scanner.py`
- [ ] Add sentiment features to dashboard UI
- [ ] Update email notifications with confidence levels
- [ ] Deploy to paper trading with sentiment filtering
- [ ] Monitor real-world performance

---

## Research Attribution

This experiment is based on insights from:
- "Predicting and Profiting from the Stock Market: A Comprehensive Strategy Research Report"
- Key recommendation: Sentiment-driven trading with emotion vs information distinction
- Academic research on Twitter/Reddit sentiment and stock returns

---

## API Usage & Costs

### Free Tier Limits

| Service | Free Tier | Usage for 11 Stocks |
|---------|-----------|---------------------|
| Twitter API v2 | 500K tweets/month | ~100 tweets/stock/day = OK |
| Reddit API (PRAW) | 60 requests/min | ~5 requests/stock/day = OK |
| Alpha Vantage News | 25 calls/day | 2-3 stocks/day = Need rotation |

**Cost:** All APIs used are FREE tier. No subscription required.

**Optimization for Alpha Vantage:**
- Cache news data daily
- Prioritize stocks with active signals
- Rotate through stock universe (3 stocks/day)

---

## Troubleshooting

### FinBERT Model Not Loading
```bash
# Install PyTorch with CUDA support (if you have GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CPU-only version
pip install torch torchvision torchaudio
```

### Twitter API Errors
- Check bearer token is correct
- Verify Twitter Developer account is approved
- Free tier only accesses last 7 days (recent search)

### Reddit API Errors
- Ensure you selected "script" type when creating app
- User agent must be unique and descriptive
- Avoid excessive requests (rate limit: 60/min)

### Alpha Vantage Rate Limits
- Free tier: 25 calls/day maximum
- Add 1-second delays between requests
- Cache results to avoid re-fetching

---

## Performance Monitoring

Once deployed, monitor:

1. **Sentiment Data Quality**
   - Check daily data availability
   - Monitor API failures
   - Validate sentiment scores make sense

2. **Signal Classification**
   - Track HIGH/MEDIUM/LOW distribution
   - Compare to baseline signal count
   - Verify LOW confidence signals are avoided

3. **Trading Performance**
   - Win rate by confidence level
   - Sharpe ratio improvement
   - Drawdown reduction

---

## Contact & Support

For questions or issues:
- Review sentiment collector test scripts
- Check API configuration files
- Verify FinBERT model downloaded successfully
- Monitor logs/experiments/ for backtest results

---

**Last Updated:** 2025-11-16
**Version:** 1.0
**Status:** Phase 1 Complete, Phase 3 In Progress

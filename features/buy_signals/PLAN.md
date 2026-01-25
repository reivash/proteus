# Recommendation Engine: Daily & Weekly Trading Recommendations

> **Project Status**: Production (Complete)
> **Version**: 1.0
> **Last Updated**: January 2026
> **Output**: Actionable buy recommendations with confidence levels

---

## Overview

### Problem Statement
Raw scanner signals need to be translated into clear, actionable recommendations for trading decisions. Users need to understand not just "what to buy" but "why" and "how much confidence" to have.

### Objectives
1. **Clear Recommendations**: Ranked buy list with honest confidence levels
2. **Edge Metrics**: Show backtest-validated expected performance
3. **Risk Context**: Include regime, sector, and earnings context
4. **Delivery**: Email, console, and JSON output formats

### Key Achievements
- High-quality recommendation system with honest confidence
- Backtest-driven edge metrics (win rate, expected return)
- Forward validation infrastructure
- Multiple delivery channels (email, console, JSON)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       RECOMMENDATION ENGINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   SCANNER    │    │ RECOMMENDATION│    │   DELIVERY   │                  │
│  │   SIGNALS    │───▶│   GENERATOR   │───▶│   CHANNELS   │                  │
│  │              │    │               │    │              │                  │
│  │ - SignalV2   │    │ - Ranking     │    │ - Email      │                  │
│  │ - Regime     │    │ - Edge calc   │    │ - Console    │                  │
│  │ - Context    │    │ - Confidence  │    │ - JSON       │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                             │
│                       RECOMMENDATION OUTPUT                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Rank │ Ticker │ Score │ Confidence │ Edge    │ Position │ Targets   │ │
│  │  1    │ MPC    │ 78    │ HIGH       │ +1.2%   │ 12%      │ +2%/+3.5% │ │
│  │  2    │ KLAC   │ 72    │ MODERATE   │ +0.9%   │ 10%      │ +2%/+3%   │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

> **Note**: This project is COMPLETE. All checkboxes document the production system.

---

### Phase 1: Recommendation Generation
**Goal**: Transform scanner signals into ranked recommendations

#### 1.1 Signal Ranking
- [x] **1.1.1 Primary ranking criteria**
  **DoD**:
  - [x] Sort by adjusted signal strength (descending)
  - [x] Apply tier bonus (elite +5, strong +3)
  - [x] Apply regime bonus (volatile/bear +5)
  - [x] Final rank score calculated

- [x] **1.1.2 Filter non-actionable**
  **DoD**:
  - [x] Remove skip_trade = True
  - [x] Remove near_earnings = True
  - [x] Remove sector-limited signals
  - [x] Log filter reasons

- [x] **1.1.3 Top N selection**
  **DoD**:
  - [x] Configurable top N (default 5)
  - [x] Respect portfolio constraints
  - [x] Never recommend more than available slots

#### 1.2 Edge Calculation
- [x] **1.2.1 Historical edge lookup**
  **DoD**:
  - [x] Lookup historical performance by:
    - Ticker + regime combination
    - Signal strength bucket
    - Day of week
  - [x] Return expected win rate, avg return

- [x] **1.2.2 Edge confidence bands**
  **DoD**:
  - [x] Calculate sample size for edge estimate
  - [x] Show confidence interval
  - [x] Flag low sample size (<30 trades)

- [x] **1.2.3 Expected value calculation**
  ```python
  expected_value = win_rate * avg_win + (1 - win_rate) * avg_loss
  ```
  **DoD**:
  - [x] Calculate per recommendation
  - [x] Positive EV required for recommendation

#### Phase 1 Quality Gate: PASSED
- [x] Ranking produces intuitive results
- [x] Edge metrics match backtest data
- [x] Expected value positive for all recommendations

---

### Phase 2: Confidence Assessment
**Goal**: Honest confidence levels based on evidence quality

#### 2.1 Confidence Scoring
- [x] **2.1.1 Confidence components**
  **DoD**:
  - [x] Signal strength (>70 = +20 confidence)
  - [x] Tier quality (elite/strong = +15)
  - [x] Regime favorability (volatile = +15)
  - [x] Sample size (>100 = +10)
  - [x] Recent performance (last 30 days)

- [x] **2.1.2 Confidence levels**
  **DoD**:
  - [x] HIGH: 70+ confidence score
  - [x] MODERATE: 50-69
  - [x] LOW: 30-49
  - [x] SPECULATIVE: <30
  - [x] Map to recommendation language

- [x] **2.1.3 Caveats generation**
  **DoD**:
  - [x] List specific concerns:
    - "Low sample size for this pattern"
    - "Earnings within 5 days"
    - "Sector showing weakness"
    - "Choppy regime reduces reliability"
  - [x] Include in recommendation

#### 2.2 Honest Communication
- [x] **2.2.1 Uncertainty acknowledgment**
  **DoD**:
  - [x] Never claim certainty
  - [x] Use probabilistic language
  - [x] "Based on 87 similar trades, expected win rate is 62%"

- [x] **2.2.2 Limitation disclosure**
  **DoD**:
  - [x] State backtest vs live difference
  - [x] Acknowledge model limitations
  - [x] "Past performance does not guarantee..."

#### Phase 2 Quality Gate: PASSED
- [x] Confidence levels match evidence quality
- [x] Caveats are specific and actionable
- [x] Language is honest, not overselling

---

### Phase 3: Context Integration
**Goal**: Provide market and stock context

#### 3.1 Market Context
- [x] **3.1.1 Regime summary**
  **DoD**:
  - [x] Current regime with confidence
  - [x] Regime implications for trading
  - [x] "Volatile regime: mean reversion working well"

- [x] **3.1.2 Bear warning integration**
  **DoD**:
  - [x] Include bear score if elevated
  - [x] Adjust recommendations if WARNING+
  - [x] "Bear score 52 (WARNING): reduce position sizes"

- [x] **3.1.3 Macro context**
  **DoD**:
  - [x] VIX level
  - [x] SPY trend (up/down/flat)
  - [x] Economic indicators summary

#### 3.2 Stock-Specific Context
- [x] **3.2.1 Recent price action**
  **DoD**:
  - [x] 5-day performance
  - [x] Distance from 50-day MA
  - [x] Drawdown from recent high

- [x] **3.2.2 Sector context**
  **DoD**:
  - [x] Sector momentum
  - [x] Stock vs sector performance
  - [x] Sector rotation status

- [x] **3.2.3 Earnings proximity**
  **DoD**:
  - [x] Days until earnings
  - [x] Warning if <5 days
  - [x] Skip if <3 days

#### Phase 3 Quality Gate: PASSED
- [x] Context adds value to recommendations
- [x] Bear warning properly integrated
- [x] Stock-specific details accurate

---

### Phase 4: Output Formats
**Goal**: Multiple delivery channels

#### 4.1 Console Output
- [x] **4.1.1 Summary table**
  ```
  ═══════════════════════════════════════════════════════════════
  TODAY'S RECOMMENDATIONS (2026-01-24)
  Regime: VOLATILE | Bear Score: 32 (WATCH) | VIX: 18.2
  ═══════════════════════════════════════════════════════════════
  Rank  Ticker  Score  Confidence  Edge     Position  Targets
  1     MPC     78     HIGH        +1.2%    12%       +2%/+3.5%
  2     KLAC    72     MODERATE    +0.9%    10%       +2%/+3%
  3     V       68     MODERATE    +0.8%    8%        +2%/+2.5%
  ═══════════════════════════════════════════════════════════════
  ```
  **DoD**:
  - [x] Clean ASCII formatting
  - [x] Color coding by confidence
  - [x] All key fields visible

- [x] **4.1.2 Detailed breakdown**
  **DoD**:
  - [x] Per-stock detail section
  - [x] Boosts/penalties applied
  - [x] Historical performance stats

#### 4.2 Email Delivery
- [x] **4.2.1 HTML email template**
  **DoD**:
  - [x] Professional formatting
  - [x] Mobile-responsive
  - [x] Color-coded confidence
  - [x] Quick action links

- [x] **4.2.2 Daily morning email**
  **DoD**:
  - [x] Sent at configurable time (default 9:15 AM ET)
  - [x] Subject: "Proteus Recommendations - [DATE]"
  - [x] Top recommendations with context

- [x] **4.2.3 Weekly summary email**
  **DoD**:
  - [x] Sent Sunday evening
  - [x] Week's performance recap
  - [x] Coming week watchlist

#### 4.3 JSON Output
- [x] **4.3.1 Structured JSON format**
  ```json
  {
    "date": "2026-01-24",
    "regime": "VOLATILE",
    "bear_score": 32,
    "recommendations": [
      {
        "rank": 1,
        "ticker": "MPC",
        "score": 78,
        "confidence": "HIGH",
        "edge": {
          "win_rate": 0.68,
          "expected_return": 0.012,
          "sample_size": 127
        },
        "position": {
          "size_pct": 12,
          "shares": 45,
          "dollar_size": 5400
        },
        "targets": {
          "profit_1": 0.02,
          "profit_2": 0.035,
          "stop_loss": -0.03,
          "max_days": 5
        },
        "caveats": []
      }
    ]
  }
  ```
  **DoD**:
  - [x] Schema documented
  - [x] Machine-readable
  - [x] Versioned format

- [x] **4.3.2 File output**
  **DoD**:
  - [x] `recommendations_YYYYMMDD.json`
  - [x] Latest symlink: `latest_recommendations.json`
  - [x] Archive directory

#### Phase 4 Quality Gate: PASSED
- [x] All output formats working
- [x] Email delivery confirmed
- [x] JSON schema consistent

---

### Phase 5: Validation & Tracking
**Goal**: Track recommendation accuracy

#### 5.1 Forward Validation
- [x] **5.1.1 Recommendation logging**
  **DoD**:
  - [x] Log all recommendations to database/file
  - [x] Track: date, ticker, score, confidence, price
  - [x] Enable post-hoc analysis

- [x] **5.1.2 Outcome tracking**
  **DoD**:
  - [x] Track 3-day, 5-day outcomes
  - [x] Hit target 1? Target 2? Stop?
  - [x] Calculate actual vs expected

- [x] **5.1.3 Accuracy reporting**
  **DoD**:
  - [x] Weekly accuracy summary
  - [x] By confidence level
  - [x] By regime
  - [x] Alert if accuracy drops

#### 5.2 Feedback Loop
- [x] **5.2.1 Confidence calibration**
  **DoD**:
  - [x] Check: HIGH confidence = 70%+ actual win rate?
  - [x] Adjust confidence thresholds if needed
  - [x] Quarterly review

#### Phase 5 Quality Gate: PASSED
- [x] All recommendations tracked
- [x] Accuracy metrics calculated
- [x] Feedback loop established

---

## Progress Summary

| Phase | Status | Checkboxes | Quality Gate |
|-------|--------|------------|--------------|
| 1. Recommendation Generation | ✅ Complete | 6/6 | ✅ |
| 2. Confidence Assessment | ✅ Complete | 5/5 | ✅ |
| 3. Context Integration | ✅ Complete | 6/6 | ✅ |
| 4. Output Formats | ✅ Complete | 7/7 | ✅ |
| 5. Validation & Tracking | ✅ Complete | 4/4 | ✅ |
| **TOTAL** | **✅ Production** | **28/28** | |

---

## Production Usage

### Daily Commands
```bash
# Generate today's recommendations
python scripts/full_recommendation.py

# Send morning email
python send_todays_recommendation.py

# Generate with specific output
python scripts/proteus_recommend.py --output recommendations.json --email
```

### Configuration
```json
{
  "recommendation": {
    "max_recommendations": 5,
    "min_signal_strength": 60,
    "min_confidence": "LOW",
    "delivery": {
      "email_time": "09:15",
      "email_enabled": true,
      "json_output": true
    }
  }
}
```

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/full_recommendation.py` | Main recommendation generator |
| `scripts/proteus_recommend.py` | CLI interface |
| `send_todays_recommendation.py` | Email sender |
| `src/notifications/sendgrid_notifier.py` | Email delivery |

---

## Sample Output

### Console
```
══════════════════════════════════════════════════════════════════
PROTEUS DAILY RECOMMENDATIONS - Friday, January 24, 2026
══════════════════════════════════════════════════════════════════

MARKET CONTEXT
--------------
Regime: VOLATILE (confidence: 78%)
Bear Score: 32 (WATCH)
VIX: 18.2 (elevated)
SPY 5-day: -2.1%

TODAY'S TOP PICKS
-----------------

#1 MPC (Marathon Petroleum)
   Signal: 78/100 | Tier: ELITE | Confidence: HIGH
   Edge: +1.2% expected (68% win rate, n=127)
   Position: 12% ($5,400) | Targets: +2%/+3.5% | Stop: -3%
   Context: Oil sector strong, 8% below 50-day MA

#2 KLAC (KLA Corporation)
   Signal: 72/100 | Tier: ELITE | Confidence: MODERATE
   Edge: +0.9% expected (62% win rate, n=89)
   Position: 10% ($4,500) | Targets: +2%/+3% | Stop: -3%
   Context: Semis oversold, earnings in 12 days
   Caveat: Sector momentum weak

#3 V (Visa)
   Signal: 68/100 | Tier: ELITE | Confidence: MODERATE
   Edge: +0.8% expected (61% win rate, n=156)
   Position: 8% ($3,600) | Targets: +2%/+2.5% | Stop: -3%
   Context: Financials stable, near support

══════════════════════════════════════════════════════════════════
DISCLAIMER: Past performance does not guarantee future results.
These are model-generated signals based on historical patterns.
══════════════════════════════════════════════════════════════════
```

---

## References

- [SYSTEM_STATE.md](../../SYSTEM_STATE.md) - Full system context
- [Smart Scanner V2](./SMART_SCANNER_V2.md) - Signal generation
- [unified_config.json](../../config/unified_config.json) - Configuration

---

*Document Version: 1.0*
*Last Updated: January 2026*
*Status: Production Complete*

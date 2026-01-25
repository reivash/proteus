# Single Stock Deep Analyzer: Revival Potential System

> **Project Codename**: Phoenix
> **Version**: 1.0 (Proposal)
> **Date**: January 2026
> **Scope**: Long-term development roadmap (3-6 months)

---

## Executive Summary

This proposal outlines the design for an **extremely powerful single-stock analyzer** focused on identifying stocks with **revival/recovery potential** - companies that have experienced significant declines but show signs of potential turnaround.

Unlike the existing SmartScannerV2 (which screens 54 stocks for mean-reversion signals), Phoenix provides **deep, comprehensive analysis of individual stocks** to answer the question: *"Will this beaten-down stock recover?"*

### Key Differentiators from Existing System

| Aspect | SmartScannerV2 | Phoenix Analyzer |
|--------|----------------|------------------|
| Scope | 54 validated stocks | Any single stock |
| Depth | Signal strength (0-100) | 200+ metrics across 8 dimensions |
| Focus | Short-term mean reversion | Medium/long-term recovery |
| Output | Buy/skip decision | Comprehensive report with thesis |
| Timeframe | 2-5 day holds | Weeks to months |

---

## Problem Statement

Investors frequently encounter stocks that have declined 30-70% and want to know:

1. **Is this a value trap or genuine opportunity?**
2. **Can the company survive financially?**
3. **What catalysts could drive recovery?**
4. **Are smart money investors accumulating?**
5. **What does the technical picture suggest about bottoming?**

**Example: Ubisoft (UBI.PA)**
- Down ~70% from 2021 highs
- Gaming industry headwinds
- But: Strong IP portfolio, potential acquisition target, new game pipeline

A powerful analyzer should synthesize technical, fundamental, sentiment, and catalyst data into an actionable assessment.

---

## Objectives

### Primary Goals

1. **Comprehensive Analysis**: Evaluate stocks across 8 key dimensions
2. **Revival Probability Score**: Quantified 0-100 score with confidence intervals
3. **Catalyst Detection**: Identify potential recovery triggers
4. **Risk Assessment**: Survival probability and downside scenarios
5. **Actionable Output**: Clear recommendation with entry zones and targets

### Secondary Goals

1. **Leverage Existing Infrastructure**: Reuse Proteus fetchers, models, and features
2. **Modular Architecture**: Each analyzer component usable independently
3. **Extensible Design**: Easy to add new data sources and metrics
4. **Backtestable**: Historical analysis capability for validation

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHOENIX: Single Stock Analyzer                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │   INPUT     │   │    DATA     │   │  ANALYSIS   │   │   OUTPUT    │     │
│  │             │   │   LAYER     │   │   ENGINE    │   │             │     │
│  │  - Ticker   │──▶│             │──▶│             │──▶│  - Report   │     │
│  │  - Options  │   │  8 Fetchers │   │ 8 Analyzers │   │  - Score    │     │
│  │             │   │             │   │             │   │  - Thesis   │     │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           8 ANALYSIS DIMENSIONS                              │
├──────────────┬──────────────┬──────────────┬──────────────┬─────────────────┤
│  TECHNICAL   │ FUNDAMENTAL  │  SENTIMENT   │   CATALYST   │  INSTITUTIONAL  │
│  - Momentum  │ - Valuation  │ - News       │ - Earnings   │  - 13F Filings  │
│  - Volume    │ - Balance    │ - Social     │ - M&A        │  - Insider Buy  │
│  - Patterns  │ - Cash Flow  │ - Analysts   │ - Management │  - Ownership    │
├──────────────┼──────────────┼──────────────┼──────────────┼─────────────────┤
│    MACRO     │    SECTOR    │  HISTORICAL  │    RISK      │                 │
│  - Regime    │ - Relative   │ - Similar    │ - Survival   │                 │
│  - Rates     │ - Rotation   │ - Patterns   │ - Scenarios  │                 │
│  - Economic  │ - Peers      │ - Cycles     │ - Downside   │                 │
└──────────────┴──────────────┴──────────────┴──────────────┴─────────────────┘
```

---

## Detailed Module Specifications

### Module 1: Technical Health Analyzer

**Purpose**: Detect capitulation, accumulation, and bottoming patterns.

#### 1.1 Capitulation Detection
```python
class CapitulationDetector:
    """
    Identifies panic selling events that often precede bottoms.

    Key Indicators:
    - Volume spike (>3x 20-day average)
    - Sharp price drop (>5% single day or >15% week)
    - VIX correlation spike
    - RSI <20 with divergence
    """

    def detect(self, ticker: str) -> CapitulationSignal:
        # Returns: severity (0-100), date, volume_multiplier, price_drop
```

**Reference**: [Capitulation in Markets](https://www.stockgro.club/blogs/stock-market-101/capitulation/) - Volume spikes are almost always marked at bottoms, with multiples higher than daily average.

#### 1.2 Accumulation/Distribution Analysis
```python
class AccumulationAnalyzer:
    """
    Detects quiet institutional accumulation.

    Patterns (from TraderLion research):
    - Tight price action on low volume = accumulation
    - High volume on up days, low volume on down days
    - Price range narrowing with support holding
    """
```

**Reference**: [Identifying Quiet Institutional Accumulation](https://traderlion.com/trading-strategies/identifying-quiet-institutional-accumulation/)

#### 1.3 Technical Indicators Suite
| Indicator | Purpose | Bullish Signal |
|-----------|---------|----------------|
| RSI (14) | Momentum | <30 oversold, divergence |
| MACD | Trend | Bullish crossover below zero |
| Bollinger Bands | Volatility | Price at lower band, bands narrowing |
| OBV | Volume flow | Rising while price flat |
| A/D Line | Accumulation | Rising divergence from price |
| Stochastic | Short-term momentum | %K crosses above %D <20 |
| ATR | Volatility | Decreasing = consolidation |
| 50/200 MA | Trend | Price reclaiming moving averages |

#### 1.4 Pattern Recognition
- **Double Bottom**: Classic reversal pattern
- **Inverse Head & Shoulders**: Bottoming formation
- **Falling Wedge**: Bullish continuation/reversal
- **Volume Climax**: Exhaustion selling

**Existing Infrastructure**: `TechnicalFeatureEngineer` provides RSI, MACD, Bollinger, ATR. Need to add OBV, A/D Line, pattern detection.

---

### Module 2: Fundamental Health Analyzer

**Purpose**: Assess financial survival probability and valuation.

#### 2.1 Balance Sheet Survival Metrics

**Reference**: [Distressed Debt Investing](http://www.distressed-debt-investing.com/2009/04/balance-sheet-analysis-introduction.html) - Seth Klarman emphasizes balance sheet analysis is critical for distressed securities.

```python
class SurvivalAnalyzer:
    """
    Key Metrics:
    - Current Ratio: >1.5 healthy, <1 danger
    - Quick Ratio: >1.0 can meet short-term obligations
    - Debt-to-Equity: Industry-relative
    - Interest Coverage: EBIT/Interest Expense >2x
    - Cash Runway: Months of operations at current burn
    - Altman Z-Score: Bankruptcy probability
    """

    def calculate_survival_probability(self) -> float:
        # 0-100 score based on weighted metrics
```

#### 2.2 Valuation Assessment
| Metric | What It Shows | Revival Signal |
|--------|---------------|----------------|
| P/E vs Historical | Relative cheapness | <50% of 5-year avg |
| P/B | Asset value | <1 (trading below book) |
| P/S | Revenue multiple | Sector low |
| EV/EBITDA | Enterprise value | <10 generally good |
| FCF Yield | Cash generation | >5% attractive |
| PEG Ratio | Growth-adjusted | <1 undervalued |

#### 2.3 Cash Flow Quality
```python
class CashFlowAnalyzer:
    """
    Cash flow is hardest to manipulate.

    Key Checks:
    - Operating CF vs Net Income (quality)
    - FCF trend (improving/declining)
    - CapEx sustainability
    - Dividend coverage
    """
```

**Data Sources**: Yahoo Finance provides financials. May need to add Financial Modeling Prep API for better historical data.

---

### Module 3: Sentiment Analyzer

**Purpose**: Gauge market psychology and contrarian signals.

#### 3.1 News Sentiment Pipeline
```python
class NewsSentimentAnalyzer:
    """
    Uses existing UnifiedSentimentScorer infrastructure.

    Enhancement: Track sentiment trajectory
    - Peak negativity often = bottom (news doesn't matter anymore)
    - Sentiment inflection point detection
    """
```

**Reference**: [Market Capitulation](https://www.evelyn.com/insights-and-events/insights/market-capitulation-what-does-it-look-like-and-are-we-there-yet/) - Bear markets end on bad news when market stops reacting.

#### 3.2 Analyst Ratings Tracker
| Signal | Meaning |
|--------|---------|
| Mass downgrades | Often lagging indicator (already priced in) |
| First upgrade after downgrades | Potential inflection |
| Price target compression | Expectations reset |

#### 3.3 Short Interest Analysis
```python
class ShortInterestAnalyzer:
    """
    High short interest can indicate:
    - Consensus bearishness (contrarian opportunity)
    - Squeeze potential (forced covering)

    Data Source: FINRA short interest, days to cover
    """
```

#### 3.4 Social Sentiment (Optional)
- Reddit mentions trend (existing `reddit_collector.py`)
- Twitter sentiment (existing `twitter_collector.py`)
- StockTwits sentiment

**Existing Infrastructure**: `UnifiedSentimentScorer` with FinBERT. Need to add analyst ratings API and short interest data.

---

### Module 4: Catalyst Detector

**Purpose**: Identify potential recovery triggers.

#### 4.1 Earnings Catalyst
```python
class EarningsCatalystAnalyzer:
    """
    Leverages existing EarningsCalendarFetcher.

    Analysis:
    - Days until next earnings
    - Historical earnings surprise pattern
    - Guidance trends
    - Estimate revisions (upward = positive)
    """
```

#### 4.2 M&A Potential Scorer

**Reference**: [How to Identify M&A Targets](https://grata.com/resources/mergers-acquisitions-targets) - Key metrics include revenue growth, EBITDA margins, and strategic fit.

```python
class MAPotentialAnalyzer:
    """
    Acquisition Target Score (0-100):

    Factors:
    - Strategic value (IP, market position, technology)
    - Valuation discount to peers
    - Industry consolidation trend
    - Previous acquisition interest/rumors
    - Balance sheet attractiveness
    """

    target_characteristics = {
        'undervalued_vs_peers': weight=0.25,
        'strategic_assets': weight=0.25,
        'manageable_debt': weight=0.20,
        'industry_consolidation': weight=0.15,
        'historical_interest': weight=0.15,
    }
```

#### 4.3 Management/Strategic Changes
- CEO/CFO changes
- Board additions
- Strategic review announcements
- Cost-cutting initiatives
- Asset sales

#### 4.4 Product/Business Catalysts
- New product launches
- FDA approvals (pharma)
- Contract wins
- Market expansion

**Data Sources**: News APIs, SEC filings (8-K), earnings call transcripts.

---

### Module 5: Institutional Flow Analyzer

**Purpose**: Track smart money movements.

#### 5.1 13F Filing Analysis

**Reference**: [How to Use 13F Filings](https://medium.com/@trading.dude/how-to-use-13f-filings-reading-the-hidden-hand-of-institutional-money-a5b7d07a514e) - Comparing positions between quarters reveals net institutional flow.

```python
class InstitutionalFlowAnalyzer:
    """
    Tracks quarterly 13F filings.

    Key Metrics:
    - Institutional ownership % change
    - Number of new positions vs exits
    - Quality of buyers (top funds vs small)
    - Crowding risk (too many holders = risk)
    """

    def analyze_flow(self, ticker: str) -> InstitutionalFlow:
        # Compare current vs previous quarter
        # Flag: price falling + institutional buying = accumulation
        # Flag: price falling + institutional selling = capitulation
```

#### 5.2 Insider Trading Signals

**Reference**: [TIKR Insider Analysis](https://www.tikr.com/blog/how-to-track-insider-buying-and-selling-for-the-stocks-you-own) - Insider buying is strongest when several executives buy around same time (cluster buying).

```python
class InsiderActivityAnalyzer:
    """
    SEC Form 4 Analysis.

    Strong Signals:
    - Cluster buying (multiple insiders)
    - CEO/CFO open market purchases
    - Buying after price decline
    - Cessation of selling

    Weak Signals:
    - Automatic plan sales (10b5-1)
    - Tax-related transactions
    """
```

**Data Sources**:
- SEC EDGAR (free, official)
- Financial Modeling Prep API (structured)
- WhaleWisdom (13F aggregator)

---

### Module 6: Macro Context Analyzer

**Purpose**: Assess broader market environment for recovery.

```python
class MacroContextAnalyzer:
    """
    Leverages existing UnifiedRegimeDetector + FREDMacroFetcher.

    Key Factors:
    - Market regime (Bull/Bear/Choppy/Volatile)
    - Recession probability
    - Interest rate trajectory
    - Liquidity conditions (M2)
    - Yield curve status
    - VIX level and trend
    """
```

**Regime Impact on Revival**:
| Regime | Revival Difficulty | Notes |
|--------|-------------------|-------|
| Bull | Easier | Rising tide lifts all boats |
| Volatile | Moderate | Sharp moves both ways |
| Choppy | Hard | No sustained moves |
| Bear | Hardest | Wait for regime change |

**Existing Infrastructure**: Fully covered by `UnifiedRegimeDetector` and `FREDMacroFetcher`.

---

### Module 7: Sector & Peer Analyzer

**Purpose**: Context through comparison.

#### 7.1 Sector Relative Performance
```python
class SectorAnalyzer:
    """
    Leverages existing CrossSectionalFeatures.

    Analysis:
    - Stock performance vs sector ETF
    - Sector rotation phase (accumulation/distribution)
    - Sector tailwinds/headwinds
    """
```

#### 7.2 Peer Comparison
```python
class PeerAnalyzer:
    """
    Compare against 5-10 closest peers.

    Metrics:
    - Relative valuation (P/E, P/S, EV/EBITDA)
    - Relative performance (1m, 3m, 6m, 1y)
    - Margin comparison
    - Growth comparison
    """
```

#### 7.3 Historical Peer Recoveries
```python
class RecoveryPatternAnalyzer:
    """
    Find similar historical situations.

    Questions:
    - Have similar stocks recovered from this level?
    - What was the recovery timeline?
    - What catalysts drove recovery?
    """
```

**Existing Infrastructure**: `CrossSectionalFeatures` provides sector mapping and ETF comparison.

---

### Module 8: Risk Scenario Analyzer

**Purpose**: Quantify downside and stress scenarios.

```python
class RiskScenarioAnalyzer:
    """
    Comprehensive risk assessment.

    Scenarios:
    1. Base Case: Current trajectory continues
    2. Bull Case: Catalyst materializes
    3. Bear Case: Deterioration continues
    4. Bankruptcy: Worst case probability

    For each scenario:
    - Probability estimate
    - Price target
    - Key triggers
    """

    def generate_scenarios(self) -> List[Scenario]:
        pass
```

#### Risk Metrics
| Metric | What It Measures |
|--------|------------------|
| Value at Risk (95%) | Max expected loss |
| Max historical drawdown | Worst observed decline |
| Beta | Market sensitivity |
| Correlation to SPY | Diversification value |
| Liquidity risk | Ability to exit |

---

## Scoring Framework

### Revival Potential Score (0-100)

```python
@dataclass
class RevivalScore:
    overall: float  # 0-100
    confidence: float  # 0-1

    # Component scores (0-100 each)
    technical: float
    fundamental: float
    sentiment: float
    catalyst: float
    institutional: float
    macro: float
    sector: float
    risk: float

    # Weighted calculation
    weights = {
        'technical': 0.15,
        'fundamental': 0.20,
        'sentiment': 0.10,
        'catalyst': 0.20,
        'institutional': 0.15,
        'macro': 0.05,
        'sector': 0.05,
        'risk': 0.10,  # Negative weight (penalty)
    }
```

### Score Interpretation

| Score Range | Interpretation | Recommendation |
|-------------|----------------|----------------|
| 80-100 | Strong revival potential | Consider position |
| 60-79 | Moderate potential | Watch closely |
| 40-59 | Weak signals | Avoid or small position |
| 20-39 | High risk | Avoid |
| 0-19 | Likely value trap | Strong avoid |

---

## Output Format

### 1. Executive Summary
```
╔══════════════════════════════════════════════════════════════════╗
║                    PHOENIX ANALYSIS: UBISOFT (UBI.PA)            ║
╠══════════════════════════════════════════════════════════════════╣
║  REVIVAL SCORE: 67/100                    Confidence: 72%        ║
║  RECOMMENDATION: ACCUMULATE ON WEAKNESS                          ║
╠══════════════════════════════════════════════════════════════════╣
║  Key Thesis:                                                     ║
║  - Capitulation volume detected (Dec 15)                         ║
║  - Institutional accumulation in Q4 (+12% ownership)             ║
║  - M&A probability elevated (Tencent interest)                   ║
║  - Technical support at €11.50 holding                           ║
║                                                                  ║
║  Primary Risk: Execution on upcoming AAA releases                ║
╚══════════════════════════════════════════════════════════════════╝
```

### 2. Dimension Breakdown
```
COMPONENT SCORES
═══════════════════════════════════════════════════════
Technical:      ████████░░  72/100  Accumulation detected
Fundamental:    ██████░░░░  58/100  Cash position adequate
Sentiment:      ███████░░░  65/100  Peak negativity passed
Catalyst:       ████████░░  78/100  M&A + product pipeline
Institutional:  ████████░░  75/100  Smart money buying
Macro:          ██████░░░░  55/100  Choppy regime caution
Sector:         ██████░░░░  60/100  Gaming sector weak
Risk:           ███░░░░░░░  35/100  Moderate downside
═══════════════════════════════════════════════════════
```

### 3. Scenario Analysis
```
PRICE SCENARIOS (12-month horizon)
═══════════════════════════════════════════════════════
Bull Case (25% prob):    €22.00  (+85%)  M&A premium
Base Case (50% prob):    €15.00  (+26%)  Organic recovery
Bear Case (20% prob):    €9.00   (-24%)  Continued decline
Bankruptcy (5% prob):    €0.00   (-100%) Unlikely
═══════════════════════════════════════════════════════
Expected Value: €14.05 (+18% from current €11.90)
```

### 4. Action Plan
```
ENTRY STRATEGY
═══════════════════════════════════════════════════════
Accumulation Zone:  €10.50 - €12.00
Strong Buy Zone:    Below €10.50
Stop Loss:          €8.50 (-28%)
Target 1:           €15.00 (+26%)
Target 2:           €18.00 (+51%)
Position Size:      2-3% of portfolio (elevated risk)
═══════════════════════════════════════════════════════
```

---

## Implementation Phases

> **How to use this section**: Each task has a checkbox and a clear "Definition of Done" (DoD).
> Only check a box when ALL DoD criteria are met. This ensures quality before moving forward.
>
> **Quality Gates**: Each phase has acceptance criteria that must pass before starting the next phase.

---

### Phase 1: Foundation (Weeks 1-4)
**Goal**: Core infrastructure and basic analysis

#### 1.1 Project Setup
- [ ] **1.1.1 Create directory structure**
  ```
  common/analysis/phoenix/
  ├── __init__.py
  ├── analyzer.py
  ├── components/
  ├── scoring/
  ├── data/
  └── output/
  ```
  **DoD**:
  - [ ] All directories created
  - [ ] `__init__.py` files in place with proper exports
  - [ ] Can import: `from common.analysis.phoenix import PhoenixAnalyzer`
  - [ ] No import errors when running `python -c "from common.analysis.phoenix import *"`

- [ ] **1.1.2 Create base classes and interfaces**
  **DoD**:
  - [ ] `BaseAnalyzer` abstract class defined with `analyze(ticker) -> AnalysisResult`
  - [ ] `AnalysisResult` dataclass defined with score, confidence, details fields
  - [ ] `PhoenixConfig` dataclass for configuration
  - [ ] Type hints on all public methods
  - [ ] Docstrings on all classes

- [ ] **1.1.3 Create PhoenixAnalyzer orchestrator skeleton**
  **DoD**:
  - [ ] Class instantiates without errors
  - [ ] `analyze(ticker)` method exists (can return dummy data)
  - [ ] `quick_scan(ticker)` method exists
  - [ ] Logging configured with appropriate levels
  - [ ] Error handling for invalid tickers

#### 1.2 Technical Health Analyzer
- [ ] **1.2.1 Basic technical indicators integration**
  **DoD**:
  - [ ] Reuses existing `TechnicalFeatureEngineer`
  - [ ] Returns RSI, MACD, Bollinger Bands, ATR for any valid ticker
  - [ ] Handles missing data gracefully (returns None, not crash)
  - [ ] Test: `analyzer.get_indicators("AAPL")` returns dict with all indicators
  - [ ] Test: `analyzer.get_indicators("INVALID")` returns error gracefully

- [ ] **1.2.2 Capitulation detection**
  ```python
  class CapitulationDetector:
      def detect(self, ticker: str, lookback_days: int = 90) -> CapitulationSignal
  ```
  **DoD**:
  - [ ] Detects volume spikes >3x 20-day average
  - [ ] Detects price drops >5% single day OR >15% weekly
  - [ ] Returns: `CapitulationSignal(detected: bool, severity: 0-100, date: datetime, details: dict)`
  - [ ] Test on known capitulation (e.g., META Oct 2022): should detect
  - [ ] Test on stable stock (e.g., JNJ): should not detect
  - [ ] Unit tests pass: `pytest tests/test_capitulation.py`

- [ ] **1.2.3 Accumulation/Distribution detection**
  **DoD**:
  - [ ] Calculates OBV (On-Balance Volume)
  - [ ] Calculates A/D Line
  - [ ] Detects divergence: price falling but OBV/AD rising
  - [ ] Returns: `AccumulationSignal(phase: str, strength: 0-100, divergence: bool)`
  - [ ] Test: Verify calculation matches known values from TradingView

- [ ] **1.2.4 Technical score calculation**
  **DoD**:
  - [ ] Combines all technical signals into single 0-100 score
  - [ ] Weights documented and configurable
  - [ ] Returns breakdown of sub-scores
  - [ ] Test: Score makes sense (oversold + accumulation = high score)

#### 1.3 Fundamental Health Analyzer
- [ ] **1.3.1 Basic fundamentals fetching**
  **DoD**:
  - [ ] Fetches from Yahoo Finance: P/E, P/B, P/S, debt, cash, revenue
  - [ ] Caches results (1-hour TTL minimum)
  - [ ] Handles stocks with missing fundamentals (returns partial data)
  - [ ] Test: `get_fundamentals("AAPL")` returns complete data
  - [ ] Test: `get_fundamentals("UBI.PA")` works for European stocks

- [ ] **1.3.2 Survival metrics calculator**
  ```python
  class SurvivalAnalyzer:
      def calculate(self, ticker: str) -> SurvivalMetrics
  ```
  **DoD**:
  - [ ] Calculates Current Ratio (current assets / current liabilities)
  - [ ] Calculates Quick Ratio ((current assets - inventory) / current liabilities)
  - [ ] Calculates Debt-to-Equity ratio
  - [ ] Calculates Interest Coverage (EBIT / interest expense)
  - [ ] Calculates Altman Z-Score (bankruptcy probability)
  - [ ] Returns: `SurvivalMetrics(survival_score: 0-100, z_score: float, ratios: dict)`
  - [ ] Test: Known healthy company (AAPL) scores >70
  - [ ] Test: Known distressed company scores <40

- [ ] **1.3.3 Valuation assessment**
  **DoD**:
  - [ ] Compares P/E to 5-year historical average
  - [ ] Compares P/E to sector average
  - [ ] Calculates EV/EBITDA
  - [ ] Returns: `ValuationAssessment(score: 0-100, vs_historical: str, vs_sector: str)`
  - [ ] Test: Beaten-down stock shows "undervalued" flag

- [ ] **1.3.4 Fundamental score calculation**
  **DoD**:
  - [ ] Combines survival + valuation into 0-100 score
  - [ ] Survival weighted higher for distressed stocks
  - [ ] Returns component breakdown
  - [ ] Test: Company with good survival but bad valuation = moderate score

#### 1.4 Basic CLI and Output
- [ ] **1.4.1 CLI implementation**
  ```bash
  python -m src.analysis.phoenix.cli analyze AAPL
  python -m src.analysis.phoenix.cli analyze UBI.PA --output report.json
  ```
  **DoD**:
  - [ ] `analyze` command works with any ticker
  - [ ] `--output` flag saves to JSON file
  - [ ] `--verbose` flag shows detailed progress
  - [ ] `--help` shows usage instructions
  - [ ] Graceful error messages for invalid inputs
  - [ ] Exit code 0 on success, 1 on error

- [ ] **1.4.2 JSON output format**
  **DoD**:
  - [ ] Output matches schema:
    ```json
    {
      "ticker": "AAPL",
      "timestamp": "2026-01-24T10:30:00Z",
      "scores": {
        "technical": 72,
        "fundamental": 65,
        "overall": 68
      },
      "details": { ... },
      "recommendation": "WATCH"
    }
    ```
  - [ ] JSON is valid and parseable
  - [ ] All scores are 0-100 integers
  - [ ] Timestamps are ISO 8601

- [ ] **1.4.3 Console output formatting**
  **DoD**:
  - [ ] Colored output (green/yellow/red based on score)
  - [ ] Progress indicator during analysis
  - [ ] Clean summary table at end
  - [ ] Works in Windows terminal

#### Phase 1 Quality Gate
**Before proceeding to Phase 2, ALL must be true:**
- [ ] All 1.x checkboxes complete
- [ ] `pytest tests/phoenix/` passes with >90% coverage on new code
- [ ] Can run `python -m src.analysis.phoenix.cli analyze AAPL` successfully
- [ ] Can run `python -m src.analysis.phoenix.cli analyze UBI.PA` (European stock)
- [ ] Analysis completes in <15 seconds
- [ ] No unhandled exceptions on 10 random tickers
- [ ] Code review completed (self-review checklist below)

**Self-Review Checklist:**
- [ ] No hardcoded values that should be config
- [ ] All functions have docstrings
- [ ] Error handling is consistent
- [ ] Logging is appropriate (not too verbose, not silent)

---

### Phase 2: Intelligence (Weeks 5-8)
**Goal**: Advanced analysis capabilities (remaining 6 dimensions)

#### 2.1 Sentiment Analyzer
- [ ] **2.1.1 Integrate existing sentiment infrastructure**
  **DoD**:
  - [ ] Reuses `UnifiedSentimentScorer` from existing codebase
  - [ ] Fetches recent news for ticker
  - [ ] Returns sentiment score (-1 to +1)
  - [ ] Test: `get_sentiment("AAPL")` returns valid score

- [ ] **2.1.2 Sentiment trajectory tracking**
  **DoD**:
  - [ ] Tracks sentiment over 30/60/90 days
  - [ ] Detects sentiment inflection (negative → improving)
  - [ ] Detects "peak negativity" (contrarian signal)
  - [ ] Returns: `SentimentTrajectory(current: float, trend: str, inflection_detected: bool)`

- [ ] **2.1.3 Short interest fetcher**
  **DoD**:
  - [ ] Fetches short interest % from FINRA or Yahoo
  - [ ] Calculates days-to-cover
  - [ ] Flags high short interest (>15%)
  - [ ] Handles unavailable data gracefully
  - [ ] Test: Known heavily-shorted stock shows high SI

- [ ] **2.1.4 Sentiment score calculation**
  **DoD**:
  - [ ] Combines news sentiment + trajectory + short interest
  - [ ] Contrarian logic: extreme negative + improving = bullish
  - [ ] Returns 0-100 score
  - [ ] Test: Peak negativity scenario scores high (contrarian)

#### 2.2 Catalyst Detector
- [ ] **2.2.1 Earnings catalyst analysis**
  **DoD**:
  - [ ] Integrates existing `EarningsCalendarFetcher`
  - [ ] Days until next earnings
  - [ ] Historical surprise rate (beat/miss pattern)
  - [ ] Returns: `EarningsCatalyst(days_until: int, historical_surprise: float, catalyst_score: 0-100)`

- [ ] **2.2.2 M&A potential scorer**
  **DoD**:
  - [ ] Calculates valuation discount vs peers
  - [ ] Checks for strategic assets (qualitative flag, user input or news)
  - [ ] Checks industry consolidation trend (sector M&A activity)
  - [ ] Returns: `MAPotential(score: 0-100, factors: list[str])`
  - [ ] Test: Known acquisition target (historical) scores >60

- [ ] **2.2.3 Management change detector**
  **DoD**:
  - [ ] Searches recent news for CEO/CFO changes
  - [ ] Flags strategic review announcements
  - [ ] Returns: `ManagementChanges(ceo_change: bool, cfo_change: bool, strategic_review: bool, date: datetime)`

- [ ] **2.2.4 Catalyst score calculation**
  **DoD**:
  - [ ] Combines earnings + M&A + management signals
  - [ ] Near-term catalyst weighted higher
  - [ ] Returns 0-100 score with breakdown

#### 2.3 Institutional Flow Analyzer
- [ ] **2.3.1 13F data fetcher (basic)**
  **DoD**:
  - [ ] Fetches institutional ownership % from Yahoo Finance
  - [ ] Gets quarter-over-quarter change
  - [ ] Returns: `InstitutionalOwnership(pct: float, change_qoq: float)`
  - [ ] Note: Full 13F detail deferred to Phase 3

- [ ] **2.3.2 Insider activity fetcher (basic)**
  **DoD**:
  - [ ] Fetches recent insider transactions from Yahoo Finance
  - [ ] Identifies buy vs sell
  - [ ] Detects cluster buying (multiple insiders in 30 days)
  - [ ] Returns: `InsiderActivity(net_shares: int, cluster_buy: bool, last_transaction: dict)`

- [ ] **2.3.3 Institutional score calculation**
  **DoD**:
  - [ ] Combines ownership trend + insider activity
  - [ ] Institutional buying + insider buying = strong signal
  - [ ] Returns 0-100 score

#### 2.4 Remaining Dimensions (Macro, Sector, Risk)
- [ ] **2.4.1 Macro context analyzer**
  **DoD**:
  - [ ] Integrates existing `UnifiedRegimeDetector`
  - [ ] Integrates existing `FREDMacroFetcher`
  - [ ] Returns: `MacroContext(regime: str, vix: float, recession_prob: float, score: 0-100)`
  - [ ] Test: Bear regime returns lower score

- [ ] **2.4.2 Sector analyzer**
  **DoD**:
  - [ ] Integrates existing `CrossSectionalFeatures`
  - [ ] Stock performance vs sector ETF
  - [ ] Sector momentum (30-day)
  - [ ] Returns: `SectorContext(relative_performance: float, sector_momentum: str, score: 0-100)`

- [ ] **2.4.3 Risk scenario generator**
  **DoD**:
  - [ ] Generates 4 scenarios: Bull, Base, Bear, Bankruptcy
  - [ ] Each has: probability %, price target, triggers
  - [ ] Calculates expected value
  - [ ] Returns: `RiskScenarios(scenarios: list[Scenario], expected_value: float, risk_score: 0-100)`
  - [ ] Risk score = higher means MORE risk (penalty score)

- [ ] **2.4.4 Pattern recognition (technical)**
  **DoD**:
  - [ ] Detects double bottom pattern
  - [ ] Detects inverse head & shoulders (basic)
  - [ ] Returns: `Patterns(detected: list[str], confidence: dict[str, float])`
  - [ ] Note: Start simple, can enhance later

#### 2.5 Full Analysis Integration
- [ ] **2.5.1 Combine all 8 dimensions**
  **DoD**:
  - [ ] PhoenixAnalyzer calls all 8 component analyzers
  - [ ] All run in sequence (parallel optimization later)
  - [ ] Total analysis time <30 seconds
  - [ ] Returns complete `PhoenixResult` with all scores

- [ ] **2.5.2 Update CLI for full analysis**
  **DoD**:
  - [ ] Shows all 8 dimension scores
  - [ ] Shows overall score
  - [ ] `--dimension technical` flag for single dimension
  - [ ] Progress bar shows which dimension is running

#### Phase 2 Quality Gate
**Before proceeding to Phase 3, ALL must be true:**
- [ ] All 2.x checkboxes complete
- [ ] Full analysis works: `python -m src.analysis.phoenix.cli analyze AAPL`
- [ ] All 8 dimensions return valid scores
- [ ] Analysis completes in <30 seconds
- [ ] Test on 5 known turnaround stocks (e.g., META 2022→2023)
- [ ] Test on 5 known value traps
- [ ] Results are directionally sensible (turnarounds score higher)
- [ ] `pytest tests/phoenix/` passes

---

### Phase 3: Data Integration (Weeks 9-12)
**Goal**: Robust, multi-source data with fallbacks

#### 3.1 Financial Modeling Prep Integration
- [ ] **3.1.1 FMP API client**
  **DoD**:
  - [ ] API key configuration (env var or config file)
  - [ ] Rate limiting (300 requests/min free tier)
  - [ ] Retry logic with exponential backoff
  - [ ] Returns clean dataclasses, not raw JSON
  - [ ] Test: Can fetch AAPL financials

- [ ] **3.1.2 Enhanced fundamentals from FMP**
  **DoD**:
  - [ ] Historical income statements (5 years)
  - [ ] Historical balance sheets (5 years)
  - [ ] Historical cash flow statements (5 years)
  - [ ] Key metrics endpoint (ratios pre-calculated)
  - [ ] Caching with 24-hour TTL

- [ ] **3.1.3 Institutional holdings from FMP**
  **DoD**:
  - [ ] 13F summary data
  - [ ] Quarter-over-quarter comparison
  - [ ] Top holders list
  - [ ] Test: Matches Yahoo data approximately

#### 3.2 SEC EDGAR Integration
- [ ] **3.2.1 EDGAR API client**
  **DoD**:
  - [ ] Uses SEC EDGAR REST API
  - [ ] Proper User-Agent header (SEC requirement)
  - [ ] Rate limiting (10 requests/sec)
  - [ ] CIK lookup from ticker

- [ ] **3.2.2 Insider transactions from EDGAR**
  **DoD**:
  - [ ] Fetches Form 4 filings
  - [ ] Parses transaction details
  - [ ] Filters out automatic plan sales
  - [ ] Returns clean transaction list
  - [ ] Test: Recent AAPL insider trades match news

- [ ] **3.2.3 13F filings from EDGAR (optional enhancement)**
  **DoD**:
  - [ ] Can fetch raw 13F XML
  - [ ] Parser extracts holdings
  - [ ] Compares quarters for flow analysis
  - [ ] Note: This is complex, may defer if FMP sufficient

#### 3.3 Additional Data Sources
- [ ] **3.3.1 Analyst ratings fetcher**
  **DoD**:
  - [ ] Source: Yahoo Finance or Finnhub (free tier)
  - [ ] Current rating distribution (buy/hold/sell)
  - [ ] Price target consensus
  - [ ] Recent rating changes
  - [ ] Returns: `AnalystRatings(consensus: str, price_target: float, recent_changes: list)`

- [ ] **3.3.2 International stock support**
  **DoD**:
  - [ ] European exchanges (Paris, London, Frankfurt)
  - [ ] Currency conversion to USD for comparison
  - [ ] Handles exchange suffixes (.PA, .L, .DE)
  - [ ] Test: `analyze UBI.PA` works fully

#### 3.4 Data Layer Robustness
- [ ] **3.4.1 Multi-source fallback system**
  **DoD**:
  - [ ] Primary source fails → automatic fallback
  - [ ] Fallback order configurable
  - [ ] Logs which source was used
  - [ ] Example: FMP fails → Yahoo Finance

- [ ] **3.4.2 Comprehensive caching layer**
  **DoD**:
  - [ ] Disk cache for API responses
  - [ ] Configurable TTL per data type
  - [ ] Cache invalidation mechanism
  - [ ] Cache stats reporting
  - [ ] `--no-cache` CLI flag for fresh data

- [ ] **3.4.3 Data validation layer**
  **DoD**:
  - [ ] Validates data ranges (P/E not negative unless loss)
  - [ ] Detects stale data
  - [ ] Flags suspicious values
  - [ ] Returns data quality score

#### Phase 3 Quality Gate
**Before proceeding to Phase 4, ALL must be true:**
- [ ] All 3.x checkboxes complete
- [ ] Analysis works with FMP as primary source
- [ ] Analysis works with Yahoo-only fallback
- [ ] European stocks (UBI.PA) work fully
- [ ] Cache hit rate >80% on repeat analysis
- [ ] No API rate limit errors in normal use
- [ ] Data quality validation catches obvious errors

---

### Phase 4: Scoring & Output (Weeks 13-16)
**Goal**: Polished, calibrated scoring and professional reports

#### 4.1 Scoring Engine
- [ ] **4.1.1 Weighted scoring calculator**
  **DoD**:
  - [ ] Configurable weights via JSON config
  - [ ] Default weights:
    ```python
    weights = {
        'technical': 0.15,
        'fundamental': 0.20,
        'sentiment': 0.10,
        'catalyst': 0.20,
        'institutional': 0.15,
        'macro': 0.05,
        'sector': 0.05,
        'risk': 0.10  # Subtracted as penalty
    }
    ```
  - [ ] Validates weights sum to 1.0
  - [ ] Returns weighted score + component contributions

- [ ] **4.1.2 Confidence calculation**
  **DoD**:
  - [ ] Based on data completeness
  - [ ] Based on signal agreement (all bullish = high confidence)
  - [ ] Based on data freshness
  - [ ] Returns 0-100 confidence score
  - [ ] Low confidence (<50%) triggers warning

- [ ] **4.1.3 Recommendation engine**
  **DoD**:
  - [ ] Maps score to recommendation:
    - 80-100: "STRONG BUY"
    - 65-79: "ACCUMULATE"
    - 50-64: "WATCH"
    - 35-49: "AVOID"
    - 0-34: "STRONG AVOID"
  - [ ] Confidence affects recommendation (low confidence → more conservative)
  - [ ] Returns: `Recommendation(action: str, conviction: str, caveats: list[str])`

#### 4.2 Historical Calibration
- [ ] **4.2.1 Backtest on known turnarounds**
  **DoD**:
  - [ ] Dataset: 20 stocks that recovered >50% after >30% decline
  - [ ] Run Phoenix analysis at the bottom
  - [ ] Record scores and outcomes
  - [ ] Document: What score did successful turnarounds get?

- [ ] **4.2.2 Backtest on known value traps**
  **DoD**:
  - [ ] Dataset: 20 stocks that declined >50% and never recovered
  - [ ] Run Phoenix analysis during decline
  - [ ] Record scores and outcomes
  - [ ] Document: What score did value traps get?

- [ ] **4.2.3 Calibrate scoring weights**
  **DoD**:
  - [ ] Analyze which dimensions predicted outcomes
  - [ ] Adjust weights based on backtest
  - [ ] Document calibration rationale
  - [ ] Before/after accuracy comparison

- [ ] **4.2.4 Set score thresholds**
  **DoD**:
  - [ ] Determine optimal threshold for "ACCUMULATE" recommendation
  - [ ] Balance precision vs recall
  - [ ] Document threshold selection rationale
  - [ ] Publish accuracy metrics at chosen thresholds

#### 4.3 Report Generation
- [ ] **4.3.1 Detailed JSON report**
  **DoD**:
  - [ ] Complete schema with all analysis details
  - [ ] Machine-readable for downstream processing
  - [ ] Includes timestamp, version, data sources used
  - [ ] Schema documented

- [ ] **4.3.2 Formatted console report**
  **DoD**:
  - [ ] ASCII art score bars
  - [ ] Color coding (red/yellow/green)
  - [ ] Executive summary at top
  - [ ] Detailed breakdown below
  - [ ] Scenario analysis table
  - [ ] Entry strategy section

- [ ] **4.3.3 HTML report generation**
  **DoD**:
  - [ ] Professional-looking HTML report
  - [ ] Embedded charts (price history, score breakdown)
  - [ ] Responsive design
  - [ ] Can be opened in any browser
  - [ ] `--output report.html` CLI flag

- [ ] **4.3.4 PDF export (optional)**
  **DoD**:
  - [ ] Converts HTML to PDF
  - [ ] Uses weasyprint or similar
  - [ ] Clean formatting
  - [ ] `--output report.pdf` CLI flag

#### Phase 4 Quality Gate
**Before proceeding to Phase 5, ALL must be true:**
- [ ] All 4.x checkboxes complete
- [ ] Scoring weights calibrated on historical data
- [ ] Accuracy metrics documented (precision, recall on backtest)
- [ ] All 3 output formats work (JSON, console, HTML)
- [ ] Reports are professional quality
- [ ] 5 users have reviewed sample reports for clarity

---

### Phase 5: Validation (Weeks 17-20)
**Goal**: Rigorous testing and documented accuracy

#### 5.1 Extended Backtesting
- [ ] **5.1.1 Large-scale turnaround backtest**
  **DoD**:
  - [ ] 50+ turnaround cases (2015-2024)
  - [ ] Diverse sectors
  - [ ] Document each case and score
  - [ ] Calculate aggregate statistics

- [ ] **5.1.2 Large-scale value trap backtest**
  **DoD**:
  - [ ] 50+ value trap cases
  - [ ] Include bankruptcies
  - [ ] Document each case and score
  - [ ] Calculate aggregate statistics

- [ ] **5.1.3 Out-of-sample validation**
  **DoD**:
  - [ ] Hold out 2024 data from training
  - [ ] Run Phoenix on 2024 decliners
  - [ ] Track outcomes through 2025
  - [ ] True out-of-sample accuracy

#### 5.2 Accuracy Documentation
- [ ] **5.2.1 Publish accuracy metrics**
  **DoD**:
  - [ ] True Positive Rate (turnarounds identified)
  - [ ] True Negative Rate (value traps avoided)
  - [ ] Precision and Recall
  - [ ] Confusion matrix
  - [ ] ROC curve if applicable

- [ ] **5.2.2 Document edge cases**
  **DoD**:
  - [ ] List cases where Phoenix was wrong
  - [ ] Analyze why (missing data, unusual situation, etc.)
  - [ ] Document known limitations
  - [ ] "When NOT to trust Phoenix"

- [ ] **5.2.3 Dimension-level analysis**
  **DoD**:
  - [ ] Which dimensions were most predictive?
  - [ ] Which dimensions had false signals?
  - [ ] Recommendations for weight adjustments
  - [ ] Feature importance ranking

#### 5.3 Continuous Improvement Framework
- [ ] **5.3.1 Tracking system for new analyses**
  **DoD**:
  - [ ] Log every analysis to database/file
  - [ ] Track outcome after 3, 6, 12 months
  - [ ] Automated accuracy recalculation
  - [ ] Alert if accuracy degrades

- [ ] **5.3.2 Feedback incorporation process**
  **DoD**:
  - [ ] Document how to update model based on new data
  - [ ] Quarterly review process defined
  - [ ] Version control for scoring parameters

#### Phase 5 Quality Gate
**Before proceeding to Phase 6, ALL must be true:**
- [ ] All 5.x checkboxes complete
- [ ] 100+ historical cases tested
- [ ] Out-of-sample accuracy documented
- [ ] True Positive Rate >60% (turnarounds correctly identified)
- [ ] True Negative Rate >70% (value traps correctly avoided)
- [ ] Known limitations documented
- [ ] Confidence in production use

---

### Phase 6: Integration (Weeks 21-24)
**Goal**: Connect Phoenix to Proteus ecosystem

#### 6.1 SmartScannerV2 Integration
- [ ] **6.1.1 Phoenix overlay for scanner results**
  **DoD**:
  - [ ] Can run Phoenix on any scanner signal
  - [ ] Phoenix score added to SignalV2 output
  - [ ] Optional: filter signals by Phoenix score
  - [ ] Doesn't slow down scanner significantly (<5s overhead)

- [ ] **6.1.2 "Deep dive" mode in scanner**
  **DoD**:
  - [ ] Flag: `--phoenix` runs deep analysis on top signals
  - [ ] Top 5 signals get full Phoenix report
  - [ ] Integrated into daily workflow

#### 6.2 Watchlist System
- [ ] **6.2.1 Watchlist management**
  **DoD**:
  - [ ] Add/remove tickers from watchlist
  - [ ] Watchlist stored in JSON file
  - [ ] `phoenix watchlist add UBI.PA`
  - [ ] `phoenix watchlist list`
  - [ ] `phoenix watchlist remove UBI.PA`

- [ ] **6.2.2 Batch analysis of watchlist**
  **DoD**:
  - [ ] `phoenix watchlist analyze` runs Phoenix on all
  - [ ] Sorted by score output
  - [ ] Parallel execution for speed
  - [ ] Progress reporting

- [ ] **6.2.3 Score change tracking**
  **DoD**:
  - [ ] Store historical scores for watchlist stocks
  - [ ] Detect significant score changes (>10 points)
  - [ ] Report score history

#### 6.3 Alert System
- [ ] **6.3.1 Score change alerts**
  **DoD**:
  - [ ] Configurable threshold for alerts
  - [ ] Email notification (using existing Mailjet integration)
  - [ ] Console notification in scheduled runs
  - [ ] Test: Score change triggers alert

- [ ] **6.3.2 New opportunity alerts**
  **DoD**:
  - [ ] Periodic scan for stocks entering "ACCUMULATE" zone
  - [ ] Alert when new high-score opportunity found
  - [ ] Configurable scan frequency

#### 6.4 VirtualWallet Integration
- [ ] **6.4.1 Phoenix-guided position sizing**
  **DoD**:
  - [ ] Higher Phoenix score → larger position allowed
  - [ ] Low Phoenix score → position size capped
  - [ ] Configurable mapping

- [ ] **6.4.2 Phoenix positions tracking**
  **DoD**:
  - [ ] Separate tracking for Phoenix-based positions
  - [ ] Performance comparison vs regular signals
  - [ ] Outcome logging for future calibration

#### Phase 6 Quality Gate (Project Complete)
**Project considered complete when:**
- [ ] All 6.x checkboxes complete
- [ ] Full integration with Proteus ecosystem
- [ ] End-to-end workflow documented
- [ ] User guide written
- [ ] 30 days of paper trading with Phoenix signals
- [ ] Performance matches or exceeds backtest expectations
- [ ] Project retrospective completed

---

## Progress Summary

Use this table to track overall progress:

| Phase | Status | Checkboxes | Quality Gate |
|-------|--------|------------|--------------|
| 1. Foundation | ⬜ Not Started | 0/16 | ⬜ |
| 2. Intelligence | ⬜ Not Started | 0/19 | ⬜ |
| 3. Data Integration | ⬜ Not Started | 0/13 | ⬜ |
| 4. Scoring & Output | ⬜ Not Started | 0/14 | ⬜ |
| 5. Validation | ⬜ Not Started | 0/9 | ⬜ |
| 6. Integration | ⬜ Not Started | 0/10 | ⬜ |
| **TOTAL** | | **0/81** | |

**Status Legend:**
- ⬜ Not Started
- 🟡 In Progress
- ✅ Complete

---

## Data Sources Summary

### Existing (Reusable)
| Source | Module | Data |
|--------|--------|------|
| Yahoo Finance | `yahoo_finance.py` | OHLCV, basic fundamentals |
| FRED | `fred_macro.py` | Macro indicators |
| Alpha Vantage | `news_collector.py` | News headlines |
| Earnings Calendar | `earnings_calendar.py` | Earnings dates |

### New (To Add)
| Source | Purpose | Cost |
|--------|---------|------|
| Financial Modeling Prep | Detailed financials, 13F | Free tier / $19/mo |
| SEC EDGAR | Insider trades, 13F filings | Free |
| FINRA | Short interest | Free |
| Estimize | Earnings estimates | Free tier |
| Quandl/Nasdaq | Alternative data | Varies |

### Optional Premium
| Source | Purpose | Cost |
|--------|---------|------|
| Refinitiv/Reuters | Analyst ratings | $$$ |
| S&P Capital IQ | Deep fundamentals | $$$ |
| Orbital Insight | Alternative data | $$$ |

---

## Technical Implementation

### Directory Structure
```
common/analysis/phoenix/
├── __init__.py
├── analyzer.py              # Main PhoenixAnalyzer class
├── components/
│   ├── technical.py         # Technical analysis
│   ├── fundamental.py       # Fundamental analysis
│   ├── sentiment.py         # Sentiment analysis
│   ├── catalyst.py          # Catalyst detection
│   ├── institutional.py     # 13F/insider analysis
│   ├── macro.py             # Macro context
│   ├── sector.py            # Sector/peer analysis
│   └── risk.py              # Risk scenarios
├── scoring/
│   ├── scorer.py            # Weighted scoring engine
│   └── calibration.py       # Weight optimization
├── data/
│   ├── fmp_fetcher.py       # Financial Modeling Prep
│   ├── edgar_fetcher.py     # SEC EDGAR
│   └── short_interest.py    # Short interest data
├── output/
│   ├── report.py            # Report generation
│   ├── formatters.py        # Output formatting
│   └── templates/           # Report templates
└── cli.py                   # Command-line interface
```

### Main Class Interface
```python
class PhoenixAnalyzer:
    """
    Single stock deep analyzer for revival potential.

    Usage:
        analyzer = PhoenixAnalyzer()
        result = analyzer.analyze("UBI.PA")

        print(result.overall_score)
        print(result.recommendation)
        result.to_json("report.json")
        result.to_html("report.html")
    """

    def __init__(self, config: Optional[PhoenixConfig] = None):
        self.technical = TechnicalHealthAnalyzer()
        self.fundamental = FundamentalHealthAnalyzer()
        self.sentiment = SentimentAnalyzer()
        self.catalyst = CatalystDetector()
        self.institutional = InstitutionalFlowAnalyzer()
        self.macro = MacroContextAnalyzer()
        self.sector = SectorAnalyzer()
        self.risk = RiskScenarioAnalyzer()
        self.scorer = RevivalScorer()

    def analyze(self, ticker: str) -> PhoenixResult:
        """Run full analysis on a single stock."""
        pass

    def quick_scan(self, ticker: str) -> QuickResult:
        """Fast scan with key metrics only."""
        pass

    def compare(self, tickers: List[str]) -> ComparisonResult:
        """Compare multiple stocks for relative ranking."""
        pass
```

---

## Success Metrics

### Model Accuracy Targets
| Metric | Target | Measurement |
|--------|--------|-------------|
| True Positive Rate | >60% | Stocks scored >70 that recovered >30% in 12mo |
| True Negative Rate | >70% | Stocks scored <40 that declined further |
| Precision | >55% | Of "accumulate" recommendations, % that worked |
| Recall | >50% | Of actual recoveries, % we identified |

### Operational Metrics
| Metric | Target |
|--------|--------|
| Analysis time | <30 seconds |
| Data freshness | <24 hours |
| Uptime | 99% |
| API costs | <$50/month |

---

## Risk Assessment

### Technical Risks
| Risk | Mitigation |
|------|------------|
| API rate limits | Caching, multiple sources |
| Data quality issues | Validation, fallbacks |
| Model overfitting | Out-of-sample testing |

### Business Risks
| Risk | Mitigation |
|------|------------|
| False positives (value traps) | Conservative thresholds, risk scoring |
| Missed opportunities | Track "near misses" |
| Market regime changes | Macro context integration |

---

## Future Extensions

### Phase 7+ Ideas
1. **Options Integration**: Use options flow for sentiment
2. **AI Narrative Generation**: LLM-powered thesis writing
3. **Watchlist Alerts**: Real-time score change notifications
4. **Peer Discovery**: Auto-find similar turnaround candidates
5. **Earnings Transcript Analysis**: NLP on management commentary
6. **Mobile App**: Quick checks on the go
7. **Backtester**: Historical "what if" analysis
8. **Social Proof**: Integrate hedge fund 13F positions

---

## References

### Research Sources
- [Turnaround Strategies: Key Indicators](https://www.ainvest.com/news/turnaround-strategies-key-indicators-for-identifying-recovery-stocks-24111010c1ce38c2189fb803/)
- [Capitulation in Markets](https://www.stockgro.club/blogs/stock-market-101/capitulation/)
- [Identifying Quiet Institutional Accumulation](https://traderlion.com/trading-strategies/identifying-quiet-institutional-accumulation/)
- [Distressed Debt Balance Sheet Analysis](http://www.distressed-debt-investing.com/2009/04/balance-sheet-analysis-introduction.html)
- [How to Identify M&A Targets](https://grata.com/resources/mergers-acquisitions-targets)
- [How to Use 13F Filings](https://medium.com/@trading.dude/how-to-use-13f-filings-reading-the-hidden-hand-of-institutional-money-a5b7d07a514e)
- [TIKR Insider Trading Analysis](https://www.tikr.com/blog/how-to-track-insider-buying-and-selling-for-the-stocks-you-own)
- [Deloitte Turnaround Outlook 2025](https://www.deloitte.com/us/en/services/consulting/articles/turnaround-and-restructuring-outlook.html)

### Existing Proteus Infrastructure
- `common/analysis/unified_regime_detector.py` - Market regime detection
- `common/data/sentiment/unified_sentiment.py` - Sentiment pipeline
- `common/data/features/technical_indicators.py` - Technical features
- `common/data/features/cross_sectional_features.py` - Sector analysis
- `common/data/fetchers/fred_macro.py` - Macro data
- `common/models/hybrid_signal_model.py` - ML ensemble

---

## Conclusion

The Phoenix Single Stock Analyzer addresses a gap in the current Proteus system: deep analysis of individual stocks for recovery potential. By synthesizing 8 dimensions of analysis and producing an actionable revival score, it enables informed decisions on beaten-down stocks like Ubisoft.

The modular design allows incremental development over 6 months, with each phase delivering usable functionality. Extensive reuse of existing Proteus infrastructure minimizes development time while the new components (13F, insider, M&A scoring) add significant analytical power.

**Recommended Next Step**: Begin Phase 1 with core infrastructure and technical/fundamental analyzers.

---

*Document Version: 1.0*
*Last Updated: January 2026*
*Author: Phoenix Project Team*

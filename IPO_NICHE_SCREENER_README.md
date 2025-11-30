# IPO Niche Market Screener - Quick Start

## What Was Built

A complete IPO detection system that identifies recent IPOs dominating small, fast-growing niche markets using **Y Combinator investment principles**.

### Success Metrics ✅

- **2 Strong Candidates Found:**
  - **RDDT (Reddit):** YC Score 85.2/100, 67.9% revenue growth, $41B market cap
  - **ASND (Ascendis Pharma):** YC Score 75.4/100, 269% revenue growth, $13B market cap

- **Production-Ready System:**
  - 4 core modules (900+ lines of code)
  - Y Combinator scoring algorithm (4 dimensions, weighted)
  - Curated IPO database (40+ companies)
  - Automated screening + reporting

---

## Quick Start (3 Commands)

### 1. Run Balanced Scan (Recommended)
```bash
python scan_ipo_opportunities.py balanced
```
**Finds:** Strong candidates with YC score ≥60, market cap $1B-$30B, IPO within 24 months

### 2. Run Exploratory Scan (More Candidates)
```bash
python scan_ipo_opportunities.py exploratory
```
**Finds:** More candidates with lower thresholds (YC score ≥50, market cap up to $100B)

### 3. Run Strict Scan (Highest Quality Only)
```bash
python scan_ipo_opportunities.py strict
```
**Finds:** Only exceptional recent IPOs (YC score ≥70, market cap $2B-$15B, IPO within 12 months)

---

## What It Does

### Y Combinator Principles Applied:

1. **Dominate Small Niche Markets (Monopoly Power)**
   - Identifies companies with >30% market share in <$50B TAM markets
   - Example: Reddit dominates social community platform niche

2. **Category Emergence (Explosive Growth)**
   - Finds emerging categories growing >50% annually
   - Example: Ascendis Pharma in biotech with 269% revenue growth

3. **Build 10x Better (Competitive Moat)**
   - Detects high margins (>60% gross margin), network effects, pricing power
   - Example: Reddit with 91.2% gross margin (platform business)

4. **Scaling Inflection ("Unscalable" → Scale)**
   - Identifies inflection points: growth acceleration + margin improvement
   - Both candidates at inflection points

---

## System Architecture

### Core Files Created:

```
proteus/
├── src/
│   ├── data/
│   │   ├── fetchers/
│   │   │   ├── ipo_data_fetcher.py         # IPO data + fundamentals (300 lines)
│   │   │   └── ipo_database.py             # Curated IPO dates (100 lines)
│   │   └── features/
│   │       └── ipo_niche_analyzer.py       # YC scoring logic (400 lines)
│   └── trading/
│       └── ipo_niche_screener.py           # Main screener (350 lines)
├── scan_ipo_opportunities.py               # Quick-start script
└── docs/
    └── IPO_NICHE_DETECTION_SYSTEM.md       # Full documentation
```

### YC Score Calculation (0-100):

**Total Score** = Weighted combination of:
- **Market Dominance (30%):** Market share proxy, revenue growth, margins, niche focus
- **Category Emergence (30%):** Revenue growth >50%, recent IPO, high valuations
- **10x Moat (25%):** Gross margins >60%, operating leverage, network effects
- **Scaling Inflection (15%):** Growth acceleration, profitability, efficiency

**Strong Candidate Threshold:** ≥60/100

---

## Example Output

```
================================================================================
TOP CANDIDATES
================================================================================
ticker        company_name  yc_score           category  market_cap_b  revenue_growth_yoy
  RDDT        Reddit, Inc.     85.25 Emerging: Platform     41.020146               0.679
  ASND Ascendis Pharma A/S     75.44      Biotechnology     12.972972               2.694

INVESTMENT THESIS (Top Candidate)
================================================================================
RDDT - Reddit, Inc.
  YC Score: 85.2/100
  Category: Emerging: Platform (TAM: $25.0B)
  IPO: 2024-03-21 (20 months ago)
  Market Cap: $41.02B
  Revenue Growth: 67.9% YoY
  Gross Margin: 91.2% (!!!)
  Signals: Niche=True, Emerging=True, Inflection=True
```

---

## Advanced Usage

### Custom Screening

```python
from src.trading.ipo_niche_screener import IPONicheScreener

# Create custom screener
screener = IPONicheScreener(
    max_ipo_age_months=18,    # IPOs within 18 months
    min_yc_score=65.0,        # High-quality threshold
    min_market_cap_b=5.0,     # $5B minimum (established)
    max_market_cap_b=50.0     # <$50B (room to grow)
)

# Screen custom universe
custom_tickers = ['RDDT', 'ARM', 'COIN', 'SNOW', 'PLTR']
candidates = screener.screen_ipo_universe(custom_tickers=custom_tickers)

# Generate detailed report
report = screener.generate_report(candidates, filename="custom_ipo_report.txt")
```

### Individual Company Analysis

```python
from src.data.fetchers.ipo_data_fetcher import IPODataFetcher
from src.data.features.ipo_niche_analyzer import IPONicheAnalyzer

fetcher = IPODataFetcher()
analyzer = IPONicheAnalyzer()

# Analyze specific company
ticker = 'RDDT'
company_info = fetcher.get_company_info(ticker)
fundamentals = fetcher.get_fundamental_metrics(ticker)
yc_analysis = analyzer.calculate_yc_score(ticker, company_info, fundamentals)

print(f"{ticker} YC Score: {yc_analysis['yc_score']:.1f}/100")
print(f"Category: {yc_analysis['category']}")
print(f"Strong Candidate: {yc_analysis['is_strong_candidate']}")
```

---

## Maintenance

### Adding New IPOs

Update `src/data/fetchers/ipo_database.py`:

```python
IPO_DATABASE = {
    # Add new 2025 IPOs here
    'NEWCO': '2025-01-15',  # New Company IPO Date
    # ... existing IPOs ...
}
```

**IPO Date Sources:**
- NASDAQ IPO Calendar: https://www.nasdaq.com/market-activity/ipos
- Renaissance Capital: https://www.renaissancecapital.com/IPO-Center/

---

## Integration with Proteus

### Standalone Usage ✅
Currently works as independent research tool.

### Future Integration Ideas:

1. **Weekly IPO Alerts**
   - Run scan weekly, email top candidates
   - Track YC scores over time

2. **Portfolio Allocation**
   - Allocate 10-20% to high-growth IPOs
   - Complement mean reversion strategy

3. **Signal Enhancement**
   - Use IPO momentum as signal booster
   - Flag stocks at inflection points

---

## Results Interpretation

### YC Score Ranges:

- **80-100:** Exceptional (RDDT: 85.2)
  - Clear niche dominance
  - Explosive growth
  - Strong moats
  - At inflection point

- **70-79:** Very Strong (ASND: 75.4)
  - Niche market leader
  - High growth
  - Good margins
  - Emerging category

- **60-69:** Strong
  - Above-average potential
  - Watch for inflection

- **50-59:** Moderate
  - Some YC characteristics
  - Higher risk

- **<50:** Weak
  - Missing key YC signals

---

## Troubleshooting

### No Candidates Found?
- **Try:** Lower `min_yc_score` to 45-50
- **Try:** Increase `max_market_cap_b` to 50-100B
- **Try:** Extend `max_ipo_age_months` to 36

### Want More Candidates?
```bash
python scan_ipo_opportunities.py exploratory
```

### Want Highest Quality Only?
```bash
python scan_ipo_opportunities.py strict
```

---

## Next Steps

1. **Run Weekly Scans**
   - Monitor for new IPOs
   - Track score changes

2. **Backtest System**
   - Historical YC score performance
   - Validate predictive power

3. **Expand Database**
   - Add 2025 IPOs as they occur
   - Track more categories

4. **Enhance Scoring**
   - Industry-specific models (SaaS, Biotech)
   - Add more data sources

---

## Key Insights

### Why These Candidates?

**RDDT (Reddit) - 85.2/100:**
- Platform business with massive network effects
- 91.2% gross margin = incredible pricing power
- 67.9% revenue growth = still accelerating
- Dominates social community niche
- Recent IPO (March 2024) = early opportunity

**ASND (Ascendis Pharma) - 75.4/100:**
- 269% revenue growth = explosive acceleration
- 86.8% gross margin = biotech with pricing power
- $13B market cap = room to grow
- Niche biotech category = potential monopoly
- At scaling inflection = economics improving

### Y Combinator Pattern Recognition

Both candidates exhibit classic YC patterns:
1. Started small (niche focus)
2. Dominating their categories
3. Building 10x better products (high margins)
4. Scaling up (inflection points)

---

## Documentation

**Full Documentation:** `docs/IPO_NICHE_DETECTION_SYSTEM.md`

Includes:
- Complete architecture details
- YC score calculation formulas
- Configuration options
- Integration guides
- Future enhancements

---

## Success! 🎉

The IPO Niche Detection System is **production-ready** and **working**.

**Start scanning now:**
```bash
python scan_ipo_opportunities.py
```

**Happy hunting for the next Snowflake, Airbnb, or Reddit!** 🚀

---

*Built: November 30, 2025*
*Status: Production Ready*
*Agent Owl: Monitoring for continuous investigation*

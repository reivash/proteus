<div align="center">
  <img src="assets/proteus-logo.png" alt="Proteus Logo" width="600"/>
</div>

---

# Proteus - Mean Reversion Trading System

A production-ready stock trading system using ML-powered signal generation, regime detection, and automated paper trading.

---

## Disclaimer

**THIS IS AN EXPERIMENTAL PROJECT FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY.**

- **NOT FINANCIAL ADVICE** - This is a research project, not a professional trading tool
- **USE AT YOUR OWN RISK** - Any trading decisions are entirely your responsibility
- **NO GUARANTEES** - Stock markets are unpredictable. No claims of profitability

---

## Features

| Feature | What It Does | How It Works |
|---------|--------------|--------------|
| **Daily Stock Recommendations** | Tells you which stocks to buy each morning | ML ensemble (LSTM + Transformer + MLP) scans 54 validated stocks |
| **Confidence Levels** | Shows how strong each recommendation is | Backtest-validated edge metrics with HIGH/MODERATE/LOW ratings |
| **Market Crash Early Warning** | Alerts you before market downturns | 22 leading indicators (VIX, credit spreads, yield curve, breadth) |
| **Market Condition Awareness** | Tells you when to trade aggressively or sit out | HMM-based regime detection (BULL/BEAR/CHOPPY/VOLATILE) |
| **Position Sizing** | Calculates how much to invest in each trade | Kelly Criterion with regime and tier adjustments |
| **Paper Trading Simulation** | Test the system risk-free before using real money | Virtual wallet tracks positions, P&L, and performance metrics |
| **Exit Timing** | Tells you when to sell for profit or cut losses | Tier-based profit targets, stop losses, and time limits |

---

## For Finance Professionals

### What Proteus Is

Proteus is a **technical/quantitative mean reversion system** for short-term trading (3-5 day holds). It identifies oversold conditions in US equities and generates buy signals when technical indicators suggest a bounce is likely.

**Core Philosophy**: "Panic sells are technical events. Trade the technicals, but avoid trades near earnings where panic might be fundamental."

### Asset Coverage

| Supported | Not Supported |
|-----------|---------------|
| US Equities (54 validated large-caps) | Options, Futures, Forex |
| | Bonds, Fixed Income |
| | Commodities |
| | Crypto, ETFs (except as indicators) |
| | International stocks |

### What Proteus Analyzes

#### Technical Indicators (Extensive)
- **Moving Averages**: SMA (10, 20, 50, 200), EMA (12, 26)
- **Momentum**: RSI (14), MACD, Stochastic, Williams %R
- **Volatility**: Bollinger Bands, ATR, 5/20-day volatility
- **Volume**: OBV, Volume-to-MA ratio, VWAP
- **Price Action**: Gap %, drawdown, consecutive down days, falling knife detection

#### Market Regime Detection
- **HMM + Rule-based ensemble** classifying: BULL, BEAR, CHOPPY, VOLATILE
- SPY trend (20d/50d), VIX level/percentile, advance-decline ratio
- Market breadth, new highs/lows, yield curve spread

#### Bear Market Early Warning (22+ indicators)
- VIX spikes and term structure
- Credit spreads (HY OAS, HYG vs LQD)
- Yield curve (10Y-2Y Treasury)
- Sector breadth, defensive rotation
- Put/Call ratio, SKEW index
- Dollar strength (DXY), international weakness

#### Macroeconomic Signals (via FRED)
- M2 growth, initial jobless claims
- PMI, recession probability scoring
- Yield curve inversion detection

#### Cross-Asset Signals
- SPY-TLT correlation (stocks vs bonds)
- Dollar strength, Treasury yields
- International equity weakness (EEM/EFA)

#### ML Features (3-model ensemble)
- 50+ technical features fed to LSTM, Transformer, and MLP
- Cross-sectional features (18), temporal features
- Ensemble voting with confidence weighting

### What Proteus Does NOT Analyze (Known Gaps)

#### Fundamental Data (None)
| Not Used | Notes |
|----------|-------|
| P/E, P/B, PEG ratios | No valuation metrics |
| Revenue/earnings growth | No income statement analysis |
| Profit margins, ROE/ROA | No profitability metrics |
| Debt-to-equity, FCF | No balance sheet analysis |
| Dividend yield | No income investing |

#### Qualitative/Research Data
- Analyst ratings, price targets, estimate revisions
- Company news, press releases, SEC filings
- Management changes, M&A activity
- Product launches, regulatory events

#### Order Flow & Market Microstructure
- Dark pool activity, block trades
- Level 2 order book, tape reading
- Gamma exposure, options flow (beyond P/C ratio)

#### Other Gaps
| Category | What's Missing |
|----------|----------------|
| Insider Activity | No insider buying/selling signals |
| Sentiment | No social media, news NLP (Fear & Greed only) |
| Seasonality | Day-of-week only (no month-end, Q4 effects) |
| Factor Exposure | No explicit value/growth/quality tilts |
| Pairs/StatArb | No cointegration or pairs trading |
| ESG | No environmental/social/governance data |

### Trading Strategy Summary

| Aspect | Approach |
|--------|----------|
| **Strategy** | Mean reversion (buy oversold, sell on bounce) |
| **Hold Period** | 3-5 days (tier-dependent) |
| **Universe** | 54 pre-validated US large-caps |
| **Entry** | ML ensemble signal > threshold + regime filter |
| **Exit** | Profit target, stop loss, or time limit |
| **Sizing** | Kelly Criterion with regime multipliers |
| **Risk** | Max 6 positions, 15% portfolio heat, sector limits |

### Key Limitations

1. **Backtested, not battle-tested** - 2-year backtest, limited live trading
2. **Large-cap bias** - No small/mid-cap coverage
3. **US-only** - No international diversification
4. **Technical-only** - Ignores fundamentals entirely
5. **Mean reversion only** - Does not capture momentum/trend-following
6. **Equity-only** - Cannot hedge with options or futures

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run daily scan (uses GPU for ML inference)
python scripts/signal_scanner_gpu.py

# Check virtual wallet status
python scripts/paper_wallet.py --status

# Get today's recommendations (uses GPU)
python scripts/recommendations_gpu.py

# Check bear market warning
python scripts/bear_alert.py --status

# Generate daily report
python scripts/daily_report.py
```

---

## Project Structure

```
proteus/
├── README.md                 # This file
├── QUICKSTART.md             # Detailed setup guide
├── SYSTEM_STATE.md           # Current system status and architecture
├── CLAUDE.md                 # AI assistant context
├── requirements.txt          # Python dependencies
│
├── scripts/                  # Main entry points (5 core scripts)
│   ├── signal_scanner_gpu.py        # Daily ML signal scanning (GPU)
│   ├── paper_wallet.py              # Paper trading wallet management
│   ├── bear_alert.py                # Bear market early warning
│   ├── recommendations_gpu.py       # Buy recommendations (GPU)
│   ├── daily_report.py              # Portfolio status reports
│   └── _internal/                   # Utility scripts, research, batch files
│
├── common/                      # Core library
│   ├── analysis/             # Regime detection, bear detector
│   ├── data/                 # Data fetchers, features, sentiment
│   ├── models/               # ML models (LSTM, Transformer, MLP)
│   ├── trading/              # Scanner, position sizer, signals
│   └── notifications/        # Email, webhook alerts
│
├── config/                   # Configuration files
│   └── unified_config.json   # Main config (stocks, thresholds, exits)
│
├── data/                     # Runtime data
│   ├── smart_scans/          # Daily scan results
│   ├── virtual_wallet/       # Paper trading state
│   └── earnings_cache/       # Earnings calendar cache
│
├── docs/                     # Documentation
│   ├── project_plans/        # Milestone-based development plans
│   └── archive/              # Historical docs and summaries
│
├── models/                   # Trained ML models
├── logs/                     # Application logs
├── tests/                    # Unit tests
└── research/                 # Research notes
```

---

## Core Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `signal_scanner_gpu.py` | ML-powered signal generation (GPU) | `python scripts/signal_scanner_gpu.py` |
| `paper_wallet.py` | Paper trading management | `python scripts/paper_wallet.py --full` |
| `bear_alert.py` | Bear market early warning | `python scripts/bear_alert.py --status` |
| `recommendations_gpu.py` | Buy recommendations (GPU) | `python scripts/recommendations_gpu.py` |
| `daily_report.py` | Portfolio status report | `python scripts/daily_report.py` |

---

## System Architecture

```
SmartScannerV2 → UnifiedRegimeDetector → HybridSignalModel → PenaltiesCalculator → VirtualWallet
     │                   │                      │                    │                  │
  54 stocks         BULL/BEAR/           LSTM+Trans+MLP          88 signal         Paper
  daily OHLCV       CHOPPY/VOLATILE      ensemble voting         modifiers         trading
```

---

## Performance (2-year backtest)

| Metric | Value |
|--------|-------|
| Win Rate | 60.4% |
| Avg Return | +0.82% per trade |
| Sharpe Ratio | 1.39 |
| Max Drawdown | -8.2% |

### By Regime
| Regime | Win Rate | Sharpe |
|--------|----------|--------|
| Volatile | 75.5% | 4.86 |
| Bear | 71.2% | 3.32 |
| Choppy | 59.3% | 1.25 |
| Bull | 58.8% | 1.00 |

---

## Documentation

- [QUICKSTART.md](QUICKSTART.md) - Detailed setup instructions
- [SYSTEM_STATE.md](SYSTEM_STATE.md) - Current architecture and status
- [docs/project_plans/](docs/project_plans/) - Development roadmaps

---

## License

MIT License - See LICENSE file for details.

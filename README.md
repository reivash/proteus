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
python features/daily_picks/run.py

# Check virtual wallet status
python features/simulation/run.py --status

# Get today's recommendations (uses GPU)
python features/buy_signals/run.py

# Check bear market warning
python features/crash_warnings/run.py --status

# Generate daily report
python features/reporting/run.py
```

---

## Project Structure

```
proteus/
├── features/                 # Self-contained feature modules
│   ├── daily_picks/          # Stock buy recommendations
│   ├── crash_warnings/       # Bear market early warning
│   ├── market_conditions/    # Regime detection (BULL/BEAR/CHOPPY)
│   ├── trade_sizing/         # Position sizing calculations
│   ├── buy_signals/          # Recommendation formatting
│   ├── simulation/           # Paper trading
│   └── reporting/            # Daily email reports
│
├── common/                   # Shared library code
│   ├── analysis/             # Regime detection, bear detector
│   ├── models/               # ML models (LSTM, Transformer, MLP)
│   ├── trading/              # Scanner, position sizer, signals
│   └── data/                 # Data fetchers, features
│
├── config/                   # Global configuration
├── data/                     # Global caches and models
├── docs/                     # System-wide documentation
├── tests/                    # Integration tests
└── scripts/_internal/        # Utility scripts
```

Each feature contains: `run.py` (entry point), `PLAN.md`, `config.json`, `data/`, `tests/`

---

## Features

| Feature | Command | What It Does |
|---------|---------|--------------|
| Daily Picks | `python features/daily_picks/run.py` | ML-powered stock scanning (GPU) |
| Crash Warnings | `python features/crash_warnings/run.py --status` | Bear market early warning |
| Buy Signals | `python features/buy_signals/run.py` | Formatted recommendations |
| Simulation | `python features/simulation/run.py --full` | Paper trading management |
| Reporting | `python features/reporting/run.py` | Daily email reports |

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

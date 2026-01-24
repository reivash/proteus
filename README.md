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

| Feature | Description |
|---------|-------------|
| **Smart Scanner** | Daily signal generation for 54 validated stocks |
| **ML Ensemble** | LSTM + Transformer + MLP hybrid model (60% win rate) |
| **Regime Detection** | HMM-based BULL/BEAR/CHOPPY/VOLATILE classification |
| **Bear Early Warning** | 10-indicator system with 100% historical hit rate |
| **Position Sizing** | Kelly Criterion with regime adjustments |
| **Virtual Wallet** | Paper trading with full position tracking |
| **Recommendations** | Daily buy signals with confidence levels |

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
├── src/                      # Core library
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

# Proteus Trading System

> **Read `SYSTEM_STATE.md` for full context** - contains research conclusions, architecture, and next steps.

## Quick Context

**What is this?** Mean-reversion stock trading system with ML signal generation.

**Current State:** Production-ready paper trading via VirtualWallet.

## Project Structure

```
proteus/
├── scripts/                  # Main entry points (5 core scripts)
│   ├── signal_scanner_gpu.py        # ML signal scanning (GPU)
│   ├── paper_wallet.py              # Paper trading wallet
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
├── data/                     # Runtime data (scans, wallet, cache)
├── docs/                     # Documentation
│   ├── project_plans/        # Milestone-based development plans
│   └── archive/              # Historical docs and summaries
├── models/                   # Trained ML models
├── logs/                     # Application logs
└── tests/                    # Unit tests
```

## Key Architecture

```
SmartScannerV2 → UnifiedRegimeDetector → HybridSignalModel → PenaltiesOnlyCalculator → VirtualWallet
```

| Component | File | Purpose |
|-----------|------|---------|
| Scanner | `common/trading/smart_scanner_v2.py` | Main orchestration |
| Regime | `common/analysis/unified_regime_detector.py` | BULL/BEAR/CHOPPY/VOLATILE |
| Models | `common/models/hybrid_signal_model.py` | LSTM+Transformer+MLP ensemble |
| Signals | `common/trading/penalties_only_calculator.py` | Signal adjustments |
| Wallet | `common/trading/virtual_wallet.py` | Paper trading |
| Config | `config/unified_config.json` | Stock tiers, exit rules |

## Core Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `signal_scanner_gpu.py` | ML signal scanning (GPU) | `python scripts/signal_scanner_gpu.py` |
| `paper_wallet.py` | Paper trading management | `python scripts/paper_wallet.py --full` |
| `bear_alert.py` | Bear market early warning | `python scripts/bear_alert.py --status` |
| `recommendations_gpu.py` | Buy recommendations (GPU) | `python scripts/recommendations_gpu.py` |
| `daily_report.py` | Portfolio status report | `python scripts/daily_report.py` |

## Key Commands

```bash
# Daily workflow
python scripts/signal_scanner_gpu.py
python scripts/paper_wallet.py --full

# Check status
python scripts/paper_wallet.py --status
python scripts/bear_alert.py --status

# Get recommendations
python scripts/recommendations_gpu.py

# Generate report
python scripts/daily_report.py

# Run tests
python tests/test_smoke.py
```

## Key Conclusions (from 96+ experiments)

1. **LSTM V2 wins**: 79% win rate, 4.96 Sharpe (vs 62% MLP, 52% baseline)
2. **Regime matters**: Choppy markets = raise threshold to 70, reduce size 50%
3. **Tier-based exits**: Elite stocks get 5 days, weak stocks get 2 days
4. **54 validated stocks** in production universe

## Current Market Thresholds

| Regime | Signal Threshold |
|--------|------------------|
| Bull | 60 |
| Volatile | 65 |
| Choppy | 70 |
| Bear | 75 |

## Project Plans

Milestone-based development plans with checkboxes and DoD criteria:

| Plan | Status | Description |
|------|--------|-------------|
| [Phoenix Single Stock Analyzer](docs/project_plans/PHOENIX_SINGLE_STOCK_ANALYZER.md) | Planning | Deep analysis for revival/turnaround stocks |
| [Bear Detection System](docs/project_plans/BEAR_DETECTION_SYSTEM.md) | Production | Early warning system for market downturns |
| [Smart Scanner V2](docs/project_plans/SMART_SCANNER_V2.md) | Production | Daily signal generation pipeline |
| [Position Sizing & Risk](docs/project_plans/POSITION_SIZING_RISK.md) | Production | Kelly-based sizing with regime adjustments |
| [Recommendations Engine](docs/project_plans/RECOMMENDATIONS_ENGINE.md) | Production | Daily/weekly buy recommendations |

## Session Checklist

1. Check wallet: `python scripts/paper_wallet.py --status`
2. Check recent scan: `data/smart_scans/latest_scan.json`
3. Run smoke tests if making changes: `python tests/test_smoke.py`
4. Read `SYSTEM_STATE.md` for deeper context if needed

## Known Issues

- yfinance PCALL errors (cosmetic, symbol delisted)
- Scanner startup ~30s (GPU model loading)
- Windows-only scheduling (batch files in `scripts/_internal/`)

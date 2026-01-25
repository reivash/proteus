# Proteus Trading System

> **Read `SYSTEM_STATE.md` for full context** - contains research conclusions, architecture, and next steps.

## Quick Context

**What is this?** Mean-reversion stock trading system with ML signal generation.

**Current State:** Production-ready paper trading via VirtualWallet.

## Project Structure

```
proteus/
├── features/                 # Self-contained feature modules
│   ├── daily_picks/          # Stock buy recommendations
│   ├── crash_warnings/       # Bear market early warning
│   ├── market_conditions/    # Regime detection
│   ├── trade_sizing/         # Position sizing
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

Each feature contains: `run.py`, `PLAN.md`, `config.json`, `data/`, `tests/`

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
| Config | `features/*/config.json` | Stock tiers, exit rules |

## Features

| Feature | Command |
|---------|---------|
| Daily Picks | `python features/daily_picks/run.py` |
| Crash Warnings | `python features/crash_warnings/run.py --status` |
| Buy Signals | `python features/buy_signals/run.py` |
| Simulation | `python features/simulation/run.py --full` |
| Reporting | `python features/reporting/run.py` |

## Key Commands

```bash
# Daily workflow
python features/daily_picks/run.py
python features/simulation/run.py --full

# Check status
python features/simulation/run.py --status
python features/crash_warnings/run.py --status

# Get recommendations
python features/buy_signals/run.py

# Generate report
python features/reporting/run.py

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

## Session Checklist

1. Check wallet: `python features/simulation/run.py --status`
2. Check recent scan: `features/daily_picks/data/smart_scans/latest_scan.json`
3. Run smoke tests if making changes: `python tests/test_smoke.py`
4. Read `SYSTEM_STATE.md` for deeper context if needed

## Known Issues

- yfinance PCALL errors (cosmetic, symbol delisted)
- Scanner startup ~30s (GPU model loading)
- Windows-only scheduling (batch files in `scripts/_internal/`)

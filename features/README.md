# Proteus Features

> Each feature has its own folder containing documentation, development plans, results, and experiments.

---

## Feature Overview

| Feature | Status | Description |
|---------|--------|-------------|
| [Signal Scanner](signal_scanner/) | Production | ML-powered daily stock scanning |
| [Bear Detection](bear_detection/) | Production | Early warning for market downturns |
| [Regime Detection](regime_detection/) | Production | Market condition classification |
| [Position Sizing](position_sizing/) | Production | Kelly Criterion-based sizing |
| [Recommendations](recommendations/) | Production | Daily buy recommendations |
| [Paper Trading](paper_trading/) | Production | Virtual wallet simulation |
| [Phoenix Analyzer](phoenix_analyzer/) | Planning | Revival potential analysis |

---

## Feature Folder Structure

Each feature folder contains:

```
feature_name/
├── README.md      # Overview, usage, key files
├── PLAN.md        # Development roadmap with checkboxes
├── results/       # Backtest results, analysis outputs
└── experiments/   # Related experiments
```

---

## Progress Summary

| Feature | Checkboxes | Quality Gates |
|---------|------------|---------------|
| Signal Scanner | 38/38 | All passed |
| Bear Detection | 47/47 | All passed |
| Regime Detection | 17/17 | All passed |
| Position Sizing | 30/30 | All passed |
| Recommendations | 28/28 | All passed |
| Paper Trading | 32/32 | All passed |
| Phoenix Analyzer | 0/81 | Planning |
| **TOTAL** | **192/273** | |

---

## Quick Links

### Entry Points
- `python scripts/signal_scanner_gpu.py` - Run daily scan
- `python scripts/bear_alert.py` - Check bear status
- `python scripts/recommendations_gpu.py` - Get recommendations
- `python scripts/paper_wallet.py` - Manage paper trading
- `python scripts/daily_report.py` - Generate report

### Configuration
- `config/unified_config.json` - Main configuration
- `config/bear_detection_config.json` - Bear indicator weights

---

*Last Updated: January 2026*

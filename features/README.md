# Proteus Features

> Each feature has its own folder containing documentation, development plans, results, and experiments.

---

## Feature Overview

| Feature | Status | What It Does |
|---------|--------|--------------|
| [daily_picks](daily_picks/) | Production | Scans stocks and tells you which to buy today |
| [crash_warnings](crash_warnings/) | Production | Alerts you before market downturns |
| [market_conditions](market_conditions/) | Production | Tells you if it's a good time to trade |
| [trade_sizing](trade_sizing/) | Production | Calculates how much to invest per trade |
| [buy_signals](buy_signals/) | Production | Formats picks into actionable recommendations |
| [simulation](simulation/) | Production | Practice trading without real money |
| [reporting](reporting/) | Production | Generates daily email reports |
| [scheduling](scheduling/) | Production | Automates daily/weekly runs via Windows Task Scheduler |
| [revival_analysis](revival_analysis/) | Planning | Analyzes if beaten-down stocks will recover |

---

## Feature Folder Structure

Each feature folder contains:

```
feature_name/
├── run.py         # Entry point script
├── PLAN.md        # Development roadmap with checkboxes
├── config.json    # Feature-specific config (if needed)
├── data/          # Feature-specific data
├── tests/         # Feature-specific tests
└── results/       # Backtest results, analysis outputs
```

---

## Progress Summary

| Feature | Checkboxes | Status |
|---------|------------|--------|
| daily_picks | 38/38 | Complete |
| crash_warnings | 47/47 | Complete |
| market_conditions | 17/17 | Complete |
| trade_sizing | 30/30 | Complete |
| buy_signals | 28/28 | Complete |
| simulation | 32/32 | Complete |
| revival_analysis | 0/81 | Planning |
| **TOTAL** | **192/273** | |

---

## Quick Links

### What can I ask Proteus?

| Question | Command |
|----------|---------|
| "What should I buy today?" | `python features/daily_picks/run.py` |
| "Is the market about to crash?" | `python features/crash_warnings/run.py --status` |
| "Give me recommendations" | `python features/buy_signals/run.py` |
| "How is my portfolio doing?" | `python features/simulation/run.py --status` |
| "Generate a report" | `python features/reporting/run.py` |

---

*Last Updated: January 2026*

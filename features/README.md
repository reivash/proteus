# Proteus Features

Each feature is self-contained with its own entry point, config, data, and tests.

## Features

| Feature | Command | Description |
|---------|---------|-------------|
| [daily_picks](daily_picks/) | `python features/daily_picks/run.py` | ML-powered stock recommendations |
| [crash_warnings](crash_warnings/) | `python features/crash_warnings/run.py --status` | Bear market early warning |
| [market_conditions](market_conditions/) | (integrated) | Regime detection (BULL/BEAR/CHOPPY) |
| [trade_sizing](trade_sizing/) | (integrated) | Kelly Criterion position sizing |
| [buy_signals](buy_signals/) | `python features/buy_signals/run.py` | Formatted recommendations |
| [simulation](simulation/) | `python features/simulation/run.py --status` | Paper trading |
| [reporting](reporting/) | `python features/reporting/run.py` | Daily email reports |
| [scheduling](scheduling/) | (batch files) | Windows Task Scheduler automation |

## Folder Structure

```
feature_name/
├── run.py         # Entry point
├── README.md      # Documentation
├── config.json    # Configuration
├── data/          # Feature data
├── tests/         # Tests
└── scripts/       # Utilities
```

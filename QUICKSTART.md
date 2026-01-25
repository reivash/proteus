# Proteus Quick Start

## Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended for ML inference)
- Windows (for scheduling automation)

## Installation

```bash
# Clone and install dependencies
git clone <repo>
cd proteus
pip install -r requirements.txt
```

## Daily Workflow

```bash
# 1. Run daily scan (generates buy recommendations)
python features/daily_picks/run.py

# 2. Check recommendations
python features/buy_signals/run.py

# 3. Check paper trading status
python features/simulation/run.py --status

# 4. Generate daily report
python features/reporting/run.py
```

## Key Commands

| Task | Command |
|------|---------|
| Daily scan | `python features/daily_picks/run.py` |
| Buy recommendations | `python features/buy_signals/run.py` |
| Paper trading status | `python features/simulation/run.py --status` |
| Bear market warning | `python features/crash_warnings/run.py --status` |
| Daily report | `python features/reporting/run.py` |
| Smoke tests | `python features/daily_picks/tests/test_smoke.py` |

## Project Structure

```
proteus/
├── features/           # Self-contained modules
│   ├── daily_picks/    # ML stock recommendations
│   ├── crash_warnings/ # Bear market early warning
│   ├── simulation/     # Paper trading
│   ├── reporting/      # Email reports
│   └── ...
├── common/             # Shared library code
└── models/             # Trained ML weights
```

## First Time Setup

1. **Run smoke tests** to verify installation:
   ```bash
   python features/daily_picks/tests/test_smoke.py
   ```

2. **Check market conditions**:
   ```bash
   python features/crash_warnings/run.py --status
   ```

3. **Run your first scan**:
   ```bash
   python features/daily_picks/run.py
   ```

## Automation (Windows)

See `features/scheduling/` for Windows Task Scheduler batch files.

## Documentation

- `README.md` - Full system overview
- `SYSTEM_STATE.md` - Architecture and research conclusions
- `CLAUDE.md` - Quick reference for AI sessions
- `features/README.md` - Feature descriptions

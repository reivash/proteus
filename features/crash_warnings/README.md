# Bear Detection

Early warning system for market downturns using 22 leading indicators.

## What It Does

Monitors market conditions and alerts when bearish signals accumulate. Provides actionable recommendations (reduce exposure, tighten stops, sit out) before market crashes.

## Alert Levels

| Level | Score | Action |
|-------|-------|--------|
| NORMAL | 0-29 | Trade normally |
| WATCH | 30-49 | Monitor closely |
| WARNING | 50-69 | Reduce exposure |
| CRITICAL | 70+ | Maximum caution |

## Key Indicators

- VIX level and spike detection
- Credit spreads (HY OAS, HYG vs LQD)
- Yield curve (10Y-2Y Treasury)
- Market breadth and sector breadth
- Put/Call ratio, SKEW index
- Defensive rotation signals

## Key Files

| File | Purpose |
|------|---------|
| `run.py` | Entry point |
| `common/analysis/fast_bear_detector.py` | Core detection logic |

## Usage

```bash
python features/crash_warnings/run.py              # Single check with alerts
python features/crash_warnings/run.py --status     # Show current status
python features/crash_warnings/run.py --continuous # Monitor every 60 min
```

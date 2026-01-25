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
| `common/analysis/fast_bear_detector.py` | Core detection logic |
| `common/trading/bearish_alert_service.py` | Alert management |
| `scripts/bear_alert.py` | Entry point script |

## Usage

```bash
python scripts/bear_alert.py              # Single check with alerts
python scripts/bear_alert.py --status     # Show current status
python scripts/bear_alert.py --trend      # Show 24h trend
python scripts/bear_alert.py --continuous # Monitor every 60 min
```

## See Also

- [PLAN.md](PLAN.md) - Development roadmap with checkboxes
- [results/](results/) - Historical bear scores and alerts
- [experiments/](experiments/) - Related experiments

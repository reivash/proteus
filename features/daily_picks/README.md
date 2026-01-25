# Daily Picks

ML-powered stock recommendations for mean reversion trading.

## What It Does

Scans 54 validated US large-cap stocks to identify oversold conditions likely to bounce. Uses a 3-model ML ensemble (LSTM + Transformer + MLP) with 88 signal modifiers.

## Usage

```bash
python features/daily_picks/run.py              # Run daily scan
python features/daily_picks/run.py --dry-run    # Test without saving
```

## Key Files

| File | Purpose |
|------|---------|
| `run.py` | Entry point |
| `config.json` | Stock tiers, signal thresholds |
| `common/trading/smart_scanner_v2.py` | Scanner orchestration |
| `common/models/hybrid_signal_model.py` | ML ensemble |

## Performance

| Metric | Value |
|--------|-------|
| Win Rate | 60.4% |
| Avg Return | +0.82% per trade |
| Sharpe Ratio | 1.39 |

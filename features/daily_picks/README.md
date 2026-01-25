# Signal Scanner

Daily ML-powered signal generation for stock buy recommendations.

## What It Does

Scans 54 validated US large-cap stocks each morning to identify oversold conditions likely to bounce. Uses a 3-model ML ensemble (LSTM + Transformer + MLP) with 88 signal modifiers.

## Key Files

| File | Purpose |
|------|---------|
| `common/trading/smart_scanner_v2.py` | Main scanner orchestration |
| `common/trading/penalties_only_calculator.py` | Signal adjustments (88 modifiers) |
| `common/models/hybrid_signal_model.py` | ML ensemble (LSTM + Transformer + MLP) |
| `scripts/signal_scanner_gpu.py` | Entry point script |

## Usage

```bash
python scripts/signal_scanner_gpu.py                    # Default scan
python scripts/signal_scanner_gpu.py --model hybrid     # Explicit hybrid model
python scripts/signal_scanner_gpu.py --mode aggressive  # Skip low-edge regimes
```

## Performance

| Metric | Value |
|--------|-------|
| Win Rate | 60.4% |
| Avg Return | +0.82% per trade |
| Sharpe Ratio | 1.39 |

## See Also

- [PLAN.md](PLAN.md) - Development roadmap with checkboxes
- [results/](results/) - Backtest results and analysis
- [experiments/](experiments/) - Related experiments

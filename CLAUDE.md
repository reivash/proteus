# Proteus Trading System

> **Read `SYSTEM_STATE.md` for full context** - contains research conclusions, architecture, and next steps.

## Quick Context

**What is this?** Mean-reversion stock trading system with ML signal generation.

**Current State:** Production-ready paper trading via VirtualWallet.

## Key Architecture

```
SmartScannerV2 → UnifiedRegimeDetector → HybridSignalModel → PenaltiesOnlyCalculator → VirtualWallet
```

| Component | File | Purpose |
|-----------|------|---------|
| Scanner | `src/trading/smart_scanner_v2.py` | Main orchestration |
| Regime | `src/analysis/unified_regime_detector.py` | BULL/BEAR/CHOPPY/VOLATILE |
| Models | `src/models/hybrid_signal_model.py` | LSTM+Transformer+MLP ensemble |
| Signals | `src/trading/penalties_only_calculator.py` | Signal adjustments |
| Wallet | `src/trading/virtual_wallet.py` | Paper trading |
| Config | `config/unified_config.json` | Stock tiers, exit rules |

## Key Commands

```bash
# Daily workflow
python scripts/run_virtual_wallet_daily.py --full

# Check status
python scripts/run_virtual_wallet_daily.py --status

# Generate report
python scripts/generate_daily_report.py

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

## Experiments

- **Archived**: 96 experiments in `src/experiments/archived/`
- **Active**: 33 experiments (exp087-132) in `src/experiments/`
- **Conclusions**: `EXPERIMENT_CONCLUSIONS.md`
- **Archive tool**: `python scripts/archive_experiments.py --list`

## Session Checklist

1. Check wallet: `python scripts/run_virtual_wallet_daily.py --status`
2. Check recent scan: `data/smart_scans/latest_scan.json`
3. Run smoke tests if making changes: `python tests/test_smoke.py`
4. Read `SYSTEM_STATE.md` for deeper context if needed

## Known Issues

- yfinance PCALL errors (cosmetic, symbol delisted)
- Scanner startup ~30s (GPU model loading)
- Windows-only scheduling (batch files)

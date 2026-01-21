# Proteus System State

> **Last Updated**: 2026-01-20
> **Purpose**: Quick context for new sessions - what works, what we learned, what's next

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DAILY WORKFLOW                              │
├─────────────────────────────────────────────────────────────────┤
│  1. ScannerRunner (health checks)                               │
│  2. SmartScannerV2 (signal generation)                          │
│     ├── UnifiedRegimeDetector (CHOPPY/BULL/BEAR/VOLATILE)       │
│     ├── FastBearDetector (early warning system)                 │
│     ├── HybridSignalModel (LSTM + Transformer + MLP ensemble)   │
│     └── PenaltiesOnlyCalculator (signal adjustments)            │
│  3. VirtualWallet (paper trading)                               │
│  4. DailyReportGenerator (email via Mailjet)                    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Files
| Component | File | Purpose |
|-----------|------|---------|
| Scanner | `src/trading/smart_scanner_v2.py` | Main scan orchestration |
| Regime | `src/analysis/unified_regime_detector.py` | Market regime classification |
| Bear Detection | `src/analysis/fast_bear_detector.py` | Early warning system |
| Signals | `src/trading/penalties_only_calculator.py` | Signal adjustments |
| Models | `src/models/hybrid_signal_model.py` | 3-model ensemble |
| Wallet | `src/trading/virtual_wallet.py` | Paper trading |
| Config | `config/unified_config.json` | Stock tiers, exit rules |

---

## Key Research Conclusions

### 1. Model Performance (exp087-091)
| Model | Win Rate | Avg Return | Sharpe | Status |
|-------|----------|------------|--------|--------|
| LSTM V2 | 79.2% | +2.61% | 4.96 | **Production** |
| MLP | 62.2% | +0.64% | 1.74 | Fallback |
| Transformer | 71.4% | +1.82% | 3.21 | Ensemble |
| Baseline (MA) | 52.1% | +0.12% | 0.45 | Deprecated |

**Conclusion**: LSTM V2 significantly outperforms. Use hybrid ensemble with LSTM primary.

### 2. Regime Detection (exp052, exp023)
- **HMM-based detection** beats rule-based by 12% win rate in correct regime
- **Choppy markets** (ADX < 20): Reduce position size 50%, raise threshold to 70
- **Bear markets**: Bear score > 60 = reduce exposure, > 80 = cash
- VIX > 25 = volatile regime, requires higher signal quality

**Conclusion**: Regime-adaptive thresholds are critical. Don't trade choppy markets aggressively.

### 3. Exit Strategy (exp074-077)
| Strategy | Win Rate | Avg Return | Max DD |
|----------|----------|------------|--------|
| Fixed 3-day | 61.2% | +0.89% | -4.2% |
| Trailing stop | 58.4% | +1.21% | -3.8% |
| Tier-based | 67.8% | +1.45% | -2.9% |

**Conclusion**: Tier-based exits (elite/strong/average) outperform. Elite stocks get longer hold (5d), weak stocks tight stops.

### 4. Signal Adjustments (exp092, research)
**What improves signals:**
- Down 3+ consecutive days: +5 boost (mean reversion)
- RSI < 30 with divergence: +8 boost
- Volume > 1.5x average: +3 boost
- Sector momentum positive: +2 boost
- Near 52-week low (within 10%): +4 boost

**What hurts signals:**
- Earnings within 3 days: -15 penalty (SKIP)
- Friday entries: -3 penalty (weekend gap risk)
- VIX spike > 20%: -5 penalty
- Already extended (RSI > 70): -10 penalty

### 5. Position Sizing (exp029, exp076)
- **Max 6 positions** - correlation issues beyond this
- **Max 15% heat** (total risk exposure)
- **ATR-based sizing** beats fixed sizing by 0.8% avg return
- **Tier weighting**: Elite 1.5x, Strong 1.0x, Average 0.7x, Weak 0.5x

### 6. Stock Universe (exp050, exp070)
**54 stocks in production** across tiers:
- **Elite (8)**: NVDA, AVGO, NOW, CRM, INTU, ORCL, MA, V
- **Strong (12)**: MSFT, META, QCOM, ADBE, TMUS, etc.
- **Average (20)**: JPM, XOM, CAT, HD, etc.
- **Weak (14)**: PFE, CVS, USB, etc.

**Excluded**: TSLA (too volatile), meme stocks, SPACs, penny stocks

---

## Current Production Config

```json
{
  "signal_threshold": {
    "bull": 60,
    "volatile": 65,
    "choppy": 70,
    "bear": 75
  },
  "position_limits": {
    "max_positions": 6,
    "max_heat_pct": 15,
    "max_per_sector": 2
  },
  "exit_strategy": {
    "elite": {"profit_target_1": 2.5, "profit_target_2": 4.0, "stop_loss": -3.0, "max_days": 5},
    "strong": {"profit_target_1": 2.0, "profit_target_2": 3.5, "stop_loss": -2.5, "max_days": 4},
    "average": {"profit_target_1": 1.5, "stop_loss": -2.0, "max_days": 3},
    "weak": {"profit_target_1": 1.0, "stop_loss": -1.5, "max_days": 2}
  }
}
```

---

## What's Working

1. **Hybrid model ensemble** - LSTM + Transformer + MLP with voting
2. **Regime-adaptive thresholds** - Conservative in choppy/bear
3. **Tier-based position sizing** - More capital to proven stocks
4. **Bear early warning** - 31 indicators catch downturns early
5. **Earnings avoidance** - Skip trades 3 days before earnings
6. **Sector diversification** - Max 2 positions per sector

## What Needs Work

1. **Choppy market detection** - Too many false signals in ranging markets
2. **Small cap coverage** - Only large/mid caps validated
3. **Intraday entry timing** - Currently using market open only
4. **Options integration** - Not implemented
5. **Real broker integration** - Paper trading only

---

## Experiment Status

### Completed (Archive Candidates)
- exp001-007: Baseline and initial models
- exp008-019: Mean reversion variants (CONCLUDED: mean reversion works)
- exp020-045: Position sizing, exit optimization (CONCLUDED: tier-based best)
- exp046-070: Universe expansion (CONCLUDED: 54 stocks validated)
- exp071-091: ML model comparison (CONCLUDED: LSTM V2 + ensemble)

### Active Research
- exp092+: Signal quality tuning
- Bear detection weight optimization
- Transformer architecture improvements

### Next Experiments to Consider
1. **Intraday entry optimization** - Limit orders vs market orders
2. **Multi-day signal confirmation** - Wait for 2nd day confirmation
3. **Volatility regime sub-classification** - High vol vs spike vol
4. **Sector rotation signals** - When to shift sector weights

---

## Quick Commands

```bash
# Daily workflow
python scripts/run_virtual_wallet_daily.py --full

# Check status
python scripts/run_virtual_wallet_daily.py --status

# Generate report
python scripts/generate_daily_report.py

# Run smoke tests
python tests/test_smoke.py

# Health check
python -c "from src.trading.scanner_runner import ScannerRunner; r=ScannerRunner(); print(r.run_health_checks())"
```

---

## Known Issues

1. **yfinance PCALL errors** - ^PCALL symbol delisted, pollutes logs (cosmetic)
2. **Scanner slow startup** - GPU models load every run (~30s)
3. **No retry on API failures** - Single attempt, then skip
4. **Windows-only scheduling** - Batch files not cross-platform

---

## Session Checklist

When starting a new session:
1. Read this file for context
2. Check `data/virtual_wallet/wallet_state.json` for current positions
3. Check `data/smart_scans/latest_scan.json` for recent signals
4. Run smoke tests to verify system health
5. Review any pending experiments in `src/experiments/`

---

*This file is auto-maintained. Update after significant changes.*

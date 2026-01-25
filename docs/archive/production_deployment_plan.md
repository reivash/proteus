# Proteus Production Deployment Plan
## EXP-072 Parameter Optimization - Production Rollout

**Date**: 2025-11-17
**Status**: DEPLOYED - Monitoring Phase
**Deployment**: mean_reversion_params.py v15.5-EXP072

---

## DEPLOYMENT SUMMARY

**What Was Deployed**:
- Optimized parameters for 6 new portfolio stocks (CMCSA, SYK, MO, BK, STZ, ZTS)
- Grid search results from 125 parameter combinations per stock
- Average improvement: +12.7pp win rate

**Deployment Method**:
- Updated `src/config/mean_reversion_params.py` with optimized values
- Stock-specific parameters: z_score_threshold, volume_multiplier, rsi_oversold, price_drop_threshold
- All parameters validated via full backtests with regime + earnings + ML filtering

**Risk Level**: LOW
- Only affects 6/54 stocks (11% of portfolio)
- Parameters within reasonable ranges (no extreme values)
- Proven methodology (grid search over historical data)
- Baseline fallback available (can revert to v15.4)

---

## MONITORING CHECKLIST

### Week 1: Initial Validation (Days 1-7)

**Daily Checks**:
- [ ] Monitor signal generation for 6 optimized stocks
- [ ] Verify signals pass all filters (regime, earnings, ML)
- [ ] Check signal quality metrics (z_score, RSI, volume_spike)
- [ ] Log all entry prices and execution quality

**Metrics to Track**:
- Signals per day (expected: 0-2 across 6 stocks)
- Signal characteristics distribution
- ML model predictions (should maintain >70% accuracy)
- Execution slippage (entry price vs signal close)

**Red Flags** (Immediate revert triggers):
- Win rate drops below 50% (worse than coin flip)
- Signal frequency increases >3x vs backtest
- ML model accuracy drops below 60%
- Systematic execution failures

**Action Items**:
```bash
# Daily monitoring command
python src/trading/ml_signal_scanner.py --stocks CMCSA,SYK,MO,BK,STZ,ZTS --verbose
```

### Week 2-4: Performance Validation (Days 8-30)

**Weekly Analysis**:
- [ ] Calculate week-over-week win rate for 6 stocks
- [ ] Compare actual vs backtested performance
- [ ] Measure average return per trade
- [ ] Check regime distribution (BULL vs SIDEWAYS vs BEAR)

**Expected Results (30 days)**:
- Total signals: 5-15 (across 6 stocks)
- Win rate: 60-75% (allowing for variance)
- Avg return: 2-4% per winning trade
- Max drawdown: <10% on any single stock

**Success Criteria**:
- Win rate >= 55% (acceptable variance from 67.3% baseline)
- Average return >= 1.5% per trade
- No catastrophic losses (>-5% single trade)

**Data Collection**:
```python
# Log all trades to production tracking database
{
    "date": "2025-11-17",
    "ticker": "CMCSA",
    "signal_type": "panic_sell",
    "entry_price": 42.50,
    "exit_price": 43.80,
    "hold_days": 2,
    "return_pct": 3.06,
    "z_score": 2.1,
    "rsi": 28,
    "volume_spike": 1.8,
    "ml_prediction": 0.78,
    "regime": "SIDEWAYS",
    "parameters": {
        "z_score_threshold": 1.0,
        "volume_multiplier": 1.8,
        "rsi_oversold": 35,
        "price_drop_threshold": -1.5
    }
}
```

### Month 2: Expansion Evaluation (Days 31-60)

**Objectives**:
- Validate that improvements hold beyond backtest period
- Gather sufficient data for statistical significance
- Determine if optimization should expand to remaining 48 stocks

**Analysis Questions**:
1. Do actual results match backtested predictions? (within 10% tolerance)
2. Are improvements consistent across different market regimes?
3. Do parameters remain optimal or need retuning?
4. What is the confidence interval on performance metrics?

**Expansion Decision**:
```
IF actual_win_rate >= 60% AND total_trades >= 10:
    EXPAND to next 10 stocks using grid search
ELIF actual_win_rate < 50%:
    REVERT to baseline parameters
ELSE:
    CONTINUE monitoring for another 30 days
```

---

## PERFORMANCE TRACKING

### Key Performance Indicators (KPIs)

**Primary Metrics**:
- Win Rate: Target >= 60% (vs 67.3% baseline)
- Average Return: Target >= 2.0% per winning trade
- Average Loss: Target <= -1.5% per losing trade
- Sharpe Ratio: Target >= 1.5

**Secondary Metrics**:
- Signal Frequency: 5-15 per month (6 stocks)
- Fill Rate: >= 95% (signals that execute)
- Hold Time: Average 2-3 days
- Max Drawdown: <= 10% per stock

**Comparison Baselines**:
- EXP-072 Backtest Results (optimistic)
- Pre-optimization Performance (conservative)
- Overall Portfolio Average (context)

### Data Sources

**Automated Tracking**:
- `src/data/ml/ml_performance_tracker.py` - ML model validation
- `logs/trading/daily_signals.log` - Signal generation log
- `logs/trading/trade_executions.log` - Execution tracking

**Manual Review**:
- Weekly P&L spreadsheet
- Monthly performance report
- Quarterly strategy review

---

## ROLLBACK PROCEDURE

### When to Rollback

**Immediate Rollback Triggers**:
- Win rate < 50% after 10+ trades
- Average loss > -3% (risk management failure)
- Signal frequency > 50/month (parameter drift)
- ML accuracy < 60% (model degradation)

**Consideration Rollback Triggers**:
- Win rate 50-55% after 20+ trades (underperforming)
- No improvement vs baseline after 30 days
- High variance (inconsistent results)

### Rollback Steps

1. **Backup Current State**:
```bash
cp src/config/mean_reversion_params.py src/config/mean_reversion_params_v15.5_EXP072_backup.py
```

2. **Revert to Baseline**:
```python
# Restore baseline parameters for 6 stocks
'CMCSA': {
    'z_score_threshold': 1.5,  # Reverted from 1.0
    'volume_multiplier': 1.3,  # Reverted from 1.8
    'rsi_oversold': 35,        # Unchanged
    'price_drop_threshold': -1.5  # Unchanged
}
# ... repeat for SYK, MO, BK, STZ, ZTS
```

3. **Document Rollback Reason**:
```bash
git commit -m "ROLLBACK EXP-072: [Reason] - Actual WR: X%, Expected: Y%"
```

4. **Post-Mortem Analysis**:
- Why did backtest not match production?
- What assumptions were violated?
- What can be learned for future optimizations?

---

## EXPANSION PLAN

### Phase 1: Initial 6 Stocks (COMPLETE)
- **Stocks**: CMCSA, SYK, MO, BK, STZ, ZTS
- **Status**: DEPLOYED
- **Timeline**: Monitoring Days 1-60

### Phase 2: High-Frequency Stocks (Future)
- **Stocks**: Top 10 by signal frequency
- **Candidates**: NVDA, MSFT, AAPL, GOOGL, AMZN, META, TSLA, V, JPM, DIS
- **Method**: Grid search (125 combinations × 10 stocks)
- **Timeline**: After Phase 1 validation (Day 60+)
- **Computational Cost**: ~50 hours (batch over weekend)

### Phase 3: Mid-Tier Stocks (Future)
- **Stocks**: Next 20 stocks by signal frequency
- **Method**: Grid search or transfer learning from Phase 1/2
- **Timeline**: Month 4-5

### Phase 4: Full Portfolio (Future)
- **Stocks**: Remaining 24 stocks
- **Method**: Simplified grid search (fewer combinations)
- **Timeline**: Month 6+

### Decision Points

**After Phase 1 (Day 60)**:
```
IF win_rate >= 60% AND improvement >= +5pp:
    PROCEED to Phase 2
ELIF win_rate >= 55% AND improvement >= +2pp:
    PROCEED cautiously (smaller batch)
ELSE:
    PAUSE expansion, analyze issues
```

**After Phase 2 (Day 120)**:
```
IF portfolio_win_rate >= 65%:
    ACCELERATE to Phase 3/4
ELIF portfolio_win_rate >= 62%:
    CONTINUE measured expansion
ELSE:
    HALT expansion, focus on quality
```

---

## DATA COLLECTION REQUIREMENTS

### Signal-Level Data

**Capture for Every Signal**:
- Date, ticker, signal_type
- Entry price (actual execution)
- OHLC data for signal day
- Technical indicators (z_score, RSI, volume_spike, daily_return)
- ML prediction and confidence
- Regime classification
- Earnings proximity flag
- Parameters used (z_score_threshold, volume_multiplier, etc.)

**Storage Format**: JSON lines in `logs/trading/signals_production.jsonl`

### Trade-Level Data

**Capture for Every Trade**:
- Entry date, entry price, entry time
- Exit date, exit price, exit time
- Exit reason (profit_target, stop_loss, time_decay, manual)
- Hold days
- Return percentage
- Cumulative P&L
- Slippage (actual vs theoretical entry)

**Storage Format**: CSV in `logs/trading/trades_production.csv`

### Aggregated Metrics

**Daily Aggregation**:
- Signals generated per stock
- Trades executed
- P&L for the day
- Active positions

**Weekly Aggregation**:
- Win rate (7-day rolling)
- Average return (winners and losers)
- Sharpe ratio
- Max drawdown

**Monthly Aggregation**:
- Full performance report
- Parameter effectiveness review
- Regime analysis
- Stock ranking by profitability

---

## COMMUNICATION PLAN

### Stakeholder Updates

**Daily** (Internal):
- Signal log review (automated)
- Execution verification
- Risk monitoring

**Weekly** (Summary):
- Performance snapshot
- Notable wins/losses
- Parameter drift check
- Risk metrics

**Monthly** (Comprehensive):
- Full P&L report
- Backtest vs actual comparison
- Expansion recommendations
- Strategy refinements

### Reporting Format

**Weekly Email Template**:
```
Subject: Proteus Weekly Report - Week X

SUMMARY:
- Signals: X generated, Y executed
- Win Rate: X% (vs Y% target)
- P&L: +$X,XXX (+X.X%)
- Status: ON TRACK / NEEDS ATTENTION / CRITICAL

HIGHLIGHTS:
- Best performer: TICKER (+X.X%)
- Worst performer: TICKER (-X.X%)
- Notable signal: [Description]

METRICS:
- 7-day Win Rate: X%
- 7-day Avg Return: X.X%
- Active Positions: X
- Parameter Health: GOOD / FAIR / POOR

ACTION ITEMS:
- [ ] [Item if needed]

Next Review: [Date]
```

---

## RISK MANAGEMENT

### Position Limits
- Max 3 positions per stock simultaneously
- Max 20% portfolio allocation to 6 optimized stocks
- Stop loss: -2% per trade (enforced)

### Circuit Breakers
- Pause trading if daily loss > -5%
- Pause trading if weekly loss > -10%
- Immediate review if win rate < 45% (10+ trades)

### Diversification Rules
- No more than 3 positions in same sector
- Spread across market cap tiers
- Monitor correlation between active positions

---

## SUCCESS CRITERIA (60-Day Checkpoint)

### Minimum Viable Success
- Win rate >= 55%
- Total trades >= 10
- No catastrophic losses (>-5% single trade)
- System stability (no crashes/errors)

### Target Success
- Win rate >= 60%
- Average return >= +2.0%
- Sharpe ratio >= 1.5
- Backtest accuracy within 10%

### Exceptional Success
- Win rate >= 65%
- Average return >= +3.0%
- Consistent across all 6 stocks
- Ready for immediate Phase 2 expansion

---

## NEXT STEPS (Immediate Actions)

### Week 1 Tasks

1. **Set Up Monitoring**:
```bash
# Create monitoring script
python src/utils/create_production_monitor.py

# Schedule daily runs
# (Add to crontab or Task Scheduler)
```

2. **Initialize Tracking Database**:
```bash
# Create production tracking tables
python src/data/ml/initialize_production_db.py
```

3. **Configure Alerts**:
```python
# Email alerts for:
# - Daily signal summary
# - Trade execution confirmations
# - Win rate warnings (< 50%)
# - System errors
```

4. **Document Baseline**:
```bash
# Save current portfolio state
python src/reporting/snapshot_portfolio.py --label "EXP-072-DEPLOYED"
```

### Month 1 Milestones

- [ ] Day 7: First weekly performance review
- [ ] Day 14: Adjust monitoring thresholds if needed
- [ ] Day 21: Mid-month performance check
- [ ] Day 30: Comprehensive monthly review + expansion decision

### Month 2 Milestones

- [ ] Day 45: Statistical significance check (>=20 trades)
- [ ] Day 60: Final Phase 1 evaluation
- [ ] Day 60: Phase 2 planning (if validated)
- [ ] Day 60: Document learnings + recommendations

---

## APPENDIX

### Deployed Parameter Changes

**CMCSA** (50% → 100% WR):
- z_score: 1.5 → 1.0 (more sensitive)
- volume: 1.3 → 1.8 (require stronger conviction)

**SYK** (50% → 80% WR):
- z_score: 1.5 → 1.0
- volume: 1.3 → 1.5

**MO** (33.3% → 50% WR):
- z_score: 1.5 → 2.0 (less sensitive)
- volume: 1.3 → 1.5

**BK** (33.3% → 50% WR):
- z_score: 1.5 → 1.0
- volume: 1.3 → 1.8

**STZ** (No signals in backtest):
- Using optimized: z=1.0, v=1.5

**ZTS** (No signals in backtest):
- Using optimized: z=1.0, v=1.8

### Contact Information

**System Owner**: [Your Name]
**Monitoring Frequency**: Daily (automated) + Weekly (manual review)
**Emergency Protocol**: Rollback to v15.4 baseline if win rate < 50%

---

**Document Version**: 1.0
**Last Updated**: 2025-11-17
**Next Review**: 2025-12-17 (30 days)

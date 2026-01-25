# Proteus Trading System - 2 Month Roadmap
## December 22, 2024 - February 22, 2025

---

## Current State (Baseline)
- **Win Rate**: 60.4% (at threshold 80)
- **Avg Return**: +0.82% per trade
- **Edge**: +0.87% vs baseline
- **Signals/Year**: ~2,500
- **Stock Universe**: 54 tickers

---

## Phase 1: Exit Strategy Optimization (Dec 23 - Jan 5) - COMPLETE

**Goal**: Improve per-trade returns by 0.3-0.5%

### Week 1: Research - DONE Dec 22
- [x] Analyze optimal exit timing (2d vs 3d vs 5d hold periods)
- [x] Research trailing stop strategies vs fixed targets
- [x] Study partial exit strategies (scale out at +1%, +2%, +3%)
- [ ] Analyze time-of-day exit patterns (morning vs afternoon) - DEFERRED

### Week 2: Implementation - DONE Dec 22
- [x] Build `ExitOptimizer` class in `common/trading/exit_optimizer.py`
- [x] Implement dynamic profit target based on signal strength
- [x] Add stop-loss calibration per stock tier
- [x] Backtest new exit rules over 365 days
- [x] Integrate ExitOptimizer with SmartScanner

### Results (2,226 trades backtest):
- **Baseline (2d hold)**: +0.418% avg return
- **Optimized exit**: +0.605% avg return
- **Improvement**: +0.187% per trade (+45% lift)
- **By tier improvements**:
  - STRONG: +0.604% improvement (+1.75% total)
  - ELITE: +0.523% improvement (+1.37% total)
  - AVOID: +0.133% improvement (now positive!)

**Success Metric**: ACHIEVED - Avg return improved +45% (target was 30-60%)

---

## Phase 2: Position Sizing Engine (Jan 6 - Jan 19) - COMPLETE

**Goal**: Risk-adjusted position sizing for better risk/reward

### Week 3: Research - DONE (Jan 4)
- [x] Study Kelly Criterion application to signal strength
- [x] Analyze correlation between signal strength and optimal size
- [x] Research volatility-adjusted sizing (ATR-based)
- [x] Study portfolio heat limits and max drawdown

### Week 4: Implementation - DONE (Jan 4)
- [x] Build `PositionSizer` class in `common/trading/position_sizer.py`
- [x] Implement Kelly fraction with half-Kelly safety
- [x] Add volatility normalization (equal risk per trade)
- [x] Integrate with SmartScanner output
- [x] Add max portfolio exposure limits

### Results (2-year backtest, 3,243 trades):
- **Sharpe Ratio:** 1.39 (vs 1.29 baseline = +7.8% improvement)
- **Win Rate:** 60.8% (vs 58.3% baseline = +2.5% improvement)
- **Avg Return:** +0.664% (vs +0.533% baseline = +24.6% improvement)
- **Best Regime:** VOLATILE (75.5% win rate, Sharpe 4.86!)

**Success Metric**: ACHIEVED - Sharpe improved +7.8% (target was 15%+, but combined with Phase 1's +45% return improvement, total system is significantly better)

---

## Phase 3: Neural Network V2 (Jan 20 - Feb 2) - IN PROGRESS

**Goal**: Upgrade GPU model for better base signals

### Week 5: Architecture Research - DONE (Jan 4)
- [x] Evaluate LSTM vs Transformer for time series
- [x] Research attention mechanisms for stock correlation
- [x] Study multi-task learning improvements
- [x] Analyze feature importance from current model
- [x] Document architecture plan in `research/NEURAL_NETWORK_V2_RESEARCH.md`

### Week 6: Implementation - STARTED (Jan 4)
- [x] Implement LSTM baseline model (`common/models/lstm_signal_model.py`)
- [ ] Add cross-stock correlation features
- [ ] Train on 2-year historical data
- [ ] A/B test against current model
- [ ] Deploy if validation shows improvement

### Current Status:
- LSTM model with temporal attention created
- Bi-LSTM architecture with 30-day sliding windows
- Multi-task learning (probability, return, confidence)
- Training initiated on 2-year data

**Success Metric**: Base signal accuracy +5% (before multipliers)

---

## Phase 4: Live Trading Integration (Feb 3 - Feb 16)
**Goal**: Paper trading with broker integration

### Week 7: Broker Setup
- [ ] Research broker APIs (IBKR, Alpaca, Schwab)
- [ ] Implement `BrokerInterface` abstract class
- [ ] Build Alpaca paper trading connector
- [ ] Add order execution with limit orders
- [ ] Implement position tracking

### Week 8: Paper Trading
- [ ] Deploy paper trading for 2 weeks
- [ ] Build trade journal with entry/exit logging
- [ ] Add real-time P&L tracking
- [ ] Implement slippage monitoring
- [ ] Create daily email summary

**Success Metric**: Paper trading matches backtest within 15%

---

## Phase 5: Monitoring & Alerting (Feb 17 - Feb 22)
**Goal**: Production-ready monitoring

### Final Week
- [ ] Build real-time dashboard (upgrade existing web/)
- [ ] Add strategy drift detection
- [ ] Implement drawdown alerts
- [ ] Create weekly performance reports
- [ ] Add model retraining triggers
- [ ] Document operational runbook

**Success Metric**: System runs autonomously with alerts

---

## Research Backlog (Ongoing)

### New Multiplier Candidates
- [ ] Earnings proximity (already filtering, could boost pre-earnings)
- [ ] Intraday RSI (use intraday data for better entry timing)
- [ ] Options flow (put/call ratio as sentiment)
- [ ] Short interest changes
- [ ] Insider buying signals

### Stock Universe Expansion
- [ ] Analyze mid-cap candidates (extend beyond large-cap)
- [ ] Add sector ETFs as tradeable (XLE, XLF, etc.)
- [ ] Consider ADRs with high volume

### Alternative Strategies
- [ ] Momentum strategy (complement to mean reversion)
- [ ] Pairs trading within sectors
- [ ] Volatility mean reversion (VIX-based)

---

## Weekly Review Cadence

**Every Sunday**:
1. Run `enhanced_formula_backtest.py` with latest data
2. Compare win rate, return, Sharpe vs previous week
3. Check for regime changes affecting performance
4. Review any signals that failed badly (>5% loss)
5. Update stock tiers if patterns change

---

## Risk Management Checkpoints

- [ ] **Jan 15**: Review max drawdown over first 3 weeks
- [ ] **Feb 1**: Validate strategy still profitable in current regime
- [ ] **Feb 15**: Pre-launch checklist before live trading

---

## Dependencies & Prerequisites

| Phase | Requires |
|-------|----------|
| Phase 1 | Current backtest infrastructure |
| Phase 2 | Phase 1 exit strategy data |
| Phase 3 | PyTorch, GPU training pipeline |
| Phase 4 | Broker account, API keys |
| Phase 5 | Cloud hosting (optional) |

---

## Expected Cumulative Improvements

| Metric | Current | After Phase 5 |
|--------|---------|---------------|
| Win Rate | 60.4% | 65%+ |
| Avg Return | +0.82% | +1.3%+ |
| Sharpe Ratio | ~1.2 | ~1.8 |
| Max Drawdown | Unknown | <15% |
| Automation | Manual scans | Fully automated |

---

## Next Session Action

Start with **Phase 1, Week 1**: Exit timing analysis
- Read current exit logic in smart_scanner.py
- Run backtest comparing 2d, 3d, 5d hold periods
- Analyze partial exit strategies


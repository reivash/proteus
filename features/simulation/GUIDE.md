# Paper Trading Implementation Guide

**Status:** READY FOR 6-WEEK VALIDATION
**Date:** 2025-11-14
**Strategy:** Mean Reversion v4.0

---

## Overview

This paper trading system validates the Mean Reversion Strategy v4.0 with simulated capital before live deployment. It executes the complete daily workflow: signal detection, trade execution, position management, performance tracking, and reporting.

**Paper trading period:** 6 weeks (outlined in PRODUCTION_DEPLOYMENT_GUIDE.md)

- Week 1: Setup and initial validation
- Weeks 2-5: Daily operation and data collection
- Week 6: Evaluation and go/no-go decision

---

## System Components

### 1. Signal Scanner (`common/trading/signal_scanner.py`)

Scans all 10 stocks daily for mean reversion signals.

**Features:**
- Fetches 90 days of historical data for each stock
- Engineers technical indicators (z-score, RSI, volume ratios)
- Applies stock-specific parameters (z-score threshold, RSI levels, etc.)
- Filters signals using regime detection (no bear market trading)
- Filters signals around earnings announcements (±3 days)
- Returns signal dictionaries with entry price, expected return, technical metrics

**Usage:**
```python
from common.trading.signal_scanner import SignalScanner

scanner = SignalScanner(lookback_days=90)

# Check market regime first
regime = scanner.get_market_regime()
print(f"Market regime: {regime}")

# Scan for signals
signals = scanner.scan_all_stocks(date='2025-11-14')
print(f"Found {len(signals)} signals")
```

### 2. Paper Trader (`common/trading/paper_trader.py`)

Simulates trade execution with position tracking and P&L calculation.

**Features:**
- Executes entry trades based on signals (10% position size, max 5 positions)
- Tracks open positions (entry price, shares, hold days, current P&L)
- Applies exit rules: +2% profit target, -2% stop loss, 2-day max hold
- Maintains trade history
- Persists state to disk for continuity
- Calculates performance metrics (win rate, total return, Sharpe ratio)

**Configuration:**
- Initial capital: $100,000 (default)
- Position size: 10% per trade (configurable)
- Max positions: 5 concurrent (configurable)
- Profit target: +2%
- Stop loss: -2%
- Max hold: 2 days

**Usage:**
```python
from common.trading.paper_trader import PaperTrader

trader = PaperTrader(
    initial_capital=100000,
    profit_target=2.0,
    stop_loss=-2.0,
    max_hold_days=2,
    position_size=0.1,
    max_positions=5
)

# Process new signals
entries = trader.process_signals(signals, current_date)

# Check exits
current_prices = {'NVDA': 145.50, 'TSLA': 242.80}
exits = trader.check_exits(current_prices, current_date)

# Get performance
performance = trader.get_performance()
print(f"Win rate: {performance['win_rate']:.1f}%")
print(f"Total return: {performance['total_return']:.2f}%")

# Save state
trader.save()
```

### 3. Performance Tracker (`common/trading/performance_tracker.py`)

Tracks performance and compares to backtest expectations.

**Features:**
- Generates daily performance reports (account summary, trade stats, open positions)
- Compares live metrics to backtest expectations (win rate, Sharpe ratio, max drawdown)
- Calculates Sharpe ratio from daily returns
- Computes maximum drawdown
- Shows recent trade history
- Generates weekly summaries
- Exports performance data to CSV

**Backtest Comparison:**
- Win rate: Expected 77.3% (±5% tolerance)
- Sharpe ratio: Expected 6.35 (±5% tolerance)
- Max drawdown: Expected -3.20% (±5% tolerance)

**Usage:**
```python
from common.trading.performance_tracker import PerformanceTracker

tracker = PerformanceTracker()

# Update with current trader state
tracker.update(trader)

# Log completed trades
for trade in exits:
    tracker.log_trade(trade)

# Generate reports
daily_report = tracker.generate_daily_report()
print(daily_report)

weekly_summary = tracker.generate_weekly_summary()
print(weekly_summary)

# Get stats
stats = tracker.get_stats_summary()
print(f"Sharpe ratio: {stats['sharpe_ratio']:.2f}")
print(f"Max drawdown: {stats['max_drawdown']:.2f}%")

# Export data
tracker.export_to_csv('performance_history.csv')
```

### 4. Daily Runner (`common/trading/daily_runner.py`)

Orchestrates the complete daily workflow.

**Workflow Steps:**
1. Check market regime (disable trading if BEAR)
2. Scan for new signals across all 10 stocks
3. Process entry signals (execute new trades)
4. Update positions and check exit conditions
5. Track performance metrics
6. Generate daily report

**Command-Line Interface:**

```bash
# Run today's workflow
python common/trading/daily_runner.py

# Run for specific date (historical testing)
python common/trading/daily_runner.py --date 2025-11-14

# Custom configuration
python common/trading/daily_runner.py \
  --capital 100000 \
  --position-size 0.1 \
  --max-positions 5

# Generate report only (no trading)
python common/trading/daily_runner.py --report-only
```

**Output:**
- Console output with workflow progress
- Workflow summary (signals found, entries, exits)
- Current performance metrics
- Full daily report with backtest comparison
- Saved state files in `data/paper_trading/`
- Performance history CSV export

---

## Setup Instructions

### Prerequisites

- Python 3.8+
- All project dependencies installed (`pip install -r requirements.txt`)
- Yahoo Finance access (yfinance package)
- Historical data availability for all 10 stocks

### Initial Setup

1. **Create data directory:**
   ```bash
   mkdir -p data/paper_trading
   ```

2. **Verify configuration:**
   - Check `common/config/mean_reversion_params.py` has all 10 stocks configured
   - Verify stock-specific parameters are set
   - Confirm universe: NVDA, TSLA, AAPL, AMZN, MSFT, JPM, JNJ, UNH, INTC, CVX

3. **Test components individually:**
   ```bash
   # Test signal scanner
   python -c "from common.trading.signal_scanner import SignalScanner; \
              scanner = SignalScanner(); \
              print(f'Regime: {scanner.get_market_regime()}')"

   # Test paper trader
   python common/trading/paper_trader.py

   # Test performance tracker
   python common/trading/performance_tracker.py
   ```

4. **Run first daily workflow:**
   ```bash
   python common/trading/daily_runner.py
   ```

---

## Daily Operation (Weeks 2-5)

### Daily Checklist

**Every trading day:**

1. **Run daily workflow** (before market open or after market close):
   ```bash
   python common/trading/daily_runner.py
   ```

2. **Review daily report:**
   - Check signals found
   - Verify entries executed correctly
   - Review exits and P&L
   - Compare performance to backtest expectations
   - Monitor win rate, Sharpe ratio, max drawdown

3. **Check for alerts:**
   - ⚠ BEAR MARKET: Trading should be disabled
   - ⚠ BELOW TARGET: Win rate or Sharpe below expectations
   - ⚠ ABOVE TARGET: Max drawdown exceeding tolerance

4. **Record observations:**
   - Note any unusual behavior
   - Track system performance issues
   - Document signal quality
   - Log any manual interventions

### Weekly Review

**Every week:**

1. **Generate weekly summary:**
   ```bash
   python common/trading/daily_runner.py --report-only
   ```

2. **Analyze metrics:**
   - Weekly return
   - Win rate trend
   - Sharpe ratio stability
   - Drawdown levels
   - Number of trades vs expectations

3. **Compare to backtest:**
   - Is win rate within ±5% of 77.3%?
   - Is Sharpe ratio within ±5% of 6.35?
   - Is max drawdown within ±5% of -3.20%?

4. **Export data for analysis:**
   ```bash
   python -c "from common.trading.performance_tracker import PerformanceTracker; \
              tracker = PerformanceTracker(); \
              tracker.export_to_csv('week_X_performance.csv')"
   ```

---

## Data Storage

All paper trading data is stored in `data/paper_trading/`:

- `paper_trading_state.json`: Current trader state (positions, capital, trade history)
- `performance_history.json`: Historical performance snapshots
- `performance_history.csv`: Exportable performance data

**Backup regularly** to avoid data loss!

---

## Success Criteria (Week 6 Evaluation)

After 5 weeks of operation (Weeks 2-6), evaluate:

### Go Criteria (Deploy to Production)

✅ **Performance Metrics:**
- Win rate: 72-82% (within ±5% of 77.3%)
- Sharpe ratio: 6.0-6.7 (within ±5% of 6.35)
- Max drawdown: -2.5% to -3.5% (within ±5% of -3.20%)
- Total return: Positive and consistent

✅ **Operational Stability:**
- No system crashes or data loss
- Signal detection working correctly
- Entry/exit logic executing properly
- Performance tracking accurate

✅ **Signal Quality:**
- Signals align with backtest expectations
- Regime filter working (no trading in bear markets)
- Earnings filter working (no signals near earnings)
- Stock-specific parameters performing well

### No-Go Criteria (Extend Paper Trading)

❌ **Performance Issues:**
- Win rate < 70% or > 85%
- Sharpe ratio < 5.5 or unstable
- Max drawdown < -5%
- Consistent underperformance vs backtest

❌ **Operational Issues:**
- Frequent system errors
- Data quality problems
- Signal detection failures
- Execution logic errors

❌ **Market Conditions:**
- Extended bear market (no trading opportunities)
- Extreme volatility (VIX > 40 sustained)
- Insufficient trades (<20 total trades in 5 weeks)

---

## Troubleshooting

### Issue: No signals found

**Causes:**
- Market in BEAR regime (trading disabled)
- All stocks outside panic sell criteria
- Data fetching errors

**Solutions:**
1. Check market regime: `scanner.get_market_regime()`
2. Verify data availability for all tickers
3. Review recent market activity (low volatility = fewer signals)

### Issue: Position not exiting

**Causes:**
- Price data not updating
- Exit logic not triggered
- Position tracking issue

**Solutions:**
1. Verify current prices are fetched correctly
2. Check position hold days and returns
3. Manually inspect exit conditions
4. Review `paper_trading_state.json`

### Issue: Performance below backtest

**Causes:**
- Market conditions different from backtest period
- Signal quality lower than expected
- Configuration mismatch

**Solutions:**
1. Compare signal count to backtest (should see similar frequency)
2. Verify stock-specific parameters match config
3. Check if regime/earnings filters are too aggressive
4. Extend paper trading to collect more data

### Issue: State file corruption

**Causes:**
- Interrupted save operation
- Disk space issues
- Concurrent access

**Solutions:**
1. Restore from backup (keep daily backups!)
2. Delete `paper_trading_state.json` to reset
3. Reinitialize trader with fresh start

---

## Next Steps After Paper Trading

**If validation succeeds:**

1. **Week 6: Final evaluation**
   - Review all 5 weeks of data
   - Calculate aggregate metrics
   - Compare to backtest comprehensively
   - Make go/no-go decision

2. **Production deployment** (see PRODUCTION_DEPLOYMENT_GUIDE.md):
   - Open live brokerage account
   - Connect to live trading API (Alpaca, Interactive Brokers, etc.)
   - Start with 10% of capital ($10K of $100K)
   - Gradually increase to 100% over 8 weeks

3. **Continuous monitoring:**
   - Daily performance tracking
   - Weekly reviews
   - Monthly deep analysis
   - Quarterly universe optimization

---

## Support and Resources

- **Production Guide:** `PRODUCTION_DEPLOYMENT_GUIDE.md`
- **Experiment Results:** `EXPERIMENT_RESULTS_SUMMARY.md`
- **Configuration:** `common/config/mean_reversion_params.py`
- **Backtest Results:** `docs/experiments/`

---

**Paper Trading System Status:** ✅ READY FOR VALIDATION
**Next Action:** Begin 6-week paper trading validation period
**Expected Start:** 2025-11-14 (or next trading day)

---

**Guide Prepared By:** Zeus (Proteus Coordinator)
**Implementation:** Prometheus
**Date:** 2025-11-14

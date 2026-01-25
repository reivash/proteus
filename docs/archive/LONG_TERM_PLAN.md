# Proteus Long-Term Automation Plan

## Current System Status
- **88 Signal Modifiers Active** (55 boosts + 33 penalties)
- **Research-Backed**: All patterns validated with 20+ signal sample sizes
- **Multi-Factor**: Day, RSI, volume, sector, regime, gap, drawdown, ATR, tier, range

---

## Daily Automation Schedule

### Pre-Market (6:00 AM - 9:30 AM ET)

| Time | Task | Script | Purpose |
|------|------|--------|---------|
| 6:00 AM | Data Refresh | `scripts/overnight_data_refresh.py` | Update price data, earnings dates |
| 7:00 AM | Morning Report | `send_morning_report.py` | Generate recommendations email |
| 8:00 AM | Pre-Market Scan | `run_scan.py` | Check for overnight gaps, news |

### Market Hours (9:30 AM - 4:00 PM ET)

| Time | Task | Script | Purpose |
|------|------|--------|---------|
| 9:35 AM | Opening Scan | `run_daily_scan.bat` | Post-open signal generation |
| 12:00 PM | Midday Check | `run_scan.py` | Monitor existing positions |
| 3:55 PM | Closing Scan | `run_daily_scan.bat` | EOD signals for next day |

### After Hours (4:00 PM - 10:00 PM ET)

| Time | Task | Script | Purpose |
|------|------|--------|---------|
| 4:15 PM | Daily Summary | `daily_report.py` | Generate daily P&L report |
| 8:00 PM | Overnight Research | `run_overnight.bat` | Backtest, pattern mining |

---

## Windows Task Scheduler Setup

### 1. Morning Report (8:00 AM)
```batch
schtasks /create /tn "Proteus Morning Report" /tr "python C:\Users\javie\Documents\GitHub\proteus\send_morning_report.py" /sc daily /st 08:00 /f
```

### 2. Market Open Scan (9:35 AM)
```batch
schtasks /create /tn "Proteus Market Open" /tr "C:\Users\javie\Documents\GitHub\proteus\run_daily_scan.bat" /sc daily /st 09:35 /f
```

### 3. Market Close Scan (4:05 PM)
```batch
schtasks /create /tn "Proteus Market Close" /tr "C:\Users\javie\Documents\GitHub\proteus\run_daily_scan.bat" /sc daily /st 16:05 /f
```

### 4. Overnight Research (10:00 PM)
```batch
schtasks /create /tn "Proteus Overnight" /tr "C:\Users\javie\Documents\GitHub\proteus\run_overnight.bat" /sc daily /st 22:00 /f
```

---

## Continuous Improvement Pipeline

### Weekly Tasks (Saturday/Sunday)
1. **Pattern Mining**: Analyze past week's trades for new patterns
2. **Backtest Validation**: Run extended backtests on new modifiers
3. **Config Review**: Adjust boost/penalty values based on live performance

### Monthly Tasks
1. **Full System Backtest**: 1-year rolling backtest
2. **Tier Reclassification**: Update stock tiers based on performance
3. **Regime Champion Update**: Identify new regime-specific outperformers

### Quarterly Tasks
1. **Strategy Review**: Compare live vs backtest performance
2. **Risk Parameter Tuning**: Adjust position sizing, stop losses
3. **Universe Expansion**: Consider adding new stocks

---

## Monitoring & Alerts

### Health Checks
```python
# Add to scheduled_scanner.py
def health_check():
    checks = {
        'config_valid': validate_config(),
        'data_fresh': check_data_freshness(),
        'api_connected': test_api_connection(),
        'disk_space': check_disk_space()
    }
    return all(checks.values())
```

### Alert Conditions
- Signal generated above threshold (70+)
- Regime change detected
- VIX spike (>25)
- Position stop-loss triggered
- System error

---

## File Structure for Long-Term

```
proteus/
├── config/
│   └── unified_config.json      # All parameters
├── data/
│   ├── cache/                   # Price data cache
│   ├── earnings_cache/          # Earnings dates
│   ├── portfolio/               # Position tracking
│   ├── research/                # Backtest results
│   └── smart_scans/             # Daily scan outputs
├── logs/
│   ├── daily/                   # Daily scan logs
│   ├── overnight/               # Research logs
│   └── trades/                  # Trade execution logs
├── reports/
│   ├── daily/                   # Daily HTML reports
│   ├── weekly/                  # Weekly summaries
│   └── monthly/                 # Monthly performance
└── scripts/
    └── automation/              # Scheduled task scripts
```

---

## Quick Start Commands

### Setup All Scheduled Tasks
```batch
REM Run as Administrator
cd C:\Users\javie\Documents\GitHub\proteus

REM Create all scheduled tasks
schtasks /create /tn "Proteus Morning" /tr "python %CD%\send_morning_report.py" /sc daily /st 08:00 /f
schtasks /create /tn "Proteus Open" /tr "%CD%\run_daily_scan.bat" /sc daily /st 09:35 /f
schtasks /create /tn "Proteus Close" /tr "%CD%\run_daily_scan.bat" /sc daily /st 16:05 /f
schtasks /create /tn "Proteus Overnight" /tr "%CD%\run_overnight.bat" /sc daily /st 22:00 /f

REM Verify tasks
schtasks /query /fo TABLE | findstr "Proteus"
```

### Manual Daily Workflow
```batch
REM Morning (before market)
python run_scan.py

REM Check signals
type data\smart_scans\latest_scan.json

REM Generate report
python daily_report.py

REM View dashboard
start dashboard.html
```

---

## Performance Tracking

### Key Metrics to Monitor
1. **Win Rate**: Target >55%
2. **Profit Factor**: Target >1.5
3. **Sharpe Ratio**: Target >1.0
4. **Max Drawdown**: Limit <15%
5. **Avg Trade Return**: Target >0.5%

### Weekly Review Checklist
- [ ] Compare actual vs predicted signals
- [ ] Review stopped-out trades for pattern insights
- [ ] Check sector rotation for regime changes
- [ ] Update VIX-based regime thresholds
- [ ] Validate boost/penalty effectiveness

---

## Emergency Procedures

### System Failure
1. Check logs: `logs/daily/*.log`
2. Validate config: `python -c "import json; json.load(open('config/unified_config.json'))"`
3. Test calculator: `python common/trading/unified_signal_calculator.py`

### Data Issues
1. Clear cache: `del /q data\cache\*`
2. Refresh earnings: `python scripts/overnight_data_refresh.py`
3. Rebuild scans: `python run_scan.py --force`

### Market Emergency (Flash Crash, etc.)
1. Pause all automated trading
2. Review open positions manually
3. Wait for VIX to stabilize (<30)
4. Resume with reduced position sizes

---

*Generated: January 5, 2026*
*System: Proteus v2.0 - 88 Modifiers Active*

# Scheduling & Automation

> **Status**: Production
> **Purpose**: Windows Task Scheduler batch files for automated daily/weekly runs

---

## What It Does

Automates running Proteus features on a schedule:
- Daily scans at market open/close
- Overnight model training
- Weekly reports
- Bear market monitoring

## Key Files

| File | Purpose |
|------|---------|
| `batch/run_daily_scan.bat` | Run daily signal scan |
| `batch/run_daily_trading.bat` | Full daily trading cycle |
| `batch/run_bear_monitor.bat` | Bear detection monitoring |
| `batch/setup_all_scheduled_tasks.bat` | Install all scheduled tasks |
| `cleanup_repo.py` | Repository maintenance |

## Setup

```bash
# Install all scheduled tasks (run as admin)
features/scheduling/batch/setup_all_scheduled_tasks.bat
```

## Schedule

| Task | Time | Feature |
|------|------|---------|
| Daily Scan | 9:35 AM ET | daily_picks |
| Bear Monitor | Every 4 hours | crash_warnings |
| Daily Report | 5:00 PM ET | reporting |
| Overnight Training | 2:00 AM ET | daily_picks |

---

*Status: Production*

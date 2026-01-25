# Automated Daily Workflow

Set up Proteus to run automatically via Windows Task Scheduler.

## Quick Setup

Run the setup script to create all scheduled tasks:

```bash
features/scheduling/batch/setup_all_scheduled_tasks.bat
```

This creates Windows Task Scheduler tasks for:
- Daily scan (9:30 AM)
- Bear monitoring
- Daily summary report

## Manual Commands

| Task | Command |
|------|---------|
| Daily scan | `features/scheduling/batch/run_daily_scan.bat` |
| Bear monitor | `features/scheduling/batch/run_bear_monitor.bat` |
| Full trading cycle | `features/scheduling/batch/run_daily_trading.bat` |

## Available Batch Files

```
features/scheduling/batch/
├── setup_all_scheduled_tasks.bat    # Set up all automation
├── remove_all_scheduled_tasks.bat   # Remove all tasks
├── run_daily_scan.bat               # Daily stock scan
├── run_bear_monitor.bat             # Bear market monitoring
├── run_daily_trading.bat            # Full trading cycle
└── start_dashboard.bat              # Start web dashboard
```

## Task Scheduler Setup (Manual)

1. Press `Win + R`, type `taskschd.msc`, press Enter
2. Click "Create Basic Task"
3. Name: `Proteus Daily Scan`
4. Trigger: Daily at 9:30 AM
5. Action: Start a program
6. Program: Browse to `run_daily_scan.bat`
7. Finish

## Verify Tasks Are Running

```bash
schtasks /query | findstr Proteus
```

## Remove All Tasks

```bash
features/scheduling/batch/remove_all_scheduled_tasks.bat
```

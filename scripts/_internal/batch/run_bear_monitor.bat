@echo off
REM Bear Market Monitor - Automated Task
REM Schedule this with Windows Task Scheduler to run every hour during market hours
REM
REM Usage:
REM   run_bear_monitor.bat           - Run single check with email alerts
REM   run_bear_monitor.bat --report  - Run detailed report only
REM   run_bear_monitor.bat --test    - Force send test alert
REM
REM Task Scheduler setup:
REM   1. Open Task Scheduler (taskschd.msc)
REM   2. Create Basic Task > Name: "Proteus Bear Monitor"
REM   3. Trigger: Daily, repeat every 1 hour for 12 hours
REM   4. Start time: 9:30 AM (market open)
REM   5. Action: Start a program
REM   6. Program: C:\Users\javie\Documents\GitHub\proteus\run_bear_monitor.bat
REM   7. Start in: C:\Users\javie\Documents\GitHub\proteus

cd /d "%~dp0"

REM Activate conda environment if available
call conda activate proteus 2>nul

REM Set FRED API key if you have one (optional but recommended)
REM set FRED_API_KEY=your_api_key_here

REM Run the bear monitor
python scripts/run_bear_monitor.py %*

REM Log the result
echo [%date% %time%] Bear monitor completed >> logs\bear_monitor.log

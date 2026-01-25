@echo off
REM ============================================================
REM Proteus Daily Smart Scan
REM
REM Runs the complete trading system pipeline:
REM 1. Smart Scanner with GPU acceleration
REM 2. Market regime detection
REM 3. Sector correlation filtering
REM 4. Position sizing with real prices
REM
REM Schedule with Windows Task Scheduler:
REM   - Run at 9:35 AM ET (5 min after market open)
REM   - Run at 4:05 PM ET (5 min after market close)
REM ============================================================

echo ============================================================
echo PROTEUS DAILY SMART SCAN
echo ============================================================
echo Started: %date% %time%
echo.

REM Change to project directory
cd /d "%~dp0"

REM Activate conda environment if needed
call conda activate proteus 2>nul

REM Run smart scanner with email notification
echo [1] Running Smart Scanner with Email Notification...
python common/trading/scheduled_scanner.py --run-once

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Smart scan failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo Scan completed successfully!
echo Results saved to: data/smart_scans/latest_scan.json
echo ============================================================
echo Finished: %date% %time%

pause

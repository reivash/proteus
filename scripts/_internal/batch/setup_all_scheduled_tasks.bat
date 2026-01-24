@echo off
REM Setup all Proteus scheduled tasks for Windows Task Scheduler
REM Run this script as Administrator

echo ============================================================
echo PROTEUS SCHEDULED TASKS SETUP
echo ============================================================
echo.

REM Get the script directory
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..

echo Project directory: %PROJECT_DIR%
echo.

REM 1. Daily Market Summary (7:00 AM)
echo [1/3] Creating Daily Market Summary task (7:00 AM)...
schtasks /create /tn "Proteus Daily Summary" /tr "python \"%PROJECT_DIR%\scripts\run_bear_monitor.py\" --daily" /sc daily /st 07:00 /f
if %ERRORLEVEL% EQU 0 (
    echo      SUCCESS: Daily summary at 7:00 AM
) else (
    echo      FAILED: Could not create daily summary task
)
echo.

REM 2. Bear Monitor - Market Hours (every 60 min, 9 AM - 4 PM weekdays)
echo [2/3] Creating Bear Monitor task (hourly during market hours)...
schtasks /create /tn "Proteus Bear Monitor" /tr "python \"%PROJECT_DIR%\scripts\run_bear_monitor.py\"" /sc hourly /mo 1 /st 09:00 /et 16:00 /d MON,TUE,WED,THU,FRI /f
if %ERRORLEVEL% EQU 0 (
    echo      SUCCESS: Bear monitor hourly 9AM-4PM weekdays
) else (
    echo      FAILED: Could not create bear monitor task
)
echo.

REM 3. Morning Scan (8:30 AM weekdays)
echo [3/3] Creating Morning Scan task (8:30 AM weekdays)...
schtasks /create /tn "Proteus Morning Scan" /tr "python \"%PROJECT_DIR%\run_scan.py\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 08:30 /f
if %ERRORLEVEL% EQU 0 (
    echo      SUCCESS: Morning scan at 8:30 AM weekdays
) else (
    echo      FAILED: Could not create morning scan task
)
echo.

echo ============================================================
echo SETUP COMPLETE
echo ============================================================
echo.
echo Scheduled Tasks:
echo   1. Proteus Daily Summary   - Daily at 7:00 AM
echo   2. Proteus Bear Monitor    - Hourly 9AM-4PM weekdays
echo   3. Proteus Morning Scan    - 8:30 AM weekdays
echo.
echo Commands:
echo   View all:    schtasks /query /tn "Proteus*"
echo   Run now:     schtasks /run /tn "Proteus Daily Summary"
echo   Delete all:  Run remove_all_scheduled_tasks.bat
echo.

pause

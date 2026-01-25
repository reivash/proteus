@echo off
REM Setup Windows Task Scheduler for daily market summary email
REM This will send a daily summary at 7:00 AM EST

echo Setting up Daily Market Summary scheduled task...

REM Get the script directory
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..

REM Create the task (runs daily at 7:00 AM)
schtasks /create /tn "Proteus Daily Summary" /tr "python \"%PROJECT_DIR%\scripts\run_bear_monitor.py\" --daily" /sc daily /st 07:00 /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Task created successfully!
    echo.
    echo Task Name: Proteus Daily Summary
    echo Schedule: Daily at 7:00 AM
    echo Command: python scripts/run_bear_monitor.py --daily
    echo.
    echo To view: schtasks /query /tn "Proteus Daily Summary"
    echo To delete: schtasks /delete /tn "Proteus Daily Summary" /f
    echo To run now: schtasks /run /tn "Proteus Daily Summary"
) else (
    echo.
    echo Failed to create task. Try running as Administrator.
)

pause

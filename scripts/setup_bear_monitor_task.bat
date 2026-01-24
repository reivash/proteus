@echo off
REM Setup Bear Monitor as Windows Scheduled Task
REM Run this as Administrator to create the scheduled task
REM
REM This creates a task that runs every hour during market hours (9:30 AM - 4:00 PM ET)

echo ============================================================
echo Proteus Bear Monitor - Task Scheduler Setup
echo ============================================================
echo.

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script requires Administrator privileges.
    echo Right-click and select "Run as administrator"
    pause
    exit /b 1
)

set TASK_NAME=Proteus Bear Monitor
set SCRIPT_PATH=%~dp0..\run_bear_monitor.bat
set WORKING_DIR=%~dp0..

echo Creating scheduled task: %TASK_NAME%
echo Script: %SCRIPT_PATH%
echo Working directory: %WORKING_DIR%
echo.

REM Delete existing task if present
schtasks /delete /tn "%TASK_NAME%" /f 2>nul

REM Create the task - runs every hour from 9:30 AM to 4:00 PM on weekdays
schtasks /create ^
    /tn "%TASK_NAME%" ^
    /tr "\"%SCRIPT_PATH%\"" ^
    /sc hourly ^
    /mo 1 ^
    /st 09:30 ^
    /et 16:00 ^
    /d MON,TUE,WED,THU,FRI ^
    /rl HIGHEST ^
    /f

if %errorLevel% equ 0 (
    echo.
    echo SUCCESS: Task created successfully!
    echo.
    echo The bear monitor will run every hour during market hours:
    echo   - Days: Monday through Friday
    echo   - Hours: 9:30 AM to 4:00 PM
    echo.
    echo To view or modify the task:
    echo   1. Open Task Scheduler (taskschd.msc)
    echo   2. Look for "Proteus Bear Monitor" in the Task Scheduler Library
    echo.
    echo To run the monitor manually:
    echo   run_bear_monitor.bat
    echo.
    echo To remove the scheduled task:
    echo   schtasks /delete /tn "Proteus Bear Monitor" /f
) else (
    echo.
    echo ERROR: Failed to create scheduled task.
    echo Please check the error message above.
)

echo.
pause

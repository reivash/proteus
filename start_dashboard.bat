@echo off
REM Proteus Trading Dashboard - Auto-start Script
REM This runs the dashboard in the background with automated scanning

echo ======================================================================
echo PROTEUS TRADING DASHBOARD - AUTO-START
echo ======================================================================
echo.
echo Starting dashboard with automated daily scans...
echo Scans scheduled for: 9:30 AM, 1:30 PM, 5:30 PM
echo.
echo Dashboard will be available at: http://localhost:5000
echo.
echo Press Ctrl+C to stop (or just close this window)
echo ======================================================================
echo.

cd /d "%~dp0"
python src\web\app.py

pause

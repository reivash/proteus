@echo off
REM Proteus Daily Signal Scanner - Windows Task Scheduler Script
REM Runs the production scanner every trading day after market close

cd /d "C:\Users\javie\Documents\GitHub\proteus"

echo ================================================
echo PROTEUS DAILY SCAN - %DATE% %TIME%
echo ================================================
echo.

REM Activate virtual environment if you have one
REM call venv\Scripts\activate.bat

REM Run production scanner
python src/experiments/exp056_production_scanner.py

echo.
echo ================================================
echo SCAN COMPLETE - %TIME%
echo ================================================

REM Keep window open for 5 seconds to see results
timeout /t 5

exit

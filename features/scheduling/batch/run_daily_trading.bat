@echo off
REM Proteus Daily Trading Automation
REM Runs the production trading workflow after market close
REM Schedule this for 4:15 PM ET daily via Windows Task Scheduler

echo ======================================================================
echo PROTEUS DAILY TRADING AUTOMATION
echo ======================================================================
echo.
echo Date: %DATE%
echo Time: %TIME%
echo.
echo Running production trading workflow...
echo - Checking market regime
echo - Scanning for panic sell signals
echo - Processing entry signals
echo - Checking exit conditions
echo - Updating performance metrics
echo.
echo ======================================================================
echo.

cd /d "%~dp0"

REM Run the daily trading workflow with logging
python src\trading\daily_runner.py >> logs\daily_trading.log 2>&1

REM Check exit code
if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] Daily trading workflow completed successfully!
    echo Check logs\daily_trading.log for details
    echo.
) else (
    echo.
    echo [ERROR] Daily trading workflow failed! Exit code: %ERRORLEVEL%
    echo Check logs\daily_trading.log for error details
    echo.
)

echo ======================================================================
echo DAILY TRADING RUN COMPLETE - %DATE% %TIME%
echo ======================================================================
echo.

REM Exit (don't pause - this will be run by task scheduler)
exit /b %ERRORLEVEL%

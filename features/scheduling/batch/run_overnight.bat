@echo off
REM Overnight Research Runner for Proteus Trading System
REM Run this before going to sleep to prepare Phase 2 data

echo ============================================================
echo PROTEUS OVERNIGHT RESEARCH
echo ============================================================
echo Starting at: %date% %time%
echo.

REM Create logs directory
if not exist logs\overnight mkdir logs\overnight

echo [1/3] Starting Position Sizing Research...
start /b cmd /c "cd /d C:\Users\javie\Documents\GitHub\proteus && python common/research/position_sizing_research.py > logs\overnight\position_sizing_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log 2>&1"

echo [2/3] Starting Extended Backtest (if exists)...
if exist src\research\extended_backtest.py (
    start /b cmd /c "cd /d C:\Users\javie\Documents\GitHub\proteus && python common/research/extended_backtest.py > logs\overnight\extended_backtest_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log 2>&1"
) else (
    echo     Skipping - extended_backtest.py not found
)

echo [3/3] Starting Data Refresh (if exists)...
if exist scripts\overnight_data_refresh.py (
    start /b cmd /c "cd /d C:\Users\javie\Documents\GitHub\proteus && python scripts/overnight_data_refresh.py > logs\overnight\data_refresh_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log 2>&1"
) else (
    echo     Skipping - overnight_data_refresh.py not found
)

echo.
echo ============================================================
echo All overnight tasks started in background!
echo.
echo Check results in the morning:
echo   - data/research/position_sizing_analysis.json
echo   - logs/overnight/*.log
echo.
echo You can close this window now.
echo ============================================================
pause

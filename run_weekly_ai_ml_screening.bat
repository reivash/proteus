@echo off
REM ============================================================================
REM Weekly AI/ML IPO Screening - Automated Task
REM ============================================================================
REM
REM This script runs the comprehensive AI/ML IPO analysis weekly
REM and saves timestamped reports for tracking changes over time.
REM
REM Recommended Schedule: Every Monday at 9:00 AM
REM ============================================================================

echo.
echo ============================================================================
echo WEEKLY AI/ML IPO SCREENING
echo ============================================================================
echo.
echo Date: %DATE%
echo Time: %TIME%
echo.

REM Navigate to project directory
cd /d "%~dp0"

REM Create timestamp for logs
set TIMESTAMP=%DATE:~-4%%DATE:~-10,2%%DATE:~-7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

REM Create weekly logs directory if it doesn't exist
if not exist "logs\weekly_screening" mkdir logs\weekly_screening

REM Run the analysis and capture output
echo Running comprehensive AI/ML IPO analysis...
echo.

python analyze_ai_ml_ipos_expanded.py > logs\weekly_screening\screening_%TIMESTAMP%.log 2>&1

REM Check if successful
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================================
    echo SUCCESS: Weekly screening complete!
    echo ============================================================================
    echo.
    echo Results saved to:
    echo   - Report: results\ai_ml_expanded_analysis_*.txt
    echo   - Log: logs\weekly_screening\screening_%TIMESTAMP%.log
    echo.
    echo Top candidates can be found in CURRENT_STATE.md
    echo.
) else (
    echo.
    echo ============================================================================
    echo ERROR: Screening failed with error code %ERRORLEVEL%
    echo ============================================================================
    echo.
    echo Check log file: logs\weekly_screening\screening_%TIMESTAMP%.log
    echo.
)

REM Optional: Send email notification (if email system is set up)
REM python send_weekly_screening_summary.py

echo.
echo Press any key to close...
pause >nul

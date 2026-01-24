@echo off
echo ======================================================================
echo OVERNIGHT REGIME OPTIMIZATION
echo Started: %date% %time%
echo ======================================================================
echo.

cd /d %~dp0

echo Running overnight regime optimization pipeline...
echo This will take 1-2 hours.
echo.
echo Progress will be logged to overnight_regime_log.txt
echo Final report: OVERNIGHT_REGIME_RESULTS.txt
echo.

python overnight_regime_optimization.py

echo.
echo ======================================================================
echo COMPLETE - Check OVERNIGHT_REGIME_RESULTS.txt
echo ======================================================================
pause

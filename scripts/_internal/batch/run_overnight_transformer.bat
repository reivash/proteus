@echo off
echo ======================================================================
echo OVERNIGHT TRANSFORMER TRAINING
echo Started: %date% %time%
echo ======================================================================
echo.

cd /d %~dp0

echo Running overnight training pipeline...
echo This will take 2-3 hours.
echo.
echo Progress will be logged to overnight_training_log.txt
echo Final report: OVERNIGHT_TRANSFORMER_RESULTS.txt
echo.

python overnight_transformer_training.py

echo.
echo ======================================================================
echo COMPLETE - Check OVERNIGHT_TRANSFORMER_RESULTS.txt
echo ======================================================================
pause

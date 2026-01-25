@echo off
REM Remove all Proteus scheduled tasks
REM Run this script as Administrator

echo ============================================================
echo REMOVING PROTEUS SCHEDULED TASKS
echo ============================================================
echo.

echo Removing Proteus Daily Summary...
schtasks /delete /tn "Proteus Daily Summary" /f 2>nul
if %ERRORLEVEL% EQU 0 (echo   Removed) else (echo   Not found or already removed)

echo Removing Proteus Bear Monitor...
schtasks /delete /tn "Proteus Bear Monitor" /f 2>nul
if %ERRORLEVEL% EQU 0 (echo   Removed) else (echo   Not found or already removed)

echo Removing Proteus Morning Scan...
schtasks /delete /tn "Proteus Morning Scan" /f 2>nul
if %ERRORLEVEL% EQU 0 (echo   Removed) else (echo   Not found or already removed)

echo.
echo All Proteus scheduled tasks have been removed.
echo.

pause

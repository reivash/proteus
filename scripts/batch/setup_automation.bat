@echo off
REM ============================================================
REM Proteus Automation Setup
REM Run as Administrator to create scheduled tasks
REM ============================================================

echo ============================================================
echo PROTEUS AUTOMATION SETUP
echo ============================================================
echo.

set PROTEUS_DIR=%~dp0
echo Project Directory: %PROTEUS_DIR%
echo.

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Please run this script as Administrator!
    echo Right-click and select "Run as administrator"
    pause
    exit /b 1
)

echo Creating scheduled tasks...
echo.

REM 1. Morning Report at 8:00 AM (weekdays only)
echo [1/4] Creating Morning Report task (8:00 AM weekdays)...
schtasks /create /tn "Proteus Morning Report" /tr "python %PROTEUS_DIR%send_morning_report.py" /sc weekly /d MON,TUE,WED,THU,FRI /st 08:00 /f
if %errorLevel% equ 0 (echo       SUCCESS) else (echo       FAILED)

REM 2. Market Open Scan at 9:35 AM (weekdays only)
echo [2/4] Creating Market Open Scan task (9:35 AM weekdays)...
schtasks /create /tn "Proteus Market Open" /tr "%PROTEUS_DIR%run_daily_scan.bat" /sc weekly /d MON,TUE,WED,THU,FRI /st 09:35 /f
if %errorLevel% equ 0 (echo       SUCCESS) else (echo       FAILED)

REM 3. Market Close Scan at 4:05 PM (weekdays only)
echo [3/4] Creating Market Close Scan task (4:05 PM weekdays)...
schtasks /create /tn "Proteus Market Close" /tr "%PROTEUS_DIR%run_daily_scan.bat" /sc weekly /d MON,TUE,WED,THU,FRI /st 16:05 /f
if %errorLevel% equ 0 (echo       SUCCESS) else (echo       FAILED)

REM 4. Overnight Research at 10:00 PM (daily)
echo [4/4] Creating Overnight Research task (10:00 PM daily)...
schtasks /create /tn "Proteus Overnight" /tr "%PROTEUS_DIR%run_overnight.bat" /sc daily /st 22:00 /f
if %errorLevel% equ 0 (echo       SUCCESS) else (echo       FAILED)

echo.
echo ============================================================
echo VERIFICATION
echo ============================================================
echo.
schtasks /query /fo TABLE | findstr "Proteus"

echo.
echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo Scheduled Tasks Created:
echo   - Proteus Morning Report  : 8:00 AM (Mon-Fri)
echo   - Proteus Market Open     : 9:35 AM (Mon-Fri)
echo   - Proteus Market Close    : 4:05 PM (Mon-Fri)
echo   - Proteus Overnight       : 10:00 PM (Daily)
echo.
echo To disable all tasks:
echo   schtasks /change /tn "Proteus Morning Report" /disable
echo   schtasks /change /tn "Proteus Market Open" /disable
echo   schtasks /change /tn "Proteus Market Close" /disable
echo   schtasks /change /tn "Proteus Overnight" /disable
echo.
echo To delete all tasks:
echo   schtasks /delete /tn "Proteus Morning Report" /f
echo   schtasks /delete /tn "Proteus Market Open" /f
echo   schtasks /delete /tn "Proteus Market Close" /f
echo   schtasks /delete /tn "Proteus Overnight" /f
echo.
pause

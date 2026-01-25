@echo off
REM Schedule the Proteus Morning Report to run at 8:00 AM daily

REM Create the scheduled task
schtasks /create /tn "Proteus Morning Report" /tr "python C:\Users\javie\Documents\GitHub\proteus\send_morning_report.py" /sc daily /st 08:00 /f

echo.
echo Scheduled task created: "Proteus Morning Report"
echo Will run daily at 8:00 AM
echo.
echo To verify: schtasks /query /tn "Proteus Morning Report"
echo To delete: schtasks /delete /tn "Proteus Morning Report" /f
echo.
pause

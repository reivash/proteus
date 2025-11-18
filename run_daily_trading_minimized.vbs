' Proteus Daily Trading Automation - Run Minimized
' This executes the daily trading workflow without showing a console window
' Perfect for Windows Task Scheduler automation
' Runs after market close to process the day's signals

Set WshShell = CreateObject("WScript.Shell")

' Run daily trading workflow in background (no window)
WshShell.Run "cmd /c """ & CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & "\run_daily_trading.bat""", 0, False

Set WshShell = Nothing

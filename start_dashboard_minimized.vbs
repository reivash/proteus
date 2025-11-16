' Proteus Trading Dashboard - Start Minimized
' This starts the dashboard without showing a console window
' Perfect for running on Windows startup

Set WshShell = CreateObject("WScript.Shell")

' Start dashboard in background
WshShell.Run "cmd /c """ & CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & "\start_dashboard.bat""", 0, False

' Wait 3 seconds for server to start
WScript.Sleep 3000

' Open browser to dashboard
WshShell.Run "http://localhost:5000", 1, False

Set WshShell = Nothing

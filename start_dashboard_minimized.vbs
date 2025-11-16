' Proteus Trading Dashboard - Start Minimized
' This starts the dashboard without showing a console window
' Perfect for running on Windows startup
' The .bat file handles opening the browser automatically

Set WshShell = CreateObject("WScript.Shell")

' Start dashboard in background (the .bat file will open the browser)
WshShell.Run "cmd /c """ & CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & "\start_dashboard.bat""", 0, False

Set WshShell = Nothing

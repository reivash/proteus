' Proteus Trading Dashboard - Start Minimized
' This starts the dashboard without showing a console window
' Perfect for running on Windows startup

Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c """ & CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & "\start_dashboard.bat""", 0, False
Set WshShell = Nothing

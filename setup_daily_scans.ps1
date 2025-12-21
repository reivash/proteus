# ============================================================
# Proteus Trading System - Windows Task Scheduler Setup
#
# This script creates scheduled tasks for daily trading scans:
# - Morning scan at 9:35 AM ET (5 min after market open)
# - Afternoon scan at 4:05 PM ET (5 min after market close)
#
# Run as Administrator:
#   powershell -ExecutionPolicy Bypass -File setup_daily_scans.ps1
# ============================================================

$ErrorActionPreference = "Stop"

# Configuration
$TaskNameMorning = "Proteus Morning Scan"
$TaskNameAfternoon = "Proteus Afternoon Scan"
$ProjectPath = $PSScriptRoot
$PythonPath = "python"  # Or full path to python.exe
$ScriptPath = Join-Path $ProjectPath "src\trading\scheduled_scanner.py"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Proteus Trading System - Task Scheduler Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project Path: $ProjectPath"
Write-Host "Script Path: $ScriptPath"
Write-Host ""

# Check if running as administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "WARNING: Not running as Administrator. Some features may not work." -ForegroundColor Yellow
    Write-Host "For full functionality, run PowerShell as Administrator." -ForegroundColor Yellow
    Write-Host ""
}

# Create action for the tasks
$Action = New-ScheduledTaskAction -Execute $PythonPath `
    -Argument "--run-once `"$ScriptPath`"" `
    -WorkingDirectory $ProjectPath

# Create settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -WakeToRun

# Morning scan trigger (9:35 AM, weekdays only)
Write-Host "[1/2] Creating Morning Scan task (9:35 AM ET, weekdays)..." -ForegroundColor Green
$TriggerMorning = New-ScheduledTaskTrigger -Daily -At "09:35"

try {
    # Remove existing task if present
    Unregister-ScheduledTask -TaskName $TaskNameMorning -Confirm:$false -ErrorAction SilentlyContinue

    # Create new task
    Register-ScheduledTask `
        -TaskName $TaskNameMorning `
        -Action $Action `
        -Trigger $TriggerMorning `
        -Settings $Settings `
        -Description "Proteus Trading System - Morning market scan (5 min after open)" `
        -Force | Out-Null

    Write-Host "   SUCCESS: Morning scan task created" -ForegroundColor Green
} catch {
    Write-Host "   ERROR: Failed to create morning scan task: $_" -ForegroundColor Red
}

# Afternoon scan trigger (4:05 PM, weekdays only)
Write-Host "[2/2] Creating Afternoon Scan task (4:05 PM ET, weekdays)..." -ForegroundColor Green
$TriggerAfternoon = New-ScheduledTaskTrigger -Daily -At "16:05"

try {
    # Remove existing task if present
    Unregister-ScheduledTask -TaskName $TaskNameAfternoon -Confirm:$false -ErrorAction SilentlyContinue

    # Create new task
    Register-ScheduledTask `
        -TaskName $TaskNameAfternoon `
        -Action $Action `
        -Trigger $TriggerAfternoon `
        -Settings $Settings `
        -Description "Proteus Trading System - Afternoon market scan (5 min after close)" `
        -Force | Out-Null

    Write-Host "   SUCCESS: Afternoon scan task created" -ForegroundColor Green
} catch {
    Write-Host "   ERROR: Failed to create afternoon scan task: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Scheduled tasks have been created:" -ForegroundColor White
Write-Host "  - $TaskNameMorning : Daily at 9:35 AM" -ForegroundColor White
Write-Host "  - $TaskNameAfternoon : Daily at 4:05 PM" -ForegroundColor White
Write-Host ""
Write-Host "To view tasks: Open Task Scheduler or run:" -ForegroundColor Gray
Write-Host "  Get-ScheduledTask -TaskName 'Proteus*'" -ForegroundColor Gray
Write-Host ""
Write-Host "To run manually:" -ForegroundColor Gray
Write-Host "  Start-ScheduledTask -TaskName '$TaskNameMorning'" -ForegroundColor Gray
Write-Host ""
Write-Host "To remove tasks:" -ForegroundColor Gray
Write-Host "  Unregister-ScheduledTask -TaskName 'Proteus Morning Scan'" -ForegroundColor Gray
Write-Host "  Unregister-ScheduledTask -TaskName 'Proteus Afternoon Scan'" -ForegroundColor Gray
Write-Host ""

# List current Proteus tasks
Write-Host "Current Proteus scheduled tasks:" -ForegroundColor Yellow
Get-ScheduledTask -TaskName "Proteus*" -ErrorAction SilentlyContinue | Format-Table TaskName, State, TaskPath -AutoSize

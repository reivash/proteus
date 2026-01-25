# Proteus Daily Trading Automation Setup

## Overview
Automate the Proteus trading workflow to run every market day at 4:15 PM ET (after market close).

## Files Created
- `run_daily_trading.bat` - Batch script that executes daily trading workflow
- `run_daily_trading_minimized.vbs` - VBScript to run without console window
- `logs/daily_trading.log` - Automated log file (created on first run)

## What It Does
1. **Checks market regime** (SIDEWAYS, BULL, BEAR)
2. **Scans 54 stocks** for panic sell signals (Z-score < -2.0, RSI < 35)
3. **Processes entry signals** using limit orders (0.3% below ask)
4. **Checks exit conditions** (time-decay: 2% day 0, 1.5% day 1, 1% day 2+)
5. **Updates performance** (tracks P&L, win rate, equity curve)
6. **Sends email** with trade details (if signals triggered)

## Windows Task Scheduler Setup

### Method 1: Quick Setup (GUI)
1. Press `Win + R`, type `taskschd.msc`, press Enter
2. Click **Create Basic Task**
3. Configure:
   - **Name:** `Proteus Daily Trading`
   - **Description:** `Automated daily trading workflow after market close`
   - **Trigger:** Daily at 4:15 PM
   - **Action:** Start a program
   - **Program/script:** `wscript.exe`
   - **Arguments:** `"C:\Users\javie\Documents\GitHub\proteus\run_daily_trading_minimized.vbs"`
   - **Start in:** `C:\Users\javie\Documents\GitHub\proteus`
4. Check **"Run whether user is logged on or not"**
5. Click **Finish**

### Method 2: PowerShell Script (Advanced)
```powershell
# Run as Administrator
$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument '"C:\Users\javie\Documents\GitHub\proteus\run_daily_trading_minimized.vbs"' -WorkingDirectory "C:\Users\javie\Documents\GitHub\proteus"
$trigger = New-ScheduledTaskTrigger -Daily -At 4:15PM
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -TaskName "Proteus Daily Trading" -Action $action -Trigger $trigger -Principal $principal -Settings $settings
```

### Method 3: Command Line (schtasks)
```batch
schtasks /create /tn "Proteus Daily Trading" /tr "wscript.exe \"C:\Users\javie\Documents\GitHub\proteus\run_daily_trading_minimized.vbs\"" /sc daily /st 16:15 /f
```

## Schedule Timing

### Recommended: 4:15 PM ET (Market Close + 15 min)
- **Why:** Market closes at 4:00 PM ET, gives 15 min for final prices to settle
- **Pro:** Captures same-day panic sell signals with closing prices
- **Con:** Must convert to local time if not in ET timezone

### Alternative: 6:00 PM ET (After Hours)
- **Why:** More conservative, ensures all data available
- **Pro:** No rush, all settlement complete
- **Con:** Slight delay in signal detection

### For Eastern Time (ET):
- Schedule: **4:15 PM**

### For Central Time (CT):
- Schedule: **3:15 PM** (ET - 1 hour)

### For Pacific Time (PT):
- Schedule: **1:15 PM** (ET - 3 hours)

## Manual Testing

### Test the batch script (with console):
```batch
cd C:\Users\javie\Documents\GitHub\proteus
run_daily_trading.bat
```

### Test the VBS script (minimized):
```batch
cd C:\Users\javie\Documents\GitHub\proteus
run_daily_trading_minimized.vbs
```

### Test the scheduled task:
```batch
schtasks /run /tn "Proteus Daily Trading"
```

## Monitoring

### Check if task is running:
```batch
schtasks /query /tn "Proteus Daily Trading"
```

### View execution history:
1. Open Task Scheduler (`taskschd.msc`)
2. Navigate to **Task Scheduler Library**
3. Find **"Proteus Daily Trading"**
4. Click **History** tab

### Check logs:
```batch
type logs\daily_trading.log
```

## Skip Weekends/Holidays

The script will run daily, but you can configure it to skip weekends:

### In Task Scheduler:
1. Right-click task → **Properties**
2. Go to **Conditions** tab
3. Check **"Start only if the following network connection is available"** → Any connection
4. Go to **Settings** tab
5. Check **"If the task fails, restart every: 1 minute"**
6. Set **"Attempt to restart up to: 3 times"**

### Advanced: Market Calendar Integration
Create `common/utils/market_calendar.py` to check if today is a trading day:
```python
import pandas_market_calendars as mcal

def is_trading_day():
    nyse = mcal.get_calendar('NYSE')
    today = pd.Timestamp.now(tz='America/New_York').normalize()
    schedule = nyse.schedule(start_date=today, end_date=today)
    return len(schedule) > 0
```

Then modify `daily_runner.py` to exit early if not a trading day.

## Troubleshooting

### Task doesn't run:
- Check task is **Enabled** in Task Scheduler
- Verify **trigger is correct** (time, frequency)
- Check **"Run whether user is logged on or not"** is checked
- Ensure **no password required** or password is stored

### Script runs but no trades:
- Check `logs/daily_trading.log` for errors
- Verify market was open (not weekend/holiday)
- Check if signals were generated (might be no panic sells today)

### Email not received:
- Check SPAM folder at reivaj.cabero@gmail.com
- Verify SendGrid API key in `email_config.json`
- Check SendGrid activity: https://app.sendgrid.com/email_activity

### Python environment issues:
- Task runs in background, ensure Python is in PATH
- Or use absolute path: `C:\Users\javie\AppData\Local\Programs\Python\Python311\python.exe`

## Disable Automation

### Temporarily disable:
```batch
schtasks /change /tn "Proteus Daily Trading" /disable
```

### Re-enable:
```batch
schtasks /change /tn "Proteus Daily Trading" /enable
```

### Permanently remove:
```batch
schtasks /delete /tn "Proteus Daily Trading" /f
```

## Performance Tracking

All trades are logged to:
- `data/paper_trading/performance_history.csv` - Daily equity snapshots
- `data/paper_trading/trade_history.csv` - Individual trade records
- `logs/daily_trading.log` - Execution logs

View performance via web dashboard:
```batch
start_dashboard_minimized.vbs
```

Then open: http://localhost:5000

## Production Checklist

Before enabling daily automation:

- [ ] Verify `email_config.json` is configured
- [ ] Test batch script manually: `run_daily_trading.bat`
- [ ] Verify email notifications working
- [ ] Check performance tracking files created
- [ ] Set up Task Scheduler with correct time (4:15 PM ET)
- [ ] Test scheduled task: `schtasks /run /tn "Proteus Daily Trading"`
- [ ] Monitor first 3 days of automated runs
- [ ] Review `logs/daily_trading.log` daily
- [ ] Verify position sizing is appropriate (default: $10k per position)

## Configuration

Edit `common/trading/daily_runner.py` to adjust:
- `initial_capital` - Starting capital (default: $100,000)
- `position_size` - Capital per position (default: 0.1 = 10%)
- `max_positions` - Max concurrent positions (default: 5)
- `profit_target` - Exit profit % (default: 2.0%)
- `stop_loss` - Exit loss % (default: -2.0%)
- `max_hold_days` - Max days to hold (default: 2)

## Support

For issues or questions:
- Check logs: `logs/daily_trading.log`
- Check Owl nudges: `logs/owl_monitor.log`
- Email notifications: reivaj.cabero@gmail.com
- Dashboard: http://localhost:5000

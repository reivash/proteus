# Automated Daily Workflow Setup

Set up Proteus to run automatically and send you daily notifications - no manual intervention needed!

## What Gets Automated

Once set up, the system will:
- ✅ Start automatically when your computer boots
- ✅ Scan for signals at **9:30 AM, 1:30 PM, 5:30 PM** daily
- ✅ Send Windows notifications after each scan
- ✅ Send email notifications (once you configure Gmail)
- ✅ Track performance automatically
- ✅ Run in background - no windows popping up

## Quick Setup (3 Options)

### Option 1: Windows Startup Folder (Simplest - 30 seconds)

**Perfect if:** You want it to start every time you log in to Windows.

1. Press `Win + R`
2. Type: `shell:startup` and press Enter
3. Right-click in the folder → New → Shortcut
4. Browse to: `C:\Users\javie\Documents\GitHub\proteus\start_dashboard_minimized.vbs`
5. Name it: "Proteus Trading Dashboard"
6. Click Finish

**That's it!** Next time you log in, dashboard auto-starts in background.

### Option 2: Run Manually (When Needed)

Just double-click: `start_dashboard.bat`

The dashboard runs until you close the window. Scans happen at 9:30 AM, 1:30 PM, 5:30 PM.

### Option 3: Windows Task Scheduler (Advanced - Most Control)

**Perfect if:** You want to start at a specific time, or run even when logged out.

#### Create Scheduled Task

1. Press `Win + R`, type `taskschd.msc`, press Enter
2. Click "Create Basic Task"
3. Name: `Proteus Trading Dashboard`
4. Description: `Automated mean reversion trading signal scanner`
5. Trigger: Choose when to start
   - **"When I log on"** = Auto-start on login
   - **"Daily"** = Start at specific time (e.g., 9:00 AM)
6. Action: **Start a program**
7. Program: Browse to `C:\Users\javie\Documents\GitHub\proteus\start_dashboard_minimized.vbs`
8. Finish

#### Advanced Settings (Optional)

- Right-click task → Properties
- **General tab:**
  - ✓ Run whether user is logged on or not
  - ✓ Run with highest privileges
- **Conditions tab:**
  - ✓ Start only if computer is on AC power (uncheck for laptop)
  - ✓ Wake computer to run this task
- **Settings tab:**
  - ✓ Allow task to be run on demand
  - ✓ If task fails, restart every: 10 minutes (3 attempts)

## How Daily Workflow Works

### Timeline

**9:30 AM EST** - Market Open Scan
- Dashboard scans all 10 stocks
- Checks for panic sell signals
- Sends Windows notification
- Sends email (if configured)

**1:30 PM EST** - Mid-Day Scan
- Re-scans for intraday opportunities
- Checks if any new signals appeared
- Sends notifications

**5:30 PM EST** - End-of-Day Scan
- Final scan after market close
- Reviews full day's price action
- Sends daily summary

### What You Receive

**Windows Notification** (always, if installed):
```
Proteus Daily Scan
No signals detected
OR
🚨 BUY Signal: TSLA
Price: $156.80 | Z=-2.10 | RSI=29.9
```

**Email Notification** (once Gmail configured):
- Subject: "📊 Proteus Daily Scan - No Signals" or "🚨 Proteus Alert - BUY Signal: TSLA"
- Beautiful HTML email with:
  - Market regime status
  - Scan results for all stocks
  - Signal details (if any)
  - Performance summary
  - Next steps

## Email Configuration (Optional but Recommended)

The workflow is already scheduled, but emails won't send until you configure Gmail.

### Quick Email Setup (5 minutes)

1. **Get Gmail App Password**
   - Go to: https://myaccount.google.com/apppasswords
   - Sign in to your Google account
   - Create app password for "Proteus Trading"
   - Copy the 16-character password

2. **Edit `email_config.json`** (in project root)
   ```json
   {
     "enabled": true,
     "recipient_email": "reivaj.cabero@gmail.com",
     "sender_email": "YOUR_GMAIL@gmail.com",
     "sender_password": "YOUR_16_CHAR_APP_PASSWORD",
     "smtp_server": "smtp.gmail.com",
     "smtp_port": 587
   }
   ```

3. **Test it**
   ```bash
   python send_test_email.py
   ```

That's it! Emails will now send automatically at scan times.

## Verify It's Working

### Check if Dashboard is Running

1. Open browser: http://localhost:5000
2. You should see the dashboard
3. Check "Last Scan" time

### Check Task Manager (Windows)

1. Press `Ctrl + Shift + Esc`
2. Go to Details tab
3. Look for `python.exe` running `app.py`

### Check Logs

The dashboard prints to console. If you started with `start_dashboard.bat` (not minimized), you'll see:
```
======================================================================
PROTEUS TRADING DASHBOARD
======================================================================

[SCHEDULER] Started - scanning at 9:30 AM, 1:30 PM, 5:30 PM daily

Notification Methods:
  ✓ Windows Notifications: Enabled (NO PASSWORD NEEDED!)
  - Email Notifications: Not configured (or ✓ Enabled if you set it up)

Dashboard ready!
Open in browser: http://localhost:5000
```

## What If My Computer is Off?

**Important:** The dashboard only runs when your computer is on.

**Solutions:**

1. **Keep computer on during market hours (9:30 AM - 5:30 PM EST)**
   - Put it to sleep outside market hours
   - Wake it up before 9:30 AM

2. **Use Task Scheduler to wake computer**
   - In Task settings: ✓ "Wake computer to run this task"
   - Computer will auto-wake at scheduled times

3. **Cloud deployment (advanced)**
   - See `PRODUCTION_DEPLOYMENT_GUIDE.md` for AWS/cloud setup
   - Runs 24/7 regardless of your computer

## Stopping the Dashboard

### If Running Manually
- Close the console window
- Or press `Ctrl + C` in the terminal

### If Running from Startup/Task Scheduler
1. Open Task Manager (`Ctrl + Shift + Esc`)
2. Find `python.exe`
3. Right-click → End Task

### Disable Auto-Start
- **Startup folder:** Delete the shortcut from `shell:startup`
- **Task Scheduler:** Right-click task → Disable (or Delete)

## Troubleshooting

### Dashboard not starting automatically

**Check:**
1. Is the VBS file in startup folder? (`Win + R` → `shell:startup`)
2. Is the path correct in the shortcut?
3. Is Python in your PATH? (Try `python --version` in cmd)

**Fix:**
- Re-create the startup shortcut
- Or use Task Scheduler instead (more reliable)

### No notifications received

**Windows notifications:**
```bash
pip install windows-toasts
```

**Email notifications:**
- Check `email_config.json` has correct Gmail credentials
- Run `python send_test_email.py` to test
- See `EMAIL_NOTIFICATIONS_SETUP.md` for troubleshooting

### Dashboard crashes or stops

**Check logs:**
- If minimized, check: `data/web_dashboard/scan_status.json`
- Run manually with `start_dashboard.bat` to see errors

**Common fixes:**
- Restart the dashboard
- Check internet connection (needed for market data)
- Update dependencies: `pip install -r requirements.txt --upgrade`

### Scans not happening at scheduled times

**Verify:**
1. Dashboard is running (check http://localhost:5000)
2. Computer is on at scan times
3. System clock is correct

**Check scheduler status:**
Dashboard prints: `[SCHEDULER] Started - scanning at 9:30 AM, 1:30 PM, 5:30 PM daily`

## Testing the Automation

### Test Windows Startup

1. Add VBS file to startup folder (Option 1 above)
2. Restart your computer
3. After login, check: http://localhost:5000
4. Dashboard should be running!

### Test Manual Scan

1. Go to dashboard: http://localhost:5000
2. Click "Scan Now" button
3. Within 30 seconds, you should receive Windows notification
4. If email configured, check your inbox

### Test Email

```bash
python send_daily_report.py
```

This sends today's actual analysis to reivaj.cabero@gmail.com (once email configured).

## Current Status

**Already Configured:**
- ✅ Dashboard scheduling (9:30 AM, 1:30 PM, 5:30 PM)
- ✅ Windows notifications (working now!)
- ✅ Recipient email: reivaj.cabero@gmail.com
- ⏳ Sender Gmail: **You need to configure this**

**To Complete Full Automation:**
1. Choose startup method (Option 1, 2, or 3 above)
2. Configure Gmail sender (optional, for email notifications)
3. Test it once
4. Done! It runs automatically from now on

## Recommended Setup

For you, I recommend:

**1. Use Windows Startup (Option 1)**
- Simplest, no Task Scheduler needed
- Auto-starts when you log in
- Just copy VBS file to startup folder

**2. Keep Email Simple**
- Start with Windows notifications only (already working!)
- Add Gmail later if you want phone notifications
- Or use SendGrid API (simpler than Gmail)

**3. Test Run**
```bash
# Test right now
start_dashboard.bat

# Open browser
http://localhost:5000

# Click "Scan Now"
# You should get Windows notification!
```

---

**Summary:**
- Copy `start_dashboard_minimized.vbs` to startup folder
- Configure Gmail in `email_config.json` (when ready)
- Dashboard auto-runs daily at 9:30 AM, 1:30 PM, 5:30 PM
- You get Windows notifications + email (once configured)
- Zero manual work needed!

**Files:**
- `start_dashboard.bat` - Run manually with window
- `start_dashboard_minimized.vbs` - Auto-start, no window
- `send_daily_report.py` - Test email sending
- `email_config.json` - Configure Gmail here

**Next:** Copy VBS file to startup folder, done!

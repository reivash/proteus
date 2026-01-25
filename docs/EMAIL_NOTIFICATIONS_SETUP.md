# Email Notifications Setup Guide

## Overview

Proteus can send automated email notifications with:
- Daily scan results at market hours (9:35 AM, 1:35 PM, 5:35 PM)
- Buy signal alerts when panic sell opportunities are detected
- Performance summaries with win rate and returns
- Market regime status

## Quick Setup (5 minutes)

### Step 1: Get Gmail App Password

Since Google requires 2-factor authentication, you need an "App Password" to send emails:

1. Go to: https://myaccount.google.com/apppasswords
2. Sign in to your Google account
3. In the "App name" field, type: **Proteus Trading**
4. Click **Create**
5. Copy the 16-character password (format: `xxxx xxxx xxxx xxxx`)
6. **IMPORTANT:** Save this password - you won't see it again!

### Step 2: Configure Email Settings

Edit `email_config.json` in the project root:

```json
{
  "enabled": true,
  "recipient_email": "reivaj.cabero@gmail.com",
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "sender_email": "YOUR_GMAIL@gmail.com",
  "sender_password": "YOUR_16_CHAR_APP_PASSWORD",
  "send_on_signals_only": false,
  "send_daily_summary": true,
  "notification_times": ["09:35", "13:35", "17:35"]
}
```

Replace:
- `YOUR_GMAIL@gmail.com` with your Gmail address
- `YOUR_16_CHAR_APP_PASSWORD` with the app password from Step 1

### Step 3: Test Email Configuration

Run the test script:

```bash
python common/notifications/email_notifier.py
```

You should see:
```
Sending test email...
[EMAIL] Sent to reivaj.cabero@gmail.com: 🧪 Proteus Trading Dashboard - Test Email
✓ Test email sent successfully to reivaj.cabero@gmail.com
```

Check your inbox for the test email!

## Configuration Options

### `enabled` (boolean)
- `true`: Email notifications are enabled
- `false`: Disable all email notifications

### `recipient_email` (string)
- Email address to receive notifications
- Already set to: `reivaj.cabero@gmail.com`

### `smtp_server` and `smtp_port` (string, number)
- SMTP server settings
- Default: Gmail (`smtp.gmail.com`, port `587`)
- For other providers:
  - Outlook: `smtp.office365.com`, port `587`
  - Yahoo: `smtp.mail.yahoo.com`, port `587`

### `sender_email` and `sender_password` (string)
- Your email credentials for sending
- **MUST use App Password**, not your regular password

### `send_on_signals_only` (boolean)
- `true`: Only send emails when signals are detected
- `false`: Send emails after every scan (including "no signals" updates)
- **Recommended:** `false` for daily confirmation

### `send_daily_summary` (boolean)
- `true`: Include performance summary in emails
- `false`: Only send signal information

### `notification_times` (array)
- Times to send notifications (24-hour format)
- Default: `["09:35", "13:35", "17:35"]` (5 minutes after scans)

## Email Content

### Subject Lines
- **No signals:** "📊 Proteus Daily Scan - No Signals"
- **1 signal:** "🚨 Proteus Alert - BUY Signal: TSLA"
- **Multiple signals:** "🚨 Proteus Alert - 3 BUY Signals!"

### Email Body Includes
- **Scan Status:** Current market regime and scan result
- **Active Signals:** Ticker, price, Z-score, RSI, expected return
- **Buy Recommendations:** Position size and entry price
- **Performance Summary:** Total trades, win rate, returns
- **Next Steps:** Direct link to dashboard and action items

## Automated Integration

The email notifier is already integrated with:

### Web Dashboard (`common/web/app.py`)
Automatically sends emails after each scheduled scan at 9:30 AM, 1:30 PM, 5:30 PM.

### Daily Runner (`common/trading/daily_runner.py`)
Can be configured to send emails after paper trading workflow.

## Troubleshooting

### "Login failed" or "Username and Password not accepted"

**Problem:** Gmail is rejecting your credentials

**Solutions:**
1. Make sure you're using an **App Password**, not your regular Gmail password
2. Enable 2-factor authentication on your Google account first
3. Create a new app password at https://myaccount.google.com/apppasswords
4. Copy the password exactly (remove spaces if pasting)

### "Email config incomplete - sender_email not configured"

**Problem:** Configuration file has placeholder values

**Solution:** Edit `email_config.json` and replace:
- `YOUR_GMAIL_HERE@gmail.com` with your actual Gmail
- `YOUR_APP_PASSWORD_HERE` with your actual app password

### No emails received

**Possible causes:**
1. Check spam/junk folder
2. Verify recipient email is correct in `email_config.json`
3. Run test script to check configuration: `python common/notifications/email_notifier.py`
4. Check terminal output for error messages

### "SMTPServerDisconnected: Connection unexpectedly closed"

**Problem:** SMTP connection issues

**Solutions:**
1. Check your internet connection
2. Verify SMTP server and port are correct
3. Some networks block port 587 - try port 465 with SSL
4. Temporarily disable VPN if using one

## Security Notes

### Why is `email_config.json` gitignored?

The config file contains your email password, so it's excluded from Git to prevent accidental exposure.

### App Password Safety

- App passwords are safer than your main password
- They can be revoked at any time without affecting your Google account
- Each app has a separate password
- If compromised, revoke it and create a new one

### What data is sent?

Only trading signals and performance data from your local paper trading system. No personal information beyond email addresses.

## Alternative Email Providers

### Using Outlook/Hotmail

```json
{
  "smtp_server": "smtp.office365.com",
  "smtp_port": 587,
  "sender_email": "your-email@outlook.com",
  "sender_password": "your-outlook-password"
}
```

### Using Yahoo Mail

```json
{
  "smtp_server": "smtp.mail.yahoo.com",
  "smtp_port": 587,
  "sender_email": "your-email@yahoo.com",
  "sender_password": "your-yahoo-app-password"
}
```

**Note:** Yahoo also requires app passwords. Generate one at: https://login.yahoo.com/account/security

## Testing Email Templates

To see what emails will look like:

```bash
# Send test email
python common/notifications/email_notifier.py
```

This sends a formatted test email showing:
- Configuration confirmation
- What you'll receive in daily notifications
- Your current settings

## Disabling Email Notifications

To temporarily disable without removing configuration:

```json
{
  "enabled": false,
  ...
}
```

Or remove `email_config.json` entirely - the system will continue working without email notifications.

## Support

If you encounter issues:
1. Run the test script and check error messages
2. Verify Gmail app password is correct
3. Check that 2-factor authentication is enabled on your Google account
4. Review terminal output when web dashboard runs

---

**Setup Guide Version:** 1.0
**Last Updated:** 2025-11-14
**Compatible With:** Proteus Web Dashboard v1.0

# Notification Options - Choose What Works Best

Proteus supports three types of notifications. Pick the one that suits you!

## Quick Comparison

| Feature | Windows Toasts | Email (Gmail) | SendGrid |
|---------|---------------|---------------|----------|
| **Setup Time** | 30 seconds | 5 minutes | 3 minutes |
| **Passwords Needed** | ❌ NONE | ✅ App Password | ✅ API Key |
| **Works When Computer Off** | ❌ No | ✅ Yes | ✅ Yes |
| **Works on Phone** | ❌ No | ✅ Yes | ✅ Yes |
| **Cost** | Free | Free | Free (100/day) |
| **Best For** | Quick local alerts | Full email archive | Simpler than Gmail |

## Option 1: Windows Notifications (RECOMMENDED - Simplest!)

**Perfect if:** You keep your computer on during market hours and just want simple popup notifications.

**Setup:**
```bash
pip install windows-toasts
```

That's it! NO passwords, NO configuration, NO setup files.

**What you get:**
- Popup notifications on your Windows desktop
- Click to open dashboard
- Zero configuration required

**Limitations:**
- Only works when computer is on
- Only shows on your computer (not phone)
- No email history

**Test it:**
```bash
python -c "from common.notifications.windows_notifier import send_test_windows_notification; send_test_windows_notification()"
```

## Option 2: Email via Gmail (Full Featured)

**Perfect if:** You want emails you can read anywhere (phone, tablet, web) with full history.

**Setup (5 minutes):**
1. Go to: https://myaccount.google.com/apppasswords
2. Create app password for "Proteus Trading"
3. Edit `email_config.json`:
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

**What you get:**
- Formatted HTML emails with signal details
- Read on any device (phone, tablet, computer)
- Email archive of all signals
- Performance summaries

**Test it:**
```bash
python send_test_email.py
```

See full guide: `docs/EMAIL_NOTIFICATIONS_SETUP.md`

## Option 3: SendGrid API (Simpler Than Gmail)

**Perfect if:** You want email notifications but don't want to deal with Gmail app passwords.

**Setup (3 minutes):**
1. Sign up: https://sendgrid.com (free tier: 100 emails/day)
2. Get API key: Settings > API Keys
3. Edit `email_config.json`:
```json
{
  "enabled": true,
  "recipient_email": "reivaj.cabero@gmail.com",
  "sendgrid_api_key": "YOUR_API_KEY_HERE",
  "sender_email": "proteus@yourdomain.com"
}
```

**Install:**
```bash
pip install sendgrid
```

**What you get:**
- Same email features as Gmail option
- Simpler setup (just one API key)
- More reliable delivery
- Better for high volume

**Test it:**
```bash
python -c "from common.notifications.sendgrid_notifier import SendGridNotifier; SendGridNotifier().send_test_email()"
```

## Can I Use Multiple Methods?

Yes! The dashboard will send to all configured methods:
- Windows toast + Email = You get both!
- All three at once = Why not!

## My Recommendation

**For you:** Start with **Windows Notifications** (30 seconds, zero passwords!)

```bash
pip install windows-toasts
```

Then later, if you want email alerts on your phone, add Gmail or SendGrid.

## Setup Instructions

### Windows Notifications (Easiest)
```bash
# 1. Install
pip install windows-toasts

# 2. Test
python -c "from common.notifications.windows_notifier import send_test_windows_notification; send_test_windows_notification()"

# 3. Run dashboard
python common/web/app.py
```

You'll see: `✓ Windows Notifications: Enabled (NO PASSWORD NEEDED!)`

### Email Notifications
See detailed guide: `docs/EMAIL_NOTIFICATIONS_SETUP.md`

### SendGrid Notifications
1. Sign up: https://sendgrid.com
2. Get API key
3. Add to `email_config.json`: `"sendgrid_api_key": "YOUR_KEY"`
4. Install: `pip install sendgrid`

## FAQ

**Q: Do I NEED notifications?**
A: No! The dashboard works fine without them. You can just check http://localhost:5000 manually.

**Q: Which is fastest to set up?**
A: Windows Notifications - literally just `pip install windows-toasts`. Done!

**Q: Which works on my phone?**
A: Email (Gmail or SendGrid). Windows notifications only show on your computer.

**Q: Can I disable notifications temporarily?**
A: Yes, just don't install the packages or set `"enabled": false` in config.

**Q: Why does email need a password?**
A: Email requires authentication to prevent spam. All email sending (SMTP, API, etc.) needs credentials.

**Q: Is SendGrid really free?**
A: Yes, 100 emails/day free forever. More than enough for 3 scans/day.

## Troubleshooting

### Windows Notifications Not Working
```bash
# Check if installed
pip list | findstr windows-toasts

# Reinstall if needed
pip install --upgrade windows-toasts

# Test
python -c "from common.notifications.windows_notifier import send_test_windows_notification; send_test_windows_notification()"
```

### Email Not Working
See troubleshooting in `docs/EMAIL_NOTIFICATIONS_SETUP.md`

### SendGrid Not Working
- Verify API key is correct
- Check SendGrid dashboard for error logs
- Make sure you verified your sender identity

---

**My Recommendation: Start with Windows Notifications**

Literally 30 seconds:
```bash
pip install windows-toasts
python common/web/app.py
```

No passwords, no configuration, no hassle!

Then add email later if you want alerts on your phone.

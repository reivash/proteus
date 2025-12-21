# Weekly AI/ML IPO Screening - Automation Setup

This guide explains how to set up automated weekly screening of AI/ML IPO companies using Windows Task Scheduler.

---

## 📋 Overview

The automated system will:
- Run comprehensive AI/ML IPO analysis every week
- Generate timestamped reports
- Track score changes over time
- Save logs for review

**Recommended Schedule:** Every Monday at 9:00 AM (after market open)

---

## 🚀 Quick Setup (Windows Task Scheduler)

### Step 1: Create the Scheduled Task

1. **Open Task Scheduler:**
   - Press `Win + R`
   - Type `taskschd.msc` and press Enter

2. **Create Basic Task:**
   - Click "Create Basic Task..." in the right panel
   - Name: `Weekly AI/ML IPO Screening`
   - Description: `Automated weekly analysis of AI/ML IPO companies for unicorn potential`
   - Click "Next"

3. **Set Trigger:**
   - Select "Weekly"
   - Click "Next"
   - Start date: Choose next Monday
   - Start time: `9:00:00 AM`
   - Recur every: `1 week`
   - Select: Monday
   - Click "Next"

4. **Set Action:**
   - Select "Start a program"
   - Click "Next"
   - Program/script: Browse to `run_weekly_ai_ml_screening.bat`
     - Full path: `C:\Users\javie\Documents\GitHub\proteus\run_weekly_ai_ml_screening.bat`
   - Start in: `C:\Users\javie\Documents\GitHub\proteus`
   - Click "Next"

5. **Finish:**
   - Check "Open the Properties dialog..."
   - Click "Finish"

6. **Configure Advanced Settings:**
   - In Properties dialog:
     - **General Tab:**
       - Check "Run whether user is logged on or not" (requires password)
       - Check "Run with highest privileges"
     - **Conditions Tab:**
       - Uncheck "Start the task only if the computer is on AC power"
     - **Settings Tab:**
       - Check "Run task as soon as possible after a scheduled start is missed"
       - Set "Stop the task if it runs longer than:" to `1 hour`
   - Click "OK"

---

## 📊 What Gets Generated

Every week, the system generates:

1. **Analysis Report** (`results/ai_ml_expanded_analysis_YYYYMMDD_HHMMSS.txt`)
   - Comprehensive AI/ML company analysis
   - Updated scores for all companies
   - Performance metrics since IPO

2. **Log File** (`logs/weekly_screening/screening_YYYYMMDD_HHMMSS.log`)
   - Detailed execution log
   - Any errors or warnings
   - Timestamp of run

3. **Updated State** (`CURRENT_STATE.md`)
   - Automatically shows latest analysis results
   - Updated next steps

---

## 🔍 Monitoring Changes Over Time

### Compare Weekly Reports

To track how companies evolve:

```bash
# View recent reports
dir results\ai_ml_expanded_analysis_*.txt /o-d

# Compare two reports (use a diff tool)
# Example: Compare Nov 1 vs Dec 1 reports
fc results\ai_ml_expanded_analysis_20251101_*.txt results\ai_ml_expanded_analysis_20251201_*.txt
```

### Key Metrics to Track

Monitor these changes week-over-week:

1. **Score Changes**
   - Companies moving between tiers (e.g., Moderate → Strong)
   - Significant score increases/decreases (+/- 5 points)

2. **Revenue Growth Acceleration**
   - YoY growth rate changes
   - Inflection point detection

3. **Stock Performance**
   - Total return since IPO
   - Recent momentum (compare last 4 weeks)

4. **New Entries**
   - Newly public AI/ML companies
   - Companies crossing market cap thresholds

---

## 🛠️ Manual Execution

To run the screening manually anytime:

```bash
cd C:\Users\javie\Documents\GitHub\proteus
run_weekly_ai_ml_screening.bat
```

Or run Python directly:

```bash
python analyze_ai_ml_ipos_expanded.py
```

---

## 📧 Optional: Email Notifications

To receive email notifications with screening results:

1. Uncomment the email line in `run_weekly_ai_ml_screening.bat`:
   ```bat
   python send_weekly_screening_summary.py
   ```

2. Create `send_weekly_screening_summary.py` to send summary emails
   - Use existing `send_weekly_stock_pick.py` as template
   - Customize to include top AI/ML candidates

---

## 🔧 Troubleshooting

### Task Doesn't Run

**Check Task Scheduler History:**
1. Open Task Scheduler
2. Find your task
3. Click "History" tab
4. Look for error codes

**Common Issues:**
- Python not in system PATH → Set full Python path in batch file
- Working directory wrong → Verify "Start in" path
- Permissions → Run as administrator

### View Logs

Check the log files if something fails:

```bash
# View latest log
type logs\weekly_screening\screening_*.log | more

# List all logs
dir logs\weekly_screening\*.log /o-d
```

### Test the Batch File

Before scheduling, test it manually:

```bash
cd C:\Users\javie\Documents\GitHub\proteus
run_weekly_ai_ml_screening.bat
```

Make sure it completes successfully.

---

## 📈 Advanced: Automated Alerts

Create alerts for significant changes:

1. **Score Tier Changes**
   - Alert when a company moves from Moderate → Strong
   - Alert for new Exceptional candidates (score > 75)

2. **New IPOs**
   - Alert when new AI/ML IPOs are detected
   - Automatic addition to watchlist

3. **Performance Milestones**
   - Alert when company hits +100%, +500%, +1000% returns
   - Alert for revenue growth acceleration

To implement: Create `alert_system.py` that:
- Compares current week's report to previous week
- Detects significant changes
- Sends email/SMS notifications

---

## 🎯 Best Practices

1. **Review Weekly** - Check reports every Monday
2. **Track Trends** - Look for companies improving over time
3. **Update Database** - Add new AI/ML IPOs as they occur
4. **Adjust Filters** - Modify thresholds based on findings
5. **Backup Reports** - Keep historical reports for trend analysis

---

## 📅 Recommended Workflow

**Monday Morning (9:00 AM):**
- Automated screening runs
- Reports generated

**Monday Afternoon:**
- Review new report
- Compare to last week
- Update watchlist
- Check for tier changes
- Note any new IPOs

**Throughout Week:**
- Monitor top candidates (PLTR, ALAB, CRWV, IONQ)
- Watch for new AI/ML IPO announcements
- Track relevant news

---

## 🎓 Next Steps

Once automation is working:

1. Set up email notifications
2. Create alert system for tier changes
3. Build dashboard for visualization
4. Integrate sentiment analysis
5. Add news tracking for top candidates

---

**Questions?** Check `CURRENT_STATE.md` for latest status and findings.

**Happy Automated Screening!** 🚀

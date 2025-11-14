# Proteus Web Dashboard - Quick Start Guide

## Overview

The Proteus Web Dashboard provides a browser-based interface for monitoring your Mean Reversion Strategy in real-time. It automatically scans for signals at scheduled times and displays them in a clean, auto-refreshing UI.

## Features

- **Automated Scanning**: Runs at 9:30 AM, 1:30 PM, and 5:30 PM daily
- **Manual Scanning**: "Scan Now" button for on-demand analysis
- **Auto-Refresh**: Dashboard updates every 60 seconds automatically
- **Signal Display**: Shows active signals with price, Z-score, RSI, and expected return
- **Performance Tracking**: Displays win rate, total return, and trade statistics
- **Local & Private**: Runs on your computer (localhost:5000), no external dependencies

## Installation

### 1. Install Dependencies

```bash
pip install flask apscheduler
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

### 2. Verify Components

Make sure you have the following components set up:
- Signal Scanner (`src/trading/signal_scanner.py`)
- Paper Trader (`src/trading/paper_trader.py`)
- Performance Tracker (`src/trading/performance_tracker.py`)

## Running the Dashboard

### Start the Server

```bash
python src/web/app.py
```

You should see:

```
======================================================================
PROTEUS TRADING DASHBOARD
======================================================================

Starting web server on http://localhost:5000

[SCHEDULER] Started - scanning at 9:30 AM, 1:30 PM, 5:30 PM daily
Running initial scan...

Dashboard ready!
Open in browser: http://localhost:5000

Press Ctrl+C to stop
```

### Open in Browser

1. Open your web browser
2. Go to: **http://localhost:5000**
3. Bookmark the page for easy access
4. Keep the terminal window running in the background

## Dashboard Sections

### Status Bar
- **Last Scan**: Shows how long ago the last scan ran (e.g., "15m ago")
- **Status**: Current system status (e.g., "Complete - Found 2 signal(s)")
- **Active Signals**: Count of current buy signals
- **Scan Now**: Button to trigger immediate scan

### Active Signals Card
Displays any current buy signals with:
- Ticker symbol
- Current price
- Z-score (oversold indicator)
- RSI (relative strength indicator)
- Expected return %

### Performance Card
Shows your paper trading results:
- Total trades executed
- Win rate %
- Total return %
- Days tracked

### Strategy Info Card
Fixed information about the strategy:
- Universe size (10 stocks)
- Expected win rate (77.3%)
- Hold period (1-2 days)
- Position size (10% per trade)

## Scheduled Scanning

The dashboard automatically scans for signals at:
- **9:30 AM** - Market open (captures opening volatility)
- **1:30 PM** - Mid-day (captures lunch-time moves)
- **5:30 PM** - After market close (reviews full day data)

You don't need to do anything - just leave it running!

## Manual Scanning

Click the **"Scan Now"** button to:
- Check for signals immediately
- Refresh data before market hours
- Verify system is working

The button shows a loading animation while scanning (usually 10-30 seconds).

## Auto-Refresh

The dashboard automatically refreshes every 60 seconds to:
- Update signal status
- Update performance metrics
- Update last scan time
- Refresh current prices

You'll see "🔄 Auto-refreshing every 60 seconds" at the bottom of the Strategy Info card.

## Data Persistence

All scan results are saved to JSON files:
- `data/web_dashboard/scan_status.json` - Last scan info and signals
- `data/performance_history/` - Performance tracking data

This means:
- Dashboard state survives restarts
- You can stop/start the server without losing data
- Historical performance is preserved

## Usage Tips

### Daily Workflow

1. **Morning**: Open dashboard after 9:30 AM to check for signals
2. **Afternoon**: Check at 1:30 PM for mid-day opportunities
3. **Evening**: Review at 5:30 PM for end-of-day signals

### Keep Computer Running

For automated scanning to work:
- Keep your computer on during market hours
- Keep the Python server running (terminal window open)
- Browser tab can be closed - just reopen http://localhost:5000 anytime

### Multiple Browsers

You can open the dashboard on:
- Multiple tabs (all will auto-refresh independently)
- Different browsers
- Mobile browser (phone on same Wi-Fi): http://YOUR_COMPUTER_IP:5000

## Troubleshooting

### Server Won't Start

**Problem**: `ModuleNotFoundError: No module named 'flask'`

**Solution**: Install Flask
```bash
pip install flask apscheduler
```

### No Signals Showing

**Possible causes**:
1. Market is in BEAR regime (strategy disabled in bear markets)
2. No panic sell opportunities today (strategy is selective, 1-2 signals per month)
3. Earnings reports nearby (strategy excludes ±3 days around earnings)

**What to do**: This is normal! The strategy is highly selective. Most days have zero signals.

### Dashboard Not Updating

1. Check browser console for errors (F12 → Console)
2. Verify server is still running (check terminal window)
3. Manually refresh browser (F5)
4. Click "Scan Now" to force update

### Port Already in Use

**Problem**: `Address already in use: 5000`

**Solution**: Stop other Flask apps or change port in `app.py`:
```python
app.run(host='0.0.0.0', port=5001, debug=False)  # Changed to 5001
```

## API Endpoints

For advanced users or custom integrations:

### GET /api/status
Returns current scan status and signals
```json
{
  "last_scan": "2025-11-14T09:35:00",
  "status": "Complete - Found 1 signal(s)",
  "signals": [
    {
      "ticker": "TSLA",
      "price": 156.80,
      "z_score": -2.10,
      "rsi": 29.9,
      "expected_return": 3.5
    }
  ],
  "signal_count": 1
}
```

### GET /api/scan
Triggers manual scan, returns immediately
```json
{
  "status": "Scan initiated"
}
```

### GET /api/performance
Returns performance metrics
```json
{
  "total_trades": 15,
  "win_rate": 80.0,
  "total_return": 25.5,
  "days_tracked": 42
}
```

## Security Notes

- Dashboard runs on **localhost only** (not accessible from internet)
- No authentication required (local use only)
- No data sent to external servers
- All data stored locally on your computer

## Next Steps

After using the dashboard for 6 weeks:
1. Review performance in the dashboard
2. Compare to backtest expectations (77.3% win rate)
3. Decide whether to deploy with real capital
4. See `PRODUCTION_DEPLOYMENT_GUIDE.md` for live trading setup

## Support

If you encounter issues:
1. Check this guide's troubleshooting section
2. Review terminal output for error messages
3. Verify all dependencies installed: `pip list | grep -i flask`
4. Check log files in `data/web_dashboard/`

---

**Dashboard Version**: 1.0
**Last Updated**: 2025-11-14
**Compatible With**: Mean Reversion Strategy v4.0

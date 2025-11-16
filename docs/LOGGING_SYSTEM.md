# Scan Logging System

Complete audit trail for all trading scans - every scan is logged to text files.

## Log Structure

```
logs/scans/
├── daily/                          # Daily consolidated logs
│   ├── 2025-11-16.log
│   ├── 2025-11-17.log
│   └── ...
├── summary/                        # Summary files
│   └── latest_scans.log           # Last 50 scans
├── scan_2025-11-16_103000.json    # Individual scan (JSON)
├── scan_2025-11-16_153000.json
└── ...
```

## Daily Log Format

Each day gets one text file with all scans appended:

```
================================================================================
SCAN LOG - 2025-11-16 10:00:00
================================================================================
Type: SCHEDULED
Market Regime: BULL
Status: Complete - Found 1 signal(s)
Signals Found: 1

SIGNALS DETECTED:
--------------------------------------------------------------------------------
  Ticker: TSLA
  Price: $156.80
  Z-Score: -2.10
  RSI: 29.9
  Expected Return: 3.50%
  Signal Date: 2025-11-16
--------------------------------------------------------------------------------

PERFORMANCE SUMMARY:
  Total Trades: 3
  Win Rate: 66.7%
  Total Return: +5.20%
  Days Tracked: 2

================================================================================
```

## Individual Scan JSON

Each scan also creates a JSON file with full data:

```json
{
  "timestamp": "2025-11-16 10:00:00",
  "type": "scheduled",
  "regime": "BULL",
  "status": "Complete - Found 1 signal(s)",
  "signals": [
    {
      "ticker": "TSLA",
      "price": 156.80,
      "z_score": -2.10,
      "rsi": 29.9,
      "expected_return": 3.5,
      "Date": "2025-11-16"
    }
  ],
  "signal_count": 1,
  "performance": {
    "total_trades": 3,
    "win_rate": 66.7,
    "total_return": 5.2,
    "days_tracked": 2
  }
}
```

## Summary Log

Shows last 50 scans in one file for quick review:

```
============================================================================================================================
RECENT SCANS SUMMARY (Last 50)
============================================================================================================================
Timestamp            | Type       | Regime          | Signals    | Status
----------------------------------------------------------------------------------------------------------------------------
2025-11-16 15:30:00  | SCHEDULED  | Regime: BULL    | Signals: 0 | Status: Complete - Found 0 signal(s)
2025-11-16 10:00:00  | SCHEDULED  | Regime: BULL    | Signals: 1 | Status: Complete - Found 1 signal(s)
2025-11-16 09:45:22  | MANUAL     | Regime: BULL    | Signals: 0 | Status: Complete - Found 0 signal(s)
```

## What Gets Logged

Every scan logs:
- ✅ **Timestamp** - Exact date/time of scan
- ✅ **Type** - SCHEDULED or MANUAL
- ✅ **Market Regime** - BULL/BEAR/SIDEWAYS
- ✅ **Status** - Scan result message
- ✅ **Signals** - All detected opportunities with full details
- ✅ **Performance** - Current trading performance stats

## Scan Types

**SCHEDULED** - Automatic scans
- 10:00 AM EST (16:00 ZRH) - 1h after market open
- 3:30 PM EST (21:30 ZRH) - End of day

**MANUAL** - User-triggered scans
- Via "Scan Now" button in dashboard
- Via API call to `/api/scan`
- Via command-line tools

## Accessing Logs

### View Today's Log

```bash
# Windows
type logs\scans\daily\2025-11-16.log

# Linux/Mac
cat logs/scans/daily/2025-11-16.log
```

### View Summary

```bash
# Windows
type logs\scans\summary\latest_scans.log

# Linux/Mac
cat logs/scans/summary/latest_scans.log
```

### Programmatic Access

```python
from src.utils.scan_logger import ScanLogger

logger = ScanLogger()

# Get today's summary
print(logger.get_today_summary())

# Get recent scans
recent = logger.get_recent_scans(count=10)
for scan in recent:
    print(f"{scan['timestamp']}: {scan['signal_count']} signals")
```

## Log Retention

**Daily logs**: Kept forever (small text files)
**Individual JSONs**: Kept forever (for data analysis)
**Summary log**: Last 50 scans only (auto-rotates)

## Benefits

✅ **Complete audit trail** - Never lose scan history
✅ **Debugging** - See exactly what happened when
✅ **Analysis** - Review historical signal patterns
✅ **Compliance** - Permanent record of all trading decisions
✅ **Stateless web** - Dashboard can restart without losing history
✅ **Human readable** - Text files you can read directly

## Manual Scan Behavior

When you click "Scan Now":
1. ✅ Scans all 10 stocks
2. ✅ Logs results to text files (marked as MANUAL)
3. ✅ Sends email notification
4. ✅ Sends Windows notification
5. ✅ Updates web UI with results
6. ✅ Saves to JSON state files

Complete traceability for every scan!

## Example Usage

**Check if system ran today:**
```bash
ls logs/scans/daily/$(date +%Y-%m-%d).log
```

**Count total scans:**
```bash
# Windows
dir /b logs\scans\scan_*.json | find /c ".json"

# Linux/Mac
ls logs/scans/scan_*.json | wc -l
```

**Find scans with signals:**
```bash
# Windows
findstr /C:"Signals Found: [1-9]" logs\scans\daily\*.log

# Linux/Mac
grep "Signals Found: [1-9]" logs/scans/daily/*.log
```

## Integration

Logging is integrated into:
- ✅ Web dashboard (automatic)
- ✅ Scheduled scans (automatic)
- ✅ Manual scans (automatic)
- ✅ Daily runner (automatic)

You don't need to do anything - all scans are logged automatically!

---

**Log Location**: `logs/scans/`
**Log Format**: Text + JSON
**Retention**: Forever (small files)
**Auto-rotation**: Summary only (last 50 scans)

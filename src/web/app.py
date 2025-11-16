"""
Proteus Trading Dashboard - Local Web UI

A simple Flask web server that:
1. Runs signal scanner on schedule (daily at market open)
2. Displays current signals and status
3. Shows trade history and performance
4. Auto-refreshes every minute

Usage:
    python src/web/app.py

Then open: http://localhost:5000
"""

from flask import Flask, render_template, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, time as dt_time
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.trading.signal_scanner import SignalScanner
from src.trading.paper_trader import PaperTrader
from src.trading.performance_tracker import PerformanceTracker
from src.utils.scan_logger import ScanLogger

# Try to import notification options (user can choose which to use)
try:
    from src.notifications.windows_notifier import WindowsNotifier
    windows_notifier = WindowsNotifier()
except:
    windows_notifier = None

# Try SendGrid first (simpler than SMTP), fall back to SMTP
email_notifier = None
try:
    from src.notifications.sendgrid_notifier import SendGridNotifier
    email_notifier = SendGridNotifier()
    if not email_notifier.is_enabled():
        email_notifier = None
except:
    pass

# If SendGrid not available, try SMTP email
if email_notifier is None:
    try:
        from src.notifications.email_notifier import EmailNotifier
        email_notifier = EmailNotifier()
        if not email_notifier.is_enabled():
            email_notifier = None
    except:
        pass

app = Flask(__name__)

# Global state
scanner = SignalScanner()
tracker = PerformanceTracker()
scan_logger = ScanLogger()
last_scan_time = None
current_signals = []
scan_status = "Not started"

# Data directory
DATA_DIR = 'data/web_dashboard'
os.makedirs(DATA_DIR, exist_ok=True)

SIGNALS_FILE = os.path.join(DATA_DIR, 'current_signals.json')
STATUS_FILE = os.path.join(DATA_DIR, 'scan_status.json')


def run_daily_scan(scan_type='scheduled'):
    """Run signal scanner and update dashboard data.

    Args:
        scan_type: 'scheduled' or 'manual'
    """
    global last_scan_time, current_signals, scan_status

    try:
        scan_time = datetime.now()
        print(f"[{scan_time}] Running {scan_type} scan...")
        scan_status = "Scanning..."

        # Check market regime
        regime = scanner.get_market_regime()

        if regime == 'BEAR':
            scan_status = "Complete (BEAR market - no trading)"
            current_signals = []
            last_scan_time = scan_time
        else:
            # Scan for signals
            signals = scanner.scan_all_stocks()

            current_signals = signals
            last_scan_time = scan_time
            scan_status = f"Complete - Found {len(signals)} signal(s)"

            print(f"  Regime: {regime}")
            print(f"  Signals: {len(signals)}")

            if signals:
                for sig in signals:
                    print(f"    {sig['ticker']}: ${sig['price']:.2f}, Z={sig['z_score']:.2f}, RSI={sig['rsi']:.1f}")

        # Get performance data
        performance = tracker.get_stats_summary()

        # LOG EVERYTHING to text files
        log_file = scan_logger.log_scan(
            scan_type=scan_type,
            regime=regime,
            signals=current_signals,
            scan_status=scan_status,
            performance=performance
        )
        print(f"  Logged to: {log_file}")

        # Save to file
        save_status()

        # Send notifications (ALWAYS, for both scheduled and manual scans)
        try:
            # Windows notifications
            if windows_notifier and windows_notifier.is_enabled():
                windows_notifier.send_scan_notification(scan_status, current_signals, performance)

            # Email notifications
            if email_notifier and email_notifier.is_enabled():
                email_notifier.send_scan_notification(scan_status, current_signals, performance)
                print(f"  Email sent to {email_notifier.config['recipient_email']}")

        except Exception as notification_error:
            print(f"[WARN] Notification failed: {notification_error}")

    except Exception as e:
        scan_status = f"Error: {str(e)}"
        print(f"[ERROR] Scan failed: {e}")
        import traceback
        traceback.print_exc()


def save_status():
    """Save current status to file."""
    status = {
        'last_scan': last_scan_time.isoformat() if last_scan_time else None,
        'status': scan_status,
        'signals': current_signals,
        'signal_count': len(current_signals)
    }

    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2, default=str)


def load_status():
    """Load status from file."""
    global last_scan_time, current_signals, scan_status

    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                status = json.load(f)

            last_scan_time = datetime.fromisoformat(status['last_scan']) if status['last_scan'] else None
            scan_status = status['status']
            current_signals = status['signals']
        except Exception as e:
            print(f"[WARN] Could not load status: {e}")


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/status')
def api_status():
    """API endpoint for current status."""
    return jsonify({
        'last_scan': last_scan_time.isoformat() if last_scan_time else None,
        'status': scan_status,
        'signals': current_signals,
        'signal_count': len(current_signals),
        'server_time': datetime.now().isoformat()
    })


@app.route('/api/scan')
def api_scan():
    """Manual scan trigger - logs, emails, and updates UI."""
    run_daily_scan(scan_type='manual')
    return jsonify({'status': 'Manual scan initiated - check email and logs'})


@app.route('/api/performance')
def api_performance():
    """Get performance metrics."""
    stats = tracker.get_stats_summary()
    return jsonify(stats)


def setup_scheduler():
    """Setup scheduled scans.

    Scans at 10:00 AM EST (16:00 ZRH) - 1 hour after market open.
    """
    scheduler = BackgroundScheduler()

    # Schedule daily scan at 10:00 AM EST (16:00 ZRH - 1h after market open)
    scheduler.add_job(
        lambda: run_daily_scan(scan_type='scheduled'),
        'cron',
        hour=10,
        minute=0,
        id='daily_scan_1000'
    )

    # Optional: Add end-of-day scan at 3:30 PM EST (21:30 ZRH - after market close)
    scheduler.add_job(
        lambda: run_daily_scan(scan_type='scheduled'),
        'cron',
        hour=15,
        minute=30,
        id='eod_scan'
    )

    scheduler.start()
    print("[SCHEDULER] Started - scanning at 10:00 AM EST (16:00 ZRH), 3:30 PM EST (21:30 ZRH)")

    return scheduler


if __name__ == '__main__':
    print("=" * 70)
    print("PROTEUS TRADING DASHBOARD")
    print("=" * 70)
    print()
    print("Starting web server on http://localhost:5000")
    print()

    # Load previous status
    load_status()

    # Setup scheduler
    scheduler = setup_scheduler()

    # Check notification methods
    print("\nNotification Methods:")
    if windows_notifier and windows_notifier.is_enabled():
        print("  ✓ Windows Notifications: Enabled (NO PASSWORD NEEDED!)")
    else:
        print("  - Windows Notifications: Not available (pip install windows-toasts)")

    if email_notifier and email_notifier.is_enabled():
        print(f"  ✓ Email Notifications: Enabled → {email_notifier.config['recipient_email']}")
    else:
        print("  - Email Notifications: Not configured (see EMAIL_NOTIFICATIONS_SETUP.md)")

    if not (windows_notifier and windows_notifier.is_enabled()) and not (email_notifier and email_notifier.is_enabled()):
        print("\n  [!] No notifications configured - dashboard will work but you won't get alerts")
        print("  [!] Quickest option: pip install windows-toasts (no password needed!)")
    print()

    # Run initial scan
    if last_scan_time is None or (datetime.now() - last_scan_time).days >= 1:
        print("Running initial scan...")
        run_daily_scan()

    print()
    print("Dashboard ready!")
    print("Open in browser: http://localhost:5000")
    print()
    print("Press Ctrl+C to stop")
    print()

    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
        scheduler.shutdown()

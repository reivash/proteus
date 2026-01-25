"""
Proteus Trading Dashboard - Local Web UI

A simple Flask web server that:
1. Runs signal scanner on schedule (daily at market open)
2. Displays current signals and status
3. Shows trade history and performance
4. Auto-refreshes every minute

Usage:
    python common/web/app.py

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

from common.trading.signal_scanner import SignalScanner
from common.trading.paper_trader import PaperTrader
from common.trading.performance_tracker import PerformanceTracker
from common.utils.scan_logger import ScanLogger

# Try to import notification options (user can choose which to use)
try:
    from common.notifications.windows_notifier import WindowsNotifier
    windows_notifier = WindowsNotifier()
except:
    windows_notifier = None

# Try SendGrid first (simpler than SMTP), fall back to SMTP
email_notifier = None
try:
    from common.notifications.sendgrid_notifier import SendGridNotifier
    email_notifier = SendGridNotifier()
    if not email_notifier.is_enabled():
        email_notifier = None
except:
    pass

# If SendGrid not available, try SMTP email
if email_notifier is None:
    try:
        from common.notifications.email_notifier import EmailNotifier
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
DATA_DIR = 'features/reporting/data/web_dashboard'
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

        # Get list of tickers we're scanning
        from common.config.mean_reversion_params import get_all_tickers
        scanned_tickers = get_all_tickers()

        print(f"  Scanning {len(scanned_tickers)} instruments: {', '.join(scanned_tickers)}")

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
            else:
                print(f"    No signals found across {len(scanned_tickers)} stocks")

        # Get performance data
        performance = tracker.get_stats_summary()

        # LOG EVERYTHING to text files (including what was scanned)
        log_file = scan_logger.log_scan(
            scan_type=scan_type,
            regime=regime,
            signals=current_signals,
            scan_status=scan_status,
            performance=performance,
            scanned_tickers=scanned_tickers
        )
        print(f"  Logged to: {log_file}")

        # Save to file
        save_status()

        # Send notifications (ALWAYS, for both scheduled and manual scans)
        try:
            # Windows notifications
            if windows_notifier and windows_notifier.is_enabled():
                windows_notifier.send_scan_notification(scan_status, current_signals, performance)

            # Email notifications (pass scanned_tickers so email shows what was scanned)
            if email_notifier and email_notifier.is_enabled():
                email_notifier.send_scan_notification(scan_status, current_signals, performance, scanned_tickers)
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
    from common.config.mean_reversion_params import get_all_tickers

    status = {
        'last_scan': last_scan_time.isoformat() if last_scan_time else None,
        'status': scan_status,
        'signals': current_signals,
        'signal_count': len(current_signals),
        'scanned_tickers': get_all_tickers()
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
    from common.config.mean_reversion_params import get_all_tickers
    return jsonify({
        'last_scan': last_scan_time.isoformat() if last_scan_time else None,
        'status': scan_status,
        'signals': current_signals,
        'signal_count': len(current_signals),
        'server_time': datetime.now().isoformat(),
        'scanned_tickers': get_all_tickers()
    })


@app.route('/api/scan')
def api_scan():
    """Manual scan trigger - logs, emails, and updates UI."""
    run_daily_scan(scan_type='manual')
    return jsonify({'status': 'Manual scan initiated - check email and logs'})


@app.route('/api/test-email')
def api_test_email():
    """Send latest report email with current signals and performance."""
    try:
        from common.config.mean_reversion_params import get_all_tickers
        scanned_tickers = get_all_tickers()

        if email_notifier and email_notifier.is_enabled():
            # Send full scan notification with current data
            email_notifier.send_scan_notification(
                scan_status=scan_status,
                signals=current_signals,
                performance=tracker.get_stats_summary(),
                scanned_tickers=scanned_tickers
            )
            return jsonify({
                'status': 'success',
                'message': f'Latest report sent to {email_notifier.config["recipient_email"]}'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Email notifications not configured'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to send report: {str(e)}'
        })


@app.route('/api/performance')
def api_performance():
    """Get performance metrics and trade history."""
    stats = tracker.get_stats_summary()

    # Add recent trades (last 10)
    recent_trades = tracker._get_recent_trades(10) if hasattr(tracker, 'trade_log') else []
    stats['recent_trades'] = recent_trades

    # Add open positions from latest snapshot
    if tracker.daily_snapshots:
        latest = tracker.daily_snapshots[-1]
        stats['open_positions'] = latest.get('positions', [])
    else:
        stats['open_positions'] = []

    return jsonify(stats)


def setup_scheduler():
    """Setup scheduled scans.

    OPTIMIZED SCHEDULE (based on EXP-020 findings):
    - 9:45 AM EST: Morning scan (15 min after open, monitor positions)
    - 3:45 PM EST: Afternoon scan (15 min before close, optimal entry timing)
    """
    scheduler = BackgroundScheduler()

    # Morning scan at 9:45 AM EST (15 minutes after market open)
    # Purpose: Monitor existing positions, check overnight gaps
    scheduler.add_job(
        lambda: run_daily_scan(scan_type='scheduled'),
        'cron',
        hour=9,
        minute=45,
        id='morning_scan_945'
    )

    # Afternoon scan at 3:45 PM EST (15 minutes before market close)
    # Purpose: Detect panic sells, execute trades before close
    # EXP-020: Panic day close entry is optimal (+49.70% vs +22.55% next day)
    scheduler.add_job(
        lambda: run_daily_scan(scan_type='scheduled'),
        'cron',
        hour=15,
        minute=45,
        id='afternoon_scan_345'
    )

    scheduler.start()
    print("[SCHEDULER] Started - scanning at 9:45 AM EST, 3:45 PM EST (optimal entry timing)")

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

    # Show what instruments we're monitoring
    from common.config.mean_reversion_params import get_all_tickers
    scanned_tickers = get_all_tickers()
    print(f"\nMonitoring {len(scanned_tickers)} instruments:")
    print(f"  {', '.join(scanned_tickers)}")

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

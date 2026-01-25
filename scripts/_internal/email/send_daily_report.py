"""
Send daily trading report via email.
This simulates what the automated dashboard would send daily.
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from common.trading.signal_scanner import SignalScanner
from common.trading.performance_tracker import PerformanceTracker
from common.notifications.email_notifier import EmailNotifier


def send_daily_report():
    """Send today's trading report via email."""
    print("=" * 70)
    print("PROTEUS DAILY TRADING REPORT")
    print("=" * 70)
    print()
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize components
    scanner = SignalScanner()
    tracker = PerformanceTracker()
    email_notifier = EmailNotifier()

    # Check email configuration
    if not email_notifier.is_enabled():
        print("[ERROR] Email notifications not configured!")
        print()
        print("To send email, you need to:")
        print("1. Edit email_config.json in project root")
        print("2. Add your Gmail address to 'sender_email'")
        print("3. Get app password from: https://myaccount.google.com/apppasswords")
        print("4. Add the 16-char password to 'sender_password'")
        print()
        print("Example email_config.json:")
        print('{')
        print('  "enabled": true,')
        print('  "recipient_email": "reivaj.cabero@gmail.com",')
        print('  "sender_email": "YOUR_GMAIL@gmail.com",')
        print('  "sender_password": "YOUR_16_CHAR_APP_PASSWORD",')
        print('  "smtp_server": "smtp.gmail.com",')
        print('  "smtp_port": 587')
        print('}')
        print()
        print("Recipient is already set to: reivaj.cabero@gmail.com")
        print()
        return False

    # Step 1: Check market regime
    print("Step 1: Checking market regime...")
    regime = scanner.get_market_regime()
    print(f"  Regime: {regime}")
    print()

    # Step 2: Scan for signals
    print("Step 2: Scanning for signals...")

    if regime == 'BEAR':
        scan_status = "Complete (BEAR market - no trading)"
        signals = []
        print("  Strategy disabled in BEAR market")
    else:
        signals = scanner.scan_all_stocks()
        scan_status = f"Complete - Found {len(signals)} signal(s)"

        if signals:
            print(f"  Found {len(signals)} signal(s):")
            for sig in signals:
                print(f"    - {sig['ticker']}: ${sig['price']:.2f}, Z={sig['z_score']:.2f}, RSI={sig['rsi']:.1f}")
        else:
            print("  No signals detected")

    print()

    # Step 3: Get performance data
    print("Step 3: Loading performance data...")
    performance = tracker.get_stats_summary()
    print(f"  Total trades: {performance.get('total_trades', 0)}")
    print(f"  Win rate: {performance.get('win_rate', 0):.1f}%")
    print()

    # Step 4: Send email
    print("Step 4: Sending email notification...")
    print(f"  To: {email_notifier.config['recipient_email']}")
    print(f"  From: {email_notifier.config['sender_email']}")
    print()

    success = email_notifier.send_scan_notification(scan_status, signals, performance)

    print()
    print("=" * 70)

    if success:
        print("SUCCESS! Email sent to reivaj.cabero@gmail.com")
        print()
        print("Check your inbox for:")
        if len(signals) == 0:
            print("  Subject: 📊 Proteus Daily Scan - No Signals")
        else:
            print(f"  Subject: 🚨 Proteus Alert - {len(signals)} BUY Signal(s)!")
        print()
        print("The email includes:")
        print("  - Market regime status")
        print("  - Scan results for all 10 stocks")
        if signals:
            print("  - Buy signal details (ticker, price, Z-score, RSI)")
            print("  - Entry recommendations")
        print("  - Performance summary")
        print("  - Next steps")
    else:
        print("FAILED to send email")
        print()
        print("Common issues:")
        print("  1. Wrong Gmail address in 'sender_email'")
        print("  2. Wrong app password in 'sender_password'")
        print("  3. Need to enable 2-factor auth on Google account first")
        print("  4. App password not created at: https://myaccount.google.com/apppasswords")

    print("=" * 70)

    return success


if __name__ == '__main__':
    send_daily_report()

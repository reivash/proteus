"""
Windows Toast Notifications - NO PASSWORD NEEDED!

Simplest option - just shows notifications on your Windows computer.
No email, no passwords, no API keys, no setup!

Just install: pip install windows-toasts
"""

import json
from typing import List, Dict
from datetime import datetime

try:
    from windows_toasts import Toast, WindowsToaster, ToastDisplayImage
    TOAST_AVAILABLE = True
except ImportError:
    TOAST_AVAILABLE = False


class WindowsNotifier:
    """
    Send Windows toast notifications - NO AUTHENTICATION NEEDED!

    Pros:
    - Zero setup, no passwords, no API keys
    - Instant notifications on your computer
    - Native Windows integration

    Cons:
    - Only works when computer is on
    - Only shows on your computer (not phone)
    - No email archive
    """

    def __init__(self):
        if TOAST_AVAILABLE:
            self.toaster = WindowsToaster('Proteus Trading Dashboard')
        self.enabled = TOAST_AVAILABLE

    def is_enabled(self) -> bool:
        """Check if Windows notifications are available."""
        return self.enabled

    def send_scan_notification(self, scan_status: str, signals: List[Dict],
                              performance: Dict = None) -> bool:
        """Send Windows toast notification."""
        if not self.is_enabled():
            return False

        try:
            # Create notification
            toast = Toast()

            # Title
            if len(signals) == 0:
                toast.text_fields = ['📊 Proteus Daily Scan', 'No signals detected']
            elif len(signals) == 1:
                ticker = signals[0]['ticker']
                price = signals[0]['price']
                z_score = signals[0]['z_score']
                rsi = signals[0]['rsi']
                toast.text_fields = [
                    f'🚨 BUY Signal: {ticker}',
                    f'Price: ${price:.2f} | Z={z_score:.2f} | RSI={rsi:.1f}',
                    f'Expected return: +{signals[0].get("expected_return", 0):.2f}%'
                ]
            else:
                toast.text_fields = [
                    f'🚨 {len(signals)} BUY Signals Found!',
                    ', '.join([s['ticker'] for s in signals]),
                    'Open dashboard for details'
                ]

            # Add button to open dashboard
            toast.AddAction('Open Dashboard', 'http://localhost:5000')

            # Show notification
            self.toaster.show_toast(toast)

            print(f"[NOTIFICATION] Sent Windows toast: {toast.text_fields[0]}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to send notification: {e}")
            return False

    def send_test_notification(self) -> bool:
        """Send test notification."""
        if not self.is_enabled():
            return False

        try:
            toast = Toast()
            toast.text_fields = [
                '✓ Proteus Notifications Working!',
                'No passwords needed - just local Windows notifications',
                f'Test sent at {datetime.now().strftime("%H:%M:%S")}'
            ]

            toast.AddAction('Open Dashboard', 'http://localhost:5000')

            self.toaster.show_toast(toast)

            print("✓ Test notification sent!")
            return True

        except Exception as e:
            print(f"✗ Test failed: {e}")
            return False


def send_test_windows_notification():
    """Quick test of Windows notifications."""
    if not TOAST_AVAILABLE:
        print("Windows Toasts not installed.")
        print("\nTo install:")
        print("  pip install windows-toasts")
        print("\nThis is the SIMPLEST option - no passwords, no API keys!")
        return False

    print("Sending test Windows notification...")
    notifier = WindowsNotifier()
    return notifier.send_test_notification()


if __name__ == '__main__':
    send_test_windows_notification()

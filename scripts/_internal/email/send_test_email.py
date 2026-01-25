"""
Quick script to send a test email notification.

Before running:
1. Edit email_config.json and add your Gmail credentials
2. Get an app password from: https://myaccount.google.com/apppasswords

Usage:
    python send_test_email.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from common.notifications.email_notifier import send_test_notification

if __name__ == '__main__':
    print("=" * 70)
    print("PROTEUS EMAIL NOTIFICATION TEST")
    print("=" * 70)
    print()

    success = send_test_notification()

    print()
    print("=" * 70)

    if success:
        print("SUCCESS! Check your email inbox.")
    else:
        print("FAILED - See instructions above to configure email.")
        print()
        print("Quick setup:")
        print("1. Edit email_config.json in project root")
        print("2. Replace YOUR_GMAIL_HERE with your Gmail address")
        print("3. Get app password from: https://myaccount.google.com/apppasswords")
        print("4. Replace YOUR_APP_PASSWORD_HERE with the 16-char password")
        print("5. Run this script again")

    print("=" * 70)

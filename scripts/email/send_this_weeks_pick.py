"""
Send This Week's Stock Pick Email

Quick script to send the weekly recommendation that was generated.
Uses the already-generated recommendation from results folder.
"""

import json
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, 'src')

from src.notifications.sendgrid_notifier import SendGridNotifier


def send_recommendation_email():
    """Send the weekly recommendation email."""

    # Load email config
    with open('email_config.json', 'r') as f:
        config = json.load(f)

    # Load the generated recommendation
    rec_file = Path('results/weekly_recommendation_20251130.txt')

    if not rec_file.exists():
        print(f"[ERROR] Recommendation file not found: {rec_file}")
        return False

    with open(rec_file, 'r', encoding='utf-8') as f:
        email_body = f.read()

    # Initialize notifier (reads email_config.json automatically)
    notifier = SendGridNotifier()

    # Email details
    subject = "Weekly IPO Stock Pick: RDDT (Reddit) - Recommendation Score 74.1/100"
    to_email = config['recipient_email']

    print("=" * 80)
    print("SENDING WEEKLY STOCK RECOMMENDATION")
    print("=" * 80)
    print(f"To: {to_email}")
    print(f"Subject: {subject}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Send email
    try:
        notifier.send_email(
            to_email=to_email,
            subject=subject,
            body=email_body
        )

        print("[SUCCESS] Weekly recommendation email sent!")
        print()
        print("Email Preview:")
        print("-" * 80)
        print(email_body[:600])
        print("...")
        print("-" * 80)
        print()
        print(f"Full email ({len(email_body)} chars) sent to {to_email}")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        return False


if __name__ == "__main__":
    success = send_recommendation_email()

    if success:
        print("\n" + "=" * 80)
        print("THIS WEEK'S PICK: RDDT (Reddit)")
        print("=" * 80)
        print("Recommendation Score: 74.1/100")
        print("Total Return Since IPO: +329%")
        print("Revenue Growth: 68% YoY")
        print("Gross Margin: 91%")
        print()
        print("Investment Thesis:")
        print("Platform economics, explosive growth, successfully scaling,")
        print("small share (0.9%) with massive growth potential")
        print("=" * 80)

    sys.exit(0 if success else 1)

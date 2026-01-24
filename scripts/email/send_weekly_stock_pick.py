"""
Send Weekly Stock Recommendation Email

Generates weekly IPO stock recommendation and sends via email.
Uses Proteus email configuration (SendGrid).

Usage:
    python send_weekly_stock_pick.py
"""

import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, 'src')

from src.trading.weekly_ipo_recommender import WeeklyIPORecommender
from src.notifications.sendgrid_notifier import SendGridNotifier


def load_email_config():
    """Load email configuration."""
    config_file = Path('email_config.json')

    if not config_file.exists():
        print("[ERROR] email_config.json not found")
        print("Please create email_config.json with your SendGrid API key")
        return None

    with open(config_file, 'r') as f:
        config = json.load(f)

    return config


def send_weekly_recommendation():
    """Generate and send weekly stock recommendation."""
    print("=" * 80)
    print("WEEKLY STOCK RECOMMENDATION EMAILER")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load email config
    print("[1/4] Loading email configuration...")
    email_config = load_email_config()

    if not email_config:
        return False

    print(f"[OK] Email config loaded")
    print()

    # Generate recommendation
    print("[2/4] Generating weekly stock recommendation...")
    recommender = WeeklyIPORecommender()

    # Use select high-quality candidates for weekly picks
    priority_candidates = ['RDDT', 'ASND', 'COIN', 'SNOW', 'RBLX', 'PLTR']

    try:
        recommendation = recommender.generate_weekly_pick(priority_candidates)

        if not recommendation:
            print("[WARN] No recommendation generated this week")
            return False

        print(f"[OK] Recommendation generated: {recommendation['ticker']}")
        print()

    except Exception as e:
        print(f"[ERROR] Failed to generate recommendation: {e}")
        return False

    # Generate email body
    print("[3/4] Generating email report...")
    email_body = recommender.generate_email_report(recommendation)
    print(f"[OK] Email report generated ({len(email_body)} characters)")
    print()

    # Send email
    print("[4/4] Sending email...")
    notifier = SendGridNotifier(
        api_key=email_config.get('sendgrid_api_key'),
        from_email=email_config.get('from_email'),
        from_name=email_config.get('from_name', 'Proteus Trading System')
    )

    subject = f"Weekly Stock Pick: {recommendation['ticker']} - {recommendation.get('company_name', 'Unknown')}"
    to_email = email_config.get('to_email')

    try:
        notifier.send_email(
            to_email=to_email,
            subject=subject,
            body=email_body
        )
        print(f"[OK] Email sent to {to_email}")
        print()

    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        print()
        print("Email content preview:")
        print("-" * 80)
        print(email_body[:500])
        print("...")
        return False

    # Save report locally
    report_file = f"results/weekly_recommendation_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(email_body)

    print(f"Report also saved locally: {report_file}")
    print()

    print("=" * 80)
    print("WEEKLY RECOMMENDATION SENT SUCCESSFULLY!")
    print("=" * 80)
    print(f"Stock: {recommendation['ticker']}")
    print(f"Score: {recommendation.get('recommendation_score', 0):.1f}/100")
    print(f"Sent to: {to_email}")
    print()

    return True


if __name__ == "__main__":
    success = send_weekly_recommendation()

    if success:
        print("Next weekly recommendation: Run this script again next week")
    else:
        print("Failed to send weekly recommendation. Check errors above.")

    sys.exit(0 if success else 1)

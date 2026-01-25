"""
Quick test to diagnose SendGrid email issue
"""
import sys
from common.notifications.sendgrid_notifier import SendGridNotifier

def test_sendgrid():
    """Test SendGrid email delivery with detailed error logging."""

    print("=" * 70)
    print("SENDGRID EMAIL DIAGNOSTIC TEST")
    print("=" * 70)

    notifier = SendGridNotifier()

    print(f"\nConfig loaded: {notifier.is_enabled()}")
    print(f"Recipient: {notifier.config.get('recipient_email')}")
    print(f"Sender: {notifier.config.get('sender_email')}")
    print(f"API Key: {notifier.config.get('sendgrid_api_key')[:20]}...{notifier.config.get('sendgrid_api_key')[-10:]}")

    print("\nSending test email...")
    success = notifier.send_test_email()

    if success:
        print("\n✓ EMAIL SENT SUCCESSFULLY!")
        print("\nCheck your inbox (and SPAM folder!) at:", notifier.config.get('recipient_email'))
        print("\nIf not received:")
        print("1. Check SPAM folder first")
        print("2. Verify sender email in SendGrid dashboard: https://app.sendgrid.com/settings/sender_auth")
        print("3. Check SendGrid activity: https://app.sendgrid.com/email_activity")
        print("4. Free tier limit: 100 emails/day - check if exceeded")
    else:
        print("\n✗ EMAIL FAILED!")
        print("Check error messages above for details.")

    print("=" * 70)

if __name__ == "__main__":
    test_sendgrid()

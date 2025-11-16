"""
SendGrid Email Notifier - Simpler Alternative to SMTP

Much easier than Gmail:
1. Sign up at https://sendgrid.com (free tier: 100 emails/day)
2. Get API key from Settings > API Keys
3. Add to email_config.json: "sendgrid_api_key": "YOUR_KEY"

No passwords, no app passwords, just one API key!
"""

import json
import os
from typing import List, Dict
from datetime import datetime

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Email, To, Content
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False


class SendGridNotifier:
    """Send email notifications via SendGrid API (simpler than SMTP)."""

    def __init__(self, config_path='email_config.json'):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load email configuration."""
        if not os.path.exists(self.config_path):
            return {'enabled': False}

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            # Check for SendGrid API key
            if 'sendgrid_api_key' not in config or not config['sendgrid_api_key']:
                return {'enabled': False}

            if 'recipient_email' not in config or not config['recipient_email']:
                return {'enabled': False}

            return config

        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            return {'enabled': False}

    def is_enabled(self) -> bool:
        """Check if SendGrid is enabled and configured."""
        return self.config.get('enabled', False) and SENDGRID_AVAILABLE

    def send_scan_notification(self, scan_status: str, signals: List[Dict],
                              performance: Dict = None, scanned_tickers: List[str] = None) -> bool:
        """Send email via SendGrid."""
        if not self.is_enabled():
            return False

        if not SENDGRID_AVAILABLE:
            print("[ERROR] SendGrid not installed. Run: pip install sendgrid")
            return False

        subject = self._create_subject(signals)
        html_content = self._create_body(scan_status, signals, performance, scanned_tickers)

        message = Mail(
            from_email=Email(self.config.get('sender_email', 'proteus@trading.local')),
            to_emails=To(self.config['recipient_email']),
            subject=subject,
            html_content=Content("text/html", html_content)
        )

        try:
            sg = SendGridAPIClient(self.config['sendgrid_api_key'])
            response = sg.send(message)

            print(f"[EMAIL] Sent via SendGrid to {self.config['recipient_email']}: {subject}")
            return True

        except Exception as e:
            print(f"[ERROR] SendGrid failed: {e}")
            return False

    def send_test_email(self) -> bool:
        """Send test email."""
        if not self.is_enabled():
            return False

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        subject = f"🧪 Proteus - SendGrid Test - {timestamp}"
        html = """
<html>
<body style="font-family: Arial, sans-serif;">
    <h2 style="color: #667eea;">✓ SendGrid Email Working!</h2>
    <p>Your Proteus dashboard is now configured to send email notifications via SendGrid.</p>
    <p><strong>Much simpler than SMTP!</strong> Just one API key, no passwords needed.</p>
    <p>You'll receive notifications at: {recipient}</p>
    <hr>
    <p style="color: #666; font-size: 0.9em;">Sent: {timestamp}</p>
</body>
</html>
""".format(
            recipient=self.config['recipient_email'],
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )

        message = Mail(
            from_email=Email(self.config.get('sender_email', 'proteus@trading.local')),
            to_emails=To(self.config['recipient_email']),
            subject=subject,
            html_content=Content("text/html", html)
        )

        try:
            sg = SendGridAPIClient(self.config['sendgrid_api_key'])
            sg.send(message)
            print(f"✓ Test email sent via SendGrid!")
            return True
        except Exception as e:
            print(f"✗ SendGrid test failed: {e}")
            return False

    def _create_subject(self, signals: List[Dict]) -> str:
        """Create email subject with timestamp to prevent threading."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if len(signals) == 0:
            return f"Proteus Daily Scan - No Signals - {timestamp}"
        elif len(signals) == 1:
            return f"ALERT: Proteus BUY Signal - {signals[0]['ticker']} - {timestamp}"
        else:
            return f"ALERT: Proteus - {len(signals)} BUY Signals! - {timestamp}"

    def _create_body(self, scan_status: str, signals: List[Dict], performance: Dict = None, scanned_tickers: List[str] = None) -> str:
        """Create email body - full HTML template."""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        html = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; color: #333; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                 color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .status {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }}
        .signal {{ background: #e0f2fe; padding: 15px; margin: 10px 0;
                 border-left: 4px solid #667eea; border-radius: 3px; }}
        .signal-header {{ font-size: 1.2em; font-weight: bold; color: #667eea; margin-bottom: 10px; }}
        .signal-detail {{ margin: 5px 0; }}
        .performance {{ background: #f0fdf4; padding: 15px; border-radius: 5px; margin: 15px 0; }}
        .metric {{ margin: 8px 0; }}
        .success {{ color: #10b981; }}
        .scanned {{ background: #fef3c7; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 4px solid #f59e0b; }}
        .footer {{ color: #666; font-size: 0.85em; margin-top: 30px; padding-top: 20px;
                  border-top: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>Proteus Trading Dashboard</h2>
        <p>Mean Reversion Strategy v4.0</p>
        <p style="font-size: 0.9em;">{now}</p>
    </div>

    <div class="status">
        <strong>Scan Status:</strong> {scan_status}
    </div>

    <div class="scanned">
        <strong>Instruments Scanned ({len(scanned_tickers) if scanned_tickers else 0}):</strong><br>
        <span style="font-size: 0.95em;">{', '.join(scanned_tickers) if scanned_tickers else 'Not specified'}</span>
    </div>
"""

        # Add signals section
        if len(signals) > 0:
            html += f"""
    <h3>Active Buy Signals ({len(signals)})</h3>
"""
            for signal in signals:
                expected_return = signal.get('expected_return', 0)
                html += f"""
    <div class="signal">
        <div class="signal-header">{signal['ticker']}</div>
        <div class="signal-detail"><strong>Entry Price:</strong> ${signal['price']:.2f}</div>
        <div class="signal-detail"><strong>Z-Score:</strong> {signal['z_score']:.2f}</div>
        <div class="signal-detail"><strong>RSI:</strong> {signal['rsi']:.1f}</div>
        <div class="signal-detail"><strong>Expected Return:</strong> <span class="success">+{expected_return:.2f}%</span></div>
    </div>
"""
        else:
            html += """
    <h3>No Signals Detected</h3>
    <div class="status">
        <p>No panic sell opportunities found. Strategy is waiting for the right moment.</p>
    </div>
"""

        # Add performance
        if performance:
            html += f"""
    <h3>Performance Summary</h3>
    <div class="performance">
        <div class="metric">Total Trades: <strong>{performance.get('total_trades', 0)}</strong></div>
        <div class="metric">Win Rate: <strong>{performance.get('win_rate', 0):.1f}%</strong> (Target: 77.3%)</div>
        <div class="metric">Total Return: <strong>{performance.get('total_return', 0):+.2f}%</strong></div>
    </div>
"""

        html += """
    <div class="footer">
        <p><em>Automated notification from Proteus Trading Dashboard</em></p>
        <p><em>Dashboard: <a href="http://localhost:5000">http://localhost:5000</a></em></p>
    </div>
</body>
</html>
"""
        return html


if __name__ == '__main__':
    print("Testing SendGrid...")
    notifier = SendGridNotifier()

    if not SENDGRID_AVAILABLE:
        print("\nSendGrid not installed. To install:")
        print("  pip install sendgrid")
    elif not notifier.is_enabled():
        print("\nSendGrid not configured. Setup:")
        print("1. Sign up at https://sendgrid.com (free)")
        print("2. Get API key from Settings > API Keys")
        print("3. Add to email_config.json:")
        print('   "sendgrid_api_key": "YOUR_API_KEY_HERE"')
    else:
        notifier.send_test_email()

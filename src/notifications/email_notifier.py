"""
Email Notification System for Proteus Trading Dashboard

Sends daily scan results and signal alerts via email.
Requires email_config.json in project root.
"""

import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict, Optional


class EmailNotifier:
    """Sends email notifications for trading signals and daily summaries."""

    def __init__(self, config_path='email_config.json'):
        """
        Initialize email notifier.

        Args:
            config_path: Path to email configuration JSON file
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load email configuration from JSON file."""
        if not os.path.exists(self.config_path):
            print(f"[WARN] Email config not found at {self.config_path}")
            return {'enabled': False}

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            # Validate required fields
            required = ['recipient_email', 'smtp_server', 'smtp_port',
                       'sender_email', 'sender_password']

            for field in required:
                if field not in config or not config[field] or 'YOUR_' in str(config[field]):
                    print(f"[WARN] Email config incomplete - {field} not configured")
                    return {'enabled': False}

            return config

        except Exception as e:
            print(f"[ERROR] Failed to load email config: {e}")
            return {'enabled': False}

    def is_enabled(self) -> bool:
        """Check if email notifications are enabled and configured."""
        return self.config.get('enabled', False)

    def send_scan_notification(self, scan_status: str, signals: List[Dict],
                              performance: Dict = None) -> bool:
        """
        Send email notification after market scan.

        Args:
            scan_status: Status message from scan
            signals: List of signal dictionaries
            performance: Optional performance stats

        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.is_enabled():
            return False

        # Check if we should send (signals only or always)
        if self.config.get('send_on_signals_only', False) and len(signals) == 0:
            return False

        subject = self._create_subject(signals)
        body = self._create_body(scan_status, signals, performance)

        return self._send_email(subject, body)

    def send_test_email(self) -> bool:
        """
        Send a test email to verify configuration.

        Returns:
            True if email sent successfully, False otherwise
        """
        subject = "🧪 Proteus Trading Dashboard - Test Email"

        body = """
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; color: #333; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                 color: white; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }
        .success { color: #10b981; font-weight: bold; }
        .info { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="header">
        <h2>🧪 Proteus Trading Dashboard</h2>
        <p>Email Configuration Test</p>
    </div>

    <div class="section">
        <p class="success">✓ Email notifications are working correctly!</p>
        <p>Your Proteus trading dashboard is now configured to send automated notifications.</p>
    </div>

    <div class="section">
        <h3>What You'll Receive:</h3>
        <ul>
            <li><strong>Daily Scan Results:</strong> Signal alerts at 9:35 AM, 1:35 PM, 5:35 PM</li>
            <li><strong>Buy Signals:</strong> Immediate notifications when panic sell opportunities are detected</li>
            <li><strong>Performance Summary:</strong> Win rate, total return, and trade statistics</li>
            <li><strong>Market Regime:</strong> Current market status (BULL/BEAR/SIDEWAYS)</li>
        </ul>
    </div>

    <div class="section">
        <h3>Configuration Details:</h3>
        <p><strong>Recipient:</strong> {recipient}</p>
        <p><strong>SMTP Server:</strong> {smtp_server}:{smtp_port}</p>
        <p><strong>Send on Signals Only:</strong> {signals_only}</p>
        <p><strong>Daily Summary:</strong> {daily_summary}</p>
    </div>

    <div class="info">
        <p><em>This is an automated test message from your Proteus Trading Dashboard.</em></p>
        <p><em>Sent: {timestamp}</em></p>
    </div>
</body>
</html>
""".format(
            recipient=self.config['recipient_email'],
            smtp_server=self.config['smtp_server'],
            smtp_port=self.config['smtp_port'],
            signals_only="Yes" if self.config.get('send_on_signals_only', False) else "No",
            daily_summary="Yes" if self.config.get('send_daily_summary', True) else "No",
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )

        return self._send_email(subject, body, is_html=True)

    def _create_subject(self, signals: List[Dict]) -> str:
        """Create email subject line based on signals."""
        if len(signals) == 0:
            return "📊 Proteus Daily Scan - No Signals"
        elif len(signals) == 1:
            ticker = signals[0]['ticker']
            return f"🚨 Proteus Alert - BUY Signal: {ticker}"
        else:
            return f"🚨 Proteus Alert - {len(signals)} BUY Signals!"

    def _create_body(self, scan_status: str, signals: List[Dict],
                    performance: Dict = None) -> str:
        """Create formatted email body."""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Build HTML email
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
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .metric-value {{ font-weight: bold; color: #333; }}
        .success {{ color: #10b981; }}
        .warning {{ color: #f59e0b; }}
        .danger {{ color: #ef4444; }}
        .footer {{ color: #666; font-size: 0.85em; margin-top: 30px; padding-top: 20px;
                  border-top: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>📈 Proteus Trading Dashboard</h2>
        <p>AI-Powered Stock Predictor</p>
        <p style="font-size: 0.9em;">{now}</p>
    </div>

    <div class="status">
        <strong>Scan Status:</strong> {scan_status}
    </div>
"""

        # Add signals section
        if len(signals) > 0:
            html += f"""
    <h3>🚨 Active Buy Signals ({len(signals)})</h3>
"""
            for signal in signals:
                expected_return = signal.get('expected_return', 0)
                html += f"""
    <div class="signal">
        <div class="signal-header">{signal['ticker']}</div>
        <div class="signal-detail"><strong>Entry Price:</strong> ${signal['price']:.2f}</div>
        <div class="signal-detail"><strong>Z-Score:</strong> {signal['z_score']:.2f} (oversold)</div>
        <div class="signal-detail"><strong>RSI:</strong> {signal['rsi']:.1f} (oversold)</div>
        <div class="signal-detail"><strong>Expected Return:</strong> <span class="success">+{expected_return:.2f}%</span></div>
        <div class="signal-detail" style="margin-top: 10px;">
            <strong>Recommendation:</strong> Enter with 10% of capital (~${signal['price'] * 100:.0f} for 100 shares)
        </div>
    </div>
"""
        else:
            html += """
    <h3>💤 No Signals Detected</h3>
    <div class="status">
        <p>No panic sell opportunities found in current scan.</p>
        <p style="font-size: 0.9em; color: #666;">
            The strategy is highly selective (~1-2 signals per month). Continue monitoring.
        </p>
    </div>
"""

        # Add performance section if available
        if performance:
            html += f"""
    <h3>📊 Performance Summary</h3>
    <div class="performance">
        <div class="metric">
            <span class="metric-label">Total Trades:</span>
            <span class="metric-value">{performance.get('total_trades', 0)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Win Rate:</span>
            <span class="metric-value {'success' if performance.get('win_rate', 0) >= 75 else 'warning'}">
                {performance.get('win_rate', 0):.1f}%
            </span>
            <span style="font-size: 0.85em; color: #666;"> (Target: 77.3%)</span>
        </div>
        <div class="metric">
            <span class="metric-label">Total Return:</span>
            <span class="metric-value {'success' if performance.get('total_return', 0) > 0 else 'danger'}">
                {performance.get('total_return', 0):+.2f}%
            </span>
        </div>
        <div class="metric">
            <span class="metric-label">Days Tracked:</span>
            <span class="metric-value">{performance.get('days_tracked', 0)}</span>
        </div>
    </div>
"""

        # Add footer
        html += """
    <div class="footer">
        <p><strong>Next Steps:</strong></p>
        <ul>
            <li>Review signals in dashboard: <a href="http://localhost:5000">http://localhost:5000</a></li>
            <li>If signals present, consider entering positions with 10% capital per trade</li>
            <li>Monitor for profit target (+2%) or stop loss (-2%)</li>
            <li>Exit all positions within 2 days maximum</li>
        </ul>
        <p style="margin-top: 15px;">
            <em>This is an automated notification from Proteus Trading Dashboard.</em><br>
            <em>System: Multi-Strategy Predictor v15.0 | Expected Win Rate: 79.3% | Avg Hold: 1-2 days</em>
        </p>
    </div>
</body>
</html>
"""

        return html

    def _send_email(self, subject: str, body: str, is_html: bool = True) -> bool:
        """
        Send email via SMTP.

        Args:
            subject: Email subject line
            body: Email body content
            is_html: Whether body is HTML (default True)

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.config['sender_email']
            msg['To'] = self.config['recipient_email']

            # Attach body
            if is_html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))

            # Connect to SMTP server
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()  # Enable TLS

            # Login
            server.login(self.config['sender_email'], self.config['sender_password'])

            # Send email
            server.send_message(msg)
            server.quit()

            print(f"[EMAIL] Sent to {self.config['recipient_email']}: {subject}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to send email: {e}")
            import traceback
            traceback.print_exc()
            return False


def send_test_notification():
    """Send a test email notification."""
    notifier = EmailNotifier()

    if not notifier.is_enabled():
        print("Email notifications are not enabled or configured.")
        print(f"Please configure {notifier.config_path} with your email settings.")
        print("\nTo get a Gmail app password:")
        print("1. Go to https://myaccount.google.com/apppasswords")
        print("2. Sign in to your Google account")
        print("3. Create a new app password for 'Proteus Trading'")
        print("4. Copy the 16-character password to email_config.json")
        return False

    print("Sending test email...")
    success = notifier.send_test_email()

    if success:
        print(f"✓ Test email sent successfully to {notifier.config['recipient_email']}")
    else:
        print("✗ Failed to send test email. Check your configuration and credentials.")

    return success


if __name__ == '__main__':
    send_test_notification()

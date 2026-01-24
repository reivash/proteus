"""
Bearish Alert Service

Monitors for bearish market conditions using FastBearDetector
and sends email alerts when warning thresholds are crossed.

Alert Levels:
- WATCH: Monitor closely, alert once per 24 hours
- WARNING: Reduce exposure, alert every 4 hours
- CRITICAL: Maximum caution, alert every hour

Usage:
    service = BearishAlertService()
    signal = service.check_and_alert()
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Optional, Dict
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.fast_bear_detector import FastBearDetector, FastBearSignal

# Try to import notifier
try:
    from notifications.sendgrid_notifier import SendGridNotifier
    NOTIFIER_AVAILABLE = True
except ImportError:
    NOTIFIER_AVAILABLE = False

# Try to import webhook notifier
try:
    from notifications.webhook_notifier import WebhookNotifier
    WEBHOOK_AVAILABLE = True
except ImportError:
    WEBHOOK_AVAILABLE = False


@dataclass
class AlertState:
    """Tracks alert state to prevent spam."""
    last_alert_level: str
    last_alert_time: Optional[datetime]
    alerts_sent_today: int
    last_daily_summary: Optional[datetime] = None


class BearishAlertService:
    """
    Service that monitors for bearish conditions and sends email alerts.

    Cooldowns prevent alert spam:
    - WATCH: Max 1 email per 24 hours
    - WARNING: Max 1 email per 4 hours
    - CRITICAL: Max 1 email per hour
    - Escalations always trigger immediately
    """

    COOLDOWNS = {
        'WATCH': timedelta(hours=24),
        'WARNING': timedelta(hours=4),
        'CRITICAL': timedelta(hours=1),
        'NORMAL': timedelta(hours=24)  # No alerts for normal
    }

    ALERT_PRIORITY = {
        'NORMAL': 0,
        'WATCH': 1,
        'WARNING': 2,
        'CRITICAL': 3
    }

    def __init__(self, state_file: str = 'data/bear_alert_state.json'):
        self.fast_bear = FastBearDetector()
        self.state_file = state_file
        self.state = self._load_state()

        if NOTIFIER_AVAILABLE:
            self.notifier = SendGridNotifier()
        else:
            self.notifier = None

        # Initialize webhook notifier
        if WEBHOOK_AVAILABLE:
            self.webhook = WebhookNotifier()
        else:
            self.webhook = None

    def _load_state(self) -> AlertState:
        """Load alert state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                return AlertState(
                    last_alert_level=data.get('last_alert_level', 'NORMAL'),
                    last_alert_time=datetime.fromisoformat(data['last_alert_time']) if data.get('last_alert_time') else None,
                    alerts_sent_today=data.get('alerts_sent_today', 0),
                    last_daily_summary=datetime.fromisoformat(data['last_daily_summary']) if data.get('last_daily_summary') else None
                )
            except Exception as e:
                print(f"[BearAlert] Failed to load state: {e}")

        return AlertState(
            last_alert_level='NORMAL',
            last_alert_time=None,
            alerts_sent_today=0
        )

    def _save_state(self):
        """Save alert state to file."""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump({
                    'last_alert_level': self.state.last_alert_level,
                    'last_alert_time': self.state.last_alert_time.isoformat() if self.state.last_alert_time else None,
                    'alerts_sent_today': self.state.alerts_sent_today,
                    'last_daily_summary': self.state.last_daily_summary.isoformat() if self.state.last_daily_summary else None
                }, f)
        except Exception as e:
            print(f"[BearAlert] Failed to save state: {e}")

    def check_and_alert(self, force_alert: bool = False) -> FastBearSignal:
        """
        Check market conditions and send alert if warranted.

        Args:
            force_alert: If True, send alert regardless of cooldowns

        Returns:
            FastBearSignal with current market assessment
        """
        signal = self.fast_bear.detect()

        # Check for "all clear" - score dropped from elevated to normal
        if self._should_send_all_clear(signal):
            success = self._send_all_clear_alert(signal)
            if success:
                self.state.last_alert_level = signal.alert_level
                self.state.last_alert_time = datetime.now()
                self._save_state()
            return signal

        # Determine if we should alert
        should_alert = force_alert or self._should_send_alert(signal)

        if should_alert and signal.alert_level != 'NORMAL':
            success = self._send_bear_alert(signal)
            if success:
                self.state.last_alert_level = signal.alert_level
                self.state.last_alert_time = datetime.now()
                self.state.alerts_sent_today += 1
                self._save_state()

        return signal

    def _should_send_all_clear(self, signal: FastBearSignal) -> bool:
        """Check if we should send an 'all clear' notification."""
        # Only send all clear if:
        # 1. Previous level was WARNING or CRITICAL
        # 2. Current level is NORMAL
        # 3. At least 4 hours since last alert (don't spam on fluctuations)
        if signal.alert_level != 'NORMAL':
            return False

        if self.state.last_alert_level not in ['WARNING', 'CRITICAL']:
            return False

        if self.state.last_alert_time:
            time_since = datetime.now() - self.state.last_alert_time
            if time_since < timedelta(hours=4):
                return False

        return True

    def _send_all_clear_alert(self, signal: FastBearSignal) -> bool:
        """Send 'all clear' notification when bear score normalizes."""
        subject = f"✅ ALL CLEAR: Bear Score Normalized ({signal.bear_score:.0f}/100) - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        html_body = f"""
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #1a1a1a; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                 color: white; padding: 25px; border-radius: 10px; margin-bottom: 25px; }}
        .section {{ background: #f0fdf4; padding: 20px; border-radius: 8px; margin: 15px 0;
                   border-left: 5px solid #10b981; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f1f5f9; font-weight: 600; }}
        .footer {{ color: #666; font-size: 0.9em; margin-top: 30px; padding-top: 20px;
                  border-top: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="header">
        <h1 style="margin: 0; font-size: 1.8em;">ALL CLEAR</h1>
        <p style="font-size: 1.3em; margin: 10px 0;">Bear Score Normalized: {signal.bear_score:.0f}/100</p>
        <p style="opacity: 0.9; margin: 5px 0;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h3 style="color: #059669; margin-top: 0;">Market Conditions Have Improved</h3>
        <p style="font-size: 1.1em;">The bear score has dropped from elevated levels back to <strong>NORMAL</strong>.</p>
        <p>Previous alert level: <strong>{self.state.last_alert_level}</strong></p>
        <p>Current alert level: <strong>{signal.alert_level}</strong></p>
    </div>

    <div class="section" style="background: #f8f9fa; border-left-color: #666;">
        <h3 style="color: #1a1a1a; margin-top: 0;">Current Market Metrics</h3>
        <table>
            <tr>
                <td>SPY 3-Day ROC</td>
                <td>{signal.spy_roc_3d:+.2f}%</td>
            </tr>
            <tr>
                <td>VIX Level</td>
                <td>{signal.vix_level:.1f}</td>
            </tr>
            <tr>
                <td>Market Breadth</td>
                <td>{signal.market_breadth_pct:.1f}% above MA</td>
            </tr>
            <tr>
                <td>Sectors Declining</td>
                <td>{signal.sectors_declining}/{signal.sectors_total}</td>
            </tr>
            <tr>
                <td>Yield Curve</td>
                <td>{signal.yield_curve_spread:+.2f}%</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h3 style="color: #059669; margin-top: 0;">Recommendation</h3>
        <p style="font-size: 1.05em;">{signal.recommendation}</p>
        <p style="margin-top: 15px;"><strong>Position sizing has returned to normal levels.</strong></p>
    </div>

    <div class="footer">
        <p><em>Automated alert from Proteus Fast Bear Detector</em></p>
        <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    </div>
</body>
</html>
"""
        email_sent = self._send_email(subject, html_body)

        # Also send webhook all-clear
        webhook_sent = False
        if self.webhook and self.webhook.is_enabled():
            try:
                results = self.webhook.send_all_clear(signal)
                if results:
                    successful = [k for k, v in results.items() if v]
                    if successful:
                        print(f"[WEBHOOK] All clear sent to: {', '.join(successful)}")
                        webhook_sent = True
            except Exception as e:
                print(f"[WEBHOOK] All clear error: {e}")

        return email_sent or webhook_sent

    def _should_send_alert(self, signal: FastBearSignal) -> bool:
        """Determine if an alert should be sent based on cooldowns."""
        # No alerts for normal conditions
        if signal.alert_level == 'NORMAL':
            return False

        current_priority = self.ALERT_PRIORITY.get(signal.alert_level, 0)
        last_priority = self.ALERT_PRIORITY.get(self.state.last_alert_level, 0)

        # Always alert on escalation
        if current_priority > last_priority:
            print(f"[BearAlert] Escalation detected: {self.state.last_alert_level} -> {signal.alert_level}")
            return True

        # Check cooldown
        if self.state.last_alert_time:
            cooldown = self.COOLDOWNS.get(signal.alert_level, timedelta(hours=24))
            time_since_last = datetime.now() - self.state.last_alert_time

            if time_since_last < cooldown:
                remaining = cooldown - time_since_last
                print(f"[BearAlert] Cooldown active. Next alert in {remaining}")
                return False

        return True

    def _send_bear_alert(self, signal: FastBearSignal) -> bool:
        """Send formatted bear alert via email and webhooks."""
        email_sent = False
        webhook_sent = False

        # Send email
        if self.notifier:
            subject = self._create_subject(signal)
            html_body = self._create_html_body(signal)

            try:
                if hasattr(self.notifier, 'send_bear_alert'):
                    email_sent = self.notifier.send_bear_alert(subject, html_body)
                else:
                    from sendgrid import SendGridAPIClient
                    from sendgrid.helpers.mail import Mail, Email, To, Content

                    if self.notifier.is_enabled():
                        message = Mail(
                            from_email=Email(self.notifier.config.get('sender_email', 'proteus@trading.local')),
                            to_emails=To(self.notifier.config['recipient_email']),
                            subject=subject,
                            html_content=Content("text/html", html_body)
                        )

                        sg = SendGridAPIClient(self.notifier.config['sendgrid_api_key'])
                        response = sg.send(message)
                        print(f"[EMAIL] Bear alert sent: {subject}")
                        email_sent = True

            except Exception as e:
                print(f"[BearAlert] Failed to send email: {e}")

        # Send webhooks
        if self.webhook and self.webhook.is_enabled():
            try:
                results = self.webhook.send_bear_alert(signal)
                if results:
                    successful = [k for k, v in results.items() if v]
                    if successful:
                        print(f"[WEBHOOK] Alert sent to: {', '.join(successful)}")
                        webhook_sent = True
                    failed = [k for k, v in results.items() if not v]
                    if failed:
                        print(f"[WEBHOOK] Failed: {', '.join(failed)}")
            except Exception as e:
                print(f"[WEBHOOK] Error: {e}")

        # Return True if at least one notification was sent
        if not email_sent and not webhook_sent:
            print("[BearAlert] No notifications available, printing to console")
            print(self.fast_bear.get_detailed_report())
            return False

        return True

    def _create_subject(self, signal: FastBearSignal) -> str:
        """Create email subject line."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

        level_emoji = {
            'CRITICAL': '🚨',
            'WARNING': '⚠️',
            'WATCH': '👀',
            'NORMAL': '✅'
        }

        emoji = level_emoji.get(signal.alert_level, '📊')

        return f"{emoji} BEARISH ALERT: {signal.alert_level} - Bear Score {signal.bear_score}/100 - {timestamp}"

    def _create_html_body(self, signal: FastBearSignal) -> str:
        """Create HTML email body."""
        level_colors = {
            'CRITICAL': '#dc2626',
            'WARNING': '#f59e0b',
            'WATCH': '#3b82f6',
            'NORMAL': '#10b981'
        }

        color = level_colors.get(signal.alert_level, '#666')

        # Build triggers list
        triggers_html = ""
        for trigger in signal.triggers:
            triggers_html += f"<li>{trigger}</li>"

        html = f"""
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #1a1a1a; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, {color} 0%, {color}dd 100%);
                 color: white; padding: 25px; border-radius: 10px; margin-bottom: 25px; }}
        .section {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0;
                   border-left: 5px solid {color}; }}
        .metric {{ margin: 10px 0; }}
        .metric-label {{ color: #666; font-weight: 500; }}
        .value-bad {{ color: #dc2626; font-weight: bold; }}
        .value-warn {{ color: #f59e0b; font-weight: bold; }}
        .value-ok {{ color: #10b981; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f1f5f9; font-weight: 600; }}
        .footer {{ color: #666; font-size: 0.9em; margin-top: 30px; padding-top: 20px;
                  border-top: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="header">
        <h1 style="margin: 0; font-size: 1.8em;">BEARISH ALERT: {signal.alert_level}</h1>
        <p style="font-size: 1.3em; margin: 10px 0;">Bear Score: {signal.bear_score}/100</p>
        <p style="opacity: 0.9; margin: 5px 0;">{signal.timestamp}</p>
    </div>

    <div class="section" style="background: #fff5f5;">
        <h3 style="color: {color}; margin-top: 0;">Triggered Indicators</h3>
        <ul style="margin: 0; padding-left: 25px;">
            {triggers_html if triggers_html else '<li>No specific triggers</li>'}
        </ul>
    </div>

    <div class="section">
        <h3 style="color: #1a1a1a; margin-top: 0;">Market Metrics</h3>
        <table>
            <tr>
                <th>Indicator</th>
                <th>Value</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>SPY 3-Day ROC</td>
                <td>{signal.spy_roc_3d:+.2f}%</td>
                <td class="{'value-bad' if signal.spy_roc_3d < -3 else 'value-warn' if signal.spy_roc_3d < -2 else 'value-ok'}"">
                    {'CRITICAL' if signal.spy_roc_3d < -5 else 'WARNING' if signal.spy_roc_3d < -3 else 'WATCH' if signal.spy_roc_3d < -2 else 'OK'}
                </td>
            </tr>
            <tr>
                <td>VIX Level</td>
                <td>{signal.vix_level:.1f}</td>
                <td class="{'value-bad' if signal.vix_level > 30 else 'value-warn' if signal.vix_level > 25 else 'value-ok'}">
                    {'CRITICAL' if signal.vix_level > 35 else 'WARNING' if signal.vix_level > 30 else 'ELEVATED' if signal.vix_level > 25 else 'NORMAL'}
                </td>
            </tr>
            <tr>
                <td>VIX 2-Day Spike</td>
                <td>{signal.vix_spike_pct:+.1f}%</td>
                <td class="{'value-bad' if signal.vix_spike_pct > 30 else 'value-warn' if signal.vix_spike_pct > 20 else 'value-ok'}">
                    {'CRITICAL' if signal.vix_spike_pct > 50 else 'WARNING' if signal.vix_spike_pct > 30 else 'ELEVATED' if signal.vix_spike_pct > 20 else 'NORMAL'}
                </td>
            </tr>
            <tr>
                <td>Market Breadth</td>
                <td>{signal.market_breadth_pct:.1f}% above MA</td>
                <td class="{'value-bad' if signal.market_breadth_pct < 30 else 'value-warn' if signal.market_breadth_pct < 40 else 'value-ok'}">
                    {'CRITICAL' if signal.market_breadth_pct < 20 else 'WARNING' if signal.market_breadth_pct < 30 else 'WEAK' if signal.market_breadth_pct < 40 else 'HEALTHY'}
                </td>
            </tr>
            <tr>
                <td>Sectors Declining</td>
                <td>{signal.sectors_declining}/{signal.sectors_total}</td>
                <td class="{'value-bad' if signal.sectors_declining >= 8 else 'value-warn' if signal.sectors_declining >= 6 else 'value-ok'}">
                    {'CRITICAL' if signal.sectors_declining >= 10 else 'WARNING' if signal.sectors_declining >= 8 else 'ELEVATED' if signal.sectors_declining >= 6 else 'NORMAL'}
                </td>
            </tr>
            <tr>
                <td>Volume Confirmation</td>
                <td>{'YES' if signal.volume_confirmation else 'NO'}</td>
                <td class="{'value-bad' if signal.volume_confirmation else 'value-ok'}">
                    {'HIGH VOLUME SELLING' if signal.volume_confirmation else 'NORMAL'}
                </td>
            </tr>
            <tr>
                <td>Yield Curve (10Y-2Y)</td>
                <td>{signal.yield_curve_spread:+.2f}%</td>
                <td class="{'value-bad' if signal.yield_curve_spread <= 0 else 'value-warn' if signal.yield_curve_spread < 0.25 else 'value-ok'}">
                    {'INVERTED' if signal.yield_curve_spread <= -0.25 else 'FLAT' if signal.yield_curve_spread <= 0.25 else 'NORMAL'}
                </td>
            </tr>
            <tr>
                <td>Credit Spread (5d)</td>
                <td>{signal.credit_spread_change:+.2f}%</td>
                <td class="{'value-bad' if signal.credit_spread_change >= 10 else 'value-warn' if signal.credit_spread_change >= 5 else 'value-ok'}">
                    {'STRESSED' if signal.credit_spread_change >= 20 else 'WIDENING' if signal.credit_spread_change >= 5 else 'NORMAL'}
                </td>
            </tr>
            <tr>
                <td>Momentum Divergence</td>
                <td>{'YES' if signal.momentum_divergence else 'NO'}</td>
                <td class="{'value-bad' if signal.momentum_divergence else 'value-ok'}">
                    {'TOPPING PATTERN' if signal.momentum_divergence else 'NORMAL'}
                </td>
            </tr>
        </table>
    </div>

    <div class="section" style="background: #eff6ff; border-left-color: #3b82f6;">
        <h3 style="color: #1d4ed8; margin-top: 0;">Recommendation</h3>
        <p style="font-size: 1.05em; margin: 0;">{signal.recommendation}</p>
    </div>

    <div class="section" style="background: #f0fdf4; border-left-color: #10b981;">
        <h3 style="color: #059669; margin-top: 0;">Score Breakdown</h3>
        <p style="margin: 5px 0;"><strong>Bear Score:</strong> {signal.bear_score}/100</p>
        <p style="margin: 5px 0;"><strong>Confidence:</strong> {signal.confidence * 100:.0f}%</p>
        <p style="margin: 5px 0;"><strong>Active Triggers:</strong> {len(signal.triggers)}</p>

        <h4 style="margin-top: 15px; margin-bottom: 10px;">Score Components (max 100):</h4>
        <ul style="margin: 0; padding-left: 25px;">
            <li>VIX Level/Spike: up to 22 points (fear indicator)</li>
            <li>SPY 3d ROC: up to 18 points (momentum)</li>
            <li>Yield Curve: up to 14 points (recession predictor)</li>
            <li>Market Breadth: up to 13 points (participation)</li>
            <li>Credit Spread: up to 10 points (corporate stress)</li>
            <li>Sector Breadth: up to 10 points (confirmation)</li>
            <li>Momentum Divergence: up to 8 points (topping pattern)</li>
            <li>Volume Confirmation: up to 5 points (conviction)</li>
        </ul>
    </div>

    <div class="footer">
        <p><em>Automated alert from Proteus Fast Bear Detector</em></p>
        <p><em>Alert Cooldowns: WATCH=24h, WARNING=4h, CRITICAL=1h</em></p>
        <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    </div>
</body>
</html>
"""
        return html

    def get_status(self) -> Dict:
        """Get current alert service status."""
        # Get webhook platforms if available
        webhook_platforms = []
        if self.webhook and self.webhook.is_enabled():
            from notifications.webhook_notifier import get_webhook_status
            webhook_status = get_webhook_status()
            webhook_platforms = webhook_status.get('platforms', [])

        return {
            'last_alert_level': self.state.last_alert_level,
            'last_alert_time': self.state.last_alert_time.isoformat() if self.state.last_alert_time else None,
            'alerts_sent_today': self.state.alerts_sent_today,
            'last_daily_summary': self.state.last_daily_summary.isoformat() if self.state.last_daily_summary else None,
            'notifier_available': self.notifier is not None and self.notifier.is_enabled() if self.notifier else False,
            'webhook_available': self.webhook is not None and self.webhook.is_enabled() if self.webhook else False,
            'webhook_platforms': webhook_platforms
        }

    def send_daily_summary(self, force: bool = False) -> bool:
        """
        Send daily market conditions summary email.

        This sends regardless of alert level, providing a consistent daily update.

        Args:
            force: If True, send even if already sent today

        Returns:
            True if email was sent
        """
        # Check if we should send
        if not force and not self._should_send_daily_summary():
            return False

        signal = self.fast_bear.detect()

        # Try to get trend data
        try:
            from analysis.fast_bear_detector import get_bear_score_trend
            trend_data = get_bear_score_trend(hours=24)
        except Exception:
            trend_data = None

        subject = self._create_daily_summary_subject(signal)
        html_body = self._create_daily_summary_html(signal, trend_data)

        success = self._send_email(subject, html_body)

        if success:
            self.state.last_daily_summary = datetime.now()
            self._save_state()
            print(f"[BearAlert] Daily summary sent: Bear Score {signal.bear_score}/100")

        return success

    def _should_send_daily_summary(self) -> bool:
        """Check if daily summary should be sent (once per day)."""
        if not self.state.last_daily_summary:
            return True

        now = datetime.now()
        last = self.state.last_daily_summary

        # Send if different day
        if last.date() < now.date():
            return True

        return False

    def _send_email(self, subject: str, html_body: str) -> bool:
        """Send email using notifier."""
        if not self.notifier:
            print("[BearAlert] Notifier not available")
            return False

        try:
            if hasattr(self.notifier, 'send_bear_alert'):
                return self.notifier.send_bear_alert(subject, html_body)
            else:
                from sendgrid import SendGridAPIClient
                from sendgrid.helpers.mail import Mail, Email, To, Content

                if not self.notifier.is_enabled():
                    print("[BearAlert] Email not enabled")
                    return False

                message = Mail(
                    from_email=Email(self.notifier.config.get('sender_email', 'proteus@trading.local')),
                    to_emails=To(self.notifier.config['recipient_email']),
                    subject=subject,
                    html_content=Content("text/html", html_body)
                )

                sg = SendGridAPIClient(self.notifier.config['sendgrid_api_key'])
                response = sg.send(message)
                print(f"[EMAIL] Sent: {subject}")
                return True

        except Exception as e:
            print(f"[BearAlert] Failed to send email: {e}")
            return False

    def _create_daily_summary_subject(self, signal: FastBearSignal) -> str:
        """Create daily summary subject line."""
        date_str = datetime.now().strftime('%Y-%m-%d')

        level_emoji = {
            'CRITICAL': '🚨',
            'WARNING': '⚠️',
            'WATCH': '👀',
            'NORMAL': '✅'
        }

        emoji = level_emoji.get(signal.alert_level, '📊')

        return f"{emoji} Daily Market Summary - {signal.alert_level} (Score: {signal.bear_score}/100) - {date_str}"

    def _create_daily_summary_html(self, signal: FastBearSignal, trend_data: Optional[Dict] = None) -> str:
        """Create HTML for daily summary email."""
        level_colors = {
            'CRITICAL': '#dc2626',
            'WARNING': '#f59e0b',
            'WATCH': '#3b82f6',
            'NORMAL': '#10b981'
        }

        color = level_colors.get(signal.alert_level, '#666')

        # Build triggers list
        triggers_html = ""
        for trigger in signal.triggers:
            triggers_html += f"<li>{trigger}</li>"

        # Build trend section if available
        trend_html = ""
        if trend_data and 'scores' in trend_data and len(trend_data['scores']) > 0:
            scores = trend_data['scores']
            min_score = min(scores)
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            change = scores[-1] - scores[0] if len(scores) > 1 else 0

            trend_direction = "↑ Deteriorating" if change > 5 else "↓ Improving" if change < -5 else "→ Stable"
            trend_color = "#dc2626" if change > 5 else "#10b981" if change < -5 else "#666"

            trend_html = f"""
            <div class="section" style="background: #fefce8; border-left-color: #eab308;">
                <h3 style="color: #854d0e; margin-top: 0;">24-Hour Trend</h3>
                <table>
                    <tr>
                        <td>Range</td>
                        <td>{min_score:.0f} - {max_score:.0f}</td>
                    </tr>
                    <tr>
                        <td>Average</td>
                        <td>{avg_score:.1f}</td>
                    </tr>
                    <tr>
                        <td>Direction</td>
                        <td style="color: {trend_color}; font-weight: bold;">{trend_direction} ({change:+.1f})</td>
                    </tr>
                    <tr>
                        <td>Data Points</td>
                        <td>{len(scores)} readings</td>
                    </tr>
                </table>
            </div>
            """

        html = f"""
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #1a1a1a; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
                 color: white; padding: 25px; border-radius: 10px; margin-bottom: 25px; }}
        .status-badge {{ display: inline-block; background: {color}; color: white; padding: 8px 16px;
                        border-radius: 20px; font-weight: bold; font-size: 0.9em; }}
        .section {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0;
                   border-left: 5px solid {color}; }}
        .metric {{ margin: 10px 0; }}
        .value-bad {{ color: #dc2626; font-weight: bold; }}
        .value-warn {{ color: #f59e0b; font-weight: bold; }}
        .value-ok {{ color: #10b981; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f1f5f9; font-weight: 600; }}
        .footer {{ color: #666; font-size: 0.9em; margin-top: 30px; padding-top: 20px;
                  border-top: 1px solid #ddd; }}
        .score-big {{ font-size: 2.5em; font-weight: bold; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1 style="margin: 0; font-size: 1.8em;">Daily Market Conditions</h1>
        <p style="opacity: 0.9; margin: 10px 0 15px 0;">{datetime.now().strftime('%A, %B %d, %Y')}</p>
        <span class="status-badge">{signal.alert_level}</span>
    </div>

    <div class="section" style="text-align: center; background: #fff;">
        <p style="color: #666; margin: 0 0 5px 0;">Bear Score</p>
        <div class="score-big" style="color: {color};">{signal.bear_score}/100</div>
        <p style="color: #666; margin: 5px 0 0 0;">Confidence: {signal.confidence * 100:.0f}%</p>
    </div>

    {'<div class="section" style="background: #fff5f5;"><h3 style="color: #dc2626; margin-top: 0;">Active Warnings</h3><ul style="margin: 0; padding-left: 25px;">' + triggers_html + '</ul></div>' if triggers_html else ''}

    {trend_html}

    <div class="section">
        <h3 style="color: #1a1a1a; margin-top: 0;">Market Metrics</h3>
        <table>
            <tr>
                <th>Indicator</th>
                <th>Value</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>SPY 3-Day ROC</td>
                <td>{signal.spy_roc_3d:+.2f}%</td>
                <td class="{'value-bad' if signal.spy_roc_3d < -3 else 'value-warn' if signal.spy_roc_3d < -2 else 'value-ok'}">
                    {'CRITICAL' if signal.spy_roc_3d < -5 else 'WARNING' if signal.spy_roc_3d < -3 else 'WATCH' if signal.spy_roc_3d < -2 else 'OK'}
                </td>
            </tr>
            <tr>
                <td>VIX Level</td>
                <td>{signal.vix_level:.1f} ({signal.vix_spike_pct:+.1f}% 2d)</td>
                <td class="{'value-bad' if signal.vix_level > 30 else 'value-warn' if signal.vix_level > 25 else 'value-ok'}">
                    {'CRITICAL' if signal.vix_level > 35 else 'WARNING' if signal.vix_level > 30 else 'ELEVATED' if signal.vix_level > 25 else 'NORMAL'}
                </td>
            </tr>
            <tr>
                <td>Market Breadth</td>
                <td>{signal.market_breadth_pct:.1f}% above MA</td>
                <td class="{'value-bad' if signal.market_breadth_pct < 30 else 'value-warn' if signal.market_breadth_pct < 40 else 'value-ok'}">
                    {'CRITICAL' if signal.market_breadth_pct < 20 else 'WARNING' if signal.market_breadth_pct < 30 else 'WEAK' if signal.market_breadth_pct < 40 else 'HEALTHY'}
                </td>
            </tr>
            <tr>
                <td>Sectors Declining</td>
                <td>{signal.sectors_declining}/{signal.sectors_total}</td>
                <td class="{'value-bad' if signal.sectors_declining >= 8 else 'value-warn' if signal.sectors_declining >= 6 else 'value-ok'}">
                    {'CRITICAL' if signal.sectors_declining >= 10 else 'WARNING' if signal.sectors_declining >= 8 else 'ELEVATED' if signal.sectors_declining >= 6 else 'NORMAL'}
                </td>
            </tr>
            <tr>
                <td>Yield Curve (10Y-2Y)</td>
                <td>{signal.yield_curve_spread:+.2f}%</td>
                <td class="{'value-bad' if signal.yield_curve_spread <= 0 else 'value-warn' if signal.yield_curve_spread < 0.25 else 'value-ok'}">
                    {'INVERTED' if signal.yield_curve_spread <= -0.25 else 'FLAT' if signal.yield_curve_spread <= 0.25 else 'NORMAL'}
                </td>
            </tr>
            <tr>
                <td>Credit Spread (5d)</td>
                <td>{signal.credit_spread_change:+.2f}%</td>
                <td class="{'value-bad' if signal.credit_spread_change >= 10 else 'value-warn' if signal.credit_spread_change >= 5 else 'value-ok'}">
                    {'STRESSED' if signal.credit_spread_change >= 20 else 'WIDENING' if signal.credit_spread_change >= 5 else 'NORMAL'}
                </td>
            </tr>
            <tr>
                <td>Volume Confirmation</td>
                <td>{'YES' if signal.volume_confirmation else 'NO'}</td>
                <td class="{'value-bad' if signal.volume_confirmation else 'value-ok'}">
                    {'HIGH SELLING' if signal.volume_confirmation else 'NORMAL'}
                </td>
            </tr>
            <tr>
                <td>Momentum Divergence</td>
                <td>{'YES' if signal.momentum_divergence else 'NO'}</td>
                <td class="{'value-bad' if signal.momentum_divergence else 'value-ok'}">
                    {'TOPPING' if signal.momentum_divergence else 'NORMAL'}
                </td>
            </tr>
        </table>
    </div>

    <div class="section" style="background: #eff6ff; border-left-color: #3b82f6;">
        <h3 style="color: #1d4ed8; margin-top: 0;">Recommendation</h3>
        <p style="font-size: 1.05em; margin: 0;">{signal.recommendation}</p>
    </div>

    <div class="footer">
        <p><strong>Alert Level Reference:</strong></p>
        <p style="margin: 5px 0;">NORMAL (0-29): Standard conditions | WATCH (30-49): Monitor closely</p>
        <p style="margin: 5px 0;">WARNING (50-69): Reduce exposure | CRITICAL (70+): Maximum caution</p>
        <p style="margin-top: 15px;"><em>Daily summary from Proteus Bear Market Monitor</em></p>
        <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    </div>
</body>
</html>
"""
        return html


def check_bear_conditions(send_alert: bool = True) -> FastBearSignal:
    """
    Quick function to check current bear conditions.

    Args:
        send_alert: If True, send email alert if warranted

    Returns:
        FastBearSignal with current assessment
    """
    service = BearishAlertService()

    if send_alert:
        return service.check_and_alert()
    else:
        return service.fast_bear.detect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Bearish Alert Service')
    parser.add_argument('--test', action='store_true', help='Force send test alert')
    parser.add_argument('--status', action='store_true', help='Show service status')
    parser.add_argument('--report', action='store_true', help='Print detailed report')
    args = parser.parse_args()

    service = BearishAlertService()

    if args.status:
        status = service.get_status()
        print("Bearish Alert Service Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")

    elif args.test:
        print("Forcing test alert...")
        signal = service.check_and_alert(force_alert=True)
        print(f"Alert Level: {signal.alert_level}")
        print(f"Bear Score: {signal.bear_score}")

    elif args.report:
        print(service.fast_bear.get_detailed_report())

    else:
        signal = service.check_and_alert()
        print(f"Bear Score: {signal.bear_score}/100 ({signal.alert_level})")
        if signal.triggers:
            print(f"Triggers: {', '.join(signal.triggers)}")

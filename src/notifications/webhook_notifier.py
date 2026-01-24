"""
Webhook Notifier - Send alerts to Discord, Slack, or custom webhooks.

Supports:
- Discord webhooks
- Slack webhooks
- Generic HTTP POST webhooks
- Pushover (mobile push notifications)

Configuration in email_config.json:
{
    "webhooks": {
        "discord": "https://discord.com/api/webhooks/...",
        "slack": "https://hooks.slack.com/services/...",
        "pushover": {
            "user_key": "your_user_key",
            "api_token": "your_api_token"
        },
        "custom": [
            {"url": "https://your-webhook.com/alert", "method": "POST"}
        ]
    }
}
"""

import json
import os
from typing import Dict, Optional, List
from datetime import datetime
import urllib.request
import urllib.error


class WebhookNotifier:
    """Send notifications via webhooks to various platforms."""

    def __init__(self, config_path: str = 'email_config.json'):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load webhook configuration."""
        if not os.path.exists(self.config_path):
            return {}

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config.get('webhooks', {})
        except Exception as e:
            print(f"[WEBHOOK] Config error: {e}")
            return {}

    def is_enabled(self) -> bool:
        """Check if any webhooks are configured."""
        return bool(self.config)

    def send_bear_alert(self, signal, force: bool = False) -> Dict[str, bool]:
        """
        Send bear alert to all configured webhooks.

        Args:
            signal: FastBearSignal object
            force: Send regardless of alert level

        Returns:
            Dict of {platform: success} results
        """
        # Only send for WARNING or CRITICAL unless forced
        if not force and signal.alert_level not in ['WARNING', 'CRITICAL']:
            return {}

        results = {}

        # Discord
        if 'discord' in self.config:
            results['discord'] = self._send_discord(signal)

        # Slack
        if 'slack' in self.config:
            results['slack'] = self._send_slack(signal)

        # Pushover
        if 'pushover' in self.config:
            results['pushover'] = self._send_pushover(signal)

        # Custom webhooks
        if 'custom' in self.config:
            for i, webhook in enumerate(self.config['custom']):
                results[f'custom_{i}'] = self._send_custom(signal, webhook)

        return results

    def _send_discord(self, signal) -> bool:
        """Send alert to Discord webhook."""
        webhook_url = self.config.get('discord')
        if not webhook_url:
            return False

        # Color based on alert level
        colors = {
            'CRITICAL': 0xDC2626,  # Red
            'WARNING': 0xF59E0B,   # Orange
            'WATCH': 0x3B82F6,     # Blue
            'NORMAL': 0x10B981    # Green
        }

        # Build embed
        embed = {
            "title": f"Bear Alert: {signal.alert_level}",
            "description": signal.recommendation,
            "color": colors.get(signal.alert_level, 0x666666),
            "fields": [
                {"name": "Bear Score", "value": f"{signal.bear_score:.0f}/100", "inline": True},
                {"name": "VIX", "value": f"{signal.vix_level:.1f}", "inline": True},
                {"name": "SPY 3d", "value": f"{signal.spy_roc_3d:+.2f}%", "inline": True},
                {"name": "Breadth", "value": f"{signal.market_breadth_pct:.0f}%", "inline": True},
                {"name": "Yield Curve", "value": f"{signal.yield_curve_spread:+.2f}%", "inline": True},
                {"name": "Sectors Down", "value": f"{signal.sectors_declining}/{signal.sectors_total}", "inline": True},
            ],
            "footer": {"text": "Proteus Bear Detection"},
            "timestamp": datetime.utcnow().isoformat()
        }

        # Add triggers if present
        if signal.triggers:
            triggers_text = "\n".join([f"• {t}" for t in signal.triggers[:5]])
            embed["fields"].append({
                "name": "Active Triggers",
                "value": triggers_text,
                "inline": False
            })

        payload = {
            "embeds": [embed],
            "username": "Proteus Bear Monitor"
        }

        return self._post_json(webhook_url, payload)

    def _send_slack(self, signal) -> bool:
        """Send alert to Slack webhook."""
        webhook_url = self.config.get('slack')
        if not webhook_url:
            return False

        # Emoji based on alert level
        emojis = {
            'CRITICAL': ':rotating_light:',
            'WARNING': ':warning:',
            'WATCH': ':eyes:',
            'NORMAL': ':white_check_mark:'
        }

        # Build Slack message
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emojis.get(signal.alert_level, '')} Bear Alert: {signal.alert_level}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Bear Score:* {signal.bear_score:.0f}/100\n{signal.recommendation}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*VIX:* {signal.vix_level:.1f}"},
                    {"type": "mrkdwn", "text": f"*SPY 3d:* {signal.spy_roc_3d:+.2f}%"},
                    {"type": "mrkdwn", "text": f"*Breadth:* {signal.market_breadth_pct:.0f}%"},
                    {"type": "mrkdwn", "text": f"*Yield Curve:* {signal.yield_curve_spread:+.2f}%"},
                ]
            }
        ]

        # Add triggers
        if signal.triggers:
            triggers_text = "\n".join([f"• {t}" for t in signal.triggers[:5]])
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Triggers:*\n{triggers_text}"}
            })

        payload = {"blocks": blocks}

        return self._post_json(webhook_url, payload)

    def _send_pushover(self, signal) -> bool:
        """Send alert to Pushover (mobile push notifications)."""
        pushover_config = self.config.get('pushover', {})
        user_key = pushover_config.get('user_key')
        api_token = pushover_config.get('api_token')

        if not user_key or not api_token:
            return False

        # Priority based on alert level
        priorities = {
            'CRITICAL': 1,   # High priority
            'WARNING': 0,    # Normal
            'WATCH': -1,     # Low
            'NORMAL': -2     # Lowest
        }

        message = (
            f"Bear Score: {signal.bear_score:.0f}/100\n"
            f"VIX: {signal.vix_level:.1f} | SPY 3d: {signal.spy_roc_3d:+.2f}%\n"
            f"{signal.recommendation}"
        )

        payload = {
            'token': api_token,
            'user': user_key,
            'title': f'Bear Alert: {signal.alert_level}',
            'message': message,
            'priority': priorities.get(signal.alert_level, 0),
            'sound': 'siren' if signal.alert_level == 'CRITICAL' else 'pushover'
        }

        # Pushover uses form-encoded data, not JSON
        data = urllib.parse.urlencode(payload).encode('utf-8')

        try:
            req = urllib.request.Request(
                'https://api.pushover.net/1/messages.json',
                data=data,
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
        except Exception as e:
            print(f"[WEBHOOK] Pushover error: {e}")
            return False

    def _send_custom(self, signal, webhook_config: Dict) -> bool:
        """Send alert to custom webhook."""
        url = webhook_config.get('url')
        if not url:
            return False

        payload = {
            'timestamp': datetime.utcnow().isoformat(),
            'alert_level': signal.alert_level,
            'bear_score': signal.bear_score,
            'confidence': signal.confidence,
            'vix_level': signal.vix_level,
            'vix_spike_pct': signal.vix_spike_pct,
            'spy_roc_3d': signal.spy_roc_3d,
            'market_breadth_pct': signal.market_breadth_pct,
            'sectors_declining': signal.sectors_declining,
            'sectors_total': signal.sectors_total,
            'yield_curve_spread': signal.yield_curve_spread,
            'credit_spread_change': signal.credit_spread_change,
            'volume_confirmation': signal.volume_confirmation,
            'momentum_divergence': signal.momentum_divergence,
            'triggers': signal.triggers,
            'recommendation': signal.recommendation
        }

        headers = webhook_config.get('headers', {})

        return self._post_json(url, payload, headers)

    def _post_json(self, url: str, payload: Dict, extra_headers: Dict = None) -> bool:
        """Send JSON POST request to webhook URL."""
        try:
            data = json.dumps(payload).encode('utf-8')

            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'Proteus-Bear-Monitor/1.0'
            }
            if extra_headers:
                headers.update(extra_headers)

            req = urllib.request.Request(url, data=data, headers=headers, method='POST')

            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status in [200, 201, 204]

        except urllib.error.HTTPError as e:
            print(f"[WEBHOOK] HTTP error {e.code}: {e.reason}")
            return False
        except Exception as e:
            print(f"[WEBHOOK] Error: {e}")
            return False

    def send_all_clear(self, signal) -> Dict[str, bool]:
        """Send 'all clear' notification when conditions normalize."""
        results = {}

        # Discord
        if 'discord' in self.config:
            embed = {
                "title": "All Clear - Bear Conditions Normalized",
                "description": f"Bear score has dropped to {signal.bear_score:.0f}/100. Normal trading conditions restored.",
                "color": 0x10B981,  # Green
                "footer": {"text": "Proteus Bear Detection"},
                "timestamp": datetime.utcnow().isoformat()
            }
            payload = {"embeds": [embed], "username": "Proteus Bear Monitor"}
            results['discord'] = self._post_json(self.config['discord'], payload)

        # Slack
        if 'slack' in self.config:
            payload = {
                "blocks": [{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":white_check_mark: *All Clear* - Bear score normalized to {signal.bear_score:.0f}/100"
                    }
                }]
            }
            results['slack'] = self._post_json(self.config['slack'], payload)

        # Pushover
        if 'pushover' in self.config:
            pushover_config = self.config['pushover']
            if pushover_config.get('user_key') and pushover_config.get('api_token'):
                import urllib.parse
                payload = {
                    'token': pushover_config['api_token'],
                    'user': pushover_config['user_key'],
                    'title': 'All Clear - Bear Normalized',
                    'message': f'Bear score dropped to {signal.bear_score:.0f}/100. Normal conditions.',
                    'priority': -1,
                    'sound': 'cashregister'
                }
                data = urllib.parse.urlencode(payload).encode('utf-8')
                try:
                    req = urllib.request.Request(
                        'https://api.pushover.net/1/messages.json',
                        data=data, method='POST'
                    )
                    with urllib.request.urlopen(req, timeout=10) as response:
                        results['pushover'] = response.status == 200
                except Exception:
                    results['pushover'] = False

        return results

    def test_webhook(self, platform: str = None) -> Dict[str, bool]:
        """
        Send test message to verify webhook configuration.

        Args:
            platform: Specific platform to test, or None for all

        Returns:
            Dict of {platform: success} results
        """
        from dataclasses import dataclass

        # Create a mock signal for testing
        @dataclass
        class MockSignal:
            bear_score: float = 55.0
            alert_level: str = 'WARNING'
            confidence: float = 0.75
            vix_level: float = 28.5
            vix_spike_pct: float = 15.0
            spy_roc_3d: float = -2.5
            market_breadth_pct: float = 35.0
            sectors_declining: int = 7
            sectors_total: int = 11
            yield_curve_spread: float = 0.10
            credit_spread_change: float = 8.0
            volume_confirmation: bool = True
            momentum_divergence: bool = False
            triggers: list = None
            recommendation: str = "TEST ALERT - This is a test of the bear detection webhook."

            def __post_init__(self):
                self.triggers = ["Test trigger 1", "Test trigger 2"]

        mock = MockSignal()
        results = {}

        platforms_to_test = [platform] if platform else ['discord', 'slack', 'pushover', 'custom']

        for p in platforms_to_test:
            if p == 'discord' and 'discord' in self.config:
                results['discord'] = self._send_discord(mock)
            elif p == 'slack' and 'slack' in self.config:
                results['slack'] = self._send_slack(mock)
            elif p == 'pushover' and 'pushover' in self.config:
                results['pushover'] = self._send_pushover(mock)
            elif p == 'custom' and 'custom' in self.config:
                for i, webhook in enumerate(self.config['custom']):
                    results[f'custom_{i}'] = self._send_custom(mock, webhook)

        return results


def get_webhook_status() -> Dict:
    """Get status of webhook configuration."""
    notifier = WebhookNotifier()

    status = {
        'enabled': notifier.is_enabled(),
        'platforms': []
    }

    if 'discord' in notifier.config:
        status['platforms'].append('Discord')
    if 'slack' in notifier.config:
        status['platforms'].append('Slack')
    if 'pushover' in notifier.config:
        status['platforms'].append('Pushover')
    if 'custom' in notifier.config:
        status['platforms'].append(f"Custom ({len(notifier.config['custom'])})")

    return status

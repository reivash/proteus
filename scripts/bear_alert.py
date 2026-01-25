#!/usr/bin/env python
"""
Bear Market Monitor

Continuously monitors for bearish market conditions and sends email alerts.

Usage:
    # Run once and exit
    python scripts/run_bear_monitor.py

    # Run continuously (every 60 minutes)
    python scripts/run_bear_monitor.py --continuous --interval 60

    # Force send test alert
    python scripts/run_bear_monitor.py --test

    # Just print report (no emails)
    python scripts/run_bear_monitor.py --report
"""

import os
import sys
import time
import argparse
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from analysis.fast_bear_detector import (
    FastBearDetector, print_fast_bear_report,
    log_bear_score, print_bear_trend
)
from trading.bearish_alert_service import BearishAlertService


def run_single_check(send_alert: bool = True, verbose: bool = True, log: bool = True) -> None:
    """Run a single bear detection check."""
    service = BearishAlertService()

    if verbose:
        print("=" * 60)
        print("BEAR MARKET MONITOR")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    signal = service.check_and_alert() if send_alert else service.fast_bear.detect()

    # Log to history
    if log:
        log_bear_score(signal)

    if verbose:
        print(f"Bear Score: {signal.bear_score}/100")
        print(f"Alert Level: {signal.alert_level}")
        print(f"Confidence: {signal.confidence * 100:.0f}%")
        print()

        if signal.triggers:
            print("Active Triggers:")
            for trigger in signal.triggers:
                print(f"  - {trigger}")
            print()

        print("Key Metrics:")
        print(f"  SPY 3d ROC: {signal.spy_roc_3d:+.2f}%")
        print(f"  VIX: {signal.vix_level:.1f} ({signal.vix_spike_pct:+.1f}% 2d)")
        print(f"  Breadth: {signal.market_breadth_pct:.1f}%")
        print(f"  Sectors Down: {signal.sectors_declining}/{signal.sectors_total}")
        print(f"  Volume: {'HIGH' if signal.volume_confirmation else 'Normal'}")
        curve_status = "INVERTED" if signal.yield_curve_spread <= 0 else ("FLAT" if signal.yield_curve_spread < 0.25 else "OK")
        print(f"  Yield Curve: {signal.yield_curve_spread:+.2f}% ({curve_status})")
        credit_status = "STRESSED" if signal.credit_spread_change >= 10 else ("WIDENING" if signal.credit_spread_change >= 5 else "OK")
        print(f"  Credit Spread: {signal.credit_spread_change:+.2f}% ({credit_status})")
        if signal.momentum_divergence:
            print(f"  DIVERGENCE: SPY near highs but breadth weak!")
        print()

        print("Recommendation:")
        print(f"  {signal.recommendation}")
        print()

        # Show service status
        status = service.get_status()
        print("Service Status:")
        print(f"  Last Alert: {status['last_alert_level']} at {status['last_alert_time'] or 'Never'}")
        print(f"  Alerts Today: {status['alerts_sent_today']}")
        print(f"  Email Enabled: {status['notifier_available']}")
        print()


def run_continuous(interval_minutes: int = 60, verbose: bool = True) -> None:
    """Run continuous monitoring."""
    print("=" * 60)
    print("BEAR MARKET MONITOR - CONTINUOUS MODE")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Interval: {interval_minutes} minutes")
    print("Press Ctrl+C to stop")
    print()

    service = BearishAlertService()

    while True:
        try:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running check...")

            signal = service.check_and_alert()

            status_emoji = {
                'CRITICAL': '!',
                'WARNING': 'W',
                'WATCH': '?',
                'NORMAL': '.'
            }

            emoji = status_emoji.get(signal.alert_level, '?')
            print(f"[{emoji}] Bear: {signal.bear_score}/100 | {signal.alert_level} | "
                  f"Triggers: {len(signal.triggers)} | "
                  f"VIX: {signal.vix_level:.1f}")

            if signal.alert_level in ['WARNING', 'CRITICAL']:
                print(f"    -> {signal.recommendation}")

            # Wait for next interval
            time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print("\n\nMonitor stopped.")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")
            print("Waiting 5 minutes before retry...")
            time.sleep(300)


def main():
    parser = argparse.ArgumentParser(
        description='Bear Market Monitor - Detect bearish conditions early',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_bear_monitor.py                    # Single check with alerts
  python scripts/run_bear_monitor.py --report           # Print detailed report
  python scripts/run_bear_monitor.py --trend            # Show 24h trend with chart
  python scripts/run_bear_monitor.py --chart            # Show ASCII chart only
  python scripts/run_bear_monitor.py --test             # Force test alert
  python scripts/run_bear_monitor.py --test-webhook     # Test webhook notifications
  python scripts/run_bear_monitor.py --daily            # Send daily summary email
  python scripts/run_bear_monitor.py --daily-force      # Force resend daily summary
  python scripts/run_bear_monitor.py --continuous       # Run every 60 min
  python scripts/run_bear_monitor.py -c -i 30           # Run every 30 min

Alert Levels:
  NORMAL   (0-29):  No concern
  WATCH    (30-49): Monitor closely
  WARNING  (50-69): Reduce exposure
  CRITICAL (70+):   Maximum caution
        """
    )

    parser.add_argument('--report', '-r', action='store_true',
                       help='Print detailed report without sending alerts')
    parser.add_argument('--trend', action='store_true',
                       help='Show bear score trend analysis (24h)')
    parser.add_argument('--test', '-t', action='store_true',
                       help='Force send a test alert')
    parser.add_argument('--continuous', '-c', action='store_true',
                       help='Run continuously')
    parser.add_argument('--interval', '-i', type=int, default=60,
                       help='Check interval in minutes (default: 60)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    parser.add_argument('--status', '-s', action='store_true',
                       help='Show service status only')
    parser.add_argument('--daily', '-d', action='store_true',
                       help='Send daily summary email (if not sent today)')
    parser.add_argument('--daily-force', action='store_true',
                       help='Force send daily summary email')
    parser.add_argument('--chart', action='store_true',
                       help='Show ASCII chart of bear score history')
    parser.add_argument('--test-webhook', action='store_true',
                       help='Test webhook notifications (Discord, Slack, Pushover)')
    parser.add_argument('--config', action='store_true',
                       help='Show current configuration')
    parser.add_argument('--validate-config', action='store_true',
                       help='Validate configuration file')

    args = parser.parse_args()

    if args.daily or args.daily_force:
        print("Sending daily market summary...")
        service = BearishAlertService()
        success = service.send_daily_summary(force=args.daily_force)
        if success:
            print("Daily summary sent successfully!")
        else:
            if not args.daily_force:
                print("Daily summary already sent today. Use --daily-force to send anyway.")
            else:
                print("Failed to send daily summary.")
        return

    if args.config:
        try:
            from config.bear_config import load_config, print_config
            config = load_config()
            print_config(config)
        except ImportError:
            print("Configuration system not available.")
            print("Using built-in defaults.")
        return

    if args.validate_config:
        try:
            from config.bear_config import load_config, validate_config
            config = load_config()
            result = validate_config(config)
            print("=" * 60)
            print("CONFIGURATION VALIDATION")
            print("=" * 60)
            if config.config_path:
                print(f"Config file: {config.config_path}")
            else:
                print("Config file: Using defaults")
            print()
            if result['valid']:
                print("Status: VALID")
                print("All configuration values are correct.")
            else:
                print("Status: INVALID")
                print("\nErrors found:")
                for error in result['errors']:
                    print(f"  - {error}")
        except ImportError:
            print("Configuration system not available.")
        return

    if args.status:
        service = BearishAlertService()
        status = service.get_status()
        print("Bearish Alert Service Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        return

    if args.report:
        print_fast_bear_report()
        return

    if args.trend:
        print_fast_bear_report()
        print_bear_trend(24, show_chart=True)
        return

    if args.chart:
        from analysis.fast_bear_detector import print_ascii_bear_chart, get_bear_score_trend
        trend = get_bear_score_trend(hours=48)
        if 'scores' in trend and len(trend['scores']) > 1:
            print_ascii_bear_chart(trend['scores'], trend.get('timestamps', []))
        else:
            print("Not enough history data for chart. Run monitor a few times first.")
        return

    if args.test:
        print("Sending test bear alert...")
        service = BearishAlertService()
        signal = service.check_and_alert(force_alert=True)
        print(f"Alert sent: {signal.alert_level} (Score: {signal.bear_score})")
        return

    if args.test_webhook:
        print("Testing webhook notifications...")
        try:
            from notifications.webhook_notifier import WebhookNotifier, get_webhook_status

            status = get_webhook_status()
            print(f"Webhook enabled: {status['enabled']}")
            print(f"Platforms configured: {', '.join(status['platforms']) if status['platforms'] else 'None'}")

            if not status['enabled']:
                print("\nNo webhooks configured. Add to email_config.json:")
                print('  "webhooks": {')
                print('    "discord": "https://discord.com/api/webhooks/...",')
                print('    "slack": "https://hooks.slack.com/services/...",')
                print('    "pushover": {"user_key": "...", "api_token": "..."}')
                print('  }')
                return

            print("\nSending test message to all configured webhooks...")
            notifier = WebhookNotifier()
            results = notifier.test_webhook()

            print("\nResults:")
            for platform, success in results.items():
                status_str = "SUCCESS" if success else "FAILED"
                print(f"  {platform}: {status_str}")

        except ImportError:
            print("Webhook notifier not available")
        return

    if args.continuous:
        run_continuous(interval_minutes=args.interval, verbose=not args.quiet)
    else:
        run_single_check(send_alert=True, verbose=not args.quiet)


if __name__ == "__main__":
    main()

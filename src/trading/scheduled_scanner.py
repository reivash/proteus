"""
Scheduled Smart Scanner

Runs the smart scanner at scheduled times and generates reports.
Can be run as a service or scheduled via Task Scheduler/cron.

Features:
1. Morning scan (9:35 AM ET) - Pre-market analysis
2. Afternoon scan (4:05 PM ET) - End-of-day analysis
3. Email alerts for strong signals
4. Logging to file
5. Web dashboard updates

Usage:
    python src/trading/scheduled_scanner.py --run-once  # Single run
    python src/trading/scheduled_scanner.py --schedule  # Run on schedule
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, time
from typing import List, Dict, Optional

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.trading.smart_scanner import SmartScanner, SmartSignal
from src.notifications.email_notifier import EmailNotifier
from src.trading.paper_trader import PaperTrader


# Configure logging
def setup_logging():
    """Set up logging to file and console."""
    log_dir = "logs/scans"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"scan_{datetime.now().strftime('%Y%m%d')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class ScheduledScanner:
    """
    Manages scheduled smart scans.
    """

    # Scan times (ET timezone)
    MORNING_SCAN = time(9, 35)    # 5 min after market open
    AFTERNOON_SCAN = time(16, 5)  # 5 min after market close

    def __init__(self, portfolio_value: float = 100000):
        self.scanner = SmartScanner(portfolio_value=portfolio_value)
        self.logger = setup_logging()
        self.email_notifier = EmailNotifier()
        self.paper_trader = PaperTrader(initial_capital=portfolio_value)
        self.last_scan_time = None
        self.last_signals = []

    def run_scan(self) -> Dict:
        """
        Run a complete smart scan and return results.

        Returns:
            Dictionary with scan results and metadata
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING SCHEDULED SCAN")
        self.logger.info("=" * 60)

        start_time = datetime.now()

        try:
            # Run the scan
            result = self.scanner.scan()

            self.last_scan_time = datetime.now()
            self.last_signals = result.signals

            # Build report
            report = {
                'timestamp': start_time.isoformat(),
                'duration_seconds': (datetime.now() - start_time).total_seconds(),
                'regime': result.regime.regime.value,
                'regime_confidence': result.regime.confidence,
                'total_scanned': result.total_scanned,
                'signals_passed': len(result.signals),
                'signals_filtered': result.filtered_count,
                'strong_signals': [s for s in result.signals if s.tier in ['ELITE', 'STRONG']],
                'top_signals': result.signals[:5],
                'status': 'success'
            }

            # Log summary
            self.logger.info(f"Scan completed in {report['duration_seconds']:.1f}s")
            self.logger.info(f"Regime: {report['regime']} ({report['regime_confidence']*100:.0f}%)")
            self.logger.info(f"Signals: {report['signals_passed']} passed / {report['signals_filtered']} filtered")

            if report['strong_signals']:
                self.logger.info("STRONG SIGNALS DETECTED:")
                for sig in report['strong_signals']:
                    self.logger.info(f"  {sig.ticker}: {sig.tier} @ ${sig.current_price:.2f} "
                                   f"({sig.suggested_shares} shares)")

            return report

        except Exception as e:
            self.logger.error(f"Scan failed: {e}")
            return {
                'timestamp': start_time.isoformat(),
                'status': 'error',
                'error': str(e)
            }

    def send_email_notification(self, report: Dict) -> bool:
        """
        Send email notification with scan results and paper trading performance.

        Args:
            report: Scan report dictionary

        Returns:
            True if email sent successfully
        """
        if not self.email_notifier.is_enabled():
            self.logger.info("Email notifications not configured")
            return False

        strong_signals = report.get('strong_signals', [])
        all_signals = report.get('top_signals', [])

        # Convert SmartSignal objects to dicts for email notifier
        signal_dicts = []
        for sig in all_signals[:5]:  # Top 5 signals
            signal_dicts.append({
                'ticker': sig.ticker,
                'price': sig.current_price,
                'z_score': -2.0,  # Placeholder - SmartScanner uses different metrics
                'rsi': 30.0,  # Placeholder
                'expected_return': sig.expected_return,
                'tier': sig.tier,
                'shares': sig.suggested_shares
            })

        # Build status message
        status = f"Regime: {report.get('regime', 'N/A')} | "
        status += f"Signals: {report.get('signals_passed', 0)} passed, "
        status += f"{report.get('signals_filtered', 0)} filtered"

        # Get paper trading performance stats
        paper_perf = self.paper_trader.get_performance()
        performance = {
            'total_trades': paper_perf.get('total_trades', 0),
            'win_rate': paper_perf.get('win_rate', 0),
            'total_return': paper_perf.get('total_return', 0),
            'days_tracked': len(self.paper_trader.daily_equity)
        }

        # Send notification
        success = self.email_notifier.send_scan_notification(
            scan_status=status,
            signals=signal_dicts,
            performance=performance
        )

        if success:
            self.logger.info(f"Email notification sent ({len(signal_dicts)} signals)")
        else:
            self.logger.warning("Failed to send email notification")

        return success

    def generate_email_alert(self, report: Dict) -> Optional[str]:
        """
        Generate email alert content for strong signals.

        Returns:
            HTML email content or None if no alert needed
        """
        strong_signals = report.get('strong_signals', [])

        if not strong_signals:
            return None

        lines = []
        lines.append("<h2>Proteus Trading Alert</h2>")
        lines.append(f"<p>Scan time: {report['timestamp']}</p>")
        lines.append(f"<p>Market regime: {report['regime']} ({report['regime_confidence']*100:.0f}% confidence)</p>")
        lines.append("<h3>Strong Signals Detected:</h3>")
        lines.append("<table border='1' cellpadding='5'>")
        lines.append("<tr><th>Ticker</th><th>Tier</th><th>Price</th><th>Shares</th><th>E[R]</th></tr>")

        for sig in strong_signals:
            lines.append(f"<tr><td>{sig.ticker}</td><td>{sig.tier}</td>"
                        f"<td>${sig.current_price:.2f}</td><td>{sig.suggested_shares}</td>"
                        f"<td>{sig.expected_return:+.1f}%</td></tr>")

        lines.append("</table>")
        lines.append("<p>Full results available in the dashboard.</p>")

        return "\n".join(lines)

    def update_dashboard(self):
        """Update the web dashboard with latest scan results."""
        try:
            from src.web.dashboard_generator import update_dashboard
            update_dashboard()
            self.logger.info("Dashboard updated")
        except ImportError:
            self.logger.warning("Dashboard generator not available")
        except Exception as e:
            self.logger.error(f"Dashboard update failed: {e}")

    def is_market_hours(self) -> bool:
        """Check if we're in market hours (9:30 AM - 4:00 PM ET)."""
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close

    def is_weekday(self) -> bool:
        """Check if today is a weekday."""
        return datetime.now().weekday() < 5

    def run_scheduled(self, interval_minutes: int = 5):
        """
        Run scanner on schedule.

        Args:
            interval_minutes: How often to check if scan is due
        """
        import time as time_module

        self.logger.info("Starting scheduled scanner...")
        self.logger.info(f"Morning scan: {self.MORNING_SCAN}")
        self.logger.info(f"Afternoon scan: {self.AFTERNOON_SCAN}")

        morning_done = False
        afternoon_done = False

        while True:
            now = datetime.now()
            current_time = now.time()

            # Reset flags at midnight
            if current_time.hour == 0 and current_time.minute < interval_minutes:
                morning_done = False
                afternoon_done = False

            # Check if we should run
            if self.is_weekday():
                # Morning scan
                if (not morning_done and
                    current_time >= self.MORNING_SCAN and
                    current_time.hour < 12):
                    self.logger.info("Running morning scan...")
                    report = self.run_scan()
                    self.update_dashboard()
                    self.send_email_notification(report)
                    morning_done = True

                # Afternoon scan
                if (not afternoon_done and
                    current_time >= self.AFTERNOON_SCAN and
                    current_time.hour >= 16):
                    self.logger.info("Running afternoon scan...")
                    report = self.run_scan()
                    self.update_dashboard()
                    self.send_email_notification(report)
                    afternoon_done = True

            # Wait before checking again
            time_module.sleep(interval_minutes * 60)


def main():
    parser = argparse.ArgumentParser(description="Proteus Scheduled Scanner")
    parser.add_argument('--run-once', action='store_true', help='Run single scan')
    parser.add_argument('--schedule', action='store_true', help='Run on schedule')
    parser.add_argument('--portfolio', type=float, default=100000, help='Portfolio value')
    args = parser.parse_args()

    scanner = ScheduledScanner(portfolio_value=args.portfolio)

    if args.schedule:
        scanner.run_scheduled()
    else:
        # Default: run once
        report = scanner.run_scan()
        scanner.update_dashboard()
        scanner.send_email_notification(report)

        # Save report
        output_dir = "data/scheduled_scans"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # Convert signals to serializable format
        export_report = {
            'timestamp': report['timestamp'],
            'status': report['status'],
            'regime': report.get('regime'),
            'signals_passed': report.get('signals_passed'),
            'signals_filtered': report.get('signals_filtered'),
            'top_signals': [
                {
                    'ticker': s.ticker,
                    'tier': s.tier,
                    'price': s.current_price,
                    'shares': s.suggested_shares,
                    'expected_return': s.expected_return
                }
                for s in report.get('top_signals', [])
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(export_report, f, indent=2)

        print(f"\n[SAVED] {output_file}")


if __name__ == "__main__":
    main()

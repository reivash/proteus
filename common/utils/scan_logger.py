"""
Scan Logger - Comprehensive logging for all scan results

Logs every scan to text files for traceability and debugging.
Creates a permanent audit trail of all trading activity.
"""

import os
from datetime import datetime
from typing import List, Dict
import json


class ScanLogger:
    """Logs all scan results to text files for audit trail."""

    def __init__(self, log_dir='logs/scans'):
        """
        Initialize scan logger.

        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Create subdirectories
        self.daily_dir = os.path.join(log_dir, 'daily')
        self.summary_dir = os.path.join(log_dir, 'summary')
        os.makedirs(self.daily_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)

    def log_scan(self, scan_type: str, regime: str, signals: List[Dict],
                 scan_status: str, performance: Dict = None, scanned_tickers: List[str] = None) -> str:
        """
        Log a complete scan result.

        Args:
            scan_type: 'scheduled' or 'manual'
            regime: Market regime (BULL/BEAR/SIDEWAYS)
            signals: List of signal dictionaries
            scan_status: Status message
            performance: Optional performance stats
            scanned_tickers: List of all tickers scanned

        Returns:
            Path to log file created
        """
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H%M%S')

        # Create daily log file
        daily_log = os.path.join(self.daily_dir, f'{date_str}.log')

        # Append to daily log
        with open(daily_log, 'a', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"SCAN LOG - {timestamp}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Type: {scan_type.upper()}\n")
            f.write(f"Market Regime: {regime}\n")
            f.write(f"Status: {scan_status}\n")
            f.write("\n")

            # Always show what was scanned
            if scanned_tickers:
                f.write(f"INSTRUMENTS SCANNED ({len(scanned_tickers)}):\n")
                f.write(f"  {', '.join(scanned_tickers)}\n")
            else:
                f.write("INSTRUMENTS SCANNED: Not specified\n")
            f.write("\n")

            f.write(f"Signals Found: {len(signals)}\n")
            f.write("\n")

            if signals:
                f.write("SIGNALS DETECTED:\n")
                f.write("-" * 80 + "\n")
                for sig in signals:
                    f.write(f"  Ticker: {sig['ticker']}\n")
                    f.write(f"  Price: ${sig['price']:.2f}\n")
                    f.write(f"  Z-Score: {sig['z_score']:.2f}\n")
                    f.write(f"  RSI: {sig['rsi']:.1f}\n")
                    f.write(f"  Expected Return: {sig.get('expected_return', 0):.2f}%\n")
                    f.write(f"  Signal Date: {sig.get('Date', 'N/A')}\n")
                    f.write("-" * 80 + "\n")
            else:
                f.write("NO SIGNALS - All stocks screened, none met criteria\n")

            f.write("\n")

            if performance:
                f.write("PERFORMANCE SUMMARY:\n")
                f.write(f"  Total Trades: {performance.get('total_trades', 0)}\n")
                f.write(f"  Win Rate: {performance.get('win_rate', 0):.1f}%\n")
                f.write(f"  Total Return: {performance.get('total_return', 0):+.2f}%\n")
                f.write(f"  Days Tracked: {performance.get('days_tracked', 0)}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("\n")

        # Create individual scan log with JSON
        scan_log = os.path.join(self.log_dir, f'scan_{date_str}_{time_str}.json')
        scan_data = {
            'timestamp': timestamp,
            'type': scan_type,
            'regime': regime,
            'status': scan_status,
            'scanned_tickers': scanned_tickers or [],
            'signals': signals,
            'signal_count': len(signals),
            'performance': performance
        }

        with open(scan_log, 'w', encoding='utf-8') as f:
            json.dump(scan_data, f, indent=2, default=str)

        # Update summary file
        self._update_summary(scan_data)

        return daily_log

    def _update_summary(self, scan_data: Dict):
        """Update the summary file with latest scan."""
        summary_file = os.path.join(self.summary_dir, 'latest_scans.log')

        # Keep last 50 scans in summary
        summary_lines = []
        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_lines = f.readlines()[-49:]  # Keep last 49 to add 1 new

        # Add new scan
        timestamp = scan_data['timestamp']
        scan_type = scan_data['type'].upper()
        signal_count = scan_data['signal_count']
        regime = scan_data['regime']

        summary_line = f"{timestamp} | {scan_type:10} | Regime: {regime:8} | Signals: {signal_count} | Status: {scan_data['status']}\n"
        summary_lines.append(summary_line)

        # Write updated summary
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 120 + "\n")
            f.write("RECENT SCANS SUMMARY (Last 50)\n")
            f.write("=" * 120 + "\n")
            f.write(f"{'Timestamp':<20} | {'Type':<10} | {'Regime':<15} | {'Signals':<10} | Status\n")
            f.write("-" * 120 + "\n")
            f.writelines(summary_lines)

    def get_today_summary(self) -> str:
        """Get summary of today's scans."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        daily_log = os.path.join(self.daily_dir, f'{date_str}.log')

        if os.path.exists(daily_log):
            with open(daily_log, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return f"No scans logged for {date_str}"

    def get_recent_scans(self, count: int = 10) -> List[Dict]:
        """Get most recent scan results."""
        scan_files = []
        for file in os.listdir(self.log_dir):
            if file.startswith('scan_') and file.endswith('.json'):
                scan_files.append(os.path.join(self.log_dir, file))

        # Sort by modified time
        scan_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # Load recent scans
        recent_scans = []
        for scan_file in scan_files[:count]:
            try:
                with open(scan_file, 'r', encoding='utf-8') as f:
                    recent_scans.append(json.load(f))
            except:
                continue

        return recent_scans


if __name__ == '__main__':
    # Test logging
    logger = ScanLogger()

    test_signals = [
        {
            'ticker': 'TSLA',
            'price': 156.80,
            'z_score': -2.10,
            'rsi': 29.9,
            'expected_return': 3.5,
            'Date': '2025-11-16'
        }
    ]

    test_performance = {
        'total_trades': 3,
        'win_rate': 66.7,
        'total_return': 5.2,
        'days_tracked': 2
    }

    log_file = logger.log_scan(
        scan_type='manual',
        regime='BULL',
        signals=test_signals,
        scan_status='Complete - Found 1 signal(s)',
        performance=test_performance
    )

    print(f"Test log created: {log_file}")
    print("\nToday's summary:")
    print(logger.get_today_summary())

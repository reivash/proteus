#!/usr/bin/env python3
"""
Bear Detection Monitor - Scheduled monitoring with alerts and logging

Usage:
    python scripts/bear_monitor.py                  # Single check with logging
    python scripts/bear_monitor.py --continuous     # Continuous monitoring
    python scripts/bear_monitor.py --interval 30    # Check every 30 minutes
    python scripts/bear_monitor.py --alert-only     # Only output on elevated risk
"""

import argparse
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.analysis.fast_bear_detector import FastBearDetector


def log_message(message: str, log_file: str = 'logs/bear_monitor.log'):
    """Log message to file and print to console."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {message}"

    # Print to console
    print(log_line)

    # Write to log file
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(log_line + '\n')
    except Exception:
        pass


def run_check(detector: FastBearDetector, alert_only: bool = False) -> dict:
    """Run single bear detection check."""
    try:
        # Get key assessments
        master = detector.get_master_summary()
        worry = detector.should_i_worry()

        # Determine if we should output
        risk_category = master['risk_category']
        is_elevated = risk_category in ['ELEVATED', 'HIGH', 'SEVERE']

        if alert_only and not is_elevated:
            return {'status': 'OK', 'elevated': False, 'logged': False}

        # Build log message
        icon = worry['icon']
        score = master['overall_risk']
        level = master['core_metrics']['alert_level']
        timing = master['core_metrics']['timing_signal']

        message = f"{icon} Risk: {score:.1f}/100 | Level: {level} | Timing: {timing} | {worry['message']}"
        log_message(message)

        # Additional details on elevated risk
        if is_elevated:
            log_message(f"    Action: {master['risk_action']}")
            if master['key_concerns']:
                log_message(f"    Concerns: {', '.join(master['key_concerns'][:2])}")

        # Save snapshot
        detector.save_bear_score_snapshot()

        return {
            'status': 'OK',
            'elevated': is_elevated,
            'logged': True,
            'risk_score': score,
            'risk_category': risk_category
        }

    except Exception as e:
        log_message(f"ERROR: {str(e)}")
        return {'status': 'ERROR', 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Bear Detection Monitor')
    parser.add_argument('--continuous', action='store_true', help='Continuous monitoring mode')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in minutes (default: 60)')
    parser.add_argument('--alert-only', action='store_true', help='Only log on elevated risk')
    parser.add_argument('--log-file', type=str, default='logs/bear_monitor.log', help='Log file path')

    args = parser.parse_args()

    log_message("=" * 50)
    log_message("BEAR DETECTION MONITOR STARTING")
    log_message("=" * 50)

    detector = FastBearDetector()

    if args.continuous:
        log_message(f"Continuous mode - checking every {args.interval} minutes")
        log_message("Press Ctrl+C to stop")
        log_message("")

        check_count = 0
        alert_count = 0

        try:
            while True:
                check_count += 1
                log_message(f"--- Check #{check_count} ---")

                result = run_check(detector, args.alert_only)

                if result.get('elevated'):
                    alert_count += 1

                log_message(f"Next check in {args.interval} minutes...")
                log_message("")

                time.sleep(args.interval * 60)

        except KeyboardInterrupt:
            log_message("")
            log_message("=" * 50)
            log_message(f"MONITOR STOPPED - {check_count} checks, {alert_count} alerts")
            log_message("=" * 50)
    else:
        # Single check
        result = run_check(detector, args.alert_only)

        if result['status'] == 'OK' and not result.get('logged'):
            log_message("[OK] No elevated risk - check complete")

        log_message("")


if __name__ == '__main__':
    main()

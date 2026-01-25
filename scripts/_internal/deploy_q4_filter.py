"""
Quick Deploy: Q4 Signal Filter

Calculate Q4 threshold from recent signals and deploy to production.
Expected impact: +14.1pp win rate (63.7% → 77.8%)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent))

from common.trading.signal_scanner import SignalScanner


def calculate_recent_q4_threshold(days_back=30):
    """
    Calculate Q4 threshold from recent signals.

    Args:
        days_back: Number of days to look back

    Returns:
        75th percentile of signal_strength
    """
    print(f"Scanning signals from last {days_back} days...")

    scanner = SignalScanner(lookback_days=90)

    # Collect signal strengths
    all_signal_strengths = []

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')

        try:
            signals = scanner.scan_all_stocks(date_str)

            for signal in signals:
                all_signal_strengths.append(signal['signal_strength'])

            print(f"  {date_str}: {len(signals)} signals found")

        except Exception as e:
            print(f"  {date_str}: Error - {e}")

        current_date += timedelta(days=1)

    if not all_signal_strengths:
        print("\n[ERROR] No signals found in recent period!")
        return None

    # Calculate statistics
    q4_threshold = np.percentile(all_signal_strengths, 75)

    print(f"\n{'='*70}")
    print("SIGNAL STRENGTH DISTRIBUTION")
    print(f"{'='*70}")
    print(f"Total signals: {len(all_signal_strengths)}")
    print(f"Mean: {np.mean(all_signal_strengths):.2f}")
    print(f"Median (Q2): {np.median(all_signal_strengths):.2f}")
    print(f"75th percentile (Q4): {q4_threshold:.2f}")
    print(f"Q4 count: {sum(1 for s in all_signal_strengths if s >= q4_threshold)}")
    print(f"Q4 percentage: {sum(1 for s in all_signal_strengths if s >= q4_threshold)/len(all_signal_strengths)*100:.1f}%")
    print(f"{'='*70}")

    return q4_threshold


def update_daily_runner(q4_threshold):
    """
    Update daily_runner.py to use Q4 filter.

    Args:
        q4_threshold: Minimum signal strength threshold
    """
    runner_path = Path('common/trading/daily_runner.py')

    if not runner_path.exists():
        print(f"\n[ERROR] daily_runner.py not found at {runner_path}")
        return False

    # Read current content
    content = runner_path.read_text()

    # Check if already has min_signal_strength parameter
    if 'min_signal_strength' in content:
        print("\n[INFO] daily_runner.py already has min_signal_strength parameter")
        print("Updating threshold value...")

        # Update the value
        import re
        pattern = r'min_signal_strength=[\d.]+|min_signal_strength=None'
        replacement = f'min_signal_strength={q4_threshold:.2f}'
        new_content = re.sub(pattern, replacement, content)

    else:
        print("\n[INFO] Adding min_signal_strength parameter to daily_runner.py")

        # Find SignalScanner initialization
        import re
        pattern = r'SignalScanner\(lookback_days=\d+\)'
        replacement = f'SignalScanner(lookback_days=90, min_signal_strength={q4_threshold:.2f})'
        new_content = re.sub(pattern, replacement, content)

    # Write updated content
    runner_path.write_text(new_content)

    print(f"\n[DEPLOYED] Q4 filter with threshold {q4_threshold:.2f}")
    print(f"Updated: {runner_path}")

    return True


def main():
    print("="*70)
    print("DEPLOY Q4 SIGNAL FILTER")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("OBJECTIVE: Deploy Q4-only filtering for +14.1pp win rate improvement")
    print("METHOD: Calculate threshold from recent signals (fast deployment)")
    print()

    # Calculate Q4 threshold
    q4_threshold = calculate_recent_q4_threshold(days_back=30)

    if q4_threshold is None:
        print("\n[FAILED] Could not calculate Q4 threshold")
        return

    # Deployment decision
    print(f"\n{'='*70}")
    print("DEPLOYMENT DECISION")
    print(f"{'='*70}")
    print(f"\nQ4 Threshold: {q4_threshold:.2f}")
    print(f"Expected Impact: +14.1pp win rate (63.7% → 77.8%)")
    print(f"Trade Frequency: -75% (quality over quantity)")
    print(f"\nRecommendation: DEPLOY NOW")
    print(f"Rationale: High-confidence improvement based on EXP-091 validation")
    print()

    # Update daily runner
    success = update_daily_runner(q4_threshold)

    if success:
        print(f"\n{'='*70}")
        print("DEPLOYMENT COMPLETE")
        print(f"{'='*70}")
        print(f"\n[SUCCESS] Q4 filter deployed to production")
        print(f"Threshold: {q4_threshold:.2f}")
        print(f"Next run will use Q4-only filtering")
        print()
        print("To verify: Check next daily_runner.py execution")
        print("To revert: Set min_signal_strength=None in daily_runner.py")
    else:
        print(f"\n[FAILED] Deployment unsuccessful")


if __name__ == "__main__":
    main()

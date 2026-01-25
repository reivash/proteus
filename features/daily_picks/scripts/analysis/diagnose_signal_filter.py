"""
Diagnose why 0 signals are being generated.
Test different min_signal_strength thresholds.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from common.trading.signal_scanner import SignalScanner
from datetime import datetime

def diagnose_filters():
    """Test different signal strength thresholds."""

    print("="*70)
    print("SIGNAL FILTER DIAGNOSTIC")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # Test different thresholds
    thresholds = [0, 30, 40, 50, 55, 60, 65, 70]

    for threshold in thresholds:
        print(f"\n--- Testing min_signal_strength = {threshold} ---")
        scanner = SignalScanner(lookback_days=90, min_signal_strength=threshold)

        try:
            signals = scanner.scan_all_stocks()
            print(f"Signals found: {len(signals)}")

            if signals:
                for sig in signals[:5]:  # Show first 5
                    ticker = sig.get('ticker', 'N/A')
                    strength = sig.get('signal_strength', 0)
                    reason = sig.get('signal_type', 'N/A')
                    print(f"  - {ticker}: strength={strength:.1f}, type={reason}")

                if len(signals) > 5:
                    print(f"  ... and {len(signals) - 5} more")
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    # Run with 0 threshold to see all potential signals
    scanner = SignalScanner(lookback_days=90, min_signal_strength=0)
    all_signals = scanner.scan_all_stocks()

    if all_signals:
        strengths = [s.get('signal_strength', 0) for s in all_signals]
        print(f"\nTotal raw signals (no filter): {len(all_signals)}")
        print(f"Signal strength range: {min(strengths):.1f} - {max(strengths):.1f}")
        print(f"Signal strength mean: {sum(strengths)/len(strengths):.1f}")

        # Distribution
        print("\nStrength distribution:")
        for bucket in [(0,30), (30,50), (50,60), (60,65), (65,70), (70,80), (80,100)]:
            count = len([s for s in strengths if bucket[0] <= s < bucket[1]])
            print(f"  {bucket[0]}-{bucket[1]}: {count} signals")

        # Recommend threshold
        if max(strengths) < 65:
            print(f"\n⚠️  ISSUE: No signals have strength >= 65")
            print(f"    Max signal strength is {max(strengths):.1f}")
            print(f"    Recommend lowering threshold to {int(max(strengths) - 5)}")
    else:
        print("\n⚠️  No signals found even with threshold=0")
        print("    Check technical indicator calculations or data fetching")

if __name__ == "__main__":
    diagnose_filters()

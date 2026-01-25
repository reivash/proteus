"""
ML Outcome Updater

Utility for updating signal outcomes in the ML Performance Tracker.
Supports manual updates and batch processing.

Usage:
    # Update single signal
    updater = MLOutcomeUpdater()
    updater.update_signal(signal_id=123, outcome='win', actual_return=2.5, days_held=2)

    # View pending signals
    updater.show_pending_signals()

    # Update from CSV
    updater.batch_update_from_csv('trade_results.csv')
"""

import os
import sys
import pandas as pd
from datetime import datetime
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.monitoring.ml_performance_tracker import MLPerformanceTracker


class MLOutcomeUpdater:
    """
    Utility for updating signal outcomes.
    """

    def __init__(self, db_path: str = 'logs/ml_performance.db'):
        """
        Initialize outcome updater.

        Args:
            db_path: Path to ML performance database
        """
        self.tracker = MLPerformanceTracker(db_path=db_path)

    def show_pending_signals(self, days: int = 7):
        """
        Show all pending signals from last N days.

        Args:
            days: Number of days to look back
        """
        df = self.tracker.get_signals(days=days, outcome='pending')

        if len(df) == 0:
            print(f"\n[INFO] No pending signals in last {days} days")
            return

        print(f"\n{'='*80}")
        print(f"PENDING SIGNALS - Last {days} Days")
        print(f"{'='*80}\n")

        print(f"{'ID':<6} {'Date':<12} {'Ticker':<8} {'ML Prob':<10} {'ML Conf':<10} "
              f"{'Z-Score':<10} {'RSI':<8} {'Exp Ret'}")
        print("-"*80)

        for _, row in df.iterrows():
            print(f"{int(row['signal_id']):<6} "
                  f"{row['timestamp'][:10]:<12} "
                  f"{row['ticker']:<8} "
                  f"{row['ml_probability']:<10.3f} "
                  f"{row['ml_confidence']:<10} "
                  f"{row.get('z_score', 0):<10.2f} "
                  f"{row.get('rsi', 0):<8.1f} "
                  f"{row.get('expected_return', 0):+.1f}%")

        print()
        print(f"Total pending: {len(df)}")

    def update_signal(self, signal_id: int, outcome: str,
                     actual_return: float = None, days_held: int = None):
        """
        Update a single signal outcome.

        Args:
            signal_id: Signal ID to update
            outcome: 'win', 'loss', or 'pending'
            actual_return: Actual return percentage
            days_held: Number of days position was held
        """
        self.tracker.update_outcome(
            signal_id=signal_id,
            outcome=outcome,
            actual_return=actual_return,
            days_held=days_held
        )

        print(f"\n[OK] Signal {signal_id} updated: {outcome} "
              f"({actual_return:+.1f}% over {days_held} days)")

    def batch_update_from_csv(self, csv_path: str):
        """
        Batch update signals from CSV file.

        CSV format:
        signal_id,outcome,actual_return,days_held
        123,win,2.5,2
        124,loss,-1.8,3

        Args:
            csv_path: Path to CSV file
        """
        if not os.path.exists(csv_path):
            print(f"[ERROR] File not found: {csv_path}")
            return

        df = pd.read_csv(csv_path)

        required_cols = ['signal_id', 'outcome']
        if not all(col in df.columns for col in required_cols):
            print(f"[ERROR] CSV must contain columns: {required_cols}")
            return

        print(f"\n[INFO] Batch updating {len(df)} signals from {csv_path}")

        for _, row in df.iterrows():
            try:
                self.tracker.update_outcome(
                    signal_id=int(row['signal_id']),
                    outcome=row['outcome'],
                    actual_return=row.get('actual_return'),
                    days_held=row.get('days_held')
                )
                print(f"  [OK] Signal {int(row['signal_id'])}: {row['outcome']}")
            except Exception as e:
                print(f"  [ERROR] Signal {int(row['signal_id'])}: {e}")

        print("\n[OK] Batch update complete")

    def generate_report(self, days: int = 30):
        """
        Generate ML performance report.

        Args:
            days: Number of days to analyze
        """
        report = self.tracker.generate_report(days=days)
        print(report)

    def export_pending_to_csv(self, filepath: str = None):
        """
        Export pending signals to CSV for manual update.

        Args:
            filepath: Output file path (default: logs/pending_signals_{date}.csv)
        """
        df = self.tracker.get_signals(days=30, outcome='pending')

        if len(df) == 0:
            print("\n[INFO] No pending signals to export")
            return

        if filepath is None:
            filepath = f"logs/pending_signals_{datetime.now().strftime('%Y%m%d')}.csv"

        # Export with template columns for manual update
        export_df = df[['signal_id', 'timestamp', 'ticker', 'ml_probability',
                       'ml_confidence', 'z_score', 'rsi', 'expected_return']].copy()
        export_df['outcome'] = ''
        export_df['actual_return'] = ''
        export_df['days_held'] = ''

        export_df.to_csv(filepath, index=False)
        print(f"\n[OK] Exported {len(df)} pending signals to {filepath}")
        print("Fill in 'outcome', 'actual_return', 'days_held' columns and re-import with batch_update_from_csv()")


if __name__ == "__main__":
    """Example usage and commands."""

    updater = MLOutcomeUpdater()

    print("\n" + "="*80)
    print("ML OUTCOME UPDATER")
    print("="*80)

    # Show pending signals
    updater.show_pending_signals(days=30)

    # Generate current performance report
    print("\n" + "="*80)
    print("CURRENT ML PERFORMANCE")
    print("="*80)
    updater.generate_report(days=30)

    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    print()
    print("# Update single signal:")
    print("updater.update_signal(signal_id=123, outcome='win', actual_return=2.5, days_held=2)")
    print()
    print("# Export pending signals to CSV:")
    print("updater.export_pending_to_csv()")
    print()
    print("# Batch update from CSV:")
    print("updater.batch_update_from_csv('logs/pending_signals_20251117.csv')")
    print()

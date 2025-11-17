"""
ML Performance Tracker

Tracks real-time ML model performance in production to validate the 2.4x precision
improvement claim. Logs all signals with ML probabilities and tracks outcomes.

Features:
- Signal logging (ML-approved vs ML-filtered)
- Outcome tracking (win/loss/pending)
- Performance metrics (precision, recall, accuracy)
- Degradation alerts
- Historical trend analysis

Usage:
    tracker = MLPerformanceTracker()
    tracker.log_signal(ticker='NVDA', ml_prob=0.75, signal_strength=85, outcome='pending')
    tracker.update_outcome(signal_id=123, outcome='win', return_pct=2.5)
    report = tracker.generate_report()
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3


class MLPerformanceTracker:
    """
    Track ML model performance in production.
    """

    def __init__(self, db_path: str = 'logs/ml_performance.db', ml_threshold: float = 0.30):
        """
        Initialize ML performance tracker.

        Args:
            db_path: Path to SQLite database for signal storage
            ml_threshold: ML probability threshold for "approved" signals
        """
        self.db_path = db_path
        self.ml_threshold = ml_threshold

        # Ensure logs directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for signal tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                ml_probability REAL NOT NULL,
                ml_confidence TEXT NOT NULL,
                signal_strength REAL,
                z_score REAL,
                rsi REAL,
                expected_return REAL,
                ml_approved INTEGER NOT NULL,
                outcome TEXT DEFAULT 'pending',
                actual_return REAL,
                days_held INTEGER,
                updated_at TEXT
            )
        ''')

        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON signals(timestamp)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_outcome ON signals(outcome)
        ''')

        conn.commit()
        conn.close()

    def log_signal(self, ticker: str, ml_probability: float,
                   ml_confidence: str, signal_strength: float = None,
                   z_score: float = None, rsi: float = None,
                   expected_return: float = None) -> int:
        """
        Log a new signal.

        Args:
            ticker: Stock ticker
            ml_probability: ML probability score
            ml_confidence: ML confidence rating (HIGH/MEDIUM/LOW)
            signal_strength: Signal strength score (0-100)
            z_score: Z-score value
            rsi: RSI value
            expected_return: Expected return %

        Returns:
            signal_id: Unique signal ID
        """
        ml_approved = 1 if ml_probability >= self.ml_threshold else 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO signals (
                timestamp, ticker, ml_probability, ml_confidence,
                signal_strength, z_score, rsi, expected_return, ml_approved
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            ticker,
            ml_probability,
            ml_confidence,
            signal_strength,
            z_score,
            rsi,
            expected_return,
            ml_approved
        ))

        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return signal_id

    def update_outcome(self, signal_id: int, outcome: str,
                      actual_return: float = None, days_held: int = None):
        """
        Update signal outcome.

        Args:
            signal_id: Signal ID to update
            outcome: 'win', 'loss', or 'pending'
            actual_return: Actual return percentage
            days_held: Number of days position was held
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE signals
            SET outcome = ?, actual_return = ?, days_held = ?, updated_at = ?
            WHERE signal_id = ?
        ''', (
            outcome,
            actual_return,
            days_held,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            signal_id
        ))

        conn.commit()
        conn.close()

    def get_signals(self, days: int = 30, outcome: str = None,
                   ml_approved: bool = None) -> pd.DataFrame:
        """
        Get signals from last N days.

        Args:
            days: Number of days to look back
            outcome: Filter by outcome ('win', 'loss', 'pending', or None for all)
            ml_approved: Filter by ML approval (True/False/None for all)

        Returns:
            DataFrame of signals
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        query = 'SELECT * FROM signals WHERE timestamp >= ?'
        params = [cutoff_date]

        if outcome:
            query += ' AND outcome = ?'
            params.append(outcome)

        if ml_approved is not None:
            query += ' AND ml_approved = ?'
            params.append(1 if ml_approved else 0)

        query += ' ORDER BY timestamp DESC'

        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    def calculate_metrics(self, days: int = 30) -> Dict:
        """
        Calculate performance metrics.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary of metrics
        """
        # Get completed signals (not pending)
        df = self.get_signals(days=days)
        df_completed = df[df['outcome'] != 'pending'].copy()

        if len(df_completed) == 0:
            return {
                'period_days': days,
                'total_signals': 0,
                'error': 'No completed signals in period'
            }

        # Separate ML-approved and ML-filtered
        ml_approved = df_completed[df_completed['ml_approved'] == 1]
        ml_filtered = df_completed[df_completed['ml_approved'] == 0]
        all_signals = df_completed

        # Calculate precision (win rate)
        def calc_precision(signals):
            if len(signals) == 0:
                return 0.0
            return (signals['outcome'] == 'win').sum() / len(signals)

        # Calculate average return
        def calc_avg_return(signals):
            if len(signals) == 0:
                return 0.0
            return signals['actual_return'].mean()

        ml_approved_precision = calc_precision(ml_approved)
        ml_filtered_precision = calc_precision(ml_filtered)
        baseline_precision = calc_precision(all_signals)

        ml_approved_avg_return = calc_avg_return(ml_approved)
        ml_filtered_avg_return = calc_avg_return(ml_filtered)
        baseline_avg_return = calc_avg_return(all_signals)

        # Calculate improvement factor
        improvement_factor = (ml_approved_precision / baseline_precision
                             if baseline_precision > 0 else 0)

        return {
            'period_days': days,
            'total_signals': len(df),
            'completed_signals': len(df_completed),
            'pending_signals': len(df[df['outcome'] == 'pending']),

            'ml_approved': {
                'count': len(ml_approved),
                'wins': int((ml_approved['outcome'] == 'win').sum()),
                'losses': int((ml_approved['outcome'] == 'loss').sum()),
                'precision': float(ml_approved_precision),
                'avg_return': float(ml_approved_avg_return)
            },

            'ml_filtered': {
                'count': len(ml_filtered),
                'wins': int((ml_filtered['outcome'] == 'win').sum()),
                'losses': int((ml_filtered['outcome'] == 'loss').sum()),
                'precision': float(ml_filtered_precision),
                'avg_return': float(ml_filtered_avg_return)
            },

            'baseline': {
                'count': len(all_signals),
                'wins': int((all_signals['outcome'] == 'win').sum()),
                'losses': int((all_signals['outcome'] == 'loss').sum()),
                'precision': float(baseline_precision),
                'avg_return': float(baseline_avg_return)
            },

            'improvement_factor': float(improvement_factor),
            'improvement_percentage': float((improvement_factor - 1) * 100) if improvement_factor > 0 else 0
        }

    def generate_report(self, days: int = 30) -> str:
        """
        Generate human-readable performance report.

        Args:
            days: Number of days to analyze

        Returns:
            Formatted report string
        """
        metrics = self.calculate_metrics(days=days)

        if 'error' in metrics:
            return f"[INFO] {metrics['error']}"

        report = f"""
={'='*70}
ML PERFORMANCE REPORT - Last {days} Days
={'='*70}

SIGNAL VOLUME:
  Total Signals: {metrics['total_signals']}
  Completed: {metrics['completed_signals']}
  Pending: {metrics['pending_signals']}

ML-APPROVED SIGNALS (>= {self.ml_threshold} probability):
  Count: {metrics['ml_approved']['count']}
  Wins: {metrics['ml_approved']['wins']}
  Losses: {metrics['ml_approved']['losses']}
  Precision: {metrics['ml_approved']['precision']*100:.1f}%
  Avg Return: {metrics['ml_approved']['avg_return']:+.1f}%

ML-FILTERED SIGNALS (< {self.ml_threshold} probability):
  Count: {metrics['ml_filtered']['count']}
  Wins: {metrics['ml_filtered']['wins']}
  Losses: {metrics['ml_filtered']['losses']}
  Precision: {metrics['ml_filtered']['precision']*100:.1f}%
  Avg Return: {metrics['ml_filtered']['avg_return']:+.1f}%

BASELINE (All Signals):
  Count: {metrics['baseline']['count']}
  Wins: {metrics['baseline']['wins']}
  Losses: {metrics['baseline']['losses']}
  Precision: {metrics['baseline']['precision']*100:.1f}%
  Avg Return: {metrics['baseline']['avg_return']:+.1f}%

ML IMPROVEMENT:
  Improvement Factor: {metrics['improvement_factor']:.2f}x
  Improvement: {metrics['improvement_percentage']:+.1f}%

{'='*70}
"""

        # Add alert if ML performance is degrading
        if metrics['ml_approved']['count'] >= 5:  # Minimum sample size
            if metrics['improvement_factor'] < 1.5:  # Below 1.5x improvement
                report += "\n[ALERT] ML improvement below target (1.5x)\n"
                report += "Consider retraining model or adjusting threshold\n"

        return report

    def export_to_csv(self, days: int = 30, filepath: str = None):
        """
        Export signals to CSV for analysis.

        Args:
            days: Number of days to export
            filepath: Output file path (default: logs/ml_performance_{date}.csv)
        """
        df = self.get_signals(days=days)

        if filepath is None:
            filepath = f"logs/ml_performance_{datetime.now().strftime('%Y%m%d')}.csv"

        df.to_csv(filepath, index=False)
        print(f"Exported {len(df)} signals to {filepath}")


if __name__ == "__main__":
    """Example usage and testing."""

    tracker = MLPerformanceTracker()

    print("ML Performance Tracker Initialized")
    print(f"Database: {tracker.db_path}")
    print(f"ML Threshold: {tracker.ml_threshold}")
    print()

    # Generate report
    print("Generating performance report...")
    print(tracker.generate_report(days=30))

    # Export to CSV
    print("\nExporting data...")
    tracker.export_to_csv(days=30)

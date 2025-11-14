"""
Performance Tracker

Tracks paper trading performance with detailed metrics and comparisons.
Generates daily, weekly, and monthly reports.

Usage:
    tracker = PerformanceTracker()
    tracker.update(trader)
    report = tracker.generate_daily_report()
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import json


class PerformanceTracker:
    """
    Track and analyze paper trading performance.
    """

    def __init__(self, data_dir: str = 'data/paper_trading'):
        """
        Initialize performance tracker.

        Args:
            data_dir: Directory to store performance data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Performance history
        self.daily_snapshots: List[Dict] = []
        self.trade_log: List[Dict] = []

        # Load existing data
        self._load_history()

    def update(self, trader) -> Dict:
        """
        Update performance tracking with current trader state.

        Args:
            trader: PaperTrader instance

        Returns:
            Current performance snapshot
        """
        current_date = datetime.now().strftime('%Y-%m-%d')
        performance = trader.get_performance()
        positions = trader.get_open_positions()

        # Create snapshot
        snapshot = {
            'date': current_date,
            'timestamp': datetime.now().isoformat(),
            'capital': performance['current_capital'],
            'equity': performance['current_equity'],
            'positions_value': performance['positions_value'],
            'total_return': performance['total_return'],
            'total_trades': performance['total_trades'],
            'winning_trades': performance['winning_trades'],
            'losing_trades': performance['losing_trades'],
            'win_rate': performance['win_rate'],
            'total_pnl': performance['total_pnl'],
            'avg_pnl': performance['avg_pnl'],
            'num_positions': performance['current_positions'],
            'positions': positions
        }

        # Add to history
        self.daily_snapshots.append(snapshot)

        # Save
        self._save_history()

        return snapshot

    def log_trade(self, trade: Dict):
        """
        Log completed trade.

        Args:
            trade: Trade dictionary from PaperTrader
        """
        trade['logged_at'] = datetime.now().isoformat()
        self.trade_log.append(trade)
        self._save_history()

    def generate_daily_report(self) -> str:
        """
        Generate daily performance report.

        Returns:
            Formatted report string
        """
        if not self.daily_snapshots:
            return "[INFO] No performance data available yet"

        current = self.daily_snapshots[-1]

        report = []
        report.append("=" * 70)
        report.append(f"PAPER TRADING DAILY REPORT - {current['date']}")
        report.append("=" * 70)
        report.append("")

        # Account Summary
        report.append("ACCOUNT SUMMARY")
        report.append("-" * 70)
        report.append(f"Current Equity:      ${current['equity']:>15,.2f}")
        report.append(f"Available Capital:   ${current['capital']:>15,.2f}")
        report.append(f"Positions Value:     ${current['positions_value']:>15,.2f}")
        report.append(f"Total Return:        {current['total_return']:>15.2f}%")
        report.append("")

        # Trading Statistics
        report.append("TRADING STATISTICS")
        report.append("-" * 70)
        report.append(f"Total Trades:        {current['total_trades']:>15}")
        report.append(f"Winning Trades:      {current['winning_trades']:>15}")
        report.append(f"Losing Trades:       {current['losing_trades']:>15}")
        report.append(f"Win Rate:            {current['win_rate']:>14.1f}%")
        report.append(f"Total P&L:           ${current['total_pnl']:>14,.2f}")
        report.append(f"Avg P&L per Trade:   ${current['avg_pnl']:>14,.2f}")
        report.append("")

        # Open Positions
        report.append(f"OPEN POSITIONS ({current['num_positions']})")
        report.append("-" * 70)
        if current['positions']:
            report.append(f"{'Ticker':<8} {'Entry':<12} {'Current':<12} {'Return':<10} {'P&L':<12} {'Days':<6}")
            report.append("-" * 70)
            for pos in current['positions']:
                report.append(
                    f"{pos['ticker']:<8} "
                    f"${pos['entry_price']:<11.2f} "
                    f"${pos['current_price']:<11.2f} "
                    f"{pos['current_return']:>+8.2f}% "
                    f"${pos['pnl']:>+10.2f} "
                    f"{pos['hold_days']:>5}"
                )
        else:
            report.append("No open positions")
        report.append("")

        # Recent Trades (last 5)
        recent_trades = self._get_recent_trades(5)
        if recent_trades:
            report.append(f"RECENT TRADES (Last {len(recent_trades)})")
            report.append("-" * 70)
            report.append(f"{'Ticker':<8} {'Entry':<12} {'Exit':<12} {'Return':<10} {'P&L':<12} {'Reason':<15}")
            report.append("-" * 70)
            for trade in recent_trades:
                report.append(
                    f"{trade['ticker']:<8} "
                    f"{trade['entry_date']:<12} "
                    f"{trade['exit_date']:<12} "
                    f"{trade['return']:>+8.2f}% "
                    f"${trade['pnl']:>+10.2f} "
                    f"{trade['exit_reason']:<15}"
                )
            report.append("")

        # Performance vs Backtest
        backtest_metrics = self._get_backtest_comparison()
        if backtest_metrics:
            report.append("PERFORMANCE VS BACKTEST EXPECTATIONS")
            report.append("-" * 70)
            report.append(f"{'Metric':<25} {'Backtest':<15} {'Live':<15} {'Status':<15}")
            report.append("-" * 70)

            for metric_name, metric_data in backtest_metrics.items():
                status = self._compare_metric(metric_data['backtest'], metric_data['live'], metric_data['type'])
                report.append(
                    f"{metric_name:<25} "
                    f"{metric_data['backtest_str']:<15} "
                    f"{metric_data['live_str']:<15} "
                    f"{status:<15}"
                )
            report.append("")

        report.append("=" * 70)

        return "\n".join(report)

    def generate_weekly_summary(self) -> str:
        """
        Generate weekly performance summary.

        Returns:
            Formatted summary string
        """
        if len(self.daily_snapshots) < 2:
            return "[INFO] Insufficient data for weekly summary"

        # Get snapshots from last 7 days
        week_ago = datetime.now() - timedelta(days=7)
        weekly_snapshots = [
            s for s in self.daily_snapshots
            if pd.to_datetime(s['date']) >= week_ago
        ]

        if not weekly_snapshots:
            return "[INFO] No data from last 7 days"

        start_snapshot = weekly_snapshots[0]
        end_snapshot = weekly_snapshots[-1]

        # Calculate weekly changes
        equity_change = end_snapshot['equity'] - start_snapshot['equity']
        equity_change_pct = (equity_change / start_snapshot['equity']) * 100
        trades_this_week = end_snapshot['total_trades'] - start_snapshot['total_trades']

        report = []
        report.append("=" * 70)
        report.append(f"WEEKLY SUMMARY ({start_snapshot['date']} to {end_snapshot['date']})")
        report.append("=" * 70)
        report.append("")
        report.append(f"Starting Equity:     ${start_snapshot['equity']:>15,.2f}")
        report.append(f"Ending Equity:       ${end_snapshot['equity']:>15,.2f}")
        report.append(f"Change:              ${equity_change:>14,.2f} ({equity_change_pct:+.2f}%)")
        report.append(f"Trades This Week:    {trades_this_week:>15}")
        report.append(f"Current Win Rate:    {end_snapshot['win_rate']:>14.1f}%")
        report.append("")
        report.append("=" * 70)

        return "\n".join(report)

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.05) -> Optional[float]:
        """
        Calculate Sharpe ratio from daily returns.

        Args:
            risk_free_rate: Annual risk-free rate (default: 5%)

        Returns:
            Sharpe ratio or None if insufficient data
        """
        if len(self.daily_snapshots) < 30:
            return None

        # Calculate daily returns
        returns = []
        for i in range(1, len(self.daily_snapshots)):
            prev_equity = self.daily_snapshots[i-1]['equity']
            curr_equity = self.daily_snapshots[i]['equity']
            daily_return = (curr_equity - prev_equity) / prev_equity
            returns.append(daily_return)

        if not returns:
            return None

        returns_array = np.array(returns)

        # Annualized metrics
        avg_return = np.mean(returns_array) * 252  # 252 trading days
        std_return = np.std(returns_array) * np.sqrt(252)

        if std_return == 0:
            return None

        sharpe = (avg_return - risk_free_rate) / std_return
        return sharpe

    def calculate_max_drawdown(self) -> Optional[float]:
        """
        Calculate maximum drawdown from equity curve.

        Returns:
            Max drawdown percentage or None
        """
        if len(self.daily_snapshots) < 2:
            return None

        equity_curve = [s['equity'] for s in self.daily_snapshots]
        peak = equity_curve[0]
        max_dd = 0

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = ((equity - peak) / peak) * 100
            if dd < max_dd:
                max_dd = dd

        return max_dd

    def _get_recent_trades(self, n: int = 5) -> List[Dict]:
        """Get n most recent trades."""
        if not self.trade_log:
            return []
        return sorted(self.trade_log, key=lambda x: x['exit_date'], reverse=True)[:n]

    def _get_backtest_comparison(self) -> Optional[Dict]:
        """
        Compare live performance to backtest expectations.

        Returns:
            Comparison dictionary or None
        """
        if not self.daily_snapshots:
            return None

        current = self.daily_snapshots[-1]

        # Backtest expectations (from PRODUCTION_DEPLOYMENT_GUIDE.md)
        backtest_win_rate = 77.3
        backtest_sharpe = 6.35
        backtest_max_dd = -3.20

        # Calculate live metrics
        live_win_rate = current['win_rate']
        live_sharpe = self.calculate_sharpe_ratio()
        live_max_dd = self.calculate_max_drawdown()

        comparison = {
            'Win Rate': {
                'backtest': backtest_win_rate,
                'live': live_win_rate,
                'backtest_str': f"{backtest_win_rate:.1f}%",
                'live_str': f"{live_win_rate:.1f}%",
                'type': 'higher_better'
            },
            'Sharpe Ratio': {
                'backtest': backtest_sharpe,
                'live': live_sharpe if live_sharpe else 0,
                'backtest_str': f"{backtest_sharpe:.2f}",
                'live_str': f"{live_sharpe:.2f}" if live_sharpe else "N/A",
                'type': 'higher_better'
            },
            'Max Drawdown': {
                'backtest': backtest_max_dd,
                'live': live_max_dd if live_max_dd else 0,
                'backtest_str': f"{backtest_max_dd:.2f}%",
                'live_str': f"{live_max_dd:.2f}%" if live_max_dd else "N/A",
                'type': 'higher_better'  # Less negative is better
            }
        }

        return comparison

    def _compare_metric(self, backtest_value: float, live_value: float, metric_type: str) -> str:
        """
        Compare metric to backtest expectation.

        Args:
            backtest_value: Expected value from backtest
            live_value: Actual live value
            metric_type: 'higher_better' or 'lower_better'

        Returns:
            Status string
        """
        if live_value == 0:
            return "N/A"

        tolerance = 0.05  # 5% tolerance

        if metric_type == 'higher_better':
            if live_value >= backtest_value * (1 - tolerance):
                return "✓ ON TARGET"
            else:
                return "⚠ BELOW TARGET"
        else:
            if live_value <= backtest_value * (1 + tolerance):
                return "✓ ON TARGET"
            else:
                return "⚠ ABOVE TARGET"

    def get_stats_summary(self) -> Dict:
        """
        Get summary statistics.

        Returns:
            Statistics dictionary
        """
        if not self.daily_snapshots:
            return {}

        current = self.daily_snapshots[-1]

        return {
            'current_equity': current['equity'],
            'total_return': current['total_return'],
            'total_trades': current['total_trades'],
            'win_rate': current['win_rate'],
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'max_drawdown': self.calculate_max_drawdown(),
            'num_positions': current['num_positions'],
            'days_tracked': len(self.daily_snapshots)
        }

    def export_to_csv(self, filename: str = 'performance_history.csv'):
        """
        Export performance history to CSV.

        Args:
            filename: Output filename
        """
        if not self.daily_snapshots:
            print("[WARN] No data to export")
            return

        df = pd.DataFrame(self.daily_snapshots)
        output_path = os.path.join(self.data_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"[SAVED] Performance history to {output_path}")

    def _save_history(self):
        """Save performance history to disk."""
        history = {
            'daily_snapshots': self.daily_snapshots,
            'trade_log': self.trade_log
        }

        history_file = os.path.join(self.data_dir, 'performance_history.json')
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def _load_history(self):
        """Load performance history from disk."""
        history_file = os.path.join(self.data_dir, 'performance_history.json')
        if not os.path.exists(history_file):
            return

        try:
            with open(history_file, 'r') as f:
                history = json.load(f)

            self.daily_snapshots = history.get('daily_snapshots', [])
            self.trade_log = history.get('trade_log', [])

            print(f"[LOADED] Performance history: {len(self.daily_snapshots)} days, {len(self.trade_log)} trades")

        except Exception as e:
            print(f"[WARN] Could not load history: {e}")


if __name__ == "__main__":
    # Example usage
    tracker = PerformanceTracker()

    # Simulate some snapshots
    print("Creating test snapshots...")

    test_snapshot = {
        'date': '2025-11-14',
        'timestamp': datetime.now().isoformat(),
        'capital': 98000,
        'equity': 102500,
        'positions_value': 4500,
        'total_return': 2.5,
        'total_trades': 5,
        'winning_trades': 4,
        'losing_trades': 1,
        'win_rate': 80.0,
        'total_pnl': 2500,
        'avg_pnl': 500,
        'num_positions': 2,
        'positions': [
            {
                'ticker': 'NVDA',
                'entry_date': '2025-11-13',
                'entry_price': 145.50,
                'current_price': 148.20,
                'current_return': 1.86,
                'pnl': 270,
                'hold_days': 1,
                'shares': 100
            }
        ]
    }

    tracker.daily_snapshots.append(test_snapshot)

    # Generate report
    report = tracker.generate_daily_report()
    print(report)

    # Stats summary
    stats = tracker.get_stats_summary()
    print("\nStats Summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

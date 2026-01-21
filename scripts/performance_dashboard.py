#!/usr/bin/env python
"""
Performance Dashboard - Track VirtualWallet vs SPY benchmark

Compares paper trading performance against SPY to validate the system.

Features:
- Daily return comparison
- Cumulative performance chart
- Risk metrics (Sharpe, max drawdown, win rate)
- 30-day rolling statistics
- HTML report generation

Usage:
    python scripts/performance_dashboard.py              # Console summary
    python scripts/performance_dashboard.py --html       # Generate HTML dashboard
    python scripts/performance_dashboard.py --update     # Update benchmark data
    python scripts/performance_dashboard.py --history    # Show historical comparison
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class PerformanceDashboard:
    """Track and compare VirtualWallet performance vs SPY."""

    def __init__(self):
        self.data_dir = Path('data/virtual_wallet')
        self.benchmark_file = self.data_dir / 'benchmark_history.json'
        self.snapshots_file = self.data_dir / 'daily_snapshots.json'
        self.wallet_file = self.data_dir / 'wallet_state.json'
        self.trade_history_file = self.data_dir / 'trade_history.json'

    def load_wallet_snapshots(self) -> List[Dict]:
        """Load wallet daily snapshots."""
        if self.snapshots_file.exists():
            with open(self.snapshots_file) as f:
                data = json.load(f)
                return data.get('snapshots', [])
        return []

    def load_wallet_state(self) -> Dict:
        """Load current wallet state."""
        if self.wallet_file.exists():
            with open(self.wallet_file) as f:
                return json.load(f)
        return {}

    def load_trade_history(self) -> Dict:
        """Load trade history."""
        if self.trade_history_file.exists():
            with open(self.trade_history_file) as f:
                return json.load(f)
        return {'trades': [], 'summary': {}}

    def load_benchmark_history(self) -> Dict:
        """Load SPY benchmark history."""
        if self.benchmark_file.exists():
            with open(self.benchmark_file) as f:
                return json.load(f)
        return {'spy_data': [], 'last_updated': None}

    def save_benchmark_history(self, data: Dict) -> None:
        """Save benchmark history."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.benchmark_file, 'w') as f:
            json.dump(data, f, indent=2)

    def fetch_spy_data(self, days: int = 60) -> List[Dict]:
        """Fetch SPY price history."""
        try:
            import yfinance as yf
            spy = yf.Ticker('SPY')
            hist = spy.history(period=f'{days}d')

            data = []
            for date, row in hist.iterrows():
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'close': float(row['Close']),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low'])
                })
            return data
        except Exception as e:
            print(f"Error fetching SPY data: {e}")
            return []

    def update_benchmark(self) -> None:
        """Update SPY benchmark data."""
        print("Fetching SPY benchmark data...")
        spy_data = self.fetch_spy_data(days=90)

        if spy_data:
            benchmark = {
                'spy_data': spy_data,
                'last_updated': datetime.now().isoformat()
            }
            self.save_benchmark_history(benchmark)
            print(f"Updated benchmark: {len(spy_data)} days of SPY data")
        else:
            print("Failed to fetch SPY data")

    def calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate daily returns from prices."""
        if len(prices) < 2:
            return []
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1] * 100
            returns.append(ret)
        return returns

    def calculate_metrics(self, returns: List[float]) -> Dict:
        """Calculate performance metrics from returns."""
        if not returns:
            return {
                'total_return': 0,
                'avg_daily_return': 0,
                'volatility': 0,
                'sharpe': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'best_day': 0,
                'worst_day': 0
            }

        import statistics

        total_return = sum(returns)
        avg_return = statistics.mean(returns)
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0

        # Annualized Sharpe (assuming 252 trading days)
        sharpe = (avg_return * 252**0.5) / volatility if volatility > 0 else 0

        # Max drawdown
        cumulative = [100]
        for r in returns:
            cumulative.append(cumulative[-1] * (1 + r/100))

        peak = cumulative[0]
        max_dd = 0
        for val in cumulative:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Win rate
        wins = len([r for r in returns if r > 0])
        win_rate = wins / len(returns) * 100 if returns else 0

        return {
            'total_return': round(total_return, 2),
            'avg_daily_return': round(avg_return, 3),
            'volatility': round(volatility, 3),
            'sharpe': round(sharpe, 2),
            'max_drawdown': round(max_dd, 2),
            'win_rate': round(win_rate, 1),
            'best_day': round(max(returns), 2) if returns else 0,
            'worst_day': round(min(returns), 2) if returns else 0
        }

    def get_wallet_returns(self, snapshots: List[Dict]) -> Tuple[List[str], List[float]]:
        """Extract dates and returns from wallet snapshots."""
        if len(snapshots) < 2:
            return [], []

        dates = []
        returns = []

        for i in range(1, len(snapshots)):
            prev = snapshots[i-1]['equity']
            curr = snapshots[i]['equity']
            ret = (curr - prev) / prev * 100

            dates.append(snapshots[i]['date'])
            returns.append(ret)

        return dates, returns

    def get_spy_returns(self, spy_data: List[Dict], start_date: str) -> Tuple[List[str], List[float]]:
        """Get SPY returns aligned with wallet dates."""
        if len(spy_data) < 2:
            return [], []

        # Filter to start from start_date
        filtered = [d for d in spy_data if d['date'] >= start_date]

        if len(filtered) < 2:
            return [], []

        dates = []
        returns = []

        for i in range(1, len(filtered)):
            prev = filtered[i-1]['close']
            curr = filtered[i]['close']
            ret = (curr - prev) / prev * 100

            dates.append(filtered[i]['date'])
            returns.append(ret)

        return dates, returns

    def generate_comparison(self) -> Dict:
        """Generate full performance comparison."""
        snapshots = self.load_wallet_snapshots()
        benchmark = self.load_benchmark_history()
        trades = self.load_trade_history()
        wallet = self.load_wallet_state()

        # Get wallet performance
        wallet_dates, wallet_returns = self.get_wallet_returns(snapshots)

        # Get SPY performance (aligned to wallet start)
        start_date = snapshots[0]['date'] if snapshots else datetime.now().strftime('%Y-%m-%d')
        spy_dates, spy_returns = self.get_spy_returns(benchmark.get('spy_data', []), start_date)

        # Align dates (use wallet dates as reference)
        aligned_spy_returns = []
        spy_by_date = {d: r for d, r in zip(spy_dates, spy_returns)}
        for date in wallet_dates:
            aligned_spy_returns.append(spy_by_date.get(date, 0))

        # Calculate metrics
        wallet_metrics = self.calculate_metrics(wallet_returns)
        spy_metrics = self.calculate_metrics(aligned_spy_returns)

        # Trade stats
        trade_list = trades.get('trades', [])
        trade_stats = {
            'total_trades': len(trade_list),
            'wins': len([t for t in trade_list if t.get('outcome') == 'win']),
            'losses': len([t for t in trade_list if t.get('outcome') == 'loss']),
            'avg_hold_days': sum(t.get('hold_days', 0) for t in trade_list) / len(trade_list) if trade_list else 0,
            'total_pnl': sum(t.get('pnl_dollars', 0) for t in trade_list)
        }
        trade_stats['win_rate'] = trade_stats['wins'] / trade_stats['total_trades'] * 100 if trade_stats['total_trades'] > 0 else 0

        # Current state
        current = {
            'equity': wallet.get('total_equity', 100000),
            'initial': wallet.get('initial_capital', 100000),
            'cash': wallet.get('cash', 100000),
            'positions': len(wallet.get('positions', {})),
            'return_pct': (wallet.get('total_equity', 100000) / wallet.get('initial_capital', 100000) - 1) * 100
        }

        return {
            'generated_at': datetime.now().isoformat(),
            'tracking_days': len(wallet_returns),
            'current': current,
            'wallet': {
                'dates': wallet_dates,
                'returns': wallet_returns,
                'metrics': wallet_metrics
            },
            'spy': {
                'dates': spy_dates[-len(wallet_dates):] if spy_dates else [],
                'returns': aligned_spy_returns,
                'metrics': spy_metrics
            },
            'trades': trade_stats,
            'alpha': round(wallet_metrics['total_return'] - spy_metrics['total_return'], 2),
            'beta_adjusted_alpha': round(
                wallet_metrics['total_return'] - spy_metrics['total_return'] * 0.5, 2
            ) if spy_metrics['total_return'] != 0 else wallet_metrics['total_return']
        }

    def format_console(self, comparison: Dict) -> str:
        """Format comparison for console output."""
        lines = []

        lines.append("=" * 70)
        lines.append("PROTEUS PERFORMANCE DASHBOARD")
        lines.append("=" * 70)
        lines.append(f"Generated: {comparison['generated_at'][:19]}")
        lines.append(f"Tracking: {comparison['tracking_days']} days")
        lines.append("")

        # Current state
        c = comparison['current']
        return_sign = '+' if c['return_pct'] >= 0 else ''
        lines.append("PORTFOLIO STATUS")
        lines.append("-" * 40)
        lines.append(f"  Initial Capital:  ${c['initial']:>12,.0f}")
        lines.append(f"  Current Equity:   ${c['equity']:>12,.0f}  ({return_sign}{c['return_pct']:.2f}%)")
        lines.append(f"  Cash:             ${c['cash']:>12,.0f}")
        lines.append(f"  Open Positions:   {c['positions']}")
        lines.append("")

        # Comparison table
        w = comparison['wallet']['metrics']
        s = comparison['spy']['metrics']

        lines.append("PERFORMANCE COMPARISON")
        lines.append("-" * 40)
        lines.append(f"{'Metric':<20} {'Wallet':>12} {'SPY':>12} {'Diff':>12}")
        lines.append("-" * 56)

        metrics = [
            ('Total Return %', w['total_return'], s['total_return']),
            ('Avg Daily %', w['avg_daily_return'], s['avg_daily_return']),
            ('Volatility %', w['volatility'], s['volatility']),
            ('Sharpe Ratio', w['sharpe'], s['sharpe']),
            ('Max Drawdown %', w['max_drawdown'], s['max_drawdown']),
            ('Win Rate %', w['win_rate'], s['win_rate']),
            ('Best Day %', w['best_day'], s['best_day']),
            ('Worst Day %', w['worst_day'], s['worst_day']),
        ]

        for name, wallet_val, spy_val in metrics:
            diff = wallet_val - spy_val
            diff_str = f"{'+' if diff >= 0 else ''}{diff:.2f}"
            lines.append(f"{name:<20} {wallet_val:>12.2f} {spy_val:>12.2f} {diff_str:>12}")

        lines.append("")

        # Alpha
        lines.append("ALPHA ANALYSIS")
        lines.append("-" * 40)
        alpha = comparison['alpha']
        alpha_sign = '+' if alpha >= 0 else ''
        lines.append(f"  Raw Alpha (vs SPY):     {alpha_sign}{alpha:.2f}%")
        lines.append(f"  Risk-Adjusted Alpha:    {alpha_sign}{comparison['beta_adjusted_alpha']:.2f}%")
        lines.append("")

        # Trade stats
        t = comparison['trades']
        lines.append("TRADE STATISTICS")
        lines.append("-" * 40)
        lines.append(f"  Total Trades:    {t['total_trades']}")
        lines.append(f"  Wins/Losses:     {t['wins']}/{t['losses']}")
        lines.append(f"  Win Rate:        {t['win_rate']:.1f}%")
        lines.append(f"  Avg Hold Days:   {t['avg_hold_days']:.1f}")
        lines.append(f"  Total P&L:       ${t['total_pnl']:,.2f}")
        lines.append("")

        # Verdict
        lines.append("=" * 70)
        if comparison['tracking_days'] < 10:
            lines.append("STATUS: Insufficient data (need 10+ days for meaningful comparison)")
        elif alpha > 2:
            lines.append(f"STATUS: OUTPERFORMING SPY by {alpha:.1f}% - System showing promise")
        elif alpha > 0:
            lines.append(f"STATUS: Slightly ahead of SPY by {alpha:.1f}% - Continue monitoring")
        elif alpha > -2:
            lines.append(f"STATUS: Tracking SPY (within 2%) - Normal variance")
        else:
            lines.append(f"STATUS: UNDERPERFORMING SPY by {abs(alpha):.1f}% - Review needed")
        lines.append("=" * 70)

        return "\n".join(lines)

    def format_html(self, comparison: Dict) -> str:
        """Format comparison as HTML dashboard."""
        c = comparison['current']
        w = comparison['wallet']['metrics']
        s = comparison['spy']['metrics']
        t = comparison['trades']
        alpha = comparison['alpha']

        return_color = '#10b981' if c['return_pct'] >= 0 else '#ef4444'
        alpha_color = '#10b981' if alpha >= 0 else '#ef4444'

        # Build metrics rows
        metrics_rows = ""
        metrics = [
            ('Total Return', w['total_return'], s['total_return'], '%'),
            ('Sharpe Ratio', w['sharpe'], s['sharpe'], ''),
            ('Max Drawdown', w['max_drawdown'], s['max_drawdown'], '%'),
            ('Volatility', w['volatility'], s['volatility'], '%'),
            ('Win Rate', w['win_rate'], s['win_rate'], '%'),
        ]

        for name, wallet_val, spy_val, suffix in metrics:
            diff = wallet_val - spy_val
            diff_color = '#10b981' if diff >= 0 else '#ef4444'
            diff_sign = '+' if diff >= 0 else ''
            metrics_rows += f"""
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #eee;">{name}</td>
                <td style="padding: 12px; border-bottom: 1px solid #eee; text-align: right;">{wallet_val:.2f}{suffix}</td>
                <td style="padding: 12px; border-bottom: 1px solid #eee; text-align: right;">{spy_val:.2f}{suffix}</td>
                <td style="padding: 12px; border-bottom: 1px solid #eee; text-align: right; color: {diff_color};">{diff_sign}{diff:.2f}{suffix}</td>
            </tr>
            """

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Proteus Performance Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .card {{ background: white; border-radius: 12px; padding: 24px; margin: 16px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  color: white; padding: 30px; border-radius: 12px; text-align: center; }}
        .metric {{ display: inline-block; margin: 15px 25px 15px 0; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; }}
        .metric-label {{ font-size: 0.9em; color: #666; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background: #f8f9fa; padding: 12px; text-align: left; font-weight: 600; }}
        .alpha-box {{ background: linear-gradient(135deg, #667eea22, #764ba222);
                     padding: 20px; border-radius: 8px; text-align: center; }}
        .alpha-value {{ font-size: 2.5em; font-weight: bold; color: {alpha_color}; }}
        .status {{ padding: 15px; border-radius: 8px; text-align: center; font-weight: 500; }}
        .status.good {{ background: #d1fae5; color: #065f46; }}
        .status.neutral {{ background: #fef3c7; color: #92400e; }}
        .status.bad {{ background: #fee2e2; color: #991b1b; }}
    </style>
</head>
<body>
    <div class="header">
        <h1 style="margin: 0;">Proteus Performance Dashboard</h1>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">Tracking: {comparison['tracking_days']} days</p>
    </div>

    <div class="card">
        <h2 style="margin-top: 0;">Portfolio Status</h2>
        <div class="metric">
            <div class="metric-value">${c['equity']:,.0f}</div>
            <div class="metric-label">Current Equity</div>
        </div>
        <div class="metric">
            <div class="metric-value" style="color: {return_color};">
                {'+' if c['return_pct'] >= 0 else ''}{c['return_pct']:.2f}%
            </div>
            <div class="metric-label">Total Return</div>
        </div>
        <div class="metric">
            <div class="metric-value">{c['positions']}</div>
            <div class="metric-label">Open Positions</div>
        </div>
        <div class="metric">
            <div class="metric-value">{t['total_trades']}</div>
            <div class="metric-label">Total Trades</div>
        </div>
    </div>

    <div class="card">
        <h2 style="margin-top: 0;">Alpha vs SPY</h2>
        <div class="alpha-box">
            <div class="alpha-value">{'+' if alpha >= 0 else ''}{alpha:.2f}%</div>
            <div style="color: #666; margin-top: 8px;">Excess Return vs Benchmark</div>
        </div>
    </div>

    <div class="card">
        <h2 style="margin-top: 0;">Performance Comparison</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th style="text-align: right;">Proteus</th>
                <th style="text-align: right;">SPY</th>
                <th style="text-align: right;">Difference</th>
            </tr>
            {metrics_rows}
        </table>
    </div>

    <div class="card">
        <h2 style="margin-top: 0;">Trade Statistics</h2>
        <div class="metric">
            <div class="metric-value">{t['win_rate']:.1f}%</div>
            <div class="metric-label">Win Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{t['wins']}/{t['losses']}</div>
            <div class="metric-label">Wins/Losses</div>
        </div>
        <div class="metric">
            <div class="metric-value">{t['avg_hold_days']:.1f}</div>
            <div class="metric-label">Avg Hold Days</div>
        </div>
        <div class="metric">
            <div class="metric-value">${t['total_pnl']:,.0f}</div>
            <div class="metric-label">Total P&L</div>
        </div>
    </div>

    <div class="card">
        <div class="status {'good' if alpha > 2 else 'neutral' if alpha > -2 else 'bad'}">
            {'OUTPERFORMING' if alpha > 2 else 'TRACKING' if alpha > -2 else 'UNDERPERFORMING'} SPY
            {'- System showing promise' if alpha > 2 else '- Continue monitoring' if alpha > -2 else '- Review needed'}
        </div>
    </div>

    <p style="text-align: center; color: #666; font-size: 0.9em;">
        Generated: {comparison['generated_at'][:19]} |
        <a href="https://github.com/reivash/proteus">Proteus Trading System</a>
    </p>
</body>
</html>
"""
        return html


def main():
    parser = argparse.ArgumentParser(description='Performance Dashboard')
    parser.add_argument('--html', action='store_true', help='Generate HTML dashboard')
    parser.add_argument('--update', action='store_true', help='Update SPY benchmark data')
    parser.add_argument('--save', type=str, help='Save output to file')

    args = parser.parse_args()

    dashboard = PerformanceDashboard()

    # Update benchmark if requested or if stale
    benchmark = dashboard.load_benchmark_history()
    if args.update or not benchmark.get('spy_data'):
        dashboard.update_benchmark()

    # Generate comparison
    comparison = dashboard.generate_comparison()

    # Output
    if args.html:
        output = dashboard.format_html(comparison)
        filename = args.save or f"performance_dashboard_{datetime.now().strftime('%Y%m%d')}.html"
        with open(filename, 'w') as f:
            f.write(output)
        print(f"Dashboard saved to: {filename}")
    else:
        output = dashboard.format_console(comparison)
        print(output)

        if args.save:
            with open(args.save, 'w') as f:
                f.write(output)
            print(f"\nSaved to: {args.save}")


if __name__ == '__main__':
    main()

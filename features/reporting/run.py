#!/usr/bin/env python
"""
Daily Report Generator - Consolidated system status report.

Generates a comprehensive daily report including:
- Virtual wallet status and positions
- Recent trade history
- Performance metrics
- Market regime analysis
- Bear detection status
- System health

Usage:
    python scripts/generate_daily_report.py              # Print to console
    python scripts/generate_daily_report.py --html       # Generate HTML report
    python scripts/generate_daily_report.py --email      # Email the report
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class DailyReportGenerator:
    """Generate comprehensive daily reports."""

    def __init__(self):
        self.data_dir = Path('data')
        self.wallet_dir = self.data_dir / 'virtual_wallet'
        self.scans_dir = self.data_dir / 'smart_scans'

    def load_wallet_state(self) -> Dict:
        """Load virtual wallet state."""
        state_file = self.wallet_dir / 'wallet_state.json'
        if state_file.exists():
            with open(state_file) as f:
                return json.load(f)
        return {}

    def load_trade_history(self) -> Dict:
        """Load trade history."""
        history_file = self.wallet_dir / 'trade_history.json'
        if history_file.exists():
            with open(history_file) as f:
                return json.load(f)
        return {'trades': [], 'summary': {}}

    def load_daily_snapshots(self) -> Dict:
        """Load daily snapshots."""
        snapshots_file = self.wallet_dir / 'daily_snapshots.json'
        if snapshots_file.exists():
            with open(snapshots_file) as f:
                return json.load(f)
        return {'snapshots': []}

    def load_latest_scan(self) -> Dict:
        """Load latest scan results."""
        scan_file = self.scans_dir / 'latest_scan.json'
        if scan_file.exists():
            with open(scan_file) as f:
                return json.load(f)
        return {}

    def calculate_metrics(self, trades: List[Dict], snapshots: List[Dict]) -> Dict:
        """Calculate performance metrics."""
        metrics = {
            'total_trades': len(trades),
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'best_trade': None,
            'worst_trade': None,
            'avg_hold_days': 0,
            'max_drawdown': 0,
            'current_streak': 0,
            'streak_type': 'none'
        }

        if not trades:
            return metrics

        wins = [t for t in trades if t.get('outcome') == 'win']
        losses = [t for t in trades if t.get('outcome') == 'loss']

        metrics['wins'] = len(wins)
        metrics['losses'] = len(losses)
        metrics['win_rate'] = round(len(wins) / len(trades) * 100, 1)

        if wins:
            metrics['avg_win'] = round(sum(t['pnl_pct'] for t in wins) / len(wins), 2)
            metrics['best_trade'] = max(wins, key=lambda x: x['pnl_pct'])

        if losses:
            metrics['avg_loss'] = round(sum(t['pnl_pct'] for t in losses) / len(losses), 2)
            metrics['worst_trade'] = min(losses, key=lambda x: x['pnl_pct'])

        # Profit factor
        gross_profit = sum(t['pnl_dollars'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl_dollars'] for t in losses)) if losses else 0
        if gross_loss > 0:
            metrics['profit_factor'] = round(gross_profit / gross_loss, 2)

        # Average hold days
        if trades:
            metrics['avg_hold_days'] = round(
                sum(t.get('hold_days', 0) for t in trades) / len(trades), 1
            )

        # Current streak
        if trades:
            streak = 0
            streak_type = trades[-1].get('outcome', 'none')
            for t in reversed(trades):
                if t.get('outcome') == streak_type:
                    streak += 1
                else:
                    break
            metrics['current_streak'] = streak
            metrics['streak_type'] = streak_type

        # Max drawdown from snapshots
        if len(snapshots) >= 2:
            equities = [s['equity'] for s in snapshots]
            peak = equities[0]
            max_dd = 0
            for equity in equities:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            metrics['max_drawdown'] = round(max_dd, 2)

        return metrics

    def get_recent_trades(self, trades: List[Dict], days: int = 7) -> List[Dict]:
        """Get trades from the last N days."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return [t for t in trades if t.get('exit_date', '') >= cutoff]

    def generate_report(self) -> Dict:
        """Generate the complete report data."""
        # Load all data
        wallet = self.load_wallet_state()
        history = self.load_trade_history()
        snapshots = self.load_daily_snapshots()
        scan = self.load_latest_scan()

        trades = history.get('trades', [])
        snapshot_list = snapshots.get('snapshots', [])

        # Calculate metrics
        metrics = self.calculate_metrics(trades, snapshot_list)
        recent_trades = self.get_recent_trades(trades, days=7)

        # Build report
        report = {
            'generated_at': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),

            'portfolio': {
                'initial_capital': wallet.get('initial_capital', 100000),
                'total_equity': wallet.get('total_equity', 100000),
                'cash': wallet.get('cash', 100000),
                'return_pct': round(
                    ((wallet.get('total_equity', 100000) / wallet.get('initial_capital', 100000)) - 1) * 100, 2
                ) if wallet.get('initial_capital') else 0,
                'positions': wallet.get('positions', {}),
                'num_positions': len(wallet.get('positions', {})),
                'last_updated': wallet.get('last_updated', 'Never')
            },

            'market': {
                'regime': scan.get('regime', 'unknown'),
                'regime_confidence': scan.get('regime_confidence', 0),
                'vix': scan.get('vix', 0),
                'bear_score': scan.get('bear_score', 0),
                'bear_alert_level': scan.get('bear_alert_level', 'UNKNOWN'),
                'bear_triggers': scan.get('bear_triggers', [])[:5],
                'is_choppy': scan.get('is_choppy', False),
                'adx': scan.get('adx', 0),
                'scan_timestamp': scan.get('timestamp', 'Never')
            },

            'signals': {
                'count': len(scan.get('signals', [])),
                'top_signals': scan.get('signals', [])[:5],
                'rebalance_actions': scan.get('rebalance_actions', [])
            },

            'performance': {
                **metrics,
                'total_pnl': history.get('summary', {}).get('total_pnl', 0)
            },

            'recent_activity': {
                'trades_7d': len(recent_trades),
                'recent_trades': recent_trades[-5:] if recent_trades else []
            },

            'equity_curve': {
                'snapshots': snapshot_list[-30:] if snapshot_list else [],
                'days_tracked': len(snapshot_list)
            }
        }

        return report

    def format_console_report(self, report: Dict) -> str:
        """Format report for console output."""
        lines = []

        # Header
        lines.append("=" * 70)
        lines.append(f"PROTEUS DAILY REPORT - {report['date']}")
        lines.append("=" * 70)

        # Portfolio Summary
        p = report['portfolio']
        return_sign = '+' if p['return_pct'] >= 0 else ''
        lines.append(f"\nPORTFOLIO SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Initial Capital:  ${p['initial_capital']:>12,.0f}")
        lines.append(f"Current Equity:   ${p['total_equity']:>12,.0f}  ({return_sign}{p['return_pct']:.2f}%)")
        lines.append(f"Cash:             ${p['cash']:>12,.0f}")
        lines.append(f"Positions Value:  ${p['total_equity'] - p['cash']:>12,.0f}")
        lines.append(f"Open Positions:   {p['num_positions']}")

        # Open Positions
        if p['positions']:
            lines.append(f"\nOPEN POSITIONS")
            lines.append("-" * 40)
            lines.append(f"{'Ticker':<8} {'Entry':>10} {'Current':>10} {'P&L':>10} {'Days':>6}")
            for ticker, pos in p['positions'].items():
                pnl = pos.get('current_pnl_pct', 0)
                pnl_sign = '+' if pnl >= 0 else ''
                lines.append(
                    f"{ticker:<8} ${pos['entry_price']:>8.2f} ${pos.get('current_price', pos['entry_price']):>8.2f} "
                    f"{pnl_sign}{pnl:>6.1f}% {pos.get('days_held', 0):>5}d"
                )

        # Market Conditions
        m = report['market']
        lines.append(f"\nMARKET CONDITIONS")
        lines.append("-" * 40)
        lines.append(f"Regime:      {m['regime'].upper()} ({m['regime_confidence']*100:.0f}% confidence)")
        lines.append(f"VIX:         {m['vix']:.1f}")
        lines.append(f"Bear Score:  {m['bear_score']:.0f}/100 ({m['bear_alert_level']})")
        lines.append(f"ADX:         {m['adx']:.1f} {'[CHOPPY]' if m['is_choppy'] else ''}")

        if m['bear_triggers']:
            lines.append(f"Bear Triggers:")
            for trigger in m['bear_triggers'][:3]:
                lines.append(f"  - {trigger}")

        # Signals
        s = report['signals']
        lines.append(f"\nSIGNALS")
        lines.append("-" * 40)
        lines.append(f"Actionable Signals: {s['count']}")

        if s['top_signals']:
            lines.append("Top Signals:")
            for sig in s['top_signals'][:3]:
                lines.append(f"  {sig['ticker']}: {sig['adjusted_strength']:.0f} ({sig['tier']})")

        if s['rebalance_actions']:
            lines.append("Rebalance Actions:")
            for action in s['rebalance_actions']:
                lines.append(f"  {action['ticker']}: {action['recommendation']}")

        # Performance
        perf = report['performance']
        lines.append(f"\nPERFORMANCE METRICS")
        lines.append("-" * 40)
        lines.append(f"Total Trades:    {perf['total_trades']}")
        lines.append(f"Win Rate:        {perf['win_rate']:.1f}%")
        lines.append(f"Avg Win:         +{perf['avg_win']:.1f}%")
        lines.append(f"Avg Loss:        {perf['avg_loss']:.1f}%")
        lines.append(f"Profit Factor:   {perf['profit_factor']:.2f}")
        lines.append(f"Max Drawdown:    {perf['max_drawdown']:.1f}%")
        lines.append(f"Avg Hold Days:   {perf['avg_hold_days']:.1f}")
        lines.append(f"Total P&L:       ${perf['total_pnl']:,.2f}")

        if perf['current_streak'] > 1:
            lines.append(f"Current Streak:  {perf['current_streak']} {perf['streak_type']}s")

        # Recent Activity
        recent = report['recent_activity']
        if recent['recent_trades']:
            lines.append(f"\nRECENT TRADES (Last 7 days: {recent['trades_7d']})")
            lines.append("-" * 40)
            for trade in recent['recent_trades'][-5:]:
                outcome = 'W' if trade['outcome'] == 'win' else 'L'
                pnl_sign = '+' if trade['pnl_pct'] >= 0 else ''
                lines.append(
                    f"[{outcome}] {trade['ticker']:<6} {pnl_sign}{trade['pnl_pct']:>5.1f}% "
                    f"${trade['pnl_dollars']:>7.0f}  {trade.get('exit_reason', '')[:25]}"
                )

        lines.append("")
        lines.append("=" * 70)
        lines.append(f"Generated: {report['generated_at']}")

        return "\n".join(lines)

    def format_html_report(self, report: Dict) -> str:
        """Format report as HTML."""
        p = report['portfolio']
        m = report['market']
        perf = report['performance']
        return_color = '#10b981' if p['return_pct'] >= 0 else '#ef4444'
        return_sign = '+' if p['return_pct'] >= 0 else ''

        # Positions HTML
        positions_html = ""
        for ticker, pos in p['positions'].items():
            pnl = pos.get('current_pnl_pct', 0)
            pnl_color = '#10b981' if pnl >= 0 else '#ef4444'
            pnl_sign = '+' if pnl >= 0 else ''
            positions_html += f"""
            <tr>
                <td style="padding: 8px;">{ticker}</td>
                <td style="padding: 8px;">${pos['entry_price']:.2f}</td>
                <td style="padding: 8px;">${pos.get('current_price', pos['entry_price']):.2f}</td>
                <td style="padding: 8px; color: {pnl_color};">{pnl_sign}{pnl:.1f}%</td>
                <td style="padding: 8px;">{pos.get('days_held', 0)}d</td>
            </tr>
            """
        if not positions_html:
            positions_html = '<tr><td colspan="5" style="padding: 8px; text-align: center;">No open positions</td></tr>'

        # Bear triggers HTML
        triggers_html = ""
        for trigger in m.get('bear_triggers', [])[:5]:
            triggers_html += f"<li>{trigger}</li>"
        if not triggers_html:
            triggers_html = "<li>No active triggers</li>"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Proteus Daily Report - {report['date']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 8px; }}
        .section {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background: #e5e7eb; padding: 10px; text-align: left; }}
        td {{ border-bottom: 1px solid #eee; }}
    </style>
</head>
<body>
    <div class="header">
        <h1 style="margin: 0;">Proteus Daily Report</h1>
        <p style="margin: 10px 0 0 0;">{report['date']}</p>
    </div>

    <div class="section" style="border-left: 5px solid {return_color};">
        <h2 style="margin-top: 0;">Portfolio Summary</h2>
        <div class="metric">
            <div>Initial Capital</div>
            <div class="metric-value">${p['initial_capital']:,.0f}</div>
        </div>
        <div class="metric">
            <div>Current Equity</div>
            <div class="metric-value" style="color: {return_color};">${p['total_equity']:,.0f} ({return_sign}{p['return_pct']:.2f}%)</div>
        </div>
        <div class="metric">
            <div>Cash</div>
            <div class="metric-value">${p['cash']:,.0f}</div>
        </div>
    </div>

    <div class="section">
        <h2 style="margin-top: 0;">Open Positions ({p['num_positions']})</h2>
        <table>
            <tr><th>Ticker</th><th>Entry</th><th>Current</th><th>P&L</th><th>Days</th></tr>
            {positions_html}
        </table>
    </div>

    <div class="section" style="border-left: 5px solid #667eea;">
        <h2 style="margin-top: 0;">Market Conditions</h2>
        <div class="metric">
            <div>Regime</div>
            <div class="metric-value">{m['regime'].upper()}</div>
        </div>
        <div class="metric">
            <div>VIX</div>
            <div class="metric-value">{m['vix']:.1f}</div>
        </div>
        <div class="metric">
            <div>Bear Score</div>
            <div class="metric-value">{m['bear_score']:.0f}/100</div>
        </div>
        <div class="metric">
            <div>Alert Level</div>
            <div class="metric-value">{m['bear_alert_level']}</div>
        </div>
        <h3>Bear Triggers</h3>
        <ul>{triggers_html}</ul>
    </div>

    <div class="section" style="border-left: 5px solid #10b981;">
        <h2 style="margin-top: 0;">Performance ({perf['total_trades']} trades)</h2>
        <div class="metric">
            <div>Win Rate</div>
            <div class="metric-value">{perf['win_rate']:.1f}%</div>
        </div>
        <div class="metric">
            <div>Profit Factor</div>
            <div class="metric-value">{perf['profit_factor']:.2f}</div>
        </div>
        <div class="metric">
            <div>Max Drawdown</div>
            <div class="metric-value">{perf['max_drawdown']:.1f}%</div>
        </div>
        <div class="metric">
            <div>Total P&L</div>
            <div class="metric-value">${perf['total_pnl']:,.2f}</div>
        </div>
    </div>

    <p style="color: #666; font-size: 0.9em; margin-top: 30px;">
        Generated: {report['generated_at']}<br>
        <em>Proteus Trading System - Virtual Wallet Report</em>
    </p>
</body>
</html>
"""
        return html


def main():
    parser = argparse.ArgumentParser(description='Generate Proteus Daily Report')
    parser.add_argument('--html', action='store_true', help='Generate HTML report')
    parser.add_argument('--save', type=str, help='Save report to file')
    parser.add_argument('--email', action='store_true', help='Email the report')

    args = parser.parse_args()

    generator = DailyReportGenerator()
    report = generator.generate_report()

    if args.html:
        output = generator.format_html_report(report)
        if args.save:
            with open(args.save, 'w') as f:
                f.write(output)
            print(f"HTML report saved to: {args.save}")
        else:
            # Save to default location
            filename = f"daily_report_{report['date']}.html"
            with open(filename, 'w') as f:
                f.write(output)
            print(f"HTML report saved to: {filename}")
    elif args.email:
        # Email the report
        try:
            from trading.virtual_wallet import VirtualWallet
            wallet = VirtualWallet()
            summary = {
                'date': report['date'],
                'initial_capital': report['portfolio']['initial_capital'],
                'total_equity': report['portfolio']['total_equity'],
                'cash': report['portfolio']['cash'],
                'return_pct': report['portfolio']['return_pct'],
                'positions': report['portfolio']['positions'],
                'num_positions': report['portfolio']['num_positions'],
                'entries': [],
                'exits': [],
                'regime': report['market']['regime'],
                'metrics': report['performance']
            }
            if wallet.send_daily_email(summary):
                print("Report emailed successfully!")
            else:
                print("Email failed - check configuration")
        except Exception as e:
            print(f"Email error: {e}")
    else:
        # Console output
        output = generator.format_console_report(report)
        print(output)

        if args.save:
            with open(args.save, 'w') as f:
                f.write(output)
            print(f"\nReport saved to: {args.save}")


if __name__ == '__main__':
    main()

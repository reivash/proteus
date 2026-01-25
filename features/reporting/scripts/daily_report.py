#!/usr/bin/env python
"""
Daily Trading Report Generator

Generates a comprehensive daily report including:
1. Market regime analysis
2. Current position status
3. Rebalancing actions needed
4. Today's signals
5. Performance summary

Usage:
    python daily_report.py           # Print to console
    python daily_report.py --html    # Generate HTML report
"""

import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


def load_positions() -> Dict:
    """Load active positions."""
    pos_file = Path('features/simulation/data/portfolio/active_positions.json')
    if pos_file.exists():
        with open(pos_file) as f:
            return json.load(f)
    return {'positions': {}}


def load_latest_scan() -> Dict:
    """Load latest scan results."""
    scan_file = Path('features/daily_picks/data/smart_scans/latest_scan.json')
    if scan_file.exists():
        with open(scan_file) as f:
            return json.load(f)
    return {}


def load_config() -> Dict:
    """Load unified config."""
    config_file = Path('features/daily_picks/config.json')
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}


def get_ticker_tier(ticker: str, config: Dict) -> str:
    """Get tier for a ticker."""
    for tier, data in config.get('stock_tiers', {}).items():
        if not tier.startswith('_') and ticker in data.get('tickers', []):
            return tier
    return 'average'


def format_position_status(positions: Dict, config: Dict) -> List[str]:
    """Format position status lines."""
    lines = []

    if not positions.get('positions'):
        lines.append("  No active positions")
        return lines

    for ticker, info in positions['positions'].items():
        entry_price = info.get('entry_price', 0)
        entry_date = info.get('entry_date', '')
        size_pct = info.get('size_pct', 0) * 100

        if entry_date:
            try:
                entry_dt = datetime.fromisoformat(entry_date)
                days_held = (datetime.now() - entry_dt).days
            except ValueError:
                days_held = 0
        else:
            days_held = 0

        tier = get_ticker_tier(ticker, config)
        exit_config = config.get('exit_strategy', {}).get(tier, {})
        max_hold = exit_config.get('max_hold_days', 5)

        hold_warning = " [!]" if days_held >= max_hold else ""

        lines.append(f"  {ticker}: {size_pct:.1f}% @ ${entry_price:.2f} "
                     f"(Day {days_held}/{max_hold}){hold_warning}")

    return lines


def format_signals(scan: Dict) -> List[str]:
    """Format signal lines."""
    lines = []

    signals = scan.get('signals', [])
    if not signals:
        lines.append("  No actionable signals")
        return lines

    for sig in signals[:5]:
        ticker = sig.get('ticker', '?')
        strength = sig.get('adjusted_strength', 0)
        tier = sig.get('tier', 'average')
        size = sig.get('size_pct', 0)
        quality = sig.get('quality', 'moderate')

        lines.append(f"  {ticker}: {strength:.1f} ({tier}) -> {size:.1f}% [{quality}]")

    if len(signals) > 5:
        lines.append(f"  ... and {len(signals) - 5} more")

    return lines


def generate_text_report() -> str:
    """Generate text report."""
    positions = load_positions()
    scan = load_latest_scan()
    config = load_config()

    lines = []
    lines.append("=" * 60)
    lines.append("PROTEUS DAILY TRADING REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    # Market Regime
    lines.append("")
    lines.append("MARKET REGIME")
    lines.append("-" * 40)
    regime = scan.get('regime', {})
    if isinstance(regime, dict):
        regime_type = regime.get('type', 'unknown').upper()
        confidence = regime.get('confidence', 0) * 100
        vix = regime.get('vix', 0)
        lines.append(f"  Regime: {regime_type} ({confidence:.0f}% confidence)")
        lines.append(f"  VIX: {vix:.1f}")
    elif isinstance(regime, str):
        lines.append(f"  Regime: {regime.upper()}")
        lines.append(f"  VIX: {scan.get('vix', 0):.1f}")
    else:
        lines.append("  Regime: Unknown")

    # Portfolio Status
    lines.append("")
    lines.append("PORTFOLIO STATUS")
    lines.append("-" * 40)

    total_positions = len(positions.get('positions', {}))
    max_positions = config.get('portfolio_constraints', {}).get('max_positions', 6)
    lines.append(f"  Positions: {total_positions}/{max_positions}")

    total_heat = sum(
        p.get('size_pct', 0) if isinstance(p, dict) else p
        for p in positions.get('positions', {}).values()
    ) * 100
    max_heat = config.get('portfolio_constraints', {}).get('max_portfolio_heat_pct', 15)
    lines.append(f"  Heat: {total_heat:.1f}% / {max_heat}%")

    lines.append("")
    lines.append("Active Positions:")
    lines.extend(format_position_status(positions, config))

    # Rebalancing
    lines.append("")
    lines.append("REBALANCING ACTIONS")
    lines.append("-" * 40)

    rebalance = scan.get('rebalance_actions', [])
    if rebalance:
        for action in rebalance:
            lines.append(f"  {action.get('ticker')}: {action.get('recommendation')}")
    else:
        lines.append("  No actions needed")

    # Today's Signals
    lines.append("")
    lines.append("TODAY'S SIGNALS")
    lines.append("-" * 40)

    scan_time = scan.get('timestamp', '')
    if scan_time:
        try:
            scan_dt = datetime.fromisoformat(scan_time.replace('Z', '+00:00'))
            age_mins = (datetime.now() - scan_dt.replace(tzinfo=None)).seconds // 60
            lines.append(f"  (Scan age: {age_mins} minutes)")
        except:
            pass

    lines.extend(format_signals(scan))

    # Summary Stats
    lines.append("")
    lines.append("SCAN STATISTICS")
    lines.append("-" * 40)

    stats = scan.get('stats', scan.get('summary', {}))
    lines.append(f"  Scanned: {stats.get('total_scanned', 0)}")
    lines.append(f"  Passed: {stats.get('passed', stats.get('passed_filter', 0))}")
    lines.append(f"  Filtered: {stats.get('filtered', stats.get('filtered_out', 0))}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def generate_html_report() -> str:
    """Generate HTML report."""
    positions = load_positions()
    scan = load_latest_scan()
    config = load_config()

    # Get regime info
    regime_info = scan.get('regime', {})
    if isinstance(regime_info, dict):
        regime_type = regime_info.get('type', 'unknown').upper()
        vix = regime_info.get('vix', 0)
    else:
        regime_type = str(regime_info).upper()
        vix = scan.get('vix', 0)

    # Regime color
    regime_colors = {
        'VOLATILE': '#ff6b6b',
        'BEAR': '#ffa94d',
        'CHOPPY': '#74c0fc',
        'BULL': '#69db7c'
    }
    regime_color = regime_colors.get(regime_type, '#868e96')

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Proteus Daily Report</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
        h1 {{ color: #212529; border-bottom: 2px solid #228be6; padding-bottom: 10px; }}
        h2 {{ color: #495057; margin-top: 30px; }}
        .card {{ background: white; border-radius: 8px; padding: 15px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .regime {{ display: inline-block; padding: 5px 15px; border-radius: 20px; color: white; font-weight: bold; background: {regime_color}; }}
        .metric {{ display: inline-block; margin-right: 30px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #228be6; }}
        .metric-label {{ color: #868e96; font-size: 12px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background: #f1f3f5; font-weight: 600; }}
        .ticker {{ font-weight: bold; color: #228be6; }}
        .positive {{ color: #2f9e44; }}
        .negative {{ color: #e03131; }}
        .warning {{ background: #fff3cd; }}
        .timestamp {{ color: #868e96; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>Proteus Daily Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="card">
        <h2 style="margin-top:0">Market Regime</h2>
        <span class="regime">{regime_type}</span>
        <span style="margin-left: 20px">VIX: {vix:.1f}</span>
    </div>

    <div class="card">
        <h2 style="margin-top:0">Portfolio Status</h2>
        <div class="metric">
            <div class="metric-value">{len(positions.get('positions', {}))}/{config.get('portfolio_constraints', {}).get('max_positions', 6)}</div>
            <div class="metric-label">Positions</div>
        </div>
        <div class="metric">
            <div class="metric-value">{sum(p.get('size_pct', 0) if isinstance(p, dict) else p for p in positions.get('positions', {}).values()) * 100:.1f}%</div>
            <div class="metric-label">Heat</div>
        </div>
    </div>

    <div class="card">
        <h2 style="margin-top:0">Active Positions</h2>
        <table>
            <tr><th>Ticker</th><th>Size</th><th>Entry</th><th>Days Held</th></tr>
"""

    if positions.get('positions'):
        for ticker, info in positions['positions'].items():
            entry_price = info.get('entry_price', 0)
            entry_date = info.get('entry_date', '')
            size_pct = info.get('size_pct', 0) * 100
            days_held = 0
            if entry_date:
                try:
                    days_held = (datetime.now() - datetime.fromisoformat(entry_date)).days
                except:
                    pass
            tier = get_ticker_tier(ticker, config)
            max_hold = config.get('exit_strategy', {}).get(tier, {}).get('max_hold_days', 5)
            row_class = 'warning' if days_held >= max_hold else ''
            html += f"""            <tr class="{row_class}">
                <td class="ticker">{ticker}</td>
                <td>{size_pct:.1f}%</td>
                <td>${entry_price:.2f}</td>
                <td>{days_held}/{max_hold}</td>
            </tr>
"""
    else:
        html += """            <tr><td colspan="4">No active positions</td></tr>
"""

    html += """        </table>
    </div>

    <div class="card">
        <h2 style="margin-top:0">Today's Signals</h2>
        <table>
            <tr><th>Ticker</th><th>Strength</th><th>Tier</th><th>Size</th><th>Quality</th></tr>
"""

    signals = scan.get('signals', [])
    if signals:
        for sig in signals[:10]:
            html += f"""            <tr>
                <td class="ticker">{sig.get('ticker', '?')}</td>
                <td>{sig.get('adjusted_strength', 0):.1f}</td>
                <td>{sig.get('tier', 'average')}</td>
                <td>{sig.get('size_pct', 0):.1f}%</td>
                <td>{sig.get('quality', 'moderate')}</td>
            </tr>
"""
    else:
        html += """            <tr><td colspan="5">No actionable signals</td></tr>
"""

    html += """        </table>
    </div>

</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description='Generate daily trading report')
    parser.add_argument('--html', action='store_true', help='Generate HTML report')
    parser.add_argument('--save', action='store_true', help='Save to file')
    args = parser.parse_args()

    if args.html:
        report = generate_html_report()
        if args.save:
            filename = f"daily_report_{datetime.now().strftime('%Y%m%d')}.html"
            with open(filename, 'w') as f:
                f.write(report)
            print(f"Saved to {filename}")
        else:
            print(report)
    else:
        report = generate_text_report()
        if args.save:
            filename = f"daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
            with open(filename, 'w') as f:
                f.write(report)
            print(f"Saved to {filename}")
        else:
            print(report)


if __name__ == '__main__':
    main()

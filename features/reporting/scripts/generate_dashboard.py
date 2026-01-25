#!/usr/bin/env python
"""
Generate Trading Dashboard HTML

Creates an interactive dashboard showing:
1. Market regime and status
2. Active positions with P&L
3. Today's signals
4. System health

Usage:
    python generate_dashboard.py
    # Opens dashboard in browser
"""

import json
import webbrowser
from datetime import datetime
from pathlib import Path


def load_data():
    """Load all data sources."""
    data = {}

    # Positions
    pos_file = Path('features/simulation/data/portfolio/active_positions.json')
    if pos_file.exists():
        with open(pos_file) as f:
            data['positions'] = json.load(f)
    else:
        data['positions'] = {'positions': {}}

    # Latest scan
    scan_file = Path('features/daily_picks/data/smart_scans/latest_scan.json')
    if scan_file.exists():
        with open(scan_file) as f:
            data['scan'] = json.load(f)
    else:
        data['scan'] = {}

    # Config
    config_file = Path('features/daily_picks/config.json')
    if config_file.exists():
        with open(config_file) as f:
            data['config'] = json.load(f)
    else:
        data['config'] = {}

    return data


def generate_dashboard():
    """Generate dashboard HTML."""
    data = load_data()

    positions = data['positions'].get('positions', {})
    scan = data['scan']
    config = data['config']

    # Get regime
    regime_info = scan.get('regime', {})
    if isinstance(regime_info, dict):
        regime = regime_info.get('type', 'unknown').upper()
        vix = regime_info.get('vix', 0)
    else:
        regime = str(regime_info).upper() if regime_info else 'UNKNOWN'
        vix = scan.get('vix', 0)

    # Get bear score from scan or detect live (Jan 2026)
    bear_score = scan.get('bear_score', None)
    bear_alert_level = scan.get('bear_alert_level', 'NORMAL')
    bear_triggers = scan.get('bear_triggers', [])

    # If no bear score in scan, detect live
    if bear_score is None:
        try:
            from common.analysis.fast_bear_detector import FastBearDetector
            detector = FastBearDetector()
            bear_signal = detector.detect()
            bear_score = bear_signal.bear_score
            bear_alert_level = bear_signal.alert_level
            bear_triggers = bear_signal.triggers
        except Exception:
            bear_score = 0
            bear_alert_level = 'NORMAL'
            bear_triggers = []

    # Bear alert colors
    bear_colors = {
        'CRITICAL': '#dc2626',
        'WARNING': '#f59e0b',
        'WATCH': '#3b82f6',
        'NORMAL': '#10b981'
    }
    bear_color = bear_colors.get(bear_alert_level, '#868e96')

    regime_colors = {
        'VOLATILE': ('#ff6b6b', 'High opportunity, high risk'),
        'BEAR': ('#ffa94d', 'Good for mean reversion'),
        'CHOPPY': ('#74c0fc', 'Standard conditions'),
        'BULL': ('#69db7c', 'Conservative approach')
    }
    regime_color, regime_desc = regime_colors.get(regime, ('#868e96', 'Unknown'))

    # Calculate position metrics
    total_heat = sum(
        p.get('size_pct', 0) if isinstance(p, dict) else p
        for p in positions.values()
    ) * 100

    max_positions = config.get('portfolio_constraints', {}).get('max_positions', 6)
    max_heat = config.get('portfolio_constraints', {}).get('max_portfolio_heat_pct', 15)

    # Generate positions table
    positions_html = ""
    if positions:
        for ticker, info in positions.items():
            entry_price = info.get('entry_price', 0)
            size_pct = info.get('size_pct', 0) * 100
            entry_date = info.get('entry_date', '')
            tier = info.get('tier', 'average')

            if entry_date:
                try:
                    days = (datetime.now() - datetime.fromisoformat(entry_date)).days
                except:
                    days = 0
            else:
                days = 0

            positions_html += f"""
            <tr>
                <td class="ticker">{ticker}</td>
                <td>{tier}</td>
                <td>${entry_price:.2f}</td>
                <td>{size_pct:.1f}%</td>
                <td>Day {days}</td>
            </tr>"""
    else:
        positions_html = '<tr><td colspan="5" style="text-align:center;color:#868e96">No active positions</td></tr>'

    # Generate signals table
    signals = scan.get('signals', [])
    signals_html = ""
    if signals:
        for sig in signals[:8]:
            ticker = sig.get('ticker', '?')
            strength = sig.get('adjusted_strength', 0)
            tier = sig.get('tier', 'average')
            size = sig.get('size_pct', 0)
            quality = sig.get('quality', 'moderate')
            boosts = sig.get('boosts_applied', [])

            quality_colors = {
                'excellent': '#2f9e44',
                'strong': '#37b24d',
                'moderate': '#74c0fc',
                'weak': '#ffa94d'
            }
            q_color = quality_colors.get(quality, '#868e96')

            signals_html += f"""
            <tr>
                <td class="ticker">{ticker}</td>
                <td>{strength:.1f}</td>
                <td>{tier}</td>
                <td>{size:.1f}%</td>
                <td style="color:{q_color}">{quality}</td>
            </tr>"""
    else:
        signals_html = '<tr><td colspan="5" style="text-align:center;color:#868e96">No actionable signals</td></tr>'

    # Get scan timestamp
    scan_time = scan.get('timestamp', '')
    if scan_time:
        try:
            scan_dt = datetime.fromisoformat(scan_time.replace('Z', '+00:00'))
            scan_age = (datetime.now() - scan_dt.replace(tzinfo=None)).seconds // 60
            scan_status = f"{scan_age} min ago"
        except:
            scan_status = "Unknown"
    else:
        scan_status = "No scan"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Proteus Dashboard</title>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="300">
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', -apple-system, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e1e1e1;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            font-size: 28px;
            margin-bottom: 5px;
            color: #fff;
        }}
        .subtitle {{
            text-align: center;
            color: #868e96;
            margin-bottom: 30px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .card h2 {{
            margin-top: 0;
            font-size: 16px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #74c0fc;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding-bottom: 10px;
        }}
        .metric-row {{
            display: flex;
            justify-content: space-between;
            margin: 15px 0;
        }}
        .metric {{
            text-align: center;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: #fff;
        }}
        .metric-label {{
            font-size: 12px;
            color: #868e96;
            text-transform: uppercase;
        }}
        .regime-badge {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 18px;
            color: #fff;
            background: {regime_color};
        }}
        .regime-desc {{
            color: #868e96;
            margin-top: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{
            color: #868e96;
            font-weight: normal;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .ticker {{
            font-weight: bold;
            color: #74c0fc;
        }}
        .status-bar {{
            display: flex;
            justify-content: space-between;
            padding: 10px 20px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
        }}
        .status-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #2f9e44;
        }}
        .progress-bar {{
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 10px;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #74c0fc, #228be6);
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PROTEUS</h1>
        <p class="subtitle">Trading Dashboard</p>

        <div class="status-bar">
            <div class="status-item">
                <span class="status-dot"></span>
                System Online
            </div>
            <div class="status-item">
                Last Scan: {scan_status}
            </div>
            <div class="status-item">
                {datetime.now().strftime('%Y-%m-%d %H:%M')}
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Market Regime</h2>
                <div style="text-align:center;padding:20px 0">
                    <span class="regime-badge">{regime}</span>
                    <p class="regime-desc">{regime_desc}</p>
                </div>
                <div class="metric-row">
                    <div class="metric">
                        <div class="metric-value">{vix:.1f}</div>
                        <div class="metric-label">VIX</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{len(positions)}/{max_positions}</div>
                        <div class="metric-label">Positions</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{total_heat:.0f}%</div>
                        <div class="metric-label">Heat</div>
                    </div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width:{min(total_heat/max_heat*100, 100):.0f}%"></div>
                </div>
            </div>

            <div class="card">
                <h2>Bear Early Warning</h2>
                <div style="text-align:center;padding:15px 0">
                    <div style="font-size:48px;font-weight:bold;color:{bear_color}">{bear_score:.0f}</div>
                    <div style="font-size:12px;color:#868e96;text-transform:uppercase">Bear Score / 100</div>
                    <div style="margin-top:15px">
                        <span style="display:inline-block;padding:6px 16px;border-radius:20px;background:{bear_color};color:#fff;font-weight:bold;font-size:14px">{bear_alert_level}</span>
                    </div>
                </div>
                <div class="progress-bar" style="margin-top:15px;background:rgba(255,255,255,0.1)">
                    <div style="height:100%;width:{bear_score}%;background:{bear_color};border-radius:3px"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:10px;color:#868e96;margin-top:5px">
                    <span>NORMAL</span>
                    <span>WATCH</span>
                    <span>WARNING</span>
                    <span>CRITICAL</span>
                </div>
                {'<div style="margin-top:15px;font-size:12px;color:#868e96"><strong>Triggers:</strong> ' + ', '.join(bear_triggers[:2]) + '</div>' if bear_triggers else ''}
            </div>

            <div class="card">
                <h2>Active Positions</h2>
                <table>
                    <tr>
                        <th>Ticker</th>
                        <th>Tier</th>
                        <th>Entry</th>
                        <th>Size</th>
                        <th>Age</th>
                    </tr>
                    {positions_html}
                </table>
            </div>

            <div class="card" style="grid-column: span 2">
                <h2>Today's Signals ({len(signals)} actionable)</h2>
                <table>
                    <tr>
                        <th>Ticker</th>
                        <th>Strength</th>
                        <th>Tier</th>
                        <th>Size</th>
                        <th>Quality</th>
                    </tr>
                    {signals_html}
                </table>
            </div>
        </div>
    </div>
</body>
</html>"""

    return html


def main():
    html = generate_dashboard()

    # Save dashboard
    output_file = Path('dashboard.html')
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"Dashboard saved to {output_file}")

    # Open in browser
    webbrowser.open(str(output_file.absolute()))


if __name__ == '__main__':
    main()

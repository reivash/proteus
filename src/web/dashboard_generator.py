"""
Dashboard Generator

Generates HTML dashboard showing:
1. Current market regime
2. Latest signals with volatility metrics
3. Paper trading performance
4. System status
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional


def generate_dashboard(scan_data: Optional[Dict] = None,
                       paper_perf: Optional[Dict] = None,
                       output_path: str = "data/web_dashboard/index.html"):
    """
    Generate HTML dashboard from latest scan data.
    """
    # Load latest scan if not provided
    if scan_data is None:
        scan_file = "data/smart_scans/latest_scan.json"
        if os.path.exists(scan_file):
            with open(scan_file, 'r') as f:
                scan_data = json.load(f)
        else:
            scan_data = {}

    # Get regime info
    regime = scan_data.get('regime', {})
    regime_type = regime.get('type', 'unknown').upper()
    regime_conf = regime.get('confidence', 0) * 100
    vix = regime.get('vix', 0)
    recommendation = regime.get('recommendation', 'N/A')

    # Get signals
    signals = scan_data.get('signals', [])[:10]  # Top 10

    # Get summary
    summary = scan_data.get('summary', {})
    total_scanned = summary.get('total_scanned', 0)
    passed = summary.get('passed_filter', 0)
    filtered = summary.get('filtered_out', 0)

    # Timestamp
    scan_time = scan_data.get('timestamp', datetime.now().isoformat())

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="300">
    <title>Proteus Trading Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{
            color: #58a6ff;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        h2 {{ color: #8b949e; margin: 20px 0 10px; font-size: 1.1rem; }}
        .timestamp {{ color: #8b949e; font-size: 0.9rem; margin-bottom: 20px; }}

        .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
        }}
        .card-title {{ color: #8b949e; font-size: 0.85rem; margin-bottom: 8px; }}
        .card-value {{ font-size: 1.8rem; font-weight: 600; }}
        .card-value.bull {{ color: #3fb950; }}
        .card-value.bear {{ color: #f85149; }}
        .card-value.choppy {{ color: #d29922; }}
        .card-value.volatile {{ color: #a371f7; }}
        .card-sub {{ color: #8b949e; font-size: 0.8rem; margin-top: 5px; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: #161b22;
            border-radius: 8px;
            overflow: hidden;
            margin-top: 10px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #30363d;
        }}
        th {{
            background: #21262d;
            color: #8b949e;
            font-weight: 500;
            font-size: 0.85rem;
        }}
        tr:hover {{ background: #21262d; }}

        .tier {{
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .tier.elite {{ background: #238636; color: #fff; }}
        .tier.strong {{ background: #2ea043; color: #fff; }}
        .tier.good {{ background: #3fb950; color: #0d1117; }}
        .tier.moderate {{ background: #d29922; color: #0d1117; }}
        .tier.weak {{ background: #f85149; color: #fff; }}

        .positive {{ color: #3fb950; }}
        .negative {{ color: #f85149; }}

        .recommendation {{
            background: #21262d;
            border-left: 4px solid #58a6ff;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }}

        footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #30363d;
            color: #8b949e;
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Proteus Trading Dashboard</h1>
        <p class="timestamp">Last scan: {scan_time}</p>

        <div class="cards">
            <div class="card">
                <div class="card-title">MARKET REGIME</div>
                <div class="card-value {regime_type.lower()}">{regime_type}</div>
                <div class="card-sub">{regime_conf:.0f}% confidence</div>
            </div>
            <div class="card">
                <div class="card-title">VIX LEVEL</div>
                <div class="card-value">{vix:.1f}</div>
                <div class="card-sub">Volatility index</div>
            </div>
            <div class="card">
                <div class="card-title">SIGNALS PASSED</div>
                <div class="card-value">{passed}</div>
                <div class="card-sub">of {total_scanned} scanned</div>
            </div>
            <div class="card">
                <div class="card-title">FILTERED OUT</div>
                <div class="card-value">{filtered}</div>
                <div class="card-sub">below threshold</div>
            </div>
        </div>

        <div class="recommendation">
            <strong>Strategy:</strong> {recommendation}
        </div>

        <h2>TOP SIGNALS</h2>
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Tier</th>
                    <th>Price</th>
                    <th>Shares</th>
                    <th>Strength</th>
                    <th>E[R]</th>
                    <th>Profit Target</th>
                    <th>Stop Loss</th>
                </tr>
            </thead>
            <tbody>
"""

    for sig in signals:
        ticker = sig.get('ticker', 'N/A')
        tier = sig.get('tier', 'N/A').lower()
        price = sig.get('current_price', 0)
        shares = sig.get('suggested_shares', 0)
        strength = sig.get('adjusted_strength', 0)
        er = sig.get('expected_return', 0)
        pt = sig.get('profit_target', 2.0)
        sl = sig.get('stop_loss', -2.0)

        er_class = 'positive' if er > 0 else 'negative'

        html += f"""
                <tr>
                    <td><strong>{ticker}</strong></td>
                    <td><span class="tier {tier}">{tier.upper()}</span></td>
                    <td>${price:.2f}</td>
                    <td>{shares}</td>
                    <td>{strength:.1f}</td>
                    <td class="{er_class}">{er:+.2f}%</td>
                    <td class="positive">+{pt:.1f}%</td>
                    <td class="negative">{sl:.1f}%</td>
                </tr>
"""

    html += """
            </tbody>
        </table>

        <footer>
            <p>Auto-refreshes every 5 minutes | Powered by Proteus Trading System</p>
        </footer>
    </div>
</body>
</html>
"""

    # Save dashboard
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"[DASHBOARD] Generated: {output_path}")
    return output_path


def update_dashboard():
    """Quick function to update dashboard from latest scan."""
    return generate_dashboard()


if __name__ == "__main__":
    generate_dashboard()
    print("Dashboard generated successfully!")

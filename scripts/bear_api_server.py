#!/usr/bin/env python
"""
Bear Detection API Server

Simple HTTP API for querying bear market detection status.
Useful for external tools, dashboards, and integrations.

Endpoints:
  GET /status       - Current bear score and alert level
  GET /signal       - Full signal with all indicators
  GET /trend        - 24-hour trend data
  GET /health       - Health check

Usage:
  python scripts/bear_api_server.py                    # Start on port 8080
  python scripts/bear_api_server.py --port 5000        # Custom port
  python scripts/bear_api_server.py --host 0.0.0.0     # Listen on all interfaces

Example:
  curl http://localhost:8080/status
  curl http://localhost:8080/signal
"""

import os
import sys
import json
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))


class BearAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Bear Detection API."""

    def _set_headers(self, status=200, content_type='application/json'):
        """Set response headers."""
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _send_json(self, data, status=200):
        """Send JSON response."""
        self._set_headers(status)
        response = json.dumps(data, indent=2, default=str)
        self.wfile.write(response.encode('utf-8'))

    def _send_error(self, message, status=500):
        """Send error response."""
        self._send_json({'error': message, 'status': status}, status)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self._set_headers(200)

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip('/')

        try:
            if path == '' or path == '/':
                self._handle_root()
            elif path == '/status':
                self._handle_status()
            elif path == '/signal':
                self._handle_signal()
            elif path == '/trend':
                self._handle_trend(parse_qs(parsed.query))
            elif path == '/health':
                self._handle_health()
            elif path == '/metrics':
                self._handle_metrics()
            elif path == '/dashboard':
                self._handle_dashboard()
            else:
                self._send_error('Not found', 404)
        except Exception as e:
            self._send_error(str(e), 500)

    def _handle_root(self):
        """API documentation."""
        docs = {
            'name': 'Proteus Bear Detection API',
            'version': '1.0',
            'endpoints': {
                '/dashboard': 'HTML dashboard with visual bear status',
                '/status': 'Current bear score and alert level (lightweight)',
                '/signal': 'Full signal with all indicators',
                '/trend': 'Historical trend data (optional: ?hours=24)',
                '/metrics': 'Prometheus-compatible metrics',
                '/health': 'Health check'
            },
            'timestamp': datetime.now().isoformat()
        }
        self._send_json(docs)

    def _handle_status(self):
        """Quick status check - lightweight."""
        from analysis.fast_bear_detector import FastBearDetector

        detector = FastBearDetector()
        signal = detector.detect()

        status = {
            'bear_score': signal.bear_score,
            'alert_level': signal.alert_level,
            'confidence': signal.confidence,
            'triggers_count': len(signal.triggers),
            'timestamp': signal.timestamp
        }
        self._send_json(status)

    def _handle_signal(self):
        """Full signal with all indicators."""
        from analysis.fast_bear_detector import FastBearDetector

        detector = FastBearDetector()
        signal = detector.detect()

        full_signal = {
            'bear_score': signal.bear_score,
            'alert_level': signal.alert_level,
            'confidence': signal.confidence,
            'timestamp': signal.timestamp,
            'indicators': {
                'vix_level': signal.vix_level,
                'vix_spike_pct': signal.vix_spike_pct,
                'spy_roc_3d': signal.spy_roc_3d,
                'market_breadth_pct': signal.market_breadth_pct,
                'sectors_declining': signal.sectors_declining,
                'sectors_total': signal.sectors_total,
                'volume_confirmation': signal.volume_confirmation,
                'yield_curve_spread': signal.yield_curve_spread,
                'credit_spread_change': signal.credit_spread_change,
                'momentum_divergence': signal.momentum_divergence
            },
            'triggers': signal.triggers,
            'recommendation': signal.recommendation
        }
        self._send_json(full_signal)

    def _handle_trend(self, query_params):
        """Historical trend data."""
        from analysis.fast_bear_detector import get_bear_score_trend

        hours = int(query_params.get('hours', [24])[0])
        hours = min(max(hours, 1), 168)  # Clamp between 1 and 168 hours

        trend = get_bear_score_trend(hours=hours)
        self._send_json(trend)

    def _handle_health(self):
        """Health check endpoint."""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'fast_bear_detector': 'ok',
                'data_sources': 'ok'
            }
        }

        # Quick check that detector works
        try:
            from analysis.fast_bear_detector import FastBearDetector
            detector = FastBearDetector()
            signal = detector.detect()
            if signal.bear_score < 0 or signal.bear_score > 100:
                health['components']['fast_bear_detector'] = 'error'
                health['status'] = 'degraded'
        except Exception as e:
            health['components']['fast_bear_detector'] = f'error: {str(e)}'
            health['status'] = 'unhealthy'

        status = 200 if health['status'] == 'healthy' else 503
        self._send_json(health, status)

    def _handle_metrics(self):
        """Prometheus-compatible metrics."""
        from analysis.fast_bear_detector import FastBearDetector

        detector = FastBearDetector()
        signal = detector.detect()

        # Prometheus format
        metrics = []
        metrics.append('# HELP proteus_bear_score Current bear score (0-100)')
        metrics.append('# TYPE proteus_bear_score gauge')
        metrics.append(f'proteus_bear_score {signal.bear_score}')

        metrics.append('# HELP proteus_bear_alert_level Current alert level (0=NORMAL, 1=WATCH, 2=WARNING, 3=CRITICAL)')
        metrics.append('# TYPE proteus_bear_alert_level gauge')
        level_map = {'NORMAL': 0, 'WATCH': 1, 'WARNING': 2, 'CRITICAL': 3}
        metrics.append(f'proteus_bear_alert_level {level_map.get(signal.alert_level, 0)}')

        metrics.append('# HELP proteus_vix_level Current VIX level')
        metrics.append('# TYPE proteus_vix_level gauge')
        metrics.append(f'proteus_vix_level {signal.vix_level}')

        metrics.append('# HELP proteus_spy_roc_3d SPY 3-day rate of change percentage')
        metrics.append('# TYPE proteus_spy_roc_3d gauge')
        metrics.append(f'proteus_spy_roc_3d {signal.spy_roc_3d}')

        metrics.append('# HELP proteus_market_breadth_pct Market breadth percentage')
        metrics.append('# TYPE proteus_market_breadth_pct gauge')
        metrics.append(f'proteus_market_breadth_pct {signal.market_breadth_pct}')

        metrics.append('# HELP proteus_yield_curve_spread Yield curve spread (10Y-2Y)')
        metrics.append('# TYPE proteus_yield_curve_spread gauge')
        metrics.append(f'proteus_yield_curve_spread {signal.yield_curve_spread}')

        metrics.append('# HELP proteus_triggers_count Number of active triggers')
        metrics.append('# TYPE proteus_triggers_count gauge')
        metrics.append(f'proteus_triggers_count {len(signal.triggers)}')

        self._set_headers(200, 'text/plain; charset=utf-8')
        self.wfile.write('\n'.join(metrics).encode('utf-8'))

    def _handle_dashboard(self):
        """Serve HTML dashboard for bear status."""
        from analysis.fast_bear_detector import FastBearDetector, get_bear_score_trend

        detector = FastBearDetector()
        signal = detector.detect()
        trend = get_bear_score_trend(hours=24)

        # Color based on alert level
        colors = {
            'CRITICAL': {'bg': '#dc2626', 'text': '#fff', 'light': '#fef2f2'},
            'WARNING': {'bg': '#f59e0b', 'text': '#fff', 'light': '#fffbeb'},
            'WATCH': {'bg': '#3b82f6', 'text': '#fff', 'light': '#eff6ff'},
            'NORMAL': {'bg': '#10b981', 'text': '#fff', 'light': '#f0fdf4'}
        }
        color = colors.get(signal.alert_level, colors['NORMAL'])

        # Build triggers HTML
        triggers_html = ""
        if signal.triggers:
            triggers_html = "<ul class='triggers'>"
            for t in signal.triggers:
                triggers_html += f"<li>{t}</li>"
            triggers_html += "</ul>"
        else:
            triggers_html = "<p class='no-triggers'>No active triggers</p>"

        # Build trend info
        trend_html = ""
        if 'scores' in trend and len(trend['scores']) > 1:
            scores = trend['scores']
            change = scores[-1] - scores[0]
            trend_dir = "rising" if change > 5 else "falling" if change < -5 else "stable"
            trend_color = "#dc2626" if change > 5 else "#10b981" if change < -5 else "#666"
            trend_html = f"""
            <div class="trend-info">
                <span>24h: {trend.get('min', 0):.0f} - {trend.get('max', 0):.0f}</span>
                <span style="color: {trend_color}">({change:+.1f} {trend_dir})</span>
            </div>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="60">
    <title>Bear Detection Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .header {{
            text-align: center;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .header h1 {{ font-size: 1.5em; color: #94a3b8; }}
        .header .timestamp {{ color: #64748b; font-size: 0.9em; margin-top: 5px; }}

        .score-card {{
            background: {color['bg']};
            border-radius: 16px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .score-card .level {{ font-size: 1.2em; opacity: 0.9; margin-bottom: 10px; }}
        .score-card .score {{ font-size: 4em; font-weight: bold; }}
        .score-card .confidence {{ opacity: 0.8; margin-top: 10px; }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric {{
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .metric .label {{ color: #94a3b8; font-size: 0.85em; margin-bottom: 8px; }}
        .metric .value {{ font-size: 1.5em; font-weight: 600; }}
        .metric .status {{ font-size: 0.8em; margin-top: 5px; }}
        .status-ok {{ color: #10b981; }}
        .status-warn {{ color: #f59e0b; }}
        .status-bad {{ color: #dc2626; }}

        .triggers-card {{
            background: {color['light']};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            color: #1a1a1a;
        }}
        .triggers-card h3 {{ color: {color['bg']}; margin-bottom: 15px; }}
        .triggers {{ list-style: none; }}
        .triggers li {{ padding: 8px 0; border-bottom: 1px solid rgba(0,0,0,0.1); }}
        .triggers li:last-child {{ border-bottom: none; }}
        .no-triggers {{ color: #666; font-style: italic; }}

        .recommendation {{
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
            border-left: 4px solid {color['bg']};
        }}
        .recommendation h3 {{ color: #94a3b8; margin-bottom: 10px; }}
        .recommendation p {{ line-height: 1.6; }}

        .trend-info {{
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 10px;
            font-size: 0.9em;
            opacity: 0.9;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: #64748b;
            font-size: 0.85em;
        }}
        .footer a {{ color: #3b82f6; text-decoration: none; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Proteus Bear Detection</h1>
            <div class="timestamp">{signal.timestamp}</div>
        </div>

        <div class="score-card">
            <div class="level">{signal.alert_level}</div>
            <div class="score">{signal.bear_score:.0f}<span style="font-size: 0.4em; opacity: 0.8">/100</span></div>
            <div class="confidence">Confidence: {signal.confidence * 100:.0f}%</div>
            {trend_html}
        </div>

        <div class="metrics-grid">
            <div class="metric">
                <div class="label">VIX Level</div>
                <div class="value">{signal.vix_level:.1f}</div>
                <div class="status {'status-bad' if signal.vix_level > 30 else 'status-warn' if signal.vix_level > 25 else 'status-ok'}">
                    {'HIGH' if signal.vix_level > 30 else 'ELEVATED' if signal.vix_level > 25 else 'NORMAL'}
                </div>
            </div>
            <div class="metric">
                <div class="label">SPY 3-Day</div>
                <div class="value">{signal.spy_roc_3d:+.2f}%</div>
                <div class="status {'status-bad' if signal.spy_roc_3d < -3 else 'status-warn' if signal.spy_roc_3d < -2 else 'status-ok'}">
                    {'FALLING' if signal.spy_roc_3d < -3 else 'WEAK' if signal.spy_roc_3d < -2 else 'OK'}
                </div>
            </div>
            <div class="metric">
                <div class="label">Market Breadth</div>
                <div class="value">{signal.market_breadth_pct:.0f}%</div>
                <div class="status {'status-bad' if signal.market_breadth_pct < 30 else 'status-warn' if signal.market_breadth_pct < 40 else 'status-ok'}">
                    {'WEAK' if signal.market_breadth_pct < 30 else 'MIXED' if signal.market_breadth_pct < 40 else 'HEALTHY'}
                </div>
            </div>
            <div class="metric">
                <div class="label">Yield Curve</div>
                <div class="value">{signal.yield_curve_spread:+.2f}%</div>
                <div class="status {'status-bad' if signal.yield_curve_spread <= 0 else 'status-warn' if signal.yield_curve_spread < 0.25 else 'status-ok'}">
                    {'INVERTED' if signal.yield_curve_spread <= 0 else 'FLAT' if signal.yield_curve_spread < 0.25 else 'NORMAL'}
                </div>
            </div>
            <div class="metric">
                <div class="label">Sectors Down</div>
                <div class="value">{signal.sectors_declining}/{signal.sectors_total}</div>
                <div class="status {'status-bad' if signal.sectors_declining >= 8 else 'status-warn' if signal.sectors_declining >= 6 else 'status-ok'}">
                    {'BROAD DECLINE' if signal.sectors_declining >= 8 else 'MIXED' if signal.sectors_declining >= 6 else 'OK'}
                </div>
            </div>
            <div class="metric">
                <div class="label">Credit Spread</div>
                <div class="value">{signal.credit_spread_change:+.1f}%</div>
                <div class="status {'status-bad' if signal.credit_spread_change >= 10 else 'status-warn' if signal.credit_spread_change >= 5 else 'status-ok'}">
                    {'STRESSED' if signal.credit_spread_change >= 10 else 'WIDENING' if signal.credit_spread_change >= 5 else 'NORMAL'}
                </div>
            </div>
        </div>

        <div class="triggers-card">
            <h3>Active Triggers</h3>
            {triggers_html}
        </div>

        <div class="recommendation">
            <h3>Recommendation</h3>
            <p>{signal.recommendation}</p>
        </div>

        <div class="footer">
            <p>Auto-refreshes every 60 seconds</p>
            <p><a href="/status">JSON API</a> | <a href="/metrics">Prometheus</a> | <a href="/">API Docs</a></p>
        </div>
    </div>
</body>
</html>"""

        self._set_headers(200, 'text/html; charset=utf-8')
        self.wfile.write(html.encode('utf-8'))

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}")


def run_server(host='127.0.0.1', port=8080):
    """Run the API server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, BearAPIHandler)

    print("=" * 60)
    print("PROTEUS BEAR DETECTION API")
    print("=" * 60)
    print(f"Server running at http://{host}:{port}")
    print()
    print("Endpoints:")
    print(f"  GET http://{host}:{port}/dashboard - HTML dashboard")
    print(f"  GET http://{host}:{port}/status    - Quick status (JSON)")
    print(f"  GET http://{host}:{port}/signal    - Full signal (JSON)")
    print(f"  GET http://{host}:{port}/trend     - 24h trend (JSON)")
    print(f"  GET http://{host}:{port}/metrics   - Prometheus metrics")
    print(f"  GET http://{host}:{port}/health    - Health check")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description='Bear Detection API Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/bear_api_server.py                    # Default: localhost:8080
  python scripts/bear_api_server.py --port 5000        # Custom port
  python scripts/bear_api_server.py --host 0.0.0.0     # All interfaces

API Usage:
  curl http://localhost:8080/status
  curl http://localhost:8080/signal
  curl http://localhost:8080/trend?hours=48
        """
    )

    parser.add_argument('--host', default='127.0.0.1',
                       help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', '-p', type=int, default=8080,
                       help='Port to listen on (default: 8080)')

    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == '__main__':
    main()

"""
Automated Price Alert System
=============================

Monitors stock prices and generates alerts when key levels are reached.

Features:
- Entry zone alerts (time to buy)
- Stop loss alerts (protect capital)
- Profit target alerts (take gains)
- Valuation alerts (extreme over/undervaluation)
- Technical alerts (RSI oversold/overbought)

Usage:
    python price_alert_system.py           # Run once
    python price_alert_system.py --watch   # Continuous monitoring (every 5 min)

Can be scheduled to run daily via Task Scheduler (Windows) or cron (Mac/Linux)
"""

import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import os
import time
import argparse


class PriceAlertSystem:
    """Automated price monitoring and alert generation."""

    def __init__(self):
        self.alerts_file = 'results/price_alerts_history.json'

        # Alert configuration for each stock
        self.alert_config = {
            'MU': {
                'name': 'Micron Technology',
                'action': 'STRONG BUY',
                'entry_low': 230,
                'entry_high': 245,
                'stop_loss': 220,
                'target_1': 262,
                'target_2': 300,
                'target_3': 375,  # DCF fair value
                'dcf_fair_value': 375.20,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
            },
            'APP': {
                'name': 'AppLovin',
                'action': 'WAIT',
                'entry_low': 500,
                'entry_high': 550,
                'stop_loss': 557,
                'target_1': 600,
                'target_2': 700,
                'target_3': 750,
                'dcf_fair_value': 326.18,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
            },
            'PLTR': {
                'name': 'Palantir',
                'action': 'TRIM',
                'entry_low': 120,
                'entry_high': 140,
                'stop_loss': 155,
                'target_1': 180,  # Bounce target to trim
                'target_2': 200,
                'target_3': 220,
                'dcf_fair_value': 36.53,
                'rsi_oversold': 25,  # Lower threshold (more extreme)
                'rsi_overbought': 75,
            },
            'ALAB': {
                'name': 'Astera Labs',
                'action': 'WAIT',
                'entry_low': 100,
                'entry_high': 120,
                'stop_loss': 150,
                'target_1': 140,
                'target_2': 180,
                'target_3': 210,
                'dcf_fair_value': 59.06,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
            },
        }

        # Load alert history
        self.alert_history = self.load_alert_history()

    def load_alert_history(self) -> Dict:
        """Load previous alerts to avoid duplicates."""
        if os.path.exists(self.alerts_file):
            try:
                with open(self.alerts_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_alert(self, ticker: str, alert_type: str, message: str, price: float):
        """Save alert to history."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if ticker not in self.alert_history:
            self.alert_history[ticker] = []

        self.alert_history[ticker].append({
            'timestamp': timestamp,
            'type': alert_type,
            'message': message,
            'price': price,
        })

        # Keep only last 100 alerts per ticker
        self.alert_history[ticker] = self.alert_history[ticker][-100:]

        # Save to file
        os.makedirs('results', exist_ok=True)
        with open(self.alerts_file, 'w') as f:
            json.dump(self.alert_history, f, indent=2)

    def get_current_price(self, ticker: str) -> Tuple[float, Dict]:
        """Fetch current price and basic info."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            prev_close = info.get('previousClose', current_price)
            day_change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0

            return current_price, {
                'price': current_price,
                'day_change': day_change_pct,
                'volume': info.get('volume', 0),
            }
        except Exception as e:
            return 0, {'error': str(e)}

    def calculate_rsi(self, ticker: str, period: int = 14) -> float:
        """Calculate RSI for technical alerts."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1mo')

            if len(hist) < period:
                return 50.0  # Neutral default

            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi.iloc[-1]
        except:
            return 50.0

    def check_alerts(self, ticker: str) -> List[Dict]:
        """Check all alert conditions for a ticker."""
        config = self.alert_config.get(ticker)
        if not config:
            return []

        current_price, info = self.get_current_price(ticker)
        if current_price == 0:
            return []

        alerts = []

        # 1. Entry Zone Alerts
        if config['entry_low'] <= current_price <= config['entry_high']:
            alerts.append({
                'type': 'ENTRY',
                'priority': 'HIGH',
                'message': f"[BUY ZONE] {ticker} at ${current_price:.2f} is in entry range ${config['entry_low']}-${config['entry_high']}",
                'action': f"Consider buying - target allocation based on {config['action']} recommendation",
            })

        # 2. Entry Below Zone (even better)
        if current_price < config['entry_low']:
            discount = ((config['entry_low'] - current_price) / config['entry_low']) * 100
            alerts.append({
                'type': 'ENTRY_BELOW',
                'priority': 'CRITICAL',
                'message': f"[STRONG BUY] {ticker} at ${current_price:.2f} is BELOW entry zone (${config['entry_low']}) by {discount:.1f}%",
                'action': "Excellent entry price - buy aggressively if this is a target stock",
            })

        # 3. Stop Loss Alerts
        if current_price <= config['stop_loss']:
            alerts.append({
                'type': 'STOP_LOSS',
                'priority': 'CRITICAL',
                'message': f"[STOP LOSS HIT] {ticker} at ${current_price:.2f} hit stop loss ${config['stop_loss']}",
                'action': "SELL immediately to protect capital",
            })
        elif current_price <= config['stop_loss'] * 1.03:  # Within 3% of stop
            alerts.append({
                'type': 'STOP_LOSS_WARNING',
                'priority': 'HIGH',
                'message': f"[WARNING] {ticker} at ${current_price:.2f} approaching stop loss ${config['stop_loss']}",
                'action': "Monitor closely - consider tightening stop or reducing position",
            })

        # 4. Profit Target Alerts
        if current_price >= config['target_3']:
            alerts.append({
                'type': 'TARGET_3',
                'priority': 'HIGH',
                'message': f"[TARGET 3 HIT] {ticker} at ${current_price:.2f} reached ${config['target_3']} (DCF fair value!)",
                'action': "Take 50-75% profits - you hit fair value",
            })
        elif current_price >= config['target_2']:
            alerts.append({
                'type': 'TARGET_2',
                'priority': 'MEDIUM',
                'message': f"[TARGET 2 HIT] {ticker} at ${current_price:.2f} reached ${config['target_2']}",
                'action': "Take 25-50% profits - let rest run",
            })
        elif current_price >= config['target_1']:
            alerts.append({
                'type': 'TARGET_1',
                'priority': 'MEDIUM',
                'message': f"[TARGET 1 HIT] {ticker} at ${current_price:.2f} reached ${config['target_1']}",
                'action': "Take 25% profits or tighten stops",
            })

        # 5. Valuation Alerts
        dcf_gap = ((config['dcf_fair_value'] - current_price) / current_price) * 100
        if dcf_gap > 50:
            alerts.append({
                'type': 'DEEP_VALUE',
                'priority': 'HIGH',
                'message': f"[DEEP VALUE] {ticker} at ${current_price:.2f} is {dcf_gap:+.1f}% below DCF fair value ${config['dcf_fair_value']:.2f}",
                'action': "Significant undervaluation - strong buy candidate",
            })
        elif dcf_gap < -50:
            alerts.append({
                'type': 'EXTREME_OVERVALUATION',
                'priority': 'MEDIUM',
                'message': f"[OVERVALUED] {ticker} at ${current_price:.2f} is {abs(dcf_gap):.1f}% above DCF fair value ${config['dcf_fair_value']:.2f}",
                'action': "Extreme overvaluation - avoid or trim position",
            })

        # 6. Technical Alerts (RSI)
        rsi = self.calculate_rsi(ticker)
        if rsi < config['rsi_oversold']:
            alerts.append({
                'type': 'RSI_OVERSOLD',
                'priority': 'MEDIUM',
                'message': f"[OVERSOLD] {ticker} RSI at {rsi:.1f} (below {config['rsi_oversold']}) - potential bounce",
                'action': "Watch for reversal - could be buying opportunity",
            })
        elif rsi > config['rsi_overbought']:
            alerts.append({
                'type': 'RSI_OVERBOUGHT',
                'priority': 'MEDIUM',
                'message': f"[OVERBOUGHT] {ticker} RSI at {rsi:.1f} (above {config['rsi_overbought']}) - potential pullback",
                'action': "Consider taking profits or tightening stops",
            })

        # 7. Special Case: PLTR Bounce Alert (RSI oversold + specific price)
        if ticker == 'PLTR' and rsi < 30 and current_price < 170:
            alerts.append({
                'type': 'PLTR_TRIM_OPPORTUNITY',
                'priority': 'HIGH',
                'message': f"[PLTR TRIM SETUP] RSI {rsi:.1f} oversold at ${current_price:.2f} - bounce to $180-190 likely",
                'action': "PREPARE TO TRIM position on bounce to $180 (target allocation 10%)",
            })

        # Add price and timestamp to each alert
        for alert in alerts:
            alert['ticker'] = ticker
            alert['price'] = current_price
            alert['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return alerts

    def run_alert_check(self, display: bool = True) -> Dict:
        """Run alert check for all tickers."""
        all_alerts = {}

        for ticker in self.alert_config.keys():
            alerts = self.check_alerts(ticker)
            if alerts:
                all_alerts[ticker] = alerts

                # Save alerts to history
                for alert in alerts:
                    self.save_alert(ticker, alert['type'], alert['message'], alert['price'])

        if display:
            self.display_alerts(all_alerts)

        return all_alerts

    def display_alerts(self, alerts: Dict):
        """Display alerts in formatted output."""
        print("\n" + "="*70)
        print("PRICE ALERT SYSTEM")
        print(f"Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        if not alerts:
            print("\n No alerts at this time. All positions within normal ranges.")
            print("\nCurrent Status:")
            for ticker, config in self.alert_config.items():
                price, info = self.get_current_price(ticker)
                day_change = info.get('day_change', 0)

                change_str = f"[+{day_change:.2f}%]" if day_change > 0 else f"[{day_change:.2f}%]"
                print(f"  {ticker:6} ${price:8.2f} {change_str:12} - {config['action']}")

            return

        # Group alerts by priority
        critical_alerts = []
        high_alerts = []
        medium_alerts = []

        for ticker, ticker_alerts in alerts.items():
            for alert in ticker_alerts:
                if alert['priority'] == 'CRITICAL':
                    critical_alerts.append(alert)
                elif alert['priority'] == 'HIGH':
                    high_alerts.append(alert)
                else:
                    medium_alerts.append(alert)

        # Display CRITICAL alerts first
        if critical_alerts:
            print("\n" + "!"*70)
            print("CRITICAL ALERTS - IMMEDIATE ACTION REQUIRED")
            print("!"*70)
            for alert in critical_alerts:
                print(f"\n{alert['ticker']} - {alert['message']}")
                print(f"   ACTION: {alert['action']}")

        # Display HIGH priority alerts
        if high_alerts:
            print("\n" + "="*70)
            print("HIGH PRIORITY ALERTS")
            print("="*70)
            for alert in high_alerts:
                print(f"\n{alert['ticker']} - {alert['message']}")
                print(f"   ACTION: {alert['action']}")

        # Display MEDIUM priority alerts
        if medium_alerts:
            print("\n" + "-"*70)
            print("INFORMATIONAL ALERTS")
            print("-"*70)
            for alert in medium_alerts:
                print(f"\n{alert['ticker']} - {alert['message']}")
                print(f"   ACTION: {alert['action']}")

        print("\n" + "="*70)
        print(f"Total Alerts: {len(critical_alerts) + len(high_alerts) + len(medium_alerts)}")
        print(f"  Critical: {len(critical_alerts)}")
        print(f"  High:     {len(high_alerts)}")
        print(f"  Medium:   {len(medium_alerts)}")
        print("="*70 + "\n")

    def watch_mode(self, interval_minutes: int = 5):
        """Continuous monitoring mode."""
        print("\n" + "="*70)
        print("PRICE ALERT SYSTEM - WATCH MODE")
        print(f"Checking prices every {interval_minutes} minutes")
        print("Press Ctrl+C to stop")
        print("="*70 + "\n")

        try:
            while True:
                self.run_alert_check(display=True)

                print(f"\nNext check in {interval_minutes} minutes...")
                print(f"(Ctrl+C to stop)\n")

                time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\n\nAlert system stopped.")

    def get_alert_summary(self, days: int = 7) -> Dict:
        """Get summary of alerts over past N days."""
        cutoff_date = datetime.now() - timedelta(days=days)

        summary = {}
        for ticker, alerts in self.alert_history.items():
            recent_alerts = []
            for alert in alerts:
                alert_date = datetime.strptime(alert['timestamp'], '%Y-%m-%d %H:%M:%S')
                if alert_date >= cutoff_date:
                    recent_alerts.append(alert)

            if recent_alerts:
                summary[ticker] = {
                    'count': len(recent_alerts),
                    'types': {},
                    'alerts': recent_alerts,
                }

                # Count by type
                for alert in recent_alerts:
                    alert_type = alert['type']
                    summary[ticker]['types'][alert_type] = summary[ticker]['types'].get(alert_type, 0) + 1

        return summary


def main():
    """Main function with command-line arguments."""
    parser = argparse.ArgumentParser(description='Proteus Price Alert System')
    parser.add_argument('--watch', action='store_true', help='Continuous monitoring mode')
    parser.add_argument('--interval', type=int, default=5, help='Check interval in minutes (watch mode)')
    parser.add_argument('--summary', type=int, help='Show alert summary for past N days')

    args = parser.parse_args()

    alert_system = PriceAlertSystem()

    if args.summary:
        print(f"\nAlert Summary - Past {args.summary} Days")
        print("="*70)
        summary = alert_system.get_alert_summary(args.summary)

        if not summary:
            print("No alerts in this period.")
        else:
            for ticker, data in summary.items():
                print(f"\n{ticker}: {data['count']} alerts")
                for alert_type, count in data['types'].items():
                    print(f"  {alert_type}: {count}")

    elif args.watch:
        alert_system.watch_mode(args.interval)

    else:
        # Single run
        alert_system.run_alert_check(display=True)

        print("\nTIP: Schedule this script to run daily:")
        print("  Windows: Use Task Scheduler")
        print("  Mac/Linux: Add to crontab (e.g., '0 9 * * * python price_alert_system.py')")
        print("\nOr run in watch mode:")
        print("  python price_alert_system.py --watch")


if __name__ == "__main__":
    main()

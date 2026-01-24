"""
Send today's stock recommendation email
"""
import sys
import json
from datetime import datetime

sys.path.insert(0, 'src')

from notifications.sendgrid_notifier import SendGridNotifier

# Load latest scan
with open('data/smart_scans/latest_scan.json') as f:
    scan = json.load(f)

# Initialize notifier
notifier = SendGridNotifier()

# Get best signal
signals = scan.get('signals', [])
if not signals:
    print("No signals to send")
    sys.exit(0)

# Best signal is first (highest adjusted strength)
best = signals[0]
second = signals[1] if len(signals) > 1 else None

# Create email
subject = f"PROTEUS BUY SIGNAL: {best['ticker']} @ ${best['price']:.2f} - Monday Jan 6, 2026"

html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto;">

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 10px; margin-bottom: 25px;">
    <h1 style="margin: 0; font-size: 1.6em;">PROTEUS TRADING SIGNAL</h1>
    <p style="margin: 10px 0 0 0; opacity: 0.9;">Monday, January 6, 2026</p>
</div>

<div style="background: #f0fdf4; padding: 20px; border-left: 5px solid #10b981; border-radius: 5px; margin-bottom: 20px;">
    <h2 style="color: #10b981; margin: 0 0 10px 0;">TOP PICK: BUY {best['ticker']}</h2>
    <p style="font-size: 1.3em; margin: 0;"><strong>Signal Strength: {best['adjusted_strength']}/100</strong></p>
</div>

<table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
    <tr style="background: #f8fafc;">
        <td style="padding: 12px; border: 1px solid #ddd;"><strong>Entry Price</strong></td>
        <td style="padding: 12px; border: 1px solid #ddd;">${best['price']:.2f}</td>
    </tr>
    <tr>
        <td style="padding: 12px; border: 1px solid #ddd;"><strong>Stop Loss</strong></td>
        <td style="padding: 12px; border: 1px solid #ddd; color: #dc2626;">${best['price'] * (1 + best['stop_loss']/100):.2f} ({best['stop_loss']}%)</td>
    </tr>
    <tr style="background: #f8fafc;">
        <td style="padding: 12px; border: 1px solid #ddd;"><strong>Target 1</strong></td>
        <td style="padding: 12px; border: 1px solid #ddd; color: #10b981;">${best['price'] * (1 + best['profit_target_1']/100):.2f} (+{best['profit_target_1']}%)</td>
    </tr>
    <tr>
        <td style="padding: 12px; border: 1px solid #ddd;"><strong>Target 2</strong></td>
        <td style="padding: 12px; border: 1px solid #ddd; color: #10b981;">${best['price'] * (1 + best['profit_target_2']/100):.2f} (+{best['profit_target_2']}%)</td>
    </tr>
    <tr style="background: #f8fafc;">
        <td style="padding: 12px; border: 1px solid #ddd;"><strong>Position Size</strong></td>
        <td style="padding: 12px; border: 1px solid #ddd;">{best['shares']} shares (${best['dollar_size']:.0f})</td>
    </tr>
    <tr>
        <td style="padding: 12px; border: 1px solid #ddd;"><strong>Risk</strong></td>
        <td style="padding: 12px; border: 1px solid #ddd;">${best['risk_dollars']:.0f}</td>
    </tr>
    <tr style="background: #f8fafc;">
        <td style="padding: 12px; border: 1px solid #ddd;"><strong>Max Hold</strong></td>
        <td style="padding: 12px; border: 1px solid #ddd;">{best['max_hold_days']} days</td>
    </tr>
</table>

<div style="background: #eff6ff; padding: 15px; border-radius: 5px; margin: 20px 0;">
    <h3 style="color: #2563eb; margin: 0 0 10px 0;">Why {best['ticker']}?</h3>
    <ul style="margin: 0; padding-left: 20px;">
        <li><strong>Sector:</strong> {best['sector']}</li>
        <li><strong>Tier:</strong> {best['tier'].upper()}</li>
        <li><strong>Boosts Applied:</strong> {', '.join(best['boosts_applied'])}</li>
        <li><strong>Regime:</strong> {scan['regime'].upper()}</li>
    </ul>
</div>
"""

if second:
    html_body += f"""
<div style="background: #fef3c7; padding: 15px; border-left: 5px solid #f59e0b; border-radius: 5px; margin: 20px 0;">
    <h3 style="color: #d97706; margin: 0 0 10px 0;">SECONDARY PICK: {second['ticker']}</h3>
    <p style="margin: 0;">Signal: {second['adjusted_strength']}/100 | Entry: ${second['price']:.2f} | {second['shares']} shares</p>
    <p style="margin: 5px 0 0 0; font-size: 0.9em;">Boosts: {', '.join(second['boosts_applied'])}</p>
</div>
"""

html_body += f"""
<div style="background: #f8fafc; padding: 20px; border-radius: 5px; margin: 20px 0;">
    <h3 style="margin: 0 0 15px 0;">Execution Notes</h3>
    <ol style="margin: 0; padding-left: 20px;">
        <li><strong>Wait 5-10 minutes after open</strong> - Let opening volatility settle</li>
        <li><strong>Verify price</strong> - Ensure no significant overnight gap</li>
        <li><strong>Set stops immediately</strong> - Don't hold without protection</li>
        <li><strong>Take partial profit at Target 1</strong> - Sell 50% at +1.5%</li>
        <li><strong>Trail remainder</strong> - Use 1% trailing stop after Target 1</li>
    </ol>
</div>

<div style="background: #1a1a1a; color: white; padding: 20px; border-radius: 5px; margin: 20px 0;">
    <h3 style="margin: 0 0 10px 0; color: #10b981;">System Status</h3>
    <p style="margin: 5px 0;"><strong>Modifiers Active:</strong> 113 (67 boosts + 46 penalties)</p>
    <p style="margin: 5px 0;"><strong>Backtest Performance:</strong> 60.8% win rate, 1.39 Sharpe</p>
    <p style="margin: 5px 0;"><strong>VIX:</strong> {scan['vix']}</p>
</div>

<div style="color: #666; font-size: 0.85em; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd;">
    <p><em>This is an automated recommendation from Proteus v2.0</em></p>
    <p><em>Always verify signals and manage risk appropriately</em></p>
    <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
</div>

</body>
</html>
"""

text_body = f"""
PROTEUS TRADING SIGNAL - Monday, January 6, 2026
================================================

TOP PICK: BUY {best['ticker']}
Signal Strength: {best['adjusted_strength']}/100

Entry: ${best['price']:.2f}
Stop: ${best['price'] * (1 + best['stop_loss']/100):.2f} ({best['stop_loss']}%)
Target 1: ${best['price'] * (1 + best['profit_target_1']/100):.2f} (+{best['profit_target_1']}%)
Target 2: ${best['price'] * (1 + best['profit_target_2']/100):.2f} (+{best['profit_target_2']}%)

Position: {best['shares']} shares (${best['dollar_size']:.0f})
Risk: ${best['risk_dollars']:.0f}
Max Hold: {best['max_hold_days']} days

Sector: {best['sector']}
Boosts: {', '.join(best['boosts_applied'])}

EXECUTION:
1. Wait 5-10 min after open
2. Verify no overnight gap
3. Set stops immediately
4. Take 50% at Target 1
5. Trail remainder with 1% stop

System: 113 Modifiers | 60.8% win rate | VIX: {scan['vix']}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# Send email
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Email, To, Content

    with open('email_config.json') as f:
        config = json.load(f)

    message = Mail(
        from_email=Email(config.get('sender_email', 'proteus@trading.local')),
        to_emails=To(config['recipient_email']),
        subject=subject,
        html_content=Content("text/html", html_body)
    )

    sg = SendGridAPIClient(config['sendgrid_api_key'])
    response = sg.send(message)

    print(f"Email sent successfully!")
    print(f"To: {config['recipient_email']}")
    print(f"Subject: {subject}")
    print(f"Status: {response.status_code}")

except Exception as e:
    print(f"Email failed: {e}")

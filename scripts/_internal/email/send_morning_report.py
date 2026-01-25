"""
Send Morning Deep Dive Report

Sends the overnight deep dive analysis results via email.
Run at 8am to deliver the summary.
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, 'src')

from notifications.sendgrid_notifier import SendGridNotifier

def load_overnight_summary():
    """Load the overnight summary."""
    summary_file = "data/deep_dives/overnight_summary_2025-12-19.json"
    with open(summary_file, 'r') as f:
        return json.load(f)

def load_todays_signals():
    """Check for today's signals."""
    try:
        from trading.signal_scanner import SignalScanner
        scanner = SignalScanner(lookback_days=90, min_signal_strength=50.0)
        signals = scanner.scan_all_stocks()
        return signals
    except Exception as e:
        return []

def generate_html_email(summary, signals):
    """Generate HTML email content."""

    # Signal section
    if signals:
        signal_html = """
        <div style="background-color: #d4edda; padding: 20px; border-left: 4px solid #28a745; margin: 20px 0;">
            <h3 style="color: #28a745; margin-top: 0;">🚨 TODAY'S TRADING SIGNALS</h3>
        """
        for sig in signals:
            ticker = sig.get('ticker', 'N/A')
            # Get conviction for this signal
            conv_file = f"data/deep_dives/{ticker}_2025-12-18.json"
            conv_score = "N/A"
            conv_tier = "N/A"
            if os.path.exists(conv_file):
                with open(conv_file, 'r') as f:
                    conv = json.load(f)
                    conv_score = conv.get('conviction_score', 'N/A')
                    conv_tier = conv.get('conviction_tier', 'N/A')

            signal_html += f"""
            <div style="background-color: white; padding: 15px; margin: 10px 0; border-radius: 5px;">
                <h4 style="margin: 0; color: #2c3e50;">{ticker}</h4>
                <p style="margin: 5px 0;">
                    <strong>Signal Strength:</strong> {sig.get('signal_strength', 0):.1f}/100<br>
                    <strong>Conviction:</strong> {conv_score}/100 ({conv_tier})<br>
                    <strong>Z-Score:</strong> {sig.get('z_score', 0):.2f}<br>
                    <strong>RSI:</strong> {sig.get('rsi', 0):.1f}<br>
                    <strong>Entry Price:</strong> ${sig.get('price', 0):.2f}
                </p>
            </div>
            """
        signal_html += "</div>"
    else:
        signal_html = """
        <div style="background-color: #fff3cd; padding: 20px; border-left: 4px solid #ffc107; margin: 20px 0;">
            <h3 style="color: #856404; margin-top: 0;">📊 NO SIGNALS TODAY</h3>
            <p>No stocks currently meet the mean reversion criteria. Continue monitoring.</p>
        </div>
        """

    # Top 10 conviction
    top10_html = ""
    for i, stock in enumerate(summary['top_10'], 1):
        color = "#28a745" if stock['tier'] == 'HIGH' else "#ffc107"
        top10_html += f"""
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{i}</td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>{stock['ticker']}</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{stock['company'][:30]}</td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd; color: {color};">{stock['conviction']}/100</td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{stock['sector']}</td>
        </tr>
        """

    # High conviction list
    high_conv = summary['tiers'].get('HIGH', [])[:15]

    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto;">

    <div style="background-color: #2c3e50; color: white; padding: 20px; text-align: center;">
        <h1 style="margin: 0;">PROTEUS MORNING REPORT</h1>
        <p style="margin: 5px 0;">Deep Dive Analysis Results - {datetime.now().strftime('%B %d, %Y')}</p>
    </div>

    {signal_html}

    <div style="background-color: #f8f9fa; padding: 20px; margin: 20px 0;">
        <h2 style="color: #2c3e50; margin-top: 0;">📈 OVERNIGHT DEEP DIVE SUMMARY</h2>

        <div style="display: flex; justify-content: space-around; text-align: center; margin: 20px 0;">
            <div style="background: white; padding: 15px 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; color: #28a745; font-size: 36px;">{summary['tier_counts']['HIGH']}</h3>
                <p style="margin: 5px 0; color: #666;">HIGH Conviction</p>
            </div>
            <div style="background: white; padding: 15px 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; color: #ffc107; font-size: 36px;">{summary['tier_counts']['MEDIUM']}</h3>
                <p style="margin: 5px 0; color: #666;">MEDIUM Conviction</p>
            </div>
            <div style="background: white; padding: 15px 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; color: #dc3545; font-size: 36px;">{summary['tier_counts']['LOW'] + summary['tier_counts']['AVOID']}</h3>
                <p style="margin: 5px 0; color: #666;">LOW/AVOID</p>
            </div>
        </div>

        <p><strong>Total Stocks Analyzed:</strong> {summary['total_stocks']}</p>
    </div>

    <div style="margin: 20px 0;">
        <h2 style="color: #2c3e50;">🏆 TOP 10 HIGHEST CONVICTION STOCKS</h2>
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background-color: #2c3e50; color: white;">
                    <th style="padding: 10px; text-align: left;">#</th>
                    <th style="padding: 10px; text-align: left;">Ticker</th>
                    <th style="padding: 10px; text-align: left;">Company</th>
                    <th style="padding: 10px; text-align: left;">Conviction</th>
                    <th style="padding: 10px; text-align: left;">Sector</th>
                </tr>
            </thead>
            <tbody>
                {top10_html}
            </tbody>
        </table>
    </div>

    <div style="background-color: #d4edda; padding: 20px; margin: 20px 0; border-radius: 8px;">
        <h3 style="color: #155724; margin-top: 0;">✅ HIGH CONVICTION STOCKS</h3>
        <p style="margin-bottom: 10px;">Trade with 1.2-1.5x position size when signals fire:</p>
        <p style="font-family: monospace; background: white; padding: 10px; border-radius: 4px;">
            {', '.join(high_conv)}
        </p>
    </div>

    <div style="background-color: #fff3cd; padding: 20px; margin: 20px 0; border-radius: 8px;">
        <h3 style="color: #856404; margin-top: 0;">⚠️ WATCH LIST (Lower Conviction)</h3>
        <p>APD scored lowest (35.3/100) - trade with reduced size or skip.</p>
    </div>

    <div style="background-color: #e9ecef; padding: 20px; margin: 20px 0;">
        <h3 style="color: #2c3e50; margin-top: 0;">📋 TRADING RULES</h3>
        <ol>
            <li><strong>HIGH conviction signal:</strong> Trade with 1.2-1.5x position</li>
            <li><strong>MEDIUM conviction signal:</strong> Trade with 1.0x position</li>
            <li><strong>LOW conviction signal:</strong> Trade with 0.5x or skip</li>
            <li><strong>AVOID conviction signal:</strong> Skip the trade</li>
        </ol>
    </div>

    <div style="background-color: #f8f9fa; padding: 15px; border-top: 3px solid #6c757d; margin-top: 30px; font-size: 0.9em; color: #6c757d;">
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Deep dive analysis ran overnight on all 54 Proteus stocks. Individual analysis files available in data/deep_dives/</p>
        <p style="margin-top: 10px;"><em>This report combines quantitative scoring with fundamental analysis to identify high-conviction trading opportunities.</em></p>
    </div>

    </body>
    </html>
    """

    return html


def generate_text_email(summary, signals):
    """Generate plain text email content."""

    signal_text = ""
    if signals:
        signal_text = "\n🚨 TODAY'S TRADING SIGNALS\n" + "="*40 + "\n"
        for sig in signals:
            ticker = sig.get('ticker', 'N/A')
            signal_text += f"""
{ticker}
  Signal Strength: {sig.get('signal_strength', 0):.1f}/100
  Z-Score: {sig.get('z_score', 0):.2f}
  RSI: {sig.get('rsi', 0):.1f}
  Entry Price: ${sig.get('price', 0):.2f}
"""
    else:
        signal_text = "\n📊 NO SIGNALS TODAY - Continue monitoring\n"

    top10_text = "\nTOP 10 HIGHEST CONVICTION:\n" + "-"*40 + "\n"
    for i, stock in enumerate(summary['top_10'], 1):
        top10_text += f"{i}. {stock['ticker']} ({stock['conviction']}/100) - {stock['company'][:25]}\n"

    text = f"""
PROTEUS MORNING REPORT
{datetime.now().strftime('%B %d, %Y')}
{'='*50}

{signal_text}

OVERNIGHT DEEP DIVE SUMMARY
{'='*50}
Total Stocks Analyzed: {summary['total_stocks']}

Conviction Breakdown:
  HIGH:   {summary['tier_counts']['HIGH']} stocks
  MEDIUM: {summary['tier_counts']['MEDIUM']} stocks
  LOW:    {summary['tier_counts']['LOW']} stocks
  AVOID:  {summary['tier_counts']['AVOID']} stocks

{top10_text}

HIGH CONVICTION STOCKS (trade 1.2-1.5x):
{', '.join(summary['tiers'].get('HIGH', [])[:15])}

TRADING RULES:
1. HIGH conviction signal: Trade with 1.2-1.5x position
2. MEDIUM conviction signal: Trade with 1.0x position
3. LOW conviction signal: Trade with 0.5x or skip
4. AVOID conviction signal: Skip the trade

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    return text


def main():
    """Send the morning report email."""
    print("="*60)
    print("PROTEUS MORNING REPORT GENERATOR")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\nLoading overnight summary...")
    summary = load_overnight_summary()
    print(f"  Loaded {summary['total_stocks']} stocks")

    print("\nChecking for today's signals...")
    signals = load_todays_signals()
    print(f"  Found {len(signals)} signals")

    # Generate email
    print("\nGenerating email content...")
    html_body = generate_html_email(summary, signals)
    text_body = generate_text_email(summary, signals)

    # Send email
    print("\nSending email...")
    try:
        notifier = SendGridNotifier()

        if notifier.is_enabled():
            # Use SendGrid's Mail class directly
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail, Email, To, Content

            message = Mail(
                from_email=Email(notifier.config.get('sender_email', 'proteus@trading.local')),
                to_emails=To(notifier.config['recipient_email']),
                subject=f"Proteus Morning Report - {datetime.now().strftime('%B %d, %Y')}",
                html_content=Content("text/html", html_body)
            )

            sg = SendGridAPIClient(notifier.config['sendgrid_api_key'])
            response = sg.send(message)
            print(f"[OK] Email sent! Status: {response.status_code}")
        else:
            print("[WARN] SendGrid not configured - saving to file instead")
            raise Exception("SendGrid not enabled")

    except Exception as e:
        print(f"[WARN] Email sending failed: {e}")

        # Save to file as backup
        backup_file = f"data/deep_dives/morning_report_{datetime.now().strftime('%Y%m%d')}.html"
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(html_body)
        print(f"[OK] Saved HTML report to: {backup_file}")

    print("\n" + "="*60)
    print("REPORT COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

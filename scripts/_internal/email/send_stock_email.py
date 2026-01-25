"""
Send NVDA and PLTR analysis email
"""
import sys
import os

sys.path.insert(0, 'src')

from notifications.sendgrid_notifier import SendGridNotifier

# Initialize notifier
notifier = SendGridNotifier()

# Compose email
subject = "NVDA & PLTR Analysis - Entry Timing Assessment (Nov 19, 2025)"

html_body = """
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">

<h2 style="color: #2c3e50;">NVIDIA (NVDA) & PALANTIR (PLTR) - Technical Analysis</h2>

<div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #e74c3c; margin: 20px 0;">
    <h3 style="color: #e74c3c; margin-top: 0;">⚠️ ENTRY RECOMMENDATION: DO NOT ENTER</h3>
    <p><strong>Both stocks show NO SIGNAL from Proteus mean reversion system.</strong></p>
</div>

<hr style="border: 1px solid #ddd; margin: 30px 0;">

<h3 style="color: #2980b9;">📊 Current State Analysis</h3>

<div style="background-color: #fff; padding: 15px; border: 1px solid #ddd; margin: 20px 0;">
    <h4 style="color: #27ae60;">NVIDIA (NVDA)</h4>
    <ul>
        <li><strong>Proteus Signal:</strong> NO SIGNAL DETECTED</li>
        <li><strong>Technical State:</strong> NOT OVERSOLD</li>
        <li><strong>Mean Reversion Opportunity:</strong> None currently</li>
    </ul>

    <p style="margin-top: 15px;"><strong>What this means:</strong> NVDA has not declined sufficiently to trigger a mean reversion signal. The stock is not in an oversold condition that would indicate a bounce-back opportunity.</p>
</div>

<div style="background-color: #fff; padding: 15px; border: 1px solid #ddd; margin: 20px 0;">
    <h4 style="color: #27ae60;">PALANTIR (PLTR)</h4>
    <ul>
        <li><strong>Proteus Signal:</strong> NO SIGNAL DETECTED</li>
        <li><strong>Technical State:</strong> NOT OVERSOLD</li>
        <li><strong>Mean Reversion Opportunity:</strong> None currently</li>
    </ul>

    <p style="margin-top: 15px;"><strong>What this means:</strong> PLTR has not declined sufficiently to trigger a mean reversion signal. The stock is not in an oversold condition that would indicate a bounce-back opportunity.</p>
</div>

<hr style="border: 1px solid #ddd; margin: 30px 0;">

<h3 style="color: #2980b9;">🔮 Outlook & Timing</h3>

<div style="background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 20px 0;">
    <h4 style="margin-top: 0;">Next Few Days (Short-term)</h4>
    <p><strong>Entry Timing:</strong> <span style="color: #e74c3c;">NOT RECOMMENDED TODAY OR TOMORROW</span></p>
    <ul>
        <li>Wait for mean reversion signal to develop</li>
        <li>Signal appears when stocks decline significantly (RSI < 30, price far below moving average)</li>
        <li>Entering without a signal means you're trend-following, not mean reverting</li>
    </ul>
</div>

<div style="background-color: #d4edda; padding: 15px; border-left: 4px solid #28a745; margin: 20px 0;">
    <h4 style="margin-top: 0;">Next Few Weeks (Medium-term)</h4>
    <p><strong>Strategy:</strong> Monitor for signal development</p>
    <ul>
        <li><strong>If stocks decline 5-10%:</strong> Proteus signal may trigger → Entry opportunity</li>
        <li><strong>If stocks continue upward:</strong> No signal = No entry (avoid chasing)</li>
        <li><strong>Best scenario:</strong> Wait for oversold bounce opportunity</li>
    </ul>
</div>

<div style="background-color: #d1ecf1; padding: 15px; border-left: 4px solid #17a2b8; margin: 20px 0;">
    <h4 style="margin-top: 0;">Next Few Months (Long-term)</h4>
    <p><strong>Framework:</strong> Proteus is optimized for 1-4 day mean reversion holds</p>
    <ul>
        <li>Proteus is <strong>NOT</strong> designed for multi-month forecasting</li>
        <li>System focuses on short-term oversold bounces (63.7% win rate overall)</li>
        <li>For long-term investing, different analysis framework is needed</li>
        <li><strong>Strength zone:</strong> Q4 signals (top 25%) achieve 77.8% win rate</li>
    </ul>
</div>

<hr style="border: 1px solid #ddd; margin: 30px 0;">

<h3 style="color: #2980b9;">📈 What Makes a Good Entry Signal?</h3>

<div style="background-color: #f8f9fa; padding: 15px; border: 1px solid #dee2e6; margin: 20px 0;">
    <p><strong>Proteus Mean Reversion System looks for:</strong></p>
    <ol>
        <li><strong>Oversold Condition:</strong> RSI < 30 (extreme fear)</li>
        <li><strong>Price Dislocation:</strong> Price significantly below moving average</li>
        <li><strong>Volume Surge:</strong> Panic selling creating opportunity</li>
        <li><strong>Signal Strength:</strong> Best trades score 75+ (Q4 quartile)</li>
    </ol>

    <p style="margin-top: 15px; padding: 15px; background-color: #fff; border-left: 4px solid #6c757d;">
        <strong>Historical Performance:</strong><br>
        • Overall Win Rate: 63.7%<br>
        • Q4 Signals (Top 25%): 77.8% win rate<br>
        • Sharpe Ratio: 2.37<br>
        • Typical Hold Period: 1-4 days
    </p>
</div>

<hr style="border: 1px solid #ddd; margin: 30px 0;">

<h3 style="color: #2980b9;">✅ Action Plan</h3>

<div style="background-color: #fff; padding: 20px; border: 2px solid #2980b9; margin: 20px 0;">
    <h4 style="color: #2980b9; margin-top: 0;">Recommended Actions:</h4>
    <ol>
        <li><strong>Today/Tomorrow:</strong> <span style="color: #e74c3c;">DO NOT ENTER</span> - No signal present</li>
        <li><strong>This Week:</strong> Monitor for signal development (if stocks decline)</li>
        <li><strong>Signal Trigger:</strong> Only enter when Proteus generates a signal</li>
        <li><strong>Alternative:</strong> Consider different stocks with active signals</li>
    </ol>
</div>

<hr style="border: 1px solid #ddd; margin: 30px 0;">

<div style="background-color: #f8f9fa; padding: 15px; border-top: 3px solid #6c757d; margin-top: 30px; font-size: 0.9em; color: #6c757d;">
    <p><strong>Disclaimer:</strong> This analysis is based on the Proteus mean reversion trading system, which is optimized for short-term (1-4 day) trades. Not financial advice. Historical performance: 63.7% overall win rate, 77.8% for top quartile signals.</p>

    <p style="margin-top: 10px;"><strong>System Status:</strong> 54 experiments running | 0 completed | Mean reversion framework operational</p>

    <p style="margin-top: 10px;">Generated: November 19, 2025</p>
</div>

</body>
</html>
"""

# Send email
try:
    notifier.send_email(
        subject=subject,
        html_body=html_body,
        text_body="""
NVIDIA (NVDA) & PALANTIR (PLTR) - Technical Analysis
================================================================================

ENTRY RECOMMENDATION: DO NOT ENTER
Both stocks show NO SIGNAL from Proteus mean reversion system.

CURRENT STATE:
- NVIDIA (NVDA): NO SIGNAL - Not oversold
- PALANTIR (PLTR): NO SIGNAL - Not oversold

OUTLOOK:
- Next Few Days: DO NOT ENTER today or tomorrow
- Next Few Weeks: Monitor for signal development (if stocks decline)
- Next Few Months: Proteus not designed for long-term forecasting

RECOMMENDATION:
Wait for mean reversion signal to develop. Entering without a signal means you're trend-following, not mean reverting.

Historical Performance: 63.7% overall win rate, 77.8% for Q4 signals (top 25%)

Generated: November 19, 2025
        """,
        to_emails=["your_email@example.com"]  # Will use default from config
    )
    print("✓ Email sent successfully!")
except Exception as e:
    print(f"✗ Email failed: {e}")

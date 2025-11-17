"""
Send comprehensive email report for EXP-024 Tier A Expansion Round 2
"""
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.notifications.sendgrid_notifier import SendGridNotifier

# Load experiment results
with open('logs/experiments/exp024_expand_tier_a_round2.json', 'r') as f:
    exp_results = json.load(f)

# Initialize notifier
notifier = SendGridNotifier()

if not notifier.is_enabled():
    print("[ERROR] SendGrid is not configured. Please set up email_config.json")
    sys.exit(1)

print("=" * 70)
print("SENDING EXP-024 EMAIL REPORT")
print("=" * 70)
print()

print(f"Experiment: {exp_results['experiment_id']}")
print(f"Date: {exp_results['date']}")
print(f"Stocks Tested: {exp_results['symbols_tested']}")
print()

# Add hypothesis and methodology to results for email
exp_results['hypothesis'] = """
Test 30 new candidate stocks to expand the Tier A portfolio beyond the
current 5 stocks (NVDA, AVGO, V, MA, AXP). Hypothesis: Several large-cap
stocks with sufficient liquidity and volatility will meet our 70%+ win rate
threshold and can be added to expand trading opportunities.
"""

exp_results['methodology'] = """
BASELINE: Current 5 Tier A stocks with proven 70%+ win rates
EXPERIMENTAL: Test 30 new stocks with same mean reversion parameters
- Z-score threshold, RSI oversold, volume multiplier
- Time-decay exit strategy
- 3-year backtest period
CRITERIA: Win rate >= 70% for Tier A promotion
"""

exp_results['stocks_tested_list'] = [
    'ADBE', 'CRM', 'NOW', 'SHOP', 'ORCL', 'NFLX', 'ASML', 'TSM', 'LRCX',
    'AMAT', 'KLAC', 'MRVL', 'JPM', 'BLK', 'SCHW', 'USB', 'PNC', 'TFC',
    'ABBV', 'TMO', 'DHR', 'ISRG', 'SYK', 'BSX', 'NKE', 'SBUX', 'TGT',
    'LOW', 'HON', 'CAT'
]

print("Sending email via SendGrid...")
print()

success = notifier.send_experiment_report('EXP-024', exp_results)

if success:
    print("✅ Email sent successfully!")
    print()
    print("Email includes:")
    print("  - Hypothesis tested")
    print("  - 30 stocks tested (full list)")
    print("  - Test methodology (baseline vs experimental)")
    print("  - Results for all stocks")
    print("  - 6 new Tier A candidates identified:")
    print("    • LOW (100% WR, 8.5% return, 3 trades)")
    print("    • KLAC (80% WR, 20.1% return, 10 trades)")
    print("    • ORCL (76.9% WR, 22.5% return, 13 trades)")
    print("    • MRVL (75% WR, 26.6% return, 16 trades)")
    print("    • ABBV (70.6% WR, 10.5% return, 17 trades)")
    print("    • SYK (70% WR, 11.3% return, 10 trades)")
    print("  - 13 Tier B candidates (60-70% WR)")
    print("  - Recommendation: Add 6 new Tier A stocks")
else:
    print("❌ Failed to send email")
    sys.exit(1)

print()
print("=" * 70)
print("EMAIL REPORT SENT")
print("=" * 70)

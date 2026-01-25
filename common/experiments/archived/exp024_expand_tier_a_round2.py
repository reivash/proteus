"""
EXPERIMENT: EXP-024
Date: 2025-11-16
Objective: Expand Tier A universe from 5 to 9-12 stocks (Round 2)

RESEARCH MOTIVATION:
Stock selection has been our biggest wins:
- EXP-014: +15.99pp improvement from selective trading
- EXP-021: Found AVGO (87.5%) and AXP (71.4%)

Current Tier A: 5 stocks (NVDA, AVGO, V, MA, AXP) → ~25 trades/year
Target: 9-12 stocks → ~45-60 trades/year (+80-140% opportunities)

EXP-023 proved Tier A stocks work in ALL market regimes.
Best path forward: Find MORE high-quality stocks (not optimize strategy).

HYPOTHESIS:
Testing 30 additional symbols from S&P 100/500 will find 4-6 new Tier A stocks.
Hit rate expectation: 15-20% (based on EXP-021: 2/20 = 10%, conservative estimate).

SYMBOLS TO TEST (30 stocks across 5 sectors):

Tech Giants (6):
- ADBE (Adobe)
- CRM (Salesforce)
- NOW (ServiceNow)
- SHOP (Shopify)
- ORCL (Oracle)
- NFLX (Netflix)

Semiconductors (6):
- ASML (ASML Holding)
- TSM (Taiwan Semiconductor)
- LRCX (Lam Research)
- AMAT (Applied Materials)
- KLAC (KLA Corporation)
- MRVL (Marvell Technology)

Finance (6):
- JPM (retest - was 55.6% in EXP-014)
- BLK (BlackRock)
- SCHW (Charles Schwab)
- USB (U.S. Bancorp)
- PNC (PNC Financial)
- TFC (Truist Financial)

Healthcare (6):
- ABBV (AbbVie)
- TMO (Thermo Fisher)
- DHR (Danaher)
- ISRG (Intuitive Surgical)
- SYK (Stryker)
- BSX (Boston Scientific)

Consumer/Industrial (6):
- NKE (Nike)
- SBUX (Starbucks)
- TGT (Target)
- LOW (Lowe's)
- HON (Honeywell)
- CAT (Caterpillar)

TIER A CRITERIA (>70% win rate):
- Win rate: >70% (CRITICAL)
- Total return: >5% over 3 years
- Sharpe ratio: >5.0 (preferred)
- Min 5 trades (statistical significance)

EXPECTED OUTCOMES:
- Find 4-6 new Tier A stocks
- Expand from 5 to 9-11 total Tier A stocks
- Trade frequency: ~25 → ~45-55 trades/year
- Return improvement: +3-5pp from increased activity
- Sector diversification: Reduce tech concentration
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.experiments.exp014_stock_selection_optimization import run_exp014_stock_selection
import pandas as pd
import json
from datetime import datetime


def run_exp024_expand_tier_a_round2(period: str = "3y"):
    """
    Test 30 additional stocks to find more Tier A candidates (Round 2).

    Args:
        period: Backtest period
    """
    print("=" * 70)
    print("EXP-024: EXPAND TIER A UNIVERSE (ROUND 2)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Find 4-6 new Tier A stocks (>70% win rate)")
    print("Current Tier A (5 stocks):")
    print("  NVDA: 87.5%, AVGO: 87.5%, V: 75.0%, MA: 71.4%, AXP: 71.4%")
    print("  Avg: 78.7% win rate, ~25 trades/year")
    print()
    print("Target: 9-11 total Tier A stocks, ~45-55 trades/year")
    print()

    # Test 30 new symbols
    new_symbols = [
        # Tech Giants
        'ADBE', 'CRM', 'NOW', 'SHOP', 'ORCL', 'NFLX',

        # Semiconductors
        'ASML', 'TSM', 'LRCX', 'AMAT', 'KLAC', 'MRVL',

        # Finance
        'JPM', 'BLK', 'SCHW', 'USB', 'PNC', 'TFC',

        # Healthcare
        'ABBV', 'TMO', 'DHR', 'ISRG', 'SYK', 'BSX',

        # Consumer/Industrial
        'NKE', 'SBUX', 'TGT', 'LOW', 'HON', 'CAT'
    ]

    print(f"Testing {len(new_symbols)} new symbols:")
    print(f"  Tech: ADBE, CRM, NOW, SHOP, ORCL, NFLX")
    print(f"  Semiconductors: ASML, TSM, LRCX, AMAT, KLAC, MRVL")
    print(f"  Finance: JPM, BLK, SCHW, USB, PNC, TFC")
    print(f"  Healthcare: ABBV, TMO, DHR, ISRG, SYK, BSX")
    print(f"  Consumer/Industrial: NKE, SBUX, TGT, LOW, HON, CAT")
    print()

    # Run stock selection backtest (reuse EXP-014 code)
    results = run_exp014_stock_selection(new_symbols, period=period)

    # Additional analysis
    print("\n" + "=" * 70)
    print("COMPARISON WITH EXISTING TIER A")
    print("=" * 70)
    print()

    existing_tier_a = {
        'NVDA': {'win_rate': 87.5, 'return': 49.70, 'sharpe': 19.58, 'trades': 8},
        'AVGO': {'win_rate': 87.5, 'return': 24.52, 'sharpe': 12.65, 'trades': 8},
        'V': {'win_rate': 75.0, 'return': 7.33, 'sharpe': 7.03, 'trades': 8},
        'MA': {'win_rate': 71.4, 'return': 5.99, 'sharpe': 6.92, 'trades': 7},
        'AXP': {'win_rate': 71.4, 'return': 29.21, 'sharpe': 7.61, 'trades': 14}
    }

    print("EXISTING TIER A (Confirmed v7.0):")
    for ticker, stats in existing_tier_a.items():
        print(f"  [{ticker}] {stats['win_rate']:.1f}% win rate, {stats['return']:+.2f}% return, {stats['trades']} trades")

    print()
    print("NEW TIER A CANDIDATES (from today's test):")
    print("  (See tier classification above)")

    # Portfolio projection
    print()
    print("=" * 70)
    print("PORTFOLIO PROJECTION")
    print("=" * 70)
    print()

    # Get new Tier A from results
    if 'tier_a' in results and results['tier_a']:
        new_tier_a_count = len(results['tier_a'])
        total_tier_a = 5 + new_tier_a_count

        print(f"NEW Tier A stocks found: {new_tier_a_count}")
        print(f"TOTAL Tier A portfolio: {total_tier_a} stocks")
        print()

        # Calculate expected trades
        avg_trades_per_stock = 5  # Conservative estimate
        expected_trades = total_tier_a * avg_trades_per_stock
        current_trades = 25

        print(f"Trade frequency projection:")
        print(f"  Current: ~{current_trades} trades/year (5 stocks)")
        print(f"  Projected: ~{expected_trades} trades/year ({total_tier_a} stocks)")
        print(f"  Improvement: +{expected_trades - current_trades} trades/year (+{((expected_trades - current_trades) / current_trades * 100):.0f}%)")
        print()

        print(f"Expected impact:")
        print(f"  - More trading opportunities (+{((expected_trades - current_trades) / current_trades * 100):.0f}%)")
        print(f"  - Better sector diversification")
        print(f"  - Reduced single-stock concentration risk")
        print(f"  - Return improvement: +3-5pp from increased activity")
    else:
        print("No new Tier A stocks found in this round.")
        print("Consider:")
        print("  1. Test additional symbols from S&P 500")
        print("  2. Adjust Tier A criteria (e.g., >65% win rate)")
        print("  3. Focus on optimizing existing 5 stocks")

    print()
    print("=" * 70)

    # Save combined results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    combined_results = {
        'experiment_id': 'EXP-024',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': period,
        'symbols_tested': len(new_symbols),
        'existing_tier_a': existing_tier_a,
        'new_test_results': results,
        'recommendation': 'Update mean_reversion_params.py with new Tier A stocks'
    }

    results_file = os.path.join(results_dir, 'exp024_expand_tier_a_round2.json')
    with open(results_file, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)

    print(f"Results saved to: {results_file}")

    # Send experiment report email if configured
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        from common.notifications.sendgrid_notifier import SendGridNotifier

        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending experiment report email...")
            notifier.send_experiment_report('EXP-024', combined_results)
        else:
            print("\n[INFO] Email notifications not configured (see email_config.json)")
    except Exception as e:
        print(f"\n[WARNING] Could not send experiment report email: {e}")

    return combined_results


if __name__ == '__main__':
    """Run EXP-024 to expand Tier A stock universe (Round 2)."""

    results = run_exp024_expand_tier_a_round2(period='3y')

    print("\n\nNext steps:")
    print("1. Add new Tier A stocks to mean_reversion_params.py")
    print("2. Update dashboard to monitor expanded universe")
    print("3. Deploy with 9-12 Tier A stocks for maximum opportunities")
    print("4. Monitor performance vs baseline")

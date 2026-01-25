"""
EXPERIMENT: EXP-030
Date: 2025-11-16
Objective: Analyze portfolio correlation and sector concentration risk

RESEARCH MOTIVATION:
We now have 22 Tier A stocks (v12.0) delivering +28.84% average return.
But are we truly diversified, or do we have hidden concentration risks?

CURRENT SECTOR BREAKDOWN:
- Semiconductors (4): NVDA, AVGO, KLAC, MRVL
- Tech/Software (3): ORCL, TXN, INTU
- Healthcare (5): ABBV, SYK, GILD, IDXX, DXCM, INSM
- Payments (2): V, MA
- Finance (1): AXP
- Energy (1): EOG
- Materials (1): MLM
- Cybersecurity (1): FTNT
- REITs (2): EXR, INVH
- Construction (1): ROAD

CONCENTRATION CONCERNS:
1. **Semiconductors = 4 stocks (18%)** - High correlation likely
2. **Healthcare = 6 stocks (27%)** - Largest sector exposure
3. **Tech broadly = 8 stocks (36%)** - If you include semis + software + cybersecurity
4. Missing sectors: Utilities, Telecom, Consumer Staples

HYPOTHESIS:
Portfolio may appear diversified (9 sectors) but could have high correlation,
especially within tech/semi cluster. This could amplify drawdowns during
tech selloffs.

ANALYSIS TO PERFORM:

1. CORRELATION MATRIX
   - Calculate pairwise correlation for all 22 stocks
   - Identify high-correlation clusters (>0.7)
   - Calculate average portfolio correlation

2. SECTOR CONCENTRATION
   - Weight by expected trade frequency
   - Herfindahl-Hirschman Index (HHI) by sector
   - Identify over-concentrated sectors

3. DRAWDOWN SIMULATION
   - Simulate portfolio behavior during:
     * 2022 tech selloff
     * COVID crash (March 2020)
     * 2018 December selloff
   - Measure if diversification actually reduces risk

4. FACTOR EXPOSURE
   - Growth vs Value
   - Large vs Small cap
   - Momentum vs Mean Reversion (ironic!)

RISK METRICS TO CALCULATE:
- Portfolio correlation (avg pairwise)
- Max drawdown
- Correlation during stress events
- Sector Herfindahl Index
- Beta to SPY (market exposure)

SUCCESS CRITERIA:
- Average correlation < 0.50 (reasonable diversification)
- No sector > 30% allocation
- Max drawdown < 25% during stress periods
- HHI < 0.15 (not over-concentrated)

POTENTIAL OUTCOMES:

1. Well diversified → No action needed
2. Tech over-concentration → Add more defensive sectors
3. High correlation → Consider removing correlated stocks
4. Stress period vulnerability → Add hedges or sector limits

NOTE: This is DIAGNOSTIC only. No trading decisions, just analysis.
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.config.mean_reversion_params import MEAN_REVERSION_PARAMS
from common.data.fetchers.yahoo_finance import YahooFinanceFetcher


def classify_sector(symbol: str) -> str:
    """Classify stock into sector based on known mapping."""
    sector_map = {
        # Semiconductors
        'NVDA': 'Semiconductors', 'AVGO': 'Semiconductors',
        'KLAC': 'Semiconductors', 'MRVL': 'Semiconductors',

        # Tech/Software
        'ORCL': 'Tech/Software', 'TXN': 'Tech/Software',
        'INTU': 'Tech/Software', 'FTNT': 'Cybersecurity',

        # Healthcare
        'ABBV': 'Healthcare/Pharma', 'SYK': 'Healthcare/Medical Devices',
        'GILD': 'Healthcare/Pharma', 'IDXX': 'Healthcare/Medical Devices',
        'DXCM': 'Healthcare/Medical Devices', 'INSM': 'Healthcare/Biotech',

        # Finance/Payments
        'V': 'Payments', 'MA': 'Payments', 'AXP': 'Finance',

        # Other
        'EOG': 'Energy', 'MLM': 'Materials',
        'EXR': 'REITs', 'INVH': 'REITs', 'ROAD': 'Construction'
    }
    return sector_map.get(symbol, 'Unknown')


def run_exp030_portfolio_correlation():
    """
    Analyze correlation and concentration risk in Tier A portfolio.
    """
    print("=" * 70)
    print("EXP-030: PORTFOLIO CORRELATION & RISK ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Analyze diversification and concentration risk")
    print("Portfolio: 22 Tier A stocks (v12.0)")
    print()

    # Get Tier A stocks
    tier_a_symbols = [symbol for symbol, params in MEAN_REVERSION_PARAMS.items()
                      if symbol != 'DEFAULT' and params.get('tier') == 'A']

    print(f"Analyzing {len(tier_a_symbols)} Tier A stocks:")
    for symbol in tier_a_symbols:
        sector = classify_sector(symbol)
        print(f"  {symbol}: {sector}")
    print()

    # Fetch price data for correlation analysis
    print("Fetching 1-year price data for correlation analysis...")
    fetcher = YahooFinanceFetcher()
    price_data = {}
    failed_symbols = []

    for symbol in tier_a_symbols:
        try:
            df = fetcher.fetch_stock_data(symbol, period='1y')
            if df is not None and len(df) > 0:
                price_data[symbol] = df['Close']
            else:
                failed_symbols.append(symbol)
                print(f"  [WARNING] No data for {symbol}")
        except Exception as e:
            failed_symbols.append(symbol)
            print(f"  [ERROR] {symbol}: {e}")

    if failed_symbols:
        print(f"\n[WARNING] Could not fetch data for: {', '.join(failed_symbols)}")
        print("Continuing with available stocks...")
        tier_a_symbols = [s for s in tier_a_symbols if s not in failed_symbols]

    print(f"\nAnalyzing {len(tier_a_symbols)} stocks with valid data")
    print()

    # Calculate returns for correlation
    returns_df = pd.DataFrame()
    for symbol in tier_a_symbols:
        returns_df[symbol] = price_data[symbol].pct_change()

    returns_df = returns_df.dropna()

    # 1. CORRELATION MATRIX
    print("=" * 70)
    print("1. CORRELATION ANALYSIS")
    print("=" * 70)
    print()

    corr_matrix = returns_df.corr()

    # Calculate average correlation (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    avg_correlation = corr_matrix.where(mask).stack().mean()

    print(f"Average pairwise correlation: {avg_correlation:.3f}")
    print()

    # Find highly correlated pairs (>0.7)
    print("Highly correlated pairs (correlation > 0.7):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val > 0.7:
                pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_val)
                high_corr_pairs.append(pair)
                print(f"  {pair[0]} <-> {pair[1]}: {pair[2]:.3f}")

    if not high_corr_pairs:
        print("  None found")

    print()

    # 2. SECTOR CONCENTRATION
    print("=" * 70)
    print("2. SECTOR CONCENTRATION ANALYSIS")
    print("=" * 70)
    print()

    sector_counts = {}
    for symbol in tier_a_symbols:
        sector = classify_sector(symbol)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    total_stocks = len(tier_a_symbols)

    print("Sector allocation (equal weighting):")
    sector_weights = []
    for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True):
        weight = count / total_stocks
        sector_weights.append(weight)
        print(f"  {sector:<30s}: {count:2d} stocks ({weight*100:5.1f}%)")

    # Calculate HHI
    hhi = sum([w**2 for w in sector_weights])
    print()
    print(f"Herfindahl-Hirschman Index (HHI): {hhi:.3f}")

    if hhi < 0.15:
        concentration_level = "Low (well diversified)"
    elif hhi < 0.25:
        concentration_level = "Moderate"
    else:
        concentration_level = "High (concentrated)"

    print(f"Concentration level: {concentration_level}")
    print()

    # 3. SECTOR CORRELATION
    print("=" * 70)
    print("3. SECTOR-LEVEL CORRELATION")
    print("=" * 70)
    print()

    # Group stocks by sector and calculate average within-sector correlation
    within_sector_corr = {}
    for sector in sector_counts.keys():
        sector_symbols = [s for s in tier_a_symbols if classify_sector(s) == sector]
        if len(sector_symbols) >= 2:
            sector_corr_matrix = corr_matrix.loc[sector_symbols, sector_symbols]
            mask = np.triu(np.ones_like(sector_corr_matrix, dtype=bool), k=1)
            avg_corr = sector_corr_matrix.where(mask).stack().mean()
            within_sector_corr[sector] = avg_corr
            print(f"  {sector:<30s}: {avg_corr:.3f} avg correlation ({len(sector_symbols)} stocks)")

    print()

    # 4. RECOMMENDATIONS
    print("=" * 70)
    print("RISK ASSESSMENT & RECOMMENDATIONS")
    print("=" * 70)
    print()

    issues = []
    recommendations = []

    # Check average correlation
    if avg_correlation > 0.50:
        issues.append(f"High average correlation ({avg_correlation:.3f})")
        recommendations.append("Consider removing highly correlated stocks")
    elif avg_correlation < 0.30:
        print(f"[EXCELLENT] Low correlation ({avg_correlation:.3f}) = good diversification")
    else:
        print(f"[GOOD] Moderate correlation ({avg_correlation:.3f}) = acceptable")

    # Check sector concentration
    max_sector_weight = max(sector_weights)
    if max_sector_weight > 0.30:
        max_sector_name = max(sector_counts.items(), key=lambda x: x[1])[0]
        issues.append(f"{max_sector_name} over-weighted ({max_sector_weight*100:.1f}%)")
        recommendations.append(f"Reduce {max_sector_name} exposure or add other sectors")
    else:
        print(f"[GOOD] No sector exceeds 30% allocation (max: {max_sector_weight*100:.1f}%)")

    # Check HHI
    if hhi > 0.25:
        issues.append(f"High concentration (HHI = {hhi:.3f})")
        recommendations.append("Add stocks from under-represented sectors")
    else:
        print(f"[GOOD] Reasonable diversification (HHI = {hhi:.3f})")

    print()
    if issues:
        print("ISSUES IDENTIFIED:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        print("RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  {rec}")
    else:
        print("[SUCCESS] Portfolio is well-diversified!")
        print("No concentration or correlation issues detected.")

    print()
    print("=" * 70)

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    combined_results = {
        'experiment_id': 'EXP-030',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stocks_analyzed': len(tier_a_symbols),
        'avg_correlation': float(avg_correlation),
        'sector_hhi': float(hhi),
        'sector_breakdown': {k: v for k, v in sector_counts.items()},
        'high_correlation_pairs': [[p[0], p[1], float(p[2])] for p in high_corr_pairs],
        'issues': issues,
        'recommendations': recommendations
    }

    results_file = os.path.join(results_dir, 'exp030_portfolio_correlation.json')
    with open(results_file, 'w') as f:
        json.dump(combined_results, f, indent=2)

    print(f"Results saved to: {results_file}")

    # Send email report
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending experiment report email...")
            notifier.send_experiment_report('EXP-030', combined_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return combined_results


if __name__ == '__main__':
    """Run EXP-030 portfolio correlation analysis."""

    results = run_exp030_portfolio_correlation()

    print("\n\nNext steps:")
    print("1. Address any concentration issues identified")
    print("2. Consider sector limits in production (optional)")
    print("3. Monitor correlation during market stress")
    print("4. Re-run quarterly to track portfolio evolution")

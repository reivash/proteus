"""
Diagnostic Tool: Why Companies Were Filtered Out

Checks each company against screening filters to understand why they didn't pass.
"""

import sys
sys.path.insert(0, 'src')

from src.data.fetchers.ipo_data_fetcher import IPODataFetcher
from src.data.features.ipo_niche_analyzer import IPONicheAnalyzer


def diagnose_company(ticker: str, filters: dict):
    """
    Diagnose why a company didn't pass filters.

    Args:
        ticker: Stock ticker
        filters: Filter criteria dict
    """
    print(f"\n{'='*80}")
    print(f"DIAGNOSING: {ticker}")
    print(f"{'='*80}")

    fetcher = IPODataFetcher(max_ipo_age_months=filters['max_ipo_age_months'])
    analyzer = IPONicheAnalyzer()

    # Get company info
    company_info = fetcher.get_company_info(ticker)

    if 'error' in company_info:
        print(f"ERROR: {company_info['error']}")
        return

    print(f"Company: {company_info.get('company_name', 'Unknown')}")
    print(f"Ticker: {ticker}")
    print()

    # Check each filter
    passed = []
    failed = []

    # 1. IPO Age Filter
    months_since_ipo = company_info.get('months_since_ipo')
    ipo_date = company_info.get('ipo_date')

    if months_since_ipo is None:
        failed.append(f"IPO Age: No IPO date in database")
    elif months_since_ipo > filters['max_ipo_age_months']:
        failed.append(f"IPO Age: {months_since_ipo:.0f} months > {filters['max_ipo_age_months']} month limit")
        failed.append(f"  IPO Date: {ipo_date.strftime('%Y-%m-%d') if ipo_date else 'Unknown'}")
    else:
        passed.append(f"IPO Age: {months_since_ipo:.0f} months (< {filters['max_ipo_age_months']} limit) [OK]")
        passed.append(f"  IPO Date: {ipo_date.strftime('%Y-%m-%d') if ipo_date else 'Unknown'}")

    # 2. Market Cap Filter
    market_cap_b = company_info.get('market_cap_b', 0)

    if market_cap_b < filters['min_market_cap_b']:
        failed.append(f"Market Cap: ${market_cap_b:.2f}B < ${filters['min_market_cap_b']}B minimum")
    elif market_cap_b > filters['max_market_cap_b']:
        failed.append(f"Market Cap: ${market_cap_b:.2f}B > ${filters['max_market_cap_b']}B maximum")
    else:
        passed.append(f"Market Cap: ${market_cap_b:.2f}B (within range) [OK]")

    # 3. Get fundamentals and YC score
    fundamentals = fetcher.get_fundamental_metrics(ticker)

    if 'error' in fundamentals:
        failed.append(f"Fundamentals: Error fetching data - {fundamentals['error']}")
    else:
        # Calculate YC score
        yc_analysis = analyzer.calculate_yc_score(ticker, company_info, fundamentals)
        yc_score = yc_analysis.get('yc_score', 0)

        if yc_score < filters['min_yc_score']:
            failed.append(f"YC Score: {yc_score:.1f}/100 < {filters['min_yc_score']:.1f} minimum")
        else:
            passed.append(f"YC Score: {yc_score:.1f}/100 (> {filters['min_yc_score']:.1f} minimum) [OK]")

        # Show key metrics
        print("KEY METRICS:")
        print(f"  Revenue Growth: {(fundamentals.get('revenue_growth_yoy', 0) or 0)*100:.1f}% YoY")
        print(f"  Gross Margin: {(fundamentals.get('gross_margin', 0) or 0)*100:.1f}%")
        print(f"  Operating Margin: {(fundamentals.get('operating_margin', 0) or 0)*100:.1f}%")
        print()

    # Print results
    if passed:
        print("PASSED FILTERS:")
        for check in passed:
            print(f"  [PASS] {check}")
        print()

    if failed:
        print("FAILED FILTERS:")
        for check in failed:
            print(f"  [FAIL] {check}")
        print()

    if not failed:
        print("STATUS: All filters PASSED - Should have been analyzed!")
    else:
        print(f"STATUS: FILTERED OUT ({len(failed)} filter(s) failed)")

    print(f"{'='*80}")


def main():
    print("="*80)
    print("DIAGNOSTIC: Why Companies Were Filtered Out")
    print("="*80)
    print()

    # Companies that were filtered out
    filtered_companies = [
        'AI',      # C3.ai
        'CRWD',    # CrowdStrike
        'ZS',      # Zscaler
        'DDOG',    # Datadog
    ]

    # Our filter settings from analyze_ai_ml_ipos.py
    filters = {
        'max_ipo_age_months': 72,    # 6 years
        'min_yc_score': 30.0,
        'min_market_cap_b': 0.1,     # $100M
        'max_market_cap_b': 500.0    # $500B
    }

    print("Filter Settings:")
    print(f"  Max IPO Age: {filters['max_ipo_age_months']} months ({filters['max_ipo_age_months']/12:.1f} years)")
    print(f"  Min YC Score: {filters['min_yc_score']:.1f}/100")
    print(f"  Market Cap Range: ${filters['min_market_cap_b']:.1f}B - ${filters['max_market_cap_b']:.1f}B")
    print()

    print(f"Analyzing {len(filtered_companies)} filtered companies...")
    print()

    for ticker in filtered_companies:
        diagnose_company(ticker, filters)

    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    print()
    print("Next Steps:")
    print("1. If companies failed due to IPO age: Increase max_ipo_age_months")
    print("2. If companies failed due to market cap: Adjust market cap range")
    print("3. If companies failed due to YC score: Lower min_yc_score threshold")
    print("4. If no IPO date: Add to IPO database (src/data/fetchers/ipo_database.py)")
    print()


if __name__ == "__main__":
    main()

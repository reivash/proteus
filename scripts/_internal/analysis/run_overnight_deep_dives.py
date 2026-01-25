"""
Overnight Deep Dive Runner

Runs comprehensive deep dive analysis on all 54 Proteus stocks.
Designed to run overnight and produce a morning summary email.

Started: 2025-12-18 ~11pm
Target completion: 2025-12-19 8am
"""

import os
import sys
import json
import time
import yfinance as yf
from datetime import datetime
from pathlib import Path

# All 54 Proteus tickers
TICKERS = [
    "NVDA", "V", "MA", "AVGO", "AXP", "KLAC", "ORCL", "MRVL", "ABBV", "SYK",
    "EOG", "TXN", "GILD", "INTU", "MSFT", "QCOM", "JPM", "JNJ", "PFE", "WMT",
    "AMAT", "ADI", "NOW", "MLM", "IDXX", "EXR", "ROAD", "INSM", "SCHW", "AIG",
    "USB", "CVS", "LOW", "LMT", "COP", "SLB", "APD", "MS", "PNC", "CRM",
    "ADBE", "TGT", "CAT", "XOM", "MPC", "ECL", "NEE", "HCA", "CMCSA", "TMUS",
    "META", "ETN", "HD", "SHW"
]

# Already analyzed today
COMPLETED = ["AVGO", "ETN"]

OUTPUT_DIR = "data/deep_dives"
SUMMARY_FILE = "data/deep_dives/overnight_summary_2025-12-19.json"

def get_stock_data(ticker):
    """Fetch comprehensive stock data."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            'ticker': ticker,
            'company_name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'price_to_book': info.get('priceToBook'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'profit_margin': info.get('profitMargins'),
            'operating_margin': info.get('operatingMargins'),
            'roe': info.get('returnOnEquity'),
            'roa': info.get('returnOnAssets'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'free_cash_flow': info.get('freeCashflow'),
            'beta': info.get('beta'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
            'current_price': info.get('currentPrice', info.get('regularMarketPrice')),
            'analyst_target': info.get('targetMeanPrice'),
            'recommendation': info.get('recommendationKey'),
            'short_percent': info.get('shortPercentOfFloat'),
            'insider_ownership': info.get('heldPercentInsiders'),
            'institutional_ownership': info.get('heldPercentInstitutions'),
            'business_summary': info.get('longBusinessSummary', '')[:1000],
        }
    except Exception as e:
        return {'ticker': ticker, 'error': str(e)}


def analyze_stock_quick(data):
    """
    Quick analysis scoring based on quantitative factors.
    This runs without API calls - pure data analysis.
    """
    scores = {}

    # Business Quality (based on margins and ROE)
    profit_margin = data.get('profit_margin') or 0
    roe = data.get('roe') or 0
    scores['business_quality'] = min(10, max(1,
        5 + (profit_margin * 20) + (roe * 10)
    ))

    # Financial Health (debt, current ratio, FCF)
    de = data.get('debt_to_equity') or 50
    cr = data.get('current_ratio') or 1
    fcf = data.get('free_cash_flow') or 0
    market_cap = data.get('market_cap') or 1
    fcf_yield = (fcf / market_cap) * 100 if market_cap > 0 else 0

    debt_score = 10 - min(10, de / 50)  # Lower debt = higher score
    liquidity_score = min(10, cr * 3)
    fcf_score = min(10, fcf_yield * 2)
    scores['financial_health'] = (debt_score + liquidity_score + fcf_score) / 3

    # Valuation (PE, forward PE, PEG)
    pe = data.get('pe_ratio') or 25
    fwd_pe = data.get('forward_pe') or pe

    if pe < 15:
        val_score = 8
    elif pe < 25:
        val_score = 6
    elif pe < 40:
        val_score = 4
    else:
        val_score = 2

    # Bonus for forward PE lower than trailing
    if fwd_pe and pe and fwd_pe < pe:
        val_score += 1

    scores['valuation'] = min(10, val_score)

    # Growth (revenue and earnings growth)
    rev_growth = data.get('revenue_growth') or 0
    earn_growth = data.get('earnings_growth') or 0
    scores['growth'] = min(10, max(1, 5 + (rev_growth * 20) + (earn_growth * 10)))

    # Analyst Sentiment
    rec = data.get('recommendation', '')
    if 'strong_buy' in str(rec).lower():
        scores['sentiment'] = 9
    elif 'buy' in str(rec).lower():
        scores['sentiment'] = 7
    elif 'hold' in str(rec).lower():
        scores['sentiment'] = 5
    else:
        scores['sentiment'] = 4

    # Institutional Confidence
    inst_own = data.get('institutional_ownership') or 0
    scores['institutional'] = min(10, inst_own * 12)

    # Risk Score (beta, short interest)
    beta = data.get('beta') or 1
    short_pct = data.get('short_percent') or 0

    beta_risk = 10 - min(5, abs(beta - 1) * 3)  # Penalize high/low beta
    short_risk = 10 - min(5, short_pct * 50)  # Penalize high short interest
    scores['risk'] = (beta_risk + short_risk) / 2

    # Composite conviction score
    weights = {
        'business_quality': 0.20,
        'financial_health': 0.20,
        'valuation': 0.15,
        'growth': 0.15,
        'sentiment': 0.10,
        'institutional': 0.10,
        'risk': 0.10
    }

    conviction = sum(scores[k] * weights[k] * 10 for k in weights)

    # Tier assignment
    if conviction >= 75:
        tier = 'HIGH'
    elif conviction >= 50:
        tier = 'MEDIUM'
    elif conviction >= 25:
        tier = 'LOW'
    else:
        tier = 'AVOID'

    return {
        'scores': scores,
        'conviction_score': round(conviction, 1),
        'conviction_tier': tier
    }


def generate_brief_analysis(data, analysis):
    """Generate brief text analysis."""
    ticker = data.get('ticker', 'Unknown')
    name = data.get('company_name', ticker)
    sector = data.get('sector', 'Unknown')

    # Bull case based on strengths
    strengths = []
    if analysis['scores'].get('business_quality', 0) >= 7:
        strengths.append('high quality business')
    if analysis['scores'].get('financial_health', 0) >= 7:
        strengths.append('strong balance sheet')
    if analysis['scores'].get('growth', 0) >= 7:
        strengths.append('solid growth')
    if analysis['scores'].get('valuation', 0) >= 7:
        strengths.append('attractive valuation')

    bull = f"Quality {sector.lower()} company" + (f" with {', '.join(strengths[:2])}" if strengths else "")

    # Bear case based on weaknesses
    weaknesses = []
    if analysis['scores'].get('valuation', 10) <= 4:
        weaknesses.append('expensive valuation')
    if analysis['scores'].get('financial_health', 10) <= 4:
        weaknesses.append('balance sheet concerns')
    if analysis['scores'].get('risk', 10) <= 4:
        weaknesses.append('elevated risk profile')

    bear = ', '.join(weaknesses[:2]).capitalize() if weaknesses else 'Sector rotation risk'

    return {
        'bull_case': bull,
        'bear_case': bear
    }


def run_deep_dive(ticker):
    """Run deep dive on single stock."""
    print(f"  Analyzing {ticker}...", end=" ", flush=True)

    try:
        # Get data
        data = get_stock_data(ticker)

        if 'error' in data:
            print(f"ERROR: {data['error']}")
            return None

        # Analyze
        analysis = analyze_stock_quick(data)
        brief = generate_brief_analysis(data, analysis)

        # Build result
        result = {
            'ticker': ticker,
            'company_name': data.get('company_name', ticker),
            'sector': data.get('sector', 'Unknown'),
            'industry': data.get('industry', 'Unknown'),
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),

            # Scores
            'business_quality_score': round(analysis['scores'].get('business_quality', 5), 1),
            'financial_health_score': round(analysis['scores'].get('financial_health', 5), 1),
            'valuation_score': round(analysis['scores'].get('valuation', 5), 1),
            'growth_score': round(analysis['scores'].get('growth', 5), 1),
            'sentiment_score': round(analysis['scores'].get('sentiment', 5), 1),
            'institutional_score': round(analysis['scores'].get('institutional', 5), 1),
            'risk_score': round(analysis['scores'].get('risk', 5), 1),

            # Composite
            'conviction_score': analysis['conviction_score'],
            'conviction_tier': analysis['conviction_tier'],

            # Analysis
            'bull_case': brief['bull_case'],
            'bear_case': brief['bear_case'],

            # Key metrics for reference
            'market_cap': data.get('market_cap'),
            'pe_ratio': data.get('pe_ratio'),
            'debt_to_equity': data.get('debt_to_equity'),
            'profit_margin': data.get('profit_margin'),
            'revenue_growth': data.get('revenue_growth'),
            'current_price': data.get('current_price'),
            'analyst_target': data.get('analyst_target'),
            'recommendation': data.get('recommendation'),

            # Position guidance
            'position_size_modifier': 1.0 + (analysis['conviction_score'] - 50) / 100,
        }

        # Save individual file
        filepath = os.path.join(OUTPUT_DIR, f"{ticker}_{result['analysis_date']}.json")
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"OK - {analysis['conviction_tier']} ({analysis['conviction_score']})")
        return result

    except Exception as e:
        print(f"FAILED: {e}")
        return None


def generate_summary(results):
    """Generate summary report."""

    # Sort by conviction
    sorted_results = sorted(
        [r for r in results if r],
        key=lambda x: x['conviction_score'],
        reverse=True
    )

    # Tier counts
    tiers = {'HIGH': [], 'MEDIUM': [], 'LOW': [], 'AVOID': []}
    for r in sorted_results:
        tiers[r['conviction_tier']].append(r['ticker'])

    # Sector breakdown
    sectors = {}
    for r in sorted_results:
        sector = r.get('sector', 'Unknown')
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append({
            'ticker': r['ticker'],
            'conviction': r['conviction_score'],
            'tier': r['conviction_tier']
        })

    summary = {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_stocks': len(sorted_results),

        'tier_counts': {
            'HIGH': len(tiers['HIGH']),
            'MEDIUM': len(tiers['MEDIUM']),
            'LOW': len(tiers['LOW']),
            'AVOID': len(tiers['AVOID'])
        },

        'tiers': tiers,

        'top_10': [
            {
                'ticker': r['ticker'],
                'company': r['company_name'],
                'conviction': r['conviction_score'],
                'tier': r['conviction_tier'],
                'sector': r.get('sector'),
                'bull_case': r.get('bull_case'),
            }
            for r in sorted_results[:10]
        ],

        'bottom_5': [
            {
                'ticker': r['ticker'],
                'company': r['company_name'],
                'conviction': r['conviction_score'],
                'tier': r['conviction_tier'],
                'bear_case': r.get('bear_case'),
            }
            for r in sorted_results[-5:]
        ],

        'by_sector': sectors,

        'all_results': [
            {
                'ticker': r['ticker'],
                'conviction': r['conviction_score'],
                'tier': r['conviction_tier']
            }
            for r in sorted_results
        ]
    }

    return summary


def generate_email_content(summary):
    """Generate email-ready summary."""

    content = f"""
PROTEUS OVERNIGHT DEEP DIVE RESULTS
Generated: {summary['generated']}
================================================================================

EXECUTIVE SUMMARY
-----------------
Total Stocks Analyzed: {summary['total_stocks']}

Conviction Tiers:
  HIGH:   {summary['tier_counts']['HIGH']} stocks
  MEDIUM: {summary['tier_counts']['MEDIUM']} stocks
  LOW:    {summary['tier_counts']['LOW']} stocks
  AVOID:  {summary['tier_counts']['AVOID']} stocks

================================================================================

TOP 10 HIGHEST CONVICTION STOCKS
--------------------------------
"""

    for i, stock in enumerate(summary['top_10'], 1):
        content += f"""
{i}. {stock['ticker']} - {stock['company']}
   Conviction: {stock['conviction']}/100 ({stock['tier']})
   Sector: {stock['sector']}
   Bull Case: {stock['bull_case']}
"""

    content += """
================================================================================

BOTTOM 5 LOWEST CONVICTION STOCKS (AVOID)
-----------------------------------------
"""

    for stock in summary['bottom_5']:
        content += f"""
- {stock['ticker']}: {stock['conviction']}/100 ({stock['tier']})
  Bear Case: {stock['bear_case']}
"""

    content += """
================================================================================

HIGH CONVICTION STOCKS (Trade with larger size when signals fire)
-----------------------------------------------------------------
"""
    content += ', '.join(summary['tiers']['HIGH']) if summary['tiers']['HIGH'] else 'None'

    content += """

================================================================================

TRADING IMPLICATIONS
--------------------
1. When a signal fires for a HIGH conviction stock: Trade with 1.2-1.5x position
2. When a signal fires for a MEDIUM conviction stock: Trade with 1.0x position
3. When a signal fires for a LOW conviction stock: Trade with 0.5-0.75x position
4. When a signal fires for an AVOID stock: Skip the trade

================================================================================

Full results saved to: data/deep_dives/
Individual stock analysis files available for each ticker.

This analysis was generated automatically by the Proteus Deep Dive Agent.
"""

    return content


def main():
    """Main overnight runner."""
    print("="*70)
    print("PROTEUS OVERNIGHT DEEP DIVE RUNNER")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Stocks to analyze: {len(TICKERS)}")
    print(f"Already completed: {len(COMPLETED)}")
    print()

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load any existing results from today
    results = []
    for ticker in COMPLETED:
        filepath = os.path.join(OUTPUT_DIR, f"{ticker}_2025-12-18.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                results.append(json.load(f))
            print(f"  Loaded existing: {ticker}")

    # Run remaining tickers
    remaining = [t for t in TICKERS if t not in COMPLETED]
    print(f"\nAnalyzing {len(remaining)} remaining stocks...")
    print()

    for ticker in remaining:
        result = run_deep_dive(ticker)
        if result:
            results.append(result)
        time.sleep(1)  # Rate limiting

    # Generate summary
    print()
    print("="*70)
    print("GENERATING SUMMARY")
    print("="*70)

    summary = generate_summary(results)

    # Save summary JSON
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {SUMMARY_FILE}")

    # Generate email content
    email_content = generate_email_content(summary)

    email_file = "data/deep_dives/email_summary_2025-12-19.txt"
    with open(email_file, 'w') as f:
        f.write(email_content)
    print(f"Email content saved to: {email_file}")

    # Print summary to console
    print()
    print(email_content)

    print()
    print("="*70)
    print(f"COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    main()

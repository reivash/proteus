"""
Analyze Established AI Companies (Non-IPO Focus)

Direct analysis of established AI companies using Yahoo Finance data.
"""

import yfinance as yf
from datetime import datetime, timedelta


def analyze_stock(ticker):
    """Analyze a single stock and return key metrics."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get price history
        hist = stock.history(period="1y")
        current_price = hist['Close'].iloc[-1] if not hist.empty else info.get('currentPrice', 0)
        year_ago_price = hist['Close'].iloc[0] if not hist.empty else current_price
        ytd_return = ((current_price - year_ago_price) / year_ago_price) * 100 if year_ago_price > 0 else 0

        return {
            'ticker': ticker,
            'name': info.get('longName', ticker),
            'current_price': current_price,
            'market_cap': info.get('marketCap', 0) / 1e9,  # in billions
            'revenue': info.get('totalRevenue', 0) / 1e9,
            'revenue_growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
            'gross_margin': info.get('grossMargins', 0) * 100 if info.get('grossMargins') else 0,
            'operating_margin': info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 0,
            'forward_pe': info.get('forwardPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'ytd_return': ytd_return,
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
        }
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return None


def score_stock(data):
    """Calculate a simple score based on key metrics."""
    score = 0

    # Revenue growth (max 30 points)
    rev_growth = data['revenue_growth']
    if rev_growth >= 50:
        score += 30
    elif rev_growth >= 30:
        score += 25
    elif rev_growth >= 20:
        score += 20
    elif rev_growth >= 10:
        score += 15
    else:
        score += max(0, rev_growth / 2)

    # Gross margin (max 20 points)
    gross_margin = data['gross_margin']
    if gross_margin >= 70:
        score += 20
    elif gross_margin >= 60:
        score += 17
    elif gross_margin >= 50:
        score += 14
    else:
        score += max(0, gross_margin / 4)

    # Operating margin/profitability (max 20 points)
    op_margin = data['operating_margin']
    if op_margin >= 30:
        score += 20
    elif op_margin >= 20:
        score += 17
    elif op_margin >= 10:
        score += 14
    elif op_margin >= 5:
        score += 10
    elif op_margin > 0:
        score += 5

    # Valuation (max 15 points)
    forward_pe = data['forward_pe']
    peg = data['peg_ratio']
    if peg > 0 and peg < 0.5:
        score += 15
    elif peg > 0 and peg < 1.0:
        score += 12
    elif peg > 0 and peg < 1.5:
        score += 9
    elif forward_pe > 0 and forward_pe < 20:
        score += 12
    elif forward_pe > 0 and forward_pe < 30:
        score += 9
    elif forward_pe > 0 and forward_pe < 40:
        score += 6

    # YTD Performance (max 15 points)
    ytd_ret = data['ytd_return']
    if ytd_ret >= 100:
        score += 15
    elif ytd_ret >= 50:
        score += 12
    elif ytd_ret >= 25:
        score += 10
    elif ytd_ret >= 0:
        score += 7
    elif ytd_ret >= -20:
        score += 3

    return min(100, score)


def main():
    print("=" * 80)
    print("ESTABLISHED AI COMPANIES ANALYSIS")
    print("=" * 80)
    print()

    candidates = {
        'APP': 'AppLovin (AI Advertising)',
        'MRVL': 'Marvell Technology (AI Chips)',
        'MU': 'Micron Technology (AI Memory)',
        'QCOM': 'Qualcomm (AI Edge Devices)',
        'NOW': 'ServiceNow (Enterprise AI Software)',
        'CRM': 'Salesforce (CRM + AI)',
        'NVDA': 'NVIDIA (AI Chips - Benchmark)',
        'AMD': 'AMD (AI Chips)',
    }

    print("Fetching data for 8 companies...")
    print()

    results = []
    for ticker, description in candidates.items():
        print(f"Analyzing {ticker}...", end=" ")
        data = analyze_stock(ticker)
        if data:
            data['score'] = score_stock(data)
            data['description'] = description
            results.append(data)
            print(f"[OK] Score: {data['score']:.1f}")
        else:
            print("[FAILED]")

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)

    print("\n" + "=" * 80)
    print("DETAILED RESULTS - RANKED BY SCORE")
    print("=" * 80)

    for i, stock in enumerate(results, 1):
        print(f"\n{i}. {stock['ticker']} - {stock['name']}")
        print("=" * 80)
        print(f"Score: {stock['score']:.1f}/100")
        print(f"Description: {stock['description']}")
        print()
        print(f"Price:            ${stock['current_price']:.2f}")
        print(f"Market Cap:       ${stock['market_cap']:.1f}B")
        print(f"Revenue:          ${stock['revenue']:.1f}B")
        print(f"Revenue Growth:   {stock['revenue_growth']:.1f}% YoY")
        print(f"Gross Margin:     {stock['gross_margin']:.1f}%")
        print(f"Operating Margin: {stock['operating_margin']:.1f}%")
        print(f"Forward P/E:      {stock['forward_pe']:.1f}x")
        print(f"PEG Ratio:        {stock['peg_ratio']:.2f}" if stock['peg_ratio'] > 0 else "PEG Ratio:        N/A")
        print(f"YTD Return:       {stock['ytd_return']:+.1f}%")
        print()

        # Tier
        score = stock['score']
        if score >= 75:
            print("TIER: EXCEPTIONAL - STRONG BUY")
            rec = "Core holding, 15-25% allocation"
        elif score >= 65:
            print("TIER: VERY STRONG - BUY")
            rec = "Strong holding, 10-20% allocation"
        elif score >= 50:
            print("TIER: STRONG - BUY")
            rec = "Good holding, 5-15% allocation"
        elif score >= 35:
            print("TIER: MODERATE - HOLD")
            rec = "Monitor, consider on dips"
        else:
            print("TIER: WEAK - AVOID")
            rec = "Better opportunities available"

        print(f"RECOMMENDATION: {rec}")
        print("-" * 80)

    # Summary rankings
    print("\n" + "=" * 80)
    print("QUICK RANKINGS")
    print("=" * 80)
    print()

    for i, stock in enumerate(results, 1):
        tier = ""
        if stock['score'] >= 75:
            tier = "⭐ EXCEPTIONAL"
        elif stock['score'] >= 65:
            tier = "🔥 VERY STRONG"
        elif stock['score'] >= 50:
            tier = "✅ STRONG"
        elif stock['score'] >= 35:
            tier = "⚠️ MODERATE"
        else:
            tier = "❌ WEAK"

        print(f"{i}. {stock['ticker']:6s} - {stock['score']:5.1f}/100  {tier}")
        print(f"   {stock['name']}")
        print(f"   Price: ${stock['current_price']:.2f} | Growth: {stock['revenue_growth']:+.1f}% | Margin: {stock['operating_margin']:.1f}%")
        print()

    # Comparison
    print("=" * 80)
    print("COMPARISON VS CURRENT TOP PICKS")
    print("=" * 80)
    print()
    print("CURRENT IPO PICKS:")
    print("  PLTR - 81.8/100 (EXCEPTIONAL)")
    print("  ALAB - 70.8/100 (VERY STRONG)")
    print("  CRWV - 65.2/100 (VERY STRONG)")
    print()
    print("ESTABLISHED AI STOCKS (TOP 3):")
    for i, stock in enumerate(results[:3], 1):
        tier = ""
        if stock['score'] >= 75:
            tier = "EXCEPTIONAL"
        elif stock['score'] >= 65:
            tier = "VERY STRONG"
        elif stock['score'] >= 50:
            tier = "STRONG"
        else:
            tier = "MODERATE"
        print(f"  {stock['ticker']} - {stock['score']:.1f}/100 ({tier})")
    print()

    # Investment recommendations
    print("=" * 80)
    print("INVESTMENT IMPLICATIONS")
    print("=" * 80)
    print()

    top_picks = [s for s in results if s['score'] >= 65 and s['ticker'] != 'NVDA'][:3]
    if top_picks:
        print("NEW OPPORTUNITIES TO CONSIDER:")
        for stock in top_picks:
            print(f"\n{stock['ticker']} - ${stock['current_price']:.2f} ({stock['score']:.1f}/100)")
            print(f"  Why: {stock['revenue_growth']:.1f}% growth, {stock['operating_margin']:.1f}% margin")
            if stock['peg_ratio'] > 0 and stock['peg_ratio'] < 1.5:
                print(f"  Valuation: Attractive (PEG {stock['peg_ratio']:.2f})")
            print(f"  YTD: {stock['ytd_return']:+.1f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

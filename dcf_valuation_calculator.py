"""
DCF Valuation Calculator
=========================

Calculates intrinsic fair value for stocks using Discounted Cash Flow (DCF) model.

For each of our top 4 picks (APP, PLTR, MU, ALAB), we'll calculate:
1. Free Cash Flow (FCF) projections
2. Present value of projected cash flows
3. Terminal value
4. Fair value per share
5. Upside/downside vs current price
"""

import yfinance as yf
from typing import Dict, List, Tuple
import json


class DCFCalculator:
    """Calculates DCF valuation for a stock."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.info = self.stock.info

    def get_financials(self) -> Dict:
        """Get current financial metrics."""
        try:
            # Get key metrics
            market_cap = self.info.get('marketCap', 0)
            enterprise_value = self.info.get('enterpriseValue', 0)
            revenue = self.info.get('totalRevenue', 0)
            operating_cf = self.info.get('operatingCashflow', 0)
            capex = self.info.get('capitalExpenditures', 0)  # Usually negative
            fcf = operating_cf + capex  # capex is negative, so this is OCF - capex
            shares = self.info.get('sharesOutstanding', 0)
            current_price = self.info.get('currentPrice') or self.info.get('regularMarketPrice', 0)
            revenue_growth = self.info.get('revenueGrowth', 0) * 100  # Convert to %

            # Get debt and cash for enterprise value adjustment
            total_debt = self.info.get('totalDebt', 0)
            cash = self.info.get('totalCash', 0)
            net_debt = total_debt - cash

            return {
                'ticker': self.ticker,
                'current_price': current_price,
                'market_cap': market_cap,
                'enterprise_value': enterprise_value,
                'revenue': revenue,
                'operating_cashflow': operating_cf,
                'capex': capex,
                'fcf': fcf,
                'shares_outstanding': shares,
                'revenue_growth': revenue_growth,
                'net_debt': net_debt,
            }
        except Exception as e:
            print(f"Error getting financials for {self.ticker}: {e}")
            return {}

    def calculate_dcf(
        self,
        base_fcf: float,
        growth_rates: List[float],  # Year 1-5 growth rates
        terminal_growth: float = 2.5,  # Long-term growth (GDP-like)
        discount_rate: float = 10.0,  # WACC (cost of capital)
        net_debt: float = 0,
        shares_outstanding: float = 1,
    ) -> Dict:
        """
        Calculate DCF valuation.

        Args:
            base_fcf: Current year free cash flow
            growth_rates: List of growth rates for years 1-5 (e.g., [30, 25, 20, 15, 10])
            terminal_growth: Long-term growth rate (typically 2-3% for mature company)
            discount_rate: Discount rate / WACC (typically 8-12% for tech stocks)
            net_debt: Total debt minus cash (negative if net cash position)
            shares_outstanding: Number of shares outstanding
        """
        if base_fcf <= 0:
            return {
                'error': 'Base FCF must be positive',
                'fair_value_per_share': 0,
            }

        # Project cash flows for 5 years
        projected_fcf = []
        current_fcf = base_fcf

        for i, growth_rate in enumerate(growth_rates):
            current_fcf = current_fcf * (1 + growth_rate / 100)
            projected_fcf.append({
                'year': i + 1,
                'fcf': current_fcf,
                'growth_rate': growth_rate,
            })

        # Calculate present value of projected cash flows
        pv_fcf = []
        total_pv_fcf = 0

        for year_data in projected_fcf:
            year = year_data['year']
            fcf = year_data['fcf']
            pv = fcf / ((1 + discount_rate / 100) ** year)
            pv_fcf.append({
                'year': year,
                'fcf': fcf,
                'pv': pv,
            })
            total_pv_fcf += pv

        # Calculate terminal value
        # Terminal FCF = Last year FCF * (1 + terminal_growth)
        terminal_fcf = projected_fcf[-1]['fcf'] * (1 + terminal_growth / 100)
        # Terminal Value = Terminal FCF / (discount_rate - terminal_growth)
        terminal_value = terminal_fcf / ((discount_rate - terminal_growth) / 100)
        # Present Value of Terminal Value
        pv_terminal_value = terminal_value / ((1 + discount_rate / 100) ** len(growth_rates))

        # Total Enterprise Value = PV of FCF + PV of Terminal Value
        enterprise_value = total_pv_fcf + pv_terminal_value

        # Equity Value = Enterprise Value - Net Debt
        equity_value = enterprise_value - net_debt

        # Fair Value Per Share = Equity Value / Shares Outstanding
        fair_value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0

        return {
            'base_fcf': base_fcf,
            'projected_fcf': projected_fcf,
            'pv_fcf': pv_fcf,
            'total_pv_fcf': total_pv_fcf,
            'terminal_fcf': terminal_fcf,
            'terminal_value': terminal_value,
            'pv_terminal_value': pv_terminal_value,
            'enterprise_value': enterprise_value,
            'net_debt': net_debt,
            'equity_value': equity_value,
            'shares_outstanding': shares_outstanding,
            'fair_value_per_share': fair_value_per_share,
            'assumptions': {
                'growth_rates': growth_rates,
                'terminal_growth': terminal_growth,
                'discount_rate': discount_rate,
            }
        }

    def run_full_analysis(
        self,
        growth_rates: List[float],
        terminal_growth: float = 2.5,
        discount_rate: float = 10.0,
        custom_fcf: float = None,
    ) -> Dict:
        """
        Run complete DCF analysis with current financials.

        If custom_fcf is provided, use that instead of calculated FCF.
        """
        print(f"\n{'='*60}")
        print(f"DCF VALUATION ANALYSIS: {self.ticker}")
        print(f"{'='*60}\n")

        # Get financials
        financials = self.get_financials()

        if not financials:
            print("Error: Could not retrieve financials")
            return {}

        # Use custom FCF or calculated FCF
        base_fcf = custom_fcf if custom_fcf else financials['fcf']

        print(f"Current Financials:")
        print(f"  Market Cap:        ${financials['market_cap']/1e9:.2f}B")
        print(f"  Current Price:     ${financials['current_price']:.2f}")
        print(f"  Revenue:           ${financials['revenue']/1e9:.2f}B")
        print(f"  Operating CF:      ${financials['operating_cashflow']/1e9:.2f}B")
        print(f"  CapEx:             ${abs(financials['capex'])/1e9:.2f}B")
        print(f"  Free Cash Flow:    ${base_fcf/1e9:.2f}B")
        print(f"  Revenue Growth:    {financials['revenue_growth']:.1f}%")
        print(f"  Shares Outstanding: {financials['shares_outstanding']/1e9:.2f}B")
        print(f"  Net Debt:          ${financials['net_debt']/1e9:.2f}B")

        # Calculate DCF
        dcf_result = self.calculate_dcf(
            base_fcf=base_fcf,
            growth_rates=growth_rates,
            terminal_growth=terminal_growth,
            discount_rate=discount_rate,
            net_debt=financials['net_debt'],
            shares_outstanding=financials['shares_outstanding'],
        )

        if 'error' in dcf_result:
            print(f"\nError: {dcf_result['error']}")
            return {}

        # Print results
        print(f"\n{'='*60}")
        print("DCF ASSUMPTIONS")
        print(f"{'='*60}")
        print(f"  FCF Growth Rates: {growth_rates}")
        print(f"  Terminal Growth:  {terminal_growth}%")
        print(f"  Discount Rate:    {discount_rate}%")

        print(f"\n{'='*60}")
        print("PROJECTED FREE CASH FLOWS")
        print(f"{'='*60}")
        print(f"  Base Year (0):     ${base_fcf/1e9:.2f}B")
        for year_data in dcf_result['projected_fcf']:
            print(f"  Year {year_data['year']:1}:           ${year_data['fcf']/1e9:.2f}B (growth {year_data['growth_rate']:.0f}%)")

        print(f"\n{'='*60}")
        print("PRESENT VALUE CALCULATIONS")
        print(f"{'='*60}")
        for pv_data in dcf_result['pv_fcf']:
            print(f"  Year {pv_data['year']} PV:        ${pv_data['pv']/1e9:.2f}B")
        print(f"  Total PV of FCF:   ${dcf_result['total_pv_fcf']/1e9:.2f}B")
        print(f"\n  Terminal Value:    ${dcf_result['terminal_value']/1e9:.2f}B")
        print(f"  PV Terminal Value: ${dcf_result['pv_terminal_value']/1e9:.2f}B")

        print(f"\n{'='*60}")
        print("VALUATION RESULTS")
        print(f"{'='*60}")
        print(f"  Enterprise Value:  ${dcf_result['enterprise_value']/1e9:.2f}B")
        print(f"  Net Debt:          ${dcf_result['net_debt']/1e9:.2f}B")
        print(f"  Equity Value:      ${dcf_result['equity_value']/1e9:.2f}B")
        print(f"  Shares Outstanding: {dcf_result['shares_outstanding']/1e9:.2f}B")
        print(f"\n  FAIR VALUE PER SHARE: ${dcf_result['fair_value_per_share']:.2f}")
        print(f"  Current Price:        ${financials['current_price']:.2f}")

        # Calculate upside/downside
        upside = ((dcf_result['fair_value_per_share'] - financials['current_price']) / financials['current_price']) * 100

        print(f"\n{'='*60}")
        if upside > 0:
            print(f"  UPSIDE: +{upside:.1f}% [UNDERVALUED]")
        else:
            print(f"  DOWNSIDE: {upside:.1f}% [OVERVALUED]")
        print(f"{'='*60}\n")

        # Recommendation
        if upside > 30:
            print("[***] STRONG BUY - Significantly undervalued")
        elif upside > 15:
            print("[+] BUY - Undervalued")
        elif upside > -10:
            print("[=] HOLD - Fairly valued")
        elif upside > -25:
            print("[-] REDUCE - Overvalued")
        else:
            print("[--] SELL - Significantly overvalued")

        # Add to result
        dcf_result['current_price'] = financials['current_price']
        dcf_result['upside_pct'] = upside
        dcf_result['ticker'] = self.ticker

        return dcf_result


def analyze_all_top_picks():
    """Analyze all 4 top picks with DCF models."""

    results = {}

    print("\n" + "="*60)
    print("PROTEUS DCF VALUATION ANALYSIS")
    print("Top 4 Investment Recommendations")
    print("="*60)

    # 1. APP (AppLovin) - Score 82.0
    print("\n\n" + "#"*60)
    print("# 1. APP (APPLOVIN)")
    print("#"*60)

    app_calc = DCFCalculator('APP')
    # APP: High growth, exceptional margins
    # Assumptions: 30%, 25%, 20%, 15%, 12% growth (decelerating from 68%)
    app_result = app_calc.run_full_analysis(
        growth_rates=[30, 25, 20, 15, 12],
        terminal_growth=3.0,  # Can sustain above-GDP growth
        discount_rate=10.0,  # High quality business
    )
    results['APP'] = app_result

    # 2. MU (Micron) - Score 81.9
    print("\n\n" + "#"*60)
    print("# 2. MU (MICRON)")
    print("#"*60)

    mu_calc = DCFCalculator('MU')
    # MU: Cyclical but in upcycle, HBM sold out
    # Assumptions: 40%, 30%, 20%, 10%, 5% growth (HBM boom then normalization)
    mu_result = mu_calc.run_full_analysis(
        growth_rates=[40, 30, 20, 10, 5],
        terminal_growth=2.0,  # Cyclical, GDP-like long term
        discount_rate=11.0,  # Cyclical risk premium
    )
    results['MU'] = mu_result

    # 3. PLTR (Palantir) - Score 81.8
    print("\n\n" + "#"*60)
    print("# 3. PLTR (PALANTIR)")
    print("#"*60)

    pltr_calc = DCFCalculator('PLTR')
    # PLTR: High growth but expensive, slowing from 63%
    # Assumptions: 45%, 40%, 35%, 30%, 25% growth (AIP driving)
    pltr_result = pltr_calc.run_full_analysis(
        growth_rates=[45, 40, 35, 30, 25],
        terminal_growth=4.0,  # Can sustain high growth longer
        discount_rate=12.0,  # Higher discount for expensive stock
    )
    results['PLTR'] = pltr_result

    # 4. ALAB (Astera Labs) - Score 70.8
    print("\n\n" + "#"*60)
    print("# 4. ALAB (ASTERA LABS)")
    print("#"*60)

    alab_calc = DCFCalculator('ALAB')
    # ALAB: Very high growth (104%), but smaller company
    # Assumptions: 50%, 40%, 30%, 20%, 15% growth (infrastructure boom)
    alab_result = alab_calc.run_full_analysis(
        growth_rates=[50, 40, 30, 20, 15],
        terminal_growth=3.0,  # Above GDP as niche leader
        discount_rate=12.0,  # Smaller company, higher risk
    )
    results['ALAB'] = alab_result

    # Summary comparison
    print("\n\n" + "="*60)
    print("VALUATION SUMMARY - ALL 4 PICKS")
    print("="*60 + "\n")

    print(f"{'Ticker':<8} {'Current':<10} {'Fair Value':<12} {'Upside':<10} {'Verdict'}")
    print("-" * 60)

    for ticker in ['APP', 'MU', 'PLTR', 'ALAB']:
        if ticker in results and results[ticker]:
            r = results[ticker]
            current = r.get('current_price', 0)
            fair_value = r.get('fair_value_per_share', 0)
            upside = r.get('upside_pct', 0)

            if upside > 15:
                verdict = "BUY"
            elif upside > -10:
                verdict = "HOLD"
            else:
                verdict = "OVERVALUED"

            print(f"{ticker:<8} ${current:<9.2f} ${fair_value:<11.2f} {upside:>+6.1f}%   {verdict}")

    # Save results
    output_file = 'results/dcf_valuations.json'
    # Make serializable
    serializable_results = {}
    for ticker, data in results.items():
        if data:
            serializable_results[ticker] = {
                'fair_value_per_share': data.get('fair_value_per_share', 0),
                'current_price': data.get('current_price', 0),
                'upside_pct': data.get('upside_pct', 0),
                'enterprise_value': data.get('enterprise_value', 0),
                'equity_value': data.get('equity_value', 0),
                'assumptions': data.get('assumptions', {}),
            }

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    analyze_all_top_picks()

"""
Management Quality Assessment System
=====================================

Analyzes leadership quality and capital allocation for investment decisions.

Evaluates:
1. CEO/Executive Track Record
2. Capital Allocation (buybacks, dividends, M&A)
3. Insider Transactions (buying vs selling)
4. Compensation Alignment (pay for performance)
5. Shareholder Returns Under Current Management

Usage:
    python management_quality_assessment.py                    # Analyze all 4 picks
    python management_quality_assessment.py --ticker MU        # Analyze specific stock
    python management_quality_assessment.py --report           # Generate detailed report
"""

import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import os


class ManagementQualityAssessor:
    """Analyzes management quality and capital allocation decisions."""

    def __init__(self):
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)

        # Key management info for our top picks
        self.management_info = {
            'MU': {
                'ceo': 'Sanjay Mehrotra',
                'ceo_since': '2017-05-01',
                'previous_role': 'CEO & Co-founder of SanDisk',
                'industry_experience_years': 40,
            },
            'APP': {
                'ceo': 'Adam Foroughi',
                'ceo_since': '2012-04-01',  # Co-founder
                'previous_role': 'Co-founder',
                'industry_experience_years': 15,
            },
            'PLTR': {
                'ceo': 'Alexander Karp',
                'ceo_since': '2003-01-01',  # Co-founder
                'previous_role': 'Co-founder',
                'industry_experience_years': 22,
            },
            'ALAB': {
                'ceo': 'Jitendra Mohan',
                'ceo_since': '2017-01-01',  # Co-founder
                'previous_role': 'Co-founder, VP at Broadcom',
                'industry_experience_years': 25,
            },
        }

    def get_stock_performance_under_ceo(self, ticker: str) -> Dict:
        """Calculate stock performance since CEO took over."""
        try:
            stock = yf.Ticker(ticker)
            ceo_info = self.management_info.get(ticker, {})

            if not ceo_info:
                return {'error': 'No CEO data available'}

            ceo_start_date = ceo_info['ceo_since']
            hist = stock.history(period='max')

            if len(hist) == 0:
                return {'error': 'No historical data'}

            # Find price when CEO started (or closest after IPO)
            ceo_start = datetime.strptime(ceo_start_date, '%Y-%m-%d')
            hist_start = hist.index[0].to_pydatetime().replace(tzinfo=None)

            # If CEO started before IPO, use IPO price
            if ceo_start < hist_start:
                start_idx = 0
                actual_start_date = hist_start
                note = f"CEO started before IPO; using IPO date {hist_start.strftime('%Y-%m-%d')}"
            else:
                # Find closest date after CEO start
                hist_dates = [d.to_pydatetime().replace(tzinfo=None) for d in hist.index]
                start_idx = next((i for i, d in enumerate(hist_dates) if d >= ceo_start), 0)
                actual_start_date = hist_dates[start_idx]
                note = f"CEO started {ceo_start_date}"

            start_price = hist['Close'].iloc[start_idx]
            current_price = hist['Close'].iloc[-1]

            # Calculate returns
            total_return = ((current_price - start_price) / start_price) * 100

            # Calculate years
            years_as_ceo = (datetime.now() - actual_start_date).days / 365.25
            annualized_return = ((current_price / start_price) ** (1 / years_as_ceo) - 1) * 100 if years_as_ceo > 0 else 0

            # Calculate vs S&P 500 (simplified - using SPY as proxy)
            try:
                spy = yf.Ticker('SPY')
                spy_hist = spy.history(period='max')

                # Find SPY price at same start date
                spy_dates = [d.to_pydatetime().replace(tzinfo=None) for d in spy_hist.index]
                spy_start_idx = next((i for i, d in enumerate(spy_dates) if d >= actual_start_date), 0)

                spy_start = spy_hist['Close'].iloc[spy_start_idx]
                spy_current = spy_hist['Close'].iloc[-1]
                spy_return = ((spy_current - spy_start) / spy_start) * 100
                spy_annual = ((spy_current / spy_start) ** (1 / years_as_ceo) - 1) * 100 if years_as_ceo > 0 else 0

                outperformance = total_return - spy_return
                annual_alpha = annualized_return - spy_annual

            except:
                spy_return = 0
                spy_annual = 0
                outperformance = 0
                annual_alpha = 0

            return {
                'ceo': ceo_info['ceo'],
                'tenure_years': round(years_as_ceo, 1),
                'start_date': actual_start_date.strftime('%Y-%m-%d'),
                'start_price': round(start_price, 2),
                'current_price': round(current_price, 2),
                'total_return': round(total_return, 1),
                'annualized_return': round(annualized_return, 1),
                'sp500_return': round(spy_return, 1),
                'sp500_annual': round(spy_annual, 1),
                'outperformance': round(outperformance, 1),
                'annual_alpha': round(annual_alpha, 1),
                'note': note,
            }

        except Exception as e:
            return {'error': str(e)}

    def analyze_capital_allocation(self, ticker: str) -> Dict:
        """Analyze how management allocates capital (buybacks, dividends, M&A)."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            cashflow = stock.cashflow

            # Get basic info
            market_cap = info.get('marketCap', 0)

            # Dividends
            dividend_rate = info.get('dividendRate', 0)
            dividend_yield = info.get('dividendYield', 0)
            payout_ratio = info.get('payoutRatio', 0)

            # Share buybacks (from cash flow statement)
            buyback_analysis = {}
            if not cashflow.empty and 'Repurchase Of Capital Stock' in cashflow.index:
                buybacks = cashflow.loc['Repurchase Of Capital Stock']
                # Buybacks are negative in cash flow statement
                recent_buyback = abs(buybacks.iloc[0]) if len(buybacks) > 0 else 0
                total_buybacks_3yr = abs(buybacks.iloc[:3].sum()) if len(buybacks) >= 3 else 0

                buyback_analysis = {
                    'recent_buyback': recent_buyback,
                    'total_3yr': total_buybacks_3yr,
                    'buyback_yield': (recent_buyback / market_cap * 100) if market_cap > 0 else 0,
                }
            else:
                buyback_analysis = {
                    'recent_buyback': 0,
                    'total_3yr': 0,
                    'buyback_yield': 0,
                }

            # Free cash flow (for evaluating capital allocation capacity)
            fcf_analysis = {}
            if not cashflow.empty and 'Free Cash Flow' in cashflow.index:
                fcf = cashflow.loc['Free Cash Flow']
                recent_fcf = fcf.iloc[0] if len(fcf) > 0 else 0
                avg_fcf_3yr = fcf.iloc[:3].mean() if len(fcf) >= 3 else 0

                # Calculate what % of FCF goes to shareholders (dividends + buybacks)
                total_returned = (dividend_rate * info.get('sharesOutstanding', 0)) + buyback_analysis['recent_buyback']
                fcf_return_rate = (total_returned / recent_fcf * 100) if recent_fcf > 0 else 0

                fcf_analysis = {
                    'recent_fcf': recent_fcf,
                    'avg_fcf_3yr': avg_fcf_3yr,
                    'fcf_return_rate': fcf_return_rate,
                }
            else:
                fcf_analysis = {
                    'recent_fcf': 0,
                    'avg_fcf_3yr': 0,
                    'fcf_return_rate': 0,
                }

            # Debt management
            total_debt = info.get('totalDebt', 0)
            total_cash = info.get('totalCash', 0)
            net_debt = total_debt - total_cash
            debt_to_equity = info.get('debtToEquity', 0)

            # Capital allocation score (0-100)
            score = 0
            reasons = []

            # 1. Returns cash to shareholders (30 points)
            total_yield = (dividend_yield or 0) * 100 + buyback_analysis['buyback_yield']
            if total_yield > 5:
                score += 30
                reasons.append(f"Excellent shareholder returns ({total_yield:.1f}% total yield)")
            elif total_yield > 2:
                score += 20
                reasons.append(f"Good shareholder returns ({total_yield:.1f}% total yield)")
            elif total_yield > 0:
                score += 10
                reasons.append(f"Some shareholder returns ({total_yield:.1f}% total yield)")
            else:
                reasons.append("No dividends or buybacks (growth focus or concern?)")

            # 2. FCF generation (30 points)
            if fcf_analysis['recent_fcf'] > 0:
                score += 30
                reasons.append(f"Positive FCF (${fcf_analysis['recent_fcf']/1e9:.1f}B)")
            elif fcf_analysis['recent_fcf'] > -1e9:
                score += 15
                reasons.append("Near breakeven FCF")
            else:
                reasons.append(f"Negative FCF (${fcf_analysis['recent_fcf']/1e9:.1f}B)")

            # 3. Balance sheet management (25 points)
            if net_debt < 0:  # Net cash position
                score += 25
                reasons.append(f"Net cash position (${abs(net_debt)/1e9:.1f}B)")
            elif debt_to_equity < 50:
                score += 20
                reasons.append(f"Conservative debt (D/E {debt_to_equity:.1f})")
            elif debt_to_equity < 100:
                score += 10
                reasons.append(f"Moderate debt (D/E {debt_to_equity:.1f})")
            else:
                reasons.append(f"High debt (D/E {debt_to_equity:.1f})")

            # 4. Efficient capital returns (15 points)
            if fcf_analysis['fcf_return_rate'] > 75:
                score += 15
                reasons.append(f"Returns {fcf_analysis['fcf_return_rate']:.0f}% of FCF to shareholders")
            elif fcf_analysis['fcf_return_rate'] > 50:
                score += 10
                reasons.append(f"Returns {fcf_analysis['fcf_return_rate']:.0f}% of FCF to shareholders")
            elif fcf_analysis['fcf_return_rate'] > 25:
                score += 5
                reasons.append(f"Returns {fcf_analysis['fcf_return_rate']:.0f}% of FCF to shareholders")
            else:
                reasons.append("Retains most/all FCF (growth investment or inefficiency?)")

            return {
                'capital_allocation_score': score,
                'dividend_yield': round((dividend_yield or 0) * 100, 2),
                'dividend_rate': round(dividend_rate, 2),
                'payout_ratio': round((payout_ratio or 0) * 100, 1),
                'buyback_yield': round(buyback_analysis['buyback_yield'], 2),
                'total_shareholder_yield': round(total_yield, 2),
                'recent_fcf_b': round(fcf_analysis['recent_fcf'] / 1e9, 2),
                'fcf_return_rate': round(fcf_analysis['fcf_return_rate'], 1),
                'net_debt_b': round(net_debt / 1e9, 2),
                'debt_to_equity': round(debt_to_equity, 1),
                'reasons': reasons,
            }

        except Exception as e:
            return {'error': str(e)}

    def analyze_insider_transactions(self, ticker: str) -> Dict:
        """Analyze insider buying/selling patterns."""
        try:
            stock = yf.Ticker(ticker)

            # Get insider transactions
            # Note: yfinance provides limited insider data
            insider_transactions = stock.insider_transactions

            if insider_transactions is None or insider_transactions.empty:
                return {
                    'insider_score': 50,  # Neutral if no data
                    'note': 'No insider transaction data available from yfinance',
                    'recommendation': 'Check SEC Form 4 filings manually',
                }

            # Analyze last 6 months of transactions
            six_months_ago = datetime.now() - timedelta(days=180)
            recent = insider_transactions[insider_transactions.index > six_months_ago]

            if recent.empty:
                return {
                    'insider_score': 50,
                    'note': 'No recent insider transactions in last 6 months',
                }

            # Count buys vs sells
            buys = recent[recent['Shares'] > 0] if 'Shares' in recent.columns else []
            sells = recent[recent['Shares'] < 0] if 'Shares' in recent.columns else []

            buy_count = len(buys)
            sell_count = len(sells)

            # Calculate score (0-100)
            if buy_count > sell_count * 2:
                score = 90
                signal = "STRONG BUY signal (insiders accumulating)"
            elif buy_count > sell_count:
                score = 75
                signal = "Positive signal (more buys than sells)"
            elif buy_count == sell_count:
                score = 50
                signal = "Neutral (balanced buying/selling)"
            elif sell_count > buy_count * 2:
                score = 25
                signal = "Caution (heavy insider selling)"
            else:
                score = 40
                signal = "Slightly negative (more sells than buys)"

            return {
                'insider_score': score,
                'buy_transactions': buy_count,
                'sell_transactions': sell_count,
                'signal': signal,
                'period': '6 months',
            }

        except Exception as e:
            # If no insider data available, return neutral score
            return {
                'insider_score': 50,
                'note': 'Limited insider data available',
                'recommendation': 'Check SEC Form 4 filings manually at sec.gov',
            }

    def assess_compensation_alignment(self, ticker: str) -> Dict:
        """Analyze if executive compensation aligns with shareholder returns."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get stock performance (1 year)
            hist = stock.history(period='1y')
            if len(hist) < 2:
                return {'error': 'Insufficient historical data'}

            year_return = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100

            # Note: yfinance doesn't provide detailed compensation data
            # We can only make general assessments based on governance scores

            # Get some proxy metrics
            insider_ownership = info.get('heldPercentInsiders', 0) * 100
            institutional_ownership = info.get('heldPercentInstitutions', 0) * 100

            # Score based on insider ownership (higher = more aligned)
            score = 0
            reasons = []

            if insider_ownership > 10:
                score += 40
                reasons.append(f"High insider ownership ({insider_ownership:.1f}%) - strong alignment")
            elif insider_ownership > 5:
                score += 30
                reasons.append(f"Good insider ownership ({insider_ownership:.1f}%)")
            elif insider_ownership > 1:
                score += 20
                reasons.append(f"Moderate insider ownership ({insider_ownership:.1f}%)")
            else:
                score += 10
                reasons.append(f"Low insider ownership ({insider_ownership:.1f}%)")

            # Institutional ownership (suggests professional investors trust management)
            if institutional_ownership > 80:
                score += 30
                reasons.append(f"Very high institutional ownership ({institutional_ownership:.1f}%)")
            elif institutional_ownership > 60:
                score += 25
                reasons.append(f"High institutional ownership ({institutional_ownership:.1f}%)")
            elif institutional_ownership > 40:
                score += 15
                reasons.append(f"Moderate institutional ownership ({institutional_ownership:.1f}%)")
            else:
                reasons.append(f"Low institutional ownership ({institutional_ownership:.1f}%)")

            # Performance-based bonus
            if year_return > 50:
                score += 30
                reasons.append(f"Exceptional 1-year return ({year_return:.1f}%) - compensation likely deserved")
            elif year_return > 20:
                score += 20
                reasons.append(f"Strong 1-year return ({year_return:.1f}%)")
            elif year_return > 0:
                score += 10
                reasons.append(f"Positive 1-year return ({year_return:.1f}%)")
            else:
                reasons.append(f"Negative 1-year return ({year_return:.1f}%) - check if comp reduced")

            return {
                'compensation_alignment_score': score,
                'insider_ownership_pct': round(insider_ownership, 2),
                'institutional_ownership_pct': round(institutional_ownership, 2),
                'year_return_pct': round(year_return, 1),
                'reasons': reasons,
                'note': 'Limited compensation data from yfinance - check proxy statements (DEF 14A) for details',
            }

        except Exception as e:
            return {'error': str(e)}

    def generate_overall_assessment(self, ticker: str) -> Dict:
        """Generate comprehensive management quality assessment."""
        print(f"\nAnalyzing management quality for {ticker}...")

        # Get all components
        performance = self.get_stock_performance_under_ceo(ticker)
        capital_allocation = self.analyze_capital_allocation(ticker)
        insider_activity = self.analyze_insider_transactions(ticker)
        compensation = self.assess_compensation_alignment(ticker)

        # Calculate overall score (weighted average)
        scores = {
            'performance': performance.get('annual_alpha', 0),  # vs S&P 500
            'capital_allocation': capital_allocation.get('capital_allocation_score', 0),
            'insider': insider_activity.get('insider_score', 50),
            'compensation': compensation.get('compensation_alignment_score', 50),
        }

        # Weights
        weights = {
            'performance': 0.40,  # CEO performance is most important
            'capital_allocation': 0.30,  # How they use capital
            'insider': 0.15,  # What insiders are doing
            'compensation': 0.15,  # Alignment
        }

        # Performance score (convert alpha to 0-100 scale)
        # +20% annual alpha = 100, 0% = 50, -20% = 0
        perf_score = max(0, min(100, 50 + scores['performance'] * 2.5))

        overall_score = (
            perf_score * weights['performance'] +
            scores['capital_allocation'] * weights['capital_allocation'] +
            scores['insider'] * weights['insider'] +
            scores['compensation'] * weights['compensation']
        )

        # Rating
        if overall_score >= 80:
            rating = "EXCELLENT"
            recommendation = "Management is exceptional - high confidence in capital allocation"
        elif overall_score >= 65:
            rating = "GOOD"
            recommendation = "Management is above average - trust with capital allocation"
        elif overall_score >= 50:
            rating = "AVERAGE"
            recommendation = "Management is competent but not exceptional"
        elif overall_score >= 35:
            rating = "BELOW AVERAGE"
            recommendation = "Management concerns - monitor closely"
        else:
            rating = "POOR"
            recommendation = "Serious management concerns - consider avoiding"

        return {
            'ticker': ticker,
            'overall_score': round(overall_score, 1),
            'rating': rating,
            'recommendation': recommendation,
            'component_scores': {
                'performance': round(perf_score, 1),
                'capital_allocation': scores['capital_allocation'],
                'insider': scores['insider'],
                'compensation': scores['compensation'],
            },
            'performance_analysis': performance,
            'capital_allocation_analysis': capital_allocation,
            'insider_analysis': insider_activity,
            'compensation_analysis': compensation,
        }

    def analyze_all_picks(self) -> Dict:
        """Analyze management quality for all top picks."""
        results = {}

        tickers = ['MU', 'APP', 'PLTR', 'ALAB']

        for ticker in tickers:
            try:
                results[ticker] = self.generate_overall_assessment(ticker)
            except Exception as e:
                results[ticker] = {'error': str(e)}

        return results

    def display_results(self, results: Dict):
        """Display management quality assessment results."""
        print("\n" + "="*80)
        print("MANAGEMENT QUALITY ASSESSMENT")
        print("="*80)

        for ticker, data in results.items():
            if 'error' in data:
                print(f"\n{ticker}: ERROR - {data['error']}")
                continue

            print(f"\n{ticker} - {data['rating']} ({data['overall_score']}/100)")
            print("-" * 80)

            # CEO Performance
            perf = data['performance_analysis']
            if 'error' not in perf:
                print(f"\n1. CEO PERFORMANCE (Score: {data['component_scores']['performance']}/100)")
                print(f"   CEO: {perf['ceo']} (Tenure: {perf['tenure_years']} years)")
                print(f"   Total Return: {perf['total_return']:+.1f}% ({perf['annualized_return']:+.1f}% annual)")
                print(f"   vs S&P 500: {perf['outperformance']:+.1f}% total ({perf['annual_alpha']:+.1f}% annual alpha)")

                if perf['annual_alpha'] > 15:
                    print(f"   [EXCELLENT] Massive outperformance vs market")
                elif perf['annual_alpha'] > 5:
                    print(f"   [GOOD] Solid outperformance vs market")
                elif perf['annual_alpha'] > -5:
                    print(f"   [AVERAGE] Roughly in line with market")
                else:
                    print(f"   [POOR] Underperforming market significantly")

            # Capital Allocation
            cap = data['capital_allocation_analysis']
            if 'error' not in cap:
                print(f"\n2. CAPITAL ALLOCATION (Score: {data['component_scores']['capital_allocation']}/100)")
                print(f"   Total Shareholder Yield: {cap['total_shareholder_yield']:.2f}%")
                print(f"     - Dividend Yield: {cap['dividend_yield']:.2f}%")
                print(f"     - Buyback Yield: {cap['buyback_yield']:.2f}%")
                print(f"   Free Cash Flow: ${cap['recent_fcf_b']:.1f}B")
                print(f"   FCF Returned to Shareholders: {cap['fcf_return_rate']:.1f}%")
                print(f"   Net Debt: ${cap['net_debt_b']:.1f}B (D/E: {cap['debt_to_equity']:.1f})")

                for reason in cap['reasons'][:3]:
                    print(f"   - {reason}")

            # Insider Activity
            insider = data['insider_analysis']
            print(f"\n3. INSIDER ACTIVITY (Score: {data['component_scores']['insider']}/100)")
            if 'signal' in insider:
                print(f"   Signal: {insider['signal']}")
                print(f"   Recent Activity: {insider['buy_transactions']} buys, {insider['sell_transactions']} sells")
            else:
                print(f"   Note: {insider.get('note', 'No data available')}")
                if 'recommendation' in insider:
                    print(f"   Recommendation: {insider['recommendation']}")

            # Compensation Alignment
            comp = data['compensation_analysis']
            if 'error' not in comp:
                print(f"\n4. COMPENSATION ALIGNMENT (Score: {data['component_scores']['compensation']}/100)")
                print(f"   Insider Ownership: {comp['insider_ownership_pct']:.2f}%")
                print(f"   Institutional Ownership: {comp['institutional_ownership_pct']:.1f}%")
                print(f"   1-Year Return: {comp['year_return_pct']:+.1f}%")

                for reason in comp['reasons'][:2]:
                    print(f"   - {reason}")

            # Overall Recommendation
            print(f"\n[{data['rating']}] {data['recommendation']}")
            print("\n" + "="*80)

        # Summary
        print("\nSUMMARY - MANAGEMENT QUALITY RANKINGS")
        print("="*80)

        ranked = sorted(results.items(),
                       key=lambda x: x[1].get('overall_score', 0),
                       reverse=True)

        for i, (ticker, data) in enumerate(ranked, 1):
            if 'overall_score' in data:
                score = data['overall_score']
                rating = data['rating']

                # Get CEO performance
                perf = data.get('performance_analysis', {})
                alpha = perf.get('annual_alpha', 0)

                print(f"{i}. {ticker:6} | Score: {score:5.1f}/100 | {rating:15} | "
                      f"CEO Alpha: {alpha:+6.1f}%/year")

        print("="*80)

    def save_results(self, results: Dict, filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"management_quality_assessment_{timestamp}.json"

        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {filepath}")
        return filepath


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Proteus Management Quality Assessment')
    parser.add_argument('--ticker', type=str, help='Analyze specific ticker')
    parser.add_argument('--report', action='store_true', help='Generate detailed report file')

    args = parser.parse_args()

    assessor = ManagementQualityAssessor()

    if args.ticker:
        # Analyze single ticker
        results = {args.ticker: assessor.generate_overall_assessment(args.ticker.upper())}
    else:
        # Analyze all picks
        results = assessor.analyze_all_picks()

    # Display results
    assessor.display_results(results)

    # Save results
    assessor.save_results(results)

    print("\nManagement Quality Assessment Complete!")
    print("\nNext Steps:")
    print("1. Review CEO track records and tenure")
    print("2. Check capital allocation efficiency")
    print("3. Monitor insider transactions via SEC Form 4")
    print("4. Review proxy statements (DEF 14A) for detailed compensation")


if __name__ == "__main__":
    main()

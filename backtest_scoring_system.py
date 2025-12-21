"""
Backtesting Framework for IPO Scoring System
==============================================

Tests whether our scoring methodology successfully identifies winners
by analyzing historical performance of high-scoring vs low-scoring stocks.

Strategy:
1. Get current scores for all companies
2. Get historical prices (6 months ago, 1 year ago)
3. Calculate actual returns
4. Analyze correlation between scores and returns
5. Validate scoring system effectiveness
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

# Our 22 tracked companies with current scores (from recent analysis)
COMPANIES = {
    # EXCEPTIONAL (80+)
    'APP': 82.0,    # AppLovin
    'PLTR': 81.8,   # Palantir
    'MU': 81.9,     # Micron
    'NVDA': 80.0,   # NVIDIA

    # VERY STRONG (65-79)
    'ALAB': 70.8,   # Astera Labs
    'CRWV': 66.4,   # CoreWeave
    'MRVL': 65.4,   # Marvell
    'QCOM': 65.0,   # Qualcomm
    'AMD': 65.0,    # AMD

    # STRONG (50-64)
    'TEM': 64.5,    # Tempus AI
    'ARM': 62.5,    # ARM Holdings
    'RBRK': 60.9,   # Rubrik
    'IONQ': 59.7,   # IonQ
    'NOW': 54.0,    # ServiceNow
    'DDOG': 52.6,   # Datadog
    'CRM': 50.9,    # Salesforce

    # MODERATE (35-49)
    'ZS': 49.9,     # Zscaler
    'FIG': 49.6,    # Figma
    'CRWD': 48.6,   # CrowdStrike

    # WEAK (<35)
    'RDDT': 30.2,   # Reddit
    'FOUR': 28.4,   # Shift4 Payments
    'BBAI': 22.2,   # BigBear.ai
}

class BacktestAnalyzer:
    """Analyzes historical performance to validate scoring system."""

    def __init__(self):
        self.results = {}

    def get_historical_returns(self, ticker: str, months_back: int) -> Tuple[float, bool]:
        """
        Calculate returns from N months ago to today.
        Returns (return_pct, data_available)
        """
        try:
            stock = yf.Ticker(ticker)

            # Calculate dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months_back * 30 + 30)  # Extra buffer

            # Get historical data
            hist = stock.history(start=start_date, end=end_date)

            if hist.empty or len(hist) < 2:
                return 0.0, False

            # Get price from N months ago (using index position instead of date arithmetic)
            # This avoids timezone issues
            days_target = months_back * 30
            target_idx = max(0, len(hist) - days_target - 1)

            if target_idx >= len(hist) - 1:
                return 0.0, False

            start_price = hist['Close'].iloc[target_idx]
            current_price = hist['Close'].iloc[-1]

            return_pct = ((current_price - start_price) / start_price) * 100

            return return_pct, True

        except Exception as e:
            # print(f"Error getting data for {ticker}: {e}")
            return 0.0, False

    def run_backtest(self, months_back: int = 6) -> Dict:
        """
        Run backtest for specified period.

        Returns dictionary with:
        - Performance by score tier
        - Correlation between score and returns
        - Top performers vs scores
        - System validation metrics
        """
        print(f"\n{'='*60}")
        print(f"BACKTESTING SCORING SYSTEM - {months_back} Month Period")
        print(f"{'='*60}\n")

        results = []

        # Get returns for each company
        for ticker, score in COMPANIES.items():
            return_pct, available = self.get_historical_returns(ticker, months_back)

            if available:
                results.append({
                    'ticker': ticker,
                    'score': score,
                    'return': return_pct,
                    'tier': self._get_tier(score)
                })
                print(f"{ticker:6} | Score: {score:5.1f} | Return: {return_pct:7.1f}% | {'Available' if available else 'No Data'}")
            else:
                print(f"{ticker:6} | Score: {score:5.1f} | No historical data (recent IPO)")

        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)

        if df.empty:
            print("\nNo data available for backtesting")
            return {}

        # Analyze by tier
        tier_performance = df.groupby('tier').agg({
            'return': ['mean', 'median', 'count'],
            'score': 'mean'
        }).round(2)

        # Calculate correlation
        correlation = df['score'].corr(df['return'])

        # Get top performers
        top_10_actual = df.nlargest(10, 'return')[['ticker', 'score', 'return']]
        top_10_scored = df.nlargest(10, 'score')[['ticker', 'score', 'return']]

        # Calculate validation metrics
        high_scorers = df[df['score'] >= 70]  # VERY STRONG or better
        avg_high_score_return = high_scorers['return'].mean() if len(high_scorers) > 0 else 0

        low_scorers = df[df['score'] < 50]  # MODERATE or worse
        avg_low_score_return = low_scorers['return'].mean() if len(low_scorers) > 0 else 0

        # Market benchmark (SPY)
        spy_return, spy_available = self.get_historical_returns('SPY', months_back)

        results_dict = {
            'period_months': months_back,
            'tier_performance': tier_performance.to_dict(),
            'correlation': correlation,
            'top_10_actual': top_10_actual.to_dict('records'),
            'top_10_scored': top_10_scored.to_dict('records'),
            'high_scorers_return': avg_high_score_return,
            'low_scorers_return': avg_low_score_return,
            'market_return': spy_return if spy_available else None,
            'total_companies': len(df),
        }

        # Print summary
        print(f"\n{'='*60}")
        print("BACKTEST RESULTS SUMMARY")
        print(f"{'='*60}\n")

        print(f"Period: {months_back} months")
        print(f"Companies analyzed: {len(df)}")
        print(f"Score-Return Correlation: {correlation:.3f}")
        print(f"\nAverage Returns by Tier:")
        print(tier_performance)

        print(f"\n{'='*60}")
        print("VALIDATION METRICS")
        print(f"{'='*60}")
        print(f"High Scorers (70+) Avg Return: {avg_high_score_return:.1f}%")
        print(f"Low Scorers (<50) Avg Return:  {avg_low_score_return:.1f}%")
        if spy_available:
            print(f"S&P 500 (SPY) Return:          {spy_return:.1f}%")
            print(f"High Scorers vs SPY:           {avg_high_score_return - spy_return:+.1f}%")

        print(f"\n{'='*60}")
        print("TOP 10 BY ACTUAL RETURNS")
        print(f"{'='*60}")
        for i, row in top_10_actual.iterrows():
            print(f"{i+1:2}. {row['ticker']:6} | Score: {row['score']:5.1f} | Return: {row['return']:7.1f}%")

        print(f"\n{'='*60}")
        print("TOP 10 BY SCORE (Our Picks)")
        print(f"{'='*60}")
        for i, row in top_10_scored.iterrows():
            print(f"{i+1:2}. {row['ticker']:6} | Score: {row['score']:5.1f} | Return: {row['return']:7.1f}%")

        # Validation assessment
        print(f"\n{'='*60}")
        print("SYSTEM VALIDATION")
        print(f"{'='*60}\n")

        if correlation > 0.5:
            print("[+] STRONG VALIDATION: Score correlates well with returns (r > 0.5)")
        elif correlation > 0.3:
            print("[+] MODERATE VALIDATION: Score shows positive correlation (r > 0.3)")
        elif correlation > 0:
            print("[!] WEAK VALIDATION: Score shows weak correlation (r > 0)")
        else:
            print("[-] POOR VALIDATION: Negative correlation - system needs revision")

        if avg_high_score_return > avg_low_score_return:
            outperformance = avg_high_score_return - avg_low_score_return
            print(f"[+] High scorers outperformed low scorers by {outperformance:.1f}%")
        else:
            print("[-] High scorers did NOT outperform low scorers")

        if spy_available and avg_high_score_return > spy_return:
            alpha = avg_high_score_return - spy_return
            print(f"[+] High scorers beat market by {alpha:.1f}% (alpha generation)")
        elif spy_available:
            print("[-] High scorers did NOT beat market")

        # Overall verdict
        validation_score = 0
        if correlation > 0.3: validation_score += 1
        if avg_high_score_return > avg_low_score_return: validation_score += 1
        if spy_available and avg_high_score_return > spy_return: validation_score += 1

        print(f"\nValidation Score: {validation_score}/3")
        if validation_score == 3:
            print("[***] SYSTEM VALIDATED - Scoring methodology works!")
        elif validation_score == 2:
            print("[+] SYSTEM PARTIALLY VALIDATED - Generally effective")
        elif validation_score == 1:
            print("[!] SYSTEM NEEDS IMPROVEMENT - Weak predictive power")
        else:
            print("[-] SYSTEM FAILED VALIDATION - Requires major revision")

        return results_dict

    def _get_tier(self, score: float) -> str:
        """Get tier classification from score."""
        if score >= 80: return "EXCEPTIONAL"
        elif score >= 65: return "VERY STRONG"
        elif score >= 50: return "STRONG"
        elif score >= 35: return "MODERATE"
        else: return "WEAK"

    def run_multi_period_backtest(self) -> Dict:
        """Run backtests for multiple periods to validate consistency."""
        periods = [3, 6, 12]  # months
        multi_results = {}

        for months in periods:
            print(f"\n\n{'#'*60}")
            print(f"# BACKTESTING: {months} MONTH PERIOD")
            print(f"{'#'*60}")

            results = self.run_backtest(months)
            multi_results[f"{months}_months"] = results

        # Cross-period analysis
        print(f"\n\n{'='*60}")
        print("CROSS-PERIOD VALIDATION")
        print(f"{'='*60}\n")

        correlations = [multi_results[f"{m}_months"].get('correlation', 0) for m in periods if f"{m}_months" in multi_results]
        avg_correlation = sum(correlations) / len(correlations) if correlations else 0

        print(f"Average Correlation Across Periods: {avg_correlation:.3f}")

        if avg_correlation > 0.4:
            print("[+] STRONG: System consistently predicts winners across time periods")
        elif avg_correlation > 0.2:
            print("[+] MODERATE: System shows consistent positive correlation")
        else:
            print("[!] WEAK: System lacks consistent predictive power")

        return multi_results


def main():
    """Run comprehensive backtest."""
    print("\n" + "="*60)
    print("PROTEUS IPO SCORING SYSTEM - BACKTEST VALIDATION")
    print("="*60)
    print("\nTesting whether our scoring methodology successfully")
    print("identifies stocks that will outperform the market.\n")
    print("Current Scoring Factors:")
    print("- 40% Base YC Score (growth, margins, profitability)")
    print("- 30% Competitive Moat (TAM, differentiation)")
    print("- 30% Growth Trajectory (acceleration, momentum)")
    print("="*60)

    analyzer = BacktestAnalyzer()

    # Run multi-period backtest
    results = analyzer.run_multi_period_backtest()

    # Save results
    output_file = 'results/backtest_results.json'
    with open(output_file, 'w') as f:
        # Convert any non-serializable objects
        serializable_results = {}
        for period, data in results.items():
            serializable_results[period] = {k: v for k, v in data.items() if k != 'tier_performance'}
        json.dump(serializable_results, f, indent=2)

    print(f"\n\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")

    print("\nNEXT STEPS:")
    print("1. Review backtest results above")
    print("2. If validation score < 2/3, consider adjusting scoring weights")
    print("3. Proceed to DCF valuation models for top picks")
    print("4. Add technical analysis layer\n")


if __name__ == "__main__":
    main()

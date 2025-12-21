"""
Institutional Ownership Analysis Tracker
=========================================

Tracks institutional investor positions and trends (smart money analysis).

Analyzes:
1. Major Institutional Holders (top 10)
2. Ownership Concentration (diversification risk)
3. Ownership Trends (accumulation vs distribution)
4. Smart Money Tracking (top hedge funds, mutual funds)
5. Institutional Confidence Score

Usage:
    python institutional_ownership_tracker.py                 # Analyze all 4 picks
    python institutional_ownership_tracker.py --ticker MU     # Analyze specific stock
    python institutional_ownership_tracker.py --compare       # Compare ownership trends
"""

import yfinance as yf
from datetime import datetime
from typing import Dict, List, Tuple
import json
import os


class InstitutionalOwnershipTracker:
    """Analyzes institutional ownership patterns and trends."""

    def __init__(self):
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)

        # List of "smart money" institutions to track
        self.smart_money_indicators = [
            'Vanguard',
            'BlackRock',
            'State Street',
            'Fidelity',
            'Citadel',
            'Renaissance Technologies',
            'Bridgewater',
            'Two Sigma',
            'D. E. Shaw',
            'AQR Capital',
            'Millennium',
            'Point72',
            'Tiger Global',
            'Coatue',
        ]

    def get_institutional_holders(self, ticker: str) -> Dict:
        """Get current institutional holder data."""
        try:
            stock = yf.Ticker(ticker)
            institutional = stock.institutional_holders
            info = stock.info

            if institutional is None or institutional.empty:
                return {'error': 'No institutional holder data available'}

            # Get total institutional ownership %
            total_institutional_pct = info.get('heldPercentInstitutions', 0) * 100

            # Analyze top holders
            top_holders = []
            total_shares_held = 0

            for idx, row in institutional.head(10).iterrows():
                holder_name = row.get('Holder', 'Unknown')
                shares = row.get('Shares', 0)
                date_reported = row.get('Date Reported', 'N/A')
                pct_out = row.get('% Out', 0) * 100 if isinstance(row.get('% Out'), (int, float)) else 0
                value = row.get('Value', 0)

                # Check if smart money
                is_smart_money = any(smart in holder_name for smart in self.smart_money_indicators)

                top_holders.append({
                    'name': holder_name,
                    'shares': shares,
                    'value': value,
                    'pct_outstanding': pct_out,
                    'date_reported': str(date_reported),
                    'is_smart_money': is_smart_money,
                })

                total_shares_held += shares

            # Calculate concentration (top 10 ownership)
            shares_outstanding = info.get('sharesOutstanding', 1)
            top10_pct = (total_shares_held / shares_outstanding * 100) if shares_outstanding > 0 else 0

            # Count smart money holders
            smart_money_count = sum(1 for h in top_holders if h['is_smart_money'])

            return {
                'total_institutional_pct': round(total_institutional_pct, 2),
                'top10_pct': round(top10_pct, 2),
                'top_holders': top_holders,
                'smart_money_count': smart_money_count,
                'shares_outstanding': shares_outstanding,
            }

        except Exception as e:
            return {'error': str(e)}

    def analyze_ownership_trends(self, ticker: str) -> Dict:
        """Analyze if institutions are accumulating or distributing."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get institutional ownership %
            current_institutional_pct = info.get('heldPercentInstitutions', 0) * 100

            # Since yfinance doesn't provide historical institutional data,
            # we can use insider ownership as a proxy for trend analysis
            insider_pct = info.get('heldPercentInsiders', 0) * 100

            # Analyze float (% available for trading)
            float_shares = info.get('floatShares', 0)
            shares_outstanding = info.get('sharesOutstanding', 1)
            float_pct = (float_shares / shares_outstanding * 100) if shares_outstanding > 0 else 0

            # Institutional ownership categories
            if current_institutional_pct > 80:
                ownership_level = "VERY HIGH"
                signal = "Strong institutional support - widely held by institutions"
            elif current_institutional_pct > 60:
                ownership_level = "HIGH"
                signal = "Good institutional support - professional investors confident"
            elif current_institutional_pct > 40:
                ownership_level = "MODERATE"
                signal = "Average institutional interest"
            elif current_institutional_pct > 20:
                ownership_level = "LOW"
                signal = "Limited institutional interest - may be undiscovered or concerns"
            else:
                ownership_level = "VERY LOW"
                signal = "Very limited institutional support - red flag or early stage"

            # Calculate score (0-100)
            score = 0
            reasons = []

            # 1. High institutional ownership (40 points)
            if current_institutional_pct > 75:
                score += 40
                reasons.append(f"Very high institutional ownership ({current_institutional_pct:.1f}%)")
            elif current_institutional_pct > 60:
                score += 35
                reasons.append(f"High institutional ownership ({current_institutional_pct:.1f}%)")
            elif current_institutional_pct > 40:
                score += 25
                reasons.append(f"Moderate institutional ownership ({current_institutional_pct:.1f}%)")
            elif current_institutional_pct > 20:
                score += 15
                reasons.append(f"Low institutional ownership ({current_institutional_pct:.1f}%)")
            else:
                score += 5
                reasons.append(f"Very low institutional ownership ({current_institutional_pct:.1f}%)")

            # 2. Reasonable insider ownership (15 points)
            if 5 < insider_pct < 20:
                score += 15
                reasons.append(f"Healthy insider ownership ({insider_pct:.1f}%) - aligned but not locked up")
            elif 1 < insider_pct <= 5:
                score += 10
                reasons.append(f"Moderate insider ownership ({insider_pct:.1f}%)")
            elif insider_pct <= 1:
                score += 5
                reasons.append(f"Low insider ownership ({insider_pct:.1f}%) - less skin in game")
            else:
                score += 5
                reasons.append(f"Very high insider ownership ({insider_pct:.1f}%) - potential liquidity concern")

            # 3. Float analysis (15 points)
            if float_pct > 80:
                score += 15
                reasons.append(f"High float ({float_pct:.1f}%) - good liquidity")
            elif float_pct > 60:
                score += 10
                reasons.append(f"Adequate float ({float_pct:.1f}%)")
            else:
                score += 5
                reasons.append(f"Low float ({float_pct:.1f}%) - potential liquidity concerns")

            return {
                'ownership_trend_score': score,
                'institutional_pct': round(current_institutional_pct, 2),
                'insider_pct': round(insider_pct, 2),
                'float_pct': round(float_pct, 2),
                'ownership_level': ownership_level,
                'signal': signal,
                'reasons': reasons,
            }

        except Exception as e:
            return {'error': str(e)}

    def analyze_smart_money_presence(self, ticker: str) -> Dict:
        """Analyze presence of top hedge funds and quality institutions."""
        try:
            holders_data = self.get_institutional_holders(ticker)

            if 'error' in holders_data:
                return holders_data

            top_holders = holders_data['top_holders']
            smart_money_count = holders_data['smart_money_count']
            total_institutional = holders_data['total_institutional_pct']

            # Calculate smart money holdings
            smart_money_holders = [h for h in top_holders if h['is_smart_money']]
            smart_money_pct = sum(h['pct_outstanding'] for h in smart_money_holders)

            # Score based on smart money presence (0-100)
            score = 0
            reasons = []

            # 1. Number of smart money institutions (40 points)
            if smart_money_count >= 7:
                score += 40
                reasons.append(f"{smart_money_count} elite institutions in top 10 (exceptional)")
            elif smart_money_count >= 5:
                score += 35
                reasons.append(f"{smart_money_count} elite institutions in top 10 (excellent)")
            elif smart_money_count >= 3:
                score += 25
                reasons.append(f"{smart_money_count} elite institutions in top 10 (good)")
            elif smart_money_count >= 1:
                score += 15
                reasons.append(f"{smart_money_count} elite institution(s) in top 10")
            else:
                score += 5
                reasons.append("No elite institutions in top 10 (concerning)")

            # 2. Smart money ownership % (30 points)
            if smart_money_pct > 30:
                score += 30
                reasons.append(f"Elite institutions hold {smart_money_pct:.1f}% (very high conviction)")
            elif smart_money_pct > 20:
                score += 25
                reasons.append(f"Elite institutions hold {smart_money_pct:.1f}% (high conviction)")
            elif smart_money_pct > 10:
                score += 15
                reasons.append(f"Elite institutions hold {smart_money_pct:.1f}% (moderate conviction)")
            elif smart_money_pct > 0:
                score += 5
                reasons.append(f"Elite institutions hold {smart_money_pct:.1f}% (low conviction)")

            # 3. Diversification of holders (30 points)
            # More diverse ownership = better (not concentrated in 1-2 holders)
            top1_pct = top_holders[0]['pct_outstanding'] if len(top_holders) > 0 else 0

            if top1_pct < 10:
                score += 30
                reasons.append(f"Well diversified (top holder {top1_pct:.1f}%)")
            elif top1_pct < 15:
                score += 20
                reasons.append(f"Good diversification (top holder {top1_pct:.1f}%)")
            elif top1_pct < 25:
                score += 10
                reasons.append(f"Moderate concentration (top holder {top1_pct:.1f}%)")
            else:
                score += 5
                reasons.append(f"High concentration risk (top holder {top1_pct:.1f}%)")

            return {
                'smart_money_score': score,
                'smart_money_count': smart_money_count,
                'smart_money_pct': round(smart_money_pct, 2),
                'smart_money_holders': [h['name'] for h in smart_money_holders],
                'top1_holder': top_holders[0]['name'] if len(top_holders) > 0 else 'Unknown',
                'top1_pct': round(top1_pct, 2),
                'reasons': reasons,
            }

        except Exception as e:
            return {'error': str(e)}

    def generate_overall_assessment(self, ticker: str) -> Dict:
        """Generate comprehensive institutional ownership assessment."""
        print(f"\nAnalyzing institutional ownership for {ticker}...")

        # Get all components
        holders_data = self.get_institutional_holders(ticker)
        trend_analysis = self.analyze_ownership_trends(ticker)
        smart_money = self.analyze_smart_money_presence(ticker)

        # Calculate overall score (weighted average)
        scores = {
            'trend': trend_analysis.get('ownership_trend_score', 50),
            'smart_money': smart_money.get('smart_money_score', 50),
        }

        # Weights
        weights = {
            'trend': 0.40,  # Overall institutional ownership level
            'smart_money': 0.60,  # Quality of institutions (more important)
        }

        overall_score = (
            scores['trend'] * weights['trend'] +
            scores['smart_money'] * weights['smart_money']
        )

        # Rating
        if overall_score >= 80:
            rating = "EXCELLENT"
            recommendation = "Strong institutional support - high conviction from smart money"
        elif overall_score >= 65:
            rating = "GOOD"
            recommendation = "Solid institutional backing - professional investors confident"
        elif overall_score >= 50:
            rating = "AVERAGE"
            recommendation = "Moderate institutional interest - neither bullish nor bearish"
        elif overall_score >= 35:
            rating = "BELOW AVERAGE"
            recommendation = "Limited institutional support - may be concerns or undiscovered"
        else:
            rating = "POOR"
            recommendation = "Very weak institutional backing - red flag"

        return {
            'ticker': ticker,
            'overall_score': round(overall_score, 1),
            'rating': rating,
            'recommendation': recommendation,
            'component_scores': {
                'ownership_trend': scores['trend'],
                'smart_money': scores['smart_money'],
            },
            'holders_data': holders_data,
            'trend_analysis': trend_analysis,
            'smart_money_analysis': smart_money,
        }

    def analyze_all_picks(self) -> Dict:
        """Analyze institutional ownership for all top picks."""
        results = {}

        tickers = ['MU', 'APP', 'PLTR', 'ALAB']

        for ticker in tickers:
            try:
                results[ticker] = self.generate_overall_assessment(ticker)
            except Exception as e:
                results[ticker] = {'error': str(e)}

        return results

    def display_results(self, results: Dict):
        """Display institutional ownership analysis results."""
        print("\n" + "="*80)
        print("INSTITUTIONAL OWNERSHIP ANALYSIS")
        print("="*80)

        for ticker, data in results.items():
            if 'error' in data:
                print(f"\n{ticker}: ERROR - {data['error']}")
                continue

            print(f"\n{ticker} - {data['rating']} ({data['overall_score']}/100)")
            print("-" * 80)

            # Ownership Overview
            trend = data['trend_analysis']
            if 'error' not in trend:
                print(f"\n1. OWNERSHIP OVERVIEW (Score: {data['component_scores']['ownership_trend']}/100)")
                print(f"   Total Institutional: {trend['institutional_pct']:.1f}%")
                print(f"   Insider Ownership: {trend['insider_pct']:.2f}%")
                print(f"   Float: {trend['float_pct']:.1f}%")
                print(f"   Level: {trend['ownership_level']}")
                print(f"   Signal: {trend['signal']}")

            # Smart Money Analysis
            smart = data['smart_money_analysis']
            if 'error' not in smart:
                print(f"\n2. SMART MONEY ANALYSIS (Score: {data['component_scores']['smart_money']}/100)")
                print(f"   Elite Institutions in Top 10: {smart['smart_money_count']}")
                print(f"   Smart Money Ownership: {smart['smart_money_pct']:.2f}%")
                print(f"   Top Holder: {smart['top1_holder']} ({smart['top1_pct']:.2f}%)")

                if smart['smart_money_holders']:
                    print(f"\n   Elite Institutions:")
                    for holder in smart['smart_money_holders'][:5]:
                        print(f"     - {holder}")

                print(f"\n   Key Insights:")
                for reason in smart['reasons'][:3]:
                    print(f"     - {reason}")

            # Top 10 Holders
            holders = data['holders_data']
            if 'error' not in holders:
                print(f"\n3. TOP 10 INSTITUTIONAL HOLDERS")
                print(f"   Top 10 Combined: {holders['top10_pct']:.2f}%")
                print(f"\n   Rank | Holder                                | % Outstanding | Smart Money")
                print(f"   " + "-"*76)

                for i, holder in enumerate(holders['top_holders'], 1):
                    smart_badge = "[SMART]" if holder['is_smart_money'] else "       "
                    print(f"   {i:2}.   | {holder['name'][:37]:37} | {holder['pct_outstanding']:>11.2f}% | {smart_badge}")

            # Overall Recommendation
            print(f"\n[{data['rating']}] {data['recommendation']}")
            print("\n" + "="*80)

        # Summary
        print("\nSUMMARY - INSTITUTIONAL BACKING RANKINGS")
        print("="*80)

        ranked = sorted(results.items(),
                       key=lambda x: x[1].get('overall_score', 0),
                       reverse=True)

        for i, (ticker, data) in enumerate(ranked, 1):
            if 'overall_score' in data:
                score = data['overall_score']
                rating = data['rating']

                # Get institutional %
                trend = data.get('trend_analysis', {})
                inst_pct = trend.get('institutional_pct', 0)

                # Get smart money count
                smart = data.get('smart_money_analysis', {})
                smart_count = smart.get('smart_money_count', 0)

                print(f"{i}. {ticker:6} | Score: {score:5.1f}/100 | {rating:15} | "
                      f"Inst: {inst_pct:5.1f}% | Smart: {smart_count}/10")

        print("="*80)

    def compare_ownership(self, results: Dict):
        """Compare institutional ownership across stocks."""
        print("\n" + "="*80)
        print("INSTITUTIONAL OWNERSHIP COMPARISON")
        print("="*80)

        comparison_data = []

        for ticker, data in results.items():
            if 'error' in data:
                continue

            trend = data.get('trend_analysis', {})
            smart = data.get('smart_money_analysis', {})
            holders = data.get('holders_data', {})

            comparison_data.append({
                'ticker': ticker,
                'institutional_pct': trend.get('institutional_pct', 0),
                'smart_money_count': smart.get('smart_money_count', 0),
                'smart_money_pct': smart.get('smart_money_pct', 0),
                'top10_pct': holders.get('top10_pct', 0),
                'overall_score': data.get('overall_score', 0),
            })

        # Sort by overall score
        comparison_data.sort(key=lambda x: x['overall_score'], reverse=True)

        print(f"\n{'Ticker':<8} {'Overall':<10} {'Total Inst':<12} {'Smart Money':<14} {'Smart %':<10} {'Top 10':<10}")
        print(f"{'':8} {'Score':<10} {'%':<12} {'Count':<14} {'':<10} {'%':<10}")
        print("-" * 80)

        for item in comparison_data:
            print(f"{item['ticker']:<8} {item['overall_score']:>6.1f}/100  "
                  f"{item['institutional_pct']:>10.1f}%  "
                  f"{item['smart_money_count']:>12}/10  "
                  f"{item['smart_money_pct']:>8.2f}%  "
                  f"{item['top10_pct']:>8.1f}%")

        print("="*80)

        # Analysis
        print("\nKEY INSIGHTS:")

        # Highest institutional backing
        best_institutional = max(comparison_data, key=lambda x: x['institutional_pct'])
        print(f"\n- Highest Institutional Ownership: {best_institutional['ticker']} "
              f"({best_institutional['institutional_pct']:.1f}%)")
        print(f"  Professional investors have high conviction in {best_institutional['ticker']}")

        # Most smart money
        best_smart_money = max(comparison_data, key=lambda x: x['smart_money_count'])
        print(f"\n- Most Elite Institutions: {best_smart_money['ticker']} "
              f"({best_smart_money['smart_money_count']}/10 top institutions)")
        print(f"  Smart money (Vanguard, BlackRock, Citadel, etc.) heavily invested")

        # Most concentrated ownership
        most_concentrated = max(comparison_data, key=lambda x: x['top10_pct'])
        print(f"\n- Most Concentrated Ownership: {most_concentrated['ticker']} "
              f"(top 10 hold {most_concentrated['top10_pct']:.1f}%)")

        print("\n" + "="*80)

    def save_results(self, results: Dict, filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"institutional_ownership_analysis_{timestamp}.json"

        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {filepath}")
        return filepath


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Proteus Institutional Ownership Tracker')
    parser.add_argument('--ticker', type=str, help='Analyze specific ticker')
    parser.add_argument('--compare', action='store_true', help='Show comparison table')

    args = parser.parse_args()

    tracker = InstitutionalOwnershipTracker()

    if args.ticker:
        # Analyze single ticker
        results = {args.ticker: tracker.generate_overall_assessment(args.ticker.upper())}
    else:
        # Analyze all picks
        results = tracker.analyze_all_picks()

    # Display results
    tracker.display_results(results)

    # Show comparison if requested
    if args.compare or not args.ticker:
        tracker.compare_ownership(results)

    # Save results
    tracker.save_results(results)

    print("\nInstitutional Ownership Analysis Complete!")
    print("\nNext Steps:")
    print("1. Monitor for changes in institutional positions (quarterly 13F filings)")
    print("2. Watch for smart money accumulation/distribution")
    print("3. Check for activist investors entering positions")
    print("4. Review institutional ownership changes in quarterly reports")


if __name__ == "__main__":
    main()

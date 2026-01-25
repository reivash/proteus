"""
Macro Economic Factor Analysis
==============================

Analyzes macroeconomic factors impacting investment recommendations.

Evaluates:
1. Federal Reserve Policy (interest rates, monetary stance)
2. Economic Cycle Position (recession risk, growth outlook)
3. Sector Tailwinds/Headwinds (AI spending, data center growth)
4. Geopolitical Risks (US-China, Taiwan, trade)
5. Currency & Inflation Impacts

Usage:
    python macro_analysis.py                    # Full macro analysis
    python macro_analysis.py --ticker MU        # Sector-specific analysis
    python macro_analysis.py --report           # Generate detailed report
"""

import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List
import json
import os


class MacroAnalyzer:
    """Analyzes macroeconomic factors and sector trends."""

    def __init__(self):
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)

        # Current macro assessment (as of Dec 2025)
        self.macro_context = {
            'date': '2025-12-01',
            'fed_funds_rate': 4.50,  # Approximate current rate
            'inflation_trend': 'MODERATING',
            'economic_growth': 'POSITIVE',
            'unemployment': 'LOW',
            'recession_risk': 'LOW',
        }

        # Sector-specific tailwinds
        self.sector_tailwinds = {
            'semiconductors': {
                'AI_spending': {
                    'trend': 'ACCELERATING',
                    'duration': '5-10 years',
                    'impact': 'VERY POSITIVE',
                    'notes': 'AI/ML driving unprecedented chip demand (GPUs, HBM, custom silicon)',
                },
                'data_center_growth': {
                    'trend': 'STRONG',
                    'duration': '3-7 years',
                    'impact': 'POSITIVE',
                    'notes': 'Cloud migration + AI workloads driving data center expansion',
                },
                'memory_upcycle': {
                    'trend': 'EARLY STAGE',
                    'duration': '2-3 years',
                    'impact': 'POSITIVE',
                    'notes': 'HBM shortage, DRAM pricing recovery, inventory normalization',
                },
                'supply_constraints': {
                    'trend': 'EASING',
                    'duration': '1-2 years',
                    'impact': 'NEUTRAL',
                    'notes': 'Capacity additions coming online, but HBM still sold out',
                },
            },
        }

    def analyze_fed_policy_impact(self, ticker: str) -> Dict:
        """Analyze how Federal Reserve policy impacts the stock."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get sector
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')

            # Interest rate sensitivity
            # Semiconductors: MODERATE sensitivity (capex-heavy, but secular growth)
            # Software: LOW sensitivity (high margins, low debt)

            if 'Semiconductor' in industry or 'Technology Hardware' in sector:
                rate_sensitivity = "MODERATE"
                impact_notes = [
                    "Capex-intensive industry (high rates = higher cost of capital)",
                    "However, secular AI tailwind overrides rate concerns",
                    "Rate cuts would be positive but not required for growth",
                ]
                current_impact = "NEUTRAL TO SLIGHTLY POSITIVE"
            elif 'Software' in industry:
                rate_sensitivity = "LOW"
                impact_notes = [
                    "High gross margins (60-80%) reduce sensitivity",
                    "SaaS revenue models provide stability",
                    "Rate cuts positive for multiple expansion",
                ]
                current_impact = "NEUTRAL TO POSITIVE"
            else:
                rate_sensitivity = "MODERATE"
                impact_notes = ["General technology sector sensitivity"]
                current_impact = "NEUTRAL"

            # Current Fed stance (as of Dec 2025)
            fed_stance = "NEUTRAL TO ACCOMMODATIVE"
            fed_notes = [
                "Fed Funds Rate: ~4.50% (down from 5.50% peak in 2023)",
                "Rate cuts likely complete or nearly complete",
                "Inflation moderating toward 2% target",
                "Focus on maintaining growth while controlling inflation",
            ]

            # Score (0-100)
            # Higher = more favorable macro environment
            score = 0
            reasons = []

            # 1. Fed policy stance (40 points)
            # Accommodative = better for stocks
            score += 30
            reasons.append("Fed in neutral/accommodative mode (supportive)")

            # 2. Rate trajectory (30 points)
            # Stable or declining = positive
            score += 25
            reasons.append("Rate cuts complete/near complete (reduces uncertainty)")

            # 3. Inflation control (30 points)
            score += 25
            reasons.append("Inflation moderating (no emergency tightening needed)")

            return {
                'fed_policy_score': score,
                'rate_sensitivity': rate_sensitivity,
                'current_impact': current_impact,
                'fed_stance': fed_stance,
                'fed_notes': fed_notes,
                'impact_notes': impact_notes,
                'reasons': reasons,
            }

        except Exception as e:
            return {'error': str(e)}

    def analyze_economic_cycle(self) -> Dict:
        """Analyze current economic cycle position."""
        # Use market indices as proxies
        try:
            # Get S&P 500
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(period='1y')

            # Calculate 1-year return
            year_return = ((spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[0]) /
                          spy_hist['Close'].iloc[0]) * 100

            # Get volatility (VIX proxy)
            try:
                vix = yf.Ticker('^VIX')
                vix_hist = vix.history(period='1mo')
                current_vix = vix_hist['Close'].iloc[-1] if len(vix_hist) > 0 else 15
            except:
                current_vix = 15  # Default neutral

            # Assess cycle
            if year_return > 20:
                cycle_phase = "EXPANSION"
                risk_level = "LOW TO MODERATE"
                notes = "Strong market returns suggest healthy economic expansion"
            elif year_return > 10:
                cycle_phase = "MODERATE GROWTH"
                risk_level = "MODERATE"
                notes = "Positive but moderate growth"
            elif year_return > 0:
                cycle_phase = "LATE CYCLE OR EARLY RECOVERY"
                risk_level = "MODERATE"
                notes = "Slower growth, monitor for signs of weakness"
            elif year_return > -10:
                cycle_phase = "SLOWDOWN"
                risk_level = "ELEVATED"
                notes = "Market weakness suggests economic concerns"
            else:
                cycle_phase = "RECESSION OR BEAR MARKET"
                risk_level = "HIGH"
                notes = "Significant market decline, recession risk high"

            # VIX assessment
            if current_vix < 15:
                volatility_regime = "LOW (complacent)"
            elif current_vix < 20:
                volatility_regime = "NORMAL (healthy)"
            elif current_vix < 30:
                volatility_regime = "ELEVATED (caution)"
            else:
                volatility_regime = "HIGH (fear/panic)"

            # Score (0-100)
            score = 0
            reasons = []

            if year_return > 15:
                score += 40
                reasons.append(f"Strong market returns (+{year_return:.1f}% YTD)")
            elif year_return > 5:
                score += 30
                reasons.append(f"Positive market returns (+{year_return:.1f}% YTD)")
            elif year_return > 0:
                score += 20
                reasons.append(f"Slightly positive returns (+{year_return:.1f}% YTD)")
            else:
                score += 10
                reasons.append(f"Negative market returns ({year_return:.1f}% YTD)")

            if current_vix < 20:
                score += 30
                reasons.append(f"Low volatility (VIX {current_vix:.1f})")
            elif current_vix < 25:
                score += 20
                reasons.append(f"Moderate volatility (VIX {current_vix:.1f})")
            else:
                score += 10
                reasons.append(f"High volatility (VIX {current_vix:.1f})")

            # Current assessment (Dec 2025)
            score += 20  # Assuming continued economic strength
            reasons.append("No recession signals, unemployment low")

            return {
                'economic_cycle_score': score,
                'cycle_phase': cycle_phase,
                'risk_level': risk_level,
                'sp500_return_ytd': round(year_return, 1),
                'vix_level': round(current_vix, 1),
                'volatility_regime': volatility_regime,
                'notes': notes,
                'reasons': reasons,
            }

        except Exception as e:
            return {'error': str(e)}

    def analyze_sector_tailwinds(self, ticker: str) -> Dict:
        """Analyze sector-specific tailwinds for the stock."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            industry = info.get('industry', '')
            sector = info.get('sector', '')

            # Determine relevant tailwinds
            tailwinds = []
            headwinds = []
            score = 0

            if 'Semiconductor' in industry:
                # Semiconductor-specific analysis
                tailwinds.extend([
                    {
                        'factor': 'AI & ML Chip Demand',
                        'strength': 'VERY STRONG',
                        'duration': '5-10 years',
                        'impact': '+40 points',
                        'notes': 'Training & inference driving GPU/HBM/custom silicon demand',
                    },
                    {
                        'factor': 'Data Center Expansion',
                        'strength': 'STRONG',
                        'duration': '3-7 years',
                        'impact': '+25 points',
                        'notes': 'Cloud migration + AI workloads = capacity additions',
                    },
                    {
                        'factor': 'Memory Upcycle (HBM)',
                        'strength': 'STRONG',
                        'duration': '2-3 years',
                        'impact': '+20 points',
                        'notes': 'HBM sold out through 2026, pricing power returning',
                    },
                ])

                headwinds.extend([
                    {
                        'factor': 'Cyclicality Risk',
                        'strength': 'MODERATE',
                        'duration': 'Structural',
                        'impact': '-10 points',
                        'notes': 'Semiconductors historically cyclical (but AI secular)',
                    },
                ])

                score += 40 + 25 + 20 - 10  # Net: 75
                sector_outlook = "VERY POSITIVE (secular AI tailwinds override cyclical concerns)"

            elif 'Software' in industry:
                # Software-specific analysis
                tailwinds.extend([
                    {
                        'factor': 'Digital Transformation',
                        'strength': 'STRONG',
                        'duration': '5+ years',
                        'impact': '+30 points',
                        'notes': 'Enterprises accelerating cloud/SaaS adoption',
                    },
                    {
                        'factor': 'AI Integration into Software',
                        'strength': 'EMERGING',
                        'duration': '3-10 years',
                        'impact': '+20 points',
                        'notes': 'Copilot features, automation driving value',
                    },
                ])

                headwinds.extend([
                    {
                        'factor': 'Valuation Pressure',
                        'strength': 'MODERATE',
                        'duration': 'Ongoing',
                        'impact': '-15 points',
                        'notes': 'High multiples make stocks rate-sensitive',
                    },
                ])

                score += 30 + 20 - 15  # Net: 35
                sector_outlook = "POSITIVE (steady growth but valuation-dependent)"

            else:
                # Generic technology
                tailwinds.append({
                    'factor': 'Technology Adoption',
                    'strength': 'MODERATE',
                    'duration': '5+ years',
                    'impact': '+25 points',
                    'notes': 'General technology adoption trends',
                })

                score += 25
                sector_outlook = "NEUTRAL TO POSITIVE"

            return {
                'sector_tailwinds_score': score,
                'sector_outlook': sector_outlook,
                'tailwinds': tailwinds,
                'headwinds': headwinds,
                'net_score': score,
            }

        except Exception as e:
            return {'error': str(e)}

    def analyze_geopolitical_risks(self, ticker: str) -> Dict:
        """Analyze geopolitical risks specific to the stock."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            industry = info.get('industry', '')

            risks = []
            score = 70  # Start neutral

            if 'Semiconductor' in industry:
                # Major geopolitical risks for semiconductors
                risks.extend([
                    {
                        'risk': 'US-China Tech Decoupling',
                        'severity': 'MODERATE',
                        'impact': '-10 points',
                        'mitigation': 'US companies benefit from export controls on China',
                        'notes': 'Restricts China access to advanced chips, benefits US companies like MU',
                    },
                    {
                        'risk': 'Taiwan Geopolitical Tension',
                        'severity': 'LOW TO MODERATE',
                        'impact': '-10 points',
                        'mitigation': 'Diversified manufacturing (US CHIPS Act, Japan, Singapore)',
                        'notes': 'Taiwan risk exists but diversification underway',
                    },
                    {
                        'risk': 'Export Control Escalation',
                        'severity': 'LOW',
                        'impact': '-5 points',
                        'mitigation': 'US companies already comply, Chinese competitors restricted',
                        'notes': 'Actually benefits US companies vs Chinese competition',
                    },
                ])

                score -= 25  # Net geopolitical drag
                geo_outlook = "MANAGEABLE RISKS (US positioning advantageous)"

            else:
                # Software/other
                risks.append({
                    'risk': 'Regulatory Risk',
                    'severity': 'LOW',
                    'impact': '-5 points',
                    'mitigation': 'Compliance programs in place',
                    'notes': 'Tech regulation ongoing but manageable',
                })

                score -= 5
                geo_outlook = "LOW RISK"

            return {
                'geopolitical_risk_score': score,
                'geo_outlook': geo_outlook,
                'risks': risks,
                'net_impact': score - 70,  # vs neutral baseline
            }

        except Exception as e:
            return {'error': str(e)}

    def generate_overall_macro_assessment(self, ticker: str = None) -> Dict:
        """Generate comprehensive macro analysis."""
        if ticker:
            print(f"\nAnalyzing macro environment for {ticker}...")
        else:
            print("\nAnalyzing overall macro environment...")

        # Get all components
        economic_cycle = self.analyze_economic_cycle()

        if ticker:
            fed_policy = self.analyze_fed_policy_impact(ticker)
            sector_tailwinds = self.analyze_sector_tailwinds(ticker)
            geopolitical = self.analyze_geopolitical_risks(ticker)

            # Calculate overall score (weighted average)
            scores = {
                'cycle': economic_cycle.get('economic_cycle_score', 50),
                'fed': fed_policy.get('fed_policy_score', 50),
                'sector': sector_tailwinds.get('sector_tailwinds_score', 50),
                'geo': geopolitical.get('geopolitical_risk_score', 50),
            }

            # Weights
            weights = {
                'cycle': 0.25,
                'fed': 0.20,
                'sector': 0.40,  # Most important - secular trends
                'geo': 0.15,
            }

            overall_score = (
                scores['cycle'] * weights['cycle'] +
                scores['fed'] * weights['fed'] +
                scores['sector'] * weights['sector'] +
                scores['geo'] * weights['geo']
            )

            # Rating
            if overall_score >= 75:
                rating = "VERY FAVORABLE"
                recommendation = "Macro environment strongly supports investment thesis"
            elif overall_score >= 60:
                rating = "FAVORABLE"
                recommendation = "Macro environment supports investment, minor headwinds"
            elif overall_score >= 45:
                rating = "NEUTRAL"
                recommendation = "Mixed macro signals, focus on company fundamentals"
            elif overall_score >= 30:
                rating = "UNFAVORABLE"
                recommendation = "Macro headwinds present, proceed with caution"
            else:
                rating = "VERY UNFAVORABLE"
                recommendation = "Significant macro risks, consider waiting"

            return {
                'ticker': ticker,
                'overall_score': round(overall_score, 1),
                'rating': rating,
                'recommendation': recommendation,
                'component_scores': {
                    'economic_cycle': scores['cycle'],
                    'fed_policy': scores['fed'],
                    'sector_tailwinds': scores['sector'],
                    'geopolitical': scores['geo'],
                },
                'economic_cycle_analysis': economic_cycle,
                'fed_policy_analysis': fed_policy,
                'sector_analysis': sector_tailwinds,
                'geopolitical_analysis': geopolitical,
            }

        else:
            # Just economic cycle (no ticker-specific)
            return {
                'overall_score': economic_cycle.get('economic_cycle_score', 50),
                'rating': "GENERAL MARKET ANALYSIS",
                'economic_cycle_analysis': economic_cycle,
            }

    def analyze_all_picks(self) -> Dict:
        """Analyze macro environment for all top picks."""
        results = {}

        tickers = ['MU', 'APP', 'PLTR', 'ALAB']

        for ticker in tickers:
            try:
                results[ticker] = self.generate_overall_macro_assessment(ticker)
            except Exception as e:
                results[ticker] = {'error': str(e)}

        return results

    def display_results(self, results: Dict):
        """Display macro analysis results."""
        print("\n" + "="*80)
        print("MACRO ECONOMIC FACTOR ANALYSIS")
        print("="*80)

        for ticker, data in results.items():
            if 'error' in data:
                print(f"\n{ticker}: ERROR - {data['error']}")
                continue

            print(f"\n{ticker} - {data['rating']} ({data['overall_score']}/100)")
            print("-" * 80)

            # Economic Cycle
            cycle = data['economic_cycle_analysis']
            if 'error' not in cycle:
                print(f"\n1. ECONOMIC CYCLE (Score: {data['component_scores']['economic_cycle']}/100)")
                print(f"   Phase: {cycle['cycle_phase']}")
                print(f"   Risk Level: {cycle['risk_level']}")
                print(f"   S&P 500 YTD: {cycle['sp500_return_ytd']:+.1f}%")
                print(f"   VIX: {cycle['vix_level']:.1f} ({cycle['volatility_regime']})")
                print(f"   Assessment: {cycle['notes']}")

            # Fed Policy
            fed = data.get('fed_policy_analysis', {})
            if 'error' not in fed:
                print(f"\n2. FED POLICY (Score: {data['component_scores']['fed_policy']}/100)")
                print(f"   Stance: {fed['fed_stance']}")
                print(f"   Rate Sensitivity: {fed['rate_sensitivity']}")
                print(f"   Current Impact: {fed['current_impact']}")

                print(f"\n   Key Notes:")
                for note in fed['fed_notes'][:3]:
                    print(f"     - {note}")

            # Sector Tailwinds
            sector = data.get('sector_analysis', {})
            if 'error' not in sector:
                print(f"\n3. SECTOR TAILWINDS (Score: {data['component_scores']['sector_tailwinds']}/100)")
                print(f"   Outlook: {sector['sector_outlook']}")

                if sector['tailwinds']:
                    print(f"\n   Tailwinds:")
                    for tw in sector['tailwinds']:
                        print(f"     [+] {tw['factor']} ({tw['strength']})")
                        print(f"         {tw['notes']}")

                if sector['headwinds']:
                    print(f"\n   Headwinds:")
                    for hw in sector['headwinds']:
                        print(f"     [-] {hw['factor']} ({hw['strength']})")
                        print(f"         {hw['notes']}")

            # Geopolitical
            geo = data.get('geopolitical_analysis', {})
            if 'error' not in geo:
                print(f"\n4. GEOPOLITICAL FACTORS (Score: {data['component_scores']['geopolitical']}/100)")
                print(f"   Outlook: {geo['geo_outlook']}")

                if geo['risks']:
                    print(f"\n   Key Risks:")
                    for risk in geo['risks']:
                        print(f"     - {risk['risk']} ({risk['severity']})")
                        print(f"       Mitigation: {risk['mitigation']}")

            # Overall Recommendation
            print(f"\n[{data['rating']}] {data['recommendation']}")
            print("\n" + "="*80)

        # Summary
        print("\nSUMMARY - MACRO FAVORABILITY RANKINGS")
        print("="*80)

        ranked = sorted(results.items(),
                       key=lambda x: x[1].get('overall_score', 0),
                       reverse=True)

        for i, (ticker, data) in enumerate(ranked, 1):
            if 'overall_score' in data:
                score = data['overall_score']
                rating = data['rating']

                # Get sector score (most important)
                sector_score = data['component_scores'].get('sector_tailwinds', 0)

                print(f"{i}. {ticker:6} | Score: {score:5.1f}/100 | {rating:18} | "
                      f"Sector: {sector_score:3.0f}/100")

        print("="*80)

    def save_results(self, results: Dict, filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"macro_analysis_{timestamp}.json"

        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {filepath}")
        return filepath


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Proteus Macro Economic Analysis')
    parser.add_argument('--ticker', type=str, help='Analyze specific ticker')
    parser.add_argument('--report', action='store_true', help='Generate detailed report file')

    args = parser.parse_args()

    analyzer = MacroAnalyzer()

    if args.ticker:
        # Analyze single ticker
        results = {args.ticker: analyzer.generate_overall_macro_assessment(args.ticker.upper())}
    else:
        # Analyze all picks
        results = analyzer.analyze_all_picks()

    # Display results
    analyzer.display_results(results)

    # Save results
    analyzer.save_results(results)

    print("\nMacro Economic Analysis Complete!")
    print("\nKey Takeaways:")
    print("1. Monitor Fed policy changes (rate decisions)")
    print("2. Track economic indicators (GDP, unemployment, inflation)")
    print("3. Watch sector-specific trends (AI spending, data center growth)")
    print("4. Stay informed on geopolitical developments (US-China, Taiwan)")


if __name__ == "__main__":
    main()

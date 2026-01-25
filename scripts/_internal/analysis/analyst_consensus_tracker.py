"""
Analyst Consensus Tracker
==========================

Systematically monitors Wall Street analyst opinions for our top picks.

Tracks:
1. Analyst Ratings (Buy/Hold/Sell distribution)
2. Price Targets (high/low/median/mean)
3. Recent Changes (upgrades/downgrades)
4. Earnings Estimates (revisions)
5. Consensus Trends (strengthening/weakening)
"""

import yfinance as yf
from datetime import datetime
from typing import Dict, List
import json


class AnalystTracker:
    """Tracks analyst consensus for a stock."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.info = self.stock.info

    def get_analyst_ratings(self) -> Dict:
        """Get current analyst ratings distribution."""
        try:
            # Get recommendations
            recommendations = self.stock.recommendations

            if recommendations is None or recommendations.empty:
                return {'error': 'No analyst data available'}

            # Get most recent period
            latest = recommendations.tail(5)  # Last 5 periods for trend

            # Current ratings
            current = latest.iloc[-1] if len(latest) > 0 else None

            if current is None:
                return {'error': 'No current ratings'}

            # Extract ratings
            strong_buy = current.get('strongBuy', 0)
            buy = current.get('buy', 0)
            hold = current.get('hold', 0)
            sell = current.get('sell', 0)
            strong_sell = current.get('strongSell', 0)

            total_analysts = strong_buy + buy + hold + sell + strong_sell

            if total_analysts == 0:
                return {'error': 'No analyst coverage'}

            # Calculate percentages
            ratings = {
                'strong_buy': strong_buy,
                'buy': buy,
                'hold': hold,
                'sell': sell,
                'strong_sell': strong_sell,
                'total': total_analysts,
                'strong_buy_pct': (strong_buy / total_analysts) * 100,
                'buy_pct': (buy / total_analysts) * 100,
                'hold_pct': (hold / total_analysts) * 100,
                'sell_pct': (sell / total_analysts) * 100,
                'strong_sell_pct': (strong_sell / total_analysts) * 100,
            }

            # Determine consensus
            bullish_count = strong_buy + buy
            bearish_count = sell + strong_sell

            bullish_pct = (bullish_count / total_analysts) * 100

            if bullish_pct >= 70:
                consensus = "STRONG BUY"
            elif bullish_pct >= 55:
                consensus = "BUY"
            elif bullish_pct >= 45:
                consensus = "HOLD"
            elif bullish_pct >= 30:
                consensus = "UNDERPERFORM"
            else:
                consensus = "SELL"

            ratings['bullish_count'] = bullish_count
            ratings['bearish_count'] = bearish_count
            ratings['consensus'] = consensus
            ratings['bullish_pct'] = bullish_pct

            # Trend (compare to previous period)
            if len(latest) >= 2:
                prev = latest.iloc[-2]
                prev_total = prev.get('strongBuy', 0) + prev.get('buy', 0) + prev.get('hold', 0) + prev.get('sell', 0) + prev.get('strongSell', 0)
                prev_bullish = prev.get('strongBuy', 0) + prev.get('buy', 0)

                if prev_total > 0:
                    prev_bullish_pct = (prev_bullish / prev_total) * 100
                    change = bullish_pct - prev_bullish_pct

                    if change > 5:
                        ratings['trend'] = "IMPROVING"
                    elif change < -5:
                        ratings['trend'] = "WEAKENING"
                    else:
                        ratings['trend'] = "STABLE"
                else:
                    ratings['trend'] = "N/A"
            else:
                ratings['trend'] = "N/A"

            return ratings

        except Exception as e:
            return {'error': str(e)}

    def get_price_targets(self) -> Dict:
        """Get analyst price targets."""
        try:
            current_price = self.info.get('currentPrice') or self.info.get('regularMarketPrice', 0)
            target_high = self.info.get('targetHighPrice', 0)
            target_low = self.info.get('targetLowPrice', 0)
            target_mean = self.info.get('targetMeanPrice', 0)
            target_median = self.info.get('targetMedianPrice', 0)

            if target_mean == 0:
                return {'error': 'No price target data'}

            # Calculate upside/downside
            upside_mean = ((target_mean - current_price) / current_price) * 100 if current_price > 0 else 0
            upside_high = ((target_high - current_price) / current_price) * 100 if current_price > 0 else 0
            upside_low = ((target_low - current_price) / current_price) * 100 if current_price > 0 else 0
            upside_median = ((target_median - current_price) / current_price) * 100 if current_price > 0 else 0

            return {
                'current_price': current_price,
                'target_high': target_high,
                'target_low': target_low,
                'target_mean': target_mean,
                'target_median': target_median,
                'upside_mean': upside_mean,
                'upside_high': upside_high,
                'upside_low': upside_low,
                'upside_median': upside_median,
            }

        except Exception as e:
            return {'error': str(e)}

    def get_earnings_estimates(self) -> Dict:
        """Get earnings estimate data."""
        try:
            # Get analyst earnings estimates
            eps_current_quarter = self.info.get('epsForward', 0)
            eps_current_year = self.info.get('epsTrailingTwelveMonths', 0)

            # Get revenue estimates
            revenue_estimate = self.info.get('revenueEstimate', 0)

            # Earnings growth
            earnings_growth = self.info.get('earningsGrowth', 0) * 100 if self.info.get('earningsGrowth') else 0
            revenue_growth = self.info.get('revenueGrowth', 0) * 100 if self.info.get('revenueGrowth') else 0

            return {
                'eps_forward': eps_current_quarter,
                'eps_ttm': eps_current_year,
                'earnings_growth': earnings_growth,
                'revenue_growth': revenue_growth,
            }

        except Exception as e:
            return {'error': str(e)}

    def analyze_consensus(self) -> str:
        """Provide narrative analysis of consensus."""
        ratings = self.get_analyst_ratings()
        targets = self.get_price_targets()

        if 'error' in ratings:
            return "Insufficient analyst coverage to determine consensus"

        consensus = ratings['consensus']
        bullish_pct = ratings['bullish_pct']
        total = ratings['total']
        trend = ratings.get('trend', 'N/A')

        narrative = f"{consensus} consensus from {total} analysts ({bullish_pct:.0f}% bullish)"

        if trend != 'N/A':
            narrative += f" - Trend: {trend}"

        if 'error' not in targets:
            upside = targets['upside_mean']
            narrative += f" | Avg target implies {upside:+.1f}% upside"

        return narrative

    def run_full_analysis(self) -> Dict:
        """Run complete analyst consensus analysis."""
        print(f"\n{'='*60}")
        print(f"ANALYST CONSENSUS: {self.ticker}")
        print(f"{'='*60}\n")

        # Get all data
        ratings = self.get_analyst_ratings()
        targets = self.get_price_targets()
        estimates = self.get_earnings_estimates()

        # Print ratings
        print("ANALYST RATINGS")
        print("-" * 60)

        if 'error' not in ratings:
            print(f"  Strong Buy: {ratings['strong_buy']:2d} ({ratings['strong_buy_pct']:4.1f}%)")
            print(f"  Buy:        {ratings['buy']:2d} ({ratings['buy_pct']:4.1f}%)")
            print(f"  Hold:       {ratings['hold']:2d} ({ratings['hold_pct']:4.1f}%)")
            print(f"  Sell:       {ratings['sell']:2d} ({ratings['sell_pct']:4.1f}%)")
            print(f"  Strong Sell:{ratings['strong_sell']:2d} ({ratings['strong_sell_pct']:4.1f}%)")
            print(f"  " + "-"*58)
            print(f"  Total:      {ratings['total']:2d} analysts")
            print(f"\n  Bullish: {ratings['bullish_count']}/{ratings['total']} ({ratings['bullish_pct']:.0f}%)")
            print(f"  Bearish: {ratings['bearish_count']}/{ratings['total']} ({100-ratings['bullish_pct']:.0f}%)")
            print(f"\n  CONSENSUS: {ratings['consensus']}")
            if ratings['trend'] != 'N/A':
                print(f"  TREND: {ratings['trend']}")
        else:
            print(f"  {ratings['error']}")

        # Print price targets
        print(f"\n{'='*60}")
        print("PRICE TARGETS")
        print("-" * 60)

        if 'error' not in targets:
            print(f"  Current Price: ${targets['current_price']:.2f}")
            print(f"\n  Target High:   ${targets['target_high']:.2f} ({targets['upside_high']:+.1f}%)")
            print(f"  Target Mean:   ${targets['target_mean']:.2f} ({targets['upside_mean']:+.1f}%)")
            print(f"  Target Median: ${targets['target_median']:.2f} ({targets['upside_median']:+.1f}%)")
            print(f"  Target Low:    ${targets['target_low']:.2f} ({targets['upside_low']:+.1f}%)")

            # Interpretation
            avg_upside = targets['upside_mean']
            if avg_upside > 20:
                print(f"\n  [+] Analysts see significant upside ({avg_upside:+.1f}%)")
            elif avg_upside > 10:
                print(f"\n  [=] Analysts see modest upside ({avg_upside:+.1f}%)")
            elif avg_upside > -10:
                print(f"\n  [=] Analysts see fair value ({avg_upside:+.1f}%)")
            else:
                print(f"\n  [-] Analysts see downside ({avg_upside:+.1f}%)")
        else:
            print(f"  {targets['error']}")

        # Print earnings estimates
        print(f"\n{'='*60}")
        print("EARNINGS ESTIMATES")
        print("-" * 60)

        if 'error' not in estimates:
            print(f"  Forward EPS:      ${estimates['eps_forward']:.2f}")
            print(f"  TTM EPS:          ${estimates['eps_ttm']:.2f}")
            print(f"  Earnings Growth:  {estimates['earnings_growth']:+.1f}%")
            print(f"  Revenue Growth:   {estimates['revenue_growth']:+.1f}%")
        else:
            print(f"  {estimates['error']}")

        # Summary
        print(f"\n{'='*60}")
        print("ANALYST SUMMARY")
        print("=" * 60)
        summary = self.analyze_consensus()
        print(f"\n{summary}\n")

        return {
            'ticker': self.ticker,
            'ratings': ratings,
            'price_targets': targets,
            'earnings_estimates': estimates,
            'summary': summary,
        }


def track_all_picks():
    """Track analyst consensus for all 4 picks."""

    print("\n" + "="*60)
    print("PROTEUS ANALYST CONSENSUS TRACKER")
    print("Monitoring Wall Street Opinion on Top 4 Picks")
    print("="*60)

    tickers = ['MU', 'APP', 'PLTR', 'ALAB']
    results = {}

    for ticker in tickers:
        print(f"\n\n{'#'*60}")
        print(f"# {ticker}")
        print(f"{'#'*60}")

        tracker = AnalystTracker(ticker)
        result = tracker.run_full_analysis()
        results[ticker] = result

    # Comparison table
    print("\n\n" + "="*60)
    print("CONSENSUS COMPARISON - ALL 4 PICKS")
    print("="*60 + "\n")

    print(f"{'Ticker':<8} {'Consensus':<15} {'Bullish %':<12} {'Avg Target':<12} {'Upside'}")
    print("-" * 70)

    for ticker in tickers:
        if ticker in results:
            r = results[ticker]
            ratings = r.get('ratings', {})
            targets = r.get('price_targets', {})

            consensus = ratings.get('consensus', 'N/A')
            bullish_pct = ratings.get('bullish_pct', 0)
            total = ratings.get('total', 0)
            avg_target = targets.get('target_mean', 0)
            upside = targets.get('upside_mean', 0)

            print(f"{ticker:<8} {consensus:<15} {bullish_pct:5.0f}% ({total:2d})   ${avg_target:<10.2f} {upside:+6.1f}%")

    # Strength ranking
    print(f"\n{'='*60}")
    print("WALL STREET CONVICTION RANKING")
    print(f"{'='*60}\n")

    # Create ranking by bullish %
    rankings = []
    for ticker in tickers:
        if ticker in results:
            r = results[ticker]
            ratings = r.get('ratings', {})
            if 'bullish_pct' in ratings:
                rankings.append((ticker, ratings['bullish_pct'], ratings['total']))

    rankings.sort(key=lambda x: x[1], reverse=True)

    for i, (ticker, bullish_pct, total) in enumerate(rankings, 1):
        stars = "***" if bullish_pct >= 70 else "**" if bullish_pct >= 55 else "*"
        print(f"  {i}. {ticker} - {bullish_pct:.0f}% bullish from {total} analysts {stars}")

    # Save results
    output_file = 'results/analyst_consensus.json'
    # Make serializable (convert numpy types to native Python)
    def convert_to_serializable(obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        else:
            return obj

    serializable_results = {}
    for ticker, data in results.items():
        serializable_results[ticker] = convert_to_serializable({
            'summary': data.get('summary', ''),
            'ratings': data.get('ratings', {}),
            'price_targets': data.get('price_targets', {}),
            'earnings_estimates': data.get('earnings_estimates', {}),
        })

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")

    print("INSIGHTS:")
    print("- Higher bullish % = stronger Wall Street conviction")
    print("- Consensus trends (IMPROVING/WEAKENING) signal momentum")
    print("- Compare analyst upside to DCF fair value for validation")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    track_all_picks()

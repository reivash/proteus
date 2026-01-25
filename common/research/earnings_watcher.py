"""
Earnings Watcher

Monitors earnings calendar and automatically analyzes:
1. Upcoming earnings for Proteus stocks
2. Recent earnings transcripts for sentiment
3. Guidance changes and management tone
4. Surprise factors and market reaction

Integrates with conviction scores to update trading decisions.
"""

import os
import json
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import anthropic


@dataclass
class EarningsAnalysis:
    """Analysis of an earnings report."""
    ticker: str
    company_name: str
    earnings_date: str
    analysis_date: str

    # Quantitative
    eps_actual: Optional[float]
    eps_estimate: Optional[float]
    eps_surprise_pct: Optional[float]
    revenue_actual: Optional[float]
    revenue_estimate: Optional[float]
    revenue_surprise_pct: Optional[float]

    # Qualitative (from Claude analysis)
    management_tone: str  # CONFIDENT, CAUTIOUS, DEFENSIVE, OPTIMISTIC
    guidance_change: str  # RAISED, MAINTAINED, LOWERED, NONE
    key_themes: List[str]
    risks_mentioned: List[str]
    opportunities_mentioned: List[str]

    # Scoring
    sentiment_score: float  # -10 to +10
    conviction_adjustment: float  # -20 to +20 points to add to conviction
    trading_implication: str  # BUY_MORE, HOLD, REDUCE, AVOID

    # Raw
    summary: str
    full_analysis: str


class EarningsWatcher:
    """
    Monitors and analyzes earnings for Proteus stocks.
    """

    PROTEUS_TICKERS = [
        "NVDA", "V", "MA", "AVGO", "AXP", "KLAC", "ORCL", "MRVL", "ABBV", "SYK",
        "EOG", "TXN", "GILD", "INTU", "MSFT", "QCOM", "JPM", "JNJ", "PFE", "WMT",
        "AMAT", "ADI", "NOW", "MLM", "IDXX", "EXR", "ROAD", "INSM", "SCHW", "AIG",
        "USB", "CVS", "LOW", "LMT", "COP", "SLB", "APD", "MS", "PNC", "CRM",
        "ADBE", "TGT", "CAT", "XOM", "MPC", "ECL", "NEE", "HCA", "CMCSA", "TMUS",
        "META", "ETN", "HD", "SHW"
    ]

    def __init__(self, output_dir: str = "features/daily_picks/data/earnings_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.client = anthropic.Anthropic()

    def get_upcoming_earnings(self, days_ahead: int = 14) -> List[Dict]:
        """Get upcoming earnings for Proteus stocks."""
        upcoming = []
        today = datetime.now()
        end_date = today + timedelta(days=days_ahead)

        print(f"Checking earnings calendar ({today.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...")

        for ticker in self.PROTEUS_TICKERS:
            try:
                stock = yf.Ticker(ticker)
                calendar = stock.calendar

                if calendar is not None and not calendar.empty:
                    # Get earnings date
                    if 'Earnings Date' in calendar.index:
                        earnings_dates = calendar.loc['Earnings Date']
                        if hasattr(earnings_dates, '__iter__') and not isinstance(earnings_dates, str):
                            earnings_date = earnings_dates.iloc[0] if len(earnings_dates) > 0 else None
                        else:
                            earnings_date = earnings_dates

                        if earnings_date:
                            if isinstance(earnings_date, str):
                                ed = datetime.strptime(earnings_date, '%Y-%m-%d')
                            else:
                                ed = earnings_date.to_pydatetime() if hasattr(earnings_date, 'to_pydatetime') else earnings_date

                            if today <= ed <= end_date:
                                upcoming.append({
                                    'ticker': ticker,
                                    'earnings_date': ed.strftime('%Y-%m-%d'),
                                    'days_until': (ed - today).days
                                })
                                print(f"  {ticker}: Earnings on {ed.strftime('%Y-%m-%d')} ({(ed-today).days} days)")
            except Exception as e:
                pass

        return sorted(upcoming, key=lambda x: x['days_until'])

    def get_recent_earnings(self, days_back: int = 7) -> List[Dict]:
        """Get stocks that reported earnings recently."""
        recent = []
        today = datetime.now()
        start_date = today - timedelta(days=days_back)

        print(f"Checking recent earnings ({start_date.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')})...")

        for ticker in self.PROTEUS_TICKERS:
            try:
                # Check earnings cache
                cache_file = f"features/daily_picks/data/earnings_cache/{ticker}_earnings.json"
                if os.path.exists(cache_file):
                    with open(cache_file, 'r') as f:
                        data = json.load(f)

                    for entry in data.get('earnings_dates', []):
                        try:
                            ed = datetime.strptime(entry['date'], '%Y-%m-%d')
                            if start_date <= ed <= today:
                                recent.append({
                                    'ticker': ticker,
                                    'earnings_date': entry['date'],
                                    'days_ago': (today - ed).days
                                })
                                print(f"  {ticker}: Reported on {entry['date']} ({(today-ed).days} days ago)")
                                break
                        except:
                            pass
            except:
                pass

        return sorted(recent, key=lambda x: x['days_ago'])

    def analyze_earnings(self, ticker: str, earnings_date: str = None) -> EarningsAnalysis:
        """
        Perform deep analysis of earnings for a stock.
        Uses Claude to analyze management tone, guidance, and implications.
        """
        print(f"\n{'='*60}")
        print(f"EARNINGS ANALYSIS: {ticker}")
        print(f"{'='*60}")

        # Get stock data
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get earnings history
        try:
            earnings_hist = stock.earnings_history
            quarterly_earnings = stock.quarterly_earnings
        except:
            earnings_hist = None
            quarterly_earnings = None

        # Build context for Claude
        context = self._build_earnings_context(ticker, info, earnings_hist, quarterly_earnings)

        # Get Claude analysis
        analysis = self._get_claude_analysis(ticker, context)

        # Build result
        result = EarningsAnalysis(
            ticker=ticker,
            company_name=info.get('longName', ticker),
            earnings_date=earnings_date or datetime.now().strftime('%Y-%m-%d'),
            analysis_date=datetime.now().strftime('%Y-%m-%d'),
            eps_actual=analysis.get('eps_actual'),
            eps_estimate=analysis.get('eps_estimate'),
            eps_surprise_pct=analysis.get('eps_surprise_pct'),
            revenue_actual=analysis.get('revenue_actual'),
            revenue_estimate=analysis.get('revenue_estimate'),
            revenue_surprise_pct=analysis.get('revenue_surprise_pct'),
            management_tone=analysis.get('management_tone', 'NEUTRAL'),
            guidance_change=analysis.get('guidance_change', 'NONE'),
            key_themes=analysis.get('key_themes', []),
            risks_mentioned=analysis.get('risks_mentioned', []),
            opportunities_mentioned=analysis.get('opportunities_mentioned', []),
            sentiment_score=analysis.get('sentiment_score', 0),
            conviction_adjustment=analysis.get('conviction_adjustment', 0),
            trading_implication=analysis.get('trading_implication', 'HOLD'),
            summary=analysis.get('summary', ''),
            full_analysis=analysis.get('full_analysis', '')
        )

        # Save result
        self._save_analysis(result)

        return result

    def _build_earnings_context(self, ticker: str, info: Dict,
                                 earnings_hist, quarterly_earnings) -> str:
        """Build context string for Claude analysis."""

        context = f"""
STOCK: {ticker} - {info.get('longName', ticker)}
SECTOR: {info.get('sector', 'Unknown')}
INDUSTRY: {info.get('industry', 'Unknown')}

CURRENT METRICS:
- Market Cap: ${info.get('marketCap', 0):,.0f}
- P/E Ratio: {info.get('trailingPE', 'N/A')}
- Forward P/E: {info.get('forwardPE', 'N/A')}
- Revenue Growth: {info.get('revenueGrowth', 0)*100:.1f}% if info.get('revenueGrowth') else 'N/A'
- Earnings Growth: {info.get('earningsGrowth', 0)*100:.1f}% if info.get('earningsGrowth') else 'N/A'
- Profit Margin: {info.get('profitMargins', 0)*100:.1f}% if info.get('profitMargins') else 'N/A'

ANALYST EXPECTATIONS:
- Recommendation: {info.get('recommendationKey', 'N/A')}
- Target Price: ${info.get('targetMeanPrice', 0):.2f}
- Current Price: ${info.get('currentPrice', 0):.2f}
- Upside: {((info.get('targetMeanPrice', 0) / info.get('currentPrice', 1)) - 1) * 100:.1f}%

BUSINESS SUMMARY:
{info.get('longBusinessSummary', 'Not available')[:800]}
"""

        if quarterly_earnings is not None and not quarterly_earnings.empty:
            context += "\n\nQUARTERLY EARNINGS HISTORY:\n"
            for idx in quarterly_earnings.index[:4]:
                row = quarterly_earnings.loc[idx]
                context += f"  {idx}: Revenue ${row.get('Revenue', 0):,.0f}, Earnings ${row.get('Earnings', 0):,.0f}\n"

        return context

    def _get_claude_analysis(self, ticker: str, context: str) -> Dict:
        """Get Claude's analysis of earnings."""

        prompt = f"""Analyze the following earnings context for {ticker} and provide a trading-focused assessment.

{context}

Based on this information, provide your analysis in the following JSON format:

```json
{{
    "eps_actual": <float or null>,
    "eps_estimate": <float or null>,
    "eps_surprise_pct": <float or null>,
    "revenue_actual": <float or null>,
    "revenue_estimate": <float or null>,
    "revenue_surprise_pct": <float or null>,
    "management_tone": "<CONFIDENT|CAUTIOUS|DEFENSIVE|OPTIMISTIC|NEUTRAL>",
    "guidance_change": "<RAISED|MAINTAINED|LOWERED|NONE>",
    "key_themes": ["<theme1>", "<theme2>", "<theme3>"],
    "risks_mentioned": ["<risk1>", "<risk2>"],
    "opportunities_mentioned": ["<opportunity1>", "<opportunity2>"],
    "sentiment_score": <-10 to +10>,
    "conviction_adjustment": <-20 to +20>,
    "trading_implication": "<BUY_MORE|HOLD|REDUCE|AVOID>",
    "summary": "<2-3 sentence summary of earnings quality and outlook>",
    "full_analysis": "<Detailed paragraph analysis>"
}}
```

Focus on:
1. Earnings quality (beat/miss magnitude, quality of beat)
2. Guidance trajectory (raised, maintained, lowered)
3. Management confidence signals
4. Key risks and opportunities
5. Trading implications for mean reversion strategy

Be objective and honest. If data is limited, say so."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

            # Parse JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response_text[json_start:json_end])

        except Exception as e:
            print(f"[ERROR] Claude analysis failed: {e}")

        return {
            'management_tone': 'NEUTRAL',
            'guidance_change': 'NONE',
            'sentiment_score': 0,
            'conviction_adjustment': 0,
            'trading_implication': 'HOLD',
            'summary': 'Analysis unavailable',
            'full_analysis': str(e) if 'e' in dir() else 'Failed to get analysis'
        }

    def _save_analysis(self, result: EarningsAnalysis):
        """Save earnings analysis to file."""
        filepath = os.path.join(
            self.output_dir,
            f"{result.ticker}_{result.analysis_date}_earnings.json"
        )
        with open(filepath, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        print(f"[SAVED] {filepath}")

    def run_earnings_sweep(self) -> List[EarningsAnalysis]:
        """
        Run earnings analysis on all stocks with recent earnings.
        """
        print("="*70)
        print("PROTEUS EARNINGS SWEEP")
        print("="*70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Get stocks with recent earnings
        recent = self.get_recent_earnings(days_back=14)

        results = []
        for stock in recent:
            try:
                analysis = self.analyze_earnings(
                    stock['ticker'],
                    stock['earnings_date']
                )
                results.append(analysis)
                print(f"\n[RESULT] {stock['ticker']}: {analysis.trading_implication} "
                      f"(Sentiment: {analysis.sentiment_score:+.1f}, "
                      f"Conviction adj: {analysis.conviction_adjustment:+.0f})")
            except Exception as e:
                print(f"[ERROR] {stock['ticker']}: {e}")

        # Summary
        print("\n" + "="*70)
        print("EARNINGS SWEEP SUMMARY")
        print("="*70)
        print(f"Analyzed: {len(results)} stocks")

        if results:
            buy_more = [r for r in results if r.trading_implication == 'BUY_MORE']
            avoid = [r for r in results if r.trading_implication == 'AVOID']

            if buy_more:
                print(f"\nBUY MORE: {', '.join(r.ticker for r in buy_more)}")
            if avoid:
                print(f"AVOID: {', '.join(r.ticker for r in avoid)}")

        return results


def run_earnings_watcher():
    """Main entry point."""
    watcher = EarningsWatcher()

    print("\n1. Checking upcoming earnings...")
    upcoming = watcher.get_upcoming_earnings(days_ahead=14)

    print("\n2. Checking recent earnings...")
    recent = watcher.get_recent_earnings(days_back=14)

    if recent:
        print("\n3. Analyzing recent earnings...")
        results = watcher.run_earnings_sweep()
        return results

    print("\nNo recent earnings to analyze.")
    return []


if __name__ == "__main__":
    run_earnings_watcher()

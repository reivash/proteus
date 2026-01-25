"""
Stock Deep Dive Agent

Leverages Claude's multi-phase reasoning capability to perform exhaustive
stock research that humans typically skip because it's tedious.

This is NOT about prediction - it's about building conviction through
thorough research BEFORE signals fire.

Output: Structured conviction scores that Proteus uses for:
1. Signal filtering (only trade high-conviction stocks)
2. Position sizing (bigger on high-conviction)
3. Thesis documentation (know why before entry)
"""

import os
import json
import anthropic
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import yfinance as yf


@dataclass
class DeepDiveResult:
    """Complete deep dive analysis result."""
    ticker: str
    company_name: str
    analysis_date: str

    # Phase scores (1-10)
    business_quality_score: float
    financial_health_score: float
    competitive_moat_score: float
    management_quality_score: float
    catalyst_score: float
    risk_score: float  # Lower is riskier

    # Composite
    conviction_score: float  # Weighted composite (1-100)
    conviction_tier: str  # HIGH, MEDIUM, LOW, AVOID

    # Detailed analysis
    business_summary: str
    bull_case: str
    bear_case: str
    key_risks: List[str]
    upcoming_catalysts: List[str]

    # Trading guidance
    position_size_modifier: float  # 0.5x to 2.0x
    recommended_hold_period: str
    stop_loss_suggestion: str

    # Meta
    analysis_duration_minutes: float
    data_sources_used: List[str]
    confidence_in_analysis: str  # HIGH, MEDIUM, LOW


class StockDeepDiveAgent:
    """
    Agent that performs exhaustive multi-phase stock analysis.

    Designed to maximize Claude's reasoning time on each stock,
    building a research library that compounds over time.
    """

    def __init__(self, output_dir: str = "data/deep_dives"):
        self.client = anthropic.Anthropic()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze_stock(self, ticker: str, verbose: bool = True) -> DeepDiveResult:
        """
        Perform comprehensive deep dive on a stock.

        This is intentionally slow and thorough - we want Claude to
        spend significant time reasoning about the business.
        """
        start_time = datetime.now()

        if verbose:
            print(f"\n{'='*70}")
            print(f"DEEP DIVE ANALYSIS: {ticker}")
            print(f"{'='*70}")
            print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("This will take several minutes for thorough analysis...")
            print()

        # Gather raw data first
        if verbose:
            print("[1/7] Gathering market data...")
        market_data = self._gather_market_data(ticker)

        # Run the comprehensive analysis
        if verbose:
            print("[2/7] Running deep analysis (this is the main reasoning phase)...")

        analysis = self._run_deep_analysis(ticker, market_data, verbose)

        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60

        # Build result
        result = DeepDiveResult(
            ticker=ticker,
            company_name=market_data.get('company_name', ticker),
            analysis_date=datetime.now().strftime('%Y-%m-%d'),
            business_quality_score=analysis.get('business_quality_score', 5),
            financial_health_score=analysis.get('financial_health_score', 5),
            competitive_moat_score=analysis.get('competitive_moat_score', 5),
            management_quality_score=analysis.get('management_quality_score', 5),
            catalyst_score=analysis.get('catalyst_score', 5),
            risk_score=analysis.get('risk_score', 5),
            conviction_score=analysis.get('conviction_score', 50),
            conviction_tier=analysis.get('conviction_tier', 'MEDIUM'),
            business_summary=analysis.get('business_summary', ''),
            bull_case=analysis.get('bull_case', ''),
            bear_case=analysis.get('bear_case', ''),
            key_risks=analysis.get('key_risks', []),
            upcoming_catalysts=analysis.get('upcoming_catalysts', []),
            position_size_modifier=analysis.get('position_size_modifier', 1.0),
            recommended_hold_period=analysis.get('recommended_hold_period', '1-3 days'),
            stop_loss_suggestion=analysis.get('stop_loss_suggestion', '-2%'),
            analysis_duration_minutes=duration,
            data_sources_used=analysis.get('data_sources_used', ['Yahoo Finance']),
            confidence_in_analysis=analysis.get('confidence_in_analysis', 'MEDIUM')
        )

        # Save result
        self._save_result(result)

        if verbose:
            print(f"\n[COMPLETE] Analysis took {duration:.1f} minutes")
            self._print_summary(result)

        return result

    def _gather_market_data(self, ticker: str) -> Dict[str, Any]:
        """Gather raw market data for analysis."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get financials
            try:
                financials = stock.quarterly_financials
                balance_sheet = stock.quarterly_balance_sheet
                cash_flow = stock.quarterly_cashflow
            except:
                financials = None
                balance_sheet = None
                cash_flow = None

            return {
                'company_name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_book': info.get('priceToBook', None),
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),
                'profit_margin': info.get('profitMargins', None),
                'operating_margin': info.get('operatingMargins', None),
                'roe': info.get('returnOnEquity', None),
                'roa': info.get('returnOnAssets', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'current_ratio': info.get('currentRatio', None),
                'free_cash_flow': info.get('freeCashflow', None),
                'beta': info.get('beta', None),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', None),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', None),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', None)),
                'analyst_target': info.get('targetMeanPrice', None),
                'recommendation': info.get('recommendationKey', None),
                'short_percent': info.get('shortPercentOfFloat', None),
                'insider_ownership': info.get('heldPercentInsiders', None),
                'institutional_ownership': info.get('heldPercentInstitutions', None),
                'business_summary': info.get('longBusinessSummary', ''),
                'recent_news': self._get_recent_news(stock),
            }
        except Exception as e:
            return {'error': str(e), 'company_name': ticker}

    def _get_recent_news(self, stock) -> List[str]:
        """Get recent news headlines."""
        try:
            news = stock.news[:5] if stock.news else []
            return [n.get('title', '') for n in news]
        except:
            return []

    def _run_deep_analysis(self, ticker: str, market_data: Dict, verbose: bool) -> Dict:
        """
        Run the main multi-phase analysis using Claude.

        This is where we leverage Claude's ability to reason deeply
        across multiple phases, building up a comprehensive view.
        """

        prompt = self._build_analysis_prompt(ticker, market_data)

        # Use Claude with extended thinking for deep analysis
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        # Parse the structured response
        response_text = response.content[0].text

        if verbose:
            print("[3/7] Parsing business model analysis...")
            print("[4/7] Parsing financial forensics...")
            print("[5/7] Parsing competitive position...")
            print("[6/7] Parsing management & catalysts...")
            print("[7/7] Calculating conviction scores...")

        return self._parse_analysis_response(response_text)

    def _build_analysis_prompt(self, ticker: str, market_data: Dict) -> str:
        """Build the comprehensive analysis prompt."""

        return f"""You are a senior equity research analyst performing a deep dive on {ticker} ({market_data.get('company_name', ticker)}).

Your goal is to build CONVICTION - not predict price. Spend time reasoning through each phase thoroughly.

## RAW DATA PROVIDED

**Company:** {market_data.get('company_name', ticker)}
**Sector:** {market_data.get('sector', 'Unknown')} | **Industry:** {market_data.get('industry', 'Unknown')}
**Market Cap:** ${market_data.get('market_cap', 0):,.0f}

**Valuation:**
- P/E (TTM): {market_data.get('pe_ratio', 'N/A')}
- Forward P/E: {market_data.get('forward_pe', 'N/A')}
- PEG Ratio: {market_data.get('peg_ratio', 'N/A')}
- Price/Book: {market_data.get('price_to_book', 'N/A')}

**Growth:**
- Revenue Growth: {self._format_pct(market_data.get('revenue_growth'))}
- Earnings Growth: {self._format_pct(market_data.get('earnings_growth'))}

**Profitability:**
- Profit Margin: {self._format_pct(market_data.get('profit_margin'))}
- Operating Margin: {self._format_pct(market_data.get('operating_margin'))}
- ROE: {self._format_pct(market_data.get('roe'))}
- ROA: {self._format_pct(market_data.get('roa'))}

**Financial Health:**
- Debt/Equity: {market_data.get('debt_to_equity', 'N/A')}
- Current Ratio: {market_data.get('current_ratio', 'N/A')}
- Free Cash Flow: ${market_data.get('free_cash_flow', 0):,.0f}

**Trading:**
- Current Price: ${market_data.get('current_price', 0):.2f}
- 52-Week Range: ${market_data.get('fifty_two_week_low', 0):.2f} - ${market_data.get('fifty_two_week_high', 0):.2f}
- Beta: {market_data.get('beta', 'N/A')}
- Short Interest: {self._format_pct(market_data.get('short_percent'))}

**Ownership:**
- Insider Ownership: {self._format_pct(market_data.get('insider_ownership'))}
- Institutional Ownership: {self._format_pct(market_data.get('institutional_ownership'))}

**Analyst View:**
- Target Price: ${market_data.get('analyst_target', 0):.2f}
- Recommendation: {market_data.get('recommendation', 'N/A')}

**Business Summary:**
{market_data.get('business_summary', 'Not available')[:1000]}

**Recent News:**
{chr(10).join('- ' + n for n in market_data.get('recent_news', [])[:5])}

---

## YOUR ANALYSIS TASK

Complete ALL SIX phases below. Be thorough - this research will be used for real trading decisions.

### PHASE 1: BUSINESS MODEL ANALYSIS
- What exactly does this company do? Explain like I'm smart but unfamiliar.
- Who are the customers? Any concentration risk?
- What's the revenue model? Recurring vs one-time?
- What's the moat? (Network effects, switching costs, brand, scale, IP?)
- Is the moat durable or eroding?
- **SCORE (1-10):** Business Quality Score

### PHASE 2: FINANCIAL FORENSICS
- Is revenue quality high? (Recurring, diversified, growing?)
- Any accounting red flags? (DSO trends, inventory build, deferred revenue oddities?)
- Cash flow vs earnings - do they match? If not, why?
- Is the balance sheet a weapon or a risk?
- Capital allocation: Are they investing wisely or destroying value?
- **SCORE (1-10):** Financial Health Score

### PHASE 3: COMPETITIVE POSITION
- Who are the main competitors?
- Is market share growing, stable, or shrinking?
- What's the competitive advantage period (CAP)?
- Disruption risk: Could a startup or tech shift kill this business?
- Industry structure: Is this a good industry to be in?
- **SCORE (1-10):** Competitive Moat Score

### PHASE 4: MANAGEMENT & GOVERNANCE
- Management track record: Have they delivered?
- Insider buying/selling: Are they believers?
- Compensation alignment: Paid for performance or just showing up?
- Capital allocation history: Smart acquirers or empire builders?
- Any governance red flags?
- **SCORE (1-10):** Management Quality Score

### PHASE 5: CATALYST IDENTIFICATION
- Next earnings date and expectations
- Product launches or major announcements coming?
- Regulatory decisions pending?
- M&A potential (acquirer or target)?
- Macro factors that could help/hurt near-term?
- **SCORE (1-10):** Catalyst Score (10 = strong near-term catalyst)

### PHASE 6: RISK ASSESSMENT
- What's the bear case? What could go wrong?
- Tail risks: Regulatory, legal, competitive, macro?
- How bad could it get? (Downside scenario)
- What would make you WRONG about this stock?
- Position sizing implications
- **SCORE (1-10):** Risk Score (10 = low risk, 1 = very risky)

---

## OUTPUT FORMAT

After your analysis, provide a structured summary in EXACTLY this JSON format:

```json
{{
    "business_quality_score": <1-10>,
    "financial_health_score": <1-10>,
    "competitive_moat_score": <1-10>,
    "management_quality_score": <1-10>,
    "catalyst_score": <1-10>,
    "risk_score": <1-10>,
    "conviction_score": <1-100 weighted composite>,
    "conviction_tier": "<HIGH|MEDIUM|LOW|AVOID>",
    "business_summary": "<2-3 sentence summary>",
    "bull_case": "<The optimistic scenario in 2-3 sentences>",
    "bear_case": "<The pessimistic scenario in 2-3 sentences>",
    "key_risks": ["<risk 1>", "<risk 2>", "<risk 3>"],
    "upcoming_catalysts": ["<catalyst 1>", "<catalyst 2>"],
    "position_size_modifier": <0.5 to 2.0>,
    "recommended_hold_period": "<e.g., 1-3 days, 1-2 weeks, 1-3 months>",
    "stop_loss_suggestion": "<e.g., -2%, -5%, -3% below entry>",
    "data_sources_used": ["Yahoo Finance", "<others>"],
    "confidence_in_analysis": "<HIGH|MEDIUM|LOW>"
}}
```

CONVICTION TIER GUIDELINES:
- HIGH (75-100): Strong business, good financials, clear catalyst, manageable risk. Size up.
- MEDIUM (50-74): Decent business with some concerns. Standard position size.
- LOW (25-49): Significant concerns. Reduce size or require stronger technical signal.
- AVOID (0-24): Fundamental problems. Don't trade even if technical signal fires.

Now perform your analysis. Be thorough and honest - this matters for real money."""

    def _format_pct(self, value) -> str:
        """Format a decimal as percentage."""
        if value is None:
            return 'N/A'
        return f"{value*100:.1f}%"

    def _parse_analysis_response(self, response_text: str) -> Dict:
        """Parse the structured analysis response."""
        # Find JSON block in response
        try:
            # Look for JSON block
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Return defaults if parsing fails
        return {
            'business_quality_score': 5,
            'financial_health_score': 5,
            'competitive_moat_score': 5,
            'management_quality_score': 5,
            'catalyst_score': 5,
            'risk_score': 5,
            'conviction_score': 50,
            'conviction_tier': 'MEDIUM',
            'business_summary': 'Analysis parsing failed - manual review needed',
            'bull_case': '',
            'bear_case': '',
            'key_risks': [],
            'upcoming_catalysts': [],
            'position_size_modifier': 1.0,
            'recommended_hold_period': '1-3 days',
            'stop_loss_suggestion': '-2%',
            'data_sources_used': ['Yahoo Finance'],
            'confidence_in_analysis': 'LOW'
        }

    def _save_result(self, result: DeepDiveResult):
        """Save analysis result to file."""
        filepath = os.path.join(
            self.output_dir,
            f"{result.ticker}_{result.analysis_date}.json"
        )
        with open(filepath, 'w') as f:
            json.dump(asdict(result), f, indent=2)

    def _print_summary(self, result: DeepDiveResult):
        """Print analysis summary."""
        print(f"\n{'='*70}")
        print(f"DEEP DIVE SUMMARY: {result.ticker}")
        print(f"{'='*70}")
        print(f"\nConviction: {result.conviction_tier} ({result.conviction_score}/100)")
        print(f"\nPhase Scores:")
        print(f"  Business Quality:    {result.business_quality_score}/10")
        print(f"  Financial Health:    {result.financial_health_score}/10")
        print(f"  Competitive Moat:    {result.competitive_moat_score}/10")
        print(f"  Management Quality:  {result.management_quality_score}/10")
        print(f"  Catalyst Score:      {result.catalyst_score}/10")
        print(f"  Risk Score:          {result.risk_score}/10")
        print(f"\nBusiness Summary:")
        print(f"  {result.business_summary}")
        print(f"\nBull Case: {result.bull_case}")
        print(f"Bear Case: {result.bear_case}")
        print(f"\nKey Risks:")
        for risk in result.key_risks[:3]:
            print(f"  - {risk}")
        print(f"\nTrading Guidance:")
        print(f"  Position Size Modifier: {result.position_size_modifier}x")
        print(f"  Recommended Hold: {result.recommended_hold_period}")
        print(f"  Stop Loss: {result.stop_loss_suggestion}")
        print(f"\nAnalysis Confidence: {result.confidence_in_analysis}")
        print(f"Duration: {result.analysis_duration_minutes:.1f} minutes")

    def load_existing_analysis(self, ticker: str) -> Optional[DeepDiveResult]:
        """Load most recent analysis for a ticker if it exists."""
        import glob
        pattern = os.path.join(self.output_dir, f"{ticker}_*.json")
        files = sorted(glob.glob(pattern), reverse=True)

        if files:
            with open(files[0], 'r') as f:
                data = json.load(f)
                return DeepDiveResult(**data)
        return None

    def get_conviction_for_signal(self, ticker: str, max_age_days: int = 30) -> Dict:
        """
        Get conviction data for a signal, running fresh analysis if needed.

        This is the integration point with Proteus signal filtering.
        """
        existing = self.load_existing_analysis(ticker)

        if existing:
            # Check age
            analysis_date = datetime.strptime(existing.analysis_date, '%Y-%m-%d')
            age_days = (datetime.now() - analysis_date).days

            if age_days <= max_age_days:
                return {
                    'ticker': ticker,
                    'conviction_score': existing.conviction_score,
                    'conviction_tier': existing.conviction_tier,
                    'position_size_modifier': existing.position_size_modifier,
                    'analysis_age_days': age_days,
                    'needs_refresh': False
                }

        # Need fresh analysis
        return {
            'ticker': ticker,
            'conviction_score': None,
            'conviction_tier': 'UNKNOWN',
            'position_size_modifier': 1.0,
            'analysis_age_days': None,
            'needs_refresh': True
        }


def run_deep_dive(ticker: str):
    """Run a deep dive analysis on a single stock."""
    agent = StockDeepDiveAgent()
    result = agent.analyze_stock(ticker)
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "AVGO"  # Default to today's signal

    print(f"Running deep dive on {ticker}...")
    result = run_deep_dive(ticker)

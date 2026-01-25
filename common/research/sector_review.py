"""
Weekly Sector Review

Analyzes sector performance and rotation to:
1. Identify which sectors are leading/lagging
2. Compare Proteus stocks against sector benchmarks
3. Detect sector rotation signals
4. Adjust conviction based on sector momentum

Run weekly (Sunday evening) to prepare for the trading week.
"""

import os
import json
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class SectorAnalysis:
    """Analysis of a sector."""
    sector: str
    stocks: List[str]
    avg_return_1w: float
    avg_return_1m: float
    avg_return_3m: float
    relative_strength: float  # vs SPY
    momentum_score: float  # -10 to +10
    rotation_signal: str  # INFLOW, OUTFLOW, NEUTRAL
    top_performer: str
    worst_performer: str
    sector_outlook: str


@dataclass
class WeeklySectorReport:
    """Complete weekly sector report."""
    report_date: str
    market_regime: str
    spy_return_1w: float
    spy_return_1m: float

    sectors: List[SectorAnalysis]
    sector_rankings: List[str]  # Best to worst

    rotation_theme: str
    actionable_insights: List[str]
    conviction_adjustments: Dict[str, float]  # ticker -> adjustment


class SectorReviewer:
    """
    Performs weekly sector analysis for Proteus stocks.
    """

    PROTEUS_TICKERS = [
        "NVDA", "V", "MA", "AVGO", "AXP", "KLAC", "ORCL", "MRVL", "ABBV", "SYK",
        "EOG", "TXN", "GILD", "INTU", "MSFT", "QCOM", "JPM", "JNJ", "PFE", "WMT",
        "AMAT", "ADI", "NOW", "MLM", "IDXX", "EXR", "ROAD", "INSM", "SCHW", "AIG",
        "USB", "CVS", "LOW", "LMT", "COP", "SLB", "APD", "MS", "PNC", "CRM",
        "ADBE", "TGT", "CAT", "XOM", "MPC", "ECL", "NEE", "HCA", "CMCSA", "TMUS",
        "META", "ETN", "HD", "SHW"
    ]

    SECTOR_ETFS = {
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        'Financial Services': 'XLF',
        'Energy': 'XLE',
        'Consumer Cyclical': 'XLY',
        'Industrials': 'XLI',
        'Consumer Defensive': 'XLP',
        'Utilities': 'XLU',
        'Real Estate': 'XLRE',
        'Basic Materials': 'XLB',
        'Communication Services': 'XLC'
    }

    def __init__(self, output_dir: str = "data/sector_reviews"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def get_stock_sectors(self) -> Dict[str, List[str]]:
        """Get sector mapping for all Proteus stocks."""
        sectors = defaultdict(list)

        print("Mapping stocks to sectors...")
        for ticker in self.PROTEUS_TICKERS:
            try:
                stock = yf.Ticker(ticker)
                sector = stock.info.get('sector', 'Unknown')
                sectors[sector].append(ticker)
            except:
                sectors['Unknown'].append(ticker)

        return dict(sectors)

    def get_returns(self, ticker: str, periods: List[int] = [5, 21, 63]) -> Dict[str, float]:
        """Get returns for different periods (in trading days)."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")

            if len(hist) < max(periods):
                return {f"{p}d": 0.0 for p in periods}

            current_price = hist['Close'].iloc[-1]
            returns = {}

            for p in periods:
                if len(hist) > p:
                    past_price = hist['Close'].iloc[-p-1]
                    returns[f"{p}d"] = ((current_price / past_price) - 1) * 100
                else:
                    returns[f"{p}d"] = 0.0

            return returns
        except Exception as e:
            return {f"{p}d": 0.0 for p in periods}

    def analyze_sector(self, sector: str, tickers: List[str], spy_returns: Dict) -> SectorAnalysis:
        """Analyze a single sector."""
        print(f"\n  Analyzing {sector} ({len(tickers)} stocks)...")

        returns_1w = []
        returns_1m = []
        returns_3m = []
        stock_returns = {}

        for ticker in tickers:
            returns = self.get_returns(ticker)
            returns_1w.append(returns.get('5d', 0))
            returns_1m.append(returns.get('21d', 0))
            returns_3m.append(returns.get('63d', 0))
            stock_returns[ticker] = returns

        avg_1w = np.mean(returns_1w) if returns_1w else 0
        avg_1m = np.mean(returns_1m) if returns_1m else 0
        avg_3m = np.mean(returns_3m) if returns_3m else 0

        # Relative strength vs SPY
        rel_strength = avg_1m - spy_returns.get('21d', 0)

        # Momentum score (-10 to +10)
        momentum = np.clip(avg_1w * 2, -10, 10)

        # Rotation signal
        if avg_1w > 2 and rel_strength > 1:
            rotation = 'INFLOW'
        elif avg_1w < -2 and rel_strength < -1:
            rotation = 'OUTFLOW'
        else:
            rotation = 'NEUTRAL'

        # Top/worst performers
        sorted_stocks = sorted(stock_returns.items(), key=lambda x: x[1].get('5d', 0), reverse=True)
        top = sorted_stocks[0][0] if sorted_stocks else 'N/A'
        worst = sorted_stocks[-1][0] if sorted_stocks else 'N/A'

        # Outlook
        if momentum > 3 and rotation == 'INFLOW':
            outlook = 'BULLISH - Strong momentum and inflows'
        elif momentum < -3 and rotation == 'OUTFLOW':
            outlook = 'BEARISH - Weak momentum and outflows'
        elif momentum > 0:
            outlook = 'NEUTRAL-POSITIVE - Moderate strength'
        elif momentum < 0:
            outlook = 'NEUTRAL-NEGATIVE - Moderate weakness'
        else:
            outlook = 'NEUTRAL - Mixed signals'

        return SectorAnalysis(
            sector=sector,
            stocks=tickers,
            avg_return_1w=round(avg_1w, 2),
            avg_return_1m=round(avg_1m, 2),
            avg_return_3m=round(avg_3m, 2),
            relative_strength=round(rel_strength, 2),
            momentum_score=round(momentum, 2),
            rotation_signal=rotation,
            top_performer=top,
            worst_performer=worst,
            sector_outlook=outlook
        )

    def generate_conviction_adjustments(self, sectors: List[SectorAnalysis]) -> Dict[str, float]:
        """Generate conviction adjustments based on sector analysis."""
        adjustments = {}

        for sector_analysis in sectors:
            # Base adjustment on momentum and rotation
            base_adj = sector_analysis.momentum_score / 2  # -5 to +5

            if sector_analysis.rotation_signal == 'INFLOW':
                base_adj += 3
            elif sector_analysis.rotation_signal == 'OUTFLOW':
                base_adj -= 3

            # Apply to all stocks in sector
            for ticker in sector_analysis.stocks:
                adjustments[ticker] = round(base_adj, 1)

        return adjustments

    def run_weekly_review(self) -> WeeklySectorReport:
        """Run complete weekly sector review."""
        print("="*70)
        print("PROTEUS WEEKLY SECTOR REVIEW")
        print("="*70)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Get SPY returns as benchmark
        print("\n1. Getting market benchmark (SPY)...")
        spy_returns = self.get_returns('SPY')
        print(f"   SPY: 1W {spy_returns.get('5d', 0):+.1f}%, 1M {spy_returns.get('21d', 0):+.1f}%")

        # Determine market regime
        if spy_returns.get('21d', 0) > 3:
            regime = 'BULL'
        elif spy_returns.get('21d', 0) < -3:
            regime = 'BEAR'
        else:
            regime = 'SIDEWAYS'
        print(f"   Regime: {regime}")

        # Get sector mapping
        print("\n2. Mapping stocks to sectors...")
        sector_map = self.get_stock_sectors()

        # Analyze each sector
        print("\n3. Analyzing sectors...")
        sector_analyses = []
        for sector, tickers in sector_map.items():
            if sector != 'Unknown' and tickers:
                analysis = self.analyze_sector(sector, tickers, spy_returns)
                sector_analyses.append(analysis)
                print(f"     {sector}: {analysis.rotation_signal} (Mom: {analysis.momentum_score:+.1f})")

        # Rank sectors
        ranked = sorted(sector_analyses, key=lambda x: x.momentum_score, reverse=True)
        rankings = [s.sector for s in ranked]

        # Generate conviction adjustments
        print("\n4. Generating conviction adjustments...")
        adjustments = self.generate_conviction_adjustments(sector_analyses)

        # Identify rotation theme
        inflows = [s.sector for s in sector_analyses if s.rotation_signal == 'INFLOW']
        outflows = [s.sector for s in sector_analyses if s.rotation_signal == 'OUTFLOW']

        if inflows and outflows:
            theme = f"Rotation FROM {', '.join(outflows[:2])} TO {', '.join(inflows[:2])}"
        elif inflows:
            theme = f"Broad inflows into {', '.join(inflows[:2])}"
        elif outflows:
            theme = f"Broad outflows from {', '.join(outflows[:2])}"
        else:
            theme = "No clear rotation pattern"

        # Actionable insights
        insights = []
        if ranked:
            insights.append(f"Strongest sector: {ranked[0].sector} ({ranked[0].avg_return_1w:+.1f}% 1W)")
            insights.append(f"Weakest sector: {ranked[-1].sector} ({ranked[-1].avg_return_1w:+.1f}% 1W)")

        if inflows:
            insights.append(f"FAVOR signals in: {', '.join(inflows)}")
        if outflows:
            insights.append(f"CAUTION on signals in: {', '.join(outflows)}")

        # Build report
        report = WeeklySectorReport(
            report_date=datetime.now().strftime('%Y-%m-%d'),
            market_regime=regime,
            spy_return_1w=spy_returns.get('5d', 0),
            spy_return_1m=spy_returns.get('21d', 0),
            sectors=sector_analyses,
            sector_rankings=rankings,
            rotation_theme=theme,
            actionable_insights=insights,
            conviction_adjustments=adjustments
        )

        # Save report
        self._save_report(report)

        # Print summary
        self._print_summary(report)

        return report

    def _save_report(self, report: WeeklySectorReport):
        """Save report to file."""
        filepath = os.path.join(
            self.output_dir,
            f"sector_review_{report.report_date}.json"
        )

        # Convert to serializable format
        data = {
            'report_date': report.report_date,
            'market_regime': report.market_regime,
            'spy_return_1w': report.spy_return_1w,
            'spy_return_1m': report.spy_return_1m,
            'sectors': [asdict(s) for s in report.sectors],
            'sector_rankings': report.sector_rankings,
            'rotation_theme': report.rotation_theme,
            'actionable_insights': report.actionable_insights,
            'conviction_adjustments': report.conviction_adjustments
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n[SAVED] {filepath}")

    def _print_summary(self, report: WeeklySectorReport):
        """Print summary of sector review."""
        print("\n" + "="*70)
        print("SECTOR REVIEW SUMMARY")
        print("="*70)

        print(f"\nMarket Regime: {report.market_regime}")
        print(f"SPY: 1W {report.spy_return_1w:+.1f}%, 1M {report.spy_return_1m:+.1f}%")
        print(f"\nRotation Theme: {report.rotation_theme}")

        print("\nSector Rankings (Best to Worst):")
        for i, sector in enumerate(report.sector_rankings[:5], 1):
            analysis = next(s for s in report.sectors if s.sector == sector)
            print(f"  {i}. {sector}: {analysis.rotation_signal} ({analysis.avg_return_1w:+.1f}% 1W)")

        print("\nActionable Insights:")
        for insight in report.actionable_insights:
            print(f"  - {insight}")

        print("\nTop Conviction Adjustments:")
        sorted_adj = sorted(report.conviction_adjustments.items(), key=lambda x: abs(x[1]), reverse=True)
        for ticker, adj in sorted_adj[:10]:
            if adj != 0:
                print(f"  {ticker}: {adj:+.1f} points")


def run_sector_review():
    """Main entry point."""
    reviewer = SectorReviewer()
    report = reviewer.run_weekly_review()
    return report


if __name__ == "__main__":
    run_sector_review()

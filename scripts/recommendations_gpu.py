"""
Proteus Daily Recommendation System

High-quality, confident recommendations only.
Stays quiet when uncertain rather than forcing weak trades.

Usage:
    python scripts/proteus_recommend.py          # Full report
    python scripts/proteus_recommend.py --quiet  # Summary only
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from contextlib import redirect_stdout, redirect_stderr
import io

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

import yfinance as yf

# Defer scanner import to allow quiet mode to work
SmartScannerV2 = None

def get_scanner():
    global SmartScannerV2
    if SmartScannerV2 is None:
        from trading.smart_scanner_v2 import SmartScannerV2 as Scanner
        SmartScannerV2 = Scanner
    return SmartScannerV2()


@dataclass
class Recommendation:
    """A trading recommendation with confidence assessment."""
    ticker: str
    action: str  # BUY, HOLD, EXIT
    confidence: str  # HIGH, MODERATE, LOW
    raw_signal: float
    adjusted_signal: float
    tier: str
    ensemble_votes: int
    reasoning: List[str]
    concerns: List[str]
    entry_price: Optional[float] = None
    current_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: Optional[str] = None


class ProteusAdvisor:
    """
    High-quality recommendation engine.

    Confidence Levels:
    - HIGH: Raw 70+, Elite/Strong tier, 3-vote ensemble, Adjusted 65+
    - MODERATE: Raw 65+, Average+ tier, 2+ votes, Adjusted 55+
    - LOW: Everything else (not recommended)
    """

    # Quality thresholds - based on backtest data
    HIGH_RAW_THRESHOLD = 70
    HIGH_ADJUSTED_THRESHOLD = 65
    MODERATE_RAW_THRESHOLD = 65
    MODERATE_ADJUSTED_THRESHOLD = 55

    # Regime edge from backtests (win rate, avg return)
    REGIME_EDGE = {
        'bear': (0.69, 1.57),      # Best regime
        'volatile': (0.62, 1.52),  # Good regime
        'choppy': (0.56, 0.17),    # Bad regime - low edge
        'bull': (0.51, 0.15)       # Worst - no edge
    }

    # Tier edge from backtests
    TIER_EDGE = {
        'strong': (0.685, 1.15),   # Best tier
        'elite': (0.60, 0.56),
        'average': (0.59, 0.57),
        'weak': (0.55, 0.30),
        'avoid': (0.45, -0.15)
    }

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self._scanner = None
        self.wallet_path = Path(__file__).parent.parent / 'data' / 'virtual_wallet' / 'wallet_state.json'

    @property
    def scanner(self):
        if self._scanner is None:
            if self.quiet:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self._scanner = get_scanner()
            else:
                self._scanner = get_scanner()
        return self._scanner

    def load_positions(self) -> Dict:
        """Load current wallet positions."""
        if self.wallet_path.exists():
            with open(self.wallet_path) as f:
                return json.load(f)
        return {'positions': {}, 'cash': 100000}

    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Fetch current prices for tickers."""
        if not tickers:
            return {}
        try:
            data = yf.download(tickers, period='1d', progress=False)
            if 'Close' in data.columns:
                if len(tickers) == 1:
                    return {tickers[0]: float(data['Close'].iloc[-1])}
                return {t: float(data['Close'][t].iloc[-1]) for t in tickers if t in data['Close'].columns}
            return {}
        except Exception:
            return {}

    def assess_signal_confidence(self, sig) -> Tuple[str, List[str], List[str]]:
        """
        Assess confidence level of a signal.
        Returns: (confidence_level, strengths, concerns)
        """
        strengths = []
        concerns = []

        # Raw signal strength
        if sig.raw_strength >= 75:
            strengths.append(f"Very strong base signal ({sig.raw_strength:.0f})")
        elif sig.raw_strength >= 70:
            strengths.append(f"Strong base signal ({sig.raw_strength:.0f})")
        elif sig.raw_strength >= 65:
            strengths.append(f"Good base signal ({sig.raw_strength:.0f})")
        elif sig.raw_strength >= 60:
            concerns.append(f"Moderate base signal ({sig.raw_strength:.0f})")
        else:
            concerns.append(f"Weak base signal ({sig.raw_strength:.0f})")

        # Tier assessment (based on backtest data - strong > elite)
        tier_wr, tier_ret = self.TIER_EDGE.get(sig.tier, (0.55, 0.3))
        if sig.tier == 'strong':
            strengths.append(f"STRONG tier - best performer ({tier_wr:.0%} win, +{tier_ret:.1f}%)")
        elif sig.tier == 'elite':
            strengths.append(f"Elite tier ({tier_wr:.0%} win rate)")
        elif sig.tier == 'average':
            pass  # Neutral
        elif sig.tier == 'weak':
            concerns.append(f"Weak tier ({tier_wr:.0%} win rate)")
        elif sig.tier == 'avoid':
            concerns.append("AVOID tier - negative expected value")

        # Penalty impact
        penalty = sig.raw_strength - sig.adjusted_signal if hasattr(sig, 'adjusted_signal') else sig.raw_strength - sig.adjusted_strength
        if penalty > 15:
            concerns.append(f"Heavy penalties applied (-{penalty:.0f} points)")
        elif penalty > 8:
            concerns.append(f"Significant penalties (-{penalty:.0f} points)")
        elif penalty > 0:
            pass  # Minor penalties are normal

        # Adjusted signal
        adj = sig.adjusted_signal if hasattr(sig, 'adjusted_signal') else sig.adjusted_strength
        if adj >= 70:
            strengths.append(f"Adjusted signal very strong ({adj:.0f})")
        elif adj >= 65:
            strengths.append(f"Adjusted signal strong ({adj:.0f})")
        elif adj >= 55:
            pass  # Acceptable
        else:
            concerns.append(f"Adjusted signal weak ({adj:.0f})")

        # Quality assessment
        if sig.quality in ['excellent', 'strong']:
            strengths.append(f"{sig.quality.capitalize()} position sizing")
        elif sig.quality == 'moderate':
            pass  # Acceptable
        elif sig.quality == 'weak':
            concerns.append("Weak position sizing")

        # Determine confidence
        adj = sig.adjusted_signal if hasattr(sig, 'adjusted_signal') else sig.adjusted_strength

        if (sig.raw_strength >= self.HIGH_RAW_THRESHOLD and
            adj >= self.HIGH_ADJUSTED_THRESHOLD and
            sig.tier in ['elite', 'strong']):
            return 'HIGH', strengths, concerns

        if (sig.raw_strength >= self.MODERATE_RAW_THRESHOLD and
            adj >= self.MODERATE_ADJUSTED_THRESHOLD and
            sig.tier not in ['weak', 'avoid']):
            return 'MODERATE', strengths, concerns

        return 'LOW', strengths, concerns

    def generate_recommendations(self) -> Dict:
        """Generate daily recommendations."""

        # Run scan
        result = self.scanner.scan()

        # Load positions
        wallet = self.load_positions()
        positions = wallet.get('positions', {})

        # Get current prices for positions
        if positions:
            prices = self.get_current_prices(list(positions.keys()))
        else:
            prices = {}

        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'market_context': {
                'regime': result.regime,
                'regime_confidence': result.regime_confidence,
                'vix': result.vix,
                'bear_score': result.bear_score,
                'bear_level': result.bear_alert_level,
                'is_choppy': result.is_choppy
            },
            'high_confidence': [],
            'moderate_confidence': [],
            'watchlist': [],
            'current_positions': [],
            'exit_signals': [],
            'summary': ''
        }

        # Process exit signals first
        for action in result.rebalance_actions:
            recommendations['exit_signals'].append({
                'ticker': action.get('ticker', 'Unknown'),
                'action': 'EXIT',
                'reason': action.get('reason', 'Exit signal triggered'),
                'pnl_pct': action.get('pnl_pct', 0)
            })

        # Process current positions
        for ticker, pos in positions.items():
            current_price = prices.get(ticker, pos.get('current_price', pos['entry_price']))
            entry_price = pos['entry_price']
            pnl_pct = ((current_price - entry_price) / entry_price) * 100

            recommendations['current_positions'].append({
                'ticker': ticker,
                'entry_price': entry_price,
                'current_price': current_price,
                'pnl_pct': round(pnl_pct, 2),
                'days_held': pos.get('days_held', 0),
                'tier': pos.get('tier', 'unknown')
            })

        # Process new signals
        for sig in result.signals:
            # Skip if already holding
            if sig.ticker in positions:
                continue

            confidence, strengths, concerns = self.assess_signal_confidence(sig)

            rec = {
                'ticker': sig.ticker,
                'action': 'BUY',
                'confidence': confidence,
                'raw_signal': round(sig.raw_strength, 1),
                'adjusted_signal': round(sig.adjusted_strength, 1),
                'tier': sig.tier,
                'strengths': strengths,
                'concerns': concerns,
                'size_pct': sig.size_pct,
                'dollar_size': sig.dollar_size
            }

            if confidence == 'HIGH':
                recommendations['high_confidence'].append(rec)
            elif confidence == 'MODERATE':
                recommendations['moderate_confidence'].append(rec)
            else:
                recommendations['watchlist'].append(rec)

        # Generate summary
        recommendations['summary'] = self._generate_summary(recommendations)

        return recommendations

    def _generate_summary(self, recs: Dict) -> str:
        """Generate human-readable summary."""
        lines = []

        ctx = recs['market_context']
        regime = ctx['regime'].lower()

        # Regime-based edge assessment
        regime_wr, regime_ret = self.REGIME_EDGE.get(regime, (0.55, 0.3))

        if regime in ['bear', 'volatile']:
            lines.append(f"FAVORABLE REGIME ({regime.upper()}) - historical {regime_wr:.0%} win rate, +{regime_ret:.1f}% avg.")
        elif regime == 'choppy':
            lines.append(f"CHOPPY MARKET - low edge ({regime_wr:.0%} win rate). Being highly selective.")
        elif regime == 'bull':
            lines.append(f"BULL MARKET - mean reversion underperforms ({regime_wr:.0%} win rate). Consider sitting out.")

        if ctx['bear_score'] >= 40:
            lines.append(f"Elevated bear score ({ctx['bear_score']}) - may transition to better regime soon.")

        # Recommendations
        high = len(recs['high_confidence'])
        mod = len(recs['moderate_confidence'])

        if high > 0:
            tickers = [r['ticker'] for r in recs['high_confidence']]
            lines.append(f"HIGH CONFIDENCE: {', '.join(tickers)}")

        if mod > 0:
            tickers = [r['ticker'] for r in recs['moderate_confidence']]
            lines.append(f"MODERATE CONFIDENCE: {', '.join(tickers)}")

        if high == 0 and mod == 0:
            lines.append("No confident recommendations today. Standing aside.")

        # Exits
        if recs['exit_signals']:
            exits = [f"{e['ticker']} ({e['reason']})" for e in recs['exit_signals']]
            lines.append(f"EXIT SIGNALS: {', '.join(exits)}")

        # Position count
        pos_count = len(recs['current_positions'])
        if pos_count > 0:
            lines.append(f"Currently holding {pos_count} position(s).")

        return ' '.join(lines)

    def print_report(self, recs: Dict = None):
        """Print formatted recommendation report."""
        if recs is None:
            recs = self.generate_recommendations()

        ctx = recs['market_context']

        print("=" * 70)
        print("PROTEUS DAILY RECOMMENDATION")
        print("=" * 70)
        print(f"Generated: {recs['timestamp'][:19]}")
        print()

        # Market Context
        print("MARKET CONTEXT")
        print("-" * 40)
        print(f"  Regime: {ctx['regime'].upper()} ({ctx['regime_confidence']:.0%} confidence)")
        print(f"  VIX: {ctx['vix']:.1f}")
        print(f"  Bear Score: {ctx['bear_score']:.0f}/100 ({ctx['bear_level']})")
        if ctx['is_choppy']:
            print("  ** CHOPPY MARKET - Lower win rates expected **")
        print()

        # High Confidence
        print("HIGH CONFIDENCE RECOMMENDATIONS")
        print("-" * 40)
        if recs['high_confidence']:
            for r in recs['high_confidence']:
                print(f"  {r['ticker']} - BUY")
                print(f"    Signal: {r['raw_signal']:.0f} -> {r['adjusted_signal']:.0f}")
                print(f"    Tier: {r['tier'].upper()}")
                print(f"    Size: {r['size_pct']:.1f}% (${r['dollar_size']:,.0f})")
                if r['strengths']:
                    print(f"    + {r['strengths'][0]}")
                print()
        else:
            print("  None - No signals meet high-confidence criteria")
            print()

        # Moderate Confidence
        print("MODERATE CONFIDENCE (Optional)")
        print("-" * 40)
        if recs['moderate_confidence']:
            for r in recs['moderate_confidence']:
                print(f"  {r['ticker']}")
                print(f"    Signal: {r['raw_signal']:.0f} -> {r['adjusted_signal']:.0f} | Tier: {r['tier']}")
                if r['concerns']:
                    print(f"    - {r['concerns'][0]}")
                print()
        else:
            print("  None")
            print()

        # Current Positions
        if recs['current_positions']:
            print("CURRENT POSITIONS")
            print("-" * 40)
            for p in recs['current_positions']:
                pnl_sign = '+' if p['pnl_pct'] >= 0 else ''
                print(f"  {p['ticker']}: ${p['entry_price']:.2f} -> ${p['current_price']:.2f} ({pnl_sign}{p['pnl_pct']:.1f}%) Day {p['days_held']}")
            print()

        # Exit Signals
        if recs['exit_signals']:
            print("ACTION REQUIRED - EXIT SIGNALS")
            print("-" * 40)
            for e in recs['exit_signals']:
                print(f"  ** {e['ticker']}: {e['reason']} ({e['pnl_pct']:+.1f}%) **")
            print()

        # Summary
        print("=" * 70)
        print("SUMMARY")
        print("-" * 40)
        print(f"  {recs['summary']}")
        print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Proteus Daily Recommendations')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress scanner output, show only recommendations')
    parser.add_argument('--json', action='store_true',
                       help='Output as JSON')
    args = parser.parse_args()

    advisor = ProteusAdvisor(quiet=args.quiet)

    if args.quiet:
        # Suppress all scanner output
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            recs = advisor.generate_recommendations()
        if args.json:
            print(json.dumps(recs, indent=2, default=str))
        else:
            advisor.print_report(recs)
    else:
        if args.json:
            recs = advisor.generate_recommendations()
            print(json.dumps(recs, indent=2, default=str))
        else:
            advisor.print_report()


if __name__ == '__main__':
    main()

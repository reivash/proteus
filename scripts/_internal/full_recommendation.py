"""
Full Proteus Recommendation System

Combines:
1. Technical signals (MLP, LSTM, Transformer ensemble)
2. Regime detection (HMM, correlation-based)
3. Enhanced market signals (sentiment, breadth, cross-asset)
4. Historical edge data

Provides actionable recommendation with full reasoning.
"""

import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

from contextlib import redirect_stdout, redirect_stderr
import io


@dataclass
class FullRecommendation:
    """Complete recommendation with all factors."""
    timestamp: str

    # Market assessment
    market_signal: str  # strong_buy, buy, neutral, sell, strong_sell
    market_confidence: float
    market_factors: List[str]

    # Regime assessment
    regime: str
    regime_confidence: float
    regime_edge: float  # Historical win rate for this regime

    # Technical signals
    top_signals: List[Dict]

    # Current positions
    positions: List[Dict]
    exit_signals: List[Dict]

    # Final recommendation
    action: str  # BUY, HOLD, SELL, WAIT
    tickers: List[str]
    reasoning: List[str]
    risk_level: str  # low, medium, high


def get_full_recommendation() -> FullRecommendation:
    """Generate comprehensive recommendation."""

    # 1. Get enhanced market signals
    from data.signals.market_signals import MarketSignalGenerator

    market_gen = MarketSignalGenerator()

    # Suppress output
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        market_signals = market_gen.generate_signals()

    # 2. Get Proteus technical signals
    from trading.smart_scanner_v2 import SmartScannerV2

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        scanner = SmartScannerV2()
        scan_result = scanner.scan()

    # 3. Load positions
    wallet_path = Path(__file__).parent.parent / 'data' / 'virtual_wallet' / 'wallet_state.json'
    positions = []
    if wallet_path.exists():
        with open(wallet_path) as f:
            wallet = json.load(f)
            for ticker, pos in wallet.get('positions', {}).items():
                positions.append({
                    'ticker': ticker,
                    'entry': pos['entry_price'],
                    'pnl_pct': pos.get('current_pnl_pct', 0),
                    'days': pos.get('days_held', 0)
                })

    # 4. Historical regime edge
    REGIME_EDGE = {
        'bear': 0.69,
        'volatile': 0.62,
        'choppy': 0.56,
        'bull': 0.51
    }

    regime = scan_result.regime.lower()
    regime_edge = REGIME_EDGE.get(regime, 0.55)

    # 5. Extract top technical signals
    top_signals = []
    for sig in scan_result.signals[:5]:
        top_signals.append({
            'ticker': sig.ticker,
            'raw': sig.raw_strength,
            'adjusted': sig.adjusted_strength,
            'tier': sig.tier
        })

    # 6. Exit signals
    exit_signals = []
    for action in scan_result.rebalance_actions:
        exit_signals.append({
            'ticker': action.get('ticker'),
            'reason': action.get('reason'),
            'pnl': action.get('pnl_pct', 0)
        })

    # 7. Determine final action
    reasoning = []

    # Market-level assessment
    if market_signals.overall_signal in ['strong_sell', 'sell']:
        market_bearish = True
        reasoning.append(f"Market sentiment BEARISH: {', '.join(market_signals.factors[:2])}")
    elif market_signals.overall_signal in ['strong_buy', 'buy']:
        market_bearish = False
        reasoning.append(f"Market sentiment BULLISH: {', '.join(market_signals.factors[:2])}")
    else:
        market_bearish = None
        reasoning.append("Market sentiment NEUTRAL")

    # Regime assessment
    if regime in ['bear', 'volatile']:
        regime_favorable = True
        reasoning.append(f"Regime FAVORABLE: {regime.upper()} has {regime_edge:.0%} historical win rate")
    elif regime == 'choppy':
        regime_favorable = False
        reasoning.append(f"Regime UNFAVORABLE: CHOPPY has only {regime_edge:.0%} win rate")
    else:
        regime_favorable = False
        reasoning.append(f"Regime POOR: BULL regime has {regime_edge:.0%} win rate for mean reversion")

    # Technical signals
    if len(top_signals) > 0 and top_signals[0]['adjusted'] >= 65:
        has_strong_signal = True
        reasoning.append(f"Strong technical signal: {top_signals[0]['ticker']} at {top_signals[0]['adjusted']:.0f}")
    else:
        has_strong_signal = False
        if len(top_signals) > 0:
            reasoning.append(f"Weak technical signals: best is {top_signals[0]['ticker']} at {top_signals[0]['adjusted']:.0f}")
        else:
            reasoning.append("No actionable technical signals")

    # Determine action
    if exit_signals:
        action = 'SELL'
        tickers = [e['ticker'] for e in exit_signals]
        reasoning.append(f"EXIT REQUIRED: {', '.join(tickers)} hit targets")
    elif market_bearish and not regime_favorable:
        action = 'WAIT'
        tickers = []
        reasoning.append("Both market sentiment and regime unfavorable - WAIT for better setup")
    elif regime_favorable and has_strong_signal and not market_bearish:
        action = 'BUY'
        tickers = [s['ticker'] for s in top_signals if s['adjusted'] >= 65]
        reasoning.append(f"Conditions align - consider {', '.join(tickers)}")
    elif has_strong_signal and market_signals.overall_signal == 'neutral':
        action = 'HOLD'
        tickers = []
        reasoning.append("Mixed signals - hold existing positions, no new entries")
    else:
        action = 'WAIT'
        tickers = []
        reasoning.append("Insufficient edge - preserve capital")

    # Risk level
    if market_signals.fear_greed_signal == 'extreme_greed' or market_signals.credit_signal == 'stress':
        risk = 'high'
    elif market_signals.breadth_signal == 'deteriorating':
        risk = 'high'
    elif market_signals.fear_greed_signal in ['fear', 'extreme_fear']:
        risk = 'low'
    else:
        risk = 'medium'

    return FullRecommendation(
        timestamp=datetime.now().isoformat(),
        market_signal=market_signals.overall_signal,
        market_confidence=market_signals.confidence,
        market_factors=market_signals.factors,
        regime=regime,
        regime_confidence=scan_result.regime_confidence,
        regime_edge=regime_edge,
        top_signals=top_signals,
        positions=positions,
        exit_signals=exit_signals,
        action=action,
        tickers=tickers,
        reasoning=reasoning,
        risk_level=risk
    )


def print_recommendation(rec: FullRecommendation):
    """Print formatted recommendation."""

    print("=" * 70)
    print("PROTEUS FULL RECOMMENDATION")
    print("=" * 70)
    print(f"Generated: {rec.timestamp[:19]}")
    print()

    # Market Assessment
    print("MARKET ASSESSMENT")
    print("-" * 40)
    signal_marker = {
        'strong_buy': '++', 'buy': '+', 'neutral': '=',
        'sell': '-', 'strong_sell': '--'
    }
    print(f"  Signal: {rec.market_signal.upper()} [{signal_marker.get(rec.market_signal, '')}]")
    print(f"  Confidence: {rec.market_confidence:.0%}")
    for factor in rec.market_factors[:3]:
        print(f"    • {factor}")
    print()

    # Regime Assessment
    print("REGIME ASSESSMENT")
    print("-" * 40)
    print(f"  Current: {rec.regime.upper()}")
    print(f"  Historical Win Rate: {rec.regime_edge:.0%}")
    print(f"  Detection Confidence: {rec.regime_confidence:.0%}")
    print()

    # Current Positions
    if rec.positions:
        print("CURRENT POSITIONS")
        print("-" * 40)
        for p in rec.positions:
            pnl = p['pnl_pct']
            sign = '+' if pnl >= 0 else ''
            print(f"  {p['ticker']}: {sign}{pnl:.1f}% (day {p['days']})")
        print()

    # Exit Signals
    if rec.exit_signals:
        print("*** EXIT SIGNALS ***")
        print("-" * 40)
        for e in rec.exit_signals:
            print(f"  ** {e['ticker']}: {e['reason']} ({e['pnl']:+.1f}%) **")
        print()

    # Top Technical Signals
    if rec.top_signals:
        print("TOP TECHNICAL SIGNALS")
        print("-" * 40)
        for s in rec.top_signals[:5]:
            status = "[Y]" if s['adjusted'] >= 65 else "[N]"
            print(f"  {status} {s['ticker']}: {s['adjusted']:.0f} (raw: {s['raw']:.0f}, {s['tier']})")
        print()

    # Final Recommendation
    print("=" * 70)
    action_style = {
        'BUY': '>>> BUY <<<', 'SELL': '>>> SELL <<<',
        'HOLD': '--- HOLD ---', 'WAIT': '... WAIT ...'
    }
    print(f"ACTION: {action_style.get(rec.action, rec.action)}")

    if rec.tickers:
        print(f"TICKERS: {', '.join(rec.tickers)}")

    print(f"RISK LEVEL: {rec.risk_level.upper()}")
    print("=" * 70)
    print()

    print("REASONING:")
    for i, reason in enumerate(rec.reasoning, 1):
        print(f"  {i}. {reason}")
    print()


def main():
    print("Analyzing all data sources...")
    print()

    rec = get_full_recommendation()
    print_recommendation(rec)


if __name__ == '__main__':
    main()

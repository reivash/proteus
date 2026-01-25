"""
Regime Transition Analysis Script

Analyzes the last 50+ regime transitions to measure:
1. Early detection rate (HMM detected before it was obvious)
2. On-time detection rate
3. Late detection rate
4. Patterns in early vs late detection

This script compares HMM predictions to "hindsight truth" based on
subsequent price action.

Author: Claude Opus 4.5
Date: January 25, 2026
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Add common to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / 'common'))

import yfinance as yf


@dataclass
class RegimeTransition:
    """Represents a regime transition."""
    date: str
    from_regime: str
    to_regime: str
    hmm_detection_date: str
    obvious_date: str  # When it became obvious via rule-based
    detection_lead_days: int  # Positive = early, negative = late
    detection_quality: str  # "early", "on_time", "late"
    confidence_at_detection: float
    vix_at_transition: float
    spy_return_5d: float  # 5-day forward return from transition


@dataclass
class TransitionAnalysis:
    """Results of transition analysis."""
    total_transitions: int
    early_detections: int
    on_time_detections: int
    late_detections: int
    early_rate: float
    avg_lead_days: float
    transitions: List[RegimeTransition]


def get_hindsight_regime(spy_data: pd.DataFrame, idx: int, lookback: int = 20) -> str:
    """
    Determine "true" regime using hindsight (looking at what happened after).

    This is the ground truth we compare HMM predictions against.
    """
    if idx < lookback or idx >= len(spy_data) - 5:
        return "unknown"

    # Get data around this point
    past = spy_data.iloc[idx-lookback:idx]
    future = spy_data.iloc[idx:idx+5]

    # Calculate metrics
    past_return = (past['Close'].iloc[-1] / past['Close'].iloc[0] - 1) * 100
    past_vol = past['Close'].pct_change().std() * np.sqrt(252) * 100
    future_return = (future['Close'].iloc[-1] / future['Close'].iloc[0] - 1) * 100

    # Hindsight regime classification
    # VOLATILE: High volatility regardless of direction
    if past_vol > 30:  # Annualized vol > 30%
        return "volatile"

    # BEAR: Declining with more decline ahead
    if past_return < -5 and future_return < -1:
        return "bear"

    # BULL: Rising with more upside ahead
    if past_return > 5 and future_return > 0:
        return "bull"

    # CHOPPY: Everything else
    return "choppy"


def get_rule_based_regime(spy_data: pd.DataFrame, vix_data: pd.DataFrame, idx: int) -> Tuple[str, float]:
    """
    Simple rule-based regime detection (what most traders would see).
    Returns (regime, confidence).
    """
    if idx < 50:
        return "unknown", 0.0

    close = spy_data['Close'].iloc[:idx+1]

    # Trends
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    current = close.iloc[-1]

    trend_20d = (current / close.iloc[-20] - 1) * 100
    trend_50d = (current / close.iloc[-50] - 1) * 100

    # VIX
    vix_level = vix_data['Close'].iloc[min(idx, len(vix_data)-1)]

    # Classification
    if vix_level > 30:
        return "volatile", 0.85

    if trend_20d > 3 and trend_50d > 5 and current > ma20 > ma50:
        return "bull", 0.75

    if trend_20d < -3 and trend_50d < -5 and current < ma20 < ma50:
        return "bear", 0.75

    return "choppy", 0.60


def run_hmm_backtest(spy_data: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Run HMM on rolling windows to get historical regime predictions.
    """
    from analysis.hmm_regime_detector import HMMRegimeDetector

    detector = HMMRegimeDetector()

    results = []

    # Need at least 100 days for training + window
    start_idx = 100

    print(f"Running HMM on {len(spy_data) - start_idx} days...")

    for i in range(start_idx, len(spy_data)):
        if i % 50 == 0:
            print(f"  Processing day {i}/{len(spy_data)}...")

        # Use data up to this point
        historical = spy_data.iloc[:i+1]

        try:
            # Train if needed (first time or every 100 days)
            if i == start_idx or (i - start_idx) % 100 == 0:
                detector.train(historical.tail(500))  # Train on last 500 days

            # Detect regime
            result = detector.detect_regime(historical, lookback=window)

            results.append({
                'date': spy_data.index[i],
                'regime': result.regime,
                'confidence': result.confidence,
                'probabilities': result.probabilities,
                'days_in_regime': result.days_in_regime,
                'transition_signal': result.transition_signal
            })
        except Exception as e:
            results.append({
                'date': spy_data.index[i],
                'regime': 'unknown',
                'confidence': 0.0,
                'probabilities': {},
                'days_in_regime': 0,
                'transition_signal': 'error'
            })

    return pd.DataFrame(results)


def identify_transitions(regimes: pd.DataFrame) -> List[Dict]:
    """
    Identify regime transitions from a sequence of regime predictions.
    """
    transitions = []

    prev_regime = None
    prev_date = None

    for _, row in regimes.iterrows():
        if prev_regime is not None and row['regime'] != prev_regime:
            transitions.append({
                'date': row['date'],
                'from_regime': prev_regime,
                'to_regime': row['regime'],
                'confidence': row['confidence']
            })

        prev_regime = row['regime']
        prev_date = row['date']

    return transitions


def analyze_detection_timing(
    hmm_transitions: List[Dict],
    spy_data: pd.DataFrame,
    vix_data: pd.DataFrame
) -> List[RegimeTransition]:
    """
    For each HMM transition, determine if it was early, on-time, or late.
    """
    analyzed = []

    for trans in hmm_transitions:
        trans_date = trans['date']

        # Find the index in spy_data
        try:
            idx = spy_data.index.get_loc(trans_date)
        except KeyError:
            continue

        # Look for when rule-based would have detected this
        # Search forward up to 10 days
        rule_detection_idx = None
        for offset in range(-5, 11):  # 5 days before to 10 days after
            check_idx = idx + offset
            if check_idx < 50 or check_idx >= len(spy_data):
                continue

            rule_regime, _ = get_rule_based_regime(spy_data, vix_data, check_idx)
            if rule_regime == trans['to_regime']:
                rule_detection_idx = check_idx
                break

        if rule_detection_idx is None:
            # Rule-based never confirmed this transition (false positive?)
            lead_days = 0
            quality = "unconfirmed"
            obvious_date = str(trans_date)
        else:
            lead_days = rule_detection_idx - idx
            obvious_date = str(spy_data.index[rule_detection_idx])

            if lead_days > 1:
                quality = "early"
            elif lead_days >= -1:
                quality = "on_time"
            else:
                quality = "late"

        # Get context
        vix_level = vix_data['Close'].iloc[min(idx, len(vix_data)-1)] if idx < len(vix_data) else 20.0

        # Forward return
        if idx + 5 < len(spy_data):
            fwd_return = (spy_data['Close'].iloc[idx+5] / spy_data['Close'].iloc[idx] - 1) * 100
        else:
            fwd_return = 0.0

        analyzed.append(RegimeTransition(
            date=str(trans_date),
            from_regime=trans['from_regime'],
            to_regime=trans['to_regime'],
            hmm_detection_date=str(trans_date),
            obvious_date=obvious_date,
            detection_lead_days=lead_days,
            detection_quality=quality,
            confidence_at_detection=trans['confidence'],
            vix_at_transition=vix_level,
            spy_return_5d=fwd_return
        ))

    return analyzed


def run_analysis(years_back: int = 3) -> TransitionAnalysis:
    """
    Run full transition analysis.
    """
    print("=" * 70)
    print("REGIME TRANSITION ANALYSIS")
    print("=" * 70)
    print()

    # Fetch data
    print("[1] Fetching historical data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)

    spy = yf.Ticker("SPY")
    spy_data = spy.history(start=start_date, end=end_date)

    vix = yf.Ticker("^VIX")
    vix_data = vix.history(start=start_date, end=end_date)

    print(f"    SPY: {len(spy_data)} days")
    print(f"    VIX: {len(vix_data)} days")

    # Run HMM backtest
    print()
    print("[2] Running HMM regime detection...")
    hmm_results = run_hmm_backtest(spy_data)

    # Identify transitions
    print()
    print("[3] Identifying transitions...")
    transitions = identify_transitions(hmm_results)
    print(f"    Found {len(transitions)} transitions")

    # Analyze timing
    print()
    print("[4] Analyzing detection timing...")
    analyzed = analyze_detection_timing(transitions, spy_data, vix_data)

    # Calculate statistics
    early = [t for t in analyzed if t.detection_quality == "early"]
    on_time = [t for t in analyzed if t.detection_quality == "on_time"]
    late = [t for t in analyzed if t.detection_quality == "late"]

    early_rate = len(early) / len(analyzed) * 100 if analyzed else 0
    avg_lead = np.mean([t.detection_lead_days for t in analyzed]) if analyzed else 0

    result = TransitionAnalysis(
        total_transitions=len(analyzed),
        early_detections=len(early),
        on_time_detections=len(on_time),
        late_detections=len(late),
        early_rate=early_rate,
        avg_lead_days=avg_lead,
        transitions=analyzed
    )

    return result


def print_results(analysis: TransitionAnalysis):
    """Print analysis results."""
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    print(f"Total transitions analyzed: {analysis.total_transitions}")
    print()

    print("Detection Timing:")
    print(f"  Early (1+ days ahead):  {analysis.early_detections:3d} ({analysis.early_detections/analysis.total_transitions*100:.1f}%)")
    print(f"  On-time (within 1 day): {analysis.on_time_detections:3d} ({analysis.on_time_detections/analysis.total_transitions*100:.1f}%)")
    print(f"  Late (1+ days behind):  {analysis.late_detections:3d} ({analysis.late_detections/analysis.total_transitions*100:.1f}%)")
    print()

    print(f"Average lead time: {analysis.avg_lead_days:+.1f} days")
    print()

    # Breakdown by transition type
    print("By Transition Type:")
    type_counts = {}
    for t in analysis.transitions:
        key = f"{t.from_regime} -> {t.to_regime}"
        if key not in type_counts:
            type_counts[key] = {'total': 0, 'early': 0}
        type_counts[key]['total'] += 1
        if t.detection_quality == 'early':
            type_counts[key]['early'] += 1

    for trans_type, counts in sorted(type_counts.items(), key=lambda x: -x[1]['total']):
        early_pct = counts['early'] / counts['total'] * 100
        print(f"  {trans_type:<20} {counts['total']:3d} transitions, {early_pct:.0f}% early")

    print()

    # Recent transitions
    print("Last 10 Transitions:")
    print(f"{'Date':<12} {'From':<10} {'To':<10} {'Lead':>6} {'Quality':<12} {'VIX':>6}")
    print("-" * 60)
    for t in analysis.transitions[-10:]:
        print(f"{t.date[:10]:<12} {t.from_regime:<10} {t.to_regime:<10} "
              f"{t.detection_lead_days:+5d}d {t.detection_quality:<12} {t.vix_at_transition:6.1f}")


def save_results(analysis: TransitionAnalysis, output_dir: str = None):
    """Save results to files."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / 'research'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_file = output_dir / 'transition_analysis.json'
    with open(json_file, 'w') as f:
        json.dump({
            'summary': {
                'total_transitions': analysis.total_transitions,
                'early_detections': analysis.early_detections,
                'on_time_detections': analysis.on_time_detections,
                'late_detections': analysis.late_detections,
                'early_rate': analysis.early_rate,
                'avg_lead_days': analysis.avg_lead_days
            },
            'transitions': [asdict(t) for t in analysis.transitions]
        }, f, indent=2, default=str)

    print(f"\n[SAVED] {json_file}")

    # Save markdown report
    md_file = output_dir / 'transition_analysis.md'
    with open(md_file, 'w') as f:
        f.write("# Regime Transition Analysis\n\n")
        f.write(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("---\n\n")

        f.write("## Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total Transitions | {analysis.total_transitions} |\n")
        f.write(f"| Early Detections | {analysis.early_detections} ({analysis.early_rate:.1f}%) |\n")
        f.write(f"| On-Time Detections | {analysis.on_time_detections} |\n")
        f.write(f"| Late Detections | {analysis.late_detections} |\n")
        f.write(f"| Avg Lead Time | {analysis.avg_lead_days:+.1f} days |\n\n")

        f.write("## Key Findings\n\n")
        if analysis.early_rate >= 50:
            f.write("- HMM achieves good early detection (50%+)\n")
        else:
            f.write("- HMM early detection needs improvement (<50%)\n")

        f.write("\n## All Transitions\n\n")
        f.write("| Date | From | To | Lead | Quality | VIX |\n")
        f.write("|------|------|----|----|---------|-----|\n")
        for t in analysis.transitions:
            f.write(f"| {t.date[:10]} | {t.from_regime} | {t.to_regime} | "
                   f"{t.detection_lead_days:+d}d | {t.detection_quality} | {t.vix_at_transition:.1f} |\n")

    print(f"[SAVED] {md_file}")


def main():
    """Main entry point."""
    analysis = run_analysis(years_back=3)
    print_results(analysis)
    save_results(analysis)


if __name__ == "__main__":
    main()

"""
Regime Misclassification Analysis

Identifies the worst regime misclassifications and root causes:
- Bull calls before major selloffs
- Bear calls before rallies
- Choppy calls during strong trends

Author: Claude Opus 4.5
Date: January 25, 2026
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / 'common'))

import yfinance as yf


@dataclass
class Misclassification:
    """Represents a regime misclassification."""
    date: str
    hmm_regime: str
    hindsight_regime: str
    confidence: float
    severity: float  # Higher = worse mistake
    forward_return_5d: float
    forward_return_10d: float
    vix_level: float
    root_cause: str
    description: str


def get_hindsight_truth(spy_data: pd.DataFrame, idx: int) -> Tuple[str, float]:
    """
    Determine the "true" regime using what actually happened.

    This is 20/20 hindsight - we look at what followed to judge
    what the regime "should have been".
    """
    if idx >= len(spy_data) - 10:
        return "unknown", 0.0

    # Forward returns
    fwd_5d = (spy_data['Close'].iloc[idx+5] / spy_data['Close'].iloc[idx] - 1) * 100
    fwd_10d = (spy_data['Close'].iloc[idx+10] / spy_data['Close'].iloc[idx] - 1) * 100

    # Volatility of next 10 days
    fwd_vol = spy_data['Close'].iloc[idx:idx+10].pct_change().std() * np.sqrt(252) * 100

    # Classification based on outcomes
    if fwd_vol > 35:
        return "volatile", fwd_vol
    elif fwd_5d < -3 and fwd_10d < -5:
        return "bear", abs(fwd_10d)
    elif fwd_5d > 3 and fwd_10d > 5:
        return "bull", fwd_10d
    else:
        return "choppy", abs(fwd_5d)


def calculate_severity(hmm_regime: str, truth_regime: str, forward_return: float) -> float:
    """
    Calculate how bad a misclassification was.

    Severity considers:
    1. How wrong the call was (bull vs bear is worse than choppy vs bull)
    2. How much money you'd lose acting on bad advice
    """
    # Base severity by regime mismatch
    mismatch_severity = {
        ('bull', 'bear'): 3.0,   # Called bull, was bear - worst!
        ('bear', 'bull'): 2.5,   # Called bear, was bull - missed gains
        ('choppy', 'bear'): 2.0, # Didn't see bear coming
        ('bull', 'volatile'): 2.0,
        ('choppy', 'bull'): 1.5, # Missed bull - but less harm
        ('bull', 'choppy'): 1.0,
        ('bear', 'choppy'): 1.0,
        ('volatile', 'choppy'): 0.5,
    }

    base = mismatch_severity.get((hmm_regime, truth_regime), 1.0)

    # Scale by actual loss (if you acted on bad advice)
    if hmm_regime == 'bull' and forward_return < 0:
        # Called bull, market went down - loss
        impact = abs(forward_return) / 5  # Normalize to ~1
    elif hmm_regime == 'bear' and forward_return > 0:
        # Called bear, market went up - missed gain
        impact = forward_return / 5
    else:
        impact = 1.0

    return base * impact


def identify_root_cause(
    spy_data: pd.DataFrame,
    vix_data: pd.DataFrame,
    idx: int,
    hmm_regime: str,
    truth_regime: str
) -> Tuple[str, str]:
    """
    Identify why the HMM made a mistake.
    """
    if idx < 20:
        return "insufficient_data", "Not enough history"

    # Get context
    past_return_10d = (spy_data['Close'].iloc[idx] / spy_data['Close'].iloc[idx-10] - 1) * 100
    past_vol = spy_data['Close'].iloc[idx-20:idx].pct_change().std() * np.sqrt(252) * 100
    vix = vix_data['Close'].iloc[min(idx, len(vix_data)-1)]
    vol_ratio = spy_data['Volume'].iloc[idx] / spy_data['Volume'].iloc[idx-20:idx].mean()

    # Root cause analysis
    if hmm_regime == 'bull' and truth_regime == 'bear':
        if past_return_10d > 2:
            return "momentum_lag", f"HMM lagged momentum reversal. 10d return was +{past_return_10d:.1f}% before selloff."
        elif vix < 20:
            return "low_vix_trap", f"Low VIX ({vix:.1f}) masked building risk."
        elif past_vol < 15:
            return "vol_compression", f"Low volatility ({past_vol:.1f}%) before expansion."
        else:
            return "feature_blindness", "HMM features didn't capture regime shift."

    elif hmm_regime == 'bear' and truth_regime == 'bull':
        if past_return_10d < -2:
            return "momentum_lag", f"HMM lagged momentum reversal. 10d return was {past_return_10d:.1f}% before rally."
        elif vix > 25:
            return "vix_overweight", f"High VIX ({vix:.1f}) caused false bear signal during recovery."
        else:
            return "late_transition", "HMM was slow to recognize bear-to-bull transition."

    elif hmm_regime == 'choppy' and truth_regime in ['bear', 'bull']:
        if abs(past_return_10d) < 3:
            return "trend_blindness", f"HMM missed emerging trend. Prior 10d return: {past_return_10d:+.1f}%"
        else:
            return "sensitivity_low", "HMM threshold too high for trend detection."

    elif hmm_regime in ['bull', 'volatile'] and truth_regime == 'volatile':
        return "vol_underweight", f"HMM underweighted volatility. Realized vol: {past_vol:.1f}%"

    return "unknown", "Unable to determine root cause."


def run_misclassification_analysis(years_back: int = 3) -> List[Misclassification]:
    """
    Find the worst misclassifications.
    """
    print("=" * 70)
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 70)
    print()

    # Fetch data
    print("[1] Fetching data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)

    spy = yf.Ticker("SPY")
    spy_data = spy.history(start=start_date, end=end_date)

    vix = yf.Ticker("^VIX")
    vix_data = vix.history(start=start_date, end=end_date)

    print(f"    SPY: {len(spy_data)} days")

    # Load transition analysis results (has HMM regime calls)
    print("[2] Loading HMM regime history...")
    transition_file = Path(__file__).resolve().parent.parent / 'research' / 'transition_analysis.json'

    if not transition_file.exists():
        print("    Running transition analysis first...")
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'analyze_transitions.py')])

    with open(transition_file, 'r') as f:
        transition_data = json.load(f)

    # Build HMM regime timeline from transitions
    transitions = transition_data['transitions']
    print(f"    Found {len(transitions)} transitions")

    # Analyze each significant period
    print("[3] Analyzing misclassifications...")

    misclassifications = []

    # Check key dates and significant market moves
    for idx in range(50, len(spy_data) - 10):
        # Only check every 5th day to reduce compute
        if idx % 5 != 0:
            continue

        date = spy_data.index[idx]
        date_str = str(date)[:10]

        # Find HMM regime for this date
        hmm_regime = "unknown"
        hmm_confidence = 0.5

        for i, t in enumerate(transitions):
            trans_date = t['date'][:10]
            if trans_date <= date_str:
                hmm_regime = t['to_regime']
                hmm_confidence = t['confidence_at_detection']
            else:
                break

        if hmm_regime == "unknown":
            continue

        # Get hindsight truth
        truth_regime, truth_score = get_hindsight_truth(spy_data, idx)

        if truth_regime == "unknown":
            continue

        # Check for mismatch
        if hmm_regime != truth_regime:
            # Calculate forward returns
            fwd_5d = (spy_data['Close'].iloc[idx+5] / spy_data['Close'].iloc[idx] - 1) * 100
            fwd_10d = (spy_data['Close'].iloc[idx+10] / spy_data['Close'].iloc[idx] - 1) * 100

            vix_level = vix_data['Close'].iloc[min(idx, len(vix_data)-1)]

            severity = calculate_severity(hmm_regime, truth_regime, fwd_10d)

            root_cause, description = identify_root_cause(
                spy_data, vix_data, idx, hmm_regime, truth_regime
            )

            misclassifications.append(Misclassification(
                date=date_str,
                hmm_regime=hmm_regime,
                hindsight_regime=truth_regime,
                confidence=hmm_confidence,
                severity=severity,
                forward_return_5d=fwd_5d,
                forward_return_10d=fwd_10d,
                vix_level=vix_level,
                root_cause=root_cause,
                description=description
            ))

    # Sort by severity and get top 5
    misclassifications.sort(key=lambda x: -x.severity)

    return misclassifications


def print_worst_misclassifications(misclassifications: List[Misclassification], top_n: int = 5):
    """Print the worst misclassifications."""
    print()
    print("=" * 70)
    print(f"TOP {top_n} WORST MISCLASSIFICATIONS")
    print("=" * 70)
    print()

    for i, m in enumerate(misclassifications[:top_n], 1):
        print(f"#{i} - {m.date}")
        print(f"   HMM said: {m.hmm_regime.upper()} (confidence: {m.confidence:.0%})")
        print(f"   Reality:  {m.hindsight_regime.upper()}")
        print(f"   Severity: {m.severity:.2f}")
        print(f"   Forward returns: 5d={m.forward_return_5d:+.1f}%, 10d={m.forward_return_10d:+.1f}%")
        print(f"   VIX: {m.vix_level:.1f}")
        print(f"   Root cause: {m.root_cause}")
        print(f"   Details: {m.description}")
        print()

    # Summary statistics
    print("-" * 70)
    print("ROOT CAUSE SUMMARY")
    print("-" * 70)

    cause_counts = {}
    for m in misclassifications:
        cause = m.root_cause
        if cause not in cause_counts:
            cause_counts[cause] = 0
        cause_counts[cause] += 1

    for cause, count in sorted(cause_counts.items(), key=lambda x: -x[1]):
        pct = count / len(misclassifications) * 100
        print(f"  {cause:<20} {count:4d} ({pct:5.1f}%)")


def save_results(misclassifications: List[Misclassification]):
    """Save analysis results."""
    output_dir = Path(__file__).resolve().parent.parent / 'research'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_file = output_dir / 'misclassification_analysis.json'
    with open(json_file, 'w') as f:
        json.dump({
            'total_misclassifications': len(misclassifications),
            'top_5': [asdict(m) for m in misclassifications[:5]],
            'all': [asdict(m) for m in misclassifications]
        }, f, indent=2)
    print(f"\n[SAVED] {json_file}")

    # Save markdown report
    md_file = output_dir / 'misclassification_analysis.md'
    with open(md_file, 'w') as f:
        f.write("# Regime Misclassification Analysis\n\n")
        f.write(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("---\n\n")

        f.write("## Top 5 Worst Misclassifications\n\n")

        for i, m in enumerate(misclassifications[:5], 1):
            f.write(f"### #{i}: {m.date}\n\n")
            f.write(f"| Aspect | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| HMM Regime | {m.hmm_regime.upper()} |\n")
            f.write(f"| Actual Regime | {m.hindsight_regime.upper()} |\n")
            f.write(f"| Severity | {m.severity:.2f} |\n")
            f.write(f"| 5-day Return | {m.forward_return_5d:+.1f}% |\n")
            f.write(f"| 10-day Return | {m.forward_return_10d:+.1f}% |\n")
            f.write(f"| VIX | {m.vix_level:.1f} |\n")
            f.write(f"| Root Cause | {m.root_cause} |\n\n")
            f.write(f"**Analysis**: {m.description}\n\n")

        f.write("---\n\n")
        f.write("## Root Cause Summary\n\n")

        cause_counts = {}
        for m in misclassifications:
            cause = m.root_cause
            if cause not in cause_counts:
                cause_counts[cause] = 0
            cause_counts[cause] += 1

        f.write("| Root Cause | Count | % |\n")
        f.write("|------------|-------|---|\n")
        for cause, count in sorted(cause_counts.items(), key=lambda x: -x[1]):
            pct = count / len(misclassifications) * 100
            f.write(f"| {cause} | {count} | {pct:.1f}% |\n")

        f.write("\n---\n\n")
        f.write("## Recommendations\n\n")
        f.write("Based on root cause analysis:\n\n")

        if 'momentum_lag' in cause_counts:
            f.write("1. **Add momentum divergence detection** - HMM is slow to recognize reversals\n")
        if 'low_vix_trap' in cause_counts or 'vix_overweight' in cause_counts:
            f.write("2. **Improve VIX integration** - Either missing VIX signals or overweighting them\n")
        if 'vol_compression' in cause_counts:
            f.write("3. **Add volatility regime feature** - Compression before expansion is predictive\n")
        if 'trend_blindness' in cause_counts:
            f.write("4. **Lower trend detection threshold** - Missing emerging trends\n")

    print(f"[SAVED] {md_file}")


def main():
    """Main entry point."""
    misclassifications = run_misclassification_analysis(years_back=3)
    print_worst_misclassifications(misclassifications, top_n=5)
    save_results(misclassifications)


if __name__ == "__main__":
    main()

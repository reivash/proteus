"""
Regime Detection Performance Dashboard

Generates a comprehensive performance view:
1. Current regime with confidence and context
2. Historical accuracy metrics
3. Recent transitions and trends
4. Feature contributions
5. Model agreement/disagreement patterns

Author: Claude Opus 4.5
Date: January 25, 2026
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / 'common'))


def load_research_data() -> Dict:
    """Load all research analysis results."""
    research_dir = Path(__file__).resolve().parent.parent / 'research'

    data = {}

    # Transition analysis
    trans_file = research_dir / 'transition_analysis.json'
    if trans_file.exists():
        with open(trans_file, 'r') as f:
            data['transitions'] = json.load(f)

    # Misclassification analysis
    misclass_file = research_dir / 'misclassification_analysis.json'
    if misclass_file.exists():
        with open(misclass_file, 'r') as f:
            data['misclassifications'] = json.load(f)

    # Benchmark results
    bench_file = research_dir / 'benchmark_results.json'
    if bench_file.exists():
        with open(bench_file, 'r') as f:
            data['benchmark'] = json.load(f)

    return data


def get_current_regime() -> Dict:
    """Get current regime detection result."""
    try:
        from analysis.unified_regime_detector import UnifiedRegimeDetector, DetectionMethod

        detector = UnifiedRegimeDetector(method=DetectionMethod.ENSEMBLE)
        result = detector.detect_regime()

        return {
            'regime': result.regime,
            'confidence': result.confidence,
            'hmm_regime': result.hmm_regime,
            'hmm_confidence': result.hmm_confidence,
            'rule_regime': result.rule_regime,
            'rule_confidence': result.rule_confidence,
            'agreement': result.agreement,
            'vix_level': result.vix_level,
            'vix_3m': result.vix_3m,
            'vix_term_structure': result.vix_term_structure,
            'days_in_regime': result.days_in_regime,
            'transition_signal': result.transition_signal,
            'hmm_probabilities': result.hmm_probabilities,
            'early_warning_score': result.early_warning_score,
            'early_warning_level': result.early_warning_level,
            'meta_regime': result.meta_regime,
            'recession_signal': result.recession_signal,
            'model_disagreement': result.model_disagreement,
            'recommendation': result.recommendation
        }
    except Exception as e:
        return {'error': str(e)}


def get_regime_history() -> List[Dict]:
    """Load recent regime history."""
    history_file = Path(__file__).resolve().parent.parent / 'data' / 'regime_history.json'

    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
            return history[-30:]  # Last 30 entries
    return []


def calculate_recent_metrics(history: List[Dict]) -> Dict:
    """Calculate metrics from recent history."""
    if len(history) < 2:
        return {}

    # Regime distribution
    regime_counts = {}
    for entry in history:
        r = entry.get('regime', 'unknown')
        regime_counts[r] = regime_counts.get(r, 0) + 1

    # Transitions
    transitions = 0
    for i in range(1, len(history)):
        if history[i].get('regime') != history[i-1].get('regime'):
            transitions += 1

    # Agreement rate
    agreements = sum(1 for e in history if e.get('agreement', False))
    agreement_rate = agreements / len(history) * 100 if history else 0

    # Average confidence
    avg_confidence = sum(e.get('confidence', 0) for e in history) / len(history) * 100 if history else 0

    return {
        'period_days': len(history),
        'regime_distribution': regime_counts,
        'total_transitions': transitions,
        'stability_days': len(history) / (transitions + 1),
        'agreement_rate': agreement_rate,
        'avg_confidence': avg_confidence
    }


def generate_dashboard():
    """Generate and display performance dashboard."""
    print()
    print("=" * 80)
    print("           REGIME DETECTION PERFORMANCE DASHBOARD")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. Current Regime
    print("-" * 80)
    print("1. CURRENT REGIME STATUS")
    print("-" * 80)

    current = get_current_regime()

    if 'error' not in current:
        regime = current['regime'].upper()
        conf = current['confidence'] * 100

        # Regime badge
        badge = {
            'volatile': '[!!!]',
            'bear': '[ ! ]',
            'choppy': '[ - ]',
            'bull': '[ + ]'
        }.get(current['regime'], '[   ]')

        print(f"\n  {badge} {regime} (Confidence: {conf:.0f}%)")
        print()

        # Details
        print(f"  HMM says:        {current['hmm_regime'].upper():<12} ({current['hmm_confidence']*100:.0f}%)")
        print(f"  Rule-based says: {current['rule_regime'].upper():<12} ({current['rule_confidence']*100:.0f}%)")
        print(f"  Agreement:       {'YES' if current['agreement'] else 'NO'}")
        print()

        print(f"  VIX:             {current['vix_level']:.1f}")
        print(f"  VIX 3M:          {current['vix_3m']:.1f}")
        print(f"  Term Structure:  {current['vix_term_structure']:.3f}", end="")
        if current['vix_term_structure'] > 1.05:
            print("  [BACKWARDATION - Fear]")
        elif current['vix_term_structure'] < 0.95:
            print("  [CONTANGO - Normal]")
        else:
            print("  [FLAT]")

        print(f"  Days in Regime:  {current['days_in_regime']}")
        print(f"  Transition:      {current['transition_signal']}")
        print()

        # HMM Probabilities
        print("  HMM Probabilities:")
        if current['hmm_probabilities']:
            for regime, prob in sorted(current['hmm_probabilities'].items(), key=lambda x: -x[1]):
                bar = "#" * int(prob * 20)
                print(f"    {regime.upper():<10} {prob*100:5.1f}% {bar}")
        print()

        # Early Warning
        print(f"  Early Warning:   {current['early_warning_score']:.0f}/100 ({current['early_warning_level']})")
        print(f"  Meta Regime:     {current['meta_regime']}")
        print(f"  Recession Risk:  {current['recession_signal']}")
        print(f"  Model Disagree:  {current['model_disagreement']*100:.0f}%")
    else:
        print(f"  ERROR: {current['error']}")

    # 2. Historical Performance
    print()
    print("-" * 80)
    print("2. HISTORICAL PERFORMANCE")
    print("-" * 80)

    research = load_research_data()

    # Benchmark results
    if 'benchmark' in research:
        bench = research['benchmark']
        print("\n  Benchmark Comparison (2-year backtest):")
        print(f"  {'Method':<15} {'Accuracy':>10} {'Stability':>12} {'FP Rate':>10}")
        print(f"  {'-'*47}")

        for method in ['HMM', 'Rule-based', 'Ensemble']:
            if method in bench:
                m = bench[method]
                print(f"  {method:<15} {m['accuracy']:>9.1f}% {m['regime_stability']:>10.1f}d {m['false_positive_rate']:>9.1f}%")

    # Transition analysis
    if 'transitions' in research:
        trans = research['transitions']
        summary = trans.get('summary', {})
        print(f"\n  Transition Analysis ({summary.get('total_transitions', 0)} transitions):")
        print(f"    Early Detection Rate: {summary.get('early_rate', 0):.1f}%")
        print(f"    Average Lead Time:    {summary.get('avg_lead_days', 0):+.1f} days")

    # 3. Recent Trends
    print()
    print("-" * 80)
    print("3. RECENT TRENDS (Last 30 Days)")
    print("-" * 80)

    history = get_regime_history()
    if history:
        metrics = calculate_recent_metrics(history)

        print(f"\n  Regime Distribution:")
        for regime, count in sorted(metrics.get('regime_distribution', {}).items(), key=lambda x: -x[1]):
            pct = count / metrics.get('period_days', 1) * 100
            bar = "#" * int(pct / 5)
            print(f"    {regime.upper():<10} {count:3d} days ({pct:5.1f}%) {bar}")

        print(f"\n  Transitions:     {metrics.get('total_transitions', 0)}")
        print(f"  Stability:       {metrics.get('stability_days', 0):.1f} days avg")
        print(f"  Agreement Rate:  {metrics.get('agreement_rate', 0):.0f}%")
        print(f"  Avg Confidence:  {metrics.get('avg_confidence', 0):.0f}%")

        # Recent regime sequence
        print(f"\n  Recent Sequence (last 10):")
        recent = history[-10:]
        seq = " -> ".join([e.get('regime', '?')[:3].upper() for e in recent])
        print(f"    {seq}")
    else:
        print("\n  No recent history available")

    # 4. Known Issues
    print()
    print("-" * 80)
    print("4. KNOWN ISSUES & RECOMMENDATIONS")
    print("-" * 80)

    if 'misclassifications' in research:
        misclass = research['misclassifications']
        top5 = misclass.get('top_5', [])

        print("\n  Top Misclassification Types:")

        # Count root causes
        causes = {}
        for m in misclass.get('all', []):
            c = m.get('root_cause', 'unknown')
            causes[c] = causes.get(c, 0) + 1

        for cause, count in sorted(causes.items(), key=lambda x: -x[1])[:5]:
            print(f"    - {cause}: {count} occurrences")

    print("\n  Improvement Priorities:")
    print("    1. Add VIX as direct HMM feature (currently indirect)")
    print("    2. Implement regime duration modeling")
    print("    3. Reduce jitter (current stability: 2.7 days)")
    print("    4. Improve bull->bear transition detection")

    # 5. Quick Stats
    print()
    print("-" * 80)
    print("5. QUICK REFERENCE")
    print("-" * 80)

    print("\n  Current Thresholds by Regime:")
    print("    BULL:     Signal >= 55")
    print("    VOLATILE: Signal >= 60")
    print("    CHOPPY:   Signal >= 65")
    print("    BEAR:     Signal >= 70")

    print("\n  Position Multipliers:")
    print("    BULL:     1.0x")
    print("    BEAR:     1.0x")
    print("    CHOPPY:   0.5x")
    print("    VOLATILE: 0.5x")

    print()
    print("=" * 80)
    print("                          END OF DASHBOARD")
    print("=" * 80)
    print()


def save_dashboard(output_file: str = None):
    """Save dashboard output to file."""
    if output_file is None:
        output_dir = Path(__file__).resolve().parent.parent / 'data'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # Capture output
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        generate_dashboard()

    output = f.getvalue()

    with open(output_file, 'w') as file:
        file.write(output)

    print(f"[SAVED] {output_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Regime Detection Performance Dashboard')
    parser.add_argument('--save', action='store_true', help='Save dashboard to file')
    args = parser.parse_args()

    generate_dashboard()

    if args.save:
        save_dashboard()


if __name__ == "__main__":
    main()

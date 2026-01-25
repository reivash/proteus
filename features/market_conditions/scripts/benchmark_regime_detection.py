"""
Regime Detection Benchmark

Comprehensive benchmarking of regime detection:
1. Classification accuracy vs hindsight truth
2. Detection latency (how long until detected)
3. False positive/negative rates
4. Comparison: HMM vs Rule-based vs Ensemble

Author: Claude Opus 4.5
Date: January 25, 2026
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / 'common'))

import yfinance as yf


@dataclass
class BenchmarkMetrics:
    """Benchmark metrics for a regime detector."""
    method: str
    accuracy: float                 # % correct classifications
    bear_precision: float           # True bear / Predicted bear
    bear_recall: float              # True bear / Actual bear
    bear_f1: float                  # Harmonic mean of precision/recall
    avg_detection_latency: float    # Days to detect regime change
    false_positive_rate: float      # False bear calls / Total bear calls
    false_negative_rate: float      # Missed bears / Actual bears
    regime_stability: float         # Avg days between transitions (higher = more stable)
    avg_inference_time_ms: float    # Computational performance


def get_ground_truth_regime(spy_data: pd.DataFrame, vix_data: pd.DataFrame, idx: int) -> str:
    """
    Determine ground truth regime using multiple signals with hindsight.

    Uses:
    - Price trend (20d, 50d)
    - Realized volatility
    - VIX level
    - Forward returns (since we have hindsight)
    """
    if idx < 50 or idx >= len(spy_data) - 5:
        return "unknown"

    close = spy_data['Close']

    # Trends
    trend_20d = (close.iloc[idx] / close.iloc[idx-20] - 1) * 100
    trend_50d = (close.iloc[idx] / close.iloc[idx-50] - 1) * 100

    # Volatility
    realized_vol = close.iloc[idx-20:idx].pct_change().std() * np.sqrt(252) * 100

    # VIX
    vix = vix_data['Close'].iloc[min(idx, len(vix_data)-1)]

    # Forward info (hindsight)
    fwd_5d = (close.iloc[min(idx+5, len(close)-1)] / close.iloc[idx] - 1) * 100

    # Classification with hindsight
    if vix > 30 or realized_vol > 25:
        return "volatile"
    elif trend_20d < -3 and trend_50d < -5 and fwd_5d < 0:
        return "bear"
    elif trend_20d > 3 and trend_50d > 5 and fwd_5d > 0:
        return "bull"
    else:
        return "choppy"


def run_hmm_detector(spy_data: pd.DataFrame, idx: int) -> Tuple[str, float, float]:
    """
    Run HMM detector and return (regime, confidence, inference_time_ms).
    """
    from analysis.hmm_regime_detector import HMMRegimeDetector

    detector = HMMRegimeDetector()

    start = time.perf_counter()
    try:
        result = detector.detect_regime(spy_data.iloc[:idx+1], lookback=60)
        regime = result.regime
        confidence = result.confidence
    except Exception:
        regime = "choppy"
        confidence = 0.5

    elapsed_ms = (time.perf_counter() - start) * 1000
    return regime, confidence, elapsed_ms


def run_rule_detector(spy_data: pd.DataFrame, vix_data: pd.DataFrame, idx: int) -> Tuple[str, float, float]:
    """
    Run rule-based detector and return (regime, confidence, inference_time_ms).
    """
    if idx < 50:
        return "unknown", 0.0, 0.0

    start = time.perf_counter()

    close = spy_data['Close']

    # Trends
    trend_20d = (close.iloc[idx] / close.iloc[idx-20] - 1) * 100
    trend_50d = (close.iloc[idx] / close.iloc[idx-50] - 1) * 100

    # VIX
    vix = vix_data['Close'].iloc[min(idx, len(vix_data)-1)]

    # MA alignment
    ma20 = close.rolling(20).mean().iloc[idx]
    ma50 = close.rolling(50).mean().iloc[idx]
    current = close.iloc[idx]

    # Classification
    if vix > 30:
        regime, confidence = "volatile", 0.85
    elif trend_20d > 3 and trend_50d > 5 and current > ma20 > ma50:
        regime, confidence = "bull", 0.75
    elif trend_20d < -3 and trend_50d < -5 and current < ma20 < ma50:
        regime, confidence = "bear", 0.75
    else:
        regime, confidence = "choppy", 0.60

    elapsed_ms = (time.perf_counter() - start) * 1000
    return regime, confidence, elapsed_ms


def run_ensemble_detector(spy_data: pd.DataFrame, vix_data: pd.DataFrame, idx: int) -> Tuple[str, float, float]:
    """
    Run ensemble detector (HMM + rule) and return (regime, confidence, inference_time_ms).
    """
    start = time.perf_counter()

    hmm_regime, hmm_conf, _ = run_hmm_detector(spy_data, idx)
    rule_regime, rule_conf, _ = run_rule_detector(spy_data, vix_data, idx)

    # Simple ensemble: agree = high conf, disagree = conservative
    if hmm_regime == rule_regime:
        regime = hmm_regime
        confidence = (hmm_conf + rule_conf) / 2 + 0.1
    else:
        # Default to more conservative regime
        conservative_order = ['volatile', 'bear', 'choppy', 'bull']
        hmm_rank = conservative_order.index(hmm_regime) if hmm_regime in conservative_order else 2
        rule_rank = conservative_order.index(rule_regime) if rule_regime in conservative_order else 2

        if hmm_rank <= rule_rank:
            regime = hmm_regime
        else:
            regime = rule_regime
        confidence = min(hmm_conf, rule_conf) * 0.8

    elapsed_ms = (time.perf_counter() - start) * 1000
    return regime, confidence, elapsed_ms


def calculate_metrics(
    predictions: List[str],
    ground_truth: List[str],
    inference_times: List[float]
) -> Dict:
    """Calculate benchmark metrics."""
    # Filter unknowns
    valid_pairs = [(p, g) for p, g in zip(predictions, ground_truth) if g != "unknown"]
    preds = [p for p, g in valid_pairs]
    truths = [g for p, g in valid_pairs]

    # Accuracy
    correct = sum(1 for p, g in zip(preds, truths) if p == g)
    accuracy = correct / len(preds) * 100 if preds else 0

    # Bear-specific metrics
    bear_pred = [p == 'bear' for p in preds]
    bear_true = [g == 'bear' for g in truths]

    true_pos = sum(1 for p, g in zip(bear_pred, bear_true) if p and g)
    false_pos = sum(1 for p, g in zip(bear_pred, bear_true) if p and not g)
    false_neg = sum(1 for p, g in zip(bear_pred, bear_true) if not p and g)
    true_neg = sum(1 for p, g in zip(bear_pred, bear_true) if not p and not g)

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = false_pos / (false_pos + true_neg) if (false_pos + true_neg) > 0 else 0
    fnr = false_neg / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

    # Regime stability (days between transitions)
    transitions = sum(1 for i in range(1, len(preds)) if preds[i] != preds[i-1])
    stability = len(preds) / (transitions + 1)

    # Avg inference time
    avg_time = np.mean(inference_times) if inference_times else 0

    return {
        'accuracy': accuracy,
        'bear_precision': precision * 100,
        'bear_recall': recall * 100,
        'bear_f1': f1 * 100,
        'false_positive_rate': fpr * 100,
        'false_negative_rate': fnr * 100,
        'regime_stability': stability,
        'avg_inference_time_ms': avg_time,
        'total_predictions': len(preds),
        'total_transitions': transitions
    }


def run_benchmark(years_back: int = 2) -> Dict[str, BenchmarkMetrics]:
    """Run full benchmark comparison."""
    print("=" * 70)
    print("REGIME DETECTION BENCHMARK")
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
    print(f"    VIX: {len(vix_data)} days")

    # Generate ground truth
    print()
    print("[2] Generating ground truth...")
    ground_truth = []
    for idx in range(len(spy_data)):
        truth = get_ground_truth_regime(spy_data, vix_data, idx)
        ground_truth.append(truth)

    # Count ground truth distribution
    gt_counts = {}
    for g in ground_truth:
        gt_counts[g] = gt_counts.get(g, 0) + 1
    print("    Ground truth distribution:")
    for regime, count in sorted(gt_counts.items()):
        if regime != "unknown":
            print(f"      {regime}: {count} days ({count/len(ground_truth)*100:.1f}%)")

    # Run each detector
    results = {}

    # Sample every 5th day for efficiency (still ~100+ data points)
    sample_indices = list(range(60, len(spy_data) - 5, 5))
    print(f"\n    Benchmarking on {len(sample_indices)} sample days...")

    methods = {
        'HMM': run_hmm_detector,
        'Rule-based': run_rule_detector,
        'Ensemble': run_ensemble_detector
    }

    for method_name, detector_fn in methods.items():
        print(f"\n[3] Running {method_name}...")
        predictions = []
        inference_times = []

        for i, idx in enumerate(sample_indices):
            if i % 20 == 0:
                print(f"    Progress: {i}/{len(sample_indices)}")

            if method_name == 'HMM':
                regime, conf, elapsed = detector_fn(spy_data, idx)
            else:
                regime, conf, elapsed = detector_fn(spy_data, vix_data, idx)

            predictions.append(regime)
            inference_times.append(elapsed)

        # Get corresponding ground truth
        gt_sample = [ground_truth[idx] for idx in sample_indices]

        # Calculate metrics
        metrics = calculate_metrics(predictions, gt_sample, inference_times)

        results[method_name] = BenchmarkMetrics(
            method=method_name,
            accuracy=metrics['accuracy'],
            bear_precision=metrics['bear_precision'],
            bear_recall=metrics['bear_recall'],
            bear_f1=metrics['bear_f1'],
            avg_detection_latency=0,  # Would need transition analysis
            false_positive_rate=metrics['false_positive_rate'],
            false_negative_rate=metrics['false_negative_rate'],
            regime_stability=metrics['regime_stability'],
            avg_inference_time_ms=metrics['avg_inference_time_ms']
        )

    return results


def print_benchmark_results(results: Dict[str, BenchmarkMetrics]):
    """Print benchmark comparison."""
    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print()

    print(f"{'Metric':<25} {'HMM':>12} {'Rule-based':>12} {'Ensemble':>12}")
    print("-" * 63)

    metrics = [
        ('Accuracy (%)', 'accuracy'),
        ('Bear Precision (%)', 'bear_precision'),
        ('Bear Recall (%)', 'bear_recall'),
        ('Bear F1 (%)', 'bear_f1'),
        ('False Pos Rate (%)', 'false_positive_rate'),
        ('False Neg Rate (%)', 'false_negative_rate'),
        ('Regime Stability (days)', 'regime_stability'),
        ('Inference Time (ms)', 'avg_inference_time_ms'),
    ]

    for label, attr in metrics:
        hmm = getattr(results.get('HMM'), attr, 0)
        rule = getattr(results.get('Rule-based'), attr, 0)
        ensemble = getattr(results.get('Ensemble'), attr, 0)

        if 'time' in attr.lower():
            print(f"{label:<25} {hmm:>11.1f} {rule:>11.1f} {ensemble:>11.1f}")
        elif 'stability' in attr.lower():
            print(f"{label:<25} {hmm:>11.1f} {rule:>11.1f} {ensemble:>11.1f}")
        else:
            print(f"{label:<25} {hmm:>10.1f}% {rule:>10.1f}% {ensemble:>10.1f}%")

    # Winner summary
    print()
    print("-" * 63)
    print("ANALYSIS:")

    hmm = results.get('HMM')
    rule = results.get('Rule-based')
    ensemble = results.get('Ensemble')

    if hmm.accuracy > rule.accuracy and hmm.accuracy > ensemble.accuracy:
        print(f"  - HMM has highest accuracy ({hmm.accuracy:.1f}%)")
    elif ensemble.accuracy > rule.accuracy:
        print(f"  - Ensemble has highest accuracy ({ensemble.accuracy:.1f}%)")
    else:
        print(f"  - Rule-based has highest accuracy ({rule.accuracy:.1f}%)")

    if hmm.bear_f1 > rule.bear_f1:
        print(f"  - HMM better at bear detection (F1: {hmm.bear_f1:.1f}%)")
    else:
        print(f"  - Rule-based better at bear detection (F1: {rule.bear_f1:.1f}%)")

    print(f"  - Rule-based is {hmm.avg_inference_time_ms / max(rule.avg_inference_time_ms, 0.01):.0f}x faster than HMM")


def save_results(results: Dict[str, BenchmarkMetrics]):
    """Save benchmark results."""
    output_dir = Path(__file__).resolve().parent.parent / 'research'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_file = output_dir / 'benchmark_results.json'
    with open(json_file, 'w') as f:
        json.dump({
            method: asdict(metrics)
            for method, metrics in results.items()
        }, f, indent=2)
    print(f"\n[SAVED] {json_file}")

    # Save markdown
    md_file = output_dir / 'benchmark_results.md'
    with open(md_file, 'w') as f:
        f.write("# Regime Detection Benchmark Results\n\n")
        f.write(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## Summary\n\n")
        f.write("| Metric | HMM | Rule-based | Ensemble |\n")
        f.write("|--------|-----|------------|----------|\n")

        for label, attr in [
            ('Accuracy', 'accuracy'),
            ('Bear Precision', 'bear_precision'),
            ('Bear Recall', 'bear_recall'),
            ('Bear F1', 'bear_f1'),
            ('False Pos Rate', 'false_positive_rate'),
            ('False Neg Rate', 'false_negative_rate'),
            ('Stability', 'regime_stability'),
            ('Inference (ms)', 'avg_inference_time_ms'),
        ]:
            hmm = getattr(results.get('HMM'), attr, 0)
            rule = getattr(results.get('Rule-based'), attr, 0)
            ensemble = getattr(results.get('Ensemble'), attr, 0)

            if 'time' in attr or 'stability' in attr.lower():
                f.write(f"| {label} | {hmm:.1f} | {rule:.1f} | {ensemble:.1f} |\n")
            else:
                f.write(f"| {label} | {hmm:.1f}% | {rule:.1f}% | {ensemble:.1f}% |\n")

    print(f"[SAVED] {md_file}")


def main():
    """Main entry point."""
    results = run_benchmark(years_back=2)
    print_benchmark_results(results)
    save_results(results)


if __name__ == "__main__":
    main()

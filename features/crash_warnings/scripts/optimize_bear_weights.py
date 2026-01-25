#!/usr/bin/env python
"""
Bear Detection Weight Optimizer

Uses historical data to find optimal indicator weights that maximize
hit rate while minimizing false positives.

Usage:
    python scripts/optimize_bear_weights.py
    python scripts/optimize_bear_weights.py --period 5y
    python scripts/optimize_bear_weights.py --iterations 500
"""

import os
import sys
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    print("Please install yfinance: pip install yfinance")
    sys.exit(1)


# Default weights (current configuration)
DEFAULT_WEIGHTS = {
    'spy_roc': 0.14,
    'vix': 0.18,
    'breadth': 0.11,
    'sector_breadth': 0.08,
    'volume': 0.05,
    'yield_curve': 0.12,
    'credit_spread': 0.08,
    'high_yield': 0.08,
    'put_call': 0.08,
    'divergence': 0.08
}

# Max points per indicator (based on scoring system)
MAX_POINTS = {
    'spy_roc': 14,
    'vix': 18,
    'breadth': 11,
    'sector_breadth': 8,
    'volume': 5,
    'yield_curve': 12,
    'credit_spread': 8,
    'high_yield': 8,
    'put_call': 8,
    'divergence': 8
}


def fetch_historical_data(period: str = '5y') -> pd.DataFrame:
    """Fetch SPY historical data."""
    print(f"Fetching SPY data for {period}...")
    spy = yf.Ticker('SPY')
    df = spy.history(period=period)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_historical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all bear detection indicator proxies historically."""
    result = df.copy()

    # Basic indicators
    result['spy_roc_3d'] = result['Close'].pct_change(3) * 100
    result['high_20d'] = result['Close'].rolling(20).max()
    result['drawdown'] = (result['Close'] - result['high_20d']) / result['high_20d'] * 100
    result['sma_20'] = result['Close'].rolling(20).mean()
    result['sma_50'] = result['Close'].rolling(50).mean()
    result['vol_sma'] = result['Volume'].rolling(20).mean()
    result['vol_ratio'] = result['Volume'] / result['vol_sma']
    result['daily_return'] = result['Close'].pct_change() * 100

    # Advanced indicators for simulation
    result['volatility_20d'] = result['daily_return'].rolling(20).std() * np.sqrt(252)
    result['vol_spike'] = result['volatility_20d'].pct_change(2) * 100
    result['rsi_14'] = calculate_rsi(result['Close'], 14)
    result['price_vs_sma20'] = (result['Close'] / result['sma_20'] - 1) * 100
    result['sma_20_slope'] = result['sma_20'].pct_change(5) * 100

    return result.dropna()


def calculate_raw_scores(row: pd.Series) -> Dict[str, float]:
    """Calculate raw (unweighted) indicator scores for a single row."""
    scores = {}

    # SPY ROC component (0-14 points)
    if row['spy_roc_3d'] <= -5:
        scores['spy_roc'] = 14
    elif row['spy_roc_3d'] <= -3:
        scores['spy_roc'] = 10
    elif row['spy_roc_3d'] <= -2:
        scores['spy_roc'] = 5
    else:
        scores['spy_roc'] = 0

    # VIX proxy using realized volatility (0-18 points)
    vol = row.get('volatility_20d', 15)
    vol_spike = row.get('vol_spike', 0)
    if vol >= 35 or vol_spike >= 50:
        scores['vix'] = 18
    elif vol >= 30 or vol_spike >= 30:
        scores['vix'] = 12
    elif vol >= 25 or vol_spike >= 20:
        scores['vix'] = 6
    else:
        scores['vix'] = 0

    # Market breadth proxy (0-11 points)
    price_vs_ma = row.get('price_vs_sma20', 0)
    if price_vs_ma <= -5:
        scores['breadth'] = 11
    elif price_vs_ma <= -3:
        scores['breadth'] = 7
    elif price_vs_ma <= -1:
        scores['breadth'] = 4
    else:
        scores['breadth'] = 0

    # Sector breadth / trend proxy (0-8 points)
    if row['Close'] < row['sma_20'] < row['sma_50']:
        scores['sector_breadth'] = 8
    elif row['Close'] < row['sma_20']:
        scores['sector_breadth'] = 5
    elif row['Close'] < row['sma_50']:
        scores['sector_breadth'] = 2
    else:
        scores['sector_breadth'] = 0

    # Volume confirmation (0-5 points)
    if row['daily_return'] < -0.5 and row['vol_ratio'] > 2:
        scores['volume'] = 5
    else:
        scores['volume'] = 0

    # Yield curve proxy using SMA slope (0-12 points)
    sma_slope = row.get('sma_20_slope', 0)
    if sma_slope <= -3:
        scores['yield_curve'] = 12
    elif sma_slope <= -1.5:
        scores['yield_curve'] = 8
    elif sma_slope <= -0.5:
        scores['yield_curve'] = 4
    else:
        scores['yield_curve'] = 0

    # Credit spread proxy using drawdown (0-8 points)
    if row['drawdown'] < -8:
        scores['credit_spread'] = 8
    elif row['drawdown'] < -5:
        scores['credit_spread'] = 5
    elif row['drawdown'] < -3:
        scores['credit_spread'] = 2
    else:
        scores['credit_spread'] = 0

    # High-yield proxy (0-8 points)
    if row['spy_roc_3d'] < -3 and row['drawdown'] < -3:
        scores['high_yield'] = 8
    elif row['spy_roc_3d'] < -2 and row['drawdown'] < -2:
        scores['high_yield'] = 5
    elif row['spy_roc_3d'] < -1 and row['drawdown'] < -1:
        scores['high_yield'] = 2
    else:
        scores['high_yield'] = 0

    # Put/Call proxy using RSI (0-8 points)
    rsi = row.get('rsi_14', 50)
    if rsi <= 25:
        scores['put_call'] = 8
    elif rsi <= 35:
        scores['put_call'] = 5
    elif rsi <= 40:
        scores['put_call'] = 2
    else:
        scores['put_call'] = 0

    # Divergence proxy (0-8 points)
    if row['drawdown'] > -2 and row['spy_roc_3d'] < -1.5:
        scores['divergence'] = 8
    else:
        scores['divergence'] = 0

    return scores


def calculate_weighted_score(raw_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """Calculate weighted bear score from raw indicator scores."""
    total = 0
    for indicator, raw_score in raw_scores.items():
        # Normalize raw score to 0-100 scale, then apply weight
        max_pts = MAX_POINTS.get(indicator, 10)
        normalized = (raw_score / max_pts) * 100
        total += normalized * weights.get(indicator, 0)
    return min(total, 100)


def simulate_with_weights(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """Simulate bear scores using specified weights."""
    result = df.copy()

    scores = []
    for i, row in result.iterrows():
        raw_scores = calculate_raw_scores(row)
        weighted = calculate_weighted_score(raw_scores, weights)
        scores.append(weighted)

    result['bear_score'] = scores
    result['alert_level'] = pd.cut(
        result['bear_score'],
        bins=[-1, 29, 49, 69, 100],
        labels=['NORMAL', 'WATCH', 'WARNING', 'CRITICAL']
    )

    return result


def find_major_drawdowns(df: pd.DataFrame, threshold: float = -5.0) -> List[Dict]:
    """Find major drawdown events."""
    drawdowns = []
    in_drawdown = False
    start_idx = None
    peak_price = None

    for i, row in df.iterrows():
        if not in_drawdown:
            if row['drawdown'] < threshold:
                in_drawdown = True
                start_idx = i
                peak_idx = df.loc[:i, 'Close'].idxmax()
                peak_price = df.loc[peak_idx, 'Close']
        else:
            if row['drawdown'] > threshold / 2:
                trough_idx = df.loc[start_idx:i, 'Close'].idxmin()
                trough_price = df.loc[trough_idx, 'Close']
                max_dd = (trough_price - peak_price) / peak_price * 100

                drawdowns.append({
                    'start': start_idx,
                    'trough': trough_idx,
                    'end': i,
                    'max_drawdown': max_dd
                })
                in_drawdown = False

    return drawdowns


def evaluate_weights(df: pd.DataFrame, weights: Dict[str, float], drawdowns: List[Dict]) -> Dict:
    """Evaluate how well given weights detect drawdowns."""
    result_df = simulate_with_weights(df, weights)

    warned_count = 0
    warning_days = []

    for dd in drawdowns:
        lookback_start = dd['start'] - timedelta(days=10)
        pre_dd_data = result_df.loc[lookback_start:dd['start']]
        warnings = pre_dd_data[pre_dd_data['alert_level'].isin(['WATCH', 'WARNING', 'CRITICAL'])]

        if len(warnings) > 0:
            warned_count += 1
            first_warning = warnings.index[0]
            days_before = (dd['start'] - first_warning).days
            warning_days.append(days_before)

    # Count false positives
    false_positives = 0
    warning_dates = result_df[result_df['alert_level'].isin(['WARNING', 'CRITICAL'])].index
    for w_date in warning_dates:
        check_end = w_date + timedelta(days=10)
        future_data = result_df.loc[w_date:check_end]
        if future_data['drawdown'].min() > -3:
            false_positives += 1

    hit_rate = warned_count / len(drawdowns) * 100 if drawdowns else 0
    avg_lead = np.mean(warning_days) if warning_days else 0
    fp_rate = false_positives / len(result_df) * 100

    # Combined fitness score: maximize hit rate, penalize false positives
    # Weight hit rate heavily, penalize false positive rate
    fitness = hit_rate - (fp_rate * 10)

    return {
        'hit_rate': hit_rate,
        'warned': warned_count,
        'total': len(drawdowns),
        'avg_lead': avg_lead,
        'false_positives': false_positives,
        'fp_rate': fp_rate,
        'fitness': fitness
    }


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights to sum to 1.0."""
    total = sum(weights.values())
    if total == 0:
        return DEFAULT_WEIGHTS.copy()
    return {k: v / total for k, v in weights.items()}


def mutate_weights(weights: Dict[str, float], mutation_rate: float = 0.2) -> Dict[str, float]:
    """Mutate weights slightly for optimization."""
    new_weights = weights.copy()
    for key in new_weights:
        if random.random() < mutation_rate:
            # Small random adjustment
            change = random.gauss(0, 0.03)
            new_weights[key] = max(0.01, new_weights[key] + change)
    return normalize_weights(new_weights)


def crossover_weights(w1: Dict[str, float], w2: Dict[str, float]) -> Dict[str, float]:
    """Crossover two weight configurations."""
    new_weights = {}
    for key in w1:
        if random.random() < 0.5:
            new_weights[key] = w1[key]
        else:
            new_weights[key] = w2[key]
    return normalize_weights(new_weights)


def random_weights() -> Dict[str, float]:
    """Generate random weights."""
    weights = {k: random.uniform(0.02, 0.25) for k in DEFAULT_WEIGHTS.keys()}
    return normalize_weights(weights)


def optimize_weights(df: pd.DataFrame, drawdowns: List[Dict],
                     iterations: int = 200, population_size: int = 20) -> Tuple[Dict, Dict]:
    """
    Optimize weights using genetic algorithm approach.

    Returns:
        best_weights: Dict of optimized weights
        best_result: Dict of evaluation metrics
    """
    print(f"\nOptimizing weights over {iterations} iterations...")
    print(f"Population size: {population_size}")
    print(f"Drawdowns to detect: {len(drawdowns)}")
    print()

    # Initialize population with default + random variations
    population = [DEFAULT_WEIGHTS.copy()]
    for _ in range(population_size - 1):
        if random.random() < 0.5:
            population.append(mutate_weights(DEFAULT_WEIGHTS, 0.4))
        else:
            population.append(random_weights())

    best_weights = DEFAULT_WEIGHTS.copy()
    best_result = evaluate_weights(df, best_weights, drawdowns)

    print(f"Starting fitness: {best_result['fitness']:.1f} "
          f"(hit: {best_result['hit_rate']:.1f}%, FP: {best_result['false_positives']})")

    for gen in range(iterations):
        # Evaluate all in population
        results = [(w, evaluate_weights(df, w, drawdowns)) for w in population]
        results.sort(key=lambda x: x[1]['fitness'], reverse=True)

        # Check for improvement
        if results[0][1]['fitness'] > best_result['fitness']:
            best_weights = results[0][0]
            best_result = results[0][1]
            print(f"Gen {gen+1}: New best fitness: {best_result['fitness']:.1f} "
                  f"(hit: {best_result['hit_rate']:.1f}%, FP: {best_result['false_positives']})")

        # Progress update every 50 generations
        if (gen + 1) % 50 == 0:
            print(f"Gen {gen+1}: Best fitness: {best_result['fitness']:.1f}")

        # Selection: keep top 25%
        survivors = [r[0] for r in results[:population_size // 4]]

        # Create new population
        new_population = survivors.copy()

        # Add mutations of survivors
        while len(new_population) < population_size * 0.6:
            parent = random.choice(survivors)
            new_population.append(mutate_weights(parent))

        # Add crossovers
        while len(new_population) < population_size * 0.9:
            p1, p2 = random.sample(survivors, 2)
            new_population.append(crossover_weights(p1, p2))

        # Add fresh random to maintain diversity
        while len(new_population) < population_size:
            new_population.append(random_weights())

        population = new_population

    return best_weights, best_result


def print_results(original: Dict, optimized: Dict, original_weights: Dict,
                  optimized_weights: Dict):
    """Print comparison of original vs optimized weights."""
    print()
    print("=" * 70)
    print("WEIGHT OPTIMIZATION RESULTS")
    print("=" * 70)
    print()

    print("--- PERFORMANCE COMPARISON ---")
    print(f"{'Metric':<20} {'Original':<15} {'Optimized':<15} {'Change':<15}")
    print("-" * 60)
    print(f"{'Hit Rate':<20} {original['hit_rate']:.1f}%{'':<10} {optimized['hit_rate']:.1f}%{'':<10} "
          f"{optimized['hit_rate'] - original['hit_rate']:+.1f}%")
    print(f"{'Warned':<20} {original['warned']}/{original['total']}{'':<8} "
          f"{optimized['warned']}/{optimized['total']}{'':<8}")
    print(f"{'Avg Lead Days':<20} {original['avg_lead']:.1f}{'':<12} {optimized['avg_lead']:.1f}{'':<12} "
          f"{optimized['avg_lead'] - original['avg_lead']:+.1f}")
    print(f"{'False Positives':<20} {original['false_positives']}{'':<13} {optimized['false_positives']}{'':<13} "
          f"{optimized['false_positives'] - original['false_positives']:+d}")
    print(f"{'Fitness Score':<20} {original['fitness']:.1f}{'':<12} {optimized['fitness']:.1f}{'':<12} "
          f"{optimized['fitness'] - original['fitness']:+.1f}")
    print()

    print("--- WEIGHT COMPARISON ---")
    print(f"{'Indicator':<18} {'Original':<12} {'Optimized':<12} {'Change':<12}")
    print("-" * 54)
    for key in original_weights:
        orig = original_weights[key] * 100
        opt = optimized_weights[key] * 100
        change = opt - orig
        print(f"{key:<18} {orig:>5.1f}%{'':<5} {opt:>5.1f}%{'':<5} {change:>+5.1f}%")
    print()


def save_optimized_config(weights: Dict[str, float], result: Dict, output_path: str):
    """Save optimized weights to JSON file."""
    config = {
        "_description": "Optimized bear detection weights",
        "_generated": datetime.now().isoformat(),
        "_metrics": {
            "hit_rate": result['hit_rate'],
            "false_positives": result['false_positives'],
            "avg_lead_days": result['avg_lead'],
            "fitness": result['fitness']
        },
        "weights": {k: round(v, 4) for k, v in weights.items()}
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Optimized weights saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Optimize Bear Detection Weights')
    parser.add_argument('--period', type=str, default='5y',
                       help='Historical period for optimization (1y, 2y, 5y)')
    parser.add_argument('--iterations', type=int, default=200,
                       help='Number of optimization iterations')
    parser.add_argument('--population', type=int, default=20,
                       help='Population size for genetic algorithm')
    parser.add_argument('--output', type=str, default='data/research/optimized_bear_weights.json',
                       help='Output file for optimized weights')
    args = parser.parse_args()

    # Fetch and prepare data
    df = fetch_historical_data(args.period)
    df = calculate_historical_indicators(df)

    # Find drawdowns
    drawdowns = find_major_drawdowns(df, threshold=-5.0)
    print(f"Found {len(drawdowns)} major drawdowns (>5%)")

    # Evaluate current weights
    print("\nEvaluating current weights...")
    original_result = evaluate_weights(df, DEFAULT_WEIGHTS, drawdowns)
    print(f"Current: Hit rate={original_result['hit_rate']:.1f}%, "
          f"FP={original_result['false_positives']}, "
          f"Avg lead={original_result['avg_lead']:.1f} days")

    # Run optimization
    optimized_weights, optimized_result = optimize_weights(
        df, drawdowns,
        iterations=args.iterations,
        population_size=args.population
    )

    # Print comparison
    print_results(original_result, optimized_result, DEFAULT_WEIGHTS, optimized_weights)

    # Save results
    output_path = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_optimized_config(optimized_weights, optimized_result, output_path)

    # Print config update instructions
    if optimized_result['hit_rate'] > original_result['hit_rate']:
        print()
        print("=" * 70)
        print("TO APPLY OPTIMIZED WEIGHTS:")
        print("=" * 70)
        print()
        print("Update config/bear_detection_config.json 'weights' section with:")
        print()
        print(json.dumps({k: round(v, 4) for k, v in optimized_weights.items()}, indent=2))
        print()


if __name__ == "__main__":
    main()

"""
EXPERIMENT: EXP-001
Date: 2025-11-13
Objective: Establish baseline performance using Moving Average Crossover

METHOD:
- Approach: 50-day and 200-day moving average crossover
- Data: S&P 500 (^GSPC), 2 years historical
- Model: Simple MA crossover (no ML)

This establishes the baseline that all future models must beat.
"""

import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.models.baseline.moving_average import MovingAverageCrossover
from common.evaluation.metrics import evaluate_model


def run_baseline_experiment():
    """Run the baseline experiment."""

    print("=" * 70)
    print("PROTEUS - EXPERIMENT EXP-001: BASELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    ticker = "^GSPC"  # S&P 500
    period = "2y"
    short_window = 50
    long_window = 200

    print(f"Configuration:")
    print(f"  Ticker: {ticker}")
    print(f"  Period: {period}")
    print(f"  Short MA: {short_window} days")
    print(f"  Long MA: {long_window} days")
    print()

    # Step 1: Fetch Data
    print("[1/4] Fetching data...")
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, period=period)
        print(f"      [OK] Fetched {len(data)} rows")
        print(f"      [OK] Date range: {data['Date'].min()} to {data['Date'].max()}")
    except Exception as e:
        print(f"      [FAIL] Error fetching data: {str(e)}")
        return None

    # Step 2: Train Model
    print("\n[2/4] Training model...")
    model = MovingAverageCrossover(short_window=short_window, long_window=long_window)
    model.fit(data)
    print("      [OK] Model trained (baseline requires no actual training)")

    # Step 3: Generate Predictions
    print("\n[3/4] Generating predictions...")
    predictions_df = model.predict(data)
    valid_predictions = predictions_df.dropna()
    print(f"      [OK] Generated {len(valid_predictions)} predictions")

    # Step 4: Evaluate Performance
    print("\n[4/4] Evaluating performance...")
    results = model.evaluate(data)

    if 'error' in results:
        print(f"      [FAIL] Evaluation error: {results['error']}")
        return None

    print("      [OK] Evaluation complete")

    # Display Results
    print()
    print("=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Directional Accuracy:  {results['directional_accuracy']:.2f}%")
    print(f"Mean Absolute Error:   ${results['mae']:.2f}")
    print(f"Total Predictions:     {results['total_samples']}")
    print(f"Correct Predictions:   {results['correct_predictions']}")
    print()
    print(f"Predictions Breakdown:")
    print(f"  Up Predicted:        {results['up_predictions']} ({results['up_predictions']/results['total_samples']*100:.1f}%)")
    print(f"  Down Predicted:      {results['down_predictions']} ({results['down_predictions']/results['total_samples']*100:.1f}%)")
    print()
    print(f"Actual Market Behavior:")
    print(f"  Up Days:             {results['actual_up_days']} ({results['actual_up_days']/results['total_samples']*100:.1f}%)")
    print(f"  Down Days:           {results['actual_down_days']} ({results['actual_down_days']/results['total_samples']*100:.1f}%)")
    print("=" * 70)

    # Prepare results for logging (convert numpy types to Python types for JSON)
    experiment_results = {
        'experiment_id': 'EXP-001',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'MovingAverageCrossover',
        'ticker': ticker,
        'period': period,
        'parameters': {
            'short_window': int(short_window),
            'long_window': int(long_window)
        },
        'metrics': {
            'directional_accuracy': float(results['directional_accuracy']),
            'mae': float(results['mae']),
            'total_samples': int(results['total_samples']),
            'correct_predictions': int(results['correct_predictions']),
            'up_predictions': int(results['up_predictions']),
            'down_predictions': int(results['down_predictions']),
            'actual_up_days': int(results['actual_up_days']),
            'actual_down_days': int(results['actual_down_days'])
        },
        'baseline': True
    }

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'experiments')
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, 'exp001_baseline_results.json')

    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return experiment_results


if __name__ == "__main__":
    results = run_baseline_experiment()

    if results:
        print("\n[SUCCESS] Baseline experiment completed successfully!")
        print(f"[SUCCESS] Baseline Directional Accuracy: {results['metrics']['directional_accuracy']:.2f}%")
    else:
        print("\n[FAILED] Baseline experiment failed!")

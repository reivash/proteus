"""
EXPERIMENT: EXP-003
Date: 2025-11-14
Objective: Leverage GPU-accelerated LSTM for temporal pattern recognition

METHOD:
- Approach: LSTM neural network with 60-day sequences
- Data: S&P 500 (^GSPC), 2 years historical
- Features: Same 36 technical indicators as EXP-002
- Model: 2-layer LSTM (128 hidden units) with dropout
- Hardware: RTX 4080 SUPER GPU acceleration
"""

import sys
import os
import json
from datetime import datetime
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.models.ml.lstm_classifier import LSTMStockPredictor


def run_lstm_experiment():
    """Run the LSTM experiment with GPU acceleration."""

    print("=" * 70)
    print("PROTEUS - EXPERIMENT EXP-003: LSTM WITH GPU ACCELERATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    ticker = "^GSPC"
    period = "2y"
    sequence_length = 60
    hidden_size = 128
    num_layers = 2
    dropout = 0.2
    learning_rate = 0.001
    batch_size = 32
    epochs = 50

    print(f"Configuration:")
    print(f"  Ticker: {ticker}")
    print(f"  Period: {period}")
    print(f"  Model: LSTM Neural Network")
    print(f"  Sequence Length: {sequence_length} days")
    print(f"  Hidden Size: {hidden_size}")
    print(f"  Layers: {num_layers}")
    print(f"  Dropout: {dropout}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print()

    # Step 1: Fetch Data
    print("[1/7] Fetching data...")
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, period=period)
        print(f"      [OK] Fetched {len(data)} rows")
        print(f"      [OK] Date range: {data['Date'].min()} to {data['Date'].max()}")
    except Exception as e:
        print(f"      [FAIL] Error fetching data: {str(e)}")
        return None

    # Step 2: Feature Engineering
    print("\n[2/7] Engineering features...")
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)
    new_features = enriched_data.shape[1] - data.shape[1]
    print(f"      [OK] Created {new_features} new features")
    print(f"      [OK] Total features: {enriched_data.shape[1]}")

    # Step 3: Prepare Target
    print("\n[3/7] Preparing target...")
    enriched_data['Actual_Direction'] = (enriched_data['Close'].shift(-1) > enriched_data['Close']).astype(int)
    enriched_data = enriched_data[:-1].dropna()
    print(f"      [OK] Valid samples: {len(enriched_data)}")

    # Step 4: Initialize Model
    print("\n[4/7] Initializing LSTM model...")
    predictor = LSTMStockPredictor(
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs
    )

    # Prepare data
    X, y = predictor.prepare_data(enriched_data)
    print(f"      [OK] Features: {X.shape[1]}")
    print(f"      [OK] Samples: {X.shape[0]}")

    # Step 5: Train-Test Split
    print("\n[5/7] Splitting data...")
    # Account for sequence length in split
    split_idx = int((len(X) - sequence_length) * 0.7) + sequence_length
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"      [OK] Train set: {len(X_train)} samples")
    print(f"      [OK] Test set: {len(X_test)} samples")

    # Step 6: Train Model
    print("\n[6/7] Training LSTM model on GPU...")
    print("      (This may take a few minutes...)")
    start_time = time.time()

    predictor.fit(X_train, y_train, validation_split=0.2, verbose=True)

    training_time = time.time() - start_time
    print(f"      [OK] Training completed in {training_time:.2f} seconds")

    # Step 7: Evaluate
    print("\n[7/7] Evaluating model...")
    results = predictor.evaluate(X_test, y_test, verbose=True)

    # Display Results
    print()
    print("=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Directional Accuracy:  {results['accuracy']:.2f}%")
    print(f"Precision:             {results['precision']:.3f}")
    print(f"Recall:                {results['recall']:.3f}")
    print(f"F1 Score:              {results['f1_score']:.3f}")
    print()
    print(f"Confusion Matrix:")
    print(f"  True Positives:      {results['true_positives']}")
    print(f"  True Negatives:      {results['true_negatives']}")
    print(f"  False Positives:     {results['false_positives']}")
    print(f"  False Negatives:     {results['false_negatives']}")
    print()
    print(f"Predictions Breakdown:")
    print(f"  Up Predicted:        {results['up_predicted']} ({results['up_predicted']/results['total_samples']*100:.1f}%)")
    print(f"  Down Predicted:      {results['down_predicted']} ({results['down_predicted']/results['total_samples']*100:.1f}%)")
    print()
    print(f"Actual Market Behavior:")
    print(f"  Up Days:             {results['actual_up']} ({results['actual_up']/results['total_samples']*100:.1f}%)")
    print(f"  Down Days:           {results['actual_down']} ({results['actual_down']/results['total_samples']*100:.1f}%)")
    print()
    print(f"Training Time:         {training_time:.2f} seconds")
    print("=" * 70)

    # Prepare results for logging
    experiment_results = {
        'experiment_id': 'EXP-003',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'LSTM Neural Network',
        'ticker': ticker,
        'period': period,
        'parameters': {
            'sequence_length': sequence_length,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs
        },
        'hardware': {
            'device': str(predictor.device),
            'gpu_name': 'RTX 4080 SUPER' if predictor.device.type == 'cuda' else 'N/A'
        },
        'metrics': results,
        'training_time_seconds': float(training_time),
        'num_features': int(X.shape[1]),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'baseline_comparison': {
            'baseline_accuracy': 53.49,
            'xgboost_accuracy': 58.94,
            'improvement_over_baseline': float(results['accuracy'] - 53.49),
            'improvement_over_xgboost': float(results['accuracy'] - 58.94)
        }
    }

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'experiments')
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, 'exp003_lstm_results.json')

    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return experiment_results


if __name__ == "__main__":
    results = run_lstm_experiment()

    if results:
        print("\n[SUCCESS] LSTM experiment completed successfully!")
        print(f"[SUCCESS] Accuracy: {results['metrics']['accuracy']:.2f}%")
        print(f"[SUCCESS] Improvement over baseline: +{results['baseline_comparison']['improvement_over_baseline']:.2f}%")
        print(f"[SUCCESS] Improvement over XGBoost: +{results['baseline_comparison']['improvement_over_xgboost']:.2f}%")
        print(f"[SUCCESS] Training time: {results['training_time_seconds']:.2f}s")
    else:
        print("\n[FAILED] LSTM experiment failed!")

"""
EXPERIMENT: EXP-006
Date: 2025-11-14
Objective: Ensemble methods for robust predictions

METHOD:
- Approach: Voting ensemble (XGBoost + LightGBM + Random Forest)
- Voting: Soft (weighted average of probabilities)
- Rationale: Multiple models reduce overfitting, more stable predictions
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.models.ml.ensemble_classifier import EnsembleStockClassifier


def run_ensemble_experiment():
    """Run ensemble model experiment."""

    print("=" * 70)
    print("PROTEUS - EXPERIMENT EXP-006: ENSEMBLE METHODS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    ticker = "^GSPC"
    period = "2y"

    print(f"Configuration:")
    print(f"  Ticker: {ticker}")
    print(f"  Period: {period}")
    print(f"  Models: XGBoost + LightGBM + Random Forest")
    print(f"  Voting: Soft (probability averaging)")
    print(f"  Baseline: 58.94% (EXP-002)")
    print()

    # Step 1: Fetch Data
    print("[1/6] Fetching data...")
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, period=period)
        print(f"      [OK] Fetched {len(data)} rows")
    except Exception as e:
        print(f"      [FAIL] Error: {str(e)}")
        return None

    # Step 2: Feature Engineering
    print("\n[2/6] Engineering features...")
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)
    print(f"      [OK] Features: {enriched_data.shape[1]}")

    # Step 3: Prepare Target
    print("\n[3/6] Preparing target...")
    enriched_data['Actual_Direction'] = (enriched_data['Close'].shift(-1) > enriched_data['Close']).astype(int)
    enriched_data = enriched_data[:-1].dropna()
    print(f"      [OK] Valid samples: {len(enriched_data)}")

    # Step 4: Train-Test Split
    print("\n[4/6] Splitting data...")
    classifier = EnsembleStockClassifier(voting='soft')
    X, y = classifier.prepare_data(enriched_data)

    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"      [OK] Train set: {len(X_train)} samples")
    print(f"      [OK] Test set: {len(X_test)} samples")
    print(f"      [OK] Features: {X.shape[1]}")

    # Step 5: Train Ensemble
    print("\n[5/6] Training ensemble...")
    classifier.fit(X_train, y_train, verbose=True)

    # Step 6: Evaluate
    print("\n[6/6] Evaluating ensemble...")
    results = classifier.evaluate(X_test, y_test, verbose=True)

    # Display Results
    print()
    print("=" * 70)
    print("RESULTS SUMMARY:")
    print("=" * 70)
    print(f"Ensemble Accuracy:     {results['accuracy']:.2f}%")
    print(f"XGBoost Only:          {results['individual_accuracies']['xgb']:.2f}%")
    print(f"LightGBM Only:         {results['individual_accuracies']['lgbm']:.2f}%")
    print(f"Random Forest Only:    {results['individual_accuracies']['rf']:.2f}%")
    print()
    print(f"Baseline (EXP-002):    58.94%")
    improvement = results['accuracy'] - 58.94
    if improvement > 0:
        print(f"Improvement:           +{improvement:.2f}%")
    else:
        print(f"Change:                {improvement:.2f}%")
    print("=" * 70)

    # Prepare results
    experiment_results = {
        'experiment_id': 'EXP-006',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'Ensemble (XGBoost + LightGBM + RandomForest)',
        'ticker': ticker,
        'period': period,
        'voting_method': 'soft',
        'metrics': results,
        'num_features': int(X.shape[1]),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'comparison': {
            'baseline': 53.49,
            'xgboost_single': 58.94,
            'ensemble': float(results['accuracy']),
            'improvement_over_baseline': float(results['accuracy'] - 53.49),
            'improvement_over_xgboost': float(results['accuracy'] - 58.94)
        }
    }

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'experiments')
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, 'exp006_ensemble_results.json')

    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return experiment_results


if __name__ == "__main__":
    results = run_ensemble_experiment()

    if results:
        print("\n[SUCCESS] Ensemble experiment complete!")
        print(f"[SUCCESS] Ensemble Accuracy: {results['metrics']['accuracy']:.2f}%")
        print(f"[INFO] XGBoost Single: 58.94%")

        improvement = results['comparison']['improvement_over_xgboost']
        if improvement > 0:
            print(f"[SUCCESS] Ensemble improves by +{improvement:.2f}%")
        elif improvement > -1:
            print(f"[INFO] Ensemble similar to single model ({improvement:.2f}%)")
        else:
            print(f"[INFO] Single XGBoost still best ({improvement:.2f}%)")
    else:
        print("\n[FAILED] Ensemble experiment failed!")

"""
EXPERIMENT: EXP-005
Date: 2025-11-14
Objective: Optimize XGBoost hyperparameters using Optuna

METHOD:
- Approach: Automated hyperparameter search with Optuna
- Base Model: XGBoost (best from EXP-002)
- Search Space: n_estimators, max_depth, learning_rate, subsample, etc.
- Optimization: Maximize directional accuracy
- Trials: 50-100 iterations
"""

import sys
import os
import json
from datetime import datetime
import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.models.ml.xgboost_classifier import XGBoostStockClassifier


# Global variables for optimization
GLOBAL_X_TRAIN = None
GLOBAL_Y_TRAIN = None
GLOBAL_X_VAL = None
GLOBAL_Y_VAL = None


def objective(trial):
    """Optuna objective function."""

    # Hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
    }

    # Train model
    classifier = XGBoostStockClassifier(**params)
    classifier.fit(GLOBAL_X_TRAIN, GLOBAL_Y_TRAIN, validation_split=0.0, verbose=False)

    # Evaluate on validation set
    results = classifier.evaluate(GLOBAL_X_VAL, GLOBAL_Y_VAL, verbose=False)

    return results['accuracy']


def run_hyperparameter_tuning():
    """Run hyperparameter optimization experiment."""

    print("=" * 70)
    print("PROTEUS - EXPERIMENT EXP-005: HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    ticker = "^GSPC"
    period = "2y"
    n_trials = 50  # Number of optimization trials

    print(f"Configuration:")
    print(f"  Ticker: {ticker}")
    print(f"  Period: {period}")
    print(f"  Optimizer: Optuna (TPE Sampler)")
    print(f"  Trials: {n_trials}")
    print(f"  Baseline: 58.94% (EXP-002)")
    print()

    # Step 1: Fetch Data
    print("[1/7] Fetching data...")
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, period=period)
        print(f"      [OK] Fetched {len(data)} rows")
    except Exception as e:
        print(f"      [FAIL] Error: {str(e)}")
        return None

    # Step 2: Feature Engineering
    print("\n[2/7] Engineering features...")
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)
    print(f"      [OK] Features: {enriched_data.shape[1]}")

    # Step 3: Prepare Target
    print("\n[3/7] Preparing target...")
    enriched_data['Actual_Direction'] = (enriched_data['Close'].shift(-1) > enriched_data['Close']).astype(int)
    enriched_data = enriched_data[:-1].dropna()
    print(f"      [OK] Valid samples: {len(enriched_data)}")

    # Step 4: Prepare Data
    print("\n[4/7] Splitting data...")
    classifier = XGBoostStockClassifier()
    X, y = classifier.prepare_data(enriched_data)

    # Split: 60% train, 20% validation, 20% test
    train_idx = int(len(X) * 0.6)
    val_idx = int(len(X) * 0.8)

    global GLOBAL_X_TRAIN, GLOBAL_Y_TRAIN, GLOBAL_X_VAL, GLOBAL_Y_VAL

    GLOBAL_X_TRAIN = X.iloc[:train_idx]
    GLOBAL_Y_TRAIN = y.iloc[:train_idx]
    GLOBAL_X_VAL = X.iloc[train_idx:val_idx]
    GLOBAL_Y_VAL = y.iloc[train_idx:val_idx]
    X_test = X.iloc[val_idx:]
    y_test = y.iloc[val_idx:]

    print(f"      [OK] Train: {len(GLOBAL_X_TRAIN)} samples")
    print(f"      [OK] Validation: {len(GLOBAL_X_VAL)} samples")
    print(f"      [OK] Test: {len(X_test)} samples")

    # Step 5: Hyperparameter Optimization
    print(f"\n[5/7] Running Optuna optimization ({n_trials} trials)...")
    print("      (This may take several minutes...)")
    print()

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[
            lambda study, trial: print(f"      Trial {trial.number + 1}/{n_trials}: Accuracy = {trial.value:.2f}%")
            if (trial.number + 1) % 5 == 0 else None
        ]
    )

    print()
    print(f"      [OK] Optimization complete!")
    print(f"      [OK] Best validation accuracy: {study.best_value:.2f}%")

    # Step 6: Train Final Model with Best Params
    print("\n[6/7] Training final model with best parameters...")
    best_params = study.best_params

    final_classifier = XGBoostStockClassifier(**best_params)

    # Train on train + validation
    X_train_full = X.iloc[:val_idx]
    y_train_full = y.iloc[:val_idx]

    final_classifier.fit(X_train_full, y_train_full, validation_split=0.0, verbose=False)
    print(f"      [OK] Final model trained")

    # Step 7: Evaluate on Test Set
    print("\n[7/7] Evaluating on test set...")
    results = final_classifier.evaluate(X_test, y_test, verbose=True)

    # Display Results
    print()
    print("=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Test Accuracy:         {results['accuracy']:.2f}%")
    print(f"Baseline (EXP-002):    58.94%")
    print(f"Improvement:           +{results['accuracy'] - 58.94:.2f}%")
    print()
    print("Best Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param:20s} {value}")
    print()
    print("Top 10 Most Important Features:")
    top_features = final_classifier.get_top_features(10)
    for idx, row in top_features.iterrows():
        print(f"  {row['feature']:20s} {row['importance']:.4f}")
    print("=" * 70)

    # Prepare results
    experiment_results = {
        'experiment_id': 'EXP-005',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'XGBoost (Hyperparameter Tuned)',
        'ticker': ticker,
        'period': period,
        'optimization': {
            'method': 'Optuna TPE',
            'n_trials': n_trials,
            'best_validation_accuracy': float(study.best_value),
            'best_params': best_params
        },
        'metrics': results,
        'num_features': int(X.shape[1]),
        'train_samples': int(len(X_train_full)),
        'test_samples': int(len(X_test)),
        'top_features': top_features.head(10).to_dict('records'),
        'comparison': {
            'baseline': 53.49,
            'xgboost_default': 58.94,
            'xgboost_tuned': float(results['accuracy']),
            'improvement_over_baseline': float(results['accuracy'] - 53.49),
            'improvement_over_default': float(results['accuracy'] - 58.94)
        }
    }

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'experiments')
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, 'exp005_xgboost_tuned_results.json')

    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return experiment_results


if __name__ == "__main__":
    results = run_hyperparameter_tuning()

    if results:
        print("\n[SUCCESS] Hyperparameter tuning complete!")
        print(f"[SUCCESS] Tuned Accuracy: {results['metrics']['accuracy']:.2f}%")
        print(f"[SUCCESS] Default Accuracy: 58.94%")
        improvement = results['comparison']['improvement_over_default']
        if improvement > 0:
            print(f"[SUCCESS] Improvement: +{improvement:.2f}%")
        else:
            print(f"[INFO] No improvement over default parameters")
    else:
        print("\n[FAILED] Hyperparameter tuning failed!")

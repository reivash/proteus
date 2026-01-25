"""
EXPERIMENT: EXP-002
Date: 2025-11-14
Objective: Beat baseline using XGBoost with technical indicators

METHOD:
- Approach: XGBoost classifier with 36 engineered features
- Data: S&P 500 (^GSPC), 2 years historical
- Features: Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Model: XGBoost with optimized hyperparameters
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.models.ml.xgboost_classifier import XGBoostStockClassifier


def run_xgboost_experiment():
    """Run the XGBoost experiment."""

    print("=" * 70)
    print("PROTEUS - EXPERIMENT EXP-002: XGBOOST WITH TECHNICAL FEATURES")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    ticker = "^GSPC"
    period = "2y"

    print(f"Configuration:")
    print(f"  Ticker: {ticker}")
    print(f"  Period: {period}")
    print(f"  Model: XGBoost Classifier")
    print()

    # Step 1: Fetch Data
    print("[1/6] Fetching data...")
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, period=period)
        print(f"      [OK] Fetched {len(data)} rows")
        print(f"      [OK] Date range: {data['Date'].min()} to {data['Date'].max()}")
    except Exception as e:
        print(f"      [FAIL] Error fetching data: {str(e)}")
        return None

    # Step 2: Feature Engineering
    print("\n[2/6] Engineering features...")
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)
    new_features = enriched_data.shape[1] - data.shape[1]
    print(f"      [OK] Created {new_features} new features")
    print(f"      [OK] Total features: {enriched_data.shape[1]}")

    # Step 3: Prepare Target
    print("\n[3/6] Preparing target...")
    enriched_data['Actual_Direction'] = (enriched_data['Close'].shift(-1) > enriched_data['Close']).astype(int)
    enriched_data = enriched_data[:-1].dropna()
    print(f"      [OK] Valid samples: {len(enriched_data)}")

    # Step 4: Train-Test Split
    print("\n[4/6] Splitting data...")
    classifier = XGBoostStockClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )

    X, y = classifier.prepare_data(enriched_data)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"      [OK] Train set: {len(X_train)} samples")
    print(f"      [OK] Test set: {len(X_test)} samples")
    print(f"      [OK] Features: {X.shape[1]}")

    # Step 5: Train Model
    print("\n[5/6] Training model...")
    classifier.fit(X_train, y_train, validation_split=0.2, verbose=False)
    print(f"      [OK] Model trained")

    # Step 6: Evaluate
    print("\n[6/6] Evaluating model...")
    results = classifier.evaluate(X_test, y_test, verbose=False)
    print(f"      [OK] Accuracy: {results['accuracy']:.2f}%")

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
    print("Top 10 Most Important Features:")
    top_features = classifier.get_top_features(10)
    for idx, row in top_features.iterrows():
        print(f"  {row['feature']:20s} {row['importance']:.4f}")
    print("=" * 70)

    # Prepare results for logging
    experiment_results = {
        'experiment_id': 'EXP-002',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'XGBoostClassifier',
        'ticker': ticker,
        'period': period,
        'parameters': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'metrics': results,
        'num_features': int(X.shape[1]),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'top_features': top_features.head(10).to_dict('records'),
        'baseline_comparison': {
            'baseline_accuracy': 53.49,
            'improvement': float(results['accuracy'] - 53.49),
            'improvement_pct': float((results['accuracy'] - 53.49) / 53.49 * 100)
        }
    }

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'experiments')
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, 'exp002_xgboost_results.json')

    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return experiment_results


if __name__ == "__main__":
    results = run_xgboost_experiment()

    if results:
        print("\n[SUCCESS] XGBoost experiment completed successfully!")
        print(f"[SUCCESS] Accuracy: {results['metrics']['accuracy']:.2f}%")
        print(f"[SUCCESS] Improvement over baseline: +{results['baseline_comparison']['improvement']:.2f}%")
    else:
        print("\n[FAILED] XGBoost experiment failed!")

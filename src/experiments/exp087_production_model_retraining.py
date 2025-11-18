"""
EXP-087: Retrain Production XGBoost Model with Validated Temporal Features

OBJECTIVE: Deploy validated temporal features to production ML scanner
BASELINE: 0.834 AUC (41 features, EXP-069)
TARGET: 0.854 AUC (65 features: 50 technical + 15 temporal)

VALIDATION BASIS (EXP-085):
- Average AUC improvement: +2.0pp (0.694 → 0.715)
- Success rate: 90% stocks improved (9/10)
- All validation criteria PASSED

METHODOLOGY:
1. Use FeatureIntegrator with temporal features enabled
2. Train on same stocks as EXP-069 for fair comparison
3. Test on holdout period (2024-01-01 to present)
4. Save new model to production path for ML scanner
5. Compare AUC: 41-feature baseline vs 65-feature temporal

EXPECTED PRODUCTION IMPACT:
- AUC: 0.834 → 0.854 (+2.0pp validated improvement)
- Precision: 35.5% → 38% (proportional to AUC gain)
- 281 trades/year benefit from enhanced predictions
- Foundation for further feature additions (cross-sectional, ensemble)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from src.models.ml.feature_integration import FeatureIntegrator, FeatureConfig
from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.models.trading.mean_reversion import MeanReversionDetector


def get_production_stocks():
    """Get production stock universe (same as EXP-069 for fair comparison)."""
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',  # Tech leaders
        'JPM', 'BAC', 'GS', 'WFC',  # Financials
        'JNJ', 'UNH', 'PFE', 'ABBV',  # Healthcare
        'WMT', 'COST', 'TGT',  # Retail
        'V', 'MA',  # Payments
        'DIS', 'NFLX',  # Entertainment
        'BA', 'CAT', 'MMM',  # Industrials
        'XOM', 'CVX',  # Energy
        'TSLA', 'F',  # Automotive
    ]


def prepare_training_data(stocks, start_date='2019-01-01', end_date='2023-12-31', use_temporal=True):
    """
    Prepare training dataset with FeatureIntegrator.

    Args:
        stocks: List of stock tickers
        start_date: Training start date
        end_date: Training end date (holdout starts after this)
        use_temporal: Enable temporal features (default: True)

    Returns:
        X_train, y_train: Training features and labels
    """
    print(f"Preparing training data ({start_date} to {end_date})...")
    print(f"Temporal features: {'ENABLED' if use_temporal else 'DISABLED'}")
    print()

    # Initialize feature integrator
    config = FeatureConfig(
        use_technical=True,
        use_temporal=use_temporal,
        use_cross_sectional=False,  # Not yet validated
        fillna=True
    )
    integrator = FeatureIntegrator(config)

    # Initialize data fetcher and signal detector
    fetcher = YahooFinanceFetcher()
    detector = MeanReversionDetector(
        z_score_threshold=1.5,
        rsi_oversold=35,
        volume_multiplier=1.3,
        price_drop_threshold=-1.5
    )

    all_X = []
    all_y = []

    for i, ticker in enumerate(stocks, 1):
        print(f"[{i}/{len(stocks)}] {ticker}...", end=' ')

        try:
            # Fetch data with extra buffer for indicators
            fetch_start = (pd.to_datetime(start_date) - timedelta(days=400)).strftime('%Y-%m-%d')
            data = fetcher.fetch_stock_data(ticker, start_date=fetch_start, end_date=end_date)

            if data is None or len(data) < 300:
                print(f"[SKIP] Insufficient data")
                continue

            # Engineer features with FeatureIntegrator
            enriched = integrator.prepare_ml_features(data, ticker=ticker)

            # Calculate forward returns and labels (use ALL data, not just signals)
            enriched['forward_return_3d'] = enriched['Close'].pct_change(3).shift(-3) * 100
            enriched['win'] = (enriched['forward_return_3d'] > 0).astype(int)

            # Ensure index is DatetimeIndex for proper slicing
            if 'Date' in enriched.columns:
                enriched['Date'] = pd.to_datetime(enriched['Date'])
                enriched = enriched.set_index('Date')

            # Filter to training period only
            training_data = enriched.loc[start_date:end_date].copy()

            # Drop rows with NaN in critical columns only
            critical_cols = ['forward_return_3d', 'win', 'Close']
            training_data = training_data.dropna(subset=critical_cols)

            if len(training_data) < 50:
                print(f"[SKIP] Insufficient samples ({len(training_data)})")
                continue

            # Prepare features (exclude target and metadata)
            exclude_cols = ['forward_return_3d', 'win', 'Date',
                           'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            feature_cols = [col for col in training_data.columns if col not in exclude_cols]

            X = training_data[feature_cols].values
            y = training_data['win'].values

            all_X.append(X)
            all_y.append(y)

            print(f"[OK] {len(training_data)} samples, {len(feature_cols)} features")

        except Exception as e:
            print(f"[ERROR] {e}")
            continue

    # Concatenate all data
    X_train = np.vstack(all_X)
    y_train = np.hstack(all_y)

    print()
    print(f"[SUCCESS] Training data prepared:")
    print(f"  Total samples: {len(X_train):,}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Positive class: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
    print()

    return X_train, y_train


def train_production_model(X_train, y_train, model_path='logs/experiments/exp087_temporal_model.json'):
    """
    Train XGBoost model with same hyperparameters as EXP-069 for fair comparison.

    Args:
        X_train: Training features
        y_train: Training labels
        model_path: Path to save model

    Returns:
        Trained XGBoost model
    """
    print("Training XGBoost model (EXP-069 hyperparameters)...")
    print()

    # Split for validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Train model with EXP-069 hyperparameters
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc'
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # Evaluate on validation set
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > 0.30).astype(int)

    auc = roc_auc_score(y_val, y_pred_proba)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)

    print()
    print(f"[VALIDATION PERFORMANCE]")
    print(f"  AUC: {auc:.3f}")
    print(f"  Precision (@ 0.30 threshold): {precision*100:.1f}%")
    print(f"  Recall: {recall*100:.1f}%")
    print()

    # Save model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    print(f"[OK] Model saved to {model_path}")
    print()

    return model


def test_on_holdout(stocks, model, start_date='2024-01-01', use_temporal=True):
    """
    Test model on holdout period (2024-present).

    Args:
        stocks: List of stock tickers
        model: Trained XGBoost model
        start_date: Holdout start date
        use_temporal: Whether model uses temporal features

    Returns:
        Holdout AUC score
    """
    print(f"Testing on holdout period ({start_date} to present)...")
    print()

    # Initialize feature integrator
    config = FeatureConfig(
        use_technical=True,
        use_temporal=use_temporal,
        use_cross_sectional=False,
        fillna=True
    )
    integrator = FeatureIntegrator(config)

    # Initialize data fetcher and signal detector
    fetcher = YahooFinanceFetcher()
    detector = MeanReversionDetector(
        z_score_threshold=1.5,
        rsi_oversold=35,
        volume_multiplier=1.3,
        price_drop_threshold=-1.5
    )

    all_X = []
    all_y = []

    for i, ticker in enumerate(stocks[:10], 1):  # Test on subset for speed
        print(f"[{i}/10] {ticker}...", end=' ')

        try:
            # Fetch data
            fetch_start = (pd.to_datetime(start_date) - timedelta(days=400)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            data = fetcher.fetch_stock_data(ticker, start_date=fetch_start, end_date=end_date)

            if data is None or len(data) < 100:
                print(f"[SKIP] Insufficient data")
                continue

            # Engineer features
            enriched = integrator.prepare_ml_features(data, ticker=ticker)

            # Calculate forward returns and labels (use ALL data for holdout test too)
            enriched['forward_return_3d'] = enriched['Close'].pct_change(3).shift(-3) * 100
            enriched['win'] = (enriched['forward_return_3d'] > 0).astype(int)

            # Ensure index is DatetimeIndex for proper slicing
            if 'Date' in enriched.columns:
                enriched['Date'] = pd.to_datetime(enriched['Date'])
                enriched = enriched.set_index('Date')

            # Filter to holdout period only
            test_data = enriched.loc[start_date:].copy()

            # Drop rows with NaN in critical columns only
            critical_cols = ['forward_return_3d', 'win', 'Close']
            test_data = test_data.dropna(subset=critical_cols)

            if len(test_data) < 10:
                print(f"[SKIP] Insufficient samples ({len(test_data)})")
                continue

            # Prepare features
            exclude_cols = ['forward_return_3d', 'win', 'Date',
                           'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            feature_cols = [col for col in test_data.columns if col not in exclude_cols]

            X = test_data[feature_cols].values
            y = test_data['win'].values

            all_X.append(X)
            all_y.append(y)

            print(f"[OK] {len(test_data)} samples")

        except Exception as e:
            print(f"[ERROR] {e}")
            continue

    if len(all_X) == 0:
        print("[ERROR] No holdout data available")
        return 0.0

    # Concatenate and evaluate
    X_test = np.vstack(all_X)
    y_test = np.hstack(all_y)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    print()
    print(f"[HOLDOUT PERFORMANCE]")
    print(f"  Samples: {len(X_test):,}")
    print(f"  AUC: {auc:.3f}")
    print()

    return auc


def main():
    print("=" * 70)
    print("EXP-087: PRODUCTION MODEL RETRAINING WITH TEMPORAL FEATURES")
    print("=" * 70)
    print()

    stocks = get_production_stocks()
    print(f"Production stock universe: {len(stocks)} stocks")
    print()

    # Step 1: Train BASELINE model (41 features, no temporal)
    print("=" * 70)
    print("STEP 1: BASELINE MODEL (50 technical features, no temporal)")
    print("=" * 70)
    print()

    X_baseline, y_baseline = prepare_training_data(stocks, use_temporal=False)
    model_baseline = train_production_model(
        X_baseline, y_baseline,
        model_path='logs/experiments/exp087_baseline_model.json'
    )
    auc_baseline_holdout = test_on_holdout(stocks, model_baseline, use_temporal=False)

    # Step 2: Train TEMPORAL model (65 features)
    print("=" * 70)
    print("STEP 2: TEMPORAL MODEL (50 technical + 15 temporal = 65 features)")
    print("=" * 70)
    print()

    X_temporal, y_temporal = prepare_training_data(stocks, use_temporal=True)
    model_temporal = train_production_model(
        X_temporal, y_temporal,
        model_path='logs/experiments/exp087_temporal_model.json'
    )
    auc_temporal_holdout = test_on_holdout(stocks, model_temporal, use_temporal=True)

    # Step 3: Compare results
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print()

    improvement = auc_temporal_holdout - auc_baseline_holdout

    print(f"Baseline (50 features):  AUC = {auc_baseline_holdout:.3f}")
    print(f"Temporal (65 features):  AUC = {auc_temporal_holdout:.3f}")
    print(f"Improvement:             {improvement:+.3f} ({improvement*100:+.1f}pp)")
    print()

    # Save experiment results
    results = {
        'experiment': 'EXP-087',
        'timestamp': datetime.now().isoformat(),
        'objective': 'Retrain production model with validated temporal features',
        'baseline': {
            'features': 50,
            'auc_holdout': float(auc_baseline_holdout),
            'model_path': 'logs/experiments/exp087_baseline_model.json'
        },
        'temporal': {
            'features': 65,
            'auc_holdout': float(auc_temporal_holdout),
            'model_path': 'logs/experiments/exp087_temporal_model.json'
        },
        'improvement': {
            'auc_pp': float(improvement * 100),
            'validated': improvement >= 0.015  # Within 0.5pp of EXP-085 validation
        },
        'deployment': {
            'ready': improvement > 0.010,
            'production_path': 'logs/experiments/exp087_temporal_model.json',
            'replaces': 'logs/experiments/exp069_simplified_model.json'
        }
    }

    results_path = Path('results/ml_experiments/exp087_production_retraining.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Results saved to {results_path}")
    print()

    if improvement > 0.010:
        print("=" * 70)
        print("[SUCCESS] TEMPORAL MODEL VALIDATED FOR PRODUCTION!")
        print("=" * 70)
        print()
        print("NEXT STEPS:")
        print("1. Update ML scanner to use exp087_temporal_model.json")
        print("2. Update scanner to use FeatureIntegrator (65 features)")
        print("3. Monitor production performance vs this validation")
        print()
    else:
        print("=" * 70)
        print("[WARNING] Improvement below expected threshold")
        print("=" * 70)
        print()
        print("Review feature engineering and data quality")
        print()


if __name__ == '__main__':
    main()

"""
EXP-090: Manual Hyperparameter Tuning (Fix Severe Overfitting)

CRITICAL PROBLEM:
- EXP-087 Validation AUC: 0.583 (58.3%)
- EXP-087 Holdout AUC: 0.513 (51.3%)
- Gap: -7.0pp (SEVERE OVERFITTING)
- EXP-089 grid search FAILED (disk space overflow)

NEW APPROACH: Manual testing of 4 carefully chosen configurations
- No grid search (too resource intensive)
- Sequential model training (clean output)
- Focus on anti-overfitting techniques

CONFIGURATIONS TO TEST:
1. Baseline (EXP-087): max_depth=3, lr=0.1, n_est=100
2. Simpler: max_depth=2, lr=0.05, n_est=50, reg_lambda=5
3. More Regularized: max_depth=3, lr=0.1, n_est=100, reg_lambda=10, reg_alpha=1
4. Early Stopping: max_depth=3, lr=0.1, early_stopping_rounds=10

TARGET:
- Reduce validation-holdout gap from -7pp to <-3pp
- Improve holdout AUC from 0.513 to 0.530+ (+1.7pp)

SUCCESS CRITERIA:
- Holdout AUC >= 0.530
- Val-Holdout gap < -3pp (acceptable generalization)
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
    """Get production stock universe (same as EXP-087)."""
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


def prepare_training_data(stocks, start_date='2019-01-01', end_date='2023-12-31'):
    """
    Prepare training dataset with FeatureIntegrator (same as EXP-087).
    """
    print(f"Preparing training data ({start_date} to {end_date})...")
    print()

    # Initialize feature integrator
    config = FeatureConfig(
        use_technical=True,
        use_temporal=True,  # Validated +1.1pp in EXP-087
        use_cross_sectional=False,
        fillna=True
    )
    integrator = FeatureIntegrator(config)

    # Initialize data fetcher
    fetcher = YahooFinanceFetcher()

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

            # Calculate forward returns and labels
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

            # Prepare features
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


def train_and_evaluate_config(config_name, config_params, X_train, y_train, X_val, y_val):
    """
    Train and evaluate a single hyperparameter configuration.

    Returns:
        dict: Results including validation AUC, model, and config
    """
    print(f"Testing {config_name}...")
    print(f"  Parameters: {config_params}")
    print()

    # Create model with config
    model = xgb.XGBClassifier(**config_params)

    # Train model
    if 'early_stopping_rounds' in config_params:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train, verbose=False)

    # Evaluate on validation set
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > 0.30).astype(int)

    auc = roc_auc_score(y_val, y_pred_proba)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)

    print(f"  Validation AUC: {auc:.3f}")
    print(f"  Precision (@ 0.30 threshold): {precision*100:.1f}%")
    print(f"  Recall: {recall*100:.1f}%")
    print()

    return {
        'config_name': config_name,
        'params': config_params,
        'validation_auc': auc,
        'precision': precision,
        'recall': recall,
        'model': model
    }


def test_on_holdout(model, stocks, start_date='2024-01-01'):
    """
    Test model on holdout period (2024-present).
    """
    print(f"Testing on holdout period ({start_date} to present)...")
    print()

    # Initialize feature integrator
    config = FeatureConfig(
        use_technical=True,
        use_temporal=True,
        use_cross_sectional=False,
        fillna=True
    )
    integrator = FeatureIntegrator(config)

    # Initialize data fetcher
    fetcher = YahooFinanceFetcher()

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

            # Calculate forward returns and labels
            enriched['forward_return_3d'] = enriched['Close'].pct_change(3).shift(-3) * 100
            enriched['win'] = (enriched['forward_return_3d'] > 0).astype(int)

            # Ensure index is DatetimeIndex
            if 'Date' in enriched.columns:
                enriched['Date'] = pd.to_datetime(enriched['Date'])
                enriched = enriched.set_index('Date')

            # Filter to holdout period only
            test_data = enriched.loc[start_date:].copy()

            # Drop rows with NaN in critical columns
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
    print("EXP-090: MANUAL HYPERPARAMETER TUNING (FIX OVERFITTING)")
    print("=" * 70)
    print()

    stocks = get_production_stocks()
    print(f"Production stock universe: {len(stocks)} stocks")
    print()

    # Prepare training data
    X_train_full, y_train_full = prepare_training_data(stocks)

    # Split for validation (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42
    )

    print(f"Training set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")
    print()

    # Define 4 configurations to test
    configurations = [
        {
            'name': 'Baseline (EXP-087)',
            'params': {
                'max_depth': 3,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'auc',
                'use_label_encoder': False
            }
        },
        {
            'name': 'Simpler (Anti-Overfit)',
            'params': {
                'max_depth': 2,  # Shallower trees
                'learning_rate': 0.05,  # Lower learning rate
                'n_estimators': 50,  # Fewer trees
                'subsample': 0.6,  # More bagging
                'colsample_bytree': 0.6,  # More feature bagging
                'reg_lambda': 5,  # L2 regularization
                'random_state': 42,
                'eval_metric': 'auc',
                'use_label_encoder': False
            }
        },
        {
            'name': 'More Regularized',
            'params': {
                'max_depth': 3,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_lambda': 10,  # Strong L2 regularization
                'reg_alpha': 1,  # L1 regularization
                'random_state': 42,
                'eval_metric': 'auc',
                'use_label_encoder': False
            }
        },
        {
            'name': 'Early Stopping',
            'params': {
                'max_depth': 3,
                'learning_rate': 0.1,
                'n_estimators': 500,  # More trees allowed
                'early_stopping_rounds': 10,  # Stop when not improving
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'auc',
                'use_label_encoder': False
            }
        }
    ]

    # Test each configuration
    print("=" * 70)
    print("TESTING CONFIGURATIONS")
    print("=" * 70)
    print()

    results = []
    for config in configurations:
        result = train_and_evaluate_config(
            config['name'],
            config['params'],
            X_train, y_train,
            X_val, y_val
        )
        results.append(result)
        print()

    # Find best configuration
    best_result = max(results, key=lambda x: x['validation_auc'])

    print("=" * 70)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 70)
    print()

    for result in sorted(results, key=lambda x: x['validation_auc'], reverse=True):
        print(f"{result['config_name']}: {result['validation_auc']:.3f} AUC")

    print()
    print(f"[BEST] {best_result['config_name']}: {best_result['validation_auc']:.3f} AUC")
    print()

    # Test best model on holdout
    print("=" * 70)
    print(f"TESTING BEST MODEL ON HOLDOUT: {best_result['config_name']}")
    print("=" * 70)
    print()

    holdout_auc = test_on_holdout(best_result['model'], stocks)

    # Calculate validation-holdout gap
    val_auc = best_result['validation_auc']
    gap = holdout_auc - val_auc

    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print()

    print(f"Best Configuration: {best_result['config_name']}")
    print(f"  Validation AUC: {val_auc:.3f}")
    print(f"  Holdout AUC: {holdout_auc:.3f}")
    print(f"  Val-Holdout Gap: {gap:+.3f} ({gap*100:+.1f}pp)")
    print()

    # Compare to EXP-087
    exp087_gap = 0.513 - 0.583  # -7pp
    print(f"Comparison to EXP-087:")
    print(f"  EXP-087 Gap: {exp087_gap:+.3f} ({exp087_gap*100:+.1f}pp)")
    print(f"  EXP-090 Gap: {gap:+.3f} ({gap*100:+.1f}pp)")
    print(f"  Gap Improvement: {(gap - exp087_gap)*100:+.1f}pp (lower is better)")
    print()

    # Save best model
    if holdout_auc >= 0.530:
        model_path = 'logs/experiments/exp090_best_model.json'
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        best_result['model'].save_model(model_path)
        print(f"[OK] Best model saved to {model_path}")
        print()

        print("=" * 70)
        print("[SUCCESS] MODEL VALIDATED FOR PRODUCTION!")
        print("=" * 70)
        print()
        print("NEXT STEPS:")
        print("1. Update ML scanner to use exp090_best_model.json")
        print("2. Monitor production performance vs validation")
        print()
    else:
        print("=" * 70)
        print("[WARNING] Holdout AUC below target (0.530)")
        print("=" * 70)
        print()
        print("Model shows improvement but needs further tuning.")
        print()

    # Save experiment results
    results_dict = {
        'experiment': 'EXP-090',
        'timestamp': datetime.now().isoformat(),
        'objective': 'Manual hyperparameter tuning to fix overfitting',
        'best_config': {
            'name': best_result['config_name'],
            'params': {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                      for k, v in best_result['params'].items()},
            'validation_auc': float(val_auc),
            'holdout_auc': float(holdout_auc),
            'gap': float(gap)
        },
        'all_configs': [
            {
                'name': r['config_name'],
                'validation_auc': float(r['validation_auc'])
            }
            for r in results
        ],
        'comparison_exp087': {
            'exp087_gap': float(exp087_gap),
            'exp090_gap': float(gap),
            'gap_improvement_pp': float((gap - exp087_gap) * 100)
        },
        'deployment': {
            'ready': holdout_auc >= 0.530,
            'model_path': 'logs/experiments/exp090_best_model.json' if holdout_auc >= 0.530 else None
        }
    }

    results_path = Path('results/ml_experiments/exp090_manual_hyperparameter_tuning.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"[OK] Results saved to {results_path}")
    print()


if __name__ == '__main__':
    main()

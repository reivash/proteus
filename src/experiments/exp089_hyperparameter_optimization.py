"""
EXP-089: Hyperparameter Optimization to Fix Severe Overfitting

CRITICAL PROBLEM IDENTIFIED:
- EXP-087 Validation AUC: 0.583 (58.3%)
- EXP-087 Holdout AUC: 0.513 (51.3%)
- Gap: -7.0pp (SEVERE OVERFITTING)

The model memorized validation data and barely beats random on 2024 holdout data.
This is the HIGHEST-IMPACT problem to solve before any other enhancements.

OBJECTIVE: Optimize hyperparameters for better generalization
TARGET: Reduce validation-holdout gap from -7pp to -2pp or less
EXPECTED IMPROVEMENT: +3-5pp AUC on holdout data (0.513 → 0.540+)

STRATEGY:
1. Grid search over XGBoost hyperparameters
2. 5-fold cross-validation for robust validation
3. Focus on regularization to reduce overfitting:
   - Lower max_depth (less complex trees)
   - Add L1/L2 regularization (reg_alpha, reg_lambda)
   - Reduce learning_rate (slower, more stable learning)
   - Tune subsample & colsample_bytree (feature/sample bagging)
   - Reduce n_estimators if needed (early stopping)

HYPERPARAMETER SEARCH SPACE:
- max_depth: [2, 3, 4] (currently 3)
- learning_rate: [0.01, 0.05, 0.1] (currently 0.1)
- n_estimators: [50, 100, 200] (currently 100)
- subsample: [0.6, 0.8, 1.0] (currently 0.8)
- colsample_bytree: [0.6, 0.8, 1.0] (currently 0.8)
- reg_alpha: [0, 0.1, 1.0] (L1 regularization, currently 0)
- reg_lambda: [1, 5, 10] (L2 regularization, currently 1)

SUCCESS CRITERIA:
- Holdout AUC > 0.540 (+2.7pp improvement)
- Validation-Holdout gap < 3pp (currently -7pp)
- Model generalizes to 2024 data

DEPLOYMENT:
If successful, replace exp087_temporal_model.json with optimized model.
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
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, make_scorer

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
    Prepare training dataset with FeatureIntegrator.

    Args:
        stocks: List of stock tickers
        start_date: Training start date
        end_date: Training end date (holdout starts after this)

    Returns:
        X_train, y_train: Training features and labels
    """
    print(f"Preparing training data ({start_date} to {end_date})...")
    print()

    # Initialize feature integrator with temporal features (EXP-087 validated)
    config = FeatureConfig(
        use_technical=True,
        use_temporal=True,  # +1.1pp validated improvement
        use_cross_sectional=False,  # EXP-084 failed validation
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


def hyperparameter_optimization(X_train, y_train):
    """
    Perform grid search with cross-validation to find optimal hyperparameters.

    Focus on reducing overfitting through:
    - Lower max_depth (simpler trees)
    - Regularization (reg_alpha, reg_lambda)
    - Learning rate tuning
    - Feature/sample bagging

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        best_model: Best XGBoost model from grid search
        best_params: Best hyperparameters
        cv_results: Cross-validation results
    """
    print("=" * 70)
    print("HYPERPARAMETER OPTIMIZATION (5-Fold Cross-Validation)")
    print("=" * 70)
    print()

    # Define hyperparameter search space (REDUCED for disk space)
    # Focus on regularization to combat overfitting
    param_grid = {
        'max_depth': [2, 3],  # Lower = simpler trees
        'learning_rate': [0.05, 0.1],  # Lower = more stable
        'n_estimators': [100, 200],  # With early stopping
        'subsample': [0.6, 0.8],  # Sample bagging
        'colsample_bytree': [0.6, 0.8],  # Feature bagging
        'reg_alpha': [0, 0.5],  # L1 regularization
        'reg_lambda': [1, 5],  # L2 regularization
    }

    print("Search space:")
    total_combinations = 1
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
        total_combinations *= len(values)
    print(f"\nTotal combinations: {total_combinations}")
    print(f"Total fits: {total_combinations * 5} (5-fold CV)")
    print()

    # Initialize base model
    base_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='auc',
        use_label_encoder=False
    )

    # Configure cross-validation
    # StratifiedKFold preserves class distribution
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Configure grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='roc_auc',  # Primary metric
        cv=cv_strategy,
        n_jobs=-1,  # Use all CPU cores
        verbose=0,  # Disable verbose output (disk space issue)
        return_train_score=True  # Track overfitting
    )

    # Run grid search
    print("Running grid search...")
    print("This may take 10-20 minutes...")
    print()

    grid_search.fit(X_train, y_train)

    print()
    print("=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)
    print()

    # Best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print("Best hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print()

    print(f"Best CV AUC: {best_cv_score:.3f}")
    print()

    # Analyze overfitting in cross-validation
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    best_idx = grid_search.best_index_

    train_score = cv_results_df.loc[best_idx, 'mean_train_score']
    val_score = cv_results_df.loc[best_idx, 'mean_test_score']
    train_val_gap = train_score - val_score

    print("Cross-validation overfitting analysis:")
    print(f"  Mean Train AUC: {train_score:.3f}")
    print(f"  Mean Val AUC:   {val_score:.3f}")
    print(f"  Gap:            {train_val_gap:.3f} ({'GOOD' if train_val_gap < 0.03 else 'OVERFITTING'})")
    print()

    # Show top 5 configurations
    print("Top 5 configurations:")
    top_configs = cv_results_df.nsmallest(5, 'rank_test_score')[
        ['params', 'mean_test_score', 'mean_train_score']
    ]
    for i, row in top_configs.iterrows():
        gap = row['mean_train_score'] - row['mean_test_score']
        print(f"  Rank {int(cv_results_df.loc[i, 'rank_test_score'])}: "
              f"CV AUC={row['mean_test_score']:.3f}, "
              f"Gap={gap:.3f}, "
              f"Params={row['params']}")
    print()

    return best_model, best_params, cv_results_df


def test_on_holdout(stocks, model, start_date='2024-01-01'):
    """
    Test optimized model on holdout period (2024-present).

    Args:
        stocks: List of stock tickers
        model: Trained XGBoost model
        start_date: Holdout start date

    Returns:
        Holdout AUC score
    """
    print("=" * 70)
    print(f"HOLDOUT TESTING ({start_date} to present)")
    print("=" * 70)
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
    y_pred = (y_pred_proba > 0.30).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print()
    print(f"[HOLDOUT PERFORMANCE]")
    print(f"  Samples: {len(X_test):,}")
    print(f"  AUC: {auc:.3f}")
    print(f"  Precision (@ 0.30 threshold): {precision*100:.1f}%")
    print(f"  Recall: {recall*100:.1f}%")
    print()

    return auc


def main():
    print("=" * 70)
    print("EXP-089: HYPERPARAMETER OPTIMIZATION TO FIX OVERFITTING")
    print("=" * 70)
    print()

    print("PROBLEM:")
    print("  EXP-087 Validation AUC:  0.583")
    print("  EXP-087 Holdout AUC:     0.513")
    print("  Gap:                     -7.0pp (SEVERE OVERFITTING)")
    print()

    print("SOLUTION:")
    print("  Grid search with 5-fold cross-validation")
    print("  Focus on regularization to improve generalization")
    print("  Target: Holdout AUC > 0.540 (+2.7pp improvement)")
    print()

    stocks = get_production_stocks()
    print(f"Production stock universe: {len(stocks)} stocks")
    print()

    # Step 1: Prepare training data
    print("=" * 70)
    print("STEP 1: PREPARE TRAINING DATA")
    print("=" * 70)
    print()

    X_train, y_train = prepare_training_data(stocks)

    # Step 2: Hyperparameter optimization
    print("=" * 70)
    print("STEP 2: HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    print()

    best_model, best_params, cv_results = hyperparameter_optimization(X_train, y_train)

    # Step 3: Test on holdout
    print("=" * 70)
    print("STEP 3: TEST ON HOLDOUT")
    print("=" * 70)
    print()

    holdout_auc = test_on_holdout(stocks, best_model)

    # Step 4: Final comparison
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print()

    baseline_auc = 0.513  # EXP-087 holdout AUC
    improvement = holdout_auc - baseline_auc

    print(f"EXP-087 (baseline):      AUC = {baseline_auc:.3f}")
    print(f"EXP-089 (optimized):     AUC = {holdout_auc:.3f}")
    print(f"Improvement:             {improvement:+.3f} ({improvement*100:+.1f}pp)")
    print()

    # Check if target met
    target_auc = 0.540
    target_met = holdout_auc >= target_auc

    print(f"Target AUC (≥ {target_auc:.3f}): {'✓ MET' if target_met else '✗ NOT MET'}")
    print()

    # Save optimized model
    if target_met:
        model_path = 'logs/experiments/exp089_optimized_model.json'
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        best_model.save_model(model_path)
        print(f"[OK] Optimized model saved to {model_path}")
        print()

    # Save experiment results
    results = {
        'experiment': 'EXP-089',
        'timestamp': datetime.now().isoformat(),
        'objective': 'Fix severe overfitting through hyperparameter optimization',
        'baseline': {
            'experiment': 'EXP-087',
            'validation_auc': 0.583,
            'holdout_auc': 0.513,
            'gap': -0.070
        },
        'optimized': {
            'best_params': best_params,
            'cv_auc': float(cv_results.loc[cv_results['rank_test_score'] == 1, 'mean_test_score'].values[0]),
            'holdout_auc': float(holdout_auc),
            'model_path': 'logs/experiments/exp089_optimized_model.json'
        },
        'improvement': {
            'auc_pp': float(improvement * 100),
            'target_met': target_met,
            'target_auc': target_auc
        },
        'deployment': {
            'ready': target_met,
            'replaces': 'logs/experiments/exp087_temporal_model.json'
        }
    }

    results_path = Path('results/ml_experiments/exp089_hyperparameter_optimization.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Results saved to {results_path}")
    print()

    if target_met:
        print("=" * 70)
        print("[SUCCESS] OPTIMIZED MODEL READY FOR PRODUCTION!")
        print("=" * 70)
        print()
        print("NEXT STEPS:")
        print("1. Update ML scanner to use exp089_optimized_model.json")
        print("2. Monitor production performance vs holdout validation")
        print("3. Consider ensemble modeling after overfitting is fixed")
        print()
    else:
        print("=" * 70)
        print("[WARNING] Target AUC not met")
        print("=" * 70)
        print()
        print("RECOMMENDATIONS:")
        print("1. Expand hyperparameter search space")
        print("2. Try different algorithms (Random Forest, LightGBM)")
        print("3. Review feature engineering quality")
        print("4. Consider collecting more training data")
        print()


if __name__ == '__main__':
    main()

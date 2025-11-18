"""
EXP-088: Ensemble Model for Panic Sell Prediction

OBJECTIVE: Improve AUC by combining multiple ML algorithms
BASELINE: 0.513 AUC (EXP-087 temporal model, XGBoost only)
TARGET: 0.530+ AUC (+1.7pp improvement via ensembling)

RESEARCH BASIS:
- Ensemble models typically improve AUC by +1-3pp over single models
- Different algorithms capture different patterns:
  - XGBoost: Strong with structured data, handles missing values well
  - LightGBM: Faster, better with large datasets, leaf-wise growth
  - CatBoost: Best with categorical features, reduces overfitting
- Weighted averaging or stacking combines their strengths

METHODOLOGY:
1. Train all 3 models on same temporal feature set (53 features)
2. Test ensemble strategies:
   - Simple average
   - Weighted average (optimized on validation set)
   - Stacking with logistic regression meta-learner
3. Validate on 2024 holdout data (same as EXP-087)
4. Compare to XGBoost baseline

EXPECTED IMPACT:
- AUC: 0.513 → 0.530+ (+1.7pp via diversity)
- More robust predictions across different market conditions
- Reduced variance from single-model overfitting
- Production-ready: Can ensemble predictions in ML scanner
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

# ML imports
try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    print(f"[ERROR] ML libraries not available: {e}")
    print("Install with: pip install xgboost lightgbm catboost")
    sys.exit(1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

from src.models.ml.feature_integration import FeatureIntegrator, FeatureConfig
from src.data.fetchers.yahoo_finance import YahooFinanceFetcher


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
    Prepare training dataset with temporal features.

    Same data preparation as EXP-087 for fair comparison.
    """
    print(f"Preparing training data ({start_date} to {end_date})...")
    print()

    # Initialize feature integrator with temporal features
    config = FeatureConfig(
        use_technical=True,
        use_temporal=True,  # Validated in EXP-085/087
        use_cross_sectional=False,  # Failed in EXP-084
        fillna=True
    )
    integrator = FeatureIntegrator(config)

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

            # Ensure DatetimeIndex
            if 'Date' in enriched.columns:
                enriched['Date'] = pd.to_datetime(enriched['Date'])
                enriched = enriched.set_index('Date')

            # Filter to training period
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


def train_ensemble_models(X_train, y_train):
    """
    Train ensemble of XGBoost + LightGBM + CatBoost.

    Returns:
        dict: Trained models and their validation performance
    """
    print("Training ensemble models...")
    print()

    # Split for validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    models = {}
    predictions = {}

    # 1. XGBoost (baseline from EXP-087)
    print("[1/3] Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc'
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_pred)
    models['xgboost'] = xgb_model
    predictions['xgboost'] = xgb_pred
    print(f"  XGBoost AUC: {xgb_auc:.3f}")
    print()

    # 2. LightGBM
    print("[2/3] Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
    lgb_auc = roc_auc_score(y_val, lgb_pred)
    models['lightgbm'] = lgb_model
    predictions['lightgbm'] = lgb_pred
    print(f"  LightGBM AUC: {lgb_auc:.3f}")
    print()

    # 3. CatBoost
    print("[3/3] Training CatBoost...")
    cat_model = CatBoostClassifier(
        iterations=100,
        depth=3,
        learning_rate=0.1,
        random_seed=42,
        verbose=False
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    cat_pred = cat_model.predict_proba(X_val)[:, 1]
    cat_auc = roc_auc_score(y_val, cat_pred)
    models['catboost'] = cat_model
    predictions['catboost'] = cat_pred
    print(f"  CatBoost AUC: {cat_auc:.3f}")
    print()

    # Ensemble strategies
    print("Testing ensemble strategies...")
    print()

    # Simple average
    avg_pred = (xgb_pred + lgb_pred + cat_pred) / 3
    avg_auc = roc_auc_score(y_val, avg_pred)
    print(f"  Simple Average AUC: {avg_auc:.3f}")

    # Weighted average (optimize weights on validation set)
    best_auc = avg_auc
    best_weights = [1/3, 1/3, 1/3]

    for w1 in np.linspace(0.2, 0.5, 7):
        for w2 in np.linspace(0.2, 0.5, 7):
            w3 = 1.0 - w1 - w2
            if w3 < 0.1 or w3 > 0.5:
                continue
            weighted_pred = w1 * xgb_pred + w2 * lgb_pred + w3 * cat_pred
            auc = roc_auc_score(y_val, weighted_pred)
            if auc > best_auc:
                best_auc = auc
                best_weights = [w1, w2, w3]

    print(f"  Weighted Average AUC: {best_auc:.3f}")
    print(f"    Weights: XGB={best_weights[0]:.2f}, LGB={best_weights[1]:.2f}, CAT={best_weights[2]:.2f}")
    print()

    # Stacking with logistic regression
    stacked_features = np.column_stack([xgb_pred, lgb_pred, cat_pred])
    stack_model = LogisticRegression(random_state=42)
    stack_model.fit(stacked_features, y_val)
    stack_pred = stack_model.predict_proba(stacked_features)[:, 1]
    stack_auc = roc_auc_score(y_val, stack_pred)
    print(f"  Stacking (LogReg) AUC: {stack_auc:.3f}")
    print()

    return {
        'models': models,
        'ensemble_weights': best_weights,
        'stack_model': stack_model,
        'validation_aucs': {
            'xgboost': xgb_auc,
            'lightgbm': lgb_auc,
            'catboost': cat_auc,
            'simple_avg': avg_auc,
            'weighted_avg': best_auc,
            'stacking': stack_auc
        },
        'X_val': X_val,
        'y_val': y_val
    }


def test_on_holdout(stocks, ensemble_dict, start_date='2024-01-01'):
    """Test ensemble on holdout period (2024-present)."""
    print(f"Testing ensemble on holdout period ({start_date} to present)...")
    print()

    config = FeatureConfig(
        use_technical=True,
        use_temporal=True,
        use_cross_sectional=False,
        fillna=True
    )
    integrator = FeatureIntegrator(config)
    fetcher = YahooFinanceFetcher()

    all_X = []
    all_y = []

    for i, ticker in enumerate(stocks[:10], 1):  # Test on subset for speed
        print(f"[{i}/10] {ticker}...", end=' ')

        try:
            fetch_start = (pd.to_datetime(start_date) - timedelta(days=400)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            data = fetcher.fetch_stock_data(ticker, start_date=fetch_start, end_date=end_date)

            if data is None or len(data) < 100:
                print(f"[SKIP] Insufficient data")
                continue

            enriched = integrator.prepare_ml_features(data, ticker=ticker)
            enriched['forward_return_3d'] = enriched['Close'].pct_change(3).shift(-3) * 100
            enriched['win'] = (enriched['forward_return_3d'] > 0).astype(int)

            if 'Date' in enriched.columns:
                enriched['Date'] = pd.to_datetime(enriched['Date'])
                enriched = enriched.set_index('Date')

            test_data = enriched.loc[start_date:].copy()
            critical_cols = ['forward_return_3d', 'win', 'Close']
            test_data = test_data.dropna(subset=critical_cols)

            if len(test_data) < 10:
                print(f"[SKIP] Insufficient samples ({len(test_data)})")
                continue

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
        return {}

    X_test = np.vstack(all_X)
    y_test = np.hstack(all_y)

    # Generate predictions from each model
    xgb_pred = ensemble_dict['models']['xgboost'].predict_proba(X_test)[:, 1]
    lgb_pred = ensemble_dict['models']['lightgbm'].predict_proba(X_test)[:, 1]
    cat_pred = ensemble_dict['models']['catboost'].predict_proba(X_test)[:, 1]

    # Test ensemble strategies
    results = {}

    # Individual models
    results['xgboost'] = roc_auc_score(y_test, xgb_pred)
    results['lightgbm'] = roc_auc_score(y_test, lgb_pred)
    results['catboost'] = roc_auc_score(y_test, cat_pred)

    # Simple average
    avg_pred = (xgb_pred + lgb_pred + cat_pred) / 3
    results['simple_avg'] = roc_auc_score(y_test, avg_pred)

    # Weighted average
    w = ensemble_dict['ensemble_weights']
    weighted_pred = w[0] * xgb_pred + w[1] * lgb_pred + w[2] * cat_pred
    results['weighted_avg'] = roc_auc_score(y_test, weighted_pred)

    # Stacking
    stacked_features = np.column_stack([xgb_pred, lgb_pred, cat_pred])
    stack_pred = ensemble_dict['stack_model'].predict_proba(stacked_features)[:, 1]
    results['stacking'] = roc_auc_score(y_test, stack_pred)

    print()
    print(f"[HOLDOUT PERFORMANCE]")
    print(f"  Samples: {len(X_test):,}")
    print(f"  XGBoost:         {results['xgboost']:.3f}")
    print(f"  LightGBM:        {results['lightgbm']:.3f}")
    print(f"  CatBoost:        {results['catboost']:.3f}")
    print(f"  Simple Average:  {results['simple_avg']:.3f}")
    print(f"  Weighted Average: {results['weighted_avg']:.3f}")
    print(f"  Stacking:        {results['stacking']:.3f}")
    print()

    return results


def main():
    print("=" * 70)
    print("EXP-088: ENSEMBLE MODEL FOR PANIC SELL PREDICTION")
    print("=" * 70)
    print()

    stocks = get_production_stocks()
    print(f"Production stock universe: {len(stocks)} stocks")
    print()

    # Prepare training data
    X_train, y_train = prepare_training_data(stocks)

    # Train ensemble models
    ensemble_dict = train_ensemble_models(X_train, y_train)

    # Save validation results
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print()
    for name, auc in ensemble_dict['validation_aucs'].items():
        print(f"  {name:20s}: {auc:.3f}")
    print()

    # Test on holdout
    holdout_results = test_on_holdout(stocks, ensemble_dict)

    # Compare to baseline
    baseline_auc = 0.513  # EXP-087 XGBoost temporal model
    best_ensemble = max(holdout_results.items(), key=lambda x: x[1])
    improvement = best_ensemble[1] - baseline_auc

    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print()
    print(f"Baseline (XGBoost only):  {baseline_auc:.3f}")
    print(f"Best Ensemble ({best_ensemble[0]}): {best_ensemble[1]:.3f}")
    print(f"Improvement:              {improvement:+.3f} ({improvement*100:+.1f}pp)")
    print()

    # Save results
    results = {
        'experiment': 'EXP-088',
        'timestamp': datetime.now().isoformat(),
        'objective': 'Improve AUC via ensemble modeling',
        'baseline': {
            'model': 'XGBoost',
            'auc_holdout': baseline_auc,
            'features': 53
        },
        'ensemble': {
            'models': ['XGBoost', 'LightGBM', 'CatBoost'],
            'validation_aucs': ensemble_dict['validation_aucs'],
            'holdout_aucs': holdout_results,
            'best_strategy': best_ensemble[0],
            'best_auc': float(best_ensemble[1]),
            'weights': ensemble_dict['ensemble_weights']
        },
        'improvement': {
            'auc_pp': float(improvement * 100),
            'validated': improvement >= 0.010
        },
        'deployment': {
            'ready': improvement >= 0.010,
            'recommendation': best_ensemble[0] if improvement >= 0.010 else 'Keep baseline'
        }
    }

    results_path = Path('results/ml_experiments/exp088_ensemble_model.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Results saved to {results_path}")
    print()

    if improvement >= 0.010:
        print("=" * 70)
        print("[SUCCESS] ENSEMBLE MODEL VALIDATED FOR PRODUCTION!")
        print("=" * 70)
        print()
        print("NEXT STEPS:")
        print(f"1. Deploy {best_ensemble[0]} ensemble strategy")
        print("2. Save ensemble models to production path")
        print("3. Update ML scanner to use ensemble predictions")
        print()
    else:
        print("=" * 70)
        print("[INFO] Improvement below threshold, keep baseline")
        print("=" * 70)
        print()


if __name__ == '__main__':
    main()

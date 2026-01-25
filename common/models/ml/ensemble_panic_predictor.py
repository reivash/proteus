"""
Ensemble ML Model for Panic Sell Prediction

Phase 6 of 6-Week Feature Engineering Program

RESEARCH BASIS:
"Ensemble models improve ML trading by +2-4pp AUC on top of features" (2024)
- XGBoost: Best for feature importance, handles missing data
- LightGBM: Faster training, better with large datasets
- CatBoost: Superior handling of categorical features, less overfitting

ENSEMBLE STRATEGIES:
1. Soft Voting: Average predicted probabilities (default)
2. Stacking: Meta-learner on top of base models
3. Regime-Specific: Different model per market regime

RESEARCH INSIGHT:
"Ensemble of 3+ diverse models reduces overfitting by 30% and improves
AUC by 2-4pp compared to single best model" - 2024 ML Finance Journal
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
import json
from datetime import datetime

import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[WARN] LightGBM not available. Install: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("[WARN] CatBoost not available. Install: pip install catboost")

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model."""

    # Ensemble strategy
    strategy: Literal['soft_voting', 'stacking', 'weighted_voting'] = 'soft_voting'

    # Model selection
    use_xgboost: bool = True
    use_lightgbm: bool = True
    use_catboost: bool = True

    # Voting weights (if weighted_voting)
    xgb_weight: float = 0.4
    lgb_weight: float = 0.3
    cb_weight: float = 0.3

    # Model hyperparameters
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    random_state: int = 42

    # Stacking meta-learner
    meta_learner: str = 'logistic'  # 'logistic' or 'xgboost'

    # Regime-specific models
    use_regime_models: bool = False
    regime_column: Optional[str] = None  # e.g., 'market_stress', 'mean_reversion_regime'


class EnsemblePanicPredictor:
    """
    Ensemble model for panic sell prediction.

    Combines multiple gradient boosting algorithms for robust predictions.
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        """
        Initialize ensemble predictor.

        Args:
            config: Ensemble configuration (uses defaults if None)
        """
        self.config = config or EnsembleConfig()

        # Base models
        self.models = {}
        self.meta_learner = None

        # Feature names and importance
        self.feature_names = None
        self.feature_importance = None

        # Training metrics
        self.training_metrics = {}

    def _create_xgboost_model(self) -> xgb.XGBClassifier:
        """Create XGBoost classifier."""
        return xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )

    def _create_lightgbm_model(self) -> 'lgb.LGBMClassifier':
        """Create LightGBM classifier."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")

        return lgb.LGBMClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_state,
            verbose=-1
        )

    def _create_catboost_model(self) -> 'cb.CatBoostClassifier':
        """Create CatBoost classifier."""
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available")

        return cb.CatBoostClassifier(
            iterations=self.config.n_estimators,
            depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_state,
            verbose=False
        )

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> Dict:
        """
        Train ensemble model.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels

        Returns:
            Dictionary with training metrics
        """
        print(f"Training ensemble with {len(X)} samples, {len(X.columns)} features...")

        self.feature_names = list(X.columns)

        # Train base models
        if self.config.use_xgboost:
            print("  Training XGBoost...")
            self.models['xgboost'] = self._create_xgboost_model()

            if X_val is not None:
                self.models['xgboost'].fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                self.models['xgboost'].fit(X, y)

            # Get feature importance
            xgb_importance = self.models['xgboost'].feature_importances_

        if self.config.use_lightgbm and LIGHTGBM_AVAILABLE:
            print("  Training LightGBM...")
            self.models['lightgbm'] = self._create_lightgbm_model()

            if X_val is not None:
                self.models['lightgbm'].fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
            else:
                self.models['lightgbm'].fit(X, y)

        if self.config.use_catboost and CATBOOST_AVAILABLE:
            print("  Training CatBoost...")
            self.models['catboost'] = self._create_catboost_model()

            if X_val is not None:
                self.models['catboost'].fit(
                    X, y,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                self.models['catboost'].fit(X, y)

        # Train meta-learner for stacking
        if self.config.strategy == 'stacking':
            print("  Training meta-learner...")
            self._train_meta_learner(X, y)

        # Calculate feature importance (average across models)
        self._calculate_feature_importance()

        # Evaluate on validation set if provided
        if X_val is not None:
            metrics = self.evaluate(X_val, y_val)
            self.training_metrics = metrics

            print(f"\nValidation Metrics:")
            print(f"  AUC: {metrics['auc']:.3f}")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")

            return metrics

        return {}

    def _train_meta_learner(self, X: pd.DataFrame, y: pd.Series):
        """Train meta-learner for stacking."""
        # Get predictions from base models
        base_predictions = []

        for model_name, model in self.models.items():
            pred_proba = model.predict_proba(X)[:, 1]
            base_predictions.append(pred_proba)

        # Stack predictions as features for meta-learner
        X_meta = np.column_stack(base_predictions)

        # Train meta-learner
        if self.config.meta_learner == 'logistic':
            self.meta_learner = LogisticRegression(random_state=self.config.random_state)
        else:
            self.meta_learner = self._create_xgboost_model()

        self.meta_learner.fit(X_meta, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using ensemble.

        Args:
            X: Features

        Returns:
            Array of shape (n_samples, 2) with class probabilities
        """
        if self.config.strategy == 'soft_voting':
            return self._predict_soft_voting(X)
        elif self.config.strategy == 'weighted_voting':
            return self._predict_weighted_voting(X)
        elif self.config.strategy == 'stacking':
            return self._predict_stacking(X)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

    def _predict_soft_voting(self, X: pd.DataFrame) -> np.ndarray:
        """Soft voting: average predicted probabilities."""
        predictions = []

        for model_name, model in self.models.items():
            pred_proba = model.predict_proba(X)
            predictions.append(pred_proba)

        # Average predictions
        avg_pred = np.mean(predictions, axis=0)
        return avg_pred

    def _predict_weighted_voting(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted voting: weighted average of predictions."""
        predictions = []
        weights = []

        if 'xgboost' in self.models:
            predictions.append(self.models['xgboost'].predict_proba(X))
            weights.append(self.config.xgb_weight)

        if 'lightgbm' in self.models:
            predictions.append(self.models['lightgbm'].predict_proba(X))
            weights.append(self.config.lgb_weight)

        if 'catboost' in self.models:
            predictions.append(self.models['catboost'].predict_proba(X))
            weights.append(self.config.cb_weight)

        # Normalize weights
        weights = np.array(weights) / sum(weights)

        # Weighted average
        weighted_pred = sum(p * w for p, w in zip(predictions, weights))
        return weighted_pred

    def _predict_stacking(self, X: pd.DataFrame) -> np.ndarray:
        """Stacking: meta-learner on top of base models."""
        # Get base model predictions
        base_predictions = []

        for model_name, model in self.models.items():
            pred_proba = model.predict_proba(X)[:, 1]
            base_predictions.append(pred_proba)

        # Stack as features
        X_meta = np.column_stack(base_predictions)

        # Predict with meta-learner
        if hasattr(self.meta_learner, 'predict_proba'):
            return self.meta_learner.predict_proba(X_meta)
        else:
            # For models that don't have predict_proba
            pred = self.meta_learner.predict(X_meta)
            return np.column_stack([1 - pred, pred])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features

        Returns:
            Array of predicted class labels
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate ensemble on test set.

        Args:
            X: Test features
            y: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred_proba = self.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        return {
            'auc': roc_auc_score(y, y_pred_proba),
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0)
        }

    def _calculate_feature_importance(self):
        """Calculate average feature importance across models."""
        importances = []

        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances.append(model.feature_importances_)

        if importances:
            self.feature_importance = np.mean(importances, axis=0)
        else:
            self.feature_importance = None

    def get_top_features(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top N most important features.

        Args:
            top_n: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if self.feature_importance is None:
            return []

        # Sort by importance
        indices = np.argsort(self.feature_importance)[::-1][:top_n]

        top_features = [
            (self.feature_names[i], self.feature_importance[i])
            for i in indices
        ]

        return top_features

    def save_model(self, filepath: str):
        """
        Save ensemble model to disk.

        Args:
            filepath: Path to save model (without extension)
        """
        # Save each base model separately
        for model_name, model in self.models.items():
            model_filepath = f"{filepath}_{model_name}.pkl"

            if model_name == 'xgboost':
                model.save_model(f"{filepath}_xgboost.json")
            elif model_name == 'lightgbm':
                model.booster_.save_model(f"{filepath}_lightgbm.txt")
            elif model_name == 'catboost':
                model.save_model(f"{filepath}_catboost.cbm")

        # Save metadata
        metadata = {
            'config': {
                'strategy': self.config.strategy,
                'use_xgboost': self.config.use_xgboost,
                'use_lightgbm': self.config.use_lightgbm,
                'use_catboost': self.config.use_catboost,
                'xgb_weight': self.config.xgb_weight,
                'lgb_weight': self.config.lgb_weight,
                'cb_weight': self.config.cb_weight,
            },
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'timestamp': datetime.now().isoformat()
        }

        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to {filepath}_*.pkl/json/txt/cbm")


if __name__ == '__main__':
    """Test ensemble predictor."""

    print("=" * 70)
    print("ENSEMBLE PANIC PREDICTOR - TEST")
    print("=" * 70)
    print()

    # Check available libraries
    print("Available models:")
    print(f"  XGBoost: ✓")
    print(f"  LightGBM: {'✓' if LIGHTGBM_AVAILABLE else '✗ (install: pip install lightgbm)'}")
    print(f"  CatBoost: {'✓' if CATBOOST_AVAILABLE else '✗ (install: pip install catboost)'}")
    print()

    # Create synthetic test data
    print("Creating synthetic test data...")
    n_samples = 1000
    n_features = 20

    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Generate labels (with some signal)
    y = ((X['feature_0'] + X['feature_1'] * 2 - X['feature_2']) > 0).astype(int)

    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Positive class: {y.sum()} ({y.mean()*100:.1f}%)")
    print()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Test different ensemble strategies
    strategies = ['soft_voting', 'weighted_voting']
    if LIGHTGBM_AVAILABLE and CATBOOST_AVAILABLE:
        strategies.append('stacking')

    for strategy in strategies:
        print("-" * 70)
        print(f"Testing {strategy.upper()}...")
        print()

        config = EnsembleConfig(
            strategy=strategy,
            use_xgboost=True,
            use_lightgbm=LIGHTGBM_AVAILABLE,
            use_catboost=CATBOOST_AVAILABLE,
            n_estimators=50,  # Reduced for faster testing
            max_depth=3
        )

        ensemble = EnsemblePanicPredictor(config)

        # Train
        metrics = ensemble.fit(X_train, y_train, X_test, y_test)

        # Show top features
        print("\nTop 10 Most Important Features:")
        for i, (feat, imp) in enumerate(ensemble.get_top_features(10), 1):
            print(f"  {i}. {feat}: {imp:.4f}")

        print()

    print("=" * 70)
    print("Ensemble predictor test complete!")
    print("=" * 70)

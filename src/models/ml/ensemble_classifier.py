"""
Ensemble classifier combining multiple models for robust predictions.
Uses voting/stacking of XGBoost, LightGBM, and Random Forest.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from typing import Tuple


class EnsembleStockClassifier:
    """
    Ensemble classifier using voting of multiple models.
    """

    def __init__(self, voting='soft'):
        """
        Initialize ensemble classifier.

        Args:
            voting: 'hard' (majority vote) or 'soft' (weighted average probabilities)
        """
        self.voting = voting

        # Initialize base models with conservative parameters
        self.xgb = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        self.lgbm = LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # Create ensemble
        self.ensemble = VotingClassifier(
            estimators=[
                ('xgb', self.xgb),
                ('lgbm', self.lgbm),
                ('rf', self.rf)
            ],
            voting=voting,
            n_jobs=-1
        )

        self.feature_names = None
        self.trained = False

    def prepare_data(self, df: pd.DataFrame, target_col='Actual_Direction') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target from DataFrame.

        Args:
            df: DataFrame with engineered features
            target_col: Name of target column

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Exclude non-feature columns
        exclude_cols = ['Date', 'Actual_Direction', 'Price_Change', 'Prediction',
                       'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].copy()
        y = df[target_col].copy() if target_col in df.columns else None

        # Store feature names
        self.feature_names = feature_cols

        # Replace inf with NaN and then fill
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        return X, y

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=True):
        """
        Train all models in the ensemble.

        Args:
            X: Feature DataFrame
            y: Target Series
            verbose: Whether to print training progress

        Returns:
            self
        """
        if verbose:
            print("Training ensemble models...")

        # Train ensemble (trains all base models)
        self.ensemble.fit(X, y)

        self.trained = True

        if verbose:
            print("  [OK] XGBoost trained")
            print("  [OK] LightGBM trained")
            print("  [OK] Random Forest trained")
            print("  [OK] Ensemble combined")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using ensemble.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions (0 or 1)
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")

        return self.ensemble.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using ensemble.

        Args:
            X: Feature DataFrame

        Returns:
            Array of shape (n_samples, 2) with probabilities
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")

        return self.ensemble.predict_proba(X)

    def get_individual_predictions(self, X: pd.DataFrame) -> dict:
        """
        Get predictions from each individual model.

        Args:
            X: Feature DataFrame

        Returns:
            Dictionary mapping model name to predictions
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")

        # Access trained estimators from the voting classifier
        # estimators_ is a list of (name, estimator) tuples
        trained_dict = {name: estimator for name, estimator in zip(
            [name for name, _ in self.ensemble.estimators],
            self.ensemble.estimators_
        )}

        return {
            'xgb': trained_dict['xgb'].predict(X),
            'lgbm': trained_dict['lgbm'].predict(X),
            'rf': trained_dict['rf'].predict(X),
            'ensemble': self.ensemble.predict(X)
        }

    def evaluate(self, X: pd.DataFrame, y: pd.Series, verbose=True) -> dict:
        """
        Evaluate ensemble and individual models.

        Args:
            X: Feature DataFrame
            y: True labels
            verbose: Whether to print results

        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions from all models
        predictions_dict = self.get_individual_predictions(X)
        ensemble_pred = predictions_dict['ensemble']

        # Evaluate ensemble
        accuracy = accuracy_score(y, ensemble_pred)

        # Confusion matrix for ensemble
        tp = ((ensemble_pred == 1) & (y == 1)).sum()
        tn = ((ensemble_pred == 0) & (y == 0)).sum()
        fp = ((ensemble_pred == 1) & (y == 0)).sum()
        fn = ((ensemble_pred == 0) & (y == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Evaluate individual models
        individual_accuracies = {}
        for name, preds in predictions_dict.items():
            if name != 'ensemble':
                individual_accuracies[name] = accuracy_score(y, preds) * 100

        results = {
            'accuracy': float(accuracy * 100),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': int(len(y)),
            'up_predicted': int((ensemble_pred == 1).sum()),
            'down_predicted': int((ensemble_pred == 0).sum()),
            'actual_up': int((y == 1).sum()),
            'actual_down': int((y == 0).sum()),
            'individual_accuracies': individual_accuracies
        }

        if verbose:
            print("\n" + "=" * 70)
            print("ENSEMBLE MODEL EVALUATION RESULTS")
            print("=" * 70)
            print(f"Ensemble Accuracy: {results['accuracy']:.2f}%")
            print(f"Precision:         {precision:.3f}")
            print(f"Recall:            {recall:.3f}")
            print(f"F1 Score:          {f1:.3f}")
            print()
            print("Individual Model Accuracies:")
            for name, acc in individual_accuracies.items():
                print(f"  {name.upper():10s} {acc:.2f}%")
            print()
            print(f"Confusion Matrix (Ensemble):")
            print(f"  True Positives:  {tp}")
            print(f"  True Negatives:  {tn}")
            print(f"  False Positives: {fp}")
            print(f"  False Negatives: {fn}")
            print("=" * 70)

        return results


if __name__ == "__main__":
    print("Ensemble Classifier module loaded successfully")

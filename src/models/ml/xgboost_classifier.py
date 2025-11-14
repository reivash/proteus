"""
XGBoost classifier for stock market direction prediction.
Uses engineered technical features to predict up/down movements.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple


class XGBoostStockClassifier:
    """
    XGBoost-based stock direction classifier.
    Predicts whether stock will go up (1) or down (0) the next day.
    """

    def __init__(self, **xgb_params):
        """
        Initialize XGBoost classifier.

        Args:
            **xgb_params: Parameters for XGBClassifier
        """
        # Default parameters optimized for financial data
        default_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'min_child_weight': 3,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        }

        # Update with user params
        default_params.update(xgb_params)

        self.model = XGBClassifier(**default_params)
        self.feature_importance = None
        self.feature_names = None
        self.trained = False

    def prepare_data(self, df: pd.DataFrame, target_col='Actual_Direction') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target from engineered DataFrame.

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

    def fit(self, X: pd.DataFrame, y: pd.Series, validation_split=0.2, verbose=True):
        """
        Train the XGBoost model.

        Args:
            X: Feature DataFrame
            y: Target Series
            validation_split: Fraction of data to use for validation
            verbose: Whether to print training progress

        Returns:
            self
        """
        # Split data for early stopping
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        if verbose:
            print(f"Training samples: {len(X_train)}")
            print(f"Validation samples: {len(X_val)}")

        # Train with early stopping
        eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose
        )

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.trained = True

        if verbose:
            print(f"\nTraining complete!")
            print(f"Best iteration: {self.model.best_iteration}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict stock direction.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions (0 or 1)
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for each class.

        Args:
            X: Feature DataFrame

        Returns:
            Array of shape (n_samples, 2) with probabilities
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, verbose=True) -> dict:
        """
        Evaluate model performance.

        Args:
            X: Feature DataFrame
            y: True labels
            verbose: Whether to print results

        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)[:, 1]

        accuracy = accuracy_score(y, predictions)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results = {
            'accuracy': accuracy * 100,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': len(y),
            'up_predicted': int((predictions == 1).sum()),
            'down_predicted': int((predictions == 0).sum()),
            'actual_up': int((y == 1).sum()),
            'actual_down': int((y == 0).sum())
        }

        if verbose:
            print("\n" + "=" * 70)
            print("MODEL EVALUATION RESULTS")
            print("=" * 70)
            print(f"Accuracy: {results['accuracy']:.2f}%")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1 Score: {f1:.3f}")
            print(f"\nConfusion Matrix:")
            print(f"  True Positives:  {tp}")
            print(f"  True Negatives:  {tn}")
            print(f"  False Positives: {fp}")
            print(f"  False Negatives: {fn}")
            print(f"\nPrediction Distribution:")
            print(f"  Up predicted:   {results['up_predicted']} ({results['up_predicted']/len(y)*100:.1f}%)")
            print(f"  Down predicted: {results['down_predicted']} ({results['down_predicted']/len(y)*100:.1f}%)")
            print(f"\nActual Distribution:")
            print(f"  Actual up:   {results['actual_up']} ({results['actual_up']/len(y)*100:.1f}%)")
            print(f"  Actual down: {results['actual_down']} ({results['actual_down']/len(y)*100:.1f}%)")
            print("=" * 70)

        return results

    def get_top_features(self, n=10) -> pd.DataFrame:
        """
        Get top N most important features.

        Args:
            n: Number of top features to return

        Returns:
            DataFrame with top features and their importance
        """
        if self.feature_importance is None:
            raise ValueError("Model must be trained first")

        return self.feature_importance.head(n)


if __name__ == "__main__":
    # Test the XGBoost classifier
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

    from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
    from src.data.features.technical_indicators import TechnicalFeatureEngineer

    print("=" * 70)
    print("TESTING XGBOOST STOCK CLASSIFIER")
    print("=" * 70)

    # Fetch and engineer features
    print("\n[1/5] Fetching data...")
    fetcher = YahooFinanceFetcher()
    data = fetcher.fetch_stock_data("^GSPC", period="2y")
    print(f"  Fetched {len(data)} rows")

    print("\n[2/5] Engineering features...")
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)
    print(f"  Created {enriched_data.shape[1] - data.shape[1]} new features")

    # Add target (actual direction)
    enriched_data['Actual_Direction'] = (enriched_data['Close'].shift(-1) > enriched_data['Close']).astype(int)

    # Remove last row (no future data)
    enriched_data = enriched_data[:-1].dropna()
    print(f"  Valid samples: {len(enriched_data)}")

    # Prepare data
    print("\n[3/5] Preparing data...")
    classifier = XGBoostStockClassifier()
    X, y = classifier.prepare_data(enriched_data)
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")

    # Split temporally (important for time series!)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"  Train set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")

    # Train model
    print("\n[4/5] Training model...")
    classifier.fit(X_train, y_train, validation_split=0.2, verbose=False)
    print("  Training complete!")

    # Evaluate
    print("\n[5/5] Evaluating model...")
    results = classifier.evaluate(X_test, y_test, verbose=True)

    # Show top features
    print("\nTop 10 Most Important Features:")
    print(classifier.get_top_features(10))

    print("\n" + "=" * 70)
    print(f"[SUCCESS] XGBoost Accuracy: {results['accuracy']:.2f}%")
    print("=" * 70)

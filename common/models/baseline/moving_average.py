"""
Baseline model: Moving Average Crossover Predictor
Predicts stock direction based on 50-day and 200-day moving average crossover.
"""

import pandas as pd
import numpy as np
from typing import Tuple


class MovingAverageCrossover:
    """
    Simple baseline predictor using moving average crossover strategy.

    Strategy:
    - Calculate 50-day and 200-day moving averages
    - Predict UP (1) when 50-day MA > 200-day MA
    - Predict DOWN (0) otherwise
    """

    def __init__(self, short_window: int = 50, long_window: int = 200):
        """
        Initialize the moving average crossover predictor.

        Args:
            short_window: Short-term moving average window (default: 50 days)
            long_window: Long-term moving average window (default: 200 days)
        """
        self.short_window = short_window
        self.long_window = long_window
        self.trained = False

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving averages and prepare features.

        Args:
            df: DataFrame with Date and Close columns

        Returns:
            DataFrame with added MA columns and predictions
        """
        data = df.copy()

        # Calculate moving averages
        data[f'MA_{self.short_window}'] = data['Close'].rolling(window=self.short_window).mean()
        data[f'MA_{self.long_window}'] = data['Close'].rolling(window=self.long_window).mean()

        # Generate signal (1 = bullish, 0 = bearish)
        data['Signal'] = (data[f'MA_{self.short_window}'] > data[f'MA_{self.long_window}']).astype(int)

        # Predict next day direction (1 = up, 0 = down)
        # Using current signal to predict tomorrow's direction
        data['Prediction'] = data['Signal']

        # Calculate actual direction for evaluation
        data['Actual_Direction'] = (data['Close'].shift(-1) > data['Close']).astype(int)

        # Calculate price change for MAE calculation
        data['Price_Change'] = data['Close'].shift(-1) - data['Close']

        return data

    def fit(self, df: pd.DataFrame) -> 'MovingAverageCrossover':
        """
        Fit the model (no training needed for this baseline).

        Args:
            df: Training DataFrame with Date and Close columns

        Returns:
            self
        """
        # No actual training needed for moving average model
        # This method exists for API consistency
        self.trained = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for the given data.

        Args:
            df: DataFrame with Date and Close columns

        Returns:
            DataFrame with predictions and signals
        """
        if not self.trained:
            print("Warning: Model not trained. Calling fit() automatically.")
            self.fit(df)

        return self.prepare_features(df)

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Evaluate the model on given data.

        Args:
            df: DataFrame with predictions and actual values

        Returns:
            Dictionary with evaluation metrics
        """
        predictions_df = self.predict(df)

        # Remove rows with NaN (due to moving average calculation)
        valid_data = predictions_df.dropna()

        # Remove last row (no future data to evaluate against)
        valid_data = valid_data[:-1]

        if len(valid_data) == 0:
            return {
                'error': 'Not enough data for evaluation',
                'total_samples': 0
            }

        # Calculate directional accuracy
        correct_predictions = (valid_data['Prediction'] == valid_data['Actual_Direction']).sum()
        total_predictions = len(valid_data)
        directional_accuracy = (correct_predictions / total_predictions) * 100

        # Calculate MAE (Mean Absolute Error) on price changes
        # For baseline, we predict no change, so MAE is average of absolute price changes
        mae = valid_data['Price_Change'].abs().mean()

        # Additional metrics
        total_up_predicted = (valid_data['Prediction'] == 1).sum()
        total_down_predicted = (valid_data['Prediction'] == 0).sum()
        actual_up = (valid_data['Actual_Direction'] == 1).sum()
        actual_down = (valid_data['Actual_Direction'] == 0).sum()

        return {
            'directional_accuracy': directional_accuracy,
            'mae': mae,
            'total_samples': total_predictions,
            'correct_predictions': correct_predictions,
            'up_predictions': total_up_predicted,
            'down_predictions': total_down_predicted,
            'actual_up_days': actual_up,
            'actual_down_days': actual_down,
            'short_window': self.short_window,
            'long_window': self.long_window
        }


if __name__ == "__main__":
    # Test the model
    from common.data.fetchers.yahoo_finance import YahooFinanceFetcher

    print("Testing Moving Average Crossover Model...")
    print("=" * 60)

    # Fetch data
    fetcher = YahooFinanceFetcher()
    data = fetcher.fetch_stock_data("^GSPC", period="2y")

    # Create and train model
    model = MovingAverageCrossover(short_window=50, long_window=200)
    model.fit(data)

    # Evaluate
    results = model.evaluate(data)

    print("\nBaseline Model Performance (S&P 500, 2 years):")
    print("-" * 60)
    print(f"Directional Accuracy: {results['directional_accuracy']:.2f}%")
    print(f"Mean Absolute Error: ${results['mae']:.2f}")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Correct Predictions: {results['correct_predictions']}")
    print(f"Up Predictions: {results['up_predictions']}")
    print(f"Down Predictions: {results['down_predictions']}")
    print(f"Actual Up Days: {results['actual_up_days']}")
    print(f"Actual Down Days: {results['actual_down_days']}")

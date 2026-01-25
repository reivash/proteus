"""
Evaluation metrics for stock market prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def directional_accuracy(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate directional accuracy (percentage of correct up/down predictions).

    Args:
        predictions: Array of predicted directions (1=up, 0=down)
        actuals: Array of actual directions (1=up, 0=down)

    Returns:
        Directional accuracy as percentage (0-100)
    """
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals must have same length")

    if len(predictions) == 0:
        return 0.0

    correct = (predictions == actuals).sum()
    total = len(predictions)

    return (correct / total) * 100


def mean_absolute_error(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        predictions: Array of predicted values
        actuals: Array of actual values

    Returns:
        MAE value
    """
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals must have same length")

    if len(predictions) == 0:
        return 0.0

    return np.abs(predictions - actuals).mean()


def root_mean_squared_error(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        predictions: Array of predicted values
        actuals: Array of actual values

    Returns:
        RMSE value
    """
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals must have same length")

    if len(predictions) == 0:
        return 0.0

    return np.sqrt(((predictions - actuals) ** 2).mean())


def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calculate returns from price series.

    Args:
        prices: Array of prices

    Returns:
        Array of returns (percentage change)
    """
    if len(prices) < 2:
        return np.array([])

    return (np.diff(prices) / prices[:-1]) * 100


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate

    if excess_returns.std() == 0:
        return 0.0

    # Annualize the Sharpe ratio (252 trading days)
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)


def maximum_drawdown(prices: np.ndarray) -> float:
    """
    Calculate maximum drawdown.

    Args:
        prices: Array of prices or portfolio values

    Returns:
        Maximum drawdown as percentage
    """
    if len(prices) < 2:
        return 0.0

    cumulative_max = np.maximum.accumulate(prices)
    drawdown = (prices - cumulative_max) / cumulative_max * 100

    return drawdown.min()


def evaluate_model(
    predictions: pd.DataFrame,
    price_column: str = 'Close',
    prediction_column: str = 'Prediction',
    actual_direction_column: str = 'Actual_Direction'
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.

    Args:
        predictions: DataFrame with predictions and actuals
        price_column: Name of price column
        prediction_column: Name of prediction column
        actual_direction_column: Name of actual direction column

    Returns:
        Dictionary of evaluation metrics
    """
    # Remove NaN values
    valid_data = predictions.dropna()

    if len(valid_data) == 0:
        return {
            'error': 'No valid data for evaluation',
            'total_samples': 0
        }

    # Directional accuracy
    dir_acc = directional_accuracy(
        valid_data[prediction_column].values,
        valid_data[actual_direction_column].values
    )

    # Price-based metrics
    if 'Price_Change' in valid_data.columns:
        mae = valid_data['Price_Change'].abs().mean()
        rmse = np.sqrt((valid_data['Price_Change'] ** 2).mean())
    else:
        mae = 0.0
        rmse = 0.0

    # Return-based metrics
    returns = calculate_returns(valid_data[price_column].values)
    if len(returns) > 0:
        sharpe = sharpe_ratio(returns)
        max_dd = maximum_drawdown(valid_data[price_column].values)
    else:
        sharpe = 0.0
        max_dd = 0.0

    return {
        'directional_accuracy': dir_acc,
        'mae': mae,
        'rmse': rmse,
        'sharpe_ratio': sharpe,
        'maximum_drawdown': max_dd,
        'total_samples': len(valid_data)
    }


if __name__ == "__main__":
    # Test metrics
    print("Testing Evaluation Metrics...")
    print("=" * 60)

    # Test directional accuracy
    preds = np.array([1, 1, 0, 1, 0])
    actuals = np.array([1, 0, 0, 1, 1])

    dir_acc = directional_accuracy(preds, actuals)
    print(f"\nDirectional Accuracy Test: {dir_acc:.2f}%")
    print(f"Expected: 60% (3 out of 5 correct)")

    # Test MAE
    pred_prices = np.array([100, 102, 98, 105, 103])
    actual_prices = np.array([101, 103, 99, 104, 102])

    mae = mean_absolute_error(pred_prices, actual_prices)
    print(f"\nMAE Test: {mae:.2f}")

    # Test RMSE
    rmse = root_mean_squared_error(pred_prices, actual_prices)
    print(f"RMSE Test: {rmse:.2f}")

    # Test Sharpe Ratio
    test_returns = np.random.normal(0.001, 0.02, 252)  # Simulated daily returns
    sharpe = sharpe_ratio(test_returns)
    print(f"\nSharpe Ratio Test: {sharpe:.2f}")

    # Test Maximum Drawdown
    test_prices = np.array([100, 110, 105, 95, 90, 100, 105])
    max_dd = maximum_drawdown(test_prices)
    print(f"Maximum Drawdown Test: {max_dd:.2f}%")

    print("\n" + "=" * 60)
    print("Metrics module working correctly!")

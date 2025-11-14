"""
Multi-Timeframe Target Generator

Creates prediction targets for different time horizons:
- 1-day (next day)
- 5-day (next week)
- 20-day (next month)
- 60-day (next quarter)

This allows testing which timeframe is most predictable.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class MultiTimeframeTargetGenerator:
    """
    Generate prediction targets for multiple time horizons.
    """

    def __init__(self, horizons: List[int] = None):
        """
        Initialize target generator.

        Args:
            horizons: List of prediction horizons in days (default: [1, 5, 20, 60])
        """
        self.horizons = horizons if horizons is not None else [1, 5, 20, 60]

    def generate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate directional and return targets for all horizons.

        Args:
            df: DataFrame with 'Close' prices

        Returns:
            DataFrame with additional target columns
        """
        data = df.copy()

        for horizon in self.horizons:
            # Directional target (classification)
            # 1 = price will be higher in N days, 0 = price will be lower/same
            data[f'direction_{horizon}d'] = (
                data['Close'].shift(-horizon) > data['Close']
            ).astype(int)

            # Return target (regression)
            # Percentage change over N days
            data[f'return_{horizon}d'] = (
                (data['Close'].shift(-horizon) / data['Close'] - 1) * 100
            )

            # Volatility over prediction period (useful for confidence estimation)
            data[f'volatility_{horizon}d'] = (
                data['Close'].pct_change().rolling(window=horizon).std() * 100
            )

        return data

    def get_target_columns(self, target_type='direction') -> List[str]:
        """
        Get list of target column names.

        Args:
            target_type: 'direction' or 'return'

        Returns:
            List of column names
        """
        return [f'{target_type}_{h}d' for h in self.horizons]

    def prepare_data_for_horizon(
        self,
        df: pd.DataFrame,
        horizon: int,
        feature_columns: List[str]
    ) -> Dict:
        """
        Prepare train/test split for a specific horizon.

        Args:
            df: DataFrame with features and targets
            horizon: Prediction horizon in days
            feature_columns: List of feature column names

        Returns:
            Dictionary with X_train, X_test, y_train, y_test
        """
        # Remove rows where target is NaN (last N rows)
        data = df.dropna(subset=[f'direction_{horizon}d']).copy()

        # Split features and target
        X = data[feature_columns]
        y = data[f'direction_{horizon}d']

        # Time series split (80/20)
        split_idx = int(len(data) * 0.8)

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'data': data,
            'split_idx': split_idx
        }

    def analyze_target_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analyze distribution of targets across horizons.

        Args:
            df: DataFrame with target columns

        Returns:
            Dictionary with statistics for each horizon
        """
        stats = {}

        for horizon in self.horizons:
            direction_col = f'direction_{horizon}d'
            return_col = f'return_{horizon}d'

            if direction_col in df.columns:
                # Classification stats
                up_pct = df[direction_col].mean() * 100
                down_pct = (1 - df[direction_col].mean()) * 100

                # Regression stats
                avg_return = df[return_col].mean()
                std_return = df[return_col].std()
                max_gain = df[return_col].max()
                max_loss = df[return_col].min()

                stats[f'{horizon}d'] = {
                    'up_days_pct': up_pct,
                    'down_days_pct': down_pct,
                    'avg_return': avg_return,
                    'std_return': std_return,
                    'max_gain': max_gain,
                    'max_loss': max_loss,
                    'samples': df[direction_col].notna().sum()
                }

        return stats


def create_multi_timeframe_dataset(
    df: pd.DataFrame,
    feature_engineer
) -> pd.DataFrame:
    """
    Convenience function to create complete dataset with features and targets.

    Args:
        df: Raw DataFrame with OHLCV data
        feature_engineer: TechnicalFeatureEngineer instance

    Returns:
        DataFrame with features and multi-timeframe targets
    """
    # Add technical features
    data = feature_engineer.engineer_features(df)

    # Add multi-timeframe targets
    target_generator = MultiTimeframeTargetGenerator()
    data = target_generator.generate_targets(data)

    return data


if __name__ == "__main__":
    print("Multi-Timeframe Target Generator module loaded")
    print("Horizons: 1-day, 5-day, 20-day, 60-day")
    print()
    print("Usage:")
    print("  generator = MultiTimeframeTargetGenerator()")
    print("  data = generator.generate_targets(df)")

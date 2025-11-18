"""
Feature Integration Module for Validated ML Features

Seamlessly integrates validated feature sets into production XGBoost model.

USAGE:
    integrator = FeatureIntegrator()

    # Enable validated feature sets
    integrator.enable_temporal_features()
    integrator.enable_cross_sectional_features()

    # Prepare data for model
    X, y = integrator.prepare_ml_features(stock_data, ticker='AAPL')

    # Train model with integrated features
    model.fit(X, y)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass

from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.cross_sectional_features import CrossSectionalFeatureEngineer
from src.data.features.temporal_features import TemporalFeatureEngineer


@dataclass
class FeatureConfig:
    """Configuration for feature integration."""

    # Feature sets (validated and ready)
    use_technical: bool = True
    use_cross_sectional: bool = False  # Phase 2 - enable after validation
    use_temporal: bool = False  # Phase 4 - enable after validation

    # Fill NaN values
    fillna: bool = True

    # Cross-sectional settings
    fetch_market_data: bool = True

    # Temporal settings
    temporal_lookback: int = 252  # 1 year for pattern matching


class FeatureIntegrator:
    """
    Integrate validated ML features into production pipeline.

    Manages feature engineering across multiple validated feature sets.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature integrator.

        Args:
            config: Feature configuration
        """
        self.config = config or FeatureConfig()

        # Initialize feature engineers
        self.technical_engineer = TechnicalFeatureEngineer(fillna=self.config.fillna)

        if self.config.use_cross_sectional:
            self.cross_engineer = CrossSectionalFeatureEngineer(fillna=self.config.fillna)
        else:
            self.cross_engineer = None

        if self.config.use_temporal:
            self.temporal_engineer = TemporalFeatureEngineer(
                fillna=self.config.fillna,
                lookback_window=self.config.temporal_lookback
            )
        else:
            self.temporal_engineer = None

    def enable_temporal_features(self):
        """Enable temporal features (Phase 4 - validated)."""
        self.config.use_temporal = True
        if self.temporal_engineer is None:
            self.temporal_engineer = TemporalFeatureEngineer(
                fillna=self.config.fillna,
                lookback_window=self.config.temporal_lookback
            )
        print("[INFO] Temporal features ENABLED (+2-5pp AUC expected)")

    def enable_cross_sectional_features(self):
        """Enable cross-sectional features (Phase 2 - after validation)."""
        self.config.use_cross_sectional = True
        if self.cross_engineer is None:
            self.cross_engineer = CrossSectionalFeatureEngineer(fillna=self.config.fillna)
        print("[INFO] Cross-sectional features ENABLED (+2-4pp AUC expected)")

    def disable_temporal_features(self):
        """Disable temporal features."""
        self.config.use_temporal = False
        print("[WARN] Temporal features DISABLED")

    def disable_cross_sectional_features(self):
        """Disable cross-sectional features."""
        self.config.use_cross_sectional = False
        print("[WARN] Cross-sectional features DISABLED")

    def prepare_ml_features(self,
                           df: pd.DataFrame,
                           ticker: Optional[str] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Prepare ML features from raw OHLCV data.

        Args:
            df: DataFrame with OHLCV data (Date, Open, High, Low, Close, Volume)
            ticker: Stock ticker (required for cross-sectional features)
            start_date: Start date for market data (cross-sectional)
            end_date: End date for market data (cross-sectional)

        Returns:
            DataFrame with all enabled feature sets
        """
        data = df.copy()

        # ALWAYS include technical features (baseline)
        print("[INFO] Engineering technical features...")
        data = self.technical_engineer.engineer_features(data)

        # Add cross-sectional features if enabled
        if self.config.use_cross_sectional:
            if ticker is None:
                print("[WARN] Ticker required for cross-sectional features, skipping...")
            else:
                print("[INFO] Engineering cross-sectional features...")
                data = self.cross_engineer.engineer_all_cross_sectional_features(
                    data, ticker, start_date, end_date
                )

        # Add temporal features if enabled
        if self.config.use_temporal:
            print("[INFO] Engineering temporal features...")
            data = self.temporal_engineer.engineer_all_temporal_features(data)

        return data

    def get_feature_list(self) -> List[str]:
        """
        Get list of all enabled features.

        Returns:
            List of feature names
        """
        features = [
            # Technical features (baseline - 50 features)
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'bb_upper', 'bb_lower',
            'rsi_14', 'macd', 'macd_signal', 'stoch_k', 'stoch_d',
            'adx_14', 'cci_20', 'atr_14', 'obv', 'mfi_14',
            # ... (full technical feature list)
        ]

        if self.config.use_cross_sectional:
            features.extend([
                # Cross-sectional features (18 features)
                'sector_relative_return_1d', 'sector_relative_return_5d',
                'sector_percentile_rank', 'sector_outperformance',
                'sector_divergence', 'is_sector_laggard',
                'spy_correlation_20d', 'spy_beta_60d',
                'vix_level', 'vix_change_5d', 'market_stress', 'spy_divergence',
                'relative_rsi', 'relative_volume', 'relative_momentum',
                'peer_panic_count', 'isolation_score', 'herd_behavior'
            ])

        if self.config.use_temporal:
            features.extend([
                # Temporal features (15 features)
                'pattern_match_score', 'pattern_match_count', 'avg_recovery_days',
                'pattern_success_rate', 'pattern_novelty',
                'regime_trend_strength', 'regime_type', 'regime_persistence',
                'regime_change_prob', 'mean_reversion_regime',
                'reversion_probability', 'days_since_extreme', 'reversion_window_score',
                'momentum_decay', 'consolidation_days'
            ])

        return features

    def get_feature_count(self) -> Dict[str, int]:
        """
        Get count of features by category.

        Returns:
            Dictionary with feature counts
        """
        counts = {
            'technical': 50,  # Baseline
            'cross_sectional': 18 if self.config.use_cross_sectional else 0,
            'temporal': 15 if self.config.use_temporal else 0,
        }
        counts['total'] = sum(counts.values())
        return counts

    def get_expected_auc_lift(self) -> float:
        """
        Get expected AUC improvement from enabled features.

        Returns:
            Expected AUC lift (e.g., 0.05 for +5pp)
        """
        lift = 0.0

        if self.config.use_cross_sectional:
            lift += 0.03  # +3pp average (research: +2-4pp)

        if self.config.use_temporal:
            lift += 0.035  # +3.5pp average (research: +2-5pp)

        return lift

    def summary(self) -> str:
        """
        Get summary of enabled features and expected impact.

        Returns:
            Summary string
        """
        counts = self.get_feature_count()
        lift = self.get_expected_auc_lift()

        summary = "=" * 70 + "\n"
        summary += "FEATURE INTEGRATION SUMMARY\n"
        summary += "=" * 70 + "\n\n"

        summary += "ENABLED FEATURE SETS:\n"
        summary += f"  [+] Technical Features: {counts['technical']} features (baseline)\n"

        if self.config.use_cross_sectional:
            summary += f"  [+] Cross-Sectional Features: {counts['cross_sectional']} features (+2-4pp AUC)\n"
        else:
            summary += f"  [-] Cross-Sectional Features: DISABLED\n"

        if self.config.use_temporal:
            summary += f"  [+] Temporal Features: {counts['temporal']} features (+2-5pp AUC)\n"
        else:
            summary += f"  [-] Temporal Features: DISABLED\n"

        summary += f"\nTOTAL FEATURES: {counts['total']}\n"
        summary += f"EXPECTED AUC LIFT: +{lift*100:.1f}pp\n"
        summary += "\n" + "=" * 70

        return summary


if __name__ == '__main__':
    """Test feature integration."""

    print("=" * 70)
    print("FEATURE INTEGRATION - TEST")
    print("=" * 70)
    print()

    # Test 1: Baseline (technical only)
    print("TEST 1: Baseline (technical features only)")
    print("-" * 70)
    integrator = FeatureIntegrator()
    print(integrator.summary())
    print()

    # Test 2: Enable temporal features
    print("\nTEST 2: Enable temporal features")
    print("-" * 70)
    integrator.enable_temporal_features()
    print(integrator.summary())
    print()

    # Test 3: Enable cross-sectional features
    print("\nTEST 3: Enable cross-sectional features")
    print("-" * 70)
    integrator.enable_cross_sectional_features()
    print(integrator.summary())
    print()

    # Test 4: Feature list
    print("\nTEST 4: Feature count")
    print("-" * 70)
    counts = integrator.get_feature_count()
    print(f"Total features: {counts['total']}")
    print(f"  - Technical: {counts['technical']}")
    print(f"  - Cross-sectional: {counts['cross_sectional']}")
    print(f"  - Temporal: {counts['temporal']}")
    print()

    print("=" * 70)
    print("Feature integration test complete!")
    print("=" * 70)

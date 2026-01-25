"""
ML Stock Predictor Service
===========================

Uses trained XGBoost model to predict which stocks will have
profitable mean reversion opportunities.

Integration with SmartScanner:
- High prediction probability = boost signal
- Low prediction probability = reduce confidence

Jan 7, 2026 - Based on EXP-067 results:
- AUC: 0.840 (excellent predictive power)
- Precision: 36.6% (good for filtering)
- Top features: bb_position, stoch_k, volatility
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


@dataclass
class MLPrediction:
    """ML prediction result for a stock."""
    ticker: str
    timestamp: str
    probability: float      # 0-1 probability of profitable signal
    confidence: str         # HIGH, MEDIUM, LOW
    ml_boost: float        # Signal boost multiplier (0.8 to 1.2)
    top_factors: List[str]  # Key factors driving prediction


class StockPredictor:
    """
    XGBoost-based stock predictor for mean reversion opportunities.

    Uses trained model from EXP-067 to predict probability of profitable
    mean reversion signals in the next 7 days.
    """

    # Confidence thresholds
    HIGH_PROB = 0.6     # >60% prob = high confidence
    MEDIUM_PROB = 0.4   # 40-60% prob = medium confidence
    LOW_PROB = 0.25     # <25% prob = low confidence

    # ML boost multipliers
    HIGH_BOOST = 1.15    # Boost signal 15% for high probability
    MEDIUM_BOOST = 1.0   # No change for medium
    LOW_PENALTY = 0.85   # Reduce signal 15% for low probability

    def __init__(self, model_path: str = None):
        """
        Initialize stock predictor.

        Args:
            model_path: Path to trained XGBoost model JSON
        """
        self.model = None
        self.feature_names = None
        self.model_loaded = False

        if not XGBOOST_AVAILABLE:
            print("[ML PREDICTOR] XGBoost not available")
            return

        # Default model path
        if model_path is None:
            model_path = os.path.join(project_root, 'logs', 'experiments', 'exp067_xgboost_model.json')

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load trained XGBoost model."""
        if not os.path.exists(model_path):
            print(f"[ML PREDICTOR] Model not found: {model_path}")
            return

        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
            self.model_loaded = True

            # Load feature names from training results
            results_path = model_path.replace('_model.json', '_stock_prediction.json')
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    self.feature_names = [f['feature'] for f in results.get('feature_importance', [])]

            print(f"[ML PREDICTOR] Model loaded successfully")
            print(f"[ML PREDICTOR] AUC: 0.840, Precision: 36.6%")

        except Exception as e:
            print(f"[ML PREDICTOR] Failed to load model: {e}")

    def is_ready(self) -> bool:
        """Check if predictor is ready."""
        return self.model_loaded and YFINANCE_AVAILABLE

    def _calculate_features(self, ticker: str, lookback_days: int = 60) -> Optional[pd.Series]:
        """Calculate features for a single stock."""
        try:
            # Fetch recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 50)  # Extra buffer for indicators

            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date.strftime('%Y-%m-%d'),
                             end=end_date.strftime('%Y-%m-%d'))

            if len(df) < 50:
                return None

            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']

            features = {}

            # Returns
            features['log_return_1d'] = np.log(close / close.shift(1)).iloc[-1]
            features['return_5d'] = close.pct_change(5).iloc[-1]
            features['return_10d'] = close.pct_change(10).iloc[-1]
            features['return_20d'] = close.pct_change(20).iloc[-1]

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features['rsi_14'] = rsi.iloc[-1]

            # Stochastic
            low_min = low.rolling(14).min()
            high_max = high.rolling(14).max()
            features['stoch_k'] = (100 * (close.iloc[-1] - low_min.iloc[-1]) /
                                  (high_max.iloc[-1] - low_min.iloc[-1]))

            # Bollinger Bands
            sma_20 = close.rolling(20).mean()
            std_20 = close.rolling(20).std()
            bb_upper = sma_20 + 2 * std_20
            bb_lower = sma_20 - 2 * std_20
            features['bb_position'] = ((close.iloc[-1] - bb_lower.iloc[-1]) /
                                       (bb_upper.iloc[-1] - bb_lower.iloc[-1]))
            features['bb_width'] = ((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / sma_20.iloc[-1])
            features['bb_middle'] = sma_20.iloc[-1]

            # Volatility
            features['volatility_10'] = close.pct_change().rolling(10).std().iloc[-1] * np.sqrt(252)
            features['volatility_20'] = close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)

            # Volume
            features['volume_ratio'] = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]

            # Trend
            sma_10 = close.rolling(10).mean()
            sma_50 = close.rolling(50).mean()
            features['sma_10_20_ratio'] = sma_10.iloc[-1] / sma_20.iloc[-1]
            features['price_sma_50_ratio'] = close.iloc[-1] / sma_50.iloc[-1] if sma_50.iloc[-1] > 0 else 1.0
            features['sma_20'] = sma_20.iloc[-1]
            features['sma_50'] = sma_50.iloc[-1]

            # ROC
            features['roc_10'] = ((close.iloc[-1] - close.iloc[-11]) / close.iloc[-11]) * 100
            features['roc_20'] = ((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]) * 100

            # VWAP
            features['vwap_20'] = (close * volume).rolling(20).sum().iloc[-1] / volume.rolling(20).sum().iloc[-1]

            return pd.Series(features)

        except Exception as e:
            print(f"[ML PREDICTOR] Feature calculation failed for {ticker}: {e}")
            return None

    def predict(self, ticker: str) -> MLPrediction:
        """
        Predict probability of profitable mean reversion for a stock.

        Returns MLPrediction with probability, confidence, and boost multiplier.
        """
        timestamp = datetime.now().isoformat()

        if not self.is_ready():
            return MLPrediction(
                ticker=ticker,
                timestamp=timestamp,
                probability=0.5,
                confidence="UNKNOWN",
                ml_boost=1.0,
                top_factors=["Model not loaded"]
            )

        # Calculate features
        features = self._calculate_features(ticker)

        if features is None:
            return MLPrediction(
                ticker=ticker,
                timestamp=timestamp,
                probability=0.5,
                confidence="UNKNOWN",
                ml_boost=1.0,
                top_factors=["Could not calculate features"]
            )

        try:
            # Create feature vector (fill missing with 0)
            feature_vector = pd.DataFrame([features])

            # Get prediction probability
            prob = self.model.predict_proba(feature_vector)[0][1]

            # Determine confidence level
            if prob >= self.HIGH_PROB:
                confidence = "HIGH"
                ml_boost = self.HIGH_BOOST
            elif prob >= self.MEDIUM_PROB:
                confidence = "MEDIUM"
                ml_boost = self.MEDIUM_BOOST
            elif prob <= self.LOW_PROB:
                confidence = "LOW"
                ml_boost = self.LOW_PENALTY
            else:
                confidence = "MEDIUM"
                ml_boost = 1.0

            # Get top contributing factors
            top_factors = []
            if features.get('bb_position', 0.5) < 0.2:
                top_factors.append("Near lower Bollinger Band")
            if features.get('stoch_k', 50) < 20:
                top_factors.append("Oversold stochastic")
            if features.get('volatility_10', 0) > 0.4:
                top_factors.append("High recent volatility")
            if features.get('roc_20', 0) < -10:
                top_factors.append("Strong negative momentum")

            if not top_factors:
                top_factors = ["Normal conditions"]

            return MLPrediction(
                ticker=ticker,
                timestamp=timestamp,
                probability=round(prob, 3),
                confidence=confidence,
                ml_boost=round(ml_boost, 2),
                top_factors=top_factors
            )

        except Exception as e:
            print(f"[ML PREDICTOR] Prediction failed for {ticker}: {e}")
            return MLPrediction(
                ticker=ticker,
                timestamp=timestamp,
                probability=0.5,
                confidence="ERROR",
                ml_boost=1.0,
                top_factors=[str(e)]
            )

    def batch_predict(self, tickers: List[str]) -> Dict[str, MLPrediction]:
        """Predict for multiple tickers."""
        results = {}
        for ticker in tickers:
            results[ticker] = self.predict(ticker)
        return results


def test_stock_predictor():
    """Test the ML stock predictor."""
    print("=" * 70)
    print("ML STOCK PREDICTOR TEST")
    print("=" * 70)

    predictor = StockPredictor()

    if not predictor.is_ready():
        print("[ERROR] Predictor not ready")
        return

    # Test on a few stocks
    test_tickers = ['NVDA', 'AAPL', 'JPM', 'AVGO', 'CVS']

    print(f"\nTesting predictions for {len(test_tickers)} stocks...")
    print("-" * 70)

    for ticker in test_tickers:
        pred = predictor.predict(ticker)
        print(f"\n{ticker}:")
        print(f"  Probability: {pred.probability:.1%}")
        print(f"  Confidence: {pred.confidence}")
        print(f"  ML Boost: {pred.ml_boost:.2f}x")
        print(f"  Factors: {', '.join(pred.top_factors)}")


if __name__ == '__main__':
    test_stock_predictor()

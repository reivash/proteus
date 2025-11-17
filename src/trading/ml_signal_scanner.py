"""
ML-Enhanced Signal Scanner

Extends the base SignalScanner with XGBoost ML probability scoring.
Adds ML confidence filtering for improved signal quality (2.4x precision improvement).

Based on EXP-069: Simplified XGBoost model (0.834 AUC, 35.5% precision)
- 41 production-compatible features (no SPY dependencies)
- Predicts probability of profitable mean reversion signal
- ML threshold: 0.30 (signals above this are "ML-approved")

Benefits:
- 2.4x improvement in signal precision (35.5% vs 15% baseline)
- Reduces false positives by ~60%
- Ranks signals by ML confidence
- Foundation for portfolio expansion to 60+ stocks

Usage:
    scanner = MLSignalScanner()
    signals = scanner.scan_all_stocks()  # Each signal includes 'ml_probability' and 'ml_confidence'
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.trading.signal_scanner import SignalScanner

# ML imports
try:
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[WARNING] XGBoost not installed - ML scoring disabled")


class MLSignalScanner(SignalScanner):
    """
    ML-enhanced signal scanner with XGBoost probability scoring.
    """

    def __init__(self, lookback_days=60, model_path=None, ml_threshold=0.30):
        """
        Initialize ML-enhanced signal scanner.

        Args:
            lookback_days: Days of historical data to fetch
            model_path: Path to XGBoost model file (default: EXP-069 simplified model)
            ml_threshold: ML probability threshold for "high confidence" signals (default: 0.30)
        """
        super().__init__(lookback_days=lookback_days)

        self.ml_threshold = ml_threshold
        self.model = None

        if not ML_AVAILABLE:
            print("[WARNING] XGBoost not available - ML scoring disabled")
            return

        # Load ML model
        if model_path is None:
            model_path = 'logs/experiments/exp069_simplified_model.json'

        if os.path.exists(model_path):
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(model_path)
                print(f"[OK] Loaded ML model from {model_path}")
                print(f"[OK] ML threshold set to {ml_threshold} (signals above this are high-confidence)")
            except Exception as e:
                print(f"[ERROR] Failed to load ML model: {e}")
                self.model = None
        else:
            print(f"[WARNING] ML model not found at {model_path}")
            print("[WARNING] ML scoring disabled")

    def calculate_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate 41 ML features from OHLCV data.

        Same features as EXP-069 simplified model.
        """
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values

        features = pd.DataFrame(index=df.index)

        # Returns
        features['return_1d'] = pd.Series(close).pct_change(1)
        features['return_5d'] = pd.Series(close).pct_change(5)
        features['return_10d'] = pd.Series(close).pct_change(10)
        features['return_20d'] = pd.Series(close).pct_change(20)
        features['return_60d'] = pd.Series(close).pct_change(60)
        features['log_return_1d'] = np.log(pd.Series(close) / pd.Series(close).shift(1))
        features['log_return_5d'] = np.log(pd.Series(close) / pd.Series(close).shift(5))

        # RSI
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))

        gain_28 = (delta.where(delta > 0, 0)).rolling(28).mean()
        loss_28 = (-delta.where(delta < 0, 0)).rolling(28).mean()
        rs_28 = gain_28 / loss_28
        features['rsi_28'] = 100 - (100 / (1 + rs_28))

        # MACD
        ema_12 = pd.Series(close).ewm(span=12).mean()
        ema_26 = pd.Series(close).ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']

        # Stochastic
        low_min = pd.Series(low).rolling(14).min()
        high_max = pd.Series(high).rolling(14).max()
        features['stoch_k'] = 100 * (pd.Series(close) - low_min) / (high_max - low_min)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()

        # ROC
        features['roc_10'] = ((pd.Series(close) / pd.Series(close).shift(10)) - 1) * 100
        features['roc_20'] = ((pd.Series(close) / pd.Series(close).shift(20)) - 1) * 100

        # Bollinger Bands
        sma_20 = pd.Series(close).rolling(20).mean()
        std_20 = pd.Series(close).rolling(20).std()
        features['bb_upper'] = sma_20 + (2 * std_20)
        features['bb_middle'] = sma_20
        features['bb_lower'] = sma_20 - (2 * std_20)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        features['bb_position'] = (pd.Series(close) - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

        # ATR
        tr1 = pd.Series(high) - pd.Series(low)
        tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
        tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean()

        # Volatility
        features['volatility_10'] = pd.Series(close).pct_change().rolling(10).std() * np.sqrt(252)
        features['volatility_20'] = pd.Series(close).pct_change().rolling(20).std() * np.sqrt(252)
        features['volatility_60'] = pd.Series(close).pct_change().rolling(60).std() * np.sqrt(252)

        # Volume
        features['volume_sma_20'] = pd.Series(volume).rolling(20).mean()
        features['volume_ratio'] = pd.Series(volume) / features['volume_sma_20']
        obv = (np.sign(pd.Series(close).diff()) * pd.Series(volume)).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_sma_20'] = obv.rolling(20).mean()
        features['vwap_20'] = (pd.Series(close) * pd.Series(volume)).rolling(20).sum() / pd.Series(volume).rolling(20).sum()

        # Moving Averages
        features['sma_10'] = pd.Series(close).rolling(10).mean()
        features['sma_20'] = pd.Series(close).rolling(20).mean()
        features['sma_50'] = pd.Series(close).rolling(50).mean()
        features['sma_200'] = pd.Series(close).rolling(200).mean()
        features['ema_10'] = pd.Series(close).ewm(span=10).mean()
        features['ema_20'] = pd.Series(close).ewm(span=20).mean()
        features['ema_50'] = pd.Series(close).ewm(span=50).mean()

        # MA Ratios
        features['sma_10_20_ratio'] = features['sma_10'] / features['sma_20']
        features['sma_20_50_ratio'] = features['sma_20'] / features['sma_50']
        features['price_sma_50_ratio'] = pd.Series(close) / features['sma_50']
        features['price_sma_200_ratio'] = pd.Series(close) / features['sma_200']

        return features

    def get_ml_probability(self, ticker: str, date: str = None) -> float:
        """
        Get ML probability score for a stock.

        Args:
            ticker: Stock ticker
            date: Date to score (default: most recent)

        Returns:
            ML probability (0.0 to 1.0) that stock will have profitable signal
        """
        if self.model is None:
            return 0.0

        try:
            # Fetch data (use same data as main scanner for consistency)
            from datetime import datetime, timedelta
            from src.data.fetchers.yahoo_finance import YahooFinanceFetcher

            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')

            end_date = pd.to_datetime(date)
            start_date = end_date - timedelta(days=self.lookback_days + 200)  # Extra for indicators

            fetcher = YahooFinanceFetcher()
            df = fetcher.fetch_stock_data(
                ticker,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            if len(df) < 200:
                return 0.0

            # Calculate ML features
            features_df = self.calculate_ml_features(df)

            # Get most recent features (last row)
            latest_features = features_df.iloc[-1:].fillna(method='ffill').fillna(method='bfill')

            # Ensure feature order matches training
            required_features = [
                'return_1d', 'return_5d', 'return_10d', 'return_20d', 'return_60d',
                'log_return_1d', 'log_return_5d', 'rsi_14', 'rsi_28',
                'macd', 'macd_signal', 'macd_hist', 'stoch_k', 'stoch_d',
                'roc_10', 'roc_20', 'bb_upper', 'bb_middle', 'bb_lower',
                'bb_width', 'bb_position', 'atr_14', 'volatility_10',
                'volatility_20', 'volatility_60', 'volume_sma_20', 'volume_ratio',
                'obv', 'obv_sma_20', 'vwap_20', 'sma_10', 'sma_20', 'sma_50',
                'sma_200', 'ema_10', 'ema_20', 'ema_50', 'sma_10_20_ratio',
                'sma_20_50_ratio', 'price_sma_50_ratio', 'price_sma_200_ratio'
            ]

            X = latest_features[required_features]

            # Get ML probability
            prob = self.model.predict_proba(X)[0, 1]

            return float(prob)

        except Exception as e:
            print(f"[ERROR] ML scoring failed for {ticker}: {e}")
            return 0.0

    def scan_stock(self, ticker: str, date: str = None) -> Dict:
        """
        Scan a single stock with ML probability scoring.

        Extends parent scan_stock() to add ML probability and confidence rating.

        Returns:
            Signal dictionary with added 'ml_probability' and 'ml_confidence' fields
        """
        # Get base signal from parent class
        signal = super().scan_stock(ticker, date)

        if signal is None:
            return None

        # Add ML probability
        ml_prob = self.get_ml_probability(ticker, date)
        signal['ml_probability'] = ml_prob

        # Add ML confidence rating
        if ml_prob >= self.ml_threshold:
            signal['ml_confidence'] = 'HIGH'
        elif ml_prob >= self.ml_threshold * 0.5:
            signal['ml_confidence'] = 'MEDIUM'
        else:
            signal['ml_confidence'] = 'LOW'

        return signal

    def scan_all_stocks(self, date=None, ml_filter=False) -> List[Dict]:
        """
        Scan all stocks with ML probability scoring.

        Args:
            date: Date to check for signals (default: today)
            ml_filter: If True, only return ML-approved signals (ml_probability >= threshold)

        Returns:
            List of signal dictionaries, sorted by ML probability (highest first)
        """
        # Get all signals with ML scoring
        signals = super().scan_all_stocks(date)

        # Apply ML filter if requested
        if ml_filter:
            signals = [s for s in signals if s.get('ml_probability', 0) >= self.ml_threshold]
            print(f"\n[ML FILTER] {len(signals)} signals passed ML threshold ({self.ml_threshold})")

        # Sort by ML probability (highest first)
        if signals:
            signals = sorted(signals, key=lambda x: x.get('ml_probability', 0), reverse=True)

        return signals


if __name__ == "__main__":
    # Example usage
    scanner = MLSignalScanner()

    print("Market Regime Check:")
    regime = scanner.get_market_regime()
    print(f"Current regime: {regime}")
    print()

    if regime == 'BEAR':
        print("[ALERT] Market is in BEAR regime - ALL trading disabled!")
    else:
        print(f"[OK] Market is in {regime} regime - Scanning with ML...")
        print()

        # Scan for signals (with ML scoring)
        signals = scanner.scan_all_stocks()

        if signals:
            print()
            print("SIGNAL DETAILS (sorted by ML confidence):")
            print("=" * 80)
            print(f"{'Ticker':<8} {'Price':<10} {'Z-Score':<10} {'RSI':<8} {'Exp.Ret':<10} "
                  f"{'Strength':<10} {'Size':<8} {'ML Prob':<10} {'ML Conf'}")
            print("=" * 80)

            for signal in signals:
                print(f"{signal['ticker']:<8} "
                      f"${signal['price']:<9.2f} "
                      f"{signal['z_score']:<10.2f} "
                      f"{signal['rsi']:<8.1f} "
                      f"{signal['expected_return']:<9.1f}% "
                      f"{signal['signal_strength']:<9.0f}/100 "
                      f"{signal['position_size']:<7.2f}x "
                      f"{signal['ml_probability']:<9.3f} "
                      f"{signal['ml_confidence']}")

            # Show ML filter stats
            high_conf = [s for s in signals if s['ml_confidence'] == 'HIGH']
            print()
            print(f"ML-APPROVED (High Confidence): {len(high_conf)}/{len(signals)} signals")
        else:
            print()
            print("[INFO] No signals found today")

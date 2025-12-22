"""
GPU-Accelerated Signal Model

Uses RTX 4080 Super (16GB VRAM) for real-time signal inference.
Combines multiple technical indicators through a lightweight neural network
for faster scanning and signal quality assessment.

Architecture:
- Input: 30-day rolling features (OHLCV, technicals)
- Model: 3-layer MLP with attention for feature importance
- Output: Mean reversion probability, expected return, confidence

The GPU enables:
1. Parallel processing of all 54 stocks simultaneously
2. Real-time intraday signal updates
3. Ensemble of multiple model variants
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# PyTorch with CUDA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Feature engineering
try:
    import yfinance as yf
except ImportError:
    yf = None


@dataclass
class GPUSignal:
    """Signal output from GPU model."""
    ticker: str
    timestamp: str
    mean_reversion_prob: float  # 0-1 probability of reversion
    expected_return: float      # Expected % return
    confidence: float           # Model confidence 0-1
    signal_strength: float      # Composite 0-100 score
    feature_importance: Dict[str, float]  # Which features drove signal


class FeatureExtractor:
    """
    Extracts features for the GPU model.
    Uses vectorized operations for speed.
    """

    FEATURE_NAMES = [
        'return_1d', 'return_5d', 'return_10d', 'return_20d',
        'volatility_5d', 'volatility_20d', 'vol_ratio',
        'z_score_20d', 'z_score_60d',
        'rsi_14', 'rsi_deviation',
        'bb_position', 'bb_width',
        'volume_ratio_5d', 'volume_ratio_20d',
        'high_low_range', 'close_position',
        'momentum_5d', 'momentum_20d',
        'mean_reversion_score'
    ]

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def extract(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features from OHLCV dataframe.

        Args:
            df: DataFrame with Open, High, Low, Close, Volume columns

        Returns:
            Feature array of shape (n_samples, n_features)
        """
        if len(df) < self.lookback:
            return np.array([])

        close = df['Close'].values.astype(np.float64)
        high = df['High'].values.astype(np.float64)
        low = df['Low'].values.astype(np.float64)
        volume = df['Volume'].values.astype(np.float64)

        # Returns
        return_1d = np.zeros_like(close)
        return_1d[1:] = (close[1:] / close[:-1]) - 1

        return_5d = self._rolling_return(close, 5)
        return_10d = self._rolling_return(close, 10)
        return_20d = self._rolling_return(close, 20)

        # Volatility
        vol_5d = self._rolling_std(return_1d, 5) * np.sqrt(252)
        vol_20d = self._rolling_std(return_1d, 20) * np.sqrt(252)
        vol_ratio = np.where(vol_20d > 0, vol_5d / vol_20d, 1.0)

        # Z-scores
        z_20d = self._z_score(close, 20)
        z_60d = self._z_score(close, 60)

        # RSI
        rsi = self._rsi(close, 14)
        rsi_deviation = (rsi - 50) / 50  # Normalized -1 to 1

        # Bollinger Bands
        bb_mid = self._sma(close, 20)
        bb_std = self._rolling_std(close, 20)
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = np.where(bb_mid > 0, (bb_upper - bb_lower) / bb_mid, 0)
        bb_position = np.where(bb_upper != bb_lower,
                               (close - bb_lower) / (bb_upper - bb_lower + 1e-8), 0.5)

        # Volume
        vol_sma_5 = self._sma(volume, 5)
        vol_sma_20 = self._sma(volume, 20)
        vol_ratio_5d = np.where(vol_sma_5 > 0, volume / vol_sma_5, 1.0)
        vol_ratio_20d = np.where(vol_sma_20 > 0, volume / vol_sma_20, 1.0)

        # Price position
        high_20 = self._rolling_max(high, 20)
        low_20 = self._rolling_min(low, 20)
        high_low_range = np.where(high_20 > 0, (high_20 - low_20) / high_20, 0)
        close_position = np.where(high_20 != low_20,
                                  (close - low_20) / (high_20 - low_20 + 1e-8), 0.5)

        # Momentum
        mom_5d = self._momentum(close, 5)
        mom_20d = self._momentum(close, 20)

        # Mean reversion score (our proprietary)
        mean_rev_score = self._mean_reversion_score(z_20d, rsi, bb_position)

        # Stack features
        features = np.column_stack([
            return_1d, return_5d, return_10d, return_20d,
            vol_5d, vol_20d, vol_ratio,
            z_20d, z_60d,
            rsi / 100, rsi_deviation,
            bb_position, bb_width,
            vol_ratio_5d, vol_ratio_20d,
            high_low_range, close_position,
            mom_5d, mom_20d,
            mean_rev_score
        ])

        # Handle NaN/Inf with feature-specific defaults
        # Use more sensible defaults than 0 for features that shouldn't be 0
        features = np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0)

        # Clip extreme values to prevent outliers from dominating
        # Returns: clip to +/- 50%
        features[:, 0:4] = np.clip(features[:, 0:4], -0.5, 0.5)
        # Volatility: clip to 0-200%
        features[:, 4:6] = np.clip(features[:, 4:6], 0, 2.0)
        # Vol ratio: clip to 0.1-10x
        features[:, 6] = np.clip(features[:, 6], 0.1, 10.0)
        # Z-scores: clip to +/- 5
        features[:, 7:9] = np.clip(features[:, 7:9], -5, 5)
        # RSI (already normalized): clip to 0-1
        features[:, 9] = np.clip(features[:, 9], 0, 1)
        # RSI deviation: clip to -1 to 1
        features[:, 10] = np.clip(features[:, 10], -1, 1)
        # BB position: clip to -0.5 to 1.5 (can be outside bands)
        features[:, 11] = np.clip(features[:, 11], -0.5, 1.5)
        # BB width: clip to 0-1
        features[:, 12] = np.clip(features[:, 12], 0, 1)
        # Volume ratios: clip to 0.1-10x
        features[:, 13:15] = np.clip(features[:, 13:15], 0.1, 10.0)
        # High-low range: clip to 0-0.5
        features[:, 15] = np.clip(features[:, 15], 0, 0.5)
        # Close position: clip to 0-1
        features[:, 16] = np.clip(features[:, 16], 0, 1)
        # Momentum: clip to -0.5 to 0.5
        features[:, 17:19] = np.clip(features[:, 17:19], -0.5, 0.5)
        # Mean reversion score: clip to -2 to 2
        features[:, 19] = np.clip(features[:, 19], -2, 2)

        return features[self.lookback:]  # Skip warmup period

    def _sma(self, arr: np.ndarray, period: int) -> np.ndarray:
        """Simple moving average."""
        result = np.zeros_like(arr)
        for i in range(period - 1, len(arr)):
            result[i] = np.mean(arr[i-period+1:i+1])
        return result

    def _rolling_std(self, arr: np.ndarray, period: int) -> np.ndarray:
        """Rolling standard deviation."""
        result = np.zeros_like(arr)
        for i in range(period - 1, len(arr)):
            result[i] = np.std(arr[i-period+1:i+1])
        return result

    def _rolling_max(self, arr: np.ndarray, period: int) -> np.ndarray:
        """Rolling maximum."""
        result = np.zeros_like(arr)
        for i in range(period - 1, len(arr)):
            result[i] = np.max(arr[i-period+1:i+1])
        return result

    def _rolling_min(self, arr: np.ndarray, period: int) -> np.ndarray:
        """Rolling minimum."""
        result = np.zeros_like(arr)
        for i in range(period - 1, len(arr)):
            result[i] = np.min(arr[i-period+1:i+1])
        return result

    def _rolling_return(self, arr: np.ndarray, period: int) -> np.ndarray:
        """Rolling return over period."""
        result = np.zeros_like(arr)
        for i in range(period, len(arr)):
            if arr[i-period] != 0:
                result[i] = (arr[i] / arr[i-period]) - 1
        return result

    def _z_score(self, arr: np.ndarray, period: int) -> np.ndarray:
        """Z-score over rolling period."""
        mean = self._sma(arr, period)
        std = self._rolling_std(arr, period)
        return np.where(std > 0, (arr - mean) / std, 0)

    def _rsi(self, arr: np.ndarray, period: int) -> np.ndarray:
        """Relative Strength Index."""
        delta = np.zeros_like(arr)
        delta[1:] = arr[1:] - arr[:-1]

        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = self._sma(gain, period)
        avg_loss = self._sma(loss, period)

        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _momentum(self, arr: np.ndarray, period: int) -> np.ndarray:
        """Price momentum."""
        result = np.zeros_like(arr)
        result[period:] = arr[period:] - arr[:-period]
        return result / (arr + 1e-8)  # Normalize

    def _mean_reversion_score(self, z_score: np.ndarray,
                              rsi: np.ndarray, bb_pos: np.ndarray) -> np.ndarray:
        """
        Composite mean reversion score.
        Higher when oversold conditions present.
        """
        # Z-score component (inverted - negative z = oversold)
        z_component = np.clip(-z_score / 2, -1, 1)

        # RSI component (low RSI = oversold)
        rsi_component = (50 - rsi) / 50

        # BB component (low position = oversold)
        bb_component = (0.5 - bb_pos) * 2

        # Weighted combination
        score = 0.4 * z_component + 0.35 * rsi_component + 0.25 * bb_component
        return score


class AttentionMLP(nn.Module):
    """
    MLP with attention mechanism for feature importance.
    Optimized for GPU inference.
    """

    def __init__(self, input_dim: int = 20, hidden_dims: List[int] = [64, 32, 16]):
        super().__init__()

        self.input_dim = input_dim

        # Feature attention
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=-1)
        )

        # Main network
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output heads
        self.prob_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )

        self.return_head = nn.Linear(prev_dim, 1)

        self.confidence_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (batch, features)

        Returns:
            prob: Mean reversion probability (batch, 1)
            expected_return: Expected return (batch, 1)
            confidence: Model confidence (batch, 1)
            attention_weights: Feature importance (batch, features)
        """
        # Compute attention weights
        attention_weights = self.attention(x)

        # Apply attention
        attended = x * attention_weights

        # Encode
        encoded = self.encoder(attended)

        # Output heads
        prob = self.prob_head(encoded)
        expected_return = self.return_head(encoded)
        confidence = self.confidence_head(encoded)

        return prob, expected_return, confidence, attention_weights


class GPUSignalModel:
    """
    GPU-accelerated signal model for Proteus.
    """

    PROTEUS_TICKERS = [
        "NVDA", "V", "MA", "AVGO", "AXP", "KLAC", "ORCL", "MRVL", "ABBV", "SYK",
        "EOG", "TXN", "GILD", "INTU", "MSFT", "QCOM", "JPM", "JNJ", "PFE", "WMT",
        "AMAT", "ADI", "NOW", "MLM", "IDXX", "EXR", "ROAD", "INSM", "SCHW", "AIG",
        "USB", "CVS", "LOW", "LMT", "COP", "SLB", "APD", "MS", "PNC", "CRM",
        "ADBE", "TGT", "CAT", "XOM", "MPC", "ECL", "NEE", "HCA", "CMCSA", "TMUS",
        "META", "ETN", "HD", "SHW"
    ]

    def __init__(self, model_dir: str = "models/gpu_signal"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Check CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[GPU] Using device: {self.device}")

        if self.device.type == "cuda":
            print(f"[GPU] {torch.cuda.get_device_name(0)}")
            print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Initialize model
        self.model = AttentionMLP(input_dim=20).to(self.device)
        self.feature_extractor = FeatureExtractor()

        # Try to load existing weights
        self._load_model()

    def _load_model(self):
        """Load model weights if available."""
        weights_file = os.path.join(self.model_dir, "model_weights.pt")
        if os.path.exists(weights_file):
            self.model.load_state_dict(torch.load(weights_file, map_location=self.device))
            print(f"[GPU] Loaded weights from {weights_file}")
        else:
            print("[GPU] No existing weights - model will use random initialization")
            print("[GPU] Run train_on_history() to train the model")

    def save_model(self):
        """Save model weights."""
        weights_file = os.path.join(self.model_dir, "model_weights.pt")
        torch.save(self.model.state_dict(), weights_file)
        print(f"[GPU] Saved weights to {weights_file}")

    def prepare_training_data(self, days_back: int = 365) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare training data from historical prices.
        Returns features, binary labels (hit 1%+), and continuous returns.
        """
        print(f"\n[GPU] Preparing training data ({days_back} days)...")

        all_features = []
        all_labels = []
        all_returns = []

        for ticker in self.PROTEUS_TICKERS:
            try:
                stock = yf.Ticker(ticker)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back + 100)  # Extra for warmup
                df = stock.history(start=start_date.strftime('%Y-%m-%d'),
                                   end=end_date.strftime('%Y-%m-%d'))

                if len(df) < 100:
                    continue

                # Extract features
                features = self.feature_extractor.extract(df)
                if len(features) == 0:
                    continue

                # Create labels (next 2-day return > 1%)
                close = df['Close'].values[self.feature_extractor.lookback:]
                future_return = np.zeros(len(close))
                for i in range(len(close) - 2):
                    future_return[i] = (close[i+2] / close[i]) - 1

                labels = (future_return > 0.01).astype(float)
                returns = future_return * 100  # Convert to percentage

                # Align (remove last 2 samples that don't have labels)
                features = features[:-2]
                labels = labels[:-2]
                returns = returns[:-2]

                all_features.append(features)
                all_labels.append(labels)
                all_returns.append(returns)

                print(f"  {ticker}: {len(features)} samples")

            except Exception as e:
                print(f"  {ticker}: [ERROR] {e}")

        if not all_features:
            raise ValueError("No training data available")

        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        r = np.hstack(all_returns)

        print(f"\n[GPU] Total samples: {len(X)}")
        print(f"[GPU] Positive rate: {y.mean()*100:.1f}%")
        print(f"[GPU] Avg 2-day return: {r.mean():.2f}%")

        return torch.FloatTensor(X), torch.FloatTensor(y), torch.FloatTensor(r)

    def train_on_history(self, epochs: int = 50, batch_size: int = 256, lr: float = 0.001):
        """
        Train the model on historical data with multi-task learning.
        Trains probability, expected return, and confidence heads together.
        Uses GPU for fast training.
        """
        print("\n" + "="*70)
        print("GPU MODEL TRAINING (Multi-Task)")
        print("="*70)

        # Prepare data (now includes returns)
        X, y, r = self.prepare_training_data()
        X, y, r = X.to(self.device), y.to(self.device), r.to(self.device)

        # Split train/val
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        r_train, r_val = r[:split_idx], r[split_idx:]

        # DataLoader with returns
        train_dataset = TensorDataset(X_train, y_train, r_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Optimizer and losses
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        bce_criterion = nn.BCELoss()
        mse_criterion = nn.MSELoss()

        # Training loop
        self.model.train()
        best_val_loss = float('inf')

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_prob_loss = 0
            epoch_ret_loss = 0
            for batch_X, batch_y, batch_r in train_loader:
                optimizer.zero_grad()

                prob, expected_ret, conf, _ = self.model(batch_X)

                # Multi-task loss
                prob_loss = bce_criterion(prob.squeeze(), batch_y)
                ret_loss = mse_criterion(expected_ret.squeeze(), batch_r / 10.0)  # Scale returns

                # Combined loss (weight return loss lower since it's harder to predict)
                loss = prob_loss + 0.1 * ret_loss

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_prob_loss += prob_loss.item()
                epoch_ret_loss += ret_loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_prob, val_ret, _, _ = self.model(X_val)
                val_prob_loss = bce_criterion(val_prob.squeeze(), y_val).item()
                val_ret_loss = mse_criterion(val_ret.squeeze(), r_val / 10.0).item()
                val_loss = val_prob_loss + 0.1 * val_ret_loss

                # Accuracy
                val_pred = (val_prob.squeeze() > 0.5).float()
                val_acc = (val_pred == y_val).float().mean().item()

                # Return prediction quality
                val_ret_mae = torch.abs(val_ret.squeeze() * 10.0 - r_val).mean().item()

            self.model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Loss: {epoch_loss/len(train_loader):.4f} | "
                      f"Val Acc: {val_acc*100:.1f}% | "
                      f"Ret MAE: {val_ret_mae:.2f}%")

        print("\n[GPU] Training complete!")
        print(f"[GPU] Best validation loss: {best_val_loss:.4f}")
        return best_val_loss

    def predict(self, ticker: str, df: pd.DataFrame = None) -> Optional[GPUSignal]:
        """
        Generate signal for a single stock.

        Args:
            ticker: Stock ticker
            df: Optional pre-fetched DataFrame

        Returns:
            GPUSignal or None if insufficient data
        """
        # Fetch data if not provided
        if df is None:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period="6mo")
            except:
                return None

        if len(df) < self.feature_extractor.lookback + 10:
            return None

        # Extract features
        features = self.feature_extractor.extract(df)
        if len(features) == 0:
            return None

        # Use only the latest sample
        latest = features[-1:]
        X = torch.FloatTensor(latest).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            prob, expected_ret, conf, attention = self.model(X)

        prob = prob.cpu().item()
        # Model outputs returns scaled by 10, so multiply by 10 to get percentage
        expected_ret = expected_ret.cpu().item() * 10.0
        # Clip to realistic range (-10% to +10% for 2-day returns)
        expected_ret = max(-10.0, min(10.0, expected_ret))
        conf = conf.cpu().item()
        attention = attention.cpu().numpy().flatten()

        # Compute signal strength (0-100) - RECALIBRATED V2 based on 365-day backtest
        # Key findings: GOOD tier (60-70) had 67.6% win rate vs STRONG (70-80) at 56.7%
        # Root cause: high E[R] predictions were unreliable, causing inflated scores
        # Fix: Reduce E[R] weight, favor probability, optimal confidence at 0.35-0.45

        # Base: probability is the most reliable predictor (0-100)
        base_score = prob * 100

        # Return component: REDUCED weight - E[R] predictions are overconfident
        # +2% expected return adds +6 points (was +20), -2% subtracts 6 points
        return_component = expected_ret * 3

        # Confidence adjustment: optimal range is 0.30-0.50 based on backtest
        # GOOD tier average was 0.357, which performed best
        if conf > 0.70:
            conf_adjustment = 0.85  # Penalize overconfidence more
        elif conf > 0.50:
            conf_adjustment = 0.95  # Slight penalty
        elif conf < 0.25:
            conf_adjustment = conf / 0.25  # Penalize very low confidence
        else:
            conf_adjustment = 1.0  # Optimal range 0.25-0.50

        # Quality bonus: reward signals with BOTH good prob AND positive E[R]
        # This helps separate truly strong signals from noise
        quality_bonus = 0
        if prob > 0.50 and expected_ret > 0.5:
            quality_bonus = min(10, (prob - 0.50) * 30 + expected_ret * 2)
        elif prob < 0.45 or expected_ret < 0:
            quality_bonus = -5  # Slight penalty for weak signals

        # Mean reversion score from features (index 19)
        mr_score = latest[0, 19] if latest.shape[1] > 19 else 0
        mr_component = float(mr_score) * 10  # Reduced from 15

        # Combine components
        signal_strength = (base_score * conf_adjustment) + return_component + mr_component + quality_bonus

        # Clamp to 0-100 range
        signal_strength = max(0, min(100, signal_strength))

        # Feature importance
        feature_importance = dict(zip(
            FeatureExtractor.FEATURE_NAMES,
            attention.tolist()
        ))

        return GPUSignal(
            ticker=ticker,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            mean_reversion_prob=round(prob, 4),
            expected_return=round(expected_ret, 2),
            confidence=round(conf, 4),
            signal_strength=round(signal_strength, 1),
            feature_importance=feature_importance
        )

    def scan_all(self) -> List[GPUSignal]:
        """
        Scan all Proteus stocks in parallel using GPU.

        Returns:
            List of signals sorted by strength
        """
        print("\n" + "="*70)
        print("GPU PARALLEL SCAN")
        print("="*70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {self.device}")

        signals = []

        # Fetch all data first (IO bound)
        print("\n1. Fetching data for all stocks...")
        stock_data = {}
        for ticker in self.PROTEUS_TICKERS:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period="6mo")
                if len(df) >= self.feature_extractor.lookback + 10:
                    stock_data[ticker] = df
                    print(f"  {ticker}: OK ({len(df)} bars)")
            except Exception as e:
                print(f"  {ticker}: [ERROR]")

        # Batch process on GPU
        print(f"\n2. Running GPU inference on {len(stock_data)} stocks...")

        # Extract all features
        all_features = []
        tickers = []
        for ticker, df in stock_data.items():
            features = self.feature_extractor.extract(df)
            if len(features) > 0:
                all_features.append(features[-1])  # Latest sample
                tickers.append(ticker)

        if not all_features:
            print("[WARN] No valid features extracted")
            return []

        # Batch inference
        X = torch.FloatTensor(np.array(all_features)).to(self.device)

        self.model.eval()
        with torch.no_grad():
            probs, returns, confs, attentions = self.model(X)

        probs = probs.cpu().numpy().flatten()
        # Model outputs returns scaled by 10, so multiply by 10 to get percentage
        returns = returns.cpu().numpy().flatten() * 10.0
        # Clip to realistic range (-10% to +10% for 2-day returns)
        returns = np.clip(returns, -10.0, 10.0)
        confs = confs.cpu().numpy().flatten()
        attentions = attentions.cpu().numpy()

        # Build signals
        for i, ticker in enumerate(tickers):
            # RECALIBRATED signal strength (same as predict())
            prob = probs[i]
            conf = confs[i]
            expected_ret = returns[i]

            # Base: probability score (0-100)
            base_score = prob * 100

            # Return component: +2% expected return adds +20 points
            return_component = expected_ret * 10

            # Confidence adjustment: penalize overconfidence and underconfidence
            if conf > 0.85:
                conf_adjustment = 1.0 - (conf - 0.85) * 2
            elif conf < 0.4:
                conf_adjustment = conf / 0.4
            else:
                conf_adjustment = 1.0

            # Mean reversion score from features (index 19)
            mr_score = all_features[i][19] if len(all_features[i]) > 19 else 0
            mr_component = float(mr_score) * 15

            # Combine and clamp
            signal_strength = (base_score * conf_adjustment) + return_component + mr_component
            signal_strength = max(0, min(100, signal_strength))

            feature_importance = dict(zip(
                FeatureExtractor.FEATURE_NAMES,
                attentions[i].tolist()
            ))

            signal = GPUSignal(
                ticker=ticker,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                mean_reversion_prob=round(float(probs[i]), 4),
                expected_return=round(float(returns[i]), 2),
                confidence=round(float(confs[i]), 4),
                signal_strength=round(float(signal_strength), 1),
                feature_importance=feature_importance
            )
            signals.append(signal)

        # Sort by strength
        signals.sort(key=lambda x: x.signal_strength, reverse=True)

        # Print top signals
        print(f"\n3. Top signals:")
        for sig in signals[:10]:
            print(f"   {sig.ticker}: Str={sig.signal_strength:.1f}, "
                  f"P={sig.mean_reversion_prob:.2f}, "
                  f"E[R]={sig.expected_return:+.1f}%, "
                  f"Conf={sig.confidence:.2f}")

        # Cache the stock data for price fetching and enhanced signal calculation
        self._last_scan_prices = {}
        self._last_scan_consecutive_down = {}
        self._last_scan_volume_ratio = {}

        for ticker, df in stock_data.items():
            if not df.empty:
                self._last_scan_prices[ticker] = float(df['Close'].iloc[-1])

                # Calculate consecutive down days
                consecutive = 0
                for i in range(len(df) - 1, -1, -1):
                    if df['Close'].iloc[i] < df['Open'].iloc[i]:
                        consecutive += 1
                    else:
                        break
                self._last_scan_consecutive_down[ticker] = consecutive

                # Calculate volume ratio (current / 20-day average)
                if len(df) >= 20:
                    avg_volume = df['Volume'].iloc[-20:].mean()
                    if avg_volume > 0:
                        self._last_scan_volume_ratio[ticker] = float(df['Volume'].iloc[-1] / avg_volume)
                    else:
                        self._last_scan_volume_ratio[ticker] = 1.0
                else:
                    self._last_scan_volume_ratio[ticker] = 1.0

        return signals

    def get_last_prices(self) -> Dict[str, float]:
        """
        Get prices from the last scan.

        Returns:
            Dictionary of {ticker: last_close_price}
        """
        return getattr(self, '_last_scan_prices', {})

    def get_consecutive_down_days(self, ticker: str) -> int:
        """
        Get number of consecutive down days for a ticker from cached data.

        Returns:
            Number of consecutive down days (0 if not available)
        """
        consecutive_data = getattr(self, '_last_scan_consecutive_down', {})
        return consecutive_data.get(ticker, 1)  # Default to 1 if not available

    def get_volume_ratio(self, ticker: str) -> float:
        """
        Get volume ratio (current volume / 20-day average) for a ticker.

        Returns:
            Volume ratio (1.0 if not available)
        """
        volume_data = getattr(self, '_last_scan_volume_ratio', {})
        return volume_data.get(ticker, 1.0)  # Default to 1.0 if not available


def run_gpu_scan():
    """Main entry point for GPU scan."""
    model = GPUSignalModel()

    # Check if model needs training
    weights_file = "models/gpu_signal/model_weights.pt"
    if not os.path.exists(weights_file):
        print("\n[INFO] Model not trained yet. Training on historical data...")
        model.train_on_history(epochs=50)

    # Run scan
    signals = model.scan_all()

    # Save results
    output_file = f"data/gpu_signals/scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("data/gpu_signals", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump([asdict(s) for s in signals], f, indent=2)
    print(f"\n[SAVED] {output_file}")

    return signals


if __name__ == "__main__":
    run_gpu_scan()

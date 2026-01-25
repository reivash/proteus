"""
LSTM Signal Model - Neural Network V2 Baseline

Architecture based on research/NEURAL_NETWORK_V2_RESEARCH.md:
- Bi-LSTM with attention for temporal patterns
- Multi-head self-attention for feature importance
- Multi-task learning (probability, return, confidence)

Target: +3-5% accuracy improvement over MLP baseline
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

try:
    import yfinance as yf
except ImportError:
    yf = None


@dataclass
class LSTMSignal:
    """Signal output from LSTM model."""
    ticker: str
    timestamp: str
    mean_reversion_prob: float
    expected_return: float
    confidence: float
    signal_strength: float
    temporal_attention: List[float]  # Which time steps were important


class SequenceDataset(Dataset):
    """Dataset for sliding window sequences."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, returns: np.ndarray,
                 seq_len: int = 30):
        """
        Args:
            features: (n_samples, n_features) array
            labels: (n_samples,) binary labels
            returns: (n_samples,) continuous returns
            seq_len: Sequence length for LSTM
        """
        self.seq_len = seq_len
        self.features = features
        self.labels = labels
        self.returns = returns

        # Valid indices (need seq_len previous samples)
        self.valid_indices = list(range(seq_len, len(features)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        # Get sequence of past seq_len samples
        seq = self.features[actual_idx - self.seq_len:actual_idx]
        label = self.labels[actual_idx]
        ret = self.returns[actual_idx]
        return torch.FloatTensor(seq), torch.FloatTensor([label]), torch.FloatTensor([ret])


class TemporalAttention(nn.Module):
    """Attention over time steps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)

        Returns:
            context: (batch, hidden_dim) - weighted sum of time steps
            attention_weights: (batch, seq_len) - importance of each time step
        """
        # Calculate attention scores
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch, hidden_dim)

        return context, weights


class LSTMSignalNetwork(nn.Module):
    """
    Bi-LSTM with temporal attention for mean reversion signals.

    Architecture:
        Input: (batch, seq_len, features)
        -> Bi-LSTM (captures forward and backward patterns)
        -> Temporal Attention (finds important time steps)
        -> Pooling (concat last state + attention context + mean)
        -> MLP Decoder
        -> Output Heads (probability, return, confidence)
    """

    def __init__(self, input_dim: int = 20, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Bi-LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Temporal attention
        self.attention = TemporalAttention(hidden_dim * 2)

        # Pooling output: last_state + attention_context + mean = 3 * hidden_dim * 2
        pooled_dim = hidden_dim * 2 * 3

        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Output heads
        self.prob_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.return_head = nn.Linear(hidden_dim // 2, 1)

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                  torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, features)

        Returns:
            prob: (batch, 1) mean reversion probability
            expected_return: (batch, 1) expected return
            confidence: (batch, 1) model confidence
            attention_weights: (batch, seq_len) temporal attention weights
        """
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden*2)

        # Temporal attention
        context, attention_weights = self.attention(lstm_out)

        # Get last hidden state (concat forward and backward)
        last_state = lstm_out[:, -1, :]  # (batch, hidden*2)

        # Mean pooling
        mean_pool = lstm_out.mean(dim=1)  # (batch, hidden*2)

        # Concatenate all representations
        pooled = torch.cat([last_state, context, mean_pool], dim=-1)  # (batch, hidden*6)

        # Decode
        decoded = self.decoder(pooled)

        # Output heads
        prob = self.prob_head(decoded)
        expected_return = self.return_head(decoded)
        confidence = self.confidence_head(decoded)

        return prob, expected_return, confidence, attention_weights


class LSTMSignalModel:
    """
    LSTM-based signal model for Proteus - Neural Network V2.
    """

    PROTEUS_TICKERS = [
        "NVDA", "V", "MA", "AVGO", "AXP", "KLAC", "ORCL", "MRVL", "ABBV", "SYK",
        "EOG", "TXN", "GILD", "INTU", "MSFT", "QCOM", "JPM", "JNJ", "PFE", "WMT",
        "AMAT", "ADI", "NOW", "MLM", "IDXX", "EXR", "ROAD", "INSM", "SCHW", "AIG",
        "USB", "CVS", "LOW", "LMT", "COP", "SLB", "APD", "MS", "PNC", "CRM",
        "ADBE", "TGT", "CAT", "XOM", "MPC", "ECL", "NEE", "HCA", "CMCSA", "TMUS",
        "META", "ETN", "HD", "SHW"
    ]

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

    def __init__(self, model_dir: str = "features/daily_picks/models/lstm_signal", seq_len: int = 30):
        self.model_dir = model_dir
        self.seq_len = seq_len
        os.makedirs(model_dir, exist_ok=True)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LSTM] Using device: {self.device}")

        if self.device.type == "cuda":
            print(f"[LSTM] {torch.cuda.get_device_name(0)}")

        # Initialize model
        self.model = LSTMSignalNetwork(
            input_dim=20,
            hidden_dim=64,
            num_layers=2,
            dropout=0.3
        ).to(self.device)

        self._load_model()

    def _load_model(self):
        """Load model weights if available."""
        weights_file = os.path.join(self.model_dir, "lstm_weights.pt")
        if os.path.exists(weights_file):
            self.model.load_state_dict(
                torch.load(weights_file, map_location=self.device, weights_only=True)
            )
            print(f"[LSTM] Loaded weights from {weights_file}")
        else:
            print("[LSTM] No existing weights - model needs training")

    def save_model(self):
        """Save model weights."""
        weights_file = os.path.join(self.model_dir, "lstm_weights.pt")
        torch.save(self.model.state_dict(), weights_file)
        print(f"[LSTM] Saved weights to {weights_file}")

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features from OHLCV dataframe."""
        if len(df) < 60:
            return np.array([])

        close = df['Close'].values.astype(np.float64)
        high = df['High'].values.astype(np.float64)
        low = df['Low'].values.astype(np.float64)
        volume = df['Volume'].values.astype(np.float64)

        # Returns
        return_1d = np.zeros_like(close)
        return_1d[1:] = (close[1:] / close[:-1]) - 1

        def rolling_return(arr, period):
            result = np.zeros_like(arr)
            for i in range(period, len(arr)):
                result[i] = (arr[i] / arr[i-period]) - 1
            return result

        def rolling_std(arr, period):
            result = np.zeros_like(arr)
            for i in range(period, len(arr)):
                result[i] = np.std(arr[i-period+1:i+1])
            return result

        def rolling_mean(arr, period):
            result = np.zeros_like(arr)
            for i in range(period, len(arr)):
                result[i] = np.mean(arr[i-period+1:i+1])
            return result

        return_5d = rolling_return(close, 5)
        return_10d = rolling_return(close, 10)
        return_20d = rolling_return(close, 20)

        # Volatility
        vol_5d = rolling_std(return_1d, 5) * np.sqrt(252)
        vol_20d = rolling_std(return_1d, 20) * np.sqrt(252)
        vol_ratio = np.where(vol_20d > 0, vol_5d / vol_20d, 1.0)

        # Z-scores
        mean_20 = rolling_mean(close, 20)
        std_20 = rolling_std(close, 20)
        z_20d = np.where(std_20 > 0, (close - mean_20) / std_20, 0)

        mean_60 = rolling_mean(close, 60)
        std_60 = rolling_std(close, 60)
        z_60d = np.where(std_60 > 0, (close - mean_60) / std_60, 0)

        # RSI
        delta = np.diff(close, prepend=close[0])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = rolling_mean(gains, 14)
        avg_loss = rolling_mean(losses, 14)
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        rsi_deviation = (rsi - 50) / 50

        # Bollinger Bands
        bb_mid = rolling_mean(close, 20)
        bb_std = rolling_std(close, 20)
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = np.where(bb_mid > 0, (bb_upper - bb_lower) / bb_mid, 0)
        bb_position = np.where(bb_upper != bb_lower,
                               (close - bb_lower) / (bb_upper - bb_lower + 1e-8), 0.5)

        # Volume
        vol_sma_5 = rolling_mean(volume, 5)
        vol_sma_20 = rolling_mean(volume, 20)
        vol_ratio_5d = np.where(vol_sma_5 > 0, volume / vol_sma_5, 1.0)
        vol_ratio_20d = np.where(vol_sma_20 > 0, volume / vol_sma_20, 1.0)

        # Price position
        high_20 = np.array([np.max(high[max(0,i-19):i+1]) for i in range(len(high))])
        low_20 = np.array([np.min(low[max(0,i-19):i+1]) for i in range(len(low))])
        high_low_range = np.where(high_20 > 0, (high_20 - low_20) / high_20, 0)
        close_position = np.where(high_20 != low_20,
                                  (close - low_20) / (high_20 - low_20 + 1e-8), 0.5)

        # Momentum
        mom_5d = rolling_return(close, 5)
        mom_20d = rolling_return(close, 20)

        # Mean reversion score
        z_component = np.clip(-z_20d / 3, -1, 1)
        rsi_component = np.clip((30 - rsi) / 30, -1, 1)
        bb_component = np.clip((0.2 - bb_position) / 0.4, -1, 1)
        mr_score = 0.4 * z_component + 0.35 * rsi_component + 0.25 * bb_component

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
            mr_score
        ])

        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        return features

    def prepare_training_data(self, days_back: int = 730) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data with sliding windows.

        Returns:
            features: (n_samples, n_features)
            labels: (n_samples,) binary labels
            returns: (n_samples,) continuous returns
        """
        print(f"\n[LSTM] Preparing training data ({days_back} days = ~{days_back//365} years)...")

        all_features = []
        all_labels = []
        all_returns = []

        for ticker in self.PROTEUS_TICKERS:
            try:
                stock = yf.Ticker(ticker)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back + 100)
                df = stock.history(start=start_date.strftime('%Y-%m-%d'),
                                   end=end_date.strftime('%Y-%m-%d'))

                if len(df) < 100:
                    continue

                features = self._extract_features(df)
                if len(features) == 0:
                    continue

                # Create labels (next 2-day return > 1%)
                close = df['Close'].values[60:]  # Skip warmup
                features = features[60:]  # Align

                future_return = np.zeros(len(close))
                for i in range(len(close) - 2):
                    future_return[i] = (close[i+2] / close[i]) - 1

                labels = (future_return > 0.01).astype(float)
                returns = future_return * 100

                # Remove last 2 samples
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

        print(f"\n[LSTM] Total samples: {len(X)}")
        print(f"[LSTM] Positive rate: {y.mean()*100:.1f}%")
        print(f"[LSTM] Avg 2-day return: {r.mean():.2f}%")

        return X, y, r

    def train(self, epochs: int = 100, batch_size: int = 64, lr: float = 1e-4):
        """
        Train LSTM model with sliding window sequences.
        """
        print("\n" + "="*70)
        print("LSTM MODEL TRAINING (Neural Network V2)")
        print("="*70)

        # Prepare data
        X, y, r = self.prepare_training_data(days_back=730)

        # Create datasets with sliding windows
        split_idx = int(len(X) * 0.8)

        train_dataset = SequenceDataset(X[:split_idx], y[:split_idx], r[:split_idx],
                                         seq_len=self.seq_len)
        val_dataset = SequenceDataset(X[split_idx:], y[split_idx:], r[split_idx:],
                                       seq_len=self.seq_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"\n[LSTM] Training samples: {len(train_dataset)}")
        print(f"[LSTM] Validation samples: {len(val_dataset)}")
        print(f"[LSTM] Sequence length: {self.seq_len}")

        # Optimizer and losses
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2
        )
        bce_criterion = nn.BCELoss()
        mse_criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            epoch_acc = 0
            n_batches = 0

            for batch_seq, batch_y, batch_r in train_loader:
                batch_seq = batch_seq.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_r = batch_r.to(self.device)

                optimizer.zero_grad()

                prob, expected_ret, conf, _ = self.model(batch_seq)

                # Multi-task loss
                prob_loss = bce_criterion(prob, batch_y)
                ret_loss = mse_criterion(expected_ret, batch_r / 10.0)

                loss = 0.5 * prob_loss + 0.3 * ret_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += ((prob > 0.5).float() == batch_y).float().mean().item()
                n_batches += 1

            scheduler.step()

            # Validation
            self.model.eval()
            val_loss = 0
            val_acc = 0
            val_batches = 0

            with torch.no_grad():
                for batch_seq, batch_y, batch_r in val_loader:
                    batch_seq = batch_seq.to(self.device)
                    batch_y = batch_y.to(self.device)
                    batch_r = batch_r.to(self.device)

                    prob, expected_ret, conf, _ = self.model(batch_seq)

                    prob_loss = bce_criterion(prob, batch_y)
                    ret_loss = mse_criterion(expected_ret, batch_r / 10.0)
                    loss = 0.5 * prob_loss + 0.3 * ret_loss

                    val_loss += loss.item()
                    val_acc += ((prob > 0.5).float() == batch_y).float().mean().item()
                    val_batches += 1

            avg_val_loss = val_loss / val_batches
            avg_val_acc = val_acc / val_batches

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {epoch_loss/n_batches:.4f} | "
                      f"Train Acc: {epoch_acc/n_batches*100:.1f}% | "
                      f"Val Acc: {avg_val_acc*100:.1f}%")

            if patience_counter >= patience:
                print(f"\n[LSTM] Early stopping at epoch {epoch+1}")
                break

        print(f"\n[LSTM] Training complete! Best val loss: {best_val_loss:.4f}")
        return best_val_loss

    def predict(self, ticker: str, df: pd.DataFrame = None) -> Optional[LSTMSignal]:
        """Generate signal for a single stock using LSTM."""
        if df is None:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period="6mo")
            except:
                return None

        if len(df) < self.seq_len + 60:
            return None

        # Extract features
        features = self._extract_features(df)
        if len(features) < self.seq_len:
            return None

        # Get last seq_len samples
        seq = features[-self.seq_len:]
        X = torch.FloatTensor(seq).unsqueeze(0).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            prob, expected_ret, conf, attention = self.model(X)

        prob = prob.cpu().item()
        expected_ret = expected_ret.cpu().item() * 10.0
        expected_ret = max(-10.0, min(10.0, expected_ret))
        conf = conf.cpu().item()
        attention = attention.cpu().numpy().flatten().tolist()

        # Compute signal strength (similar to MLP but using LSTM outputs)
        signal_strength = prob * 100

        # Return component
        signal_strength += expected_ret * 3

        # Confidence adjustment
        if conf > 0.70:
            signal_strength *= 0.90
        elif conf < 0.25:
            signal_strength *= conf / 0.25

        signal_strength = max(0, min(100, signal_strength))

        return LSTMSignal(
            ticker=ticker,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            mean_reversion_prob=round(prob, 4),
            expected_return=round(expected_ret, 2),
            confidence=round(conf, 4),
            signal_strength=round(signal_strength, 1),
            temporal_attention=attention[-10:]  # Last 10 time steps
        )


def train_lstm_model():
    """Train LSTM model."""
    model = LSTMSignalModel()
    model.train(epochs=100, batch_size=64, lr=1e-4)
    return model


if __name__ == "__main__":
    train_lstm_model()

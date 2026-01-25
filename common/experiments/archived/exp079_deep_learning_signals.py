"""
EXPERIMENT: EXP-079
Date: 2025-11-17
Objective: Deep Learning Signal Prediction using LSTM/GRU

HYPOTHESIS:
Current XGBoost model (0.834 AUC) uses static features and misses temporal patterns.
LSTM/GRU networks can learn time-series dependencies in price action, volume, and
technical indicators that lead to successful mean reversion trades.

Deep learning advantages:
1. Temporal pattern recognition (20-day sequences)
2. Non-linear feature interactions
3. Attention to critical time windows
4. Multi-task learning (win/loss + return prediction)

ALGORITHM:
1. Prepare time-series datasets:
   - 20-day sequences of OHLCV + technical indicators
   - Label: Trade outcome (win/loss) + actual return
2. Build LSTM/GRU architecture:
   - Input: (batch, 20 days, 41 features)
   - LSTM layers with dropout
   - Multi-task head: classification + regression
3. Train on GPU with early stopping
4. Validate against XGBoost baseline (0.834 AUC)
5. Measure improvement in production

EXPECTED RESULTS:
- AUC improvement: 0.834 → 0.85-0.90 (+6-7%)
- Win rate improvement: +3-5pp over current ML
- Return prediction: Mean Absolute Error < 1.5%
- Annual impact: +15-25% cumulative return boost

SUCCESS CRITERIA:
- AUC >= 0.85 (vs 0.834 baseline)
- Win rate improvement >= +3pp
- Model generalizes across 70%+ of stocks
- Training time < 2 hours on RTX GPU
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from common.config.mean_reversion_params import get_all_tickers


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time-series signal data."""

    def __init__(self, sequences, labels_classification, labels_regression):
        """
        Args:
            sequences: (N, seq_len, features) numpy array
            labels_classification: (N,) binary labels (win/loss)
            labels_regression: (N,) continuous labels (actual return)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels_clf = torch.FloatTensor(labels_classification)
        self.labels_reg = torch.FloatTensor(labels_regression)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels_clf[idx], self.labels_reg[idx]


class LSTMSignalPredictor(nn.Module):
    """LSTM-based multi-task signal predictor."""

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMSignalPredictor, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Classification head (win/loss)
        self.fc_clf = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Regression head (return prediction)
        self.fc_reg = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            clf_output: (batch, 1) - probability of winning trade
            reg_output: (batch, 1) - predicted return
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]

        # Multi-task outputs
        clf_output = self.fc_clf(last_hidden)
        reg_output = self.fc_reg(last_hidden)

        return clf_output, reg_output


def prepare_time_series_data(ticker: str,
                              start_date: str = '2020-01-01',
                              end_date: str = '2024-11-17',
                              seq_len: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare time-series sequences for LSTM training.

    Returns:
        sequences: (N, seq_len, features) array
        labels_clf: (N,) binary array (1 = win, 0 = loss)
        labels_reg: (N,) continuous array (actual return %)
    """
    try:
        # Fetch data
        buffer_start = (pd.to_datetime(start_date) - timedelta(days=180)).strftime('%Y-%m-%d')
        fetcher = YahooFinanceFetcher()
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)

        if len(data) < 100:
            return None, None, None

        # Engineer features
        engineer = TechnicalFeatureEngineer(fillna=True)
        enriched_data = engineer.engineer_features(data)

        # Detect signals
        detector = MeanReversionDetector(
            z_score_threshold=1.5,
            rsi_oversold=35,
            volume_multiplier=1.3,
            price_drop_threshold=-1.5
        )

        signals = detector.detect_overcorrections(enriched_data)

        if len(signals) == 0:
            return None, None, None

        # For LSTM training, use ALL panic_sell signals (don't apply regime/earnings filters)
        # We want maximum data for deep learning to learn patterns
        panic_signals = signals[signals['panic_sell'] == 1].copy()

        if len(panic_signals) == 0:
            return None, None, None

        # Select features for LSTM input
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'rsi', 'z_score', 'daily_return',
            'atr_14', 'bb_upper', 'bb_lower', 'bb_width',
            'macd', 'macd_signal', 'macd_hist',
            'ema_12', 'ema_26', 'ema_50',
            'adx', 'cci', 'stoch_k', 'stoch_d',
            'obv', 'cmf', 'mfi',
            'volatility_20', 'volume_ratio'
        ]

        # Build sequences and labels
        sequences = []
        labels_clf = []
        labels_reg = []

        # Get all panic_sell signal positions
        panic_indices = panic_signals.index.tolist()

        for signal_idx in panic_indices:

            # Need seq_len days before signal
            if signal_idx < seq_len:
                continue

            # Need 3 days after signal to calculate outcome
            if signal_idx + 3 >= len(enriched_data):
                continue

            # Extract sequence (seq_len days ending at signal date)
            sequence_data = enriched_data.iloc[signal_idx - seq_len + 1:signal_idx + 1]

            # Check if all features exist
            if not all(col in sequence_data.columns for col in feature_cols):
                continue

            sequence_features = sequence_data[feature_cols].values

            # Handle NaN values
            if np.isnan(sequence_features).any():
                sequence_features = np.nan_to_num(sequence_features, nan=0.0)

            # Calculate actual outcome (using backtester logic)
            entry_price = enriched_data.iloc[signal_idx]['Close']
            future_data = enriched_data.iloc[signal_idx + 1:signal_idx + 4]

            if len(future_data) < 2:
                continue

            # Simulate time_decay exit
            exit_price = None
            exit_return = 0

            for days_held, (date, row) in enumerate(future_data.iterrows()):
                current_price = row['Close']
                return_pct = (current_price / entry_price - 1) * 100

                # Time-decay targets
                if days_held == 0:
                    profit_target, stop_loss = 2.0, -2.0
                elif days_held == 1:
                    profit_target, stop_loss = 1.5, -1.5
                else:
                    profit_target, stop_loss = 1.0, -1.0

                if return_pct >= profit_target or return_pct <= stop_loss or days_held >= 2:
                    exit_price = current_price
                    exit_return = return_pct
                    break

            if exit_price is None:
                continue

            # Labels
            label_clf = 1 if exit_return > 0 else 0
            label_reg = exit_return

            sequences.append(sequence_features)
            labels_clf.append(label_clf)
            labels_reg.append(label_reg)

        if len(sequences) == 0:
            return None, None, None

        return (
            np.array(sequences),
            np.array(labels_clf),
            np.array(labels_reg)
        )

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def train_lstm_model(train_loader, val_loader, input_size, device, epochs=50):
    """Train LSTM model with multi-task learning."""

    model = LSTMSignalPredictor(input_size=input_size).to(device)

    # Loss functions
    criterion_clf = nn.BCELoss()
    criterion_reg = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    print("\nTraining LSTM model...")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        for sequences, labels_clf, labels_reg in train_loader:
            sequences = sequences.to(device)
            labels_clf = labels_clf.to(device).unsqueeze(1)
            labels_reg = labels_reg.to(device).unsqueeze(1)

            optimizer.zero_grad()

            clf_output, reg_output = model(sequences)

            loss_clf = criterion_clf(clf_output, labels_clf)
            loss_reg = criterion_reg(reg_output, labels_reg)

            # Combined loss (weighted)
            loss = loss_clf + 0.1 * loss_reg

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        val_clf_preds = []
        val_clf_labels = []
        val_reg_preds = []
        val_reg_labels = []

        with torch.no_grad():
            for sequences, labels_clf, labels_reg in val_loader:
                sequences = sequences.to(device)
                labels_clf = labels_clf.to(device).unsqueeze(1)
                labels_reg = labels_reg.to(device).unsqueeze(1)

                clf_output, reg_output = model(sequences)

                loss_clf = criterion_clf(clf_output, labels_clf)
                loss_reg = criterion_reg(reg_output, labels_reg)
                loss = loss_clf + 0.1 * loss_reg

                val_loss += loss.item()

                val_clf_preds.extend(clf_output.cpu().numpy())
                val_clf_labels.extend(labels_clf.cpu().numpy())
                val_reg_preds.extend(reg_output.cpu().numpy())
                val_reg_labels.extend(labels_reg.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'logs/experiments/exp079_best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load('logs/experiments/exp079_best_model.pth', weights_only=True))

    return model


def run_exp079_deep_learning_signals():
    """
    Train and validate deep learning signal predictor.
    """
    print("="*70)
    print("EXP-079: DEEP LEARNING SIGNAL PREDICTION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("OBJECTIVE: Use LSTM/GRU to learn temporal patterns in mean reversion signals")
    print("METHODOLOGY: 20-day sequences, multi-task learning (win/loss + return)")
    print("TARGET: AUC >= 0.85 (vs 0.834 XGBoost baseline)")
    print()

    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Get all portfolio tickers
    tickers = get_all_tickers()
    print(f"Preparing data for {len(tickers)} stocks...")
    print()

    # Collect all sequences
    all_sequences = []
    all_labels_clf = []
    all_labels_reg = []
    stock_signal_counts = {}

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Loading {ticker}...", end=" ")

        sequences, labels_clf, labels_reg = prepare_time_series_data(ticker)

        if sequences is not None:
            all_sequences.append(sequences)
            all_labels_clf.append(labels_clf)
            all_labels_reg.append(labels_reg)
            stock_signal_counts[ticker] = len(sequences)
            print(f"{len(sequences)} signals")
        else:
            print("No signals")

    if len(all_sequences) == 0:
        print("\n[ERROR] No training data")
        return None

    # Concatenate all data
    X = np.vstack(all_sequences)
    y_clf = np.concatenate(all_labels_clf)
    y_reg = np.concatenate(all_labels_reg)

    print(f"\nTotal sequences: {len(X)}")
    print(f"Sequence shape: {X.shape}")
    print(f"Positive samples: {y_clf.sum()}/{len(y_clf)} ({y_clf.mean()*100:.1f}%)")
    print()

    # Train/val split
    X_train, X_val, y_clf_train, y_clf_val, y_reg_train, y_reg_val = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print()

    # Create PyTorch datasets
    train_dataset = TimeSeriesDataset(X_train, y_clf_train, y_reg_train)
    val_dataset = TimeSeriesDataset(X_val, y_clf_val, y_reg_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_size = X.shape[2]

    # Train model
    model = train_lstm_model(train_loader, val_loader, input_size, device, epochs=50)

    # Evaluate on validation set
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    print()

    model.eval()
    val_clf_preds = []
    val_reg_preds = []

    with torch.no_grad():
        for sequences, _, _ in val_loader:
            sequences = sequences.to(device)
            clf_output, reg_output = model(sequences)
            val_clf_preds.extend(clf_output.cpu().numpy())
            val_reg_preds.extend(reg_output.cpu().numpy())

    val_clf_preds = np.array(val_clf_preds).flatten()
    val_reg_preds = np.array(val_reg_preds).flatten()

    # Classification metrics
    auc = roc_auc_score(y_clf_val, val_clf_preds)

    # Use 0.5 threshold for binary predictions
    val_clf_binary = (val_clf_preds >= 0.5).astype(int)
    precision = precision_score(y_clf_val, val_clf_binary)
    recall = recall_score(y_clf_val, val_clf_binary)

    # Regression metrics
    mae = mean_absolute_error(y_reg_val, val_reg_preds)

    # Compare to baseline
    baseline_auc = 0.834
    auc_improvement = auc - baseline_auc
    auc_improvement_pct = (auc_improvement / baseline_auc) * 100

    print(f"CLASSIFICATION (Win/Loss Prediction):")
    print(f"  AUC: {auc:.4f} (baseline: {baseline_auc:.4f})")
    print(f"  AUC Improvement: {auc_improvement:+.4f} ({auc_improvement_pct:+.1f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print()

    print(f"REGRESSION (Return Prediction):")
    print(f"  Mean Absolute Error: {mae:.2f}%")
    print()

    # Calculate estimated win rate improvement
    # Baseline: 67.3% (time_decay)
    # Estimated with LSTM filtering (assume 0.6 threshold)
    val_clf_high_confidence = (val_clf_preds >= 0.6).astype(int)
    high_conf_indices = np.where(val_clf_high_confidence == 1)[0]

    if len(high_conf_indices) > 0:
        high_conf_win_rate = y_clf_val[high_conf_indices].mean() * 100
        baseline_win_rate = 67.3
        estimated_improvement = high_conf_win_rate - baseline_win_rate

        print(f"HIGH CONFIDENCE SIGNALS (>60% predicted probability):")
        print(f"  Signals: {len(high_conf_indices)}/{len(y_clf_val)} ({len(high_conf_indices)/len(y_clf_val)*100:.1f}%)")
        print(f"  Win Rate: {high_conf_win_rate:.1f}%")
        print(f"  Baseline Win Rate: {baseline_win_rate:.1f}%")
        print(f"  Improvement: {estimated_improvement:+.1f}pp")
        print()

    # Deployment recommendation
    print("="*70)
    print("DEPLOYMENT RECOMMENDATION")
    print("="*70)
    print()

    if auc >= 0.85:
        print(f"[SUCCESS] Deep learning model VALIDATED!")
        print()
        print(f"RESULTS: AUC {auc:.4f} (target: 0.85, baseline: 0.834)")
        print(f"IMPROVEMENT: {auc_improvement:+.4f} ({auc_improvement_pct:+.1f}%)")
        print()
        print("NEXT STEPS:")
        print("  1. Integrate LSTM predictions into production scanner")
        print("  2. Use 0.6 probability threshold for signal filtering")
        print(f"  3. Monitor win rate improvement (estimated: {estimated_improvement:+.1f}pp)")
        print("  4. Track return prediction accuracy")
        print()
        print(f"ESTIMATED ANNUAL IMPACT: {estimated_improvement * 2.81:.1f}% cumulative (281 trades/year)")
    elif auc >= 0.84:
        print(f"[PARTIAL SUCCESS] AUC {auc:.4f} (target: 0.85, baseline: 0.834)")
        print()
        print("Model shows improvement but below target:")
        print(f"  - AUC improvement: {auc_improvement:+.4f}")
        print(f"  - Consider architecture tuning (more layers, attention)")
        print()
        print("Promising for deployment with refinement")
    else:
        print(f"[NEEDS WORK] AUC {auc:.4f} below baseline {baseline_auc:.4f}")
        print()
        print("Deep learning not capturing patterns effectively")
        print("Consider: more data, feature engineering, or stick with XGBoost")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-079',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'objective': 'Deep learning signal prediction with LSTM',
        'algorithm': 'Multi-task LSTM (classification + regression)',
        'device': str(device),
        'total_sequences': int(len(X)),
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'classification': {
            'auc': float(auc),
            'baseline_auc': float(baseline_auc),
            'improvement': float(auc_improvement),
            'improvement_pct': float(auc_improvement_pct),
            'precision': float(precision),
            'recall': float(recall)
        },
        'regression': {
            'mae': float(mae)
        },
        'high_confidence': {
            'threshold': 0.6,
            'count': int(len(high_conf_indices)) if len(high_conf_indices) > 0 else 0,
            'win_rate': float(high_conf_win_rate) if len(high_conf_indices) > 0 else 0,
            'improvement_pp': float(estimated_improvement) if len(high_conf_indices) > 0 else 0
        },
        'stock_signal_counts': stock_signal_counts,
        'validated': auc >= 0.85,
        'next_step': 'Deploy to production' if auc >= 0.85 else 'Needs refinement'
    }

    results_file = os.path.join(results_dir, 'exp079_deep_learning_signals.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")
    print(f"Model saved to: logs/experiments/exp079_best_model.pth")

    # Send email
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending deep learning signal prediction report...")
            notifier.send_experiment_report('EXP-079', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-079: Deep Learning Signal Prediction."""

    print("\n[DEEP LEARNING] Training LSTM for temporal pattern recognition")
    print("Using GPU acceleration on RTX")
    print()

    results = run_exp079_deep_learning_signals()

    print("\n" + "="*70)
    print("DEEP LEARNING SIGNAL PREDICTION COMPLETE")
    print("="*70)

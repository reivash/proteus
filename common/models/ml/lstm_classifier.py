"""
LSTM neural network for stock market direction prediction.
Uses PyTorch with GPU acceleration for training time series models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler


class StockDataset(Dataset):
    """PyTorch Dataset for stock time series."""

    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMClassifier(nn.Module):
    """
    LSTM-based binary classifier for stock direction prediction.
    """

    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False
    ):
        """
        Initialize LSTM classifier.

        Args:
            input_size: Number of input features
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Dense layers
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass."""
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Take the last output
        last_output = lstm_out[:, -1, :]

        # Dense layers
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out


class LSTMStockPredictor:
    """
    Wrapper class for LSTM stock prediction.
    """

    def __init__(
        self,
        sequence_length=60,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
        learning_rate=0.001,
        batch_size=32,
        epochs=50,
        device=None
    ):
        """
        Initialize LSTM stock predictor.

        Args:
            sequence_length: Number of days to use for prediction
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: Device to use ('cuda' or 'cpu')
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.trained = False

    def create_sequences(self, X, y=None):
        """
        Create sequences for LSTM input.

        Args:
            X: Feature DataFrame
            y: Target Series (optional)

        Returns:
            Sequences and targets (if y provided)
        """
        sequences = []
        targets = []

        for i in range(len(X) - self.sequence_length):
            seq = X.iloc[i:i + self.sequence_length].values
            sequences.append(seq)

            if y is not None:
                target = y.iloc[i + self.sequence_length]
                targets.append(target)

        sequences = np.array(sequences)

        if y is not None:
            targets = np.array(targets)
            return sequences, targets
        else:
            return sequences

    def prepare_data(self, df: pd.DataFrame, target_col='Actual_Direction'):
        """
        Prepare features and target from DataFrame.

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

    def fit(self, X, y, validation_split=0.2, verbose=True):
        """
        Train the LSTM model.

        Args:
            X: Feature DataFrame
            y: Target Series
            validation_split: Fraction of data to use for validation
            verbose: Whether to print training progress

        Returns:
            self
        """
        # Normalize features
        X_normalized = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Create sequences
        sequences, targets = self.create_sequences(X_normalized, y)

        # Split data
        split_idx = int(len(sequences) * (1 - validation_split))
        X_train_seq, X_val_seq = sequences[:split_idx], sequences[split_idx:]
        y_train, y_val = targets[:split_idx], targets[split_idx:]

        if verbose:
            print(f"Training sequences: {len(X_train_seq)}")
            print(f"Validation sequences: {len(X_val_seq)}")
            print(f"Sequence shape: {X_train_seq.shape}")

        # Create datasets
        train_dataset = StockDataset(X_train_seq, y_train)
        val_dataset = StockDataset(X_val_seq, y_val)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize model
        input_size = X.shape[1]
        self.model = LSTMClassifier(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device).unsqueeze(1)

                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        self.trained = True
        if verbose:
            print("Training complete!")

        return self

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions (0 or 1)
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()

        # Normalize
        X_normalized = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

        # Create sequences
        sequences = self.create_sequences(X_normalized)

        # Create dataset
        dataset = StockDataset(sequences, np.zeros(len(sequences)))  # Dummy targets
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                preds = (outputs >= 0.5).float().cpu().numpy().flatten()
                predictions.extend(preds)

        return np.array(predictions)

    def predict_proba(self, X):
        """
        Predict probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Array of probabilities
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()

        # Normalize
        X_normalized = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

        # Create sequences
        sequences = self.create_sequences(X_normalized)

        # Create dataset
        dataset = StockDataset(sequences, np.zeros(len(sequences)))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        probabilities = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probs = outputs.cpu().numpy().flatten()
                probabilities.extend(probs)

        return np.array(probabilities)

    def evaluate(self, X, y, verbose=True):
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

        # Adjust y to match predictions length (account for sequence length)
        y_adjusted = y.iloc[self.sequence_length:].values

        # Calculate metrics
        accuracy = (predictions == y_adjusted).mean() * 100

        # Confusion matrix
        tp = ((predictions == 1) & (y_adjusted == 1)).sum()
        tn = ((predictions == 0) & (y_adjusted == 0)).sum()
        fp = ((predictions == 1) & (y_adjusted == 0)).sum()
        fn = ((predictions == 0) & (y_adjusted == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': int(len(predictions)),
            'up_predicted': int((predictions == 1).sum()),
            'down_predicted': int((predictions == 0).sum()),
            'actual_up': int((y_adjusted == 1).sum()),
            'actual_down': int((y_adjusted == 0).sum())
        }

        if verbose:
            print("\n" + "=" * 70)
            print("LSTM MODEL EVALUATION RESULTS")
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
            print("=" * 70)

        return results


if __name__ == "__main__":
    # Quick test
    print("LSTM Classifier module loaded successfully")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

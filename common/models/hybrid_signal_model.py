"""
Hybrid Signal Model - 3-Model Ensemble (Transformer + LSTM + MLP)
=================================================================

BACKTEST VALIDATED (Jan 6, 2026):
---------------------------------
| Model        | Trades | Win Rate | Avg Ret | Sharpe |
|--------------|--------|----------|---------|--------|
| Ensemble-2+  |   199  |   63.8%  | +0.86%  |  0.28  | <- BEST
| LSTM@40      |   434  |   60.4%  | +0.74%  |  0.23  |
| MLP@40       |   229  |   60.7%  | +0.64%  |  0.21  |
| Transformer  |   100  |   56.0%  | +0.38%  |  0.14  |
| Ensemble-3   |    19  |   52.6%  | +0.41%  |  0.11  |

Strategy: 3-model weighted ensemble with 2+ vote consensus
- Require 2/3 models to agree (optimal selectivity)
- Weighted average of signal strengths
- 10% boost for 2-vote, 20% boost for 3-vote consensus
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import yfinance as yf
except ImportError:
    yf = None


@dataclass
class HybridSignal:
    """Signal from 3-model ensemble."""
    ticker: str
    timestamp: str
    mean_reversion_prob: float
    expected_return: float
    confidence: float
    signal_strength: float
    source: str  # 'TRANSFORMER', 'LSTM', 'MLP', or 'ENSEMBLE'
    transformer_strength: float
    lstm_strength: float
    mlp_strength: float
    votes: int  # Number of models agreeing (1, 2, or 3)
    consensus: bool  # True if 2+ models agree


class HybridSignalModel:
    """
    3-Model Ensemble combining Transformer + LSTM + MLP.

    Voting logic:
    1. Get predictions from all 3 models
    2. Count how many models exceed their thresholds (votes)
    3. Calculate weighted ensemble strength
    4. Boost signals with consensus (2+ votes)

    Based on overnight backtest (Jan 5-6, 2026):
    - All 3 models have similar performance (0.24-0.26 Sharpe)
    - Ensemble should improve consistency through voting
    """

    PROTEUS_TICKERS = [
        "NVDA", "V", "MA", "AVGO", "AXP", "KLAC", "ORCL", "MRVL", "ABBV", "SYK",
        "EOG", "TXN", "GILD", "INTU", "MSFT", "QCOM", "JPM", "JNJ", "PFE", "WMT",
        "AMAT", "ADI", "NOW", "MLM", "IDXX", "EXR", "ROAD", "INSM", "SCHW", "AIG",
        "USB", "CVS", "LOW", "LMT", "COP", "SLB", "APD", "MS", "PNC", "CRM",
        "ADBE", "TGT", "CAT", "XOM", "MPC", "ECL", "NEE", "HCA", "CMCSA", "TMUS",
        "META", "ETN", "HD", "SHW"
    ]

    # Thresholds based on overnight backtest results
    TRANSFORMER_THRESHOLD = 50.0  # Transformer @ 50: 60% win, 0.26 Sharpe
    LSTM_THRESHOLD = 40.0         # LSTM @ 40: 60% win, 0.24 Sharpe
    MLP_THRESHOLD = 40.0          # MLP @ 40: 62.8% win, 0.26 Sharpe

    # Weights based on Sharpe ratios (normalized)
    TRANSFORMER_WEIGHT = 0.34  # 0.26 / 0.76
    LSTM_WEIGHT = 0.32         # 0.24 / 0.76
    MLP_WEIGHT = 0.34          # 0.26 / 0.76

    # Consensus boosts
    CONSENSUS_2_BOOST = 1.10   # 10% boost for 2/3 agreement
    CONSENSUS_3_BOOST = 1.20   # 20% boost for 3/3 agreement

    # Minimum votes required (backtest shows 2+ is optimal)
    MIN_VOTES = 2  # Require at least 2/3 models to agree

    def __init__(self):
        """Initialize hybrid model with all 3 models."""
        self.transformer_model = None
        self.lstm_model = None
        self.mlp_model = None
        self._load_models()

    def _load_models(self):
        """Load all 3 models."""
        print("\n[HYBRID] Loading 3-model ensemble...")

        # Load Transformer
        try:
            from common.models.transformer_signal_model import TransformerSignalModel
            self.transformer_model = TransformerSignalModel()
            print("[HYBRID] Transformer model loaded")
        except Exception as e:
            print(f"[HYBRID] Transformer load failed: {e}")

        # Load LSTM
        try:
            from common.models.lstm_signal_model import LSTMSignalModel
            self.lstm_model = LSTMSignalModel()
            print("[HYBRID] LSTM model loaded")
        except Exception as e:
            print(f"[HYBRID] LSTM load failed: {e}")

        # Load MLP
        try:
            from common.models.gpu_signal_model import GPUSignalModel
            self.mlp_model = GPUSignalModel()
            print("[HYBRID] MLP model loaded")
        except Exception as e:
            print(f"[HYBRID] MLP load failed: {e}")

        # Need at least one model
        models_loaded = sum([
            self.transformer_model is not None,
            self.lstm_model is not None,
            self.mlp_model is not None
        ])
        print(f"[HYBRID] {models_loaded}/3 models loaded")

        if models_loaded == 0:
            raise RuntimeError("No models available!")

    def predict(self, ticker: str, df: pd.DataFrame = None) -> Optional[HybridSignal]:
        """
        Generate ensemble signal using 3-model voting.

        Logic:
        1. Get predictions from all available models
        2. Count votes (models above threshold)
        3. Calculate weighted ensemble strength
        4. Apply consensus boost
        """
        if df is None:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period="6mo")
            except:
                return None

        if len(df) < 90:
            return None

        # Get predictions from all models
        transformer_signal = None
        lstm_signal = None
        mlp_signal = None
        transformer_strength = 0.0
        lstm_strength = 0.0
        mlp_strength = 0.0

        # Transformer prediction
        if self.transformer_model:
            try:
                transformer_signal = self.transformer_model.predict(ticker, df)
                if transformer_signal:
                    transformer_strength = transformer_signal.signal_strength
            except:
                pass

        # LSTM prediction
        if self.lstm_model:
            try:
                lstm_signal = self.lstm_model.predict(ticker, df)
                if lstm_signal:
                    lstm_strength = lstm_signal.signal_strength
            except:
                pass

        # MLP prediction
        if self.mlp_model:
            try:
                mlp_signal = self.mlp_model.predict(ticker, df)
                if mlp_signal:
                    mlp_strength = mlp_signal.signal_strength
            except:
                pass

        # Need at least one signal
        if not any([transformer_signal, lstm_signal, mlp_signal]):
            return None

        # Count votes (models above threshold)
        transformer_vote = transformer_strength >= self.TRANSFORMER_THRESHOLD
        lstm_vote = lstm_strength >= self.LSTM_THRESHOLD
        mlp_vote = mlp_strength >= self.MLP_THRESHOLD
        votes = sum([transformer_vote, lstm_vote, mlp_vote])

        # Calculate weighted ensemble strength
        total_weight = 0.0
        weighted_strength = 0.0

        if transformer_signal:
            weighted_strength += transformer_strength * self.TRANSFORMER_WEIGHT
            total_weight += self.TRANSFORMER_WEIGHT

        if lstm_signal:
            weighted_strength += lstm_strength * self.LSTM_WEIGHT
            total_weight += self.LSTM_WEIGHT

        if mlp_signal:
            weighted_strength += mlp_strength * self.MLP_WEIGHT
            total_weight += self.MLP_WEIGHT

        if total_weight > 0:
            ensemble_strength = weighted_strength / total_weight
        else:
            return None

        # Apply consensus boost
        if votes >= 3:
            ensemble_strength = min(100.0, ensemble_strength * self.CONSENSUS_3_BOOST)
            source = 'ENSEMBLE-3'
        elif votes >= 2:
            ensemble_strength = min(100.0, ensemble_strength * self.CONSENSUS_2_BOOST)
            source = 'ENSEMBLE-2'
        elif votes == 1:
            # Single model active - use that model's name
            if transformer_vote:
                source = 'TRANSFORMER'
            elif lstm_vote:
                source = 'LSTM'
            else:
                source = 'MLP'
        else:
            # No model above threshold - use strongest
            source = 'WEAK'

        # Get primary signal for probability/return (use strongest model)
        primary = None
        max_strength = max(transformer_strength, lstm_strength, mlp_strength)
        if transformer_strength == max_strength and transformer_signal:
            primary = transformer_signal
        elif lstm_strength == max_strength and lstm_signal:
            primary = lstm_signal
        elif mlp_signal:
            primary = mlp_signal
        else:
            primary = transformer_signal or lstm_signal or mlp_signal

        return HybridSignal(
            ticker=ticker,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            mean_reversion_prob=primary.mean_reversion_prob,
            expected_return=primary.expected_return,
            confidence=primary.confidence,
            signal_strength=round(ensemble_strength, 1),
            source=source,
            transformer_strength=round(transformer_strength, 1),
            lstm_strength=round(lstm_strength, 1),
            mlp_strength=round(mlp_strength, 1),
            votes=votes,
            consensus=votes >= 2
        )

    def scan_all(self, threshold: float = 50.0, min_votes: int = None) -> List[HybridSignal]:
        """
        Scan all tickers and return signals above threshold.

        Args:
            threshold: Minimum signal strength
            min_votes: Minimum votes required (default: self.MIN_VOTES = 2)

        Returns:
            List of signals sorted by strength descending
        """
        if min_votes is None:
            min_votes = self.MIN_VOTES

        signals = []

        print(f"\n[ENSEMBLE] Scanning {len(self.PROTEUS_TICKERS)} tickers (min_votes={min_votes})...")

        for ticker in self.PROTEUS_TICKERS:
            try:
                signal = self.predict(ticker)
                if signal and signal.signal_strength >= threshold and signal.votes >= min_votes:
                    signals.append(signal)
                    # Icon based on votes
                    if signal.votes == 3:
                        icon = "+++"
                    elif signal.votes == 2:
                        icon = "++"
                    else:
                        icon = "+"
                    print(f"  {icon} {ticker}: {signal.signal_strength:.1f} ({signal.source}) v={signal.votes}")
            except Exception as e:
                pass

        # Sort by strength
        signals.sort(key=lambda x: x.signal_strength, reverse=True)

        print(f"\n[ENSEMBLE] Found {len(signals)} signals (threshold={threshold}, min_votes={min_votes})")
        ensemble3 = sum(1 for s in signals if s.votes == 3)
        ensemble2 = sum(1 for s in signals if s.votes == 2)
        print(f"  3-vote: {ensemble3}, 2-vote: {ensemble2}")

        return signals

    def get_top_signals(self, n: int = 5, threshold: float = 50.0, min_votes: int = None) -> List[HybridSignal]:
        """Get top N signals with minimum vote requirement."""
        all_signals = self.scan_all(threshold, min_votes)
        return all_signals[:n]


def test_hybrid_model():
    """Test 3-model ensemble."""
    print("\n" + "="*70)
    print("3-MODEL ENSEMBLE TEST (Transformer + LSTM + MLP)")
    print("="*70)

    model = HybridSignalModel()

    # Test single prediction
    print("\n--- Single Prediction Test ---")
    signal = model.predict("NVDA")
    if signal:
        print(f"NVDA: {signal.signal_strength:.1f} ({signal.source})")
        print(f"  Transformer: {signal.transformer_strength:.1f}")
        print(f"  LSTM: {signal.lstm_strength:.1f}")
        print(f"  MLP: {signal.mlp_strength:.1f}")
        print(f"  Votes: {signal.votes}/3, Consensus: {signal.consensus}")

    # Test scan
    print("\n--- Full Scan Test ---")
    top = model.get_top_signals(n=10, threshold=40.0)

    print("\n[TOP 10 SIGNALS]")
    print(f"{'Ticker':<8} {'Strength':>8} {'Source':<12} {'Trans':>6} {'LSTM':>6} {'MLP':>6} {'Votes':<6}")
    print("-"*60)
    for s in top:
        print(f"{s.ticker:<8} {s.signal_strength:>8.1f} {s.source:<12} {s.transformer_strength:>6.1f} {s.lstm_strength:>6.1f} {s.mlp_strength:>6.1f} {s.votes}/3")


if __name__ == "__main__":
    test_hybrid_model()

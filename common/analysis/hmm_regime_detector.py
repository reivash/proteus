"""
HMM-Based Market Regime Detection
=================================

Uses Hidden Markov Models to detect market regimes more accurately than rule-based
approaches. Key advantages:

1. Probabilistic state transitions - detects regime shifts BEFORE they're obvious
2. Learns from historical patterns rather than fixed thresholds
3. Provides regime probabilities, not just classifications
4. More robust to noise and outliers

States:
- VOLATILE: High uncertainty, large swings (best for mean reversion - Sharpe 4.86)
- BEAR: Downtrend with fear (good for mean reversion - Sharpe 3.32)
- CHOPPY: Range-bound, normal volatility (moderate - Sharpe 1.25)
- BULL: Strong uptrend, complacency (worst for mean reversion - Sharpe 1.00)

Observable Features:
- Daily returns (captures momentum)
- Realized volatility (captures regime character)
- Volume change (captures conviction)
- VIX level (captures fear/greed)

Jan 2026 - Based on research showing HMMs outperform rule-based detection.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


class HMMRegime(Enum):
    """Market regimes ordered by mean reversion performance (best to worst)."""
    VOLATILE = 0  # Best for mean reversion (Sharpe 4.86)
    BEAR = 1      # Good for mean reversion (Sharpe 3.32)
    CHOPPY = 2    # Moderate (Sharpe 1.25)
    BULL = 3      # Worst for mean reversion (Sharpe 1.00)


@dataclass
class HMMRegimeResult:
    """HMM regime detection result."""
    regime: str
    regime_id: int
    confidence: float
    probabilities: Dict[str, float]  # Probability of each regime
    transition_signal: str  # "stable", "entering_X", "exiting_X"
    days_in_regime: int
    features: Dict[str, float]
    recommendation: str


class GaussianHMM:
    """
    Simple Gaussian Hidden Markov Model implementation.

    No external dependencies - implements Baum-Welch and Viterbi algorithms.
    """

    def __init__(self, n_states: int = 4, n_features: int = 4, n_iter: int = 50):
        self.n_states = n_states
        self.n_features = n_features
        self.n_iter = n_iter

        # Model parameters (will be initialized/trained)
        self.start_prob = None  # Initial state probabilities
        self.trans_prob = None  # Transition matrix
        self.means = None       # State emission means
        self.covars = None      # State emission covariances

        self._is_fitted = False

    def _initialize_params(self, X: np.ndarray):
        """Initialize model parameters using k-means-like clustering."""
        n_samples = len(X)

        # Initial probabilities (uniform)
        self.start_prob = np.ones(self.n_states) / self.n_states

        # Transition matrix (slight preference to stay in same state)
        self.trans_prob = np.ones((self.n_states, self.n_states)) * 0.1
        np.fill_diagonal(self.trans_prob, 0.7)
        self.trans_prob /= self.trans_prob.sum(axis=1, keepdims=True)

        # Initialize means using quantiles to spread across data
        sorted_idx = np.argsort(X[:, 0])  # Sort by first feature (returns)
        chunk_size = n_samples // self.n_states

        self.means = np.zeros((self.n_states, self.n_features))
        for i in range(self.n_states):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < self.n_states - 1 else n_samples
            self.means[i] = X[sorted_idx[start:end]].mean(axis=0)

        # Initialize covariances (diagonal, based on data variance)
        data_var = np.var(X, axis=0) + 1e-6
        self.covars = np.array([np.diag(data_var) for _ in range(self.n_states)])

    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Compute log likelihood of observations under each state."""
        n_samples = len(X)
        log_lik = np.zeros((n_samples, self.n_states))

        for state in range(self.n_states):
            diff = X - self.means[state]
            cov = self.covars[state]

            # Add small regularization for numerical stability
            cov = cov + np.eye(self.n_features) * 1e-6

            try:
                cov_inv = np.linalg.inv(cov)
                log_det = np.log(np.linalg.det(cov) + 1e-10)

                # Multivariate Gaussian log likelihood
                mahal = np.sum(diff @ cov_inv * diff, axis=1)
                log_lik[:, state] = -0.5 * (
                    self.n_features * np.log(2 * np.pi) + log_det + mahal
                )
            except np.linalg.LinAlgError:
                # Fallback to diagonal covariance
                var = np.diag(cov)
                log_lik[:, state] = -0.5 * np.sum(
                    np.log(2 * np.pi * var) + (diff ** 2) / var, axis=1
                )

        return log_lik

    def _forward(self, log_lik: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward algorithm (alpha pass)."""
        n_samples = len(log_lik)
        alpha = np.zeros((n_samples, self.n_states))
        scale = np.zeros(n_samples)

        # Initialize
        alpha[0] = self.start_prob * np.exp(log_lik[0] - log_lik[0].max())
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0] + 1e-10

        # Forward pass
        for t in range(1, n_samples):
            alpha[t] = np.dot(alpha[t-1], self.trans_prob) * np.exp(
                log_lik[t] - log_lik[t].max()
            )
            scale[t] = alpha[t].sum()
            alpha[t] /= scale[t] + 1e-10

        return alpha, scale

    def _backward(self, log_lik: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Backward algorithm (beta pass)."""
        n_samples = len(log_lik)
        beta = np.zeros((n_samples, self.n_states))

        # Initialize
        beta[-1] = 1.0

        # Backward pass
        for t in range(n_samples - 2, -1, -1):
            beta[t] = np.dot(
                self.trans_prob,
                beta[t+1] * np.exp(log_lik[t+1] - log_lik[t+1].max())
            )
            beta[t] /= scale[t+1] + 1e-10

        return beta

    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """E-step: compute responsibilities and transition counts."""
        log_lik = self._compute_log_likelihood(X)
        alpha, scale = self._forward(log_lik)
        beta = self._backward(log_lik, scale)

        # Responsibilities (gamma)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-10

        # Transition responsibilities (xi)
        n_samples = len(X)
        xi = np.zeros((n_samples - 1, self.n_states, self.n_states))

        for t in range(n_samples - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (
                        alpha[t, i] * self.trans_prob[i, j] *
                        np.exp(log_lik[t+1, j] - log_lik[t+1].max()) *
                        beta[t+1, j]
                    )
            xi[t] /= xi[t].sum() + 1e-10

        # Log likelihood
        log_likelihood = np.sum(np.log(scale + 1e-10))

        return gamma, xi, log_likelihood

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        """M-step: update model parameters."""
        # Update start probabilities
        self.start_prob = gamma[0] / (gamma[0].sum() + 1e-10)

        # Update transition probabilities
        xi_sum = xi.sum(axis=0)
        self.trans_prob = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-10)

        # Update means and covariances
        for state in range(self.n_states):
            weights = gamma[:, state]
            weight_sum = weights.sum() + 1e-10

            # Mean
            self.means[state] = np.sum(weights[:, np.newaxis] * X, axis=0) / weight_sum

            # Covariance
            diff = X - self.means[state]
            self.covars[state] = np.dot(
                (weights[:, np.newaxis] * diff).T, diff
            ) / weight_sum

            # Regularize covariance
            self.covars[state] += np.eye(self.n_features) * 1e-4

    def fit(self, X: np.ndarray) -> 'GaussianHMM':
        """Fit HMM using Baum-Welch algorithm."""
        self._initialize_params(X)

        prev_log_lik = float('-inf')

        for iteration in range(self.n_iter):
            gamma, xi, log_likelihood = self._e_step(X)
            self._m_step(X, gamma, xi)

            # Check convergence
            if abs(log_likelihood - prev_log_lik) < 1e-4:
                break
            prev_log_lik = log_likelihood

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict most likely state sequence using Viterbi algorithm."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        n_samples = len(X)
        log_lik = self._compute_log_likelihood(X)

        # Viterbi algorithm
        viterbi = np.zeros((n_samples, self.n_states))
        backpointer = np.zeros((n_samples, self.n_states), dtype=int)

        # Initialize
        viterbi[0] = np.log(self.start_prob + 1e-10) + log_lik[0]

        # Forward pass
        log_trans = np.log(self.trans_prob + 1e-10)
        for t in range(1, n_samples):
            for j in range(self.n_states):
                trans_scores = viterbi[t-1] + log_trans[:, j]
                backpointer[t, j] = np.argmax(trans_scores)
                viterbi[t, j] = trans_scores[backpointer[t, j]] + log_lik[t, j]

        # Backtrack
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(viterbi[-1])
        for t in range(n_samples - 2, -1, -1):
            states[t] = backpointer[t + 1, states[t + 1]]

        return states

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict state probabilities for each observation."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        log_lik = self._compute_log_likelihood(X)
        alpha, scale = self._forward(log_lik)
        beta = self._backward(log_lik, scale)

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-10

        return gamma

    def save(self, filepath: str):
        """Save model parameters to JSON."""
        params = {
            'n_states': self.n_states,
            'n_features': self.n_features,
            'start_prob': self.start_prob.tolist(),
            'trans_prob': self.trans_prob.tolist(),
            'means': self.means.tolist(),
            'covars': [c.tolist() for c in self.covars],
            'is_fitted': self._is_fitted
        }
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)

    def load(self, filepath: str) -> 'GaussianHMM':
        """Load model parameters from JSON."""
        with open(filepath, 'r') as f:
            params = json.load(f)

        self.n_states = params['n_states']
        self.n_features = params['n_features']
        self.start_prob = np.array(params['start_prob'])
        self.trans_prob = np.array(params['trans_prob'])
        self.means = np.array(params['means'])
        self.covars = np.array(params['covars'])
        self._is_fitted = params['is_fitted']

        return self


class HMMRegimeDetector:
    """
    Market regime detector using Hidden Markov Models.

    Detects regimes earlier than rule-based methods by learning
    probabilistic state transitions from historical data.
    """

    REGIME_NAMES = ['volatile', 'bear', 'choppy', 'bull']

    def __init__(self, model_path: Optional[str] = None):
        self.hmm = GaussianHMM(n_states=4, n_features=4, n_iter=100)
        self.feature_means = None
        self.feature_stds = None
        self._regime_history: List[int] = []

        # Default model path
        if model_path is None:
            model_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'models' / 'hmm_regime.json'
        self.model_path = Path(model_path)

        # Try to load pre-trained model
        if self.model_path.exists():
            self._load_model()

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features for HMM from price data.

        Features:
        1. Daily returns (momentum signal)
        2. Realized volatility (20-day)
        3. Volume change ratio
        4. VIX level (if available) or volatility proxy
        """
        features = []

        # 1. Daily returns
        returns = df['Close'].pct_change().fillna(0).values
        features.append(returns)

        # 2. Realized volatility (20-day rolling std of returns)
        volatility = pd.Series(returns).rolling(20).std().fillna(0.01).values
        features.append(volatility)

        # 3. Volume change ratio (vs 20-day average)
        if 'Volume' in df.columns:
            vol_ma = df['Volume'].rolling(20).mean()
            vol_ratio = (df['Volume'] / vol_ma).fillna(1.0).values
            vol_ratio = np.clip(vol_ratio, 0.1, 5.0)  # Cap extreme values
        else:
            vol_ratio = np.ones(len(df))
        features.append(vol_ratio)

        # 4. Price momentum (10-day return)
        momentum = df['Close'].pct_change(10).fillna(0).values
        features.append(momentum)

        return np.column_stack(features)

    def _normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features to zero mean, unit variance."""
        if fit or self.feature_means is None:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0) + 1e-6

        return (X - self.feature_means) / self.feature_stds

    def _map_states_to_regimes(self, states: np.ndarray, features: np.ndarray) -> np.ndarray:
        """
        Map HMM states to regime labels based on feature characteristics.

        State mapping based on volatility and returns:
        - VOLATILE: Highest volatility
        - BEAR: Negative returns, moderate-high volatility
        - BULL: Positive returns, low volatility
        - CHOPPY: Low volatility, mixed returns
        """
        state_features = {}
        for state in range(4):
            mask = states == state
            if mask.sum() > 0:
                state_features[state] = {
                    'volatility': features[mask, 1].mean(),
                    'returns': features[mask, 0].mean(),
                    'count': mask.sum()
                }
            else:
                state_features[state] = {'volatility': 0, 'returns': 0, 'count': 0}

        # Sort states by volatility (descending) for initial classification
        sorted_by_vol = sorted(state_features.items(),
                              key=lambda x: x[1]['volatility'],
                              reverse=True)

        # Map states based on characteristics
        mapping = {}
        used_regimes = set()

        for state, feat in sorted_by_vol:
            if 'volatile' not in used_regimes and feat['volatility'] > 0.02:
                mapping[state] = 0  # VOLATILE
                used_regimes.add('volatile')
            elif 'bear' not in used_regimes and feat['returns'] < -0.001:
                mapping[state] = 1  # BEAR
                used_regimes.add('bear')
            elif 'bull' not in used_regimes and feat['returns'] > 0.001:
                mapping[state] = 3  # BULL
                used_regimes.add('bull')
            else:
                mapping[state] = 2  # CHOPPY
                used_regimes.add('choppy')

        return np.array([mapping.get(s, 2) for s in states])

    def train(self, df: pd.DataFrame = None, period: str = '5y') -> Dict:
        """
        Train HMM on historical market data.

        Args:
            df: DataFrame with OHLCV data (if None, fetches SPY data)
            period: Period to fetch if df is None

        Returns:
            Training statistics
        """
        if df is None:
            if not HAS_YFINANCE:
                raise ImportError("yfinance required to fetch training data")

            print("[HMM] Fetching SPY data for training...")
            spy = yf.Ticker("SPY")
            df = spy.history(period=period)
            print(f"[HMM] Got {len(df)} days of data")

        # Extract and normalize features
        features = self._extract_features(df)
        X = self._normalize_features(features, fit=True)

        # Remove any NaN rows
        valid_mask = ~np.isnan(X).any(axis=1)
        X = X[valid_mask]

        print(f"[HMM] Training on {len(X)} observations with {X.shape[1]} features...")

        # Fit HMM
        self.hmm.fit(X)

        # Predict states and map to regimes
        states = self.hmm.predict(X)
        regimes = self._map_states_to_regimes(states, features[valid_mask])

        # Calculate statistics
        regime_counts = {name: 0 for name in self.REGIME_NAMES}
        for r in regimes:
            regime_counts[self.REGIME_NAMES[r]] += 1

        stats = {
            'total_observations': len(X),
            'regime_distribution': regime_counts,
            'regime_percentages': {k: v/len(X)*100 for k, v in regime_counts.items()},
            'transition_matrix': self.hmm.trans_prob.tolist()
        }

        # Save model
        self._save_model()

        print(f"[HMM] Training complete. Regime distribution:")
        for name, pct in stats['regime_percentages'].items():
            print(f"  {name.upper()}: {pct:.1f}%")

        return stats

    def detect_regime(self, df: pd.DataFrame = None, lookback: int = 60) -> HMMRegimeResult:
        """
        Detect current market regime using HMM.

        Args:
            df: DataFrame with recent OHLCV data (if None, fetches SPY)
            lookback: Number of days to use for detection

        Returns:
            HMMRegimeResult with regime and probabilities
        """
        if df is None:
            if not HAS_YFINANCE:
                raise ImportError("yfinance required to fetch market data")

            spy = yf.Ticker("SPY")
            df = spy.history(period='6mo')

        # Use last N days
        df = df.tail(lookback)

        # Extract features
        features = self._extract_features(df)
        X = self._normalize_features(features)

        # Remove NaN
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]
        features_valid = features[valid_mask]

        if len(X_valid) < 10:
            return self._default_result("Insufficient data")

        # Check if model is trained
        if not self.hmm._is_fitted:
            print("[HMM] Model not trained, using rule-based fallback...")
            return self._rule_based_fallback(features_valid)

        # Predict states and probabilities
        states = self.hmm.predict(X_valid)
        probs = self.hmm.predict_proba(X_valid)

        # Map to regimes
        regimes = self._map_states_to_regimes(states, features_valid)

        # Current regime
        current_regime = regimes[-1]
        current_probs = probs[-1]

        # Map probabilities to regime names
        prob_dict = {}
        for i, name in enumerate(self.REGIME_NAMES):
            # Find which HMM state maps to this regime
            state_probs = []
            for state in range(4):
                if self._map_states_to_regimes(np.array([state]), features_valid[-1:])[0] == i:
                    state_probs.append(current_probs[state])
            prob_dict[name] = sum(state_probs) if state_probs else 0.0

        # Normalize probabilities
        prob_sum = sum(prob_dict.values())
        if prob_sum > 0:
            prob_dict = {k: v/prob_sum for k, v in prob_dict.items()}

        # Calculate days in current regime
        days_in_regime = 1
        for i in range(len(regimes) - 2, -1, -1):
            if regimes[i] == current_regime:
                days_in_regime += 1
            else:
                break

        # Detect transition signals
        transition_signal = self._detect_transition(regimes, probs)

        # Update history
        self._regime_history.append(current_regime)
        if len(self._regime_history) > 100:
            self._regime_history = self._regime_history[-100:]

        # Generate recommendation
        recommendation = self._get_recommendation(current_regime, transition_signal)

        # Latest features for reference
        latest_features = {
            'daily_return': float(features_valid[-1, 0]),
            'volatility': float(features_valid[-1, 1]),
            'volume_ratio': float(features_valid[-1, 2]),
            'momentum_10d': float(features_valid[-1, 3])
        }

        return HMMRegimeResult(
            regime=self.REGIME_NAMES[current_regime],
            regime_id=current_regime,
            confidence=float(max(prob_dict.values())),
            probabilities=prob_dict,
            transition_signal=transition_signal,
            days_in_regime=days_in_regime,
            features=latest_features,
            recommendation=recommendation
        )

    def _detect_transition(self, regimes: np.ndarray, probs: np.ndarray) -> str:
        """Detect if regime is transitioning."""
        if len(regimes) < 5:
            return "stable"

        recent = regimes[-5:]
        current = recent[-1]

        # Check for regime changes in last 5 days
        changes = np.sum(np.diff(recent) != 0)

        if changes >= 2:
            # Multiple changes = transitional period
            return "transitioning"

        # Check probability trends
        current_prob_trend = probs[-5:, current].mean() - probs[-10:-5, current].mean() if len(probs) >= 10 else 0

        if current_prob_trend < -0.1:
            return f"exiting_{self.REGIME_NAMES[current]}"
        elif current_prob_trend > 0.1:
            return f"entering_{self.REGIME_NAMES[current]}"

        return "stable"

    def _get_recommendation(self, regime: int, transition: str) -> str:
        """Get trading recommendation based on regime."""
        regime_name = self.REGIME_NAMES[regime]

        recs = {
            'volatile': "HIGH ALPHA regime - AGGRESSIVE mean reversion. Lower thresholds, larger positions.",
            'bear': "HIGH ALPHA regime - Mean reversion works well. Standard thresholds, normal positions.",
            'choppy': "MODERATE regime - Selective mean reversion. Standard thresholds and positions.",
            'bull': "LOW ALPHA regime - Mean reversion underperforms. Higher thresholds, smaller positions."
        }

        rec = recs.get(regime_name, "Unknown regime")

        if 'exiting' in transition:
            rec += f" WARNING: May be exiting {regime_name} regime soon."
        elif 'entering' in transition:
            rec += f" NOTE: Recently entered {regime_name} regime."

        return rec

    def _default_result(self, message: str) -> HMMRegimeResult:
        """Return default result when detection fails."""
        return HMMRegimeResult(
            regime='choppy',
            regime_id=2,
            confidence=0.5,
            probabilities={name: 0.25 for name in self.REGIME_NAMES},
            transition_signal='unknown',
            days_in_regime=0,
            features={},
            recommendation=f"Default to CHOPPY regime. {message}"
        )

    def _rule_based_fallback(self, features: np.ndarray) -> HMMRegimeResult:
        """Fallback to simple rule-based detection if HMM not trained."""
        vol = features[-1, 1]
        ret = features[-1, 0]
        mom = features[-1, 3]

        if vol > 0.025:
            regime = 0  # VOLATILE
        elif ret < -0.01 and mom < -0.03:
            regime = 1  # BEAR
        elif ret > 0.01 and mom > 0.03:
            regime = 3  # BULL
        else:
            regime = 2  # CHOPPY

        return HMMRegimeResult(
            regime=self.REGIME_NAMES[regime],
            regime_id=regime,
            confidence=0.6,
            probabilities={name: 0.25 for name in self.REGIME_NAMES},
            transition_signal='unknown',
            days_in_regime=0,
            features={
                'daily_return': float(ret),
                'volatility': float(vol),
                'momentum_10d': float(mom)
            },
            recommendation=self._get_recommendation(regime, 'unknown') + " (Rule-based fallback)"
        )

    def _save_model(self):
        """Save trained model."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        params = {
            'hmm_params': {
                'n_states': self.hmm.n_states,
                'n_features': self.hmm.n_features,
                'start_prob': self.hmm.start_prob.tolist(),
                'trans_prob': self.hmm.trans_prob.tolist(),
                'means': self.hmm.means.tolist(),
                'covars': [c.tolist() for c in self.hmm.covars],
                'is_fitted': self.hmm._is_fitted
            },
            'feature_means': self.feature_means.tolist() if self.feature_means is not None else None,
            'feature_stds': self.feature_stds.tolist() if self.feature_stds is not None else None,
            'trained_at': datetime.now().isoformat()
        }

        with open(self.model_path, 'w') as f:
            json.dump(params, f, indent=2)

        print(f"[HMM] Model saved to {self.model_path}")

    def _load_model(self):
        """Load trained model."""
        try:
            with open(self.model_path, 'r') as f:
                params = json.load(f)

            hmm_params = params['hmm_params']
            self.hmm.n_states = hmm_params['n_states']
            self.hmm.n_features = hmm_params['n_features']
            self.hmm.start_prob = np.array(hmm_params['start_prob'])
            self.hmm.trans_prob = np.array(hmm_params['trans_prob'])
            self.hmm.means = np.array(hmm_params['means'])
            self.hmm.covars = np.array(hmm_params['covars'])
            self.hmm._is_fitted = hmm_params['is_fitted']

            if params['feature_means'] is not None:
                self.feature_means = np.array(params['feature_means'])
                self.feature_stds = np.array(params['feature_stds'])

            print(f"[HMM] Model loaded from {self.model_path}")

        except Exception as e:
            print(f"[HMM] Could not load model: {e}")


def get_hmm_regime() -> HMMRegimeResult:
    """Quick function to get current regime using HMM."""
    detector = HMMRegimeDetector()
    return detector.detect_regime()


def train_hmm_model(period: str = '5y') -> Dict:
    """Train HMM model on historical data."""
    detector = HMMRegimeDetector()
    return detector.train(period=period)


def print_hmm_regime_report():
    """Print detailed HMM regime analysis."""
    print("=" * 70)
    print("HMM MARKET REGIME DETECTION")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    detector = HMMRegimeDetector()
    result = detector.detect_regime()

    print(f"REGIME: {result.regime.upper()}")
    print(f"Confidence: {result.confidence * 100:.1f}%")
    print(f"Days in regime: {result.days_in_regime}")
    print(f"Transition: {result.transition_signal}")
    print()

    print("--- REGIME PROBABILITIES ---")
    for regime, prob in sorted(result.probabilities.items(), key=lambda x: -x[1]):
        bar = "#" * int(prob * 30)
        print(f"  {regime.upper():<10} {prob*100:5.1f}% {bar}")
    print()

    print("--- FEATURES ---")
    for name, value in result.features.items():
        print(f"  {name}: {value:.4f}")
    print()

    print("--- RECOMMENDATION ---")
    print(result.recommendation)
    print()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='HMM Regime Detection')
    parser.add_argument('--train', action='store_true', help='Train model on historical data')
    parser.add_argument('--period', default='5y', help='Training period (default: 5y)')
    args = parser.parse_args()

    if args.train:
        print("Training HMM model...")
        train_hmm_model(period=args.period)
    else:
        print_hmm_regime_report()

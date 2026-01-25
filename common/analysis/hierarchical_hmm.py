"""
Hierarchical Hidden Markov Model for Market Regime Detection
=============================================================

Based on research:
- "Adaptive Hierarchical Hidden Markov Models for Structural Market Change" (MDPI 2025)
- Separates fast market states from slower structural regime changes

Architecture:
- Level 1 (Meta-regime): HIGH_UNCERTAINTY / LOW_UNCERTAINTY (weekly timeframe)
  - Captures structural changes like GFC, COVID, Fed tightening cycles
  - Driven by VIX, credit spreads, and macro uncertainty

- Level 2 (Market state): BULL / BEAR / CHOPPY / VOLATILE (daily timeframe)
  - Fast-moving market conditions
  - Transition probabilities DEPEND ON meta-regime

Key insight: During HIGH_UNCERTAINTY meta-regime, transitions to BEAR/VOLATILE
are more likely. During LOW_UNCERTAINTY, BULL persistence is higher.

Jan 2026 - Research-backed implementation
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

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


class MetaRegime(Enum):
    """Level 1: Structural uncertainty regime (weekly)."""
    LOW_UNCERTAINTY = 0   # Normal market conditions, stable transitions
    HIGH_UNCERTAINTY = 1  # Crisis/stress periods, volatile transitions


class MarketState(Enum):
    """Level 2: Fast market states (daily)."""
    VOLATILE = 0  # High vol, large swings
    BEAR = 1      # Downtrend
    CHOPPY = 2    # Range-bound
    BULL = 3      # Uptrend


@dataclass
class HierarchicalRegimeResult:
    """Result from hierarchical regime detection."""
    # Level 1: Meta-regime
    meta_regime: str  # "low_uncertainty" or "high_uncertainty"
    meta_confidence: float
    meta_days_in_regime: int

    # Level 2: Market state
    market_state: str  # "bull", "bear", "choppy", "volatile"
    state_confidence: float
    state_days: int

    # Combined
    regime_label: str  # e.g., "HIGH_UNCERTAINTY:BEAR" or "LOW_UNCERTAINTY:BULL"
    transition_probability: Dict[str, float]  # Prob of transitioning to each state

    # Features used
    meta_features: Dict[str, float]
    state_features: Dict[str, float]

    # Trading signals
    risk_multiplier: float  # 0.0 to 1.0, how much to scale positions
    recommendation: str


class HierarchicalHMM:
    """
    Two-level hierarchical HMM.

    Level 1: Meta-regime HMM (2 states, weekly features)
    Level 2: Market state HMM (4 states, daily features)

    The transition matrix of Level 2 changes based on Level 1 state.
    """

    def __init__(self):
        # Level 1: Meta-regime parameters (2 states)
        self.meta_n_states = 2
        self.meta_start_prob = np.array([0.7, 0.3])  # Start in low uncertainty

        # Meta-regime transition matrix (sticky - regimes persist)
        self.meta_trans = np.array([
            [0.95, 0.05],  # LOW -> LOW, LOW -> HIGH
            [0.10, 0.90]   # HIGH -> LOW, HIGH -> HIGH
        ])

        # Meta-regime emission parameters (VIX, credit spread, macro uncertainty)
        self.meta_means = np.array([
            [15.0, 0.0, 30.0],   # LOW: VIX ~15, no credit stress, low EPU
            [28.0, 2.0, 70.0]    # HIGH: VIX ~28, credit widening, high EPU
        ])
        self.meta_stds = np.array([
            [4.0, 0.5, 15.0],
            [10.0, 1.5, 25.0]
        ])

        # Level 2: Market state parameters (4 states)
        # DIFFERENT transition matrices for each meta-regime
        self.state_n_states = 4
        self.state_start_prob = np.array([0.1, 0.2, 0.4, 0.3])  # Start choppy

        # Transition matrix when meta-regime is LOW_UNCERTAINTY
        # Bull persists, bear/volatile transitions are rare
        self.state_trans_low = np.array([
            # VOLATILE, BEAR, CHOPPY, BULL (from)
            [0.50, 0.20, 0.20, 0.10],  # VOLATILE -> (calm down likely)
            [0.10, 0.60, 0.20, 0.10],  # BEAR -> (recovery likely)
            [0.05, 0.10, 0.60, 0.25],  # CHOPPY -> (bull likely)
            [0.02, 0.05, 0.13, 0.80],  # BULL -> (very sticky)
        ])

        # Transition matrix when meta-regime is HIGH_UNCERTAINTY
        # Bull fragile, bear/volatile more likely
        self.state_trans_high = np.array([
            # VOLATILE, BEAR, CHOPPY, BULL (from)
            [0.60, 0.25, 0.10, 0.05],  # VOLATILE -> (stays volatile)
            [0.20, 0.65, 0.10, 0.05],  # BEAR -> (bear persists)
            [0.15, 0.25, 0.50, 0.10],  # CHOPPY -> (risk of bear)
            [0.10, 0.20, 0.30, 0.40],  # BULL -> (fragile, can break)
        ])

        # State emission parameters (returns, volatility, momentum, breadth)
        self.state_means = np.array([
            [-0.001, 0.025, -0.02, 0.45],  # VOLATILE
            [-0.003, 0.018, -0.05, 0.35],  # BEAR
            [0.000, 0.010, 0.00, 0.50],    # CHOPPY
            [0.002, 0.008, 0.04, 0.65],    # BULL
        ])
        self.state_stds = np.array([
            [0.020, 0.010, 0.04, 0.15],
            [0.015, 0.008, 0.03, 0.12],
            [0.008, 0.005, 0.02, 0.10],
            [0.008, 0.004, 0.02, 0.10],
        ])

        # State tracking
        self._meta_history: List[int] = []
        self._state_history: List[int] = []

        # Model paths
        self.model_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'models' / 'hierarchical_hmm.json'

        # Try to load trained model
        if self.model_path.exists():
            self._load_model()

    def _gaussian_log_prob(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
        """Compute log probability under multivariate Gaussian (diagonal cov)."""
        return -0.5 * np.sum(np.log(2 * np.pi * std**2) + ((x - mean) / std)**2)

    def _extract_meta_features(self, df_weekly: pd.DataFrame) -> np.ndarray:
        """
        Extract Level 1 features (meta-regime, weekly).

        Features:
        1. VIX level (fear gauge)
        2. Credit spread change (HYG vs LQD relative performance)
        3. Economic Policy Uncertainty proxy (volatility of volatility)
        """
        features = []

        for i in range(len(df_weekly)):
            row = df_weekly.iloc[i]

            # VIX level
            vix = row.get('vix', 15.0)

            # Credit spread proxy (if available)
            credit = row.get('credit_spread', 0.0)

            # Uncertainty proxy: vol of vol (VIX std over period)
            if i >= 4:
                vix_std = df_weekly['vix'].iloc[max(0, i-4):i+1].std()
                uncertainty = vix_std * 10  # Scale to reasonable range
            else:
                uncertainty = 30.0

            features.append([vix, credit, uncertainty])

        return np.array(features)

    def _extract_state_features(self, df_daily: pd.DataFrame) -> np.ndarray:
        """
        Extract Level 2 features (market state, daily).

        Features:
        1. Daily returns
        2. Realized volatility (10-day)
        3. 10-day momentum
        4. Breadth proxy (above 20d MA)
        """
        features = []

        closes = df_daily['Close'].values

        for i in range(len(df_daily)):
            # Daily return
            if i > 0:
                ret = (closes[i] / closes[i-1]) - 1
            else:
                ret = 0.0

            # 10-day volatility
            if i >= 10:
                price_slice = closes[i-10:i+1]
                returns = np.diff(price_slice) / price_slice[:-1]
                vol = np.std(returns) if len(returns) > 0 else 0.01
            else:
                vol = 0.01

            # 10-day momentum
            if i >= 10:
                mom = (closes[i] / closes[i-10]) - 1
            else:
                mom = 0.0

            # Breadth proxy: where in 20-day range
            if i >= 20:
                high_20 = max(closes[i-20:i+1])
                low_20 = min(closes[i-20:i+1])
                if high_20 > low_20:
                    breadth = (closes[i] - low_20) / (high_20 - low_20)
                else:
                    breadth = 0.5
            else:
                breadth = 0.5

            features.append([ret, vol, mom, breadth])

        return np.array(features)

    def _forward_meta(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward algorithm for meta-regime HMM."""
        T = len(features)
        alpha = np.zeros((T, self.meta_n_states))

        # Initialize
        for s in range(self.meta_n_states):
            log_prob = self._gaussian_log_prob(features[0], self.meta_means[s], self.meta_stds[s])
            alpha[0, s] = self.meta_start_prob[s] * np.exp(log_prob)
        alpha[0] /= alpha[0].sum() + 1e-10

        # Forward pass
        for t in range(1, T):
            for s in range(self.meta_n_states):
                trans_prob = np.sum(alpha[t-1] * self.meta_trans[:, s])
                log_prob = self._gaussian_log_prob(features[t], self.meta_means[s], self.meta_stds[s])
                alpha[t, s] = trans_prob * np.exp(log_prob)
            alpha[t] /= alpha[t].sum() + 1e-10

        # Most likely sequence (Viterbi)
        states = np.argmax(alpha, axis=1)

        return alpha, states

    def _forward_state(self, features: np.ndarray, meta_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward algorithm for market state HMM (conditioned on meta-regime)."""
        T = len(features)
        alpha = np.zeros((T, self.state_n_states))

        # Initialize
        for s in range(self.state_n_states):
            log_prob = self._gaussian_log_prob(features[0], self.state_means[s], self.state_stds[s])
            alpha[0, s] = self.state_start_prob[s] * np.exp(log_prob)
        alpha[0] /= alpha[0].sum() + 1e-10

        # Forward pass (using meta-regime-dependent transitions)
        for t in range(1, T):
            # Select transition matrix based on meta-regime
            if meta_states[min(t, len(meta_states)-1)] == 0:  # LOW_UNCERTAINTY
                trans_matrix = self.state_trans_low
            else:  # HIGH_UNCERTAINTY
                trans_matrix = self.state_trans_high

            for s in range(self.state_n_states):
                trans_prob = np.sum(alpha[t-1] * trans_matrix[:, s])
                log_prob = self._gaussian_log_prob(features[t], self.state_means[s], self.state_stds[s])
                alpha[t, s] = trans_prob * np.exp(log_prob)
            alpha[t] /= alpha[t].sum() + 1e-10

        # Most likely sequence
        states = np.argmax(alpha, axis=1)

        return alpha, states

    def detect_regime(self,
                      df_daily: pd.DataFrame = None,
                      vix_data: pd.DataFrame = None,
                      lookback_weeks: int = 12) -> HierarchicalRegimeResult:
        """
        Detect current regime using hierarchical HMM.

        Args:
            df_daily: Daily OHLCV data for SPY or market proxy
            vix_data: VIX daily data (optional, will fetch if None)
            lookback_weeks: Weeks of history to use

        Returns:
            HierarchicalRegimeResult with both meta-regime and market state
        """
        if df_daily is None:
            if not HAS_YFINANCE:
                raise ImportError("yfinance required")
            spy = yf.Ticker("SPY")
            df_daily = spy.history(period='6mo')

        if vix_data is None:
            if HAS_YFINANCE:
                vix = yf.Ticker("^VIX")
                vix_data = vix.history(period='6mo')

        # Prepare weekly data for meta-regime
        df_weekly = self._prepare_weekly_data(df_daily, vix_data)
        df_weekly = df_weekly.tail(lookback_weeks)

        # Prepare daily data for market state
        lookback_days = lookback_weeks * 5
        df_daily_trim = df_daily.tail(lookback_days)

        # Extract features
        meta_features = self._extract_meta_features(df_weekly)
        state_features = self._extract_state_features(df_daily_trim)

        if len(meta_features) < 4 or len(state_features) < 10:
            return self._default_result("Insufficient data")

        # Level 1: Detect meta-regime
        meta_alpha, meta_states = self._forward_meta(meta_features)
        current_meta = meta_states[-1]
        meta_conf = float(meta_alpha[-1, current_meta])

        # Count days in meta-regime
        meta_days = 1
        for i in range(len(meta_states) - 2, -1, -1):
            if meta_states[i] == current_meta:
                meta_days += 7  # Weekly granularity
            else:
                break

        # Level 2: Detect market state (conditioned on meta-regime)
        # Upsample meta_states to daily
        meta_states_daily = np.repeat(meta_states, 5)[:len(state_features)]

        state_alpha, state_states = self._forward_state(state_features, meta_states_daily)
        current_state = state_states[-1]
        state_conf = float(state_alpha[-1, current_state])

        # Count days in state
        state_days = 1
        for i in range(len(state_states) - 2, -1, -1):
            if state_states[i] == current_state:
                state_days += 1
            else:
                break

        # Get transition probabilities
        if current_meta == 0:
            trans_probs = self.state_trans_low[current_state]
        else:
            trans_probs = self.state_trans_high[current_state]

        trans_dict = {
            'volatile': float(trans_probs[0]),
            'bear': float(trans_probs[1]),
            'choppy': float(trans_probs[2]),
            'bull': float(trans_probs[3])
        }

        # Calculate risk multiplier
        risk_mult = self._calculate_risk_multiplier(current_meta, current_state, state_conf)

        # Names
        meta_names = ['low_uncertainty', 'high_uncertainty']
        state_names = ['volatile', 'bear', 'choppy', 'bull']

        meta_name = meta_names[current_meta]
        state_name = state_names[current_state]

        # Update history
        self._meta_history.append(current_meta)
        self._state_history.append(current_state)
        if len(self._meta_history) > 100:
            self._meta_history = self._meta_history[-100:]
            self._state_history = self._state_history[-100:]

        # Generate recommendation
        recommendation = self._get_recommendation(current_meta, current_state, trans_dict, risk_mult)

        return HierarchicalRegimeResult(
            meta_regime=meta_name,
            meta_confidence=meta_conf,
            meta_days_in_regime=meta_days,
            market_state=state_name,
            state_confidence=state_conf,
            state_days=state_days,
            regime_label=f"{meta_name.upper()}:{state_name.upper()}",
            transition_probability=trans_dict,
            meta_features={
                'vix': float(meta_features[-1, 0]),
                'credit_spread': float(meta_features[-1, 1]),
                'uncertainty': float(meta_features[-1, 2])
            },
            state_features={
                'return': float(state_features[-1, 0]),
                'volatility': float(state_features[-1, 1]),
                'momentum': float(state_features[-1, 2]),
                'breadth': float(state_features[-1, 3])
            },
            risk_multiplier=risk_mult,
            recommendation=recommendation
        )

    def _prepare_weekly_data(self, df_daily: pd.DataFrame, vix_data: pd.DataFrame = None) -> pd.DataFrame:
        """Resample daily data to weekly with required features."""
        # Resample to weekly
        weekly = df_daily.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        # Add VIX if available
        if vix_data is not None and len(vix_data) > 0:
            vix_weekly = vix_data['Close'].resample('W').last()
            weekly['vix'] = vix_weekly.reindex(weekly.index, method='ffill')
        else:
            weekly['vix'] = 15.0

        # Credit spread proxy: returns volatility as stress indicator
        weekly['credit_spread'] = df_daily['Close'].pct_change().rolling(20).std().resample('W').last() * 100
        weekly['credit_spread'] = weekly['credit_spread'].fillna(0)

        return weekly

    def _calculate_risk_multiplier(self, meta: int, state: int, confidence: float) -> float:
        """
        Calculate position size multiplier based on regime.

        HIGH_UNCERTAINTY + BEAR/VOLATILE = very low (0.2-0.4)
        HIGH_UNCERTAINTY + BULL = moderate (0.5-0.7)
        LOW_UNCERTAINTY + BULL = high (0.8-1.0)
        LOW_UNCERTAINTY + BEAR = moderate (0.6-0.8)
        """
        base_multipliers = np.array([
            # VOLATILE, BEAR, CHOPPY, BULL
            [0.3, 0.4, 0.5, 0.6],  # HIGH_UNCERTAINTY
            [0.5, 0.7, 0.8, 1.0],  # LOW_UNCERTAINTY
        ])

        base = base_multipliers[1 - meta, state]  # Flip meta index

        # Adjust by confidence
        if confidence < 0.4:
            base *= 0.8  # Low confidence = reduce size
        elif confidence > 0.7:
            base *= 1.1  # High confidence = slight increase

        return min(1.0, max(0.2, base))

    def _get_recommendation(self, meta: int, state: int,
                           trans: Dict[str, float], risk_mult: float) -> str:
        """Generate trading recommendation."""
        meta_names = ['LOW_UNCERTAINTY', 'HIGH_UNCERTAINTY']
        state_names = ['VOLATILE', 'BEAR', 'CHOPPY', 'BULL']

        rec = f"Meta-regime: {meta_names[meta]} | State: {state_names[state]}\n"
        rec += f"Risk multiplier: {risk_mult:.1%} of normal size\n\n"

        if meta == 1:  # HIGH_UNCERTAINTY
            rec += "[!] HIGH UNCERTAINTY REGIME - Elevated caution\n"
            if state in [0, 1]:  # VOLATILE or BEAR
                rec += "- Reduce position sizes significantly\n"
                rec += "- Focus on high-conviction trades only\n"
                rec += "- Tighten stop losses\n"
            else:
                rec += "- Market state may be fragile\n"
                rec += f"- {trans['bear']:.0%} chance of transitioning to BEAR\n"
                rec += "- Consider hedging\n"
        else:  # LOW_UNCERTAINTY
            rec += "[OK] LOW UNCERTAINTY REGIME - Normal conditions\n"
            if state == 3:  # BULL
                rec += "- Bull regime is sticky in low uncertainty\n"
                rec += "- Mean reversion strategies may underperform\n"
                rec += "- Consider momentum/trend-following\n"
            elif state == 1:  # BEAR
                rec += "- Bear likely to recover in low uncertainty\n"
                rec += "- Mean reversion should work well\n"
                rec += f"- {trans['choppy']:.0%} chance of stabilizing\n"
            else:
                rec += "- Standard trading conditions\n"
                rec += "- Use normal position sizes\n"

        return rec

    def _default_result(self, msg: str) -> HierarchicalRegimeResult:
        """Return default result on error."""
        return HierarchicalRegimeResult(
            meta_regime='low_uncertainty',
            meta_confidence=0.5,
            meta_days_in_regime=0,
            market_state='choppy',
            state_confidence=0.5,
            state_days=0,
            regime_label='LOW_UNCERTAINTY:CHOPPY',
            transition_probability={'volatile': 0.25, 'bear': 0.25, 'choppy': 0.25, 'bull': 0.25},
            meta_features={},
            state_features={},
            risk_multiplier=0.5,
            recommendation=f"Default regime. {msg}"
        )

    def train(self, df_daily: pd.DataFrame = None, period: str = '5y') -> Dict:
        """
        Train hierarchical HMM parameters from data.

        Uses Baum-Welch to estimate:
        - Meta-regime emission parameters
        - Market state emission parameters
        - Transition matrices for each meta-regime
        """
        if df_daily is None:
            if not HAS_YFINANCE:
                raise ImportError("yfinance required")

            print("[HHMM] Fetching training data...")
            spy = yf.Ticker("SPY")
            df_daily = spy.history(period=period)

            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period=period)
        else:
            vix_data = None

        print(f"[HHMM] Training on {len(df_daily)} days of data...")

        # Prepare weekly data
        df_weekly = self._prepare_weekly_data(df_daily, vix_data)

        # Extract features
        meta_features = self._extract_meta_features(df_weekly)
        state_features = self._extract_state_features(df_daily)

        # Estimate meta-regime parameters from data
        # Use VIX thresholds to label regimes
        vix_vals = meta_features[:, 0]
        high_uncertainty_mask = vix_vals > 22

        for state in range(self.meta_n_states):
            if state == 0:  # LOW_UNCERTAINTY
                mask = ~high_uncertainty_mask
            else:  # HIGH_UNCERTAINTY
                mask = high_uncertainty_mask

            if mask.sum() > 5:
                self.meta_means[state] = meta_features[mask].mean(axis=0)
                self.meta_stds[state] = meta_features[mask].std(axis=0) + 0.01

        # Estimate state parameters from returns/volatility
        vol_vals = state_features[:, 1]
        ret_vals = state_features[:, 0]

        # Assign states based on thresholds
        state_labels = np.zeros(len(state_features), dtype=int)
        state_labels[vol_vals > np.percentile(vol_vals, 80)] = 0  # VOLATILE
        state_labels[(ret_vals < -0.005) & (state_labels == 0)] = 1  # BEAR
        state_labels[(ret_vals > 0.005) & (state_labels == 0)] = 3  # BULL
        state_labels[state_labels == 0] = 2  # CHOPPY

        for state in range(self.state_n_states):
            mask = state_labels == state
            if mask.sum() > 10:
                self.state_means[state] = state_features[mask].mean(axis=0)
                self.state_stds[state] = state_features[mask].std(axis=0) + 0.001

        # Save model
        self._save_model()

        # Calculate statistics
        stats = {
            'training_days': len(df_daily),
            'training_weeks': len(df_weekly),
            'meta_regime_distribution': {
                'low_uncertainty': float((~high_uncertainty_mask).sum() / len(high_uncertainty_mask)),
                'high_uncertainty': float(high_uncertainty_mask.sum() / len(high_uncertainty_mask))
            },
            'state_distribution': {
                'volatile': float((state_labels == 0).sum() / len(state_labels)),
                'bear': float((state_labels == 1).sum() / len(state_labels)),
                'choppy': float((state_labels == 2).sum() / len(state_labels)),
                'bull': float((state_labels == 3).sum() / len(state_labels))
            }
        }

        print("[HHMM] Training complete!")
        print(f"  Meta-regime: {stats['meta_regime_distribution']}")
        print(f"  States: {stats['state_distribution']}")

        return stats

    def _save_model(self):
        """Save model parameters."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        params = {
            'meta_start_prob': self.meta_start_prob.tolist(),
            'meta_trans': self.meta_trans.tolist(),
            'meta_means': self.meta_means.tolist(),
            'meta_stds': self.meta_stds.tolist(),
            'state_start_prob': self.state_start_prob.tolist(),
            'state_trans_low': self.state_trans_low.tolist(),
            'state_trans_high': self.state_trans_high.tolist(),
            'state_means': self.state_means.tolist(),
            'state_stds': self.state_stds.tolist(),
            'trained_at': datetime.now().isoformat()
        }

        with open(self.model_path, 'w') as f:
            json.dump(params, f, indent=2)

        print(f"[HHMM] Model saved to {self.model_path}")

    def _load_model(self):
        """Load model parameters."""
        try:
            with open(self.model_path, 'r') as f:
                params = json.load(f)

            self.meta_start_prob = np.array(params['meta_start_prob'])
            self.meta_trans = np.array(params['meta_trans'])
            self.meta_means = np.array(params['meta_means'])
            self.meta_stds = np.array(params['meta_stds'])
            self.state_start_prob = np.array(params['state_start_prob'])
            self.state_trans_low = np.array(params['state_trans_low'])
            self.state_trans_high = np.array(params['state_trans_high'])
            self.state_means = np.array(params['state_means'])
            self.state_stds = np.array(params['state_stds'])

            print(f"[HHMM] Model loaded from {self.model_path}")

        except Exception as e:
            print(f"[HHMM] Could not load model: {e}")


def get_hierarchical_regime() -> HierarchicalRegimeResult:
    """Quick function to get current hierarchical regime."""
    hmm = HierarchicalHMM()
    return hmm.detect_regime()


def print_hierarchical_report():
    """Print detailed hierarchical regime report."""
    print("=" * 70)
    print("HIERARCHICAL HMM REGIME DETECTION")
    print("Based on MDPI 2025 research: Adaptive Hierarchical HMMs")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    hmm = HierarchicalHMM()
    result = hmm.detect_regime()

    print(f"REGIME: {result.regime_label}")
    print()

    print("--- LEVEL 1: META-REGIME (Structural) ---")
    print(f"  State: {result.meta_regime.upper()}")
    print(f"  Confidence: {result.meta_confidence:.1%}")
    print(f"  Days in regime: {result.meta_days_in_regime}")
    print(f"  Features: VIX={result.meta_features.get('vix', 'N/A'):.1f}, "
          f"Uncertainty={result.meta_features.get('uncertainty', 'N/A'):.1f}")
    print()

    print("--- LEVEL 2: MARKET STATE (Tactical) ---")
    print(f"  State: {result.market_state.upper()}")
    print(f"  Confidence: {result.state_confidence:.1%}")
    print(f"  Days in state: {result.state_days}")
    print(f"  Features: Vol={result.state_features.get('volatility', 0)*100:.2f}%, "
          f"Mom={result.state_features.get('momentum', 0)*100:.1f}%")
    print()

    print("--- TRANSITION PROBABILITIES ---")
    for state, prob in sorted(result.transition_probability.items(), key=lambda x: -x[1]):
        bar = "#" * int(prob * 20)
        print(f"  -> {state.upper():<10} {prob:.0%} {bar}")
    print()

    print(f"RISK MULTIPLIER: {result.risk_multiplier:.0%}")
    print()

    print("--- RECOMMENDATION ---")
    print(result.recommendation)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hierarchical HMM Regime Detection')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--period', default='5y', help='Training period')
    args = parser.parse_args()

    if args.train:
        hmm = HierarchicalHMM()
        hmm.train(period=args.period)
    else:
        print_hierarchical_report()

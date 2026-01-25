"""
Backtest: HMM vs Rule-Based Regime Detection
=============================================

Compare accuracy of HMM-based and rule-based regime detection
by measuring mean reversion trade performance in each detected regime.

The better detector will show:
- Higher Sharpe in detected HIGH_ALPHA regimes (volatile, bear)
- Lower Sharpe in detected LOW_ALPHA regimes (bull)
- More consistent regime-performance correlation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, 'src')

import yfinance as yf
from analysis.hmm_regime_detector import HMMRegimeDetector
from analysis.market_regime import MarketRegimeDetector


def get_spy_data(period='2y'):
    """Fetch SPY data for regime detection."""
    spy = yf.Ticker("SPY")
    return spy.history(period=period)


def detect_regimes_historical(spy_df):
    """
    Detect regimes for each historical day using both methods.

    Returns DataFrame with date, hmm_regime, rule_regime, actual_performance
    """
    hmm_detector = HMMRegimeDetector()
    rule_detector = MarketRegimeDetector()

    results = []

    # Need at least 60 days for regime detection
    for i in range(60, len(spy_df)):
        date = spy_df.index[i]
        subset = spy_df.iloc[:i+1]

        # HMM detection
        try:
            features = hmm_detector._extract_features(subset)
            X = hmm_detector._normalize_features(features)
            valid_mask = ~np.isnan(X).any(axis=1)
            X_valid = X[valid_mask]
            features_valid = features[valid_mask]

            if len(X_valid) > 10 and hmm_detector.hmm._is_fitted:
                states = hmm_detector.hmm.predict(X_valid[-30:])  # Use last 30 days
                regimes = hmm_detector._map_states_to_regimes(states, features_valid[-30:])
                hmm_regime = hmm_detector.REGIME_NAMES[regimes[-1]]
            else:
                hmm_regime = 'choppy'
        except:
            hmm_regime = 'choppy'

        # Rule-based detection (simplified for backtest)
        try:
            close = subset['Close']
            trend_20d = (close.iloc[-1] / close.iloc[-20] - 1) * 100
            trend_50d = (close.iloc[-1] / close.iloc[-50] - 1) * 100 if len(close) >= 50 else trend_20d

            # Volatility proxy
            returns = close.pct_change().tail(20)
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized

            if volatility > 25:
                rule_regime = 'volatile'
            elif trend_20d > 3 and trend_50d > 5:
                rule_regime = 'bull'
            elif trend_20d < -3 and trend_50d < -5:
                rule_regime = 'bear'
            else:
                rule_regime = 'choppy'
        except:
            rule_regime = 'choppy'

        # Calculate next 3-day mean reversion performance (would a bounce happen?)
        if i + 3 < len(spy_df):
            today_return = (spy_df['Close'].iloc[i] / spy_df['Close'].iloc[i-1] - 1)

            # Only measure if today was a down day (mean reversion candidate)
            if today_return < -0.005:  # Down more than 0.5%
                next_3d_return = (spy_df['Close'].iloc[i+3] / spy_df['Close'].iloc[i] - 1) * 100
                is_reversion = next_3d_return > 0
            else:
                next_3d_return = None
                is_reversion = None
        else:
            next_3d_return = None
            is_reversion = None

        results.append({
            'date': date,
            'hmm_regime': hmm_regime,
            'rule_regime': rule_regime,
            'agreement': hmm_regime == rule_regime,
            'next_3d_return': next_3d_return,
            'is_reversion': is_reversion
        })

    return pd.DataFrame(results)


def analyze_regime_accuracy(df):
    """
    Analyze which regime detector better predicts mean reversion success.
    """
    # Filter to only days with mean reversion opportunities
    mr_df = df[df['is_reversion'].notna()].copy()

    if len(mr_df) == 0:
        print("No mean reversion opportunities found")
        return {}

    print(f"Analyzing {len(mr_df)} mean reversion opportunities")
    print()

    results = {'hmm': {}, 'rule': {}}

    for method in ['hmm', 'rule']:
        regime_col = f'{method}_regime'

        print(f"{'='*50}")
        print(f"{method.upper()} REGIME DETECTION")
        print(f"{'='*50}")
        print()

        for regime in ['volatile', 'bear', 'choppy', 'bull']:
            regime_data = mr_df[mr_df[regime_col] == regime]

            if len(regime_data) == 0:
                results[method][regime] = None
                continue

            win_rate = regime_data['is_reversion'].mean() * 100
            avg_return = regime_data['next_3d_return'].mean()
            count = len(regime_data)

            # Calculate Sharpe proxy (return / std)
            std = regime_data['next_3d_return'].std()
            sharpe = avg_return / std if std > 0 else 0

            results[method][regime] = {
                'count': count,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'sharpe': sharpe
            }

            print(f"{regime.upper():<10} | {count:>4} trades | {win_rate:>5.1f}% win | {avg_return:>+5.2f}% avg | Sharpe {sharpe:>4.2f}")

        print()

    return results


def calculate_detection_score(results):
    """
    Score each detector based on how well regimes predict performance.

    A good detector should:
    - Have highest Sharpe in VOLATILE/BEAR (high alpha regimes)
    - Have lowest Sharpe in BULL (low alpha regime)
    - Show clear differentiation between regimes
    """
    scores = {}

    for method in ['hmm', 'rule']:
        method_results = results[method]

        # Get Sharpe ratios (default to 0 if None)
        sharpes = {}
        for regime in ['volatile', 'bear', 'choppy', 'bull']:
            if method_results[regime]:
                sharpes[regime] = method_results[regime]['sharpe']
            else:
                sharpes[regime] = 0

        # Score 1: Volatile/Bear should have higher Sharpe than Bull
        high_alpha_avg = (sharpes['volatile'] + sharpes['bear']) / 2
        alpha_spread = high_alpha_avg - sharpes['bull']

        # Score 2: Volatility should be highest Sharpe
        if sharpes['volatile'] >= sharpes['bear'] >= sharpes['choppy'] >= sharpes['bull']:
            order_score = 10
        elif sharpes['volatile'] >= sharpes['choppy'] >= sharpes['bull']:
            order_score = 5
        else:
            order_score = 0

        # Score 3: Regime differentiation (variance of Sharpe across regimes)
        sharpe_values = [s for s in sharpes.values() if s != 0]
        if len(sharpe_values) > 1:
            differentiation = np.std(sharpe_values)
        else:
            differentiation = 0

        scores[method] = {
            'alpha_spread': alpha_spread,
            'order_score': order_score,
            'differentiation': differentiation,
            'total_score': alpha_spread * 10 + order_score + differentiation * 5
        }

    return scores


def main():
    print("=" * 70)
    print("HMM vs RULE-BASED REGIME DETECTION BACKTEST")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Get data
    print("[1] Fetching SPY data...")
    spy_df = get_spy_data('2y')
    print(f"    Got {len(spy_df)} days of data")
    print()

    # Detect regimes
    print("[2] Running regime detection on historical data...")
    print("    This may take a minute...")
    regime_df = detect_regimes_historical(spy_df)
    print()

    # Agreement analysis
    agreement_rate = regime_df['agreement'].mean() * 100
    print(f"[3] Regime Agreement Rate: {agreement_rate:.1f}%")
    print()

    # Regime distribution
    print("[4] Regime Distribution:")
    print()
    print(f"{'Regime':<12} | {'HMM':<10} | {'Rule':<10}")
    print("-" * 40)
    for regime in ['volatile', 'bear', 'choppy', 'bull']:
        hmm_pct = (regime_df['hmm_regime'] == regime).mean() * 100
        rule_pct = (regime_df['rule_regime'] == regime).mean() * 100
        print(f"{regime.upper():<12} | {hmm_pct:>8.1f}% | {rule_pct:>8.1f}%")
    print()

    # Performance analysis
    print("[5] Mean Reversion Performance by Detected Regime:")
    print()
    results = analyze_regime_accuracy(regime_df)

    # Calculate scores
    print("[6] Detection Quality Score:")
    print()
    scores = calculate_detection_score(results)

    print(f"{'Method':<10} | {'Alpha Spread':<12} | {'Order Score':<12} | {'Diff':<8} | {'TOTAL':<10}")
    print("-" * 60)
    for method, score in scores.items():
        print(f"{method.upper():<10} | {score['alpha_spread']:>10.2f} | {score['order_score']:>10} | "
              f"{score['differentiation']:>6.2f} | {score['total_score']:>8.2f}")

    print()

    # Winner
    winner = 'hmm' if scores['hmm']['total_score'] > scores['rule']['total_score'] else 'rule'
    margin = abs(scores['hmm']['total_score'] - scores['rule']['total_score'])

    print("=" * 70)
    print(f"WINNER: {winner.upper()} (by {margin:.2f} points)")
    print("=" * 70)

    if winner == 'hmm':
        print("HMM provides better regime-performance correlation.")
        print("Recommendation: Use HMM as primary detector with rule-based validation.")
    else:
        print("Rule-based provides better regime-performance correlation.")
        print("Recommendation: Use rule-based as primary detector.")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'data_days': len(spy_df),
        'mr_opportunities': len(regime_df[regime_df['is_reversion'].notna()]),
        'agreement_rate': agreement_rate,
        'results': {
            'hmm': {k: v for k, v in results['hmm'].items() if v},
            'rule': {k: v for k, v in results['rule'].items() if v}
        },
        'scores': scores,
        'winner': winner
    }

    Path('data/research').mkdir(parents=True, exist_ok=True)
    with open('data/research/hmm_vs_rule_backtest.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print()
    print("Results saved to data/research/hmm_vs_rule_backtest.json")


if __name__ == '__main__':
    main()

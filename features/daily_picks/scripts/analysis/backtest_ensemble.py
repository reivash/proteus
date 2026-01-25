"""
Ensemble vs Single Model Backtest
=================================

Compare performance of:
1. Individual models (Transformer, LSTM, MLP)
2. Ensemble with voting (2/3, 3/3 consensus)

Jan 2026 - Proteus Trading System
"""

import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import yfinance as yf

# Import models
from models.transformer_signal_model import TransformerSignalModel
from models.lstm_signal_model import LSTMSignalModel
from models.gpu_signal_model import GPUSignalModel


def get_historical_data(tickers, period='1y'):
    """Get historical data for all tickers."""
    print(f"\nFetching historical data for {len(tickers)} tickers...")
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if len(df) >= 120:  # Need enough data
                data[ticker] = df
        except:
            pass
    print(f"Got data for {len(data)} tickers")
    return data


def backtest_single_model(model, model_name, data, threshold, hold_days=2):
    """Backtest a single model."""
    results = []

    for ticker, df in data.items():
        close = df['Close'].values

        # Test on last 60 trading days
        for i in range(-80, -hold_days):
            try:
                # Get data up to this point
                test_df = df.iloc[:i] if i < -1 else df

                # Get signal
                signal = model.predict(ticker, test_df)
                if not signal:
                    continue

                strength = signal.signal_strength

                if strength >= threshold:
                    # Calculate actual return
                    entry_idx = len(df) + i
                    exit_idx = min(entry_idx + hold_days, len(df) - 1)

                    if exit_idx > entry_idx:
                        actual_return = (close[exit_idx] / close[entry_idx] - 1) * 100

                        results.append({
                            'ticker': ticker,
                            'strength': strength,
                            'actual_return': actual_return,
                            'win': actual_return > 0
                        })
            except:
                pass

    return results


def backtest_ensemble(transformer, lstm, mlp, data,
                     transformer_thresh=50, lstm_thresh=40, mlp_thresh=40,
                     min_votes=2, hold_days=2):
    """Backtest ensemble with voting."""
    results = []

    # Weights from Sharpe ratios
    TRANSFORMER_WEIGHT = 0.34
    LSTM_WEIGHT = 0.32
    MLP_WEIGHT = 0.34

    for ticker, df in data.items():
        close = df['Close'].values

        # Test on last 60 trading days
        for i in range(-80, -hold_days):
            try:
                test_df = df.iloc[:i] if i < -1 else df

                # Get signals from all models
                trans_signal = transformer.predict(ticker, test_df) if transformer else None
                lstm_signal = lstm.predict(ticker, test_df) if lstm else None
                mlp_signal = mlp.predict(ticker, test_df) if mlp else None

                trans_strength = trans_signal.signal_strength if trans_signal else 0
                lstm_strength = lstm_signal.signal_strength if lstm_signal else 0
                mlp_strength = mlp_signal.signal_strength if mlp_signal else 0

                # Count votes
                votes = sum([
                    trans_strength >= transformer_thresh,
                    lstm_strength >= lstm_thresh,
                    mlp_strength >= mlp_thresh
                ])

                if votes >= min_votes:
                    # Calculate weighted ensemble strength
                    total_weight = 0
                    weighted_strength = 0

                    if trans_signal:
                        weighted_strength += trans_strength * TRANSFORMER_WEIGHT
                        total_weight += TRANSFORMER_WEIGHT
                    if lstm_signal:
                        weighted_strength += lstm_strength * LSTM_WEIGHT
                        total_weight += LSTM_WEIGHT
                    if mlp_signal:
                        weighted_strength += mlp_strength * MLP_WEIGHT
                        total_weight += MLP_WEIGHT

                    if total_weight > 0:
                        ensemble_strength = weighted_strength / total_weight

                        # Apply consensus boost
                        if votes == 3:
                            ensemble_strength *= 1.20
                        elif votes == 2:
                            ensemble_strength *= 1.10

                        # Calculate actual return
                        entry_idx = len(df) + i
                        exit_idx = min(entry_idx + hold_days, len(df) - 1)

                        if exit_idx > entry_idx:
                            actual_return = (close[exit_idx] / close[entry_idx] - 1) * 100

                            results.append({
                                'ticker': ticker,
                                'strength': ensemble_strength,
                                'votes': votes,
                                'actual_return': actual_return,
                                'win': actual_return > 0
                            })
            except:
                pass

    return results


def analyze_results(results, name):
    """Analyze backtest results."""
    if not results:
        return {'trades': 0, 'win_rate': 0, 'avg_return': 0, 'sharpe': 0}

    trades = len(results)
    wins = sum(1 for r in results if r['win'])
    win_rate = wins / trades * 100
    returns = [r['actual_return'] for r in results]
    avg_return = np.mean(returns)
    std_return = np.std(returns) + 0.001
    sharpe = avg_return / std_return

    return {
        'name': name,
        'trades': trades,
        'wins': wins,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'std_return': std_return,
        'sharpe': sharpe
    }


def main():
    print("="*70)
    print("ENSEMBLE VS SINGLE MODEL BACKTEST")
    print("="*70)
    print(f"Started: {datetime.now()}")

    # Tickers to test
    tickers = [
        "NVDA", "V", "MA", "AVGO", "AXP", "KLAC", "ORCL", "MRVL", "ABBV",
        "EOG", "TXN", "GILD", "MSFT", "QCOM", "JPM", "JNJ", "XOM", "MPC",
        "CAT", "COP", "SLB", "CVS", "USB", "MS", "CRM", "META"
    ]

    # Get data
    data = get_historical_data(tickers, period='1y')

    # Load models
    print("\nLoading models...")
    transformer = TransformerSignalModel()
    lstm = LSTMSignalModel()
    mlp = GPUSignalModel()
    print("All models loaded")

    all_results = []

    # Test single models at optimal thresholds
    print("\n" + "-"*70)
    print("SINGLE MODEL BACKTESTS")
    print("-"*70)

    # Transformer @ 50
    print("\nTesting Transformer @ 50...")
    trans_results = backtest_single_model(transformer, "Transformer", data, threshold=50)
    trans_stats = analyze_results(trans_results, "Transformer@50")
    all_results.append(trans_stats)
    print(f"  Trades: {trans_stats['trades']}, Win: {trans_stats['win_rate']:.1f}%, "
          f"Avg: {trans_stats['avg_return']:+.2f}%, Sharpe: {trans_stats['sharpe']:.2f}")

    # LSTM @ 40
    print("\nTesting LSTM @ 40...")
    lstm_results = backtest_single_model(lstm, "LSTM", data, threshold=40)
    lstm_stats = analyze_results(lstm_results, "LSTM@40")
    all_results.append(lstm_stats)
    print(f"  Trades: {lstm_stats['trades']}, Win: {lstm_stats['win_rate']:.1f}%, "
          f"Avg: {lstm_stats['avg_return']:+.2f}%, Sharpe: {lstm_stats['sharpe']:.2f}")

    # MLP @ 40
    print("\nTesting MLP @ 40...")
    mlp_results = backtest_single_model(mlp, "MLP", data, threshold=40)
    mlp_stats = analyze_results(mlp_results, "MLP@40")
    all_results.append(mlp_stats)
    print(f"  Trades: {mlp_stats['trades']}, Win: {mlp_stats['win_rate']:.1f}%, "
          f"Avg: {mlp_stats['avg_return']:+.2f}%, Sharpe: {mlp_stats['sharpe']:.2f}")

    # Test ensemble configurations
    print("\n" + "-"*70)
    print("ENSEMBLE BACKTESTS")
    print("-"*70)

    # Ensemble with 2+ votes
    print("\nTesting Ensemble (2+ votes)...")
    ens2_results = backtest_ensemble(transformer, lstm, mlp, data, min_votes=2)
    ens2_stats = analyze_results(ens2_results, "Ensemble-2+")
    all_results.append(ens2_stats)
    print(f"  Trades: {ens2_stats['trades']}, Win: {ens2_stats['win_rate']:.1f}%, "
          f"Avg: {ens2_stats['avg_return']:+.2f}%, Sharpe: {ens2_stats['sharpe']:.2f}")

    # Vote distribution
    if ens2_results:
        votes_2 = sum(1 for r in ens2_results if r['votes'] == 2)
        votes_3 = sum(1 for r in ens2_results if r['votes'] == 3)
        print(f"  Vote distribution: 2-vote={votes_2}, 3-vote={votes_3}")

    # Ensemble with 3 votes only (unanimous)
    print("\nTesting Ensemble (3 votes only - unanimous)...")
    ens3_results = backtest_ensemble(transformer, lstm, mlp, data, min_votes=3)
    ens3_stats = analyze_results(ens3_results, "Ensemble-3")
    all_results.append(ens3_stats)
    print(f"  Trades: {ens3_stats['trades']}, Win: {ens3_stats['win_rate']:.1f}%, "
          f"Avg: {ens3_stats['avg_return']:+.2f}%, Sharpe: {ens3_stats['sharpe']:.2f}")

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"\n{'Model':<20} {'Trades':>8} {'Win Rate':>10} {'Avg Ret':>10} {'Sharpe':>10}")
    print("-"*60)

    # Sort by Sharpe
    all_results.sort(key=lambda x: x['sharpe'], reverse=True)

    for r in all_results:
        if r['trades'] > 0:
            print(f"{r['name']:<20} {r['trades']:>8} {r['win_rate']:>9.1f}% "
                  f"{r['avg_return']:>+9.2f}% {r['sharpe']:>10.2f}")

    # Best configuration
    best = all_results[0] if all_results else None
    if best and best['trades'] >= 10:
        print(f"\nBEST CONFIGURATION: {best['name']}")
        print(f"  Sharpe: {best['sharpe']:.2f}")
        print(f"  Win Rate: {best['win_rate']:.1f}%")
        print(f"  Avg Return: {best['avg_return']:+.2f}%")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

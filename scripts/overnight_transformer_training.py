"""
Overnight Transformer Training and Comprehensive Backtest
==========================================================

This script runs overnight to:
1. Train the new Transformer model (150 epochs, ~30-60 min)
2. Run comparative backtest: Transformer vs LSTM vs MLP
3. Test on held-out recent data (last 2 months)
4. Generate comprehensive report for morning review

Expected runtime: 2-3 hours with GPU

Jan 2026 - Proteus Trading System
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Suppress numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

sys.path.insert(0, 'src')

import yfinance as yf
import torch

# Import all models
from models.transformer_signal_model import TransformerSignalModel
from models.lstm_signal_model import LSTMSignalModel
from models.gpu_signal_model import GPUSignalModel


def log(msg):
    """Log with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")

    # Also write to log file
    with open('overnight_training_log.txt', 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")


def train_transformer():
    """Train Transformer model."""
    log("=" * 70)
    log("PHASE 1: TRANSFORMER MODEL TRAINING")
    log("=" * 70)

    start_time = time.time()

    try:
        model = TransformerSignalModel()
        best_acc, history = model.train(epochs=150, batch_size=64, lr=5e-5, warmup_epochs=10)

        elapsed = (time.time() - start_time) / 60
        log(f"Transformer training complete in {elapsed:.1f} minutes")
        log(f"Best validation accuracy: {best_acc*100:.1f}%")

        return True, best_acc

    except Exception as e:
        log(f"ERROR in Transformer training: {e}")
        return False, 0


def get_test_data(tickers: list, days: int = 60):
    """Get recent test data (last N days)."""
    test_data = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period='6mo')

            if len(df) >= days + 60:  # Need warmup
                test_data[ticker] = df
        except:
            pass

    return test_data


def backtest_model(model, model_name: str, test_data: dict, threshold: float = 50.0):
    """
    Backtest a model on held-out data.

    For each day in test period:
    - Generate signal
    - Check if above threshold
    - Measure next 2-day return
    """
    log(f"Backtesting {model_name}...")

    results = []

    for ticker, df in test_data.items():
        close = df['Close'].values

        # Test on last 40 trading days (leaving 20 days for forward returns)
        for i in range(-60, -20):
            try:
                # Subset data up to this point
                test_df = df.iloc[:i] if i < -1 else df

                # Get prediction
                if model_name == 'MLP':
                    signal = model.predict(ticker, test_df)
                    if signal:
                        signal_strength = signal.signal_strength
                        prob = signal.mean_reversion_prob
                    else:
                        continue
                else:
                    signal = model.predict(ticker, test_df)
                    if signal:
                        signal_strength = signal.signal_strength
                        prob = signal.mean_reversion_prob
                    else:
                        continue

                # Check if signal passes threshold
                if signal_strength >= threshold:
                    # Calculate actual 2-day return
                    entry_idx = len(df) + i
                    exit_idx = min(entry_idx + 2, len(df) - 1)

                    if exit_idx > entry_idx:
                        actual_return = (close[exit_idx] / close[entry_idx] - 1) * 100
                        win = actual_return > 0

                        results.append({
                            'ticker': ticker,
                            'signal_strength': signal_strength,
                            'probability': prob,
                            'actual_return': actual_return,
                            'win': win
                        })

            except Exception as e:
                pass

    return results


def compare_models(test_data: dict):
    """Compare all models on same test data."""
    log("=" * 70)
    log("PHASE 2: COMPARATIVE BACKTEST")
    log("=" * 70)

    models = {}

    # Load models
    try:
        log("Loading Transformer model...")
        models['Transformer'] = TransformerSignalModel()
    except Exception as e:
        log(f"Failed to load Transformer: {e}")

    try:
        log("Loading LSTM model...")
        models['LSTM'] = LSTMSignalModel()
    except Exception as e:
        log(f"Failed to load LSTM: {e}")

    try:
        log("Loading MLP model...")
        models['MLP'] = GPUSignalModel()
    except Exception as e:
        log(f"Failed to load MLP: {e}")

    # Test thresholds
    thresholds = [40, 50, 60, 70]

    all_results = {}

    for model_name, model in models.items():
        all_results[model_name] = {}

        for threshold in thresholds:
            log(f"Testing {model_name} at threshold {threshold}...")

            results = backtest_model(model, model_name, test_data, threshold)

            if results:
                wins = sum(1 for r in results if r['win'])
                total = len(results)
                win_rate = wins / total * 100 if total > 0 else 0
                avg_return = np.mean([r['actual_return'] for r in results])
                sharpe = avg_return / (np.std([r['actual_return'] for r in results]) + 0.001)

                all_results[model_name][threshold] = {
                    'trades': total,
                    'wins': wins,
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'sharpe': sharpe
                }

                log(f"  {model_name} @ {threshold}: {total} trades, {win_rate:.1f}% win, {avg_return:.2f}% avg, Sharpe {sharpe:.2f}")
            else:
                all_results[model_name][threshold] = {
                    'trades': 0, 'wins': 0, 'win_rate': 0, 'avg_return': 0, 'sharpe': 0
                }

    return all_results


def generate_report(transformer_success: bool, transformer_acc: float,
                   backtest_results: dict):
    """Generate comprehensive morning report."""
    log("=" * 70)
    log("PHASE 3: GENERATING REPORT")
    log("=" * 70)

    report = []
    report.append("=" * 70)
    report.append("OVERNIGHT TRANSFORMER TRAINING RESULTS")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    report.append("")

    # Training results
    report.append("TRANSFORMER TRAINING:")
    report.append("-" * 40)
    if transformer_success:
        report.append(f"  Status: SUCCESS")
        report.append(f"  Best Validation Accuracy: {transformer_acc*100:.1f}%")
    else:
        report.append(f"  Status: FAILED")
    report.append("")

    # Backtest comparison
    report.append("MODEL COMPARISON BACKTEST (Last 60 Days):")
    report.append("-" * 70)
    report.append(f"{'Model':<15} {'Threshold':<10} {'Trades':<8} {'Win Rate':<10} {'Avg Ret':<10} {'Sharpe':<10}")
    report.append("-" * 70)

    for model_name, thresholds in backtest_results.items():
        for threshold, metrics in thresholds.items():
            report.append(
                f"{model_name:<15} {threshold:<10} {metrics['trades']:<8} "
                f"{metrics['win_rate']:.1f}%{'':<5} {metrics['avg_return']:+.2f}%{'':<5} "
                f"{metrics['sharpe']:.2f}"
            )
    report.append("")

    # Find best model
    best_model = None
    best_sharpe = -999
    best_threshold = None

    for model_name, thresholds in backtest_results.items():
        for threshold, metrics in thresholds.items():
            if metrics['trades'] >= 10 and metrics['sharpe'] > best_sharpe:
                best_sharpe = metrics['sharpe']
                best_model = model_name
                best_threshold = threshold

    report.append("RECOMMENDATION:")
    report.append("-" * 40)
    if best_model:
        report.append(f"  Best Model: {best_model} @ threshold {best_threshold}")
        report.append(f"  Sharpe: {best_sharpe:.2f}")

        if best_model == 'Transformer':
            report.append("")
            report.append("  ACTION: Transformer outperforms! Consider switching.")
            report.append("  To enable: Update hybrid_signal_model.py to use Transformer")
        elif best_model == 'LSTM':
            report.append("")
            report.append("  ACTION: Current LSTM is best. No changes needed.")
        else:
            report.append("")
            report.append("  ACTION: MLP still competitive. Consider ensemble.")
    else:
        report.append("  Insufficient data for recommendation")

    report.append("")
    report.append("=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)

    # Write report
    report_text = "\n".join(report)

    with open('OVERNIGHT_TRANSFORMER_RESULTS.txt', 'w') as f:
        f.write(report_text)

    log("Report saved to OVERNIGHT_TRANSFORMER_RESULTS.txt")

    # Also save JSON for programmatic access
    json_results = {
        'timestamp': datetime.now().isoformat(),
        'transformer_training': {
            'success': transformer_success,
            'best_accuracy': transformer_acc
        },
        'backtest_results': backtest_results,
        'recommendation': {
            'best_model': best_model,
            'best_threshold': best_threshold,
            'best_sharpe': best_sharpe
        }
    }

    Path('data/research').mkdir(parents=True, exist_ok=True)
    with open('data/research/overnight_transformer_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    return report_text


def main():
    """Main overnight training pipeline."""
    log("=" * 70)
    log("OVERNIGHT TRANSFORMER TRAINING PIPELINE")
    log("=" * 70)
    log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log("")

    start_time = time.time()

    # Phase 1: Train Transformer
    transformer_success, transformer_acc = train_transformer()

    # Phase 2: Get test data
    log("")
    log("Fetching test data...")
    tickers = [
        "NVDA", "V", "MA", "AVGO", "AXP", "KLAC", "ORCL", "MRVL", "ABBV",
        "EOG", "TXN", "GILD", "MSFT", "QCOM", "JPM", "JNJ", "XOM", "MPC",
        "CAT", "COP", "SLB", "CVS", "USB", "MS", "CRM", "META"
    ]
    test_data = get_test_data(tickers, days=60)
    log(f"Got test data for {len(test_data)} stocks")

    # Phase 3: Compare models
    backtest_results = compare_models(test_data)

    # Phase 4: Generate report
    report = generate_report(transformer_success, transformer_acc, backtest_results)

    # Summary
    elapsed = (time.time() - start_time) / 60
    log("")
    log("=" * 70)
    log("OVERNIGHT PIPELINE COMPLETE")
    log(f"Total runtime: {elapsed:.1f} minutes")
    log("=" * 70)
    log("")
    log("Review OVERNIGHT_TRANSFORMER_RESULTS.txt for full report")

    # Print report to console too
    print("\n" + report)


if __name__ == '__main__':
    # Clear previous log
    with open('overnight_training_log.txt', 'w') as f:
        f.write(f"Overnight Training Log - Started {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n")

    main()

"""
Overnight Regime Optimization Pipeline
=======================================

This script runs overnight to:
1. Analyze ensemble performance by market regime (BULL/BEAR/CHOPPY)
2. Find optimal thresholds per regime
3. Test regime-adaptive position sizing
4. Generate comprehensive morning report

Expected runtime: 1-2 hours with GPU

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
from collections import defaultdict

warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import yfinance as yf
import torch

# Import models and analysis tools
from models.hybrid_signal_model import HybridSignalModel
from models.transformer_signal_model import TransformerSignalModel
from models.lstm_signal_model import LSTMSignalModel
from models.gpu_signal_model import GPUSignalModel
from analysis.market_regime import MarketRegimeDetector, MarketRegime


LOG_FILE = 'overnight_regime_log.txt'


def log(msg):
    """Log with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")

    with open(LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")


def get_historical_regimes(spy_data: pd.DataFrame) -> pd.DataFrame:
    """
    Classify historical periods into regimes.
    Uses simplified rule-based detection for backtesting.
    """
    regimes = []

    close = spy_data['Close'].values

    for i in range(60, len(spy_data)):
        # Get lookback data
        lookback = close[i-60:i]
        current = close[i]

        # Calculate indicators
        sma_20 = np.mean(close[i-20:i])
        sma_50 = np.mean(close[i-50:i])

        # Recent volatility
        recent_prices = lookback[-20:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns) * np.sqrt(252) * 100

        # Trend
        trend_20d = (current / close[i-20] - 1) * 100
        trend_50d = (current / close[i-50] - 1) * 100

        # Classify regime
        if trend_20d > 3 and current > sma_20 > sma_50:
            regime = 'BULL'
        elif trend_20d < -3 and current < sma_20 < sma_50:
            regime = 'BEAR'
        else:
            regime = 'CHOPPY'

        regimes.append({
            'date': spy_data.index[i],
            'regime': regime,
            'trend_20d': trend_20d,
            'volatility': volatility
        })

    return pd.DataFrame(regimes)


def backtest_by_regime(model, model_name: str, stock_data: dict,
                       regime_data: pd.DataFrame, threshold: float,
                       min_votes: int = 1) -> dict:
    """
    Backtest a model and separate results by regime.
    """
    results_by_regime = defaultdict(list)

    for ticker, df in stock_data.items():
        close = df['Close'].values

        # Test on historical periods
        for i in range(-200, -5):
            try:
                test_date = df.index[i]

                # Find regime for this date
                regime_match = regime_data[regime_data['date'].dt.date == test_date.date()]
                if len(regime_match) == 0:
                    continue
                regime = regime_match.iloc[0]['regime']

                # Get prediction
                test_df = df.iloc[:i]
                if len(test_df) < 90:
                    continue

                signal = model.predict(ticker, test_df)
                if not signal:
                    continue

                strength = signal.signal_strength
                votes = getattr(signal, 'votes', 1)

                # Apply filters
                if strength < threshold:
                    continue
                if votes < min_votes:
                    continue

                # Calculate 2-day return
                entry_idx = len(df) + i
                exit_idx = min(entry_idx + 2, len(df) - 1)

                if exit_idx > entry_idx:
                    actual_return = (close[exit_idx] / close[entry_idx] - 1) * 100

                    results_by_regime[regime].append({
                        'ticker': ticker,
                        'date': test_date,
                        'strength': strength,
                        'votes': votes,
                        'return': actual_return,
                        'win': actual_return > 0
                    })
            except:
                pass

    return dict(results_by_regime)


def analyze_regime_results(results_by_regime: dict) -> dict:
    """Analyze results for each regime."""
    analysis = {}

    for regime, results in results_by_regime.items():
        if not results:
            analysis[regime] = {
                'trades': 0, 'win_rate': 0, 'avg_return': 0, 'sharpe': 0
            }
            continue

        trades = len(results)
        wins = sum(1 for r in results if r['win'])
        returns = [r['return'] for r in results]

        win_rate = wins / trades * 100
        avg_return = np.mean(returns)
        std_return = np.std(returns) + 0.001
        sharpe = avg_return / std_return

        analysis[regime] = {
            'trades': trades,
            'wins': wins,
            'win_rate': round(win_rate, 1),
            'avg_return': round(avg_return, 3),
            'std_return': round(std_return, 3),
            'sharpe': round(sharpe, 2)
        }

    return analysis


def find_optimal_threshold(model, model_name: str, stock_data: dict,
                          regime_data: pd.DataFrame, regime: str,
                          min_votes: int = 1) -> dict:
    """Find optimal threshold for a specific regime."""
    log(f"  Finding optimal threshold for {regime}...")

    thresholds = [30, 35, 40, 45, 50, 55, 60, 65, 70]
    best_sharpe = -999
    best_threshold = 50
    results = []

    for thresh in thresholds:
        regime_results = backtest_by_regime(
            model, model_name, stock_data, regime_data,
            threshold=thresh, min_votes=min_votes
        )

        if regime in regime_results and len(regime_results[regime]) >= 10:
            stats = analyze_regime_results({regime: regime_results[regime]})[regime]
            results.append({
                'threshold': thresh,
                **stats
            })

            if stats['sharpe'] > best_sharpe:
                best_sharpe = stats['sharpe']
                best_threshold = thresh

    return {
        'best_threshold': best_threshold,
        'best_sharpe': best_sharpe,
        'all_results': results
    }


def run_regime_optimization():
    """Main regime optimization pipeline."""
    log("=" * 70)
    log("PHASE 1: LOADING DATA AND MODELS")
    log("=" * 70)

    start_time = time.time()

    # Load ensemble model
    log("Loading 3-model ensemble...")
    ensemble = HybridSignalModel()
    log(f"Ensemble loaded: {ensemble.MIN_VOTES} min votes")

    # Get SPY data for regime classification
    log("\nFetching SPY data for regime classification...")
    spy = yf.Ticker("SPY")
    spy_data = spy.history(period="2y")
    log(f"SPY data: {len(spy_data)} days")

    # Classify historical regimes
    log("Classifying historical regimes...")
    regime_data = get_historical_regimes(spy_data)

    regime_counts = regime_data['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(regime_data) * 100
        log(f"  {regime}: {count} days ({pct:.1f}%)")

    # Get stock data
    log("\nFetching stock data...")
    tickers = [
        "NVDA", "V", "MA", "AVGO", "AXP", "KLAC", "ORCL", "MRVL", "ABBV",
        "EOG", "TXN", "GILD", "MSFT", "QCOM", "JPM", "JNJ", "XOM", "MPC",
        "CAT", "COP", "SLB", "CVS", "USB", "MS", "CRM", "META", "HD", "NOW"
    ]

    stock_data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="2y")
            if len(df) >= 200:
                stock_data[ticker] = df
        except:
            pass
    log(f"Loaded data for {len(stock_data)} stocks")

    # ===== PHASE 2: BASELINE ANALYSIS =====
    log("\n" + "=" * 70)
    log("PHASE 2: BASELINE ENSEMBLE PERFORMANCE BY REGIME")
    log("=" * 70)

    # Test ensemble with current settings (threshold 50, min_votes 2)
    log("\nTesting ensemble with current settings (threshold=50, min_votes=2)...")
    baseline_results = backtest_by_regime(
        ensemble, "Ensemble", stock_data, regime_data,
        threshold=50, min_votes=2
    )
    baseline_analysis = analyze_regime_results(baseline_results)

    log("\nBaseline results by regime:")
    for regime in ['BULL', 'BEAR', 'CHOPPY']:
        if regime in baseline_analysis:
            stats = baseline_analysis[regime]
            log(f"  {regime}: {stats['trades']} trades, {stats['win_rate']:.1f}% win, "
                f"{stats['avg_return']:+.2f}% avg, Sharpe {stats['sharpe']:.2f}")

    # ===== PHASE 3: THRESHOLD OPTIMIZATION PER REGIME =====
    log("\n" + "=" * 70)
    log("PHASE 3: FINDING OPTIMAL THRESHOLDS PER REGIME")
    log("=" * 70)

    optimal_thresholds = {}

    for regime in ['BULL', 'BEAR', 'CHOPPY']:
        log(f"\nOptimizing for {regime} regime...")

        # Test with min_votes=2 (ensemble)
        result = find_optimal_threshold(
            ensemble, "Ensemble", stock_data, regime_data,
            regime=regime, min_votes=2
        )
        optimal_thresholds[regime] = result

        log(f"  Best threshold: {result['best_threshold']} (Sharpe: {result['best_sharpe']:.2f})")

        # Show all thresholds tested
        for r in result['all_results']:
            log(f"    Threshold {r['threshold']}: {r['trades']} trades, "
                f"{r['win_rate']:.1f}% win, Sharpe {r['sharpe']:.2f}")

    # ===== PHASE 4: TEST REGIME-ADAPTIVE STRATEGY =====
    log("\n" + "=" * 70)
    log("PHASE 4: TESTING REGIME-ADAPTIVE STRATEGY")
    log("=" * 70)

    # Apply optimal thresholds per regime
    log("\nBacktesting with regime-adaptive thresholds...")

    adaptive_results = defaultdict(list)

    for ticker, df in stock_data.items():
        close = df['Close'].values

        for i in range(-200, -5):
            try:
                test_date = df.index[i]

                # Find regime
                regime_match = regime_data[regime_data['date'].dt.date == test_date.date()]
                if len(regime_match) == 0:
                    continue
                regime = regime_match.iloc[0]['regime']

                # Get regime-specific threshold
                regime_threshold = optimal_thresholds.get(regime, {}).get('best_threshold', 50)

                # Get prediction
                test_df = df.iloc[:i]
                if len(test_df) < 90:
                    continue

                signal = ensemble.predict(ticker, test_df)
                if not signal:
                    continue

                # Apply regime-specific threshold
                if signal.signal_strength < regime_threshold:
                    continue
                if signal.votes < 2:
                    continue

                # Calculate return
                entry_idx = len(df) + i
                exit_idx = min(entry_idx + 2, len(df) - 1)

                if exit_idx > entry_idx:
                    actual_return = (close[exit_idx] / close[entry_idx] - 1) * 100

                    adaptive_results[regime].append({
                        'ticker': ticker,
                        'threshold_used': regime_threshold,
                        'return': actual_return,
                        'win': actual_return > 0
                    })
            except:
                pass

    adaptive_analysis = analyze_regime_results(dict(adaptive_results))

    log("\nRegime-adaptive results:")
    for regime in ['BULL', 'BEAR', 'CHOPPY']:
        if regime in adaptive_analysis:
            stats = adaptive_analysis[regime]
            thresh = optimal_thresholds.get(regime, {}).get('best_threshold', 50)
            log(f"  {regime} (thresh={thresh}): {stats['trades']} trades, "
                f"{stats['win_rate']:.1f}% win, Sharpe {stats['sharpe']:.2f}")

    # ===== PHASE 5: COMPARE STRATEGIES =====
    log("\n" + "=" * 70)
    log("PHASE 5: STRATEGY COMPARISON")
    log("=" * 70)

    # Calculate overall stats
    def calc_overall(results_dict):
        all_results = []
        for regime_results in results_dict.values():
            all_results.extend(regime_results)
        if not all_results:
            return {'trades': 0, 'win_rate': 0, 'avg_return': 0, 'sharpe': 0}

        trades = len(all_results)
        wins = sum(1 for r in all_results if r['win'])
        returns = [r['return'] for r in all_results]

        return {
            'trades': trades,
            'win_rate': round(wins / trades * 100, 1),
            'avg_return': round(np.mean(returns), 3),
            'sharpe': round(np.mean(returns) / (np.std(returns) + 0.001), 2)
        }

    baseline_overall = calc_overall(baseline_results)
    adaptive_overall = calc_overall(dict(adaptive_results))

    log("\nOverall comparison:")
    log(f"  Fixed threshold (50):     {baseline_overall['trades']} trades, "
        f"{baseline_overall['win_rate']:.1f}% win, Sharpe {baseline_overall['sharpe']:.2f}")
    log(f"  Regime-adaptive:          {adaptive_overall['trades']} trades, "
        f"{adaptive_overall['win_rate']:.1f}% win, Sharpe {adaptive_overall['sharpe']:.2f}")

    improvement = adaptive_overall['sharpe'] - baseline_overall['sharpe']
    log(f"\n  Sharpe improvement: {improvement:+.2f}")

    # ===== PHASE 6: GENERATE REPORT =====
    log("\n" + "=" * 70)
    log("PHASE 6: GENERATING REPORT")
    log("=" * 70)

    elapsed = (time.time() - start_time) / 60

    report = {
        'timestamp': datetime.now().isoformat(),
        'runtime_minutes': round(elapsed, 1),
        'stocks_tested': len(stock_data),
        'regime_distribution': regime_counts.to_dict(),
        'baseline': {
            'threshold': 50,
            'min_votes': 2,
            'overall': baseline_overall,
            'by_regime': baseline_analysis
        },
        'optimal_thresholds': {
            regime: {
                'threshold': data['best_threshold'],
                'sharpe': data['best_sharpe']
            }
            for regime, data in optimal_thresholds.items()
        },
        'adaptive_strategy': {
            'overall': adaptive_overall,
            'by_regime': adaptive_analysis
        },
        'recommendation': {
            'use_adaptive': improvement > 0.05,
            'sharpe_improvement': round(improvement, 2),
            'thresholds': {
                regime: data['best_threshold']
                for regime, data in optimal_thresholds.items()
            }
        }
    }

    # Save JSON report
    Path('data/research').mkdir(parents=True, exist_ok=True)
    with open('data/research/overnight_regime_results.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Generate text report
    text_report = []
    text_report.append("=" * 70)
    text_report.append("OVERNIGHT REGIME OPTIMIZATION RESULTS")
    text_report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    text_report.append(f"Runtime: {elapsed:.1f} minutes")
    text_report.append("=" * 70)
    text_report.append("")

    text_report.append("REGIME DISTRIBUTION (2 years):")
    text_report.append("-" * 40)
    for regime, count in regime_counts.items():
        pct = count / len(regime_data) * 100
        text_report.append(f"  {regime}: {count} days ({pct:.1f}%)")
    text_report.append("")

    text_report.append("OPTIMAL THRESHOLDS BY REGIME:")
    text_report.append("-" * 40)
    for regime in ['BULL', 'BEAR', 'CHOPPY']:
        if regime in optimal_thresholds:
            thresh = optimal_thresholds[regime]['best_threshold']
            sharpe = optimal_thresholds[regime]['best_sharpe']
            text_report.append(f"  {regime}: threshold {thresh} (Sharpe {sharpe:.2f})")
    text_report.append("")

    text_report.append("STRATEGY COMPARISON:")
    text_report.append("-" * 40)
    text_report.append(f"  Fixed (threshold=50):  {baseline_overall['trades']} trades, "
                       f"{baseline_overall['win_rate']:.1f}% win, Sharpe {baseline_overall['sharpe']:.2f}")
    text_report.append(f"  Regime-adaptive:       {adaptive_overall['trades']} trades, "
                       f"{adaptive_overall['win_rate']:.1f}% win, Sharpe {adaptive_overall['sharpe']:.2f}")
    text_report.append("")
    text_report.append(f"  Sharpe improvement: {improvement:+.2f}")
    text_report.append("")

    text_report.append("RECOMMENDATION:")
    text_report.append("-" * 40)
    if improvement > 0.05:
        text_report.append("  ACTION: Use regime-adaptive thresholds!")
        text_report.append("  Update smart_scanner.py with these thresholds:")
        for regime in ['BULL', 'BEAR', 'CHOPPY']:
            if regime in optimal_thresholds:
                text_report.append(f"    {regime}: {optimal_thresholds[regime]['best_threshold']}")
    else:
        text_report.append("  ACTION: Keep fixed threshold (50)")
        text_report.append("  Regime-adaptive did not show significant improvement.")

    text_report.append("")
    text_report.append("=" * 70)
    text_report.append("END OF REPORT")
    text_report.append("=" * 70)

    report_text = "\n".join(text_report)

    with open('OVERNIGHT_REGIME_RESULTS.txt', 'w') as f:
        f.write(report_text)

    log("Report saved to OVERNIGHT_REGIME_RESULTS.txt")
    log(f"\nTotal runtime: {elapsed:.1f} minutes")

    print("\n" + report_text)

    return report


def main():
    """Main entry point."""
    log("=" * 70)
    log("OVERNIGHT REGIME OPTIMIZATION PIPELINE")
    log("=" * 70)
    log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log("")

    try:
        report = run_regime_optimization()

        log("\n" + "=" * 70)
        log("OVERNIGHT PIPELINE COMPLETE")
        log("=" * 70)
        log("Review OVERNIGHT_REGIME_RESULTS.txt for full report")

    except Exception as e:
        log(f"\nERROR: {e}")
        import traceback
        log(traceback.format_exc())


if __name__ == '__main__':
    # Clear previous log
    with open(LOG_FILE, 'w') as f:
        f.write(f"Overnight Regime Optimization Log - Started {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n")

    main()

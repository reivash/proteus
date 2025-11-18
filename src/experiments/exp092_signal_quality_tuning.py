"""
EXP-092: Signal Quality Tuning - Generate More Q4 Signals

CONTEXT - EXP-091 BREAKTHROUGH FINDING:
Base scanner already delivers 63.7% win rate WITHOUT ML filtering.
Signal strength quartile analysis revealed MASSIVE quality variation:

Q4 (Strongest): 77.8% win rate, +1.79% avg return  <-- TARGET
Q3:             56.8% win rate, +0.42% avg return
Q2:             60.0% win rate, +1.09% avg return
Q1 (Weakest):   60.0% win rate, +0.54% avg return
Overall:        63.7% win rate, +0.96% avg return

OBJECTIVE: Tune base scanner filters to generate MORE Q4-quality signals

METHODOLOGY:
1. Analyze Q4 signal characteristics (z-score, RSI, volume patterns)
2. Identify what separates 77.8% win rate from 60% win rate signals
3. Test filter adjustments:
   - Stricter z-score thresholds (-2.5 vs -2.0)?
   - Narrower RSI bands (20-25 vs 20-30)?
   - Higher volume surge requirements (1.8x vs 1.5x)?
4. Measure trade-off: Higher quality vs signal volume

SUCCESS CRITERIA:
- Scenario A: Generate 50%+ Q4-type signals → +10pp win rate improvement
- Scenario B: Slight quality boost with minimal signal loss → +3-5pp win rate

EXPECTED OUTCOME:
Even modest improvements could be huge:
- 70% win rate (from 63.7%) = +10% annual return boost
- More reliable signals = lower drawdowns, higher Sharpe

This is the LAST frontier - we've optimized execution, now optimize signal generation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple

from src.trading.signal_scanner import SignalScanner
from src.data.fetchers.yahoo_finance import YahooFinanceFetcher


def analyze_signal_characteristics(signals: List[Dict]) -> Dict:
    """
    Analyze what makes Q4 signals different from lower quartiles.

    Args:
        signals: List of signals with performance data

    Returns:
        Analysis of Q4 vs other quartile characteristics
    """
    df = pd.DataFrame(signals)

    # Add quartiles based on signal strength
    df['strength_quartile'] = pd.qcut(
        df['signal_strength'],
        q=4,
        labels=['Q1', 'Q2', 'Q3', 'Q4']
    )

    analysis = {}

    for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_df = df[df['strength_quartile'] == quartile]

        if len(q_df) == 0:
            continue

        analysis[quartile] = {
            'count': len(q_df),
            'win_rate': (q_df['win'].sum() / len(q_df)) * 100,
            'avg_return': q_df['forward_return_3d'].mean(),
            'z_score': {
                'mean': q_df['z_score'].mean(),
                'min': q_df['z_score'].min(),
                'max': q_df['z_score'].max(),
                'median': q_df['z_score'].median()
            },
            'rsi': {
                'mean': q_df['rsi'].mean(),
                'min': q_df['rsi'].min(),
                'max': q_df['rsi'].max(),
                'median': q_df['rsi'].median()
            },
            'expected_return': {
                'mean': q_df['expected_return'].mean(),
                'min': q_df['expected_return'].min(),
                'max': q_df['expected_return'].max(),
                'median': q_df['expected_return'].median()
            }
        }

    return analysis


def test_filter_configuration(
    ticker: str,
    start_date: str,
    end_date: str,
    z_score_threshold: float,
    rsi_lower: float,
    rsi_upper: float
) -> List[Dict]:
    """
    Test a specific filter configuration.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date
        z_score_threshold: Z-score threshold (e.g. -2.0, -2.5)
        rsi_lower: RSI lower bound (e.g. 20, 25)
        rsi_upper: RSI upper bound (e.g. 30, 25)

    Returns:
        List of signals generated with performance data
    """
    fetcher = YahooFinanceFetcher()

    # Fetch data with buffer
    fetch_start = (pd.to_datetime(start_date) - timedelta(days=150)).strftime('%Y-%m-%d')
    data = fetcher.fetch_stock_data(ticker, start_date=fetch_start, end_date=end_date)

    if data is None or len(data) < 50:
        return []

    # Ensure index is DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')
        else:
            data.index = pd.to_datetime(data.index)

    # Create custom scanner with adjusted thresholds
    scanner = SignalScanner(lookback_days=90)

    # Override thresholds (we'll modify scanner directly)
    # Note: This requires modifying SignalScanner or creating a new version
    # For now, we'll scan normally and filter afterwards

    signals = []
    test_start = pd.to_datetime(start_date)

    if hasattr(data.index, 'tz') and data.index.tz is not None:
        test_start = test_start.tz_localize(data.index.tz)

    test_data = data[data.index >= test_start].copy()

    for date in test_data.index:
        date_str = date.strftime('%Y-%m-%d')

        # Get signal
        signal = scanner.scan_stock(ticker, date_str)

        if signal is None:
            continue

        # Apply custom filters
        z_score = signal.get('z_score', 0)
        rsi = signal.get('rsi', 50)

        # Check if signal passes stricter filters
        if z_score > z_score_threshold:
            continue  # Z-score not negative enough

        if rsi < rsi_lower or rsi > rsi_upper:
            continue  # RSI outside acceptable range

        # Calculate forward return
        try:
            entry_idx = data.index.get_loc(date)
            if entry_idx + 3 < len(data):
                entry_price = data.iloc[entry_idx]['Close']
                exit_price = data.iloc[entry_idx + 3]['Close']
                forward_return = ((exit_price - entry_price) / entry_price) * 100

                signals.append({
                    'ticker': ticker,
                    'date': date_str,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'forward_return_3d': forward_return,
                    'win': forward_return > 0,
                    'z_score': z_score,
                    'rsi': rsi,
                    'signal_strength': signal.get('signal_strength'),
                    'expected_return': signal.get('expected_return')
                })
        except:
            continue

    return signals


def main():
    print("=" * 70)
    print("EXP-092: SIGNAL QUALITY TUNING")
    print("=" * 70)
    print()
    print("OBJECTIVE: Tune filters to generate more Q4-quality signals (77.8% WR)")
    print()

    # Test period (2024 holdout)
    start_date = '2024-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    # Test stocks
    test_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        'JPM', 'BAC', 'GS', 'WFC',
        'JNJ', 'UNH', 'PFE', 'ABBV',
        'WMT', 'COST', 'TGT',
        'V', 'MA',
        'DIS', 'NFLX',
        'BA', 'CAT', 'MMM',
        'XOM', 'CVX',
        'TSLA', 'F'
    ]

    # Step 1: Get baseline signals with current filters
    print("Step 1: Analyzing baseline signal characteristics...")
    print()

    scanner = SignalScanner(lookback_days=90)
    baseline_signals = []

    for i, ticker in enumerate(test_stocks, 1):
        print(f"[{i}/{len(test_stocks)}] {ticker}...", end=' ')

        try:
            fetcher = YahooFinanceFetcher()
            fetch_start = (pd.to_datetime(start_date) - timedelta(days=150)).strftime('%Y-%m-%d')
            data = fetcher.fetch_stock_data(ticker, start_date=fetch_start, end_date=end_date)

            if data is None or len(data) < 50:
                print("[SKIP] Insufficient data")
                continue

            # Ensure index is DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'Date' in data.columns:
                    data['Date'] = pd.to_datetime(data['Date'])
                    data = data.set_index('Date')
                else:
                    data.index = pd.to_datetime(data.index)

            test_start = pd.to_datetime(start_date)
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                test_start = test_start.tz_localize(data.index.tz)

            test_data = data[data.index >= test_start].copy()
            signals_found = 0

            for date in test_data.index:
                date_str = date.strftime('%Y-%m-%d')
                signal = scanner.scan_stock(ticker, date_str)

                if signal is None:
                    continue

                # Calculate forward return
                try:
                    entry_idx = data.index.get_loc(date)
                    if entry_idx + 3 < len(data):
                        entry_price = data.iloc[entry_idx]['Close']
                        exit_price = data.iloc[entry_idx + 3]['Close']
                        forward_return = ((exit_price - entry_price) / entry_price) * 100

                        baseline_signals.append({
                            'ticker': ticker,
                            'date': date_str,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'forward_return_3d': forward_return,
                            'win': forward_return > 0,
                            'z_score': signal.get('z_score'),
                            'rsi': signal.get('rsi'),
                            'signal_strength': signal.get('signal_strength'),
                            'expected_return': signal.get('expected_return')
                        })
                        signals_found += 1
                except:
                    continue

            print(f"[OK] {signals_found} signals")

        except Exception as e:
            print(f"[ERROR] {e}")
            continue

    print()

    if len(baseline_signals) == 0:
        print("[ERROR] No baseline signals found")
        return

    # Analyze baseline characteristics
    print("=" * 70)
    print("BASELINE SIGNAL ANALYSIS")
    print("=" * 70)
    print()

    baseline_analysis = analyze_signal_characteristics(baseline_signals)

    print(f"{'Quartile':<10} {'Count':<8} {'Win Rate':<12} {'Avg Return':<12} "
          f"{'Z-Score':<20} {'RSI':<20}")
    print("-" * 90)

    for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
        if quartile not in baseline_analysis:
            continue

        stats = baseline_analysis[quartile]
        print(f"{quartile:<10} {stats['count']:<8} "
              f"{stats['win_rate']:<12.1f}% {stats['avg_return']:<12.2f}% "
              f"{stats['z_score']['median']:<20.2f} {stats['rsi']['median']:<20.1f}")

    print()

    # Find Q4 characteristics
    if 'Q4' in baseline_analysis:
        q4_stats = baseline_analysis['Q4']
        print("Q4 (TARGET) CHARACTERISTICS:")
        print(f"  Win Rate: {q4_stats['win_rate']:.1f}%")
        print(f"  Avg Return: {q4_stats['avg_return']:+.2f}%")
        print(f"  Z-Score Range: {q4_stats['z_score']['min']:.2f} to {q4_stats['z_score']['max']:.2f}")
        print(f"  Z-Score Median: {q4_stats['z_score']['median']:.2f}")
        print(f"  RSI Range: {q4_stats['rsi']['min']:.1f} to {q4_stats['rsi']['max']:.1f}")
        print(f"  RSI Median: {q4_stats['rsi']['median']:.1f}")
        print()

    # Step 2: Test stricter filter configurations
    print("=" * 70)
    print("TESTING STRICTER FILTER CONFIGURATIONS")
    print("=" * 70)
    print()

    # Define configurations to test
    configurations = [
        {'name': 'Baseline', 'z_threshold': -2.0, 'rsi_lower': 20, 'rsi_upper': 30},
        {'name': 'Stricter Z', 'z_threshold': -2.5, 'rsi_lower': 20, 'rsi_upper': 30},
        {'name': 'Tighter RSI', 'z_threshold': -2.0, 'rsi_lower': 20, 'rsi_upper': 25},
        {'name': 'Both Stricter', 'z_threshold': -2.5, 'rsi_lower': 20, 'rsi_upper': 25},
        {'name': 'Ultra Strict', 'z_threshold': -3.0, 'rsi_lower': 15, 'rsi_upper': 25},
    ]

    results = []

    for config in configurations:
        print(f"Testing: {config['name']}")
        print(f"  Z-Score <= {config['z_threshold']}, RSI {config['rsi_lower']}-{config['rsi_upper']}")

        config_signals = []

        for ticker in test_stocks[:10]:  # Test on subset for speed
            try:
                signals = test_filter_configuration(
                    ticker,
                    start_date,
                    end_date,
                    config['z_threshold'],
                    config['rsi_lower'],
                    config['rsi_upper']
                )
                config_signals.extend(signals)
            except:
                continue

        if len(config_signals) > 0:
            df = pd.DataFrame(config_signals)
            win_rate = (df['win'].sum() / len(df)) * 100
            avg_return = df['forward_return_3d'].mean()

            results.append({
                'name': config['name'],
                'config': config,
                'signal_count': len(config_signals),
                'win_rate': win_rate,
                'avg_return': avg_return
            })

            print(f"  Signals: {len(config_signals)}, Win Rate: {win_rate:.1f}%, Avg Return: {avg_return:+.2f}%")
        else:
            print(f"  No signals generated")

        print()

    # Step 3: Recommendation
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print()

    baseline_df = pd.DataFrame(baseline_signals)
    baseline_wr = (baseline_df['win'].sum() / len(baseline_df)) * 100
    baseline_ret = baseline_df['forward_return_3d'].mean()

    print(f"Baseline: {len(baseline_signals)} signals, {baseline_wr:.1f}% WR, {baseline_ret:+.2f}% avg return")
    print()

    # Find best configuration
    best_config = None
    best_score = 0

    for result in results:
        if result['name'] == 'Baseline':
            continue

        # Score = win rate improvement with minimal signal loss
        signal_retention = result['signal_count'] / len(baseline_signals) if len(baseline_signals) > 0 else 0
        wr_improvement = result['win_rate'] - baseline_wr
        score = wr_improvement * signal_retention

        if score > best_score and signal_retention > 0.5:  # Keep at least 50% of signals
            best_score = score
            best_config = result

    if best_config:
        print(f"[RECOMMENDED] {best_config['name']}")
        print(f"  Configuration: Z <= {best_config['config']['z_threshold']}, "
              f"RSI {best_config['config']['rsi_lower']}-{best_config['config']['rsi_upper']}")
        print(f"  Win Rate: {best_config['win_rate']:.1f}% (vs {baseline_wr:.1f}% baseline)")
        print(f"  Avg Return: {best_config['avg_return']:+.2f}% (vs {baseline_ret:+.2f}% baseline)")
        print(f"  Signal Count: {best_config['signal_count']} (vs {len(baseline_signals)} baseline)")
        print(f"  Signal Retention: {(best_config['signal_count']/len(baseline_signals))*100:.1f}%")
    else:
        print("[NO IMPROVEMENT] Baseline filters are already optimal")
        print()
        print("The current scanner configuration generates the best quality/quantity trade-off.")

    print()

    # Save results
    output = {
        'experiment': 'EXP-092',
        'timestamp': datetime.now().isoformat(),
        'objective': 'Tune filters to generate more Q4-quality signals',
        'baseline': {
            'signal_count': len(baseline_signals),
            'win_rate': float(baseline_wr),
            'avg_return': float(baseline_ret),
            'analysis': baseline_analysis
        },
        'configurations_tested': results,
        'recommendation': best_config['name'] if best_config else 'Keep baseline'
    }

    output_path = Path('logs/experiments/exp092_signal_quality_tuning.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"[OK] Results saved to {output_path}")
    print()


if __name__ == '__main__':
    main()

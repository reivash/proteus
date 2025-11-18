"""
EXP-091: Validate Base Signal Scanner Performance (No ML)

CRITICAL QUESTION: Is the ML model (AUC 0.525) adding value or destroying it?

HYPOTHESIS: The base SignalScanner may already have good precision without ML filtering.
If true, the ML model (barely better than random) is unnecessary complexity.

METHODOLOGY:
1. Run base SignalScanner on historical data (2024 holdout period)
2. Check 3-day forward returns for ALL signals found
3. Calculate win rate and average return
4. Compare to ML-filtered signals (threshold=0.30)

SUCCESS CRITERIA:
- If base scanner has >55% win rate: ML adds little value, use base scanner
- If base scanner has <50% win rate: ML is necessary (even at 0.525 AUC)
- If ML improves win rate by <5pp: Not worth the complexity

EXPECTED OUTCOME:
Based on mean reversion strategy design (z-score, RSI, volume filters),
the base scanner should already have good signal quality. If so, we should
SIMPLIFY by removing ML filtering entirely.
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
import traceback

from src.trading.signal_scanner import SignalScanner
from src.data.fetchers.yahoo_finance import YahooFinanceFetcher


def test_signal_performance(ticker, start_date, end_date, scanner):
    """
    Test signal performance for a single stock.

    Returns:
        List of dictionaries with signal data and forward returns
    """
    fetcher = YahooFinanceFetcher()

    # Fetch data with buffer for indicators (150 days to ensure 60+ trading days)
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
            # Index might already be dates but not DatetimeIndex
            data.index = pd.to_datetime(data.index)

    signals = []

    # Scan each day in the test period
    test_start = pd.to_datetime(start_date)

    # Handle timezone if data index is timezone-aware
    if hasattr(data.index, 'tz') and data.index.tz is not None:
        test_start = test_start.tz_localize(data.index.tz)

    test_data = data[data.index >= test_start].copy()

    for date in test_data.index:
        date_str = date.strftime('%Y-%m-%d')

        # Get signal for this date
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

                signals.append({
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
        except:
            continue

    return signals


def main():
    print("=" * 70)
    print("EXP-091: BASE SIGNAL SCANNER VALIDATION")
    print("=" * 70)
    print()

    # Test on holdout period (2024-01-01 to present)
    start_date = '2024-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"Test Period: {start_date} to {end_date}")
    print()

    # Production stock universe (subset for speed)
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

    print(f"Testing {len(test_stocks)} stocks...")
    print()

    # Initialize base scanner (NO ML)
    # Use 100 lookback days to ensure 60+ trading days (accounts for weekends/holidays)
    scanner = SignalScanner(lookback_days=100)

    # Collect all signals
    all_signals = []

    for i, ticker in enumerate(test_stocks, 1):
        print(f"[{i}/{len(test_stocks)}] {ticker}...", end=' ')

        try:
            signals = test_signal_performance(ticker, start_date, end_date, scanner)
            all_signals.extend(signals)
            print(f"[OK] {len(signals)} signals")
        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()
            continue

    print()

    if len(all_signals) == 0:
        print("[ERROR] No signals found in test period")
        return

    # Analyze results
    df = pd.DataFrame(all_signals)

    total_signals = len(df)
    winners = df[df['win'] == True]
    losers = df[df['win'] == False]

    win_rate = (len(winners) / total_signals) * 100
    avg_return = df['forward_return_3d'].mean()
    avg_win = winners['forward_return_3d'].mean() if len(winners) > 0 else 0
    avg_loss = losers['forward_return_3d'].mean() if len(losers) > 0 else 0

    print("=" * 70)
    print("BASE SCANNER PERFORMANCE (NO ML FILTERING)")
    print("=" * 70)
    print()

    print(f"Total Signals: {total_signals}")
    print(f"Winners: {len(winners)} ({win_rate:.1f}%)")
    print(f"Losers: {len(losers)} ({100-win_rate:.1f}%)")
    print()

    print(f"Average Return: {avg_return:+.2f}%")
    print(f"Average Win: {avg_win:+.2f}%")
    print(f"Average Loss: {avg_loss:+.2f}%")
    print()

    # Calculate by signal strength quartiles
    print("PERFORMANCE BY SIGNAL STRENGTH:")
    print("-" * 70)

    df['strength_quartile'] = pd.qcut(df['signal_strength'], q=4, labels=['Q1 (Weak)', 'Q2', 'Q3', 'Q4 (Strong)'])

    for quartile in ['Q1 (Weak)', 'Q2', 'Q3', 'Q4 (Strong)']:
        q_df = df[df['strength_quartile'] == quartile]
        if len(q_df) == 0:
            continue

        q_win_rate = (q_df['win'].sum() / len(q_df)) * 100
        q_avg_return = q_df['forward_return_3d'].mean()

        print(f"{quartile}: {len(q_df)} signals, {q_win_rate:.1f}% win rate, {q_avg_return:+.2f}% avg return")

    print()

    # Compare to random baseline
    print("=" * 70)
    print("COMPARISON TO BASELINES")
    print("=" * 70)
    print()

    print(f"Base Scanner: {win_rate:.1f}% win rate, {avg_return:+.2f}% avg return")
    print(f"Random Baseline: 50.0% win rate, ~0.0% avg return")
    print(f"ML Model (AUC 0.525): ~52.5% win rate (estimated)")
    print()

    # Decision
    print("=" * 70)
    print("ASSESSMENT")
    print("=" * 70)
    print()

    if win_rate >= 55:
        print("[STRONG PERFORMANCE] Base scanner already effective!")
        print(f"Win rate {win_rate:.1f}% >> ML improvement would be marginal")
        print()
        print("RECOMMENDATION: Use base scanner WITHOUT ML filtering")
        print("- Simpler implementation")
        print("- No ML model maintenance overhead")
        print("- Already profitable signal quality")
        recommendation = "USE_BASE_SCANNER"
    elif win_rate >= 50 and win_rate < 55:
        print("[MODERATE PERFORMANCE] Base scanner slightly profitable")
        print(f"Win rate {win_rate:.1f}% - ML could add 2-5pp improvement")
        print()
        print("RECOMMENDATION: Deploy ML if it improves win rate by >3pp")
        print("- Test ML-filtered signals vs base signals")
        print("- If ML improvement <3pp, use base scanner instead")
        recommendation = "TEST_ML_VALUE"
    else:
        print("[WEAK PERFORMANCE] Base scanner needs ML filtering")
        print(f"Win rate {win_rate:.1f}% - below breakeven after costs")
        print()
        print("RECOMMENDATION: Continue ML model improvements")
        print("- Current AUC 0.525 insufficient")
        print("- Try ensemble, new features, or different approach")
        recommendation = "IMPROVE_ML"

    print()

    # Save results
    results = {
        'experiment': 'EXP-091',
        'timestamp': datetime.now().isoformat(),
        'objective': 'Validate base scanner performance without ML',
        'test_period': {
            'start': start_date,
            'end': end_date
        },
        'results': {
            'total_signals': int(total_signals),
            'win_rate': float(win_rate),
            'avg_return': float(avg_return),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss)
        },
        'comparison': {
            'base_scanner_win_rate': float(win_rate),
            'random_baseline': 50.0,
            'ml_model_auc': 0.525,
            'ml_estimated_win_rate': 52.5
        },
        'recommendation': recommendation
    }

    results_path = Path('results/ml_experiments/exp091_base_scanner_validation.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Results saved to {results_path}")
    print()


if __name__ == '__main__':
    main()

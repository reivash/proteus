"""
Optimize threshold for Penalties-Only System

The penalties-only system needs a different threshold than the full boost system.
Test thresholds 55-80 to find optimal balance of quality vs opportunity.
"""
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import yfinance as yf

import sys
sys.path.insert(0, 'src')

from trading.penalties_only_calculator import PenaltiesOnlyCalculator


def get_stock_data(ticker: str, period: str = '2y'):
    """Fetch stock data."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if len(df) < 50:
            return None
        return df
    except:
        return None


def calculate_indicators(df) -> dict:
    """Calculate technical indicators."""
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values

    # RSI
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(14)/14, mode='valid')[-1]
    avg_loss = np.convolve(loss, np.ones(14)/14, mode='valid')[-1]
    rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))

    # SMA200
    sma200 = np.mean(close[-200:]) if len(close) >= 200 else np.mean(close)
    sma200_distance = (close[-1] / sma200 - 1) * 100

    # Consecutive down days
    down_days = 0
    for i in range(len(close)-1, 0, -1):
        if close[i] < close[i-1]:
            down_days += 1
        else:
            break

    # Drawdown from 20d high
    high_20d = np.max(high[-20:])
    drawdown = (close[-1] / high_20d - 1) * 100

    # Volume ratio
    vol_avg = np.mean(volume[-20:])
    vol_ratio = volume[-1] / (vol_avg + 1)

    return {
        'rsi': rsi,
        'sma200_distance': sma200_distance,
        'consecutive_down': down_days,
        'drawdown': drawdown,
        'volume_ratio': vol_ratio,
        'close': close[-1]
    }


def backtest_threshold(tickers: list, calc: PenaltiesOnlyCalculator, threshold: int):
    """Backtest penalties-only at specific threshold."""

    all_trades = []

    for ticker in tickers:
        df = get_stock_data(ticker, '2y')
        if df is None or len(df) < 250:
            continue

        for i in range(250, len(df) - 5):
            sub_df = df.iloc[:i+1]
            indicators = calculate_indicators(sub_df)

            date = df.index[i]
            dow = date.weekday()

            # Generate base signal (simulated from RSI + drawdown)
            base_signal = 50 + (30 - indicators['rsi']) * 0.5 + abs(indicators['drawdown']) * 1.5
            base_signal = max(30, min(80, base_signal))

            # Calculate with penalties-only
            result = calc.calculate(
                ticker=ticker,
                base_signal=base_signal,
                regime='choppy',
                is_monday=dow == 0,
                is_tuesday=dow == 1,
                is_wednesday=dow == 2,
                is_thursday=dow == 3,
                is_friday=dow == 4,
                consecutive_down_days=indicators['consecutive_down'],
                volume_ratio=indicators['volume_ratio'],
                sma200_distance=indicators['sma200_distance'],
                drawdown_pct=indicators['drawdown']
            )

            if result.final_signal >= threshold:
                entry_price = indicators['close']
                exit_price = df['Close'].iloc[i + 2]
                return_pct = (exit_price / entry_price - 1) * 100
                win = return_pct > 0

                all_trades.append({
                    'ticker': ticker,
                    'date': date,
                    'base': base_signal,
                    'final': result.final_signal,
                    'penalties': result.total_penalty,
                    'return': return_pct,
                    'win': win
                })

    if not all_trades:
        return {
            'threshold': threshold,
            'trades': 0,
            'wins': 0,
            'win_rate': 0,
            'avg_return': 0,
            'total_return': 0,
            'avg_penalty': 0,
            'trades_per_year': 0
        }

    wins = sum(1 for t in all_trades if t['win'])

    return {
        'threshold': threshold,
        'trades': len(all_trades),
        'wins': wins,
        'win_rate': wins / len(all_trades) * 100,
        'avg_return': np.mean([t['return'] for t in all_trades]),
        'total_return': sum(t['return'] for t in all_trades),
        'avg_penalty': np.mean([t['penalties'] for t in all_trades]),
        'trades_per_year': len(all_trades) / 2  # 2 years of data
    }


def main():
    print("=" * 70)
    print("PENALTIES-ONLY THRESHOLD OPTIMIZATION")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    calc = PenaltiesOnlyCalculator()
    print(f"Loaded penalties-only calculator with {calc.penalty_count} penalties")
    print()

    tickers = [
        'NVDA', 'AVGO', 'MSFT', 'JPM', 'JNJ', 'XOM', 'CAT',
        'ORCL', 'MRVL', 'INSM', 'SCHW', 'COP', 'ETN', 'ABBV',
        'MPC', 'V', 'KLAC', 'GILD', 'TXN', 'CVS'  # Added more stocks
    ]

    thresholds = [55, 60, 65, 70, 75, 80]

    print(f"Testing {len(tickers)} stocks across {len(thresholds)} thresholds")
    print()

    results = []

    for threshold in thresholds:
        print(f"Testing threshold {threshold}...", end=" ")
        result = backtest_threshold(tickers, calc, threshold)
        results.append(result)
        print(f"{result['trades']} trades, {result['win_rate']:.1f}% win rate")

    # Results table
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'Threshold':<10} | {'Trades':>8} | {'Trades/Yr':>10} | {'Win Rate':>10} | {'Avg Ret':>10} | {'Total Ret':>10}")
    print("-" * 75)

    for r in results:
        print(f"{r['threshold']:<10} | {r['trades']:>8} | {r['trades_per_year']:>10.0f} | "
              f"{r['win_rate']:>9.1f}% | {r['avg_return']:>9.2f}% | {r['total_return']:>9.1f}%")

    # Find optimal threshold
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Filter to thresholds with reasonable trade count
    viable = [r for r in results if r['trades'] >= 20]

    if not viable:
        print("No viable thresholds (need at least 20 trades)")
        return

    # Best by win rate
    best_wr = max(viable, key=lambda x: x['win_rate'])
    print(f"\nBest by WIN RATE: Threshold {best_wr['threshold']}")
    print(f"  {best_wr['trades']} trades, {best_wr['win_rate']:.1f}% win, {best_wr['avg_return']:.2f}% avg return")

    # Best by avg return
    best_ret = max(viable, key=lambda x: x['avg_return'])
    print(f"\nBest by AVG RETURN: Threshold {best_ret['threshold']}")
    print(f"  {best_ret['trades']} trades, {best_ret['win_rate']:.1f}% win, {best_ret['avg_return']:.2f}% avg return")

    # Best by risk-adjusted (win_rate * avg_return)
    for r in viable:
        r['score'] = r['win_rate'] * r['avg_return'] if r['avg_return'] > 0 else 0

    best_score = max(viable, key=lambda x: x['score'])
    print(f"\nBest by RISK-ADJUSTED: Threshold {best_score['threshold']}")
    print(f"  {best_score['trades']} trades, {best_score['win_rate']:.1f}% win, {best_score['avg_return']:.2f}% avg return")
    print(f"  Score: {best_score['score']:.1f}")

    # Recommendation
    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Find threshold with best balance
    # Want: ~50+ trades/year, 60%+ win rate, positive avg return
    optimal = None
    for r in sorted(viable, key=lambda x: x['threshold']):
        if r['trades_per_year'] >= 25 and r['win_rate'] >= 58 and r['avg_return'] > 0:
            optimal = r
            break

    if optimal:
        print(f"\nOPTIMAL THRESHOLD: {optimal['threshold']}")
        print(f"  Trades/year: {optimal['trades_per_year']:.0f}")
        print(f"  Win rate: {optimal['win_rate']:.1f}%")
        print(f"  Avg return: {optimal['avg_return']:.2f}%")
        print(f"  Avg penalty: {optimal['avg_penalty']:.1f}")
    else:
        # Fall back to best win rate
        print(f"\nFALLBACK - Best Win Rate: {best_wr['threshold']}")
        print(f"  Trades/year: {best_wr['trades_per_year']:.0f}")
        print(f"  Win rate: {best_wr['win_rate']:.1f}%")

    # Save results
    Path('data/research').mkdir(parents=True, exist_ok=True)
    with open('data/research/penalties_threshold_optimization.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'calculator': 'penalties_only',
            'penalty_count': calc.penalty_count,
            'tickers_tested': len(tickers),
            'results': results,
            'recommendation': optimal['threshold'] if optimal else best_wr['threshold']
        }, f, indent=2, default=str)

    print()
    print("Results saved to data/research/penalties_threshold_optimization.json")


if __name__ == '__main__':
    main()

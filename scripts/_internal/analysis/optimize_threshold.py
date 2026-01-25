"""
Threshold Optimization - Find the best signal threshold for trading
Tests thresholds from 65-85 with and without modifiers
"""
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import yfinance as yf

# Add src to path
import sys
sys.path.insert(0, 'src')

from trading.unified_signal_calculator import UnifiedSignalCalculator


def get_stock_data(ticker: str, period: str = '2y') -> dict:
    """Fetch stock data for backtesting."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if len(df) < 50:
            return None
        return df
    except:
        return None


def calculate_indicators(df) -> dict:
    """Calculate technical indicators for a given date."""
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

    # Day range
    day_range = (high[-1] - low[-1]) / close[-1] * 100

    # Close position in range
    close_pos = (close[-1] - low[-1]) / (high[-1] - low[-1] + 0.001)

    # ATR
    tr = np.maximum(high[-14:] - low[-14:],
                    np.abs(high[-14:] - np.roll(close[-14:], 1)))
    atr = np.mean(tr) / close[-1] * 100

    # Gap
    if len(close) >= 2:
        gap = (df['Open'].values[-1] / close[-2] - 1) * 100
    else:
        gap = 0

    return {
        'rsi': rsi,
        'sma200_distance': sma200_distance,
        'consecutive_down': down_days,
        'drawdown': drawdown,
        'volume_ratio': vol_ratio,
        'day_range': day_range,
        'close_position': close_pos,
        'atr': atr,
        'gap': gap,
        'close': close[-1]
    }


def backtest_threshold(ticker: str, calc: UnifiedSignalCalculator,
                       threshold: int, use_modifiers: bool = True):
    """Backtest a single stock with specific threshold."""
    df = get_stock_data(ticker, '2y')
    if df is None or len(df) < 250:
        return None

    tier = calc.get_tier(ticker)
    sector_map = {
        'NVDA': 'Technology', 'AVGO': 'Technology', 'MSFT': 'Technology',
        'JPM': 'Financials', 'MS': 'Financials', 'SCHW': 'Financials',
        'JNJ': 'Healthcare', 'ABBV': 'Healthcare', 'INSM': 'Healthcare',
        'XOM': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
        'CAT': 'Industrials', 'ETN': 'Industrials', 'LMT': 'Industrials',
    }
    sector = sector_map.get(ticker, 'Technology')

    trades = []

    for i in range(250, len(df) - 5):
        sub_df = df.iloc[:i+1]
        indicators = calculate_indicators(sub_df)

        date = df.index[i]
        dow = date.weekday()

        # Generate base signal
        base_signal = 50 + (30 - indicators['rsi']) * 0.5 + abs(indicators['drawdown']) * 1.5
        base_signal = max(30, min(80, base_signal))

        if use_modifiers:
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
                rsi_level=indicators['rsi'],
                volume_ratio=indicators['volume_ratio'],
                is_down_day=indicators['consecutive_down'] > 0,
                sector=sector,
                sector_momentum=0,
                close_position=indicators['close_position'],
                gap_pct=indicators['gap'],
                sma200_distance=indicators['sma200_distance'],
                day_range_pct=indicators['day_range'],
                drawdown_pct=indicators['drawdown'],
                atr_pct=indicators['atr']
            )
            final_signal = result.final_signal
        else:
            tier_mult = calc.get_tier_multiplier(tier)
            final_signal = base_signal * tier_mult

        # Trade if signal >= threshold
        if final_signal >= threshold:
            entry_price = indicators['close']
            exit_price = df['Close'].iloc[i + 2]
            return_pct = (exit_price / entry_price - 1) * 100
            win = return_pct > 0

            trades.append({
                'signal': final_signal,
                'return': return_pct,
                'win': win
            })

    if not trades:
        return {'trades': 0, 'wins': 0, 'win_rate': 0, 'avg_return': 0}

    return {
        'trades': len(trades),
        'wins': sum(1 for t in trades if t['win']),
        'win_rate': sum(1 for t in trades if t['win']) / len(trades) * 100,
        'avg_return': np.mean([t['return'] for t in trades])
    }


def main():
    print("=" * 70)
    print("THRESHOLD OPTIMIZATION TEST")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    calc = UnifiedSignalCalculator()

    tickers = [
        'NVDA', 'AVGO', 'MSFT', 'JPM', 'JNJ', 'XOM', 'CAT',
        'ORCL', 'MRVL', 'INSM', 'SCHW', 'COP', 'ETN', 'ABBV'
    ]

    # Cache stock data
    print(f"Loading data for {len(tickers)} stocks...")

    thresholds = [65, 70, 75, 80, 85]

    results = {}

    for threshold in thresholds:
        print(f"\nTesting threshold {threshold}...")

        with_mods = {'trades': 0, 'wins': 0, 'returns': []}
        without_mods = {'trades': 0, 'wins': 0, 'returns': []}

        for ticker in tickers:
            # With modifiers
            r = backtest_threshold(ticker, calc, threshold, use_modifiers=True)
            if r:
                with_mods['trades'] += r['trades']
                with_mods['wins'] += r['wins']
                if r['avg_return']:
                    with_mods['returns'].append(r['avg_return'])

            # Without modifiers
            r = backtest_threshold(ticker, calc, threshold, use_modifiers=False)
            if r:
                without_mods['trades'] += r['trades']
                without_mods['wins'] += r['wins']
                if r['avg_return']:
                    without_mods['returns'].append(r['avg_return'])

        with_wr = with_mods['wins'] / with_mods['trades'] * 100 if with_mods['trades'] > 0 else 0
        without_wr = without_mods['wins'] / without_mods['trades'] * 100 if without_mods['trades'] > 0 else 0
        with_avg = np.mean(with_mods['returns']) if with_mods['returns'] else 0
        without_avg = np.mean(without_mods['returns']) if without_mods['returns'] else 0

        results[threshold] = {
            'with_modifiers': {
                'trades': with_mods['trades'],
                'win_rate': with_wr,
                'avg_return': with_avg
            },
            'without_modifiers': {
                'trades': without_mods['trades'],
                'win_rate': without_wr,
                'avg_return': without_avg
            }
        }

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Threshold':<10} | {'WITH MODIFIERS':^30} | {'WITHOUT MODIFIERS':^30}")
    print(f"{'':10} | {'Trades':>8} {'WinRate':>10} {'AvgRet':>10} | {'Trades':>8} {'WinRate':>10} {'AvgRet':>10}")
    print("-" * 70)

    for threshold in thresholds:
        r = results[threshold]
        wm = r['with_modifiers']
        wo = r['without_modifiers']
        print(f"{threshold:<10} | {wm['trades']:>8} {wm['win_rate']:>9.1f}% {wm['avg_return']:>9.2f}% | "
              f"{wo['trades']:>8} {wo['win_rate']:>9.1f}% {wo['avg_return']:>9.2f}%")

    print()
    print("ANALYSIS:")

    # Find best threshold with modifiers
    best_threshold = max(results.keys(),
                        key=lambda t: results[t]['with_modifiers']['win_rate']
                                      if results[t]['with_modifiers']['trades'] >= 50 else 0)

    r = results[best_threshold]
    print(f"  Best threshold WITH modifiers: {best_threshold}")
    print(f"    {r['with_modifiers']['trades']} trades, "
          f"{r['with_modifiers']['win_rate']:.1f}% win, "
          f"{r['with_modifiers']['avg_return']:.2f}% avg return")

    # Compare to without modifiers at same threshold
    wo = r['without_modifiers']
    print(f"  Without modifiers at {best_threshold}:")
    print(f"    {wo['trades']} trades, {wo['win_rate']:.1f}% win, {wo['avg_return']:.2f}% avg return")

    # Save results
    Path('data/research').mkdir(parents=True, exist_ok=True)
    with open('data/research/threshold_optimization.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'thresholds_tested': thresholds,
            'results': results,
            'recommendation': best_threshold
        }, f, indent=2)

    print()
    print("Results saved to data/research/threshold_optimization.json")


if __name__ == '__main__':
    main()

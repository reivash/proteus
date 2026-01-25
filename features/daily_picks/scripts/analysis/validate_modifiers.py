"""
Validate Modifier System - Compare performance WITH vs WITHOUT modifiers
This tells us if our 113 modifiers are actually improving performance.
"""
import json
import numpy as np
from datetime import datetime, timedelta
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


def backtest_stock(ticker: str, calc: UnifiedSignalCalculator, use_modifiers: bool = True):
    """Backtest a single stock with or without modifiers."""
    df = get_stock_data(ticker, '2y')
    if df is None or len(df) < 250:
        return None

    # Get stock tier and sector
    tier = calc.get_tier(ticker)

    # Simulate sector (simplified)
    sector_map = {
        'NVDA': 'Technology', 'AVGO': 'Technology', 'MSFT': 'Technology',
        'JPM': 'Financials', 'MS': 'Financials', 'SCHW': 'Financials',
        'JNJ': 'Healthcare', 'ABBV': 'Healthcare', 'INSM': 'Healthcare',
        'XOM': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
        'CAT': 'Industrials', 'ETN': 'Industrials', 'LMT': 'Industrials',
    }
    sector = sector_map.get(ticker, 'Technology')

    trades = []

    # Walk through data
    for i in range(250, len(df) - 5):  # Need 5 days for exit
        # Get indicators for this day
        sub_df = df.iloc[:i+1]
        indicators = calculate_indicators(sub_df)

        # Day of week
        date = df.index[i]
        dow = date.weekday()
        is_monday = dow == 0
        is_tuesday = dow == 1
        is_wednesday = dow == 2
        is_thursday = dow == 3
        is_friday = dow == 4

        # Generate base signal (simplified - would use actual model)
        # Use RSI and drawdown as proxy for base signal
        base_signal = 50 + (30 - indicators['rsi']) * 0.5 + abs(indicators['drawdown']) * 1.5
        base_signal = max(30, min(80, base_signal))

        if use_modifiers:
            # Use full modifier system
            result = calc.calculate(
                ticker=ticker,
                base_signal=base_signal,
                regime='choppy',
                is_monday=is_monday,
                is_tuesday=is_tuesday,
                is_wednesday=is_wednesday,
                is_thursday=is_thursday,
                is_friday=is_friday,
                consecutive_down_days=indicators['consecutive_down'],
                rsi_level=indicators['rsi'],
                volume_ratio=indicators['volume_ratio'],
                is_down_day=indicators['consecutive_down'] > 0,
                sector=sector,
                sector_momentum=0,  # Simplified
                close_position=indicators['close_position'],
                gap_pct=indicators['gap'],
                sma200_distance=indicators['sma200_distance'],
                day_range_pct=indicators['day_range'],
                drawdown_pct=indicators['drawdown'],
                atr_pct=indicators['atr']
            )
            final_signal = result.final_signal
        else:
            # No modifiers - just use base signal with tier/regime mult
            tier_mult = calc.get_tier_multiplier(tier)
            final_signal = base_signal * tier_mult

        # Only trade if signal > 70
        if final_signal >= 70:
            entry_price = indicators['close']

            # Check 2-day return
            exit_price = df['Close'].iloc[i + 2]
            return_pct = (exit_price / entry_price - 1) * 100
            win = return_pct > 0

            trades.append({
                'date': date,
                'signal': final_signal,
                'return': return_pct,
                'win': win
            })

    if not trades:
        return None

    return {
        'ticker': ticker,
        'trades': len(trades),
        'wins': sum(1 for t in trades if t['win']),
        'win_rate': sum(1 for t in trades if t['win']) / len(trades) * 100,
        'avg_return': np.mean([t['return'] for t in trades]),
        'total_return': sum(t['return'] for t in trades)
    }


def main():
    print("=" * 70)
    print("PROTEUS MODIFIER VALIDATION BACKTEST")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    calc = UnifiedSignalCalculator()

    # Test stocks
    tickers = [
        'NVDA', 'AVGO', 'MSFT', 'JPM', 'JNJ', 'XOM', 'CAT',
        'ORCL', 'MRVL', 'INSM', 'SCHW', 'COP', 'ETN', 'ABBV'
    ]

    print(f"Testing {len(tickers)} stocks...")
    print()

    # Backtest WITH modifiers
    print("Running WITH modifiers (113 active)...")
    results_with = []
    for ticker in tickers:
        result = backtest_stock(ticker, calc, use_modifiers=True)
        if result:
            results_with.append(result)
            print(f"  {ticker}: {result['trades']} trades, {result['win_rate']:.1f}% win")

    print()

    # Backtest WITHOUT modifiers
    print("Running WITHOUT modifiers (baseline)...")
    results_without = []
    for ticker in tickers:
        result = backtest_stock(ticker, calc, use_modifiers=False)
        if result:
            results_without.append(result)
            print(f"  {ticker}: {result['trades']} trades, {result['win_rate']:.1f}% win")

    print()
    print("=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    # Aggregate results
    if results_with and results_without:
        with_trades = sum(r['trades'] for r in results_with)
        with_wins = sum(r['wins'] for r in results_with)
        with_wr = with_wins / with_trades * 100 if with_trades > 0 else 0
        with_avg = np.mean([r['avg_return'] for r in results_with])

        without_trades = sum(r['trades'] for r in results_without)
        without_wins = sum(r['wins'] for r in results_without)
        without_wr = without_wins / without_trades * 100 if without_trades > 0 else 0
        without_avg = np.mean([r['avg_return'] for r in results_without])

        print()
        print(f"{'Metric':<25} | {'WITH Modifiers':>15} | {'WITHOUT':>15} | {'Diff':>10}")
        print("-" * 70)
        print(f"{'Total Trades':<25} | {with_trades:>15} | {without_trades:>15} | {with_trades - without_trades:>+10}")
        print(f"{'Win Rate':<25} | {with_wr:>14.1f}% | {without_wr:>14.1f}% | {with_wr - without_wr:>+9.1f}%")
        print(f"{'Avg Return/Trade':<25} | {with_avg:>14.2f}% | {without_avg:>14.2f}% | {with_avg - without_avg:>+9.2f}%")

        print()
        print("CONCLUSION:")
        if with_wr > without_wr:
            print(f"  [+] Modifiers IMPROVE win rate by {with_wr - without_wr:.1f}pp")
        else:
            print(f"  [-] Modifiers DECREASE win rate by {without_wr - with_wr:.1f}pp")

        if with_avg > without_avg:
            print(f"  [+] Modifiers IMPROVE avg return by {with_avg - without_avg:.2f}%")
        else:
            print(f"  [-] Modifiers DECREASE avg return by {without_avg - with_avg:.2f}%")

        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'with_modifiers': {
                'trades': with_trades,
                'win_rate': with_wr,
                'avg_return': with_avg,
                'stocks': results_with
            },
            'without_modifiers': {
                'trades': without_trades,
                'win_rate': without_wr,
                'avg_return': without_avg,
                'stocks': results_without
            },
            'improvement': {
                'win_rate': with_wr - without_wr,
                'avg_return': with_avg - without_avg
            }
        }

        Path('data/research').mkdir(parents=True, exist_ok=True)
        with open('data/research/modifier_validation.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print()
        print("Results saved to data/research/modifier_validation.json")


if __name__ == '__main__':
    main()

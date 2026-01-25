"""
Volume Profile Analysis for Mean Reversion Signals

Investigates whether volume patterns at signal time affect mean reversion performance.

Hypothesis:
- High volume on the oversold day may indicate capitulation/panic selling -> stronger bounce
- Low volume may indicate orderly selling -> weaker bounce
- Volume spike relative to average may be a key indicator

This analysis will determine optimal volume-based adjustments.
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
from pathlib import Path


# All stocks in the trading universe
TICKERS = [
    'NVDA', 'AVGO', 'MSFT', 'ORCL', 'INTU', 'ADBE', 'CRM', 'NOW', 'AMAT', 'KLAC',
    'MRVL', 'QCOM', 'TXN', 'ADI', 'JPM', 'MS', 'SCHW', 'AXP', 'AIG', 'USB',
    'PNC', 'V', 'MA', 'ABBV', 'SYK', 'GILD', 'PFE', 'JNJ', 'CVS', 'HCA',
    'IDXX', 'INSM', 'COP', 'SLB', 'XOM', 'MPC', 'EOG', 'CAT', 'ETN', 'LMT',
    'ROAD', 'HD', 'LOW', 'TGT', 'WMT', 'TMUS', 'CMCSA', 'META', 'APD', 'ECL',
    'MLM', 'SHW', 'EXR', 'NEE'
]


def calculate_rsi(prices, period=14):
    """Calculate RSI."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_volume_ratio(df, position, lookback=20):
    """Calculate current volume relative to average."""
    if position < lookback:
        return 1.0
    avg_volume = df['Volume'].iloc[position - lookback:position].mean()
    current_volume = df['Volume'].iloc[position]
    if avg_volume > 0:
        return current_volume / avg_volume
    return 1.0


def calculate_volume_trend(df, position, lookback=5):
    """Calculate volume trend over past N days."""
    if position < lookback:
        return 0
    volumes = df['Volume'].iloc[position - lookback:position + 1].values
    # Simple linear trend: positive = increasing, negative = decreasing
    x = np.arange(len(volumes))
    if len(x) > 1 and np.std(volumes) > 0:
        slope = np.polyfit(x, volumes, 1)[0]
        # Normalize by average volume
        return slope / np.mean(volumes) * 100
    return 0


def run_volume_profile_analysis(days_back=365):
    """Analyze volume profile effect on signal performance."""

    print('=' * 70)
    print('VOLUME PROFILE ANALYSIS')
    print('=' * 70)
    print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'Period: {days_back} days')
    print()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 50)

    # Collect all signals with volume info
    results = []

    print(f'Analyzing {len(TICKERS)} stocks...')

    for i, ticker in enumerate(TICKERS):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if len(df) < 50:
                continue

            # Calculate RSI
            df['RSI'] = calculate_rsi(df['Close'])

            # Find RSI < 35 signals
            signals = df[df['RSI'] < 35].copy()

            for idx in signals.index:
                try:
                    pos = df.index.get_loc(idx)
                    if pos + 5 > len(df) or pos < 20:
                        continue

                    entry_price = df['Close'].iloc[pos]

                    # Calculate volume metrics
                    volume_ratio = calculate_volume_ratio(df, pos)
                    volume_trend = calculate_volume_trend(df, pos)

                    # Price range on signal day (volatility indicator)
                    day_range = ((df['High'].iloc[pos] - df['Low'].iloc[pos]) / df['Close'].iloc[pos]) * 100

                    # Volume on down move (if it was a down day)
                    is_down_day = df['Close'].iloc[pos] < df['Open'].iloc[pos]

                    # Calculate forward returns
                    exit_price_2d = df['Close'].iloc[min(pos + 2, len(df)-1)]
                    ret_2d = ((exit_price_2d / entry_price) - 1) * 100

                    exit_price_3d = df['Close'].iloc[min(pos + 3, len(df)-1)]
                    ret_3d = ((exit_price_3d / entry_price) - 1) * 100

                    future = df.iloc[pos:min(pos+5, len(df))]
                    max_gain = ((future['High'].max() / entry_price) - 1) * 100
                    max_loss = ((future['Low'].min() / entry_price) - 1) * 100

                    # Categorize volume ratio
                    if volume_ratio >= 2.5:
                        vol_category = 'very_high'
                    elif volume_ratio >= 1.5:
                        vol_category = 'high'
                    elif volume_ratio >= 0.8:
                        vol_category = 'normal'
                    else:
                        vol_category = 'low'

                    # Categorize day range (volatility)
                    if day_range >= 4.0:
                        range_category = 'very_wide'
                    elif day_range >= 2.5:
                        range_category = 'wide'
                    elif day_range >= 1.5:
                        range_category = 'normal'
                    else:
                        range_category = 'narrow'

                    results.append({
                        'ticker': ticker,
                        'date': idx.strftime('%Y-%m-%d'),
                        'rsi': df['RSI'].iloc[pos],
                        'volume_ratio': volume_ratio,
                        'volume_category': vol_category,
                        'volume_trend': volume_trend,
                        'day_range': day_range,
                        'range_category': range_category,
                        'is_down_day': is_down_day,
                        'ret_2d': ret_2d,
                        'ret_3d': ret_3d,
                        'max_gain': max_gain,
                        'max_loss': max_loss,
                        'win_2d': ret_2d > 0,
                        'hit_target': max_gain >= 2.0
                    })

                except:
                    continue

            if (i + 1) % 10 == 0:
                print(f'  Processed {i+1}/{len(TICKERS)} stocks ({len(results)} signals)')

        except Exception as e:
            continue

    if not results:
        print('No signals found!')
        return None

    df = pd.DataFrame(results)

    print()
    print(f'Total signals: {len(df)}')
    print()

    # Analyze by volume ratio
    print('=' * 70)
    print('RESULTS BY VOLUME RATIO')
    print('=' * 70)
    print()

    summary_by_volume = {}

    for cat in ['low', 'normal', 'high', 'very_high']:
        cat_df = df[df['volume_category'] == cat]
        if len(cat_df) > 0:
            n = len(cat_df)
            win_rate = cat_df['win_2d'].mean() * 100
            hit_rate = cat_df['hit_target'].mean() * 100
            avg_ret = cat_df['ret_2d'].mean()
            avg_vol_ratio = cat_df['volume_ratio'].mean()

            summary_by_volume[cat] = {
                'signals': n,
                'win_rate': round(win_rate, 1),
                'hit_rate': round(hit_rate, 1),
                'avg_ret': round(avg_ret, 3),
                'avg_volume_ratio': round(avg_vol_ratio, 2)
            }

            print(f'{cat.upper()} VOLUME (avg ratio: {avg_vol_ratio:.2f}x):')
            print(f'  Signals: {n}')
            print(f'  Win rate: {win_rate:.1f}%')
            print(f'  Hit rate (2%): {hit_rate:.1f}%')
            print(f'  Avg 2-day return: {avg_ret:+.3f}%')
            print()

    # Analyze by day range (volatility)
    print('=' * 70)
    print('RESULTS BY DAY RANGE (VOLATILITY)')
    print('=' * 70)
    print()

    summary_by_range = {}

    for cat in ['narrow', 'normal', 'wide', 'very_wide']:
        cat_df = df[df['range_category'] == cat]
        if len(cat_df) > 0:
            n = len(cat_df)
            win_rate = cat_df['win_2d'].mean() * 100
            hit_rate = cat_df['hit_target'].mean() * 100
            avg_ret = cat_df['ret_2d'].mean()
            avg_range = cat_df['day_range'].mean()

            summary_by_range[cat] = {
                'signals': n,
                'win_rate': round(win_rate, 1),
                'hit_rate': round(hit_rate, 1),
                'avg_ret': round(avg_ret, 3),
                'avg_range': round(avg_range, 2)
            }

            print(f'{cat.upper()} RANGE (avg: {avg_range:.2f}%):')
            print(f'  Signals: {n}')
            print(f'  Win rate: {win_rate:.1f}%')
            print(f'  Hit rate (2%): {hit_rate:.1f}%')
            print(f'  Avg 2-day return: {avg_ret:+.3f}%')
            print()

    # Cross analysis: Volume x Range
    print('=' * 70)
    print('VOLUME x RANGE CROSS-ANALYSIS')
    print('=' * 70)
    print()

    cross_analysis = []
    for vol_cat in ['low', 'normal', 'high', 'very_high']:
        for range_cat in ['narrow', 'normal', 'wide', 'very_wide']:
            subset = df[(df['volume_category'] == vol_cat) & (df['range_category'] == range_cat)]
            if len(subset) >= 10:
                cross_analysis.append({
                    'volume': vol_cat,
                    'range': range_cat,
                    'signals': len(subset),
                    'win_rate': subset['win_2d'].mean() * 100,
                    'avg_ret': subset['ret_2d'].mean()
                })

    # Sort by avg return
    cross_analysis.sort(key=lambda x: x['avg_ret'], reverse=True)

    print('TOP 10 VOLUME + RANGE COMBINATIONS:')
    for i, combo in enumerate(cross_analysis[:10]):
        print(f'  {i+1}. {combo["volume"]} vol + {combo["range"]} range: n={combo["signals"]}, '
              f'win={combo["win_rate"]:.0f}%, ret={combo["avg_ret"]:+.3f}%')

    print()
    print('BOTTOM 5 COMBINATIONS:')
    for i, combo in enumerate(cross_analysis[-5:]):
        print(f'  {i+1}. {combo["volume"]} vol + {combo["range"]} range: n={combo["signals"]}, '
              f'win={combo["win_rate"]:.0f}%, ret={combo["avg_ret"]:+.3f}%')

    # Capitulation analysis: High volume + wide range + down day
    print()
    print('=' * 70)
    print('CAPITULATION ANALYSIS')
    print('=' * 70)
    print()

    capitulation_df = df[
        (df['volume_ratio'] >= 1.5) &
        (df['day_range'] >= 2.0) &
        (df['is_down_day'])
    ]

    normal_df = df[
        (df['volume_ratio'] < 1.5) &
        (df['day_range'] < 2.0)
    ]

    if len(capitulation_df) > 0 and len(normal_df) > 0:
        cap_win = capitulation_df['win_2d'].mean() * 100
        cap_ret = capitulation_df['ret_2d'].mean()
        cap_hit = capitulation_df['hit_target'].mean() * 100

        norm_win = normal_df['win_2d'].mean() * 100
        norm_ret = normal_df['ret_2d'].mean()
        norm_hit = normal_df['hit_target'].mean() * 100

        print(f'CAPITULATION SIGNALS (high vol + wide range + down day):')
        print(f'  Signals: {len(capitulation_df)}')
        print(f'  Win rate: {cap_win:.1f}%')
        print(f'  Hit rate (2%): {cap_hit:.1f}%')
        print(f'  Avg 2-day return: {cap_ret:+.3f}%')
        print()
        print(f'NORMAL SIGNALS (low/normal vol + narrow range):')
        print(f'  Signals: {len(normal_df)}')
        print(f'  Win rate: {norm_win:.1f}%')
        print(f'  Hit rate (2%): {norm_hit:.1f}%')
        print(f'  Avg 2-day return: {norm_ret:+.3f}%')
        print()

        edge = cap_ret - norm_ret
        if edge > 0:
            print(f'>>> CAPITULATION EDGE: {edge:+.3f}% per trade <<<')
        else:
            print(f'>>> NO CAPITULATION EDGE: {edge:+.3f}% vs normal <<<')

    # Calculate strength adjustments
    print()
    print('=' * 70)
    print('RECOMMENDED ADJUSTMENTS')
    print('=' * 70)
    print()

    # Use normal volume as baseline
    baseline_ret = summary_by_volume.get('normal', {}).get('avg_ret', 0)
    print(f'Baseline (normal volume) avg return: {baseline_ret:+.3f}%')
    print()

    adjustments = {}
    for cat, stats in summary_by_volume.items():
        diff = stats['avg_ret'] - baseline_ret
        if diff > 0.15:
            adj = 1.10  # 10% boost
        elif diff > 0.08:
            adj = 1.05  # 5% boost
        elif diff < -0.15:
            adj = 0.90  # 10% penalty
        elif diff < -0.08:
            adj = 0.95  # 5% penalty
        else:
            adj = 1.0

        adjustments[cat] = {
            'diff_from_baseline': round(diff, 3),
            'strength_multiplier': adj
        }

        adj_str = f"{adj:.2f}x" if adj != 1.0 else "1.0x (no change)"
        diff_str = f"{diff:+.3f}%"
        print(f'{cat} volume: {diff_str} vs baseline -> {adj_str}')

    # Key insights
    print()
    print('=' * 70)
    print('KEY INSIGHTS')
    print('=' * 70)
    print()

    # Find best and worst
    best_cat = max(summary_by_volume.items(), key=lambda x: x[1]['avg_ret'])
    worst_cat = min(summary_by_volume.items(), key=lambda x: x[1]['avg_ret'])

    print(f'Best volume profile: {best_cat[0]}')
    print(f'  Win rate: {best_cat[1]["win_rate"]}%')
    print(f'  Avg return: {best_cat[1]["avg_ret"]:+.3f}%')
    print()
    print(f'Worst volume profile: {worst_cat[0]}')
    print(f'  Win rate: {worst_cat[1]["win_rate"]}%')
    print(f'  Avg return: {worst_cat[1]["avg_ret"]:+.3f}%')
    print()

    edge = best_cat[1]['avg_ret'] - worst_cat[1]['avg_ret']
    print(f'>>> EDGE FROM VOLUME SELECTION: {edge:+.3f}% per trade <<<')

    # Volume trend correlation
    print()
    print('VOLUME TREND EFFECT:')
    correlation = df['volume_trend'].corr(df['ret_2d'])
    print(f'  Correlation (volume trend vs 2d return): {correlation:.3f}')

    if correlation > 0.05:
        print('  >>> Increasing volume before signal helps <<<')
    elif correlation < -0.05:
        print('  >>> Decreasing volume before signal helps <<<')
    else:
        print('  >>> Volume trend has minimal effect <<<')

    # Save results
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'days_analyzed': days_back,
        'total_signals': len(df),
        'baseline_return': baseline_ret,
        'by_volume_ratio': summary_by_volume,
        'by_day_range': summary_by_range,
        'adjustments': adjustments,
        'best_volume_profile': best_cat[0],
        'worst_volume_profile': worst_cat[0],
        'edge_from_volume_selection': edge,
        'volume_trend_correlation': correlation,
        'capitulation_signals': len(capitulation_df) if 'capitulation_df' in dir() else 0,
        'top_combinations': cross_analysis[:10],
        'bottom_combinations': cross_analysis[-5:]
    }

    output_path = Path('data/research/volume_profile_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print()
    print(f'Results saved to {output_path}')

    return results_data


if __name__ == '__main__':
    run_volume_profile_analysis(days_back=365)

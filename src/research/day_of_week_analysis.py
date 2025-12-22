"""
Day-of-Week Analysis for Mean Reversion Signals

Investigates whether RSI<35 signals perform differently on specific days of the week.

Hypothesis:
- Monday: Potential "weekend effect" - stocks may gap down Monday morning after negative weekend news
- Tuesday-Thursday: Normal trading days, should be baseline
- Friday: Pre-weekend caution may affect signal performance

This analysis will determine optimal day-of-week adjustments for the signal strength formula.
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

DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


def calculate_rsi(prices, period=14):
    """Calculate RSI."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def run_day_of_week_analysis(days_back=365):
    """Analyze day-of-week effect on signal performance."""

    print('=' * 70)
    print('DAY-OF-WEEK ANALYSIS')
    print('=' * 70)
    print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'Period: {days_back} days')
    print()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 50)

    # Collect all signals with day-of-week info
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
                    if pos + 5 > len(df):
                        continue

                    entry_price = df['Close'].iloc[pos]
                    day_of_week = idx.dayofweek  # 0=Monday, 4=Friday

                    # Skip weekends (shouldn't happen but just in case)
                    if day_of_week > 4:
                        continue

                    # Calculate forward returns
                    exit_price_2d = df['Close'].iloc[min(pos + 2, len(df)-1)]
                    ret_2d = ((exit_price_2d / entry_price) - 1) * 100

                    exit_price_3d = df['Close'].iloc[min(pos + 3, len(df)-1)]
                    ret_3d = ((exit_price_3d / entry_price) - 1) * 100

                    future = df.iloc[pos:min(pos+5, len(df))]
                    max_gain = ((future['High'].max() / entry_price) - 1) * 100
                    max_loss = ((future['Low'].min() / entry_price) - 1) * 100

                    # Check next day return (day 1)
                    if pos + 1 < len(df):
                        ret_1d = ((df['Close'].iloc[pos + 1] / entry_price) - 1) * 100
                    else:
                        ret_1d = 0

                    results.append({
                        'ticker': ticker,
                        'date': idx.strftime('%Y-%m-%d'),
                        'day_of_week': day_of_week,
                        'day_name': DAY_NAMES[day_of_week],
                        'rsi': df['RSI'].iloc[pos],
                        'ret_1d': ret_1d,
                        'ret_2d': ret_2d,
                        'ret_3d': ret_3d,
                        'max_gain': max_gain,
                        'max_loss': max_loss,
                        'win_1d': ret_1d > 0,
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

    # Analyze by day of week
    print('=' * 70)
    print('RESULTS BY DAY OF WEEK')
    print('=' * 70)
    print()

    summary = {}

    for dow in range(5):
        day_df = df[df['day_of_week'] == dow]
        if len(day_df) > 0:
            day_name = DAY_NAMES[dow]
            n = len(day_df)
            win_rate_1d = day_df['win_1d'].mean() * 100
            win_rate_2d = day_df['win_2d'].mean() * 100
            hit_rate = day_df['hit_target'].mean() * 100
            avg_ret_1d = day_df['ret_1d'].mean()
            avg_ret_2d = day_df['ret_2d'].mean()
            avg_ret_3d = day_df['ret_3d'].mean()
            avg_max_gain = day_df['max_gain'].mean()
            avg_max_loss = day_df['max_loss'].mean()

            summary[day_name] = {
                'signals': n,
                'win_rate_1d': round(win_rate_1d, 1),
                'win_rate_2d': round(win_rate_2d, 1),
                'hit_rate': round(hit_rate, 1),
                'avg_ret_1d': round(avg_ret_1d, 3),
                'avg_ret_2d': round(avg_ret_2d, 3),
                'avg_ret_3d': round(avg_ret_3d, 3),
                'avg_max_gain': round(avg_max_gain, 2),
                'avg_max_loss': round(avg_max_loss, 2)
            }

            print(f'{day_name.upper()}:')
            print(f'  Signals: {n}')
            print(f'  Win rate (1d): {win_rate_1d:.1f}%')
            print(f'  Win rate (2d): {win_rate_2d:.1f}%')
            print(f'  Hit rate (2%): {hit_rate:.1f}%')
            print(f'  Avg return (1d): {avg_ret_1d:+.3f}%')
            print(f'  Avg return (2d): {avg_ret_2d:+.3f}%')
            print(f'  Avg return (3d): {avg_ret_3d:+.3f}%')
            print()

    # Calculate strength adjustments
    print('=' * 70)
    print('RECOMMENDED ADJUSTMENTS')
    print('=' * 70)
    print()

    # Use Tuesday-Thursday as baseline
    baseline_days = ['Tuesday', 'Wednesday', 'Thursday']
    baseline_ret = np.mean([summary[d]['avg_ret_2d'] for d in baseline_days if d in summary])

    print(f'Baseline (Tue-Thu) avg return: {baseline_ret:+.3f}%')
    print()

    adjustments = {}
    for day_name, stats in summary.items():
        diff = stats['avg_ret_2d'] - baseline_ret
        if diff > 0.10:
            adj = 1.05  # 5% boost
        elif diff > 0.05:
            adj = 1.02  # 2% boost
        elif diff < -0.10:
            adj = 0.95  # 5% penalty
        elif diff < -0.05:
            adj = 0.98  # 2% penalty
        else:
            adj = 1.0

        adjustments[day_name] = {
            'diff_from_baseline': round(diff, 3),
            'strength_multiplier': adj
        }

        adj_str = f"{adj:.2f}x" if adj != 1.0 else "1.0x (no change)"
        diff_str = f"{diff:+.3f}%"
        print(f'{day_name}: {diff_str} vs baseline -> {adj_str}')

    # Cross-analysis: Day + Tier
    print()
    print('=' * 70)
    print('DAY x STRENGTH CROSS-ANALYSIS')
    print('=' * 70)
    print()

    # Create strength tiers based on RSI
    df['strength_tier'] = df['rsi'].apply(
        lambda x: 'very_oversold' if x < 25 else ('moderately_oversold' if x < 30 else 'slightly_oversold')
    )

    cross_analysis = []
    for dow in range(5):
        day_name = DAY_NAMES[dow]
        for tier in ['very_oversold', 'moderately_oversold', 'slightly_oversold']:
            subset = df[(df['day_of_week'] == dow) & (df['strength_tier'] == tier)]
            if len(subset) >= 10:
                cross_analysis.append({
                    'day': day_name,
                    'tier': tier,
                    'signals': len(subset),
                    'win_rate_2d': subset['win_2d'].mean() * 100,
                    'avg_ret_2d': subset['ret_2d'].mean()
                })

    # Sort by avg return
    cross_analysis.sort(key=lambda x: x['avg_ret_2d'], reverse=True)

    print('TOP 10 DAY + OVERSOLD TIER COMBINATIONS:')
    for i, combo in enumerate(cross_analysis[:10]):
        print(f'  {i+1}. {combo["day"]} + {combo["tier"]}: n={combo["signals"]}, '
              f'win={combo["win_rate_2d"]:.0f}%, ret={combo["avg_ret_2d"]:+.3f}%')

    print()
    print('BOTTOM 5 DAY + OVERSOLD TIER COMBINATIONS:')
    for i, combo in enumerate(cross_analysis[-5:]):
        print(f'  {i+1}. {combo["day"]} + {combo["tier"]}: n={combo["signals"]}, '
              f'win={combo["win_rate_2d"]:.0f}%, ret={combo["avg_ret_2d"]:+.3f}%')

    # Key insights
    print()
    print('=' * 70)
    print('KEY INSIGHTS')
    print('=' * 70)
    print()

    # Find best and worst days
    best_day = max(summary.items(), key=lambda x: x[1]['avg_ret_2d'])
    worst_day = min(summary.items(), key=lambda x: x[1]['avg_ret_2d'])

    print(f'Best day for mean reversion: {best_day[0]}')
    print(f'  Win rate (2d): {best_day[1]["win_rate_2d"]}%')
    print(f'  Avg return (2d): {best_day[1]["avg_ret_2d"]:+.3f}%')
    print()
    print(f'Worst day for mean reversion: {worst_day[0]}')
    print(f'  Win rate (2d): {worst_day[1]["win_rate_2d"]}%')
    print(f'  Avg return (2d): {worst_day[1]["avg_ret_2d"]:+.3f}%')
    print()

    edge = best_day[1]['avg_ret_2d'] - worst_day[1]['avg_ret_2d']
    print(f'>>> EDGE FROM DAY SELECTION: {edge:+.3f}% per trade <<<')

    # Save results
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'days_analyzed': days_back,
        'total_signals': len(df),
        'baseline_return': baseline_ret,
        'by_day': summary,
        'adjustments': adjustments,
        'best_day': best_day[0],
        'worst_day': worst_day[0],
        'edge_from_day_selection': edge,
        'top_combinations': cross_analysis[:10],
        'bottom_combinations': cross_analysis[-5:]
    }

    output_path = Path('data/research/day_of_week_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print()
    print(f'Results saved to {output_path}')

    return results_data


if __name__ == '__main__':
    run_day_of_week_analysis(days_back=365)

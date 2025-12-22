"""
Consecutive Down-Days Analysis for Mean Reversion Signals

Investigates whether stocks with multiple consecutive down days before a signal
perform better for mean reversion than stocks with just 1 down day.

Hypothesis:
- 3+ consecutive down days may indicate panic selling and stronger reversion potential
- 1-2 down days may just be normal pullbacks
- The "rubber band effect" should be stronger after extended selling

This analysis will determine optimal consecutive down-day adjustments.
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


def count_consecutive_down_days(df, position):
    """Count consecutive down days before a given position."""
    count = 0
    for i in range(position - 1, -1, -1):
        if df['Close'].iloc[i] < df['Open'].iloc[i]:
            count += 1
        else:
            break
    return count


def calculate_total_drawdown(df, position, lookback=5):
    """Calculate total drawdown over the past N days."""
    if position < lookback:
        return 0
    start_price = df['Close'].iloc[position - lookback]
    current_price = df['Close'].iloc[position]
    return ((current_price / start_price) - 1) * 100


def run_consecutive_down_days_analysis(days_back=365):
    """Analyze consecutive down-days effect on signal performance."""

    print('=' * 70)
    print('CONSECUTIVE DOWN-DAYS ANALYSIS')
    print('=' * 70)
    print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'Period: {days_back} days')
    print()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 50)

    # Collect all signals with consecutive down-day info
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
                    if pos + 5 > len(df) or pos < 5:
                        continue

                    entry_price = df['Close'].iloc[pos]

                    # Count consecutive down days
                    consecutive_down = count_consecutive_down_days(df, pos)

                    # Calculate total drawdown over past 5 days
                    drawdown_5d = calculate_total_drawdown(df, pos, 5)

                    # Calculate forward returns
                    exit_price_2d = df['Close'].iloc[min(pos + 2, len(df)-1)]
                    ret_2d = ((exit_price_2d / entry_price) - 1) * 100

                    exit_price_3d = df['Close'].iloc[min(pos + 3, len(df)-1)]
                    ret_3d = ((exit_price_3d / entry_price) - 1) * 100

                    future = df.iloc[pos:min(pos+5, len(df))]
                    max_gain = ((future['High'].max() / entry_price) - 1) * 100
                    max_loss = ((future['Low'].min() / entry_price) - 1) * 100

                    # Create category for consecutive down days
                    if consecutive_down >= 5:
                        down_category = '5+'
                    elif consecutive_down >= 3:
                        down_category = '3-4'
                    elif consecutive_down >= 2:
                        down_category = '2'
                    elif consecutive_down >= 1:
                        down_category = '1'
                    else:
                        down_category = '0'

                    # Create category for drawdown severity
                    if drawdown_5d <= -10:
                        drawdown_category = 'severe (-10%+)'
                    elif drawdown_5d <= -5:
                        drawdown_category = 'moderate (-5% to -10%)'
                    elif drawdown_5d <= -3:
                        drawdown_category = 'mild (-3% to -5%)'
                    else:
                        drawdown_category = 'minimal (>-3%)'

                    results.append({
                        'ticker': ticker,
                        'date': idx.strftime('%Y-%m-%d'),
                        'rsi': df['RSI'].iloc[pos],
                        'consecutive_down': consecutive_down,
                        'down_category': down_category,
                        'drawdown_5d': drawdown_5d,
                        'drawdown_category': drawdown_category,
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

    # Analyze by consecutive down days
    print('=' * 70)
    print('RESULTS BY CONSECUTIVE DOWN DAYS')
    print('=' * 70)
    print()

    summary_by_down_days = {}

    for cat in ['0', '1', '2', '3-4', '5+']:
        cat_df = df[df['down_category'] == cat]
        if len(cat_df) > 0:
            n = len(cat_df)
            win_rate = cat_df['win_2d'].mean() * 100
            hit_rate = cat_df['hit_target'].mean() * 100
            avg_ret = cat_df['ret_2d'].mean()
            avg_consec = cat_df['consecutive_down'].mean()

            summary_by_down_days[cat] = {
                'signals': n,
                'win_rate': round(win_rate, 1),
                'hit_rate': round(hit_rate, 1),
                'avg_ret': round(avg_ret, 3),
                'avg_consecutive': round(avg_consec, 1)
            }

            print(f'{cat} CONSECUTIVE DOWN DAYS (avg: {avg_consec:.1f}):')
            print(f'  Signals: {n}')
            print(f'  Win rate: {win_rate:.1f}%')
            print(f'  Hit rate (2%): {hit_rate:.1f}%')
            print(f'  Avg 2-day return: {avg_ret:+.3f}%')
            print()

    # Analyze by drawdown severity
    print('=' * 70)
    print('RESULTS BY 5-DAY DRAWDOWN SEVERITY')
    print('=' * 70)
    print()

    summary_by_drawdown = {}

    for cat in ['minimal (>-3%)', 'mild (-3% to -5%)', 'moderate (-5% to -10%)', 'severe (-10%+)']:
        cat_df = df[df['drawdown_category'] == cat]
        if len(cat_df) > 0:
            n = len(cat_df)
            win_rate = cat_df['win_2d'].mean() * 100
            hit_rate = cat_df['hit_target'].mean() * 100
            avg_ret = cat_df['ret_2d'].mean()
            avg_dd = cat_df['drawdown_5d'].mean()

            summary_by_drawdown[cat] = {
                'signals': n,
                'win_rate': round(win_rate, 1),
                'hit_rate': round(hit_rate, 1),
                'avg_ret': round(avg_ret, 3),
                'avg_drawdown': round(avg_dd, 1)
            }

            print(f'{cat.upper()} (avg dd: {avg_dd:.1f}%):')
            print(f'  Signals: {n}')
            print(f'  Win rate: {win_rate:.1f}%')
            print(f'  Hit rate (2%): {hit_rate:.1f}%')
            print(f'  Avg 2-day return: {avg_ret:+.3f}%')
            print()

    # Cross analysis: Consecutive days x Drawdown severity
    print('=' * 70)
    print('CONSECUTIVE DAYS x DRAWDOWN CROSS-ANALYSIS')
    print('=' * 70)
    print()

    cross_analysis = []
    for down_cat in ['0', '1', '2', '3-4', '5+']:
        for dd_cat in ['minimal (>-3%)', 'mild (-3% to -5%)', 'moderate (-5% to -10%)', 'severe (-10%+)']:
            subset = df[(df['down_category'] == down_cat) & (df['drawdown_category'] == dd_cat)]
            if len(subset) >= 10:
                cross_analysis.append({
                    'down_days': down_cat,
                    'drawdown': dd_cat,
                    'signals': len(subset),
                    'win_rate': subset['win_2d'].mean() * 100,
                    'avg_ret': subset['ret_2d'].mean()
                })

    # Sort by avg return
    cross_analysis.sort(key=lambda x: x['avg_ret'], reverse=True)

    print('TOP 10 CONSECUTIVE DAYS + DRAWDOWN COMBINATIONS:')
    for i, combo in enumerate(cross_analysis[:10]):
        print(f'  {i+1}. {combo["down_days"]}d down + {combo["drawdown"]}: n={combo["signals"]}, '
              f'win={combo["win_rate"]:.0f}%, ret={combo["avg_ret"]:+.3f}%')

    print()
    print('BOTTOM 5 COMBINATIONS:')
    for i, combo in enumerate(cross_analysis[-5:]):
        print(f'  {i+1}. {combo["down_days"]}d down + {combo["drawdown"]}: n={combo["signals"]}, '
              f'win={combo["win_rate"]:.0f}%, ret={combo["avg_ret"]:+.3f}%')

    # Calculate strength adjustments
    print()
    print('=' * 70)
    print('RECOMMENDED ADJUSTMENTS')
    print('=' * 70)
    print()

    # Use 1 day as baseline
    baseline_ret = summary_by_down_days.get('1', {}).get('avg_ret', 0)
    print(f'Baseline (1 down day) avg return: {baseline_ret:+.3f}%')
    print()

    adjustments = {}
    for cat, stats in summary_by_down_days.items():
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
        print(f'{cat} down days: {diff_str} vs baseline -> {adj_str}')

    # Key insights
    print()
    print('=' * 70)
    print('KEY INSIGHTS')
    print('=' * 70)
    print()

    # Find best and worst
    best_cat = max(summary_by_down_days.items(), key=lambda x: x[1]['avg_ret'])
    worst_cat = min(summary_by_down_days.items(), key=lambda x: x[1]['avg_ret'])

    print(f'Best consecutive down pattern: {best_cat[0]} days')
    print(f'  Win rate: {best_cat[1]["win_rate"]}%')
    print(f'  Avg return: {best_cat[1]["avg_ret"]:+.3f}%')
    print()
    print(f'Worst consecutive down pattern: {worst_cat[0]} days')
    print(f'  Win rate: {worst_cat[1]["win_rate"]}%')
    print(f'  Avg return: {worst_cat[1]["avg_ret"]:+.3f}%')
    print()

    edge = best_cat[1]['avg_ret'] - worst_cat[1]['avg_ret']
    print(f'>>> EDGE FROM DOWN-DAY SELECTION: {edge:+.3f}% per trade <<<')

    # Analyze "rubber band" effect - do deeper drawdowns bounce harder?
    print()
    print('RUBBER BAND EFFECT ANALYSIS:')

    # Correlation between drawdown depth and bounce magnitude
    correlation = df['drawdown_5d'].corr(df['ret_2d'])
    print(f'  Correlation (drawdown vs 2d return): {correlation:.3f}')

    if correlation < -0.05:
        print('  >>> Confirmed: Deeper drawdowns lead to stronger bounces <<<')
    elif correlation > 0.05:
        print('  >>> Opposite: Shallow drawdowns bounce better (momentum?) <<<')
    else:
        print('  >>> No clear rubber band effect detected <<<')

    # Save results
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'days_analyzed': days_back,
        'total_signals': len(df),
        'baseline_return': baseline_ret,
        'by_consecutive_down_days': summary_by_down_days,
        'by_drawdown_severity': summary_by_drawdown,
        'adjustments': adjustments,
        'best_pattern': best_cat[0],
        'worst_pattern': worst_cat[0],
        'edge_from_pattern_selection': edge,
        'rubber_band_correlation': correlation,
        'top_combinations': cross_analysis[:10],
        'bottom_combinations': cross_analysis[-5:]
    }

    output_path = Path('data/research/consecutive_down_days_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print()
    print(f'Results saved to {output_path}')

    return results_data


if __name__ == '__main__':
    run_consecutive_down_days_analysis(days_back=365)

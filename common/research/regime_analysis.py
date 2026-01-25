"""
Regime-Specific Signal Performance Analysis

Analyzes how RSI<35 mean reversion signals perform under different market regimes:
- CHOPPY: Sideways market (ideal for mean reversion)
- BULL: Strong uptrend
- BEAR: Strong downtrend
- VOLATILE: High VIX environment
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
from pathlib import Path


def calculate_regime(spy_df, vix_df, date):
    """Determine regime for a specific date."""
    try:
        # Get data up to this date
        spy_slice = spy_df.loc[:date].tail(50)
        vix_slice = vix_df.loc[:date].tail(5)

        if len(spy_slice) < 50 or len(vix_slice) < 1:
            return 'UNKNOWN', 0, 0

        # VIX level
        vix_level = vix_slice['Close'].iloc[-1]

        # SPY trends
        spy_close = spy_slice['Close']
        trend_20d = ((spy_close.iloc[-1] / spy_close.iloc[-20]) - 1) * 100
        trend_50d = ((spy_close.iloc[-1] / spy_close.iloc[-50]) - 1) * 100

        # Regime classification
        if vix_level > 30:
            return 'VOLATILE', vix_level, trend_20d
        elif trend_20d > 3 and trend_50d > 5:
            return 'BULL', vix_level, trend_20d
        elif trend_20d < -3 and trend_50d < -5:
            return 'BEAR', vix_level, trend_20d
        else:
            return 'CHOPPY', vix_level, trend_20d

    except Exception as e:
        return 'UNKNOWN', 0, 0


def run_regime_analysis(days_back=365):
    """Run regime-specific analysis on RSI<35 signals."""

    print('='*70)
    print('REGIME-SPECIFIC SIGNAL PERFORMANCE ANALYSIS')
    print('='*70)
    print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print()

    # Fetch SPY and VIX for 1 year
    print('Fetching market data...')
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 50)

    spy = yf.Ticker('SPY').history(start=start_date, end=end_date)
    vix = yf.Ticker('^VIX').history(start=start_date, end=end_date)

    print(f'SPY data: {len(spy)} days')
    print(f'VIX data: {len(vix)} days')

    # Load stock universe
    tickers = ['NVDA', 'V', 'MA', 'AVGO', 'AXP', 'KLAC', 'ORCL', 'MRVL', 'ABBV', 'SYK',
               'EOG', 'TXN', 'GILD', 'INTU', 'MSFT', 'QCOM', 'JPM', 'JNJ', 'PFE', 'WMT',
               'AMAT', 'ADI', 'NOW', 'MLM', 'IDXX', 'EXR', 'ROAD', 'INSM', 'SCHW', 'AIG',
               'USB', 'CVS', 'LOW', 'LMT', 'COP', 'SLB', 'APD', 'MS', 'PNC', 'CRM',
               'ADBE', 'TGT', 'CAT', 'XOM', 'MPC', 'ECL', 'NEE', 'HCA', 'CMCSA', 'TMUS',
               'META', 'ETN', 'HD', 'SHW']

    # Analyze
    print()
    print('Analyzing RSI<35 signals by regime...')
    print()

    results_by_regime = {'BULL': [], 'BEAR': [], 'CHOPPY': [], 'VOLATILE': []}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if len(df) < 50:
                continue

            # Calculate RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Find RSI < 35 signals
            signals = df[df['RSI'] < 35].copy()

            for idx in signals.index:
                # Get regime at signal date
                regime, vix_level, trend = calculate_regime(spy, vix, idx)

                if regime == 'UNKNOWN':
                    continue

                # Calculate forward returns
                try:
                    pos = df.index.get_loc(idx)
                    if pos + 3 > len(df):
                        continue

                    entry_price = df['Close'].iloc[pos]

                    # 2-day return
                    exit_price = df['Close'].iloc[min(pos + 2, len(df)-1)]
                    ret_2d = ((exit_price / entry_price) - 1) * 100

                    # Max gain/loss in 5 days
                    future = df.iloc[pos:min(pos+5, len(df))]
                    max_gain = ((future['High'].max() / entry_price) - 1) * 100
                    max_loss = ((future['Low'].min() / entry_price) - 1) * 100

                    results_by_regime[regime].append({
                        'ticker': ticker,
                        'date': idx.strftime('%Y-%m-%d'),
                        'rsi': float(df['RSI'].iloc[pos]),
                        'vix': float(vix_level),
                        'trend': float(trend),
                        'ret_2d': float(ret_2d),
                        'max_gain': float(max_gain),
                        'max_loss': float(max_loss),
                        'win': ret_2d > 0
                    })
                except:
                    continue

        except Exception as e:
            continue

    # Print results
    print('='*70)
    print('RESULTS BY MARKET REGIME')
    print('='*70)
    print()

    summary = {}

    for regime in ['CHOPPY', 'BULL', 'BEAR', 'VOLATILE']:
        data = results_by_regime[regime]
        if not data:
            print(f'{regime}: No signals')
            continue

        df = pd.DataFrame(data)
        n = len(df)
        win_rate = df['win'].mean() * 100
        avg_ret = df['ret_2d'].mean()
        avg_max_gain = df['max_gain'].mean()
        avg_max_loss = df['max_loss'].mean()
        hit_rate = (df['max_gain'] >= 2).mean() * 100  # 2% target hit rate

        # Calculate expectancy
        wins = df[df['ret_2d'] > 0]['ret_2d']
        losses = df[df['ret_2d'] <= 0]['ret_2d']
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

        summary[regime] = {
            'signals': n,
            'win_rate': round(win_rate, 1),
            'hit_rate': round(hit_rate, 1),
            'avg_ret_2d': round(avg_ret, 2),
            'avg_max_gain': round(avg_max_gain, 2),
            'avg_max_loss': round(avg_max_loss, 2),
            'expectancy': round(expectancy, 2),
            'avg_vix': round(df['vix'].mean(), 1),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2)
        }

        print(f'{regime}:')
        print(f'  Signals: {n}')
        print(f'  Win Rate (2-day): {win_rate:.1f}%')
        print(f'  Hit Rate (2% target): {hit_rate:.1f}%')
        print(f'  Avg 2-day Return: {avg_ret:+.2f}%')
        print(f'  Avg Max Gain: {avg_max_gain:+.2f}%')
        print(f'  Avg Max Loss: {avg_max_loss:-.2f}%')
        print(f'  Expectancy: {expectancy:+.2f}%')
        print(f'  Avg VIX: {df["vix"].mean():.1f}')
        print()

    # Best stocks per regime
    print()
    print('='*70)
    print('TOP PERFORMERS BY REGIME')
    print('='*70)

    top_by_regime = {}

    for regime in ['CHOPPY', 'BULL', 'BEAR', 'VOLATILE']:
        data = results_by_regime[regime]
        if not data:
            continue

        df = pd.DataFrame(data)

        # Group by ticker
        by_ticker = df.groupby('ticker').agg({
            'ret_2d': ['count', 'mean'],
            'win': 'mean',
            'max_gain': 'mean'
        })
        by_ticker.columns = ['count', 'avg_ret', 'win_rate', 'avg_max_gain']

        # Filter to stocks with at least 3 signals
        by_ticker = by_ticker[by_ticker['count'] >= 3]

        # Sort by avg return
        by_ticker = by_ticker.sort_values('avg_ret', ascending=False)

        top_by_regime[regime] = by_ticker.head(5).to_dict('index')

        print(f'\n{regime} - Best Performers (n>=3):')
        for ticker in by_ticker.head(5).index:
            row = by_ticker.loc[ticker]
            print(f'  {ticker}: {row["count"]:.0f} signals, {row["win_rate"]*100:.0f}% win, {row["avg_ret"]:+.2f}% avg')

    print()
    print('='*70)
    print('REGIME DISTRIBUTION')
    print('='*70)
    total = sum(len(results_by_regime[r]) for r in results_by_regime)

    distribution = {}
    for regime in ['CHOPPY', 'BULL', 'BEAR', 'VOLATILE']:
        n = len(results_by_regime[regime])
        pct = n / total * 100 if total > 0 else 0
        distribution[regime] = {'count': n, 'percent': round(pct, 1)}
        print(f'{regime}: {n} signals ({pct:.1f}%)')

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'days_analyzed': days_back,
        'total_signals': total,
        'summary_by_regime': summary,
        'distribution': distribution,
        'top_performers_by_regime': top_by_regime
    }

    output_path = Path('data/research/regime_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f'Results saved to {output_path}')

    return results


if __name__ == '__main__':
    run_regime_analysis(days_back=365)

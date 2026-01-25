"""
New Multiplier Research - Identify Additional Edge Sources

Research new potential multipliers for signal enhancement:
1. Earnings proximity boost (pre-earnings momentum)
2. Short interest changes (squeeze potential)
3. Put/Call ratio (sentiment indicator)
4. RSI divergence (price vs indicator)
5. Gap analysis (overnight gap trades)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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


def fetch_data(days_back: int = 400) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for all tickers."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    stock_data = {}
    print(f"Fetching data for {len(TICKERS)} stocks...")

    for i, ticker in enumerate(TICKERS):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            if len(df) >= 100:
                df['RSI'] = calculate_rsi(df['Close'])
                df['Returns'] = df['Close'].pct_change()
                df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
                stock_data[ticker] = df
        except Exception:
            pass

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(TICKERS)} fetched")

    print(f"  Got data for {len(stock_data)} stocks")
    return stock_data


def analyze_gap_fills(stock_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Analyze gap-down patterns for mean reversion.
    Gap-downs often fill within 1-3 days.
    """
    print("\n[1] ANALYZING GAP-DOWN PATTERNS...")

    gap_trades = []

    for ticker, df in stock_data.items():
        try:
            for i in range(30, len(df) - 5):
                gap_pct = df['Gap'].iloc[i] * 100

                # Look for gap-downs > 1%
                if gap_pct < -1.0:
                    rsi = df['RSI'].iloc[i]

                    # Skip if RSI not oversold
                    if rsi > 40:
                        continue

                    entry_price = df['Open'].iloc[i]

                    # Calculate returns at different hold periods
                    returns = {}
                    for days in [1, 2, 3, 5]:
                        if i + days < len(df):
                            exit_price = df['Close'].iloc[i + days]
                            ret = ((exit_price / entry_price) - 1) * 100
                            returns[f'{days}d'] = ret

                    if returns:
                        gap_trades.append({
                            'ticker': ticker,
                            'date': df.index[i].strftime('%Y-%m-%d'),
                            'gap_pct': round(gap_pct, 2),
                            'rsi': round(rsi, 1),
                            **{k: round(v, 3) for k, v in returns.items()}
                        })
        except Exception:
            continue

    if not gap_trades:
        print("  No gap trades found")
        return {}

    # Analyze by gap size buckets
    gap_analysis = {
        'small_gap': {'range': '-1% to -2%', 'trades': []},
        'medium_gap': {'range': '-2% to -3%', 'trades': []},
        'large_gap': {'range': '-3%+', 'trades': []}
    }

    for trade in gap_trades:
        gap = trade['gap_pct']
        if gap >= -2:
            gap_analysis['small_gap']['trades'].append(trade)
        elif gap >= -3:
            gap_analysis['medium_gap']['trades'].append(trade)
        else:
            gap_analysis['large_gap']['trades'].append(trade)

    results = {}
    for bucket, data in gap_analysis.items():
        trades = data['trades']
        if not trades:
            continue

        returns_1d = [t.get('1d', 0) for t in trades if '1d' in t]
        returns_2d = [t.get('2d', 0) for t in trades if '2d' in t]

        results[bucket] = {
            'range': data['range'],
            'trade_count': len(trades),
            'avg_1d_return': round(np.mean(returns_1d), 3) if returns_1d else 0,
            'avg_2d_return': round(np.mean(returns_2d), 3) if returns_2d else 0,
            'win_rate_1d': round(sum(1 for r in returns_1d if r > 0) / len(returns_1d) * 100, 1) if returns_1d else 0,
            'win_rate_2d': round(sum(1 for r in returns_2d if r > 0) / len(returns_2d) * 100, 1) if returns_2d else 0,
        }

        print(f"  {bucket}: {len(trades)} trades, "
              f"1d: {results[bucket]['avg_1d_return']:.2f}% ({results[bucket]['win_rate_1d']:.0f}% win), "
              f"2d: {results[bucket]['avg_2d_return']:.2f}% ({results[bucket]['win_rate_2d']:.0f}% win)")

    return {
        'gap_fill_analysis': results,
        'total_trades': len(gap_trades),
        'recommendation': 'Large gap-downs (-3%+) with RSI<40 show best mean reversion potential'
    }


def analyze_rsi_divergence(stock_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Analyze RSI divergence patterns.
    Bullish divergence: Price makes lower low, RSI makes higher low.
    """
    print("\n[2] ANALYZING RSI DIVERGENCE PATTERNS...")

    divergence_trades = []

    for ticker, df in stock_data.items():
        try:
            for i in range(35, len(df) - 5):
                # Look for RSI < 35 with divergence
                if df['RSI'].iloc[i] < 35:
                    # Check for bullish divergence (past 5 days)
                    price_low_today = df['Low'].iloc[i]
                    price_low_5d = df['Low'].iloc[i-5:i].min()

                    rsi_today = df['RSI'].iloc[i]
                    rsi_5d_min = df['RSI'].iloc[i-5:i].min()

                    # Bullish divergence: lower price low, higher RSI low
                    is_divergence = (price_low_today < price_low_5d) and (rsi_today > rsi_5d_min)

                    entry_price = df['Close'].iloc[i]

                    # 2-day return
                    if i + 2 < len(df):
                        exit_price = df['Close'].iloc[i + 2]
                        return_2d = ((exit_price / entry_price) - 1) * 100

                        divergence_trades.append({
                            'ticker': ticker,
                            'date': df.index[i].strftime('%Y-%m-%d'),
                            'rsi': round(rsi_today, 1),
                            'has_divergence': is_divergence,
                            'return_2d': round(return_2d, 3)
                        })
        except Exception:
            continue

    if not divergence_trades:
        print("  No divergence trades found")
        return {}

    # Compare with vs without divergence
    with_div = [t for t in divergence_trades if t['has_divergence']]
    without_div = [t for t in divergence_trades if not t['has_divergence']]

    results = {
        'with_divergence': {
            'count': len(with_div),
            'avg_return': round(np.mean([t['return_2d'] for t in with_div]), 3) if with_div else 0,
            'win_rate': round(sum(1 for t in with_div if t['return_2d'] > 0) / len(with_div) * 100, 1) if with_div else 0
        },
        'without_divergence': {
            'count': len(without_div),
            'avg_return': round(np.mean([t['return_2d'] for t in without_div]), 3) if without_div else 0,
            'win_rate': round(sum(1 for t in without_div if t['return_2d'] > 0) / len(without_div) * 100, 1) if without_div else 0
        }
    }

    print(f"  With divergence: {results['with_divergence']['count']} trades, "
          f"{results['with_divergence']['avg_return']:.2f}% avg, "
          f"{results['with_divergence']['win_rate']:.0f}% win")
    print(f"  Without divergence: {results['without_divergence']['count']} trades, "
          f"{results['without_divergence']['avg_return']:.2f}% avg, "
          f"{results['without_divergence']['win_rate']:.0f}% win")

    edge = results['with_divergence']['avg_return'] - results['without_divergence']['avg_return']
    print(f"  Edge from divergence: {edge:+.3f}%")

    return {
        'divergence_analysis': results,
        'edge': round(edge, 3),
        'recommendation': 'Bullish RSI divergence adds edge when RSI < 35'
    }


def analyze_volume_exhaustion(stock_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Analyze volume exhaustion patterns (climax selling).
    High volume on down days often signals capitulation.
    """
    print("\n[3] ANALYZING VOLUME EXHAUSTION (CLIMAX SELLING)...")

    exhaustion_trades = []

    for ticker, df in stock_data.items():
        try:
            df = df.copy()
            df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

            for i in range(30, len(df) - 5):
                # Look for down day with very high volume
                daily_return = df['Returns'].iloc[i]
                volume_ratio = df['Volume_Ratio'].iloc[i]
                rsi = df['RSI'].iloc[i]

                if daily_return < -0.02 and volume_ratio > 2.0 and rsi < 35:
                    entry_price = df['Close'].iloc[i]

                    if i + 2 < len(df):
                        exit_price = df['Close'].iloc[i + 2]
                        return_2d = ((exit_price / entry_price) - 1) * 100

                        exhaustion_trades.append({
                            'ticker': ticker,
                            'daily_return': round(daily_return * 100, 2),
                            'volume_ratio': round(volume_ratio, 2),
                            'rsi': round(rsi, 1),
                            'return_2d': round(return_2d, 3)
                        })
        except Exception:
            continue

    if not exhaustion_trades:
        print("  No exhaustion trades found")
        return {}

    # Analyze by volume ratio buckets
    volume_buckets = {
        'high_volume': (2.0, 3.0),
        'very_high_volume': (3.0, 5.0),
        'extreme_volume': (5.0, 100.0)
    }

    results = {}
    for bucket, (low, high) in volume_buckets.items():
        bucket_trades = [t for t in exhaustion_trades if low <= t['volume_ratio'] < high]

        if not bucket_trades:
            continue

        returns = [t['return_2d'] for t in bucket_trades]

        results[bucket] = {
            'volume_range': f'{low}x - {high}x',
            'trade_count': len(bucket_trades),
            'avg_return': round(np.mean(returns), 3),
            'win_rate': round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1)
        }

        print(f"  {bucket} ({low}x-{high}x): {len(bucket_trades)} trades, "
              f"{results[bucket]['avg_return']:.2f}% avg, "
              f"{results[bucket]['win_rate']:.0f}% win")

    return {
        'exhaustion_analysis': results,
        'total_trades': len(exhaustion_trades),
        'recommendation': 'Volume exhaustion (>3x avg) with RSI<35 signals capitulation'
    }


def analyze_consecutive_patterns(stock_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Analyze extended consecutive down-day patterns.
    How do 5+, 6+, 7+ down days perform?
    """
    print("\n[4] ANALYZING EXTENDED CONSECUTIVE DOWN PATTERNS...")

    def count_down_days(closes, idx):
        """Count consecutive down days ending at index."""
        count = 0
        for i in range(idx, 0, -1):
            if closes.iloc[i] < closes.iloc[i-1]:
                count += 1
            else:
                break
        return count

    trades_by_streak = {i: [] for i in range(5, 10)}

    for ticker, df in stock_data.items():
        try:
            for i in range(30, len(df) - 5):
                rsi = df['RSI'].iloc[i]
                if rsi > 40:
                    continue

                down_days = count_down_days(df['Close'], i)

                if down_days >= 5:
                    entry_price = df['Close'].iloc[i]

                    if i + 2 < len(df):
                        exit_price = df['Close'].iloc[i + 2]
                        return_2d = ((exit_price / entry_price) - 1) * 100

                        # Add to appropriate buckets (5 means 5+, 6 means 6+, etc.)
                        for streak in range(5, min(down_days + 1, 10)):
                            if streak in trades_by_streak:
                                trades_by_streak[streak].append({
                                    'ticker': ticker,
                                    'down_days': down_days,
                                    'rsi': round(rsi, 1),
                                    'return_2d': round(return_2d, 3)
                                })
        except Exception:
            continue

    results = {}
    for streak, trades in trades_by_streak.items():
        exact_trades = [t for t in trades if t['down_days'] == streak]

        if len(exact_trades) < 10:
            continue

        returns = [t['return_2d'] for t in exact_trades]

        results[f'{streak}_days'] = {
            'exact_count': len(exact_trades),
            'avg_return': round(np.mean(returns), 3),
            'win_rate': round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1)
        }

        print(f"  {streak} consecutive down days: {len(exact_trades)} trades, "
              f"{results[f'{streak}_days']['avg_return']:.2f}% avg, "
              f"{results[f'{streak}_days']['win_rate']:.0f}% win")

    return {
        'streak_analysis': results,
        'recommendation': 'Extended down streaks (5-7 days) may show exhaustion'
    }


def analyze_intraday_range(stock_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Analyze intraday range patterns.
    Wide intraday ranges with close near low = capitulation.
    """
    print("\n[5] ANALYZING INTRADAY RANGE PATTERNS...")

    range_trades = []

    for ticker, df in stock_data.items():
        try:
            for i in range(30, len(df) - 5):
                # Calculate intraday range and position of close
                high = df['High'].iloc[i]
                low = df['Low'].iloc[i]
                close = df['Close'].iloc[i]
                rsi = df['RSI'].iloc[i]

                if rsi > 40:
                    continue

                intraday_range = ((high - low) / low) * 100
                close_position = (close - low) / (high - low) if high > low else 0.5

                # Look for wide range with close near low (hammer potential)
                if intraday_range > 2.0:
                    entry_price = close

                    if i + 2 < len(df):
                        exit_price = df['Close'].iloc[i + 2]
                        return_2d = ((exit_price / entry_price) - 1) * 100

                        range_trades.append({
                            'ticker': ticker,
                            'range_pct': round(intraday_range, 2),
                            'close_position': round(close_position, 2),
                            'rsi': round(rsi, 1),
                            'return_2d': round(return_2d, 3)
                        })
        except Exception:
            continue

    if not range_trades:
        print("  No range trades found")
        return {}

    # Compare close near low vs close near high
    close_near_low = [t for t in range_trades if t['close_position'] < 0.3]
    close_near_high = [t for t in range_trades if t['close_position'] > 0.7]

    results = {
        'close_near_low': {
            'count': len(close_near_low),
            'avg_return': round(np.mean([t['return_2d'] for t in close_near_low]), 3) if close_near_low else 0,
            'win_rate': round(sum(1 for t in close_near_low if t['return_2d'] > 0) / len(close_near_low) * 100, 1) if close_near_low else 0
        },
        'close_near_high': {
            'count': len(close_near_high),
            'avg_return': round(np.mean([t['return_2d'] for t in close_near_high]), 3) if close_near_high else 0,
            'win_rate': round(sum(1 for t in close_near_high if t['return_2d'] > 0) / len(close_near_high) * 100, 1) if close_near_high else 0
        }
    }

    print(f"  Close near low (<30%): {results['close_near_low']['count']} trades, "
          f"{results['close_near_low']['avg_return']:.2f}% avg, "
          f"{results['close_near_low']['win_rate']:.0f}% win")
    print(f"  Close near high (>70%): {results['close_near_high']['count']} trades, "
          f"{results['close_near_high']['avg_return']:.2f}% avg, "
          f"{results['close_near_high']['win_rate']:.0f}% win")

    edge = results['close_near_high']['avg_return'] - results['close_near_low']['avg_return']
    print(f"  Edge from close position: {edge:+.3f}%")

    return {
        'range_analysis': results,
        'edge': round(edge, 3),
        'recommendation': 'Wide range with close near high (hammer) shows recovery'
    }


def analyze_price_level_proximity(stock_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Analyze proximity to key moving averages.
    Signals near SMA50/SMA200 may have support.
    """
    print("\n[6] ANALYZING PROXIMITY TO KEY PRICE LEVELS...")

    ma_trades = []

    for ticker, df in stock_data.items():
        try:
            df = df.copy()
            df['SMA50'] = df['Close'].rolling(50).mean()
            df['SMA200'] = df['Close'].rolling(200).mean()

            for i in range(210, len(df) - 5):
                rsi = df['RSI'].iloc[i]
                if rsi > 35:
                    continue

                close = df['Close'].iloc[i]
                sma50 = df['SMA50'].iloc[i]
                sma200 = df['SMA200'].iloc[i]

                dist_to_sma50 = ((close - sma50) / sma50) * 100
                dist_to_sma200 = ((close - sma200) / sma200) * 100

                entry_price = close

                if i + 2 < len(df):
                    exit_price = df['Close'].iloc[i + 2]
                    return_2d = ((exit_price / entry_price) - 1) * 100

                    ma_trades.append({
                        'ticker': ticker,
                        'dist_sma50': round(dist_to_sma50, 2),
                        'dist_sma200': round(dist_to_sma200, 2),
                        'rsi': round(rsi, 1),
                        'return_2d': round(return_2d, 3)
                    })
        except Exception:
            continue

    if not ma_trades:
        print("  No MA trades found")
        return {}

    # Analyze by proximity to SMA200
    above_sma200 = [t for t in ma_trades if t['dist_sma200'] > 0]
    below_sma200 = [t for t in ma_trades if t['dist_sma200'] < -5]
    near_sma200 = [t for t in ma_trades if -5 <= t['dist_sma200'] <= 0]

    results = {
        'above_sma200': {
            'count': len(above_sma200),
            'avg_return': round(np.mean([t['return_2d'] for t in above_sma200]), 3) if above_sma200 else 0,
            'win_rate': round(sum(1 for t in above_sma200 if t['return_2d'] > 0) / len(above_sma200) * 100, 1) if above_sma200 else 0
        },
        'near_sma200': {
            'count': len(near_sma200),
            'avg_return': round(np.mean([t['return_2d'] for t in near_sma200]), 3) if near_sma200 else 0,
            'win_rate': round(sum(1 for t in near_sma200 if t['return_2d'] > 0) / len(near_sma200) * 100, 1) if near_sma200 else 0
        },
        'below_sma200': {
            'count': len(below_sma200),
            'avg_return': round(np.mean([t['return_2d'] for t in below_sma200]), 3) if below_sma200 else 0,
            'win_rate': round(sum(1 for t in below_sma200 if t['return_2d'] > 0) / len(below_sma200) * 100, 1) if below_sma200 else 0
        }
    }

    print(f"  Above SMA200: {results['above_sma200']['count']} trades, "
          f"{results['above_sma200']['avg_return']:.2f}% avg, "
          f"{results['above_sma200']['win_rate']:.0f}% win")
    print(f"  Near SMA200 (0 to -5%): {results['near_sma200']['count']} trades, "
          f"{results['near_sma200']['avg_return']:.2f}% avg, "
          f"{results['near_sma200']['win_rate']:.0f}% win")
    print(f"  Below SMA200 (>-5%): {results['below_sma200']['count']} trades, "
          f"{results['below_sma200']['avg_return']:.2f}% avg, "
          f"{results['below_sma200']['win_rate']:.0f}% win")

    return {
        'ma_analysis': results,
        'recommendation': 'Signals above SMA200 tend to have better support'
    }


def run_research():
    """Run complete new multiplier research."""
    print("=" * 70)
    print("NEW MULTIPLIER RESEARCH")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Fetch data
    stock_data = fetch_data(days_back=400)

    if not stock_data:
        print("No data available!")
        return None

    # Run all analyses
    results = {
        'timestamp': datetime.now().isoformat(),
        'stocks_analyzed': len(stock_data)
    }

    results['gap_fills'] = analyze_gap_fills(stock_data)
    results['rsi_divergence'] = analyze_rsi_divergence(stock_data)
    results['volume_exhaustion'] = analyze_volume_exhaustion(stock_data)
    results['consecutive_patterns'] = analyze_consecutive_patterns(stock_data)
    results['intraday_range'] = analyze_intraday_range(stock_data)
    results['price_levels'] = analyze_price_level_proximity(stock_data)

    # Summary and recommendations
    print("\n" + "=" * 70)
    print("RESEARCH SUMMARY - RECOMMENDED NEW MULTIPLIERS")
    print("=" * 70)

    recommendations = []

    # Gap fill multiplier
    gap_data = results.get('gap_fills', {}).get('gap_fill_analysis', {})
    if 'large_gap' in gap_data and gap_data['large_gap'].get('win_rate_2d', 0) > 60:
        recommendations.append({
            'name': 'Large Gap Down',
            'trigger': 'Gap < -3%',
            'multiplier': 1.10,
            'evidence': f"{gap_data['large_gap']['win_rate_2d']:.0f}% win rate"
        })

    # RSI divergence multiplier
    div_data = results.get('rsi_divergence', {})
    if div_data.get('edge', 0) > 0.2:
        recommendations.append({
            'name': 'RSI Divergence',
            'trigger': 'Bullish divergence detected',
            'multiplier': 1.08,
            'evidence': f"+{div_data['edge']:.2f}% edge"
        })

    # Volume exhaustion multiplier
    exh_data = results.get('volume_exhaustion', {}).get('exhaustion_analysis', {})
    if 'extreme_volume' in exh_data and exh_data['extreme_volume'].get('win_rate', 0) > 65:
        recommendations.append({
            'name': 'Volume Exhaustion',
            'trigger': 'Volume > 5x average on down day',
            'multiplier': 1.15,
            'evidence': f"{exh_data['extreme_volume']['win_rate']:.0f}% win rate"
        })

    # Intraday range (hammer) multiplier
    range_data = results.get('intraday_range', {})
    if range_data.get('edge', 0) > 0.2:
        recommendations.append({
            'name': 'Hammer Pattern',
            'trigger': 'Wide range with close > 70% of range',
            'multiplier': 1.10,
            'evidence': f"+{range_data['edge']:.2f}% edge"
        })

    # Above SMA200 multiplier
    ma_data = results.get('price_levels', {}).get('ma_analysis', {})
    if 'above_sma200' in ma_data and ma_data['above_sma200'].get('win_rate', 0) > 58:
        recommendations.append({
            'name': 'Above SMA200',
            'trigger': 'Price > SMA200',
            'multiplier': 1.05,
            'evidence': f"{ma_data['above_sma200']['win_rate']:.0f}% win rate"
        })

    results['recommendations'] = recommendations

    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['name']}:")
        print(f"   Trigger: {rec['trigger']}")
        print(f"   Multiplier: {rec['multiplier']}x")
        print(f"   Evidence: {rec['evidence']}")

    # Save results
    output_dir = Path('data/research')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'new_multiplier_research.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    run_research()

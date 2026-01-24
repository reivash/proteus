"""
Exit Strategy Analysis for Mean Reversion Signals

Comprehensive analysis of exit timing and strategies:
1. Hold period optimization (1d, 2d, 3d, 4d, 5d)
2. Fixed profit targets (1%, 1.5%, 2%, 2.5%, 3%)
3. Trailing stops (0.5%, 1%, 1.5%, 2%)
4. Stop-loss levels (-1%, -1.5%, -2%, -2.5%, -3%)
5. Partial exit strategies (scale out at multiple levels)
6. Signal strength vs optimal exit correlation
7. Stock tier vs optimal exit analysis
8. Regime-based exit adjustments

Goal: Find optimal exit strategy to improve avg return from +0.82% to +1.1%+
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# Stock universe
TICKERS = [
    'NVDA', 'AVGO', 'MSFT', 'ORCL', 'INTU', 'ADBE', 'CRM', 'NOW', 'AMAT', 'KLAC',
    'MRVL', 'QCOM', 'TXN', 'ADI', 'JPM', 'MS', 'SCHW', 'AXP', 'AIG', 'USB',
    'PNC', 'V', 'MA', 'ABBV', 'SYK', 'GILD', 'PFE', 'JNJ', 'CVS', 'HCA',
    'IDXX', 'INSM', 'COP', 'SLB', 'XOM', 'MPC', 'EOG', 'CAT', 'ETN', 'LMT',
    'ROAD', 'HD', 'LOW', 'TGT', 'WMT', 'TMUS', 'CMCSA', 'META', 'APD', 'ECL',
    'MLM', 'SHW', 'EXR', 'NEE'
]

# Stock tiers from backtest
ELITE_STOCKS = ['COP', 'CVS', 'SLB', 'XOM', 'ADI', 'GILD', 'JPM', 'EOG', 'IDXX', 'TXN']
STRONG_STOCKS = ['QCOM', 'JNJ', 'V', 'MPC', 'SHW', 'KLAC', 'MS', 'AMAT']
AVOID_STOCKS = ['NOW', 'CAT', 'CRM', 'HCA', 'TGT', 'ETN', 'HD', 'ORCL', 'ADBE', 'INTU']


def calculate_rsi(prices, period=14):
    """Calculate RSI."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def get_stock_tier(ticker: str) -> str:
    """Get stock performance tier."""
    if ticker in ELITE_STOCKS:
        return 'elite'
    elif ticker in STRONG_STOCKS:
        return 'strong'
    elif ticker in AVOID_STOCKS:
        return 'avoid'
    else:
        return 'average'


@dataclass
class TradeResult:
    """Result of a single trade with various exit strategies."""
    ticker: str
    entry_date: str
    entry_price: float
    rsi: float
    signal_strength: float
    stock_tier: str

    # Returns at different hold periods
    ret_1d: float
    ret_2d: float
    ret_3d: float
    ret_4d: float
    ret_5d: float

    # Intraday data for each day
    high_1d: float
    high_2d: float
    high_3d: float
    high_4d: float
    high_5d: float
    low_1d: float
    low_2d: float
    low_3d: float
    low_4d: float
    low_5d: float

    # Maximum gain/loss within 5 days
    max_gain_5d: float
    max_loss_5d: float

    # Day of max gain (1-5)
    max_gain_day: int


def fetch_stock_data(days_back: int = 400) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for all tickers."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    stock_data = {}
    print(f"Fetching data for {len(TICKERS)} stocks...")

    for i, ticker in enumerate(TICKERS):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            # Handle multi-index columns from newer yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            if len(df) >= 50:
                stock_data[ticker] = df
        except:
            pass

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(TICKERS)} fetched")

    print(f"  Got data for {len(stock_data)} stocks")
    return stock_data


def find_signals_and_outcomes(stock_data: Dict[str, pd.DataFrame],
                               days_back: int = 365) -> List[TradeResult]:
    """Find all RSI < 35 signals and calculate outcomes."""

    results = []
    cutoff_date = datetime.now() - timedelta(days=days_back)
    cutoff_str = cutoff_date.strftime('%Y-%m-%d')

    print(f"\nAnalyzing signals from last {days_back} days...")
    print(f"  Cutoff date: {cutoff_str}")

    for ticker, df in stock_data.items():
        try:
            df = df.copy()
            df['RSI'] = calculate_rsi(df['Close'])

            # Find RSI < 35 signals
            for i in range(20, len(df) - 6):  # Need 5 days forward
                if df['RSI'].iloc[i] < 35:
                    signal_date = df.index[i]

                    # Use string comparison for dates to avoid timezone issues
                    signal_str = signal_date.strftime('%Y-%m-%d')
                    if signal_str < cutoff_str:
                        continue

                    entry_price = df['Close'].iloc[i]
                    rsi = df['RSI'].iloc[i]

                    # Simple signal strength approximation
                    signal_strength = max(0, min(100, (35 - rsi) * 3 + 50))

                    # Get returns and highs/lows for each day
                    ret_1d = ((df['Close'].iloc[i+1] / entry_price) - 1) * 100
                    ret_2d = ((df['Close'].iloc[i+2] / entry_price) - 1) * 100
                    ret_3d = ((df['Close'].iloc[i+3] / entry_price) - 1) * 100
                    ret_4d = ((df['Close'].iloc[i+4] / entry_price) - 1) * 100
                    ret_5d = ((df['Close'].iloc[i+5] / entry_price) - 1) * 100

                    # Cumulative highs/lows
                    high_1d = ((df['High'].iloc[i+1] / entry_price) - 1) * 100
                    high_2d = ((df['High'].iloc[i+1:i+3].max() / entry_price) - 1) * 100
                    high_3d = ((df['High'].iloc[i+1:i+4].max() / entry_price) - 1) * 100
                    high_4d = ((df['High'].iloc[i+1:i+5].max() / entry_price) - 1) * 100
                    high_5d = ((df['High'].iloc[i+1:i+6].max() / entry_price) - 1) * 100

                    low_1d = ((df['Low'].iloc[i+1] / entry_price) - 1) * 100
                    low_2d = ((df['Low'].iloc[i+1:i+3].min() / entry_price) - 1) * 100
                    low_3d = ((df['Low'].iloc[i+1:i+4].min() / entry_price) - 1) * 100
                    low_4d = ((df['Low'].iloc[i+1:i+5].min() / entry_price) - 1) * 100
                    low_5d = ((df['Low'].iloc[i+1:i+6].min() / entry_price) - 1) * 100

                    # Find day of max gain
                    daily_highs = [
                        ((df['High'].iloc[i+1] / entry_price) - 1) * 100,
                        ((df['High'].iloc[i+2] / entry_price) - 1) * 100,
                        ((df['High'].iloc[i+3] / entry_price) - 1) * 100,
                        ((df['High'].iloc[i+4] / entry_price) - 1) * 100,
                        ((df['High'].iloc[i+5] / entry_price) - 1) * 100,
                    ]
                    max_gain_day = daily_highs.index(max(daily_highs)) + 1

                    results.append(TradeResult(
                        ticker=ticker,
                        entry_date=signal_date.strftime('%Y-%m-%d'),
                        entry_price=entry_price,
                        rsi=rsi,
                        signal_strength=signal_strength,
                        stock_tier=get_stock_tier(ticker),
                        ret_1d=ret_1d,
                        ret_2d=ret_2d,
                        ret_3d=ret_3d,
                        ret_4d=ret_4d,
                        ret_5d=ret_5d,
                        high_1d=high_1d,
                        high_2d=high_2d,
                        high_3d=high_3d,
                        high_4d=high_4d,
                        high_5d=high_5d,
                        low_1d=low_1d,
                        low_2d=low_2d,
                        low_3d=low_3d,
                        low_4d=low_4d,
                        low_5d=low_5d,
                        max_gain_5d=high_5d,
                        max_loss_5d=low_5d,
                        max_gain_day=max_gain_day
                    ))

        except Exception as e:
            continue

    print(f"  Found {len(results)} signals")
    return results


def analyze_hold_periods(trades: List[TradeResult]) -> Dict:
    """Analyze optimal hold period."""
    print("\n" + "="*70)
    print("HOLD PERIOD ANALYSIS")
    print("="*70)

    results = {}

    for period in [1, 2, 3, 4, 5]:
        returns = [getattr(t, f'ret_{period}d') for t in trades]
        wins = sum(1 for r in returns if r > 0)

        results[f'{period}d'] = {
            'signals': len(returns),
            'win_rate': round(wins / len(returns) * 100, 1),
            'avg_return': round(np.mean(returns), 3),
            'median_return': round(np.median(returns), 3),
            'std_dev': round(np.std(returns), 3),
            'sharpe': round(np.mean(returns) / np.std(returns) * np.sqrt(252/period), 2) if np.std(returns) > 0 else 0,
            'max_return': round(max(returns), 2),
            'min_return': round(min(returns), 2),
        }

        print(f"\n{period}-DAY HOLD:")
        print(f"  Win rate: {results[f'{period}d']['win_rate']}%")
        print(f"  Avg return: {results[f'{period}d']['avg_return']:+.3f}%")
        print(f"  Median return: {results[f'{period}d']['median_return']:+.3f}%")
        print(f"  Std dev: {results[f'{period}d']['std_dev']:.3f}%")
        print(f"  Sharpe (annualized): {results[f'{period}d']['sharpe']}")

    # Find optimal
    best_period = max(results.keys(), key=lambda k: results[k]['avg_return'])
    print(f"\n>>> OPTIMAL HOLD PERIOD: {best_period} (avg return: {results[best_period]['avg_return']:+.3f}%) <<<")

    return results


def analyze_profit_targets(trades: List[TradeResult]) -> Dict:
    """Analyze fixed profit target strategies."""
    print("\n" + "="*70)
    print("PROFIT TARGET ANALYSIS")
    print("="*70)

    targets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    results = {}

    for target in targets:
        hits = 0
        total_return = 0
        days_to_target = []

        for t in trades:
            # Check each day if target was hit
            hit = False
            for day in range(1, 6):
                high = getattr(t, f'high_{day}d')
                if high >= target:
                    hits += 1
                    total_return += target
                    days_to_target.append(day)
                    hit = True
                    break

            if not hit:
                # Exit at day 5 close if target not hit
                total_return += t.ret_5d

        hit_rate = hits / len(trades) * 100
        avg_return = total_return / len(trades)
        avg_days = np.mean(days_to_target) if days_to_target else 5

        results[f'{target}%'] = {
            'hit_rate': round(hit_rate, 1),
            'avg_return': round(avg_return, 3),
            'avg_days_to_target': round(avg_days, 2),
            'trades_hit': hits,
            'trades_missed': len(trades) - hits
        }

        print(f"\n{target}% TARGET:")
        print(f"  Hit rate: {hit_rate:.1f}%")
        print(f"  Avg return: {avg_return:+.3f}%")
        print(f"  Avg days to target: {avg_days:.2f}")

    # Find optimal target
    best_target = max(results.keys(), key=lambda k: results[k]['avg_return'])
    print(f"\n>>> OPTIMAL PROFIT TARGET: {best_target} (avg return: {results[best_target]['avg_return']:+.3f}%) <<<")

    return results


def analyze_stop_losses(trades: List[TradeResult]) -> Dict:
    """Analyze stop-loss strategies."""
    print("\n" + "="*70)
    print("STOP-LOSS ANALYSIS")
    print("="*70)

    stops = [-1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0]
    results = {}

    for stop in stops:
        stopped_out = 0
        total_return = 0

        for t in trades:
            # Check each day if stop was hit
            stopped = False
            for day in range(1, 6):
                low = getattr(t, f'low_{day}d')
                if low <= stop:
                    stopped_out += 1
                    total_return += stop  # Exit at stop
                    stopped = True
                    break

            if not stopped:
                # No stop hit, exit at day 2 close (default)
                total_return += t.ret_2d

        stop_rate = stopped_out / len(trades) * 100
        avg_return = total_return / len(trades)

        results[f'{stop}%'] = {
            'stop_rate': round(stop_rate, 1),
            'avg_return': round(avg_return, 3),
            'trades_stopped': stopped_out,
            'trades_survived': len(trades) - stopped_out
        }

        print(f"\n{stop}% STOP-LOSS:")
        print(f"  Stop rate: {stop_rate:.1f}%")
        print(f"  Avg return: {avg_return:+.3f}%")
        print(f"  Trades stopped: {stopped_out}")

    # Find optimal
    best_stop = max(results.keys(), key=lambda k: results[k]['avg_return'])
    print(f"\n>>> OPTIMAL STOP-LOSS: {best_stop} (avg return: {results[best_stop]['avg_return']:+.3f}%) <<<")

    return results


def analyze_combined_strategy(trades: List[TradeResult]) -> Dict:
    """Analyze combined profit target + stop-loss strategies."""
    print("\n" + "="*70)
    print("COMBINED TARGET + STOP ANALYSIS")
    print("="*70)

    targets = [1.5, 2.0, 2.5, 3.0]
    stops = [-1.5, -2.0, -2.5, -3.0]
    max_days = [2, 3, 4, 5]

    results = []

    for target in targets:
        for stop in stops:
            for max_day in max_days:
                total_return = 0
                wins = 0

                for t in trades:
                    trade_return = None

                    for day in range(1, max_day + 1):
                        high = getattr(t, f'high_{day}d')
                        low = getattr(t, f'low_{day}d')

                        # Check stop first (conservative)
                        if low <= stop:
                            trade_return = stop
                            break
                        # Then check target
                        if high >= target:
                            trade_return = target
                            wins += 1
                            break

                    if trade_return is None:
                        # Exit at max_day close
                        trade_return = getattr(t, f'ret_{max_day}d')
                        if trade_return > 0:
                            wins += 1

                    total_return += trade_return

                avg_return = total_return / len(trades)
                win_rate = wins / len(trades) * 100

                results.append({
                    'target': target,
                    'stop': stop,
                    'max_days': max_day,
                    'avg_return': round(avg_return, 3),
                    'win_rate': round(win_rate, 1)
                })

    # Sort by avg return
    results.sort(key=lambda x: x['avg_return'], reverse=True)

    print("\nTOP 10 COMBINED STRATEGIES:")
    print("-" * 70)
    for i, r in enumerate(results[:10]):
        print(f"  {i+1}. Target: +{r['target']}%, Stop: {r['stop']}%, Max: {r['max_days']}d")
        print(f"     Win rate: {r['win_rate']}%, Avg return: {r['avg_return']:+.3f}%")

    print("\nBOTTOM 5 STRATEGIES:")
    print("-" * 70)
    for i, r in enumerate(results[-5:]):
        print(f"  Target: +{r['target']}%, Stop: {r['stop']}%, Max: {r['max_days']}d")
        print(f"  Win rate: {r['win_rate']}%, Avg return: {r['avg_return']:+.3f}%")

    return {'strategies': results[:20], 'best': results[0]}


def analyze_partial_exits(trades: List[TradeResult]) -> Dict:
    """Analyze partial exit (scale-out) strategies."""
    print("\n" + "="*70)
    print("PARTIAL EXIT (SCALE-OUT) ANALYSIS")
    print("="*70)

    strategies = [
        {'name': '50/50 at +1%/+2%', 'exits': [(0.5, 1.0), (0.5, 2.0)]},
        {'name': '33/33/34 at +1%/+2%/+3%', 'exits': [(0.33, 1.0), (0.33, 2.0), (0.34, 3.0)]},
        {'name': '50/50 at +1.5%/+3%', 'exits': [(0.5, 1.5), (0.5, 3.0)]},
        {'name': '25/25/50 at +1%/+2%/close', 'exits': [(0.25, 1.0), (0.25, 2.0), (0.5, None)]},
        {'name': '33/67 at +1%/+2%', 'exits': [(0.33, 1.0), (0.67, 2.0)]},
        {'name': '50% at +2%, 50% trail', 'exits': [(0.5, 2.0), (0.5, 'trail_1%')]},
    ]

    results = {}

    for strategy in strategies:
        total_return = 0

        for t in trades:
            trade_return = 0
            remaining = 1.0

            for portion, target in strategy['exits']:
                if target is None:
                    # Exit remaining at day 2 close
                    trade_return += remaining * t.ret_2d
                    remaining = 0
                elif target == 'trail_1%':
                    # Simple trailing: use max_gain - 1% or day 3 close
                    trail_exit = max(t.high_3d - 1.0, t.ret_3d)
                    trade_return += remaining * trail_exit
                    remaining = 0
                else:
                    # Check if target hit within 5 days
                    for day in range(1, 6):
                        high = getattr(t, f'high_{day}d')
                        if high >= target:
                            trade_return += portion * target
                            remaining -= portion
                            break
                    else:
                        # Target not hit, exit at day 5
                        trade_return += portion * t.ret_5d
                        remaining -= portion

            total_return += trade_return

        avg_return = total_return / len(trades)
        results[strategy['name']] = {
            'avg_return': round(avg_return, 3)
        }

        print(f"\n{strategy['name']}:")
        print(f"  Avg return: {avg_return:+.3f}%")

    # Compare to simple hold
    simple_2d = np.mean([t.ret_2d for t in trades])
    print(f"\n(Baseline 2d hold: {simple_2d:+.3f}%)")

    best_strategy = max(results.keys(), key=lambda k: results[k]['avg_return'])
    print(f"\n>>> BEST PARTIAL EXIT: {best_strategy} ({results[best_strategy]['avg_return']:+.3f}%) <<<")

    return results


def analyze_by_signal_strength(trades: List[TradeResult]) -> Dict:
    """Analyze optimal exit by signal strength."""
    print("\n" + "="*70)
    print("EXIT STRATEGY BY SIGNAL STRENGTH")
    print("="*70)

    # Group trades by signal strength
    strength_groups = {
        'weak (50-60)': [t for t in trades if 50 <= t.signal_strength < 60],
        'moderate (60-70)': [t for t in trades if 60 <= t.signal_strength < 70],
        'strong (70-80)': [t for t in trades if 70 <= t.signal_strength < 80],
        'very_strong (80+)': [t for t in trades if t.signal_strength >= 80],
    }

    results = {}

    for group_name, group_trades in strength_groups.items():
        if len(group_trades) < 10:
            continue

        print(f"\n{group_name.upper()} ({len(group_trades)} trades):")

        # Find best hold period for this group
        best_period = None
        best_return = -float('inf')

        for period in [1, 2, 3, 4, 5]:
            returns = [getattr(t, f'ret_{period}d') for t in group_trades]
            avg = np.mean(returns)
            if avg > best_return:
                best_return = avg
                best_period = period

        # Find hit rate for +2% target
        hit_rate_2pct = sum(1 for t in group_trades if t.high_5d >= 2.0) / len(group_trades) * 100

        # Average max gain
        avg_max_gain = np.mean([t.max_gain_5d for t in group_trades])
        avg_max_gain_day = np.mean([t.max_gain_day for t in group_trades])

        results[group_name] = {
            'count': len(group_trades),
            'best_hold_period': best_period,
            'best_hold_return': round(best_return, 3),
            'hit_rate_2pct': round(hit_rate_2pct, 1),
            'avg_max_gain': round(avg_max_gain, 2),
            'avg_max_gain_day': round(avg_max_gain_day, 1)
        }

        print(f"  Best hold period: {best_period}d ({best_return:+.3f}%)")
        print(f"  2% target hit rate: {hit_rate_2pct:.1f}%")
        print(f"  Avg max gain: {avg_max_gain:.2f}% (on day {avg_max_gain_day:.1f})")

    return results


def analyze_by_stock_tier(trades: List[TradeResult]) -> Dict:
    """Analyze optimal exit by stock tier."""
    print("\n" + "="*70)
    print("EXIT STRATEGY BY STOCK TIER")
    print("="*70)

    tier_groups = {
        'elite': [t for t in trades if t.stock_tier == 'elite'],
        'strong': [t for t in trades if t.stock_tier == 'strong'],
        'average': [t for t in trades if t.stock_tier == 'average'],
        'avoid': [t for t in trades if t.stock_tier == 'avoid'],
    }

    results = {}

    for tier, tier_trades in tier_groups.items():
        if len(tier_trades) < 10:
            continue

        print(f"\n{tier.upper()} TIER ({len(tier_trades)} trades):")

        # Best hold period
        best_period = None
        best_return = -float('inf')

        for period in [1, 2, 3, 4, 5]:
            returns = [getattr(t, f'ret_{period}d') for t in tier_trades]
            avg = np.mean(returns)
            if avg > best_return:
                best_return = avg
                best_period = period

        # Best profit target
        best_target = None
        best_target_return = -float('inf')

        for target in [1.5, 2.0, 2.5, 3.0]:
            total = 0
            for t in tier_trades:
                hit = False
                for day in range(1, 6):
                    if getattr(t, f'high_{day}d') >= target:
                        total += target
                        hit = True
                        break
                if not hit:
                    total += t.ret_5d

            avg = total / len(tier_trades)
            if avg > best_target_return:
                best_target_return = avg
                best_target = target

        # Metrics
        avg_max_gain = np.mean([t.max_gain_5d for t in tier_trades])
        avg_max_loss = np.mean([t.max_loss_5d for t in tier_trades])

        results[tier] = {
            'count': len(tier_trades),
            'best_hold_period': best_period,
            'best_hold_return': round(best_return, 3),
            'best_target': best_target,
            'best_target_return': round(best_target_return, 3),
            'avg_max_gain': round(avg_max_gain, 2),
            'avg_max_loss': round(avg_max_loss, 2)
        }

        print(f"  Best hold: {best_period}d ({best_return:+.3f}%)")
        print(f"  Best target: +{best_target}% ({best_target_return:+.3f}%)")
        print(f"  Avg max gain/loss: {avg_max_gain:+.2f}% / {avg_max_loss:.2f}%")

    return results


def generate_recommendations(all_results: Dict) -> Dict:
    """Generate final exit strategy recommendations."""
    print("\n" + "="*70)
    print("FINAL RECOMMENDATIONS")
    print("="*70)

    recommendations = {}

    # Overall best strategy
    best_combined = all_results['combined']['best']
    recommendations['default'] = {
        'profit_target': best_combined['target'],
        'stop_loss': best_combined['stop'],
        'max_hold_days': best_combined['max_days'],
        'expected_return': best_combined['avg_return'],
        'expected_win_rate': best_combined['win_rate']
    }

    print(f"\nDEFAULT STRATEGY:")
    print(f"  Profit target: +{best_combined['target']}%")
    print(f"  Stop-loss: {best_combined['stop']}%")
    print(f"  Max hold: {best_combined['max_days']} days")
    print(f"  Expected return: {best_combined['avg_return']:+.3f}%")
    print(f"  Expected win rate: {best_combined['win_rate']}%")

    # Tier-specific recommendations
    if 'by_tier' in all_results:
        print(f"\nTIER-SPECIFIC ADJUSTMENTS:")
        for tier, data in all_results['by_tier'].items():
            print(f"  {tier}: target +{data.get('best_target', 2.0)}%, hold {data.get('best_hold_period', 2)}d")
            recommendations[f'tier_{tier}'] = {
                'profit_target': data.get('best_target', 2.0),
                'max_hold_days': data.get('best_hold_period', 2)
            }

    # Signal strength adjustments
    if 'by_strength' in all_results:
        print(f"\nSIGNAL STRENGTH ADJUSTMENTS:")
        for strength, data in all_results['by_strength'].items():
            print(f"  {strength}: hold {data['best_hold_period']}d, 2% hit rate: {data['hit_rate_2pct']}%")

    return recommendations


def run_exit_strategy_analysis(days_back: int = 365):
    """Run complete exit strategy analysis."""
    print("="*70)
    print("EXIT STRATEGY ANALYSIS")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Period: {days_back} days")
    print()

    # Fetch data
    stock_data = fetch_stock_data(days_back + 50)

    # Find signals and outcomes
    trades = find_signals_and_outcomes(stock_data, days_back)

    if len(trades) < 50:
        print("Not enough trades for analysis!")
        return None

    print(f"\nTotal trades for analysis: {len(trades)}")

    # Run all analyses
    all_results = {}

    all_results['hold_periods'] = analyze_hold_periods(trades)
    all_results['profit_targets'] = analyze_profit_targets(trades)
    all_results['stop_losses'] = analyze_stop_losses(trades)
    all_results['combined'] = analyze_combined_strategy(trades)
    all_results['partial_exits'] = analyze_partial_exits(trades)
    all_results['by_strength'] = analyze_by_signal_strength(trades)
    all_results['by_tier'] = analyze_by_stock_tier(trades)

    # Generate recommendations
    recommendations = generate_recommendations(all_results)
    all_results['recommendations'] = recommendations

    # Calculate improvement vs baseline
    baseline_return = all_results['hold_periods']['2d']['avg_return']
    best_return = all_results['combined']['best']['avg_return']
    improvement = best_return - baseline_return

    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Baseline (2d hold): {baseline_return:+.3f}%")
    print(f"Best strategy: {best_return:+.3f}%")
    print(f">>> IMPROVEMENT: {improvement:+.3f}% per trade <<<")

    # Save results
    output_path = Path('data/research/exit_strategy_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'days_analyzed': days_back,
        'total_trades': len(trades),
        'baseline_return': baseline_return,
        'best_return': best_return,
        'improvement': improvement,
        'hold_periods': all_results['hold_periods'],
        'profit_targets': all_results['profit_targets'],
        'stop_losses': all_results['stop_losses'],
        'best_combined_strategies': all_results['combined']['strategies'][:10],
        'partial_exits': all_results['partial_exits'],
        'by_signal_strength': all_results['by_strength'],
        'by_stock_tier': all_results['by_tier'],
        'recommendations': recommendations
    }

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == '__main__':
    run_exit_strategy_analysis(days_back=365)

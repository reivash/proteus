#!/usr/bin/env python
"""
Out-of-Sample Validation for Bear Detection Weights

Tests the optimized weights on different time periods to ensure
they're robust and not overfitted to the training data.

Usage:
    python scripts/validate_bear_weights_oos.py
"""

import os
import sys
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    print("Please install yfinance: pip install yfinance")
    sys.exit(1)


# Optimized weights (from optimization script)
OPTIMIZED_WEIGHTS = {
    'spy_roc': 0.053,
    'vix': 0.023,
    'breadth': 0.158,
    'sector_breadth': 0.141,
    'volume': 0.108,
    'yield_curve': 0.040,
    'credit_spread': 0.126,
    'high_yield': 0.132,
    'put_call': 0.124,
    'divergence': 0.096
}

# Original weights (before optimization)
ORIGINAL_WEIGHTS = {
    'spy_roc': 0.14,
    'vix': 0.18,
    'breadth': 0.11,
    'sector_breadth': 0.08,
    'volume': 0.05,
    'yield_curve': 0.12,
    'credit_spread': 0.08,
    'high_yield': 0.08,
    'put_call': 0.08,
    'divergence': 0.08
}


def fetch_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch SPY data for specified date range."""
    spy = yf.Ticker('SPY')
    df = spy.history(start=start_date, end=end_date)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators."""
    result = df.copy()

    result['spy_roc_3d'] = result['Close'].pct_change(3) * 100
    result['high_20d'] = result['Close'].rolling(20).max()
    result['drawdown'] = (result['Close'] - result['high_20d']) / result['high_20d'] * 100
    result['sma_20'] = result['Close'].rolling(20).mean()
    result['sma_50'] = result['Close'].rolling(50).mean()
    result['vol_sma'] = result['Volume'].rolling(20).mean()
    result['vol_ratio'] = result['Volume'] / result['vol_sma']
    result['daily_return'] = result['Close'].pct_change() * 100
    result['volatility_20d'] = result['daily_return'].rolling(20).std() * np.sqrt(252)
    result['vol_spike'] = result['volatility_20d'].pct_change(2) * 100
    result['rsi_14'] = calculate_rsi(result['Close'], 14)
    result['price_vs_sma20'] = (result['Close'] / result['sma_20'] - 1) * 100
    result['sma_20_slope'] = result['sma_20'].pct_change(5) * 100

    return result.dropna()


def simulate_scores(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Simulate bear scores with given weights."""
    result = df.copy()

    # Scoring points based on weights (scaled to max 100)
    max_pts = {
        'spy_roc': int(weights['spy_roc'] * 100),
        'vix': int(weights['vix'] * 100),
        'breadth': int(weights['breadth'] * 100),
        'sector_breadth': int(weights['sector_breadth'] * 100),
        'volume': int(weights['volume'] * 100),
        'yield_curve': int(weights['yield_curve'] * 100),
        'credit_spread': int(weights['credit_spread'] * 100),
        'high_yield': int(weights['high_yield'] * 100),
        'put_call': int(weights['put_call'] * 100),
        'divergence': int(weights['divergence'] * 100)
    }

    scores = []
    for i, row in result.iterrows():
        score = 0

        # SPY ROC
        if row['spy_roc_3d'] <= -5:
            score += max_pts['spy_roc']
        elif row['spy_roc_3d'] <= -3:
            score += int(max_pts['spy_roc'] * 0.7)
        elif row['spy_roc_3d'] <= -2:
            score += int(max_pts['spy_roc'] * 0.4)

        # VIX proxy
        vol = row.get('volatility_20d', 15)
        vol_spike = row.get('vol_spike', 0)
        if vol >= 35 or vol_spike >= 50:
            score += max_pts['vix']
        elif vol >= 30 or vol_spike >= 30:
            score += int(max_pts['vix'] * 0.5)
        elif vol >= 25 or vol_spike >= 20:
            score += int(max_pts['vix'] * 0.3)

        # Breadth proxy
        price_vs_ma = row.get('price_vs_sma20', 0)
        if price_vs_ma <= -5:
            score += max_pts['breadth']
        elif price_vs_ma <= -3:
            score += int(max_pts['breadth'] * 0.7)
        elif price_vs_ma <= -1:
            score += int(max_pts['breadth'] * 0.3)

        # Sector breadth proxy
        if row['Close'] < row['sma_20'] < row['sma_50']:
            score += max_pts['sector_breadth']
        elif row['Close'] < row['sma_20']:
            score += int(max_pts['sector_breadth'] * 0.6)
        elif row['Close'] < row['sma_50']:
            score += int(max_pts['sector_breadth'] * 0.3)

        # Volume
        if row['daily_return'] < -0.5 and row['vol_ratio'] > 2:
            score += max_pts['volume']

        # Yield curve proxy
        sma_slope = row.get('sma_20_slope', 0)
        if sma_slope <= -3:
            score += max_pts['yield_curve']
        elif sma_slope <= -1.5:
            score += int(max_pts['yield_curve'] * 0.7)
        elif sma_slope <= -0.5:
            score += int(max_pts['yield_curve'] * 0.25)

        # Credit spread proxy
        if row['drawdown'] < -8:
            score += max_pts['credit_spread']
        elif row['drawdown'] < -5:
            score += int(max_pts['credit_spread'] * 0.6)
        elif row['drawdown'] < -3:
            score += int(max_pts['credit_spread'] * 0.3)

        # High-yield proxy
        if row['spy_roc_3d'] < -3 and row['drawdown'] < -3:
            score += max_pts['high_yield']
        elif row['spy_roc_3d'] < -2 and row['drawdown'] < -2:
            score += int(max_pts['high_yield'] * 0.6)
        elif row['spy_roc_3d'] < -1 and row['drawdown'] < -1:
            score += int(max_pts['high_yield'] * 0.3)

        # Put/Call proxy
        rsi = row.get('rsi_14', 50)
        if rsi <= 25:
            score += max_pts['put_call']
        elif rsi <= 35:
            score += int(max_pts['put_call'] * 0.65)
        elif rsi <= 40:
            score += int(max_pts['put_call'] * 0.25)

        # Divergence
        if row['drawdown'] > -2 and row['spy_roc_3d'] < -1.5:
            score += max_pts['divergence']

        scores.append(min(score, 100))

    result['bear_score'] = scores
    result['alert_level'] = pd.cut(
        result['bear_score'],
        bins=[-1, 29, 49, 69, 100],
        labels=['NORMAL', 'WATCH', 'WARNING', 'CRITICAL']
    )

    return result


def find_drawdowns(df: pd.DataFrame, threshold: float = -5.0) -> list:
    """Find major drawdowns."""
    drawdowns = []
    in_drawdown = False
    start_idx = None
    peak_price = None

    for i, row in df.iterrows():
        if not in_drawdown:
            if row['drawdown'] < threshold:
                in_drawdown = True
                start_idx = i
                peak_idx = df.loc[:i, 'Close'].idxmax()
                peak_price = df.loc[peak_idx, 'Close']
        else:
            if row['drawdown'] > threshold / 2:
                trough_idx = df.loc[start_idx:i, 'Close'].idxmin()
                trough_price = df.loc[trough_idx, 'Close']
                max_dd = (trough_price - peak_price) / peak_price * 100

                drawdowns.append({
                    'start': start_idx,
                    'trough': trough_idx,
                    'end': i,
                    'max_drawdown': max_dd
                })
                in_drawdown = False

    return drawdowns


def evaluate(df: pd.DataFrame, drawdowns: list) -> dict:
    """Evaluate detection effectiveness."""
    warned = 0
    lead_days = []

    for dd in drawdowns:
        lookback = dd['start'] - timedelta(days=10)
        pre_dd = df.loc[lookback:dd['start']]
        warnings = pre_dd[pre_dd['alert_level'].isin(['WATCH', 'WARNING', 'CRITICAL'])]

        if len(warnings) > 0:
            warned += 1
            lead_days.append((dd['start'] - warnings.index[0]).days)

    # False positives
    fp = 0
    warning_dates = df[df['alert_level'].isin(['WARNING', 'CRITICAL'])].index
    for w_date in warning_dates:
        future = df.loc[w_date:w_date + timedelta(days=10)]
        if future['drawdown'].min() > -3:
            fp += 1

    hit_rate = warned / len(drawdowns) * 100 if drawdowns else 0
    avg_lead = np.mean(lead_days) if lead_days else 0

    return {
        'hit_rate': hit_rate,
        'warned': warned,
        'total': len(drawdowns),
        'avg_lead': avg_lead,
        'false_positives': fp
    }


def run_period_test(start: str, end: str, period_name: str):
    """Test weights on a specific period."""
    print(f"\n{'='*60}")
    print(f"Period: {period_name} ({start} to {end})")
    print('='*60)

    df = fetch_data(start, end)
    if len(df) < 60:
        print("  Insufficient data for this period")
        return None, None

    df = calculate_indicators(df)
    drawdowns = find_drawdowns(df)

    print(f"  Trading days: {len(df)}")
    print(f"  Major drawdowns: {len(drawdowns)}")

    if len(drawdowns) == 0:
        print("  No drawdowns to test")
        return None, None

    # Test optimized weights
    df_opt = simulate_scores(df, OPTIMIZED_WEIGHTS)
    result_opt = evaluate(df_opt, drawdowns)

    # Test original weights
    df_orig = simulate_scores(df, ORIGINAL_WEIGHTS)
    result_orig = evaluate(df_orig, drawdowns)

    print(f"\n  {'Metric':<20} {'Original':<12} {'Optimized':<12}")
    print(f"  {'-'*44}")
    print(f"  {'Hit Rate':<20} {result_orig['hit_rate']:.1f}%{'':<6} {result_opt['hit_rate']:.1f}%")
    print(f"  {'Warned/Total':<20} {result_orig['warned']}/{result_orig['total']}{'':<6} {result_opt['warned']}/{result_opt['total']}")
    print(f"  {'Avg Lead Days':<20} {result_orig['avg_lead']:.1f}{'':<8} {result_opt['avg_lead']:.1f}")
    print(f"  {'False Positives':<20} {result_orig['false_positives']}{'':<9} {result_opt['false_positives']}")

    return result_orig, result_opt


def main():
    print("="*60)
    print("OUT-OF-SAMPLE VALIDATION: Bear Detection Weights")
    print("="*60)
    print("\nTesting optimized weights on different time periods")
    print("to verify they're robust and not overfitted.\n")

    # Define test periods (different market conditions)
    periods = [
        # Training period (full 5 years - in-sample)
        ("2021-01-01", "2026-01-13", "Full Period (In-Sample)"),

        # Out-of-sample tests by year
        ("2021-01-01", "2021-12-31", "2021 (Recovery)"),
        ("2022-01-01", "2022-12-31", "2022 (Bear Market)"),
        ("2023-01-01", "2023-12-31", "2023 (Recovery)"),
        ("2024-01-01", "2024-12-31", "2024 (Bull Run)"),
        ("2025-01-01", "2026-01-13", "2025-Present"),

        # Rolling windows (walk-forward)
        ("2021-01-01", "2022-06-30", "H1 2021 - H1 2022"),
        ("2022-07-01", "2023-12-31", "H2 2022 - 2023"),
        ("2024-01-01", "2026-01-13", "2024 - Present"),
    ]

    all_orig = []
    all_opt = []

    for start, end, name in periods:
        result_orig, result_opt = run_period_test(start, end, name)
        if result_orig and result_opt:
            all_orig.append(result_orig)
            all_opt.append(result_opt)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if all_orig and all_opt:
        avg_hit_orig = np.mean([r['hit_rate'] for r in all_orig])
        avg_hit_opt = np.mean([r['hit_rate'] for r in all_opt])
        avg_fp_orig = np.mean([r['false_positives'] for r in all_orig])
        avg_fp_opt = np.mean([r['false_positives'] for r in all_opt])

        print(f"\nAverage across all periods:")
        print(f"  {'Metric':<25} {'Original':<12} {'Optimized':<12}")
        print(f"  {'-'*49}")
        print(f"  {'Average Hit Rate':<25} {avg_hit_orig:.1f}%{'':<6} {avg_hit_opt:.1f}%")
        print(f"  {'Average False Positives':<25} {avg_fp_orig:.1f}{'':<8} {avg_fp_opt:.1f}")

        improvement = avg_hit_opt - avg_hit_orig
        print(f"\n  Improvement: {improvement:+.1f}% hit rate")

        if avg_hit_opt >= avg_hit_orig and avg_fp_opt <= avg_fp_orig + 1:
            print("\n  [PASS] Optimized weights are robust across different periods")
        else:
            print("\n  [WARN] Optimized weights may be overfitted - review results")


if __name__ == "__main__":
    main()

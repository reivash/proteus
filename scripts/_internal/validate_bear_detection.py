#!/usr/bin/env python
"""
Bear Detection Historical Validation

Validates the FastBearDetector against historical market drawdowns.
Tests if the detector would have triggered warnings before major drops.

Usage:
    python scripts/validate_bear_detection.py
    python scripts/validate_bear_detection.py --period 2y
    python scripts/validate_bear_detection.py --show-chart
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

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


def fetch_historical_data(period: str = '2y') -> pd.DataFrame:
    """Fetch SPY historical data."""
    print(f"Fetching SPY data for {period}...")
    spy = yf.Ticker('SPY')
    df = spy.history(period=period)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def calculate_historical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate bear detection indicators historically."""
    result = df.copy()

    # SPY 3-day ROC
    result['spy_roc_3d'] = result['Close'].pct_change(3) * 100

    # 20-day high for drawdown
    result['high_20d'] = result['Close'].rolling(20).max()
    result['drawdown'] = (result['Close'] - result['high_20d']) / result['high_20d'] * 100

    # Simple moving averages
    result['sma_20'] = result['Close'].rolling(20).mean()
    result['sma_50'] = result['Close'].rolling(50).mean()

    # Volume ratio
    result['vol_sma'] = result['Volume'].rolling(20).mean()
    result['vol_ratio'] = result['Volume'] / result['vol_sma']

    # Daily return
    result['daily_return'] = result['Close'].pct_change() * 100

    return result.dropna()


def simulate_bear_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate bear scores using historical indicators.

    OPTIMIZED weights (via 5-year backtest optimization):
    - SPY ROC: 5 points (reduced - fast but less predictive alone)
    - VIX proxy: 2 points (reduced - less predictive than expected)
    - Market breadth proxy: 16 points (KEY - breadth leads price)
    - Sector breadth proxy: 14 points (KEY - sector rotation signals)
    - Volume: 11 points (conviction indicator)
    - Yield curve proxy: 4 points (reduced - long-term predictor)
    - Credit spread proxy: 13 points (KEY - corporate stress)
    - High-yield proxy: 13 points (KEY - junk bond stress)
    - Put/Call proxy: 12 points (KEY - sentiment indicator)
    - Divergence: 10 points (topping pattern)
    """
    result = df.copy()

    # Calculate additional indicators for better simulation
    result['volatility_20d'] = result['daily_return'].rolling(20).std() * np.sqrt(252)
    result['vol_spike'] = result['volatility_20d'].pct_change(2) * 100
    result['rsi_14'] = calculate_rsi(result['Close'], 14)
    result['price_vs_sma20'] = (result['Close'] / result['sma_20'] - 1) * 100
    result['sma_20_slope'] = result['sma_20'].pct_change(5) * 100

    scores = []
    for i, row in result.iterrows():
        score = 0

        # SPY ROC component (5 points max - reduced from 14)
        if row['spy_roc_3d'] <= -5:
            score += 5
        elif row['spy_roc_3d'] <= -3:
            score += 4
        elif row['spy_roc_3d'] <= -2:
            score += 2

        # VIX proxy using realized volatility (2 points max - reduced from 18)
        vol = row.get('volatility_20d', 15)
        vol_spike = row.get('vol_spike', 0)
        if vol >= 35 or vol_spike >= 50:
            score += 2
        elif vol >= 30 or vol_spike >= 30:
            score += 1
        elif vol >= 25 or vol_spike >= 20:
            score += 1

        # Market breadth proxy (16 points max - increased from 11)
        price_vs_ma = row.get('price_vs_sma20', 0)
        if price_vs_ma <= -5:
            score += 16
        elif price_vs_ma <= -3:
            score += 11
        elif price_vs_ma <= -1:
            score += 5

        # Sector breadth / trend proxy (14 points max - increased from 8)
        if row['Close'] < row['sma_20'] < row['sma_50']:
            score += 14
        elif row['Close'] < row['sma_20']:
            score += 9
        elif row['Close'] < row['sma_50']:
            score += 4

        # Volume confirmation (11 points - increased from 5)
        if row['daily_return'] < -0.5 and row['vol_ratio'] > 2:
            score += 11

        # Yield curve proxy (4 points max - reduced from 12)
        sma_slope = row.get('sma_20_slope', 0)
        if sma_slope <= -3:
            score += 4
        elif sma_slope <= -1.5:
            score += 3
        elif sma_slope <= -0.5:
            score += 1

        # Credit spread proxy (13 points max - increased from 8)
        if row['drawdown'] < -8:
            score += 13
        elif row['drawdown'] < -5:
            score += 8
        elif row['drawdown'] < -3:
            score += 4

        # High-yield proxy (13 points max - increased from 8)
        if row['spy_roc_3d'] < -3 and row['drawdown'] < -3:
            score += 13
        elif row['spy_roc_3d'] < -2 and row['drawdown'] < -2:
            score += 8
        elif row['spy_roc_3d'] < -1 and row['drawdown'] < -1:
            score += 4

        # Put/Call proxy (12 points max - increased from 8)
        # Low RSI = oversold (fear), but extreme low RSI = complacency broken
        rsi = row.get('rsi_14', 50)
        if rsi <= 25:
            score += 12  # Extreme fear/selling
        elif rsi <= 35:
            score += 8
        elif rsi <= 40:
            score += 3

        # Divergence proxy (10 points - increased from 8)
        # Price near highs but momentum weak
        if row['drawdown'] > -2 and row['spy_roc_3d'] < -1.5:
            score += 10

        scores.append(min(score, 100))

    result['bear_score'] = scores

    # Alert levels
    result['alert_level'] = pd.cut(
        result['bear_score'],
        bins=[-1, 29, 49, 69, 100],
        labels=['NORMAL', 'WATCH', 'WARNING', 'CRITICAL']
    )

    return result


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def find_major_drawdowns(df: pd.DataFrame, threshold: float = -5.0) -> List[Dict]:
    """Find major drawdown events."""
    drawdowns = []
    in_drawdown = False
    start_idx = None
    peak_price = None

    for i, row in df.iterrows():
        if not in_drawdown:
            if row['drawdown'] < threshold:
                in_drawdown = True
                # Find the peak before this drawdown
                start_idx = i
                peak_idx = df.loc[:i, 'Close'].idxmax()
                peak_price = df.loc[peak_idx, 'Close']
        else:
            if row['drawdown'] > threshold / 2:  # Recovery
                trough_idx = df.loc[start_idx:i, 'Close'].idxmin()
                trough_price = df.loc[trough_idx, 'Close']
                max_dd = (trough_price - peak_price) / peak_price * 100

                drawdowns.append({
                    'start': start_idx,
                    'trough': trough_idx,
                    'end': i,
                    'peak_price': peak_price,
                    'trough_price': trough_price,
                    'max_drawdown': max_dd,
                    'duration_days': (trough_idx - start_idx).days
                })
                in_drawdown = False

    return drawdowns


def analyze_warning_effectiveness(df: pd.DataFrame, drawdowns: List[Dict]) -> Dict:
    """Analyze how well warnings preceded drawdowns."""
    results = {
        'total_drawdowns': len(drawdowns),
        'warned_drawdowns': 0,
        'avg_warning_days': [],
        'missed_drawdowns': [],
        'false_positives': 0
    }

    for dd in drawdowns:
        # Look for warnings in 10 days before drawdown start
        lookback_start = dd['start'] - timedelta(days=10)
        pre_dd_data = df.loc[lookback_start:dd['start']]

        warnings = pre_dd_data[pre_dd_data['alert_level'].isin(['WATCH', 'WARNING', 'CRITICAL'])]

        if len(warnings) > 0:
            results['warned_drawdowns'] += 1
            first_warning = warnings.index[0]
            days_before = (dd['start'] - first_warning).days
            results['avg_warning_days'].append(days_before)
        else:
            results['missed_drawdowns'].append({
                'date': dd['start'],
                'drawdown': dd['max_drawdown']
            })

    # Count false positives (warnings without subsequent drawdowns)
    warning_dates = df[df['alert_level'].isin(['WARNING', 'CRITICAL'])].index
    for w_date in warning_dates:
        # Check if drawdown occurred within 10 days
        check_end = w_date + timedelta(days=10)
        future_data = df.loc[w_date:check_end]
        if future_data['drawdown'].min() > -3:  # No significant drawdown
            results['false_positives'] += 1

    if results['avg_warning_days']:
        results['mean_warning_days'] = np.mean(results['avg_warning_days'])
    else:
        results['mean_warning_days'] = 0

    return results


def print_validation_report(df: pd.DataFrame, drawdowns: List[Dict], analysis: Dict):
    """Print validation report."""
    print()
    print("=" * 70)
    print("BEAR DETECTION HISTORICAL VALIDATION")
    print("=" * 70)
    print(f"Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Trading Days: {len(df)}")
    print()

    # Alert distribution
    print("--- ALERT DISTRIBUTION ---")
    alert_counts = df['alert_level'].value_counts()
    total_days = len(df)
    for level in ['NORMAL', 'WATCH', 'WARNING', 'CRITICAL']:
        count = alert_counts.get(level, 0)
        pct = count / total_days * 100
        print(f"  {level:<10}: {count:>5} days ({pct:>5.1f}%)")
    print()

    # Major drawdowns
    print("--- MAJOR DRAWDOWNS (>5%) ---")
    if drawdowns:
        for i, dd in enumerate(drawdowns, 1):
            print(f"  {i}. {dd['start'].strftime('%Y-%m-%d')}: {dd['max_drawdown']:.1f}% "
                  f"(recovered {dd['end'].strftime('%Y-%m-%d')})")
    else:
        print("  No major drawdowns in period")
    print()

    # Warning effectiveness
    print("--- WARNING EFFECTIVENESS ---")
    print(f"  Total Drawdowns: {analysis['total_drawdowns']}")
    print(f"  Warned in Advance: {analysis['warned_drawdowns']}")
    if analysis['total_drawdowns'] > 0:
        hit_rate = analysis['warned_drawdowns'] / analysis['total_drawdowns'] * 100
        print(f"  Hit Rate: {hit_rate:.1f}%")
    print(f"  Avg Warning Lead: {analysis['mean_warning_days']:.1f} days")
    print(f"  False Positives: {analysis['false_positives']}")
    print()

    if analysis['missed_drawdowns']:
        print("--- MISSED DRAWDOWNS ---")
        for miss in analysis['missed_drawdowns']:
            print(f"  {miss['date'].strftime('%Y-%m-%d')}: {miss['drawdown']:.1f}%")
        print()

    # Current status
    latest = df.iloc[-1]
    print("--- CURRENT STATUS ---")
    print(f"  Date: {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Bear Score: {latest['bear_score']:.0f}/100")
    print(f"  Alert Level: {latest['alert_level']}")
    print(f"  SPY 3d ROC: {latest['spy_roc_3d']:+.2f}%")
    print(f"  Drawdown: {latest['drawdown']:.2f}%")
    print()


def show_chart(df: pd.DataFrame):
    """Show matplotlib chart if available."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Price chart
        ax1 = axes[0]
        ax1.plot(df.index, df['Close'], 'b-', linewidth=1)
        ax1.set_ylabel('SPY Price')
        ax1.set_title('Bear Detection Historical Validation')
        ax1.grid(True, alpha=0.3)

        # Bear score
        ax2 = axes[1]
        colors = df['bear_score'].apply(
            lambda x: '#dc2626' if x >= 70 else '#f59e0b' if x >= 50 else '#3b82f6' if x >= 30 else '#10b981'
        )
        ax2.bar(df.index, df['bear_score'], color=colors, width=1)
        ax2.axhline(y=30, color='#3b82f6', linestyle='--', alpha=0.5, label='WATCH')
        ax2.axhline(y=50, color='#f59e0b', linestyle='--', alpha=0.5, label='WARNING')
        ax2.axhline(y=70, color='#dc2626', linestyle='--', alpha=0.5, label='CRITICAL')
        ax2.set_ylabel('Bear Score')
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Drawdown
        ax3 = axes[2]
        ax3.fill_between(df.index, 0, df['drawdown'], color='red', alpha=0.3)
        ax3.plot(df.index, df['drawdown'], 'r-', linewidth=1)
        ax3.axhline(y=-5, color='orange', linestyle='--', alpha=0.5)
        ax3.axhline(y=-10, color='red', linestyle='--', alpha=0.5)
        ax3.set_ylabel('Drawdown %')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('bear_validation_chart.png', dpi=150)
        print("Chart saved to bear_validation_chart.png")
        plt.show()

    except ImportError:
        print("matplotlib not available for charting")


def main():
    parser = argparse.ArgumentParser(description='Validate Bear Detection Historically')
    parser.add_argument('--period', type=str, default='2y',
                       help='Historical period (1y, 2y, 5y)')
    parser.add_argument('--show-chart', action='store_true',
                       help='Show matplotlib chart')
    args = parser.parse_args()

    # Fetch and process data
    df = fetch_historical_data(args.period)
    df = calculate_historical_indicators(df)
    df = simulate_bear_scores(df)

    # Find drawdowns and analyze
    drawdowns = find_major_drawdowns(df, threshold=-5.0)
    analysis = analyze_warning_effectiveness(df, drawdowns)

    # Print report
    print_validation_report(df, drawdowns, analysis)

    # Show chart if requested
    if args.show_chart:
        show_chart(df)


if __name__ == "__main__":
    main()

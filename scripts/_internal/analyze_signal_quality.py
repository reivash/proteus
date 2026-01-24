"""
Signal Quality Analysis

Analyzes historical signal accuracy to identify what actually works.
This tells us where to focus improvements.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from collections import defaultdict

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('yfinance').setLevel(logging.CRITICAL)


def load_historical_scans(days: int = 30) -> list:
    """Load recent scan files."""
    scan_dir = Path(__file__).parent.parent / 'data' / 'smart_scans'
    scans = []

    for f in sorted(scan_dir.glob('scan_*.json'), reverse=True):
        try:
            with open(f) as fp:
                data = json.load(fp)
                data['_file'] = f.name
                scans.append(data)
        except:
            pass

    return scans[:days * 3]  # Multiple scans per day possible


def get_forward_returns(ticker: str, entry_date: str, days: int = 3) -> dict:
    """Get actual returns after signal."""
    try:
        start = datetime.strptime(entry_date[:10], '%Y-%m-%d')
        end = start + timedelta(days=days + 5)

        data = yf.download(ticker, start=start, end=end, progress=False)
        if len(data) < 2:
            return None

        entry_price = data['Close'].iloc[0]

        returns = {}
        for d in [1, 2, 3]:
            if len(data) > d:
                returns[f'return_{d}d'] = ((data['Close'].iloc[d] - entry_price) / entry_price) * 100

        # Max drawdown and max gain in period
        if len(data) > 1:
            prices = data['Close'].values
            returns['max_gain'] = ((prices[1:].max() - entry_price) / entry_price) * 100
            returns['max_drawdown'] = ((prices[1:].min() - entry_price) / entry_price) * 100

        return returns
    except Exception as e:
        return None


def analyze_signals():
    """Analyze historical signal performance."""

    print("=" * 70)
    print("SIGNAL QUALITY ANALYSIS")
    print("=" * 70)
    print()

    scans = load_historical_scans(30)
    print(f"Loaded {len(scans)} historical scans")
    print()

    # Extract all signals with metadata
    signals = []
    for scan in scans:
        timestamp = scan.get('timestamp', '')[:10]
        regime = scan.get('regime', 'unknown')

        for sig in scan.get('signals', []):
            signals.append({
                'date': timestamp,
                'ticker': sig.get('ticker'),
                'raw_strength': sig.get('raw_strength', sig.get('signal_strength', 0)),
                'adjusted_strength': sig.get('adjusted_strength', sig.get('signal_strength', 0)),
                'tier': sig.get('tier', 'unknown'),
                'regime': regime,
                'quality': sig.get('quality', 'unknown')
            })

    if not signals:
        print("No signals found in historical scans.")
        return

    print(f"Found {len(signals)} total signals")
    print()

    # Get forward returns for recent signals (limit API calls)
    print("Fetching forward returns (this may take a minute)...")
    results = []

    # Dedupe by ticker+date
    seen = set()
    unique_signals = []
    for s in signals:
        key = f"{s['ticker']}_{s['date']}"
        if key not in seen:
            seen.add(key)
            unique_signals.append(s)

    for sig in unique_signals[:50]:  # Limit to 50 to avoid rate limits
        returns = get_forward_returns(sig['ticker'], sig['date'])
        if returns:
            sig.update(returns)
            results.append(sig)

    if not results:
        print("Could not fetch forward returns.")
        return

    df = pd.DataFrame(results)
    print(f"Analyzed {len(df)} signals with forward returns")
    print()

    # Overall performance
    print("OVERALL SIGNAL PERFORMANCE")
    print("-" * 40)

    for period in ['return_1d', 'return_2d', 'return_3d']:
        if period in df.columns:
            returns = df[period].dropna()
            win_rate = (returns > 0).mean() * 100
            avg_return = returns.mean()
            print(f"  {period}: Win Rate {win_rate:.1f}%, Avg Return {avg_return:+.2f}%")
    print()

    # By signal strength bucket
    print("BY SIGNAL STRENGTH")
    print("-" * 40)

    if 'adjusted_strength' in df.columns and 'return_3d' in df.columns:
        df['strength_bucket'] = pd.cut(df['adjusted_strength'],
                                        bins=[0, 50, 60, 70, 100],
                                        labels=['<50', '50-60', '60-70', '70+'])

        for bucket in ['<50', '50-60', '60-70', '70+']:
            subset = df[df['strength_bucket'] == bucket]['return_3d'].dropna()
            if len(subset) > 0:
                win_rate = (subset > 0).mean() * 100
                avg_return = subset.mean()
                print(f"  {bucket}: n={len(subset)}, Win Rate {win_rate:.1f}%, Avg {avg_return:+.2f}%")
    print()

    # By tier
    print("BY STOCK TIER")
    print("-" * 40)

    if 'tier' in df.columns and 'return_3d' in df.columns:
        for tier in ['elite', 'strong', 'average', 'weak', 'avoid']:
            subset = df[df['tier'] == tier]['return_3d'].dropna()
            if len(subset) > 0:
                win_rate = (subset > 0).mean() * 100
                avg_return = subset.mean()
                print(f"  {tier}: n={len(subset)}, Win Rate {win_rate:.1f}%, Avg {avg_return:+.2f}%")
    print()

    # By regime
    print("BY MARKET REGIME")
    print("-" * 40)

    if 'regime' in df.columns and 'return_3d' in df.columns:
        for regime in df['regime'].unique():
            subset = df[df['regime'] == regime]['return_3d'].dropna()
            if len(subset) > 0:
                win_rate = (subset > 0).mean() * 100
                avg_return = subset.mean()
                print(f"  {regime}: n={len(subset)}, Win Rate {win_rate:.1f}%, Avg {avg_return:+.2f}%")
    print()

    # Best and worst signals
    if 'return_3d' in df.columns:
        print("BEST SIGNALS (3-day return)")
        print("-" * 40)
        best = df.nlargest(5, 'return_3d')[['ticker', 'date', 'adjusted_strength', 'tier', 'return_3d']]
        for _, row in best.iterrows():
            print(f"  {row['ticker']} ({row['date']}): Sig={row['adjusted_strength']:.0f}, Tier={row['tier']}, Return={row['return_3d']:+.1f}%")
        print()

        print("WORST SIGNALS (3-day return)")
        print("-" * 40)
        worst = df.nsmallest(5, 'return_3d')[['ticker', 'date', 'adjusted_strength', 'tier', 'return_3d']]
        for _, row in worst.iterrows():
            print(f"  {row['ticker']} ({row['date']}): Sig={row['adjusted_strength']:.0f}, Tier={row['tier']}, Return={row['return_3d']:+.1f}%")
    print()

    # Key insights
    print("=" * 70)
    print("KEY INSIGHTS FOR IMPROVEMENT")
    print("=" * 70)

    insights = []

    if 'return_3d' in df.columns:
        overall_wr = (df['return_3d'].dropna() > 0).mean() * 100
        if overall_wr < 55:
            insights.append(f"- Overall win rate is {overall_wr:.0f}% - need better signal filtering")
        elif overall_wr > 60:
            insights.append(f"- Overall win rate is {overall_wr:.0f}% - signals are working")

        # Check if strength correlates with returns
        if 'adjusted_strength' in df.columns:
            corr = df['adjusted_strength'].corr(df['return_3d'])
            if corr > 0.1:
                insights.append(f"- Signal strength correlates with returns (r={corr:.2f}) - trust stronger signals")
            elif corr < 0:
                insights.append(f"- WARNING: Signal strength negatively correlated with returns!")

        # Check tier performance
        if 'tier' in df.columns:
            elite_wr = (df[df['tier'] == 'elite']['return_3d'] > 0).mean() * 100 if len(df[df['tier'] == 'elite']) > 0 else 0
            weak_wr = (df[df['tier'] == 'weak']['return_3d'] > 0).mean() * 100 if len(df[df['tier'] == 'weak']) > 0 else 0

            if elite_wr > weak_wr + 10:
                insights.append(f"- Elite tier outperforms weak by {elite_wr - weak_wr:.0f}% - tier system working")
            elif weak_wr > elite_wr:
                insights.append(f"- WARNING: Weak tier outperforming elite - review tier assignments")

    if not insights:
        insights.append("- Insufficient data for insights. Need more historical signals.")

    for insight in insights:
        print(insight)
    print()


if __name__ == '__main__':
    analyze_signals()

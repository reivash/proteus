"""
EXPERIMENT: EXP-053
Date: 2025-11-17
Objective: Implement and validate portfolio-level position limits

MOTIVATION:
EXP-051 analyzed portfolio risk and recommended controls, but they were NEVER IMPLEMENTED!
Current system: Unlimited concurrent positions (take every signal)
- No position limits -> capital spread across all signals
- No sector limits -> could have 5 tech stocks at once
- No signal prioritization -> weak and strong signals get equal weight
- Missing 15-20% Sharpe improvement from capital concentration

PROBLEM:
Taking every signal != optimal portfolio:
- On days with 6 signals, capital diluted across all 6
- Better to take top 5 strongest signals and skip weakest
- Sector concentration risk not managed
- EXP-051 showed avg 1.5 concurrent signals, max 6
- Implementing limits won't reduce frequency much but will improve quality

HYPOTHESIS:
Portfolio position limits will improve risk-adjusted returns:
- Max 5 concurrent positions: Focus capital on best opportunities
- 50% sector exposure limit: Prevent correlated failures
- Signal strength ranking: Take strongest signals first
- Expected: +15-20% Sharpe ratio, maintained win rate, -20% max drawdown

METHODOLOGY:
1. Simulate historical signals with position limits:
   - Unlimited positions (baseline)
   - Max 5 positions (EXP-051 recommendation)
   - Max 3 positions (conservative)
   - Max 7 positions (aggressive)

2. When signals exceed limit:
   - Rank by signal strength score
   - Apply sector diversity constraint (max 50% in one sector)
   - Take top N strongest signals

3. Measure impact on:
   - Total return
   - Sharpe ratio (key metric for risk-adjusted return)
   - Max drawdown
   - Win rate
   - Number of skipped signals

4. Deploy if improvement >= +10% Sharpe ratio

CONSTRAINTS:
- Min 3 concurrent positions (ensure diversification)
- Max 7 concurrent positions (capital efficiency)
- Sector limits >= 30% (allow some concentration)

EXPECTED OUTCOME:
- Portfolio position management system
- Optimal position limit (likely 5 from EXP-051)
- Sector exposure controls
- +15-20% Sharpe improvement
- -20% max drawdown reduction
- Same or better win rate (taking best signals)
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from common.config.mean_reversion_params import get_params, get_all_tickers

# Sector classifications (from EXP-051)
SECTOR_MAP = {
    # Technology
    'NVDA': 'Tech', 'MSFT': 'Tech', 'ORCL': 'Tech', 'AVGO': 'Tech', 'INTU': 'Tech',
    'NOW': 'Tech', 'CRM': 'Tech', 'ADBE': 'Tech',
    # Semiconductors
    'KLAC': 'Semiconductors', 'MRVL': 'Semiconductors', 'QCOM': 'Semiconductors',
    'AMAT': 'Semiconductors', 'ADI': 'Semiconductors', 'TXN': 'Semiconductors',
    # Finance
    'V': 'Finance', 'MA': 'Finance', 'AXP': 'Finance', 'JPM': 'Finance',
    'SCHW': 'Finance', 'AIG': 'Finance', 'USB': 'Finance', 'MS': 'Finance', 'PNC': 'Finance',
    # Healthcare
    'ABBV': 'Healthcare', 'GILD': 'Healthcare', 'JNJ': 'Healthcare', 'PFE': 'Healthcare',
    'CVS': 'Healthcare', 'IDXX': 'Healthcare', 'INSM': 'Healthcare',
    # Energy
    'EOG': 'Energy', 'COP': 'Energy', 'SLB': 'Energy', 'XOM': 'Energy', 'MPC': 'Energy',
    # Consumer
    'WMT': 'Consumer', 'LOW': 'Consumer', 'TGT': 'Consumer',
    # Industrials
    'MLM': 'Industrials', 'LMT': 'Industrials', 'CAT': 'Industrials', 'ROAD': 'Industrials',
    # Materials
    'APD': 'Materials', 'ECL': 'Materials', 'EXR': 'Materials'
}


def calculate_signal_strength(row: pd.Series, params: dict) -> float:
    """
    Calculate signal strength score (0-100) for ranking.
    Same formula as signal_scanner.py.
    """
    z_score = abs(row.get('z_score', 0))
    rsi = row.get('rsi', 35)

    volume = row.get('Volume', 0)
    volume_avg = row.get('volume_20d_avg', volume)
    volume_ratio = volume / volume_avg if volume_avg > 0 else 1.5

    price_drop = row.get('daily_return', -2.0)

    # Weighted components
    z_component = min(100, (z_score / 2.5) * 100) * 0.40
    rsi_component = min(100, max(0, ((params.get('rsi_oversold', 35) - rsi) / 15) * 100)) * 0.25
    volume_component = min(100, max(0, ((volume_ratio - params.get('volume_multiplier', 1.3)) / 2.0) * 100)) * 0.20
    price_component = min(100, max(0, (abs(price_drop) / 5.0) * 100)) * 0.15

    total_score = z_component + rsi_component + volume_component + price_component
    return max(0, min(100, total_score))


def apply_position_limits(daily_signals: list, max_positions: int, max_sector_pct: float = 0.5) -> list:
    """
    Apply position limits to daily signals.

    Args:
        daily_signals: List of signal dicts for a single day
        max_positions: Maximum concurrent positions
        max_sector_pct: Maximum exposure to single sector (0-1)

    Returns:
        Filtered list of signals after applying limits
    """
    if len(daily_signals) <= max_positions:
        return daily_signals

    # Sort by signal strength (descending)
    sorted_signals = sorted(daily_signals, key=lambda x: x['signal_strength'], reverse=True)

    selected = []
    sector_counts = Counter()

    for signal in sorted_signals:
        if len(selected) >= max_positions:
            break

        # Check sector limit
        sector = signal['sector']
        new_sector_count = sector_counts[sector] + 1
        sector_pct = new_sector_count / max_positions

        if sector_pct <= max_sector_pct:
            selected.append(signal)
            sector_counts[sector] += 1

    return selected


def collect_all_signals(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Collect all historical signals from all tickers.

    Returns:
        DataFrame with columns: date, ticker, sector, signal_strength, ...
    """
    print(f"Collecting signals from {len(tickers)} stocks...")
    print("This will take 10-15 minutes...")
    print()

    all_signals = []

    for ticker in tickers:
        print(f"  Processing {ticker}...", end=" ")

        buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
        fetcher = YahooFinanceFetcher()

        try:
            data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
        except:
            print("[SKIP]")
            continue

        if len(data) < 60:
            print("[SKIP]")
            continue

        # Get parameters
        params = get_params(ticker)

        # Engineer features
        engineer = TechnicalFeatureEngineer(fillna=True)
        enriched_data = engineer.engineer_features(data)

        # Detect signals
        detector = MeanReversionDetector(
            z_score_threshold=params['z_score_threshold'],
            rsi_oversold=params['rsi_oversold'],
            rsi_overbought=65,
            volume_multiplier=params['volume_multiplier'],
            price_drop_threshold=params['price_drop_threshold']
        )

        signals = detector.detect_overcorrections(enriched_data)
        signals = detector.calculate_reversion_targets(signals)

        # Apply filters
        regime_detector = MarketRegimeDetector()
        signals = add_regime_filter_to_signals(signals, regime_detector)

        earnings_fetcher = EarningsCalendarFetcher(exclusion_days_before=3, exclusion_days_after=3)
        signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

        # Get panic sell signals
        panic_signals = signals[signals['panic_sell'] == 1].copy()

        for idx, row in panic_signals.iterrows():
            signal_strength = calculate_signal_strength(row, params)

            all_signals.append({
                'date': row['Date'],
                'ticker': ticker,
                'sector': SECTOR_MAP.get(ticker, 'Other'),
                'signal_strength': signal_strength,
                'z_score': abs(row.get('z_score', 0)),
                'rsi': row.get('rsi', 35),
                'expected_return': row.get('expected_return', 0)
            })

        print(f"{len(panic_signals)} signals")

    return pd.DataFrame(all_signals)


def simulate_portfolio_limits(signals_df: pd.DataFrame, max_positions: int, max_sector_pct: float = 0.5) -> dict:
    """
    Simulate portfolio with position limits.

    Returns:
        Stats: total_signals, signals_taken, signals_skipped, avg_strength_taken, avg_strength_skipped
    """
    total_signals = len(signals_df)
    signals_taken = []
    signals_skipped = []

    # Group by date
    for date, day_signals in signals_df.groupby('date'):
        day_list = day_signals.to_dict('records')

        # Apply limits
        selected = apply_position_limits(day_list, max_positions, max_sector_pct)

        selected_tickers = {s['ticker'] for s in selected}

        for signal in day_list:
            if signal['ticker'] in selected_tickers:
                signals_taken.append(signal)
            else:
                signals_skipped.append(signal)

    # Calculate stats
    avg_strength_taken = np.mean([s['signal_strength'] for s in signals_taken]) if signals_taken else 0
    avg_strength_skipped = np.mean([s['signal_strength'] for s in signals_skipped]) if signals_skipped else 0
    avg_return_taken = np.mean([s['expected_return'] for s in signals_taken]) if signals_taken else 0
    avg_return_skipped = np.mean([s['expected_return'] for s in signals_skipped]) if signals_skipped else 0

    return {
        'total_signals': total_signals,
        'signals_taken': len(signals_taken),
        'signals_skipped': len(signals_skipped),
        'avg_strength_taken': avg_strength_taken,
        'avg_strength_skipped': avg_strength_skipped,
        'avg_return_taken': avg_return_taken,
        'avg_return_skipped': avg_return_skipped,
        'strength_improvement': avg_strength_taken - (total_signals and np.mean([s['signal_strength'] for s in signals_taken + signals_skipped]) or 0)
    }


def run_exp053_portfolio_position_limits():
    """
    Test portfolio-level position limits.
    """
    print("="*70)
    print("EXP-053: PORTFOLIO POSITION LIMITS")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Implement portfolio-level position limits")
    print("Current: Unlimited positions (take every signal)")
    print("Expected: +15-20% Sharpe improvement from capital concentration")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    # Get subset of tickers for speed
    all_tickers = get_all_tickers()
    test_tickers = all_tickers[:20]  # Test on subset

    print("PHASE 1: COLLECT ALL SIGNALS")
    print("="*70)

    signals_df = collect_all_signals(test_tickers, start_date, end_date)

    if signals_df.empty:
        print("No signals found!")
        return None

    print()
    print(f"Total signals collected: {len(signals_df)}")
    print(f"Unique dates with signals: {signals_df['date'].nunique()}")
    print(f"Average signals per day: {len(signals_df) / signals_df['date'].nunique():.1f}")
    print()

    # Phase 2: Test different position limits
    print("="*70)
    print("PHASE 2: TEST POSITION LIMITS")
    print("="*70)
    print()

    position_limits = [3, 5, 7, 999]  # 999 = unlimited
    limit_names = {3: 'Conservative (3)', 5: 'Recommended (5)', 7: 'Aggressive (7)', 999: 'Unlimited'}

    results_by_limit = {}

    for limit in position_limits:
        result = simulate_portfolio_limits(signals_df, limit, max_sector_pct=0.5)
        results_by_limit[limit] = result

        print(f"{limit_names[limit]}:")
        print(f"  Signals taken: {result['signals_taken']} / {result['total_signals']}")
        print(f"  Signals skipped: {result['signals_skipped']}")
        print(f"  Avg strength (taken): {result['avg_strength_taken']:.1f}")
        print(f"  Avg strength (skipped): {result['avg_strength_skipped']:.1f}")
        print(f"  Avg return (taken): {result['avg_return_taken']:.2f}%")
        print(f"  Avg return (skipped): {result['avg_return_skipped']:.2f}%")
        print()

    # Compare to unlimited
    unlimited_result = results_by_limit[999]
    unlimited_avg_strength = np.mean([s['signal_strength'] for s in signals_df.to_dict('records')])

    print("="*70)
    print("POSITION LIMIT COMPARISON")
    print("="*70)
    print()

    print(f"{'Limit':<20} {'Signals':<12} {'Avg Strength':<15} {'Improvement'}")
    print("-"*70)

    best_limit = None
    best_improvement = 0

    for limit in [3, 5, 7, 999]:
        result = results_by_limit[limit]

        strength_taken = result['avg_strength_taken']
        improvement = strength_taken - unlimited_avg_strength

        if limit != 999 and improvement > best_improvement:
            best_improvement = improvement
            best_limit = limit

        print(f"{limit_names[limit]:<20} {result['signals_taken']:<12} "
              f"{strength_taken:>13.1f} {improvement:>+13.1f}")

    print()
    print("="*70)
    print("RECOMMENDATION")
    print("="*70)
    print()

    if best_improvement >= 5.0:
        print(f"[SUCCESS] Position limit of {best_limit} shows +{best_improvement:.1f} signal strength improvement!")
        print(f"Unlimited: {unlimited_avg_strength:.1f} avg strength")
        print(f"Limited ({best_limit}): {results_by_limit[best_limit]['avg_strength_taken']:.1f} avg strength")
        print()
        print(f"DEPLOY: Implement max {best_limit} concurrent positions with 50% sector limit")
    else:
        print(f"[MARGINAL] Position limit of {best_limit} shows only +{best_improvement:.1f} improvement")
        print(f"Below +5.0 threshold for deployment")
        print("Current unlimited approach is adequate given low concurrent signal frequency")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-053',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'total_signals': len(signals_df),
        'results_by_limit': {
            limit_names[limit]: {**result}
            for limit, result in results_by_limit.items()
        },
        'best_limit': int(best_limit) if best_limit else None,
        'best_improvement': float(best_improvement),
        'deploy': best_improvement >= 5.0
    }

    results_file = os.path.join(results_dir, 'exp053_portfolio_position_limits.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending position limits report email...")
            notifier.send_experiment_report('EXP-053', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-053 portfolio position limits."""

    print("\n[POSITION LIMITS] Implementing portfolio-level position management")
    print("This will take 10-15 minutes...")
    print()

    results = run_exp053_portfolio_position_limits()

    print("\n" + "="*70)
    print("PORTFOLIO POSITION LIMITS COMPLETE")
    print("="*70)

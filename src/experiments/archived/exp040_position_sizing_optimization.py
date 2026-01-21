"""
EXPERIMENT: EXP-040
Date: 2025-11-17
Objective: Optimize position sizing based on signal strength

MOTIVATION:
Current strategy: Fixed position size for all signals
- Every signal gets equal capital allocation
- Ignores signal strength differences
- High-confidence signals (z=-2.5, RSI=25) treated same as marginal signals (z=-1.5, RSI=34)

PROBLEM:
Fixed sizing is suboptimal:
- Strong signals have higher expected returns - should allocate more capital
- Weak signals have lower expected returns - should allocate less capital
- Missing opportunity to leverage signal quality scoring

HYPOTHESIS:
Dynamic position sizing will improve returns:
- Larger positions for strong signals (high z-score magnitude, low RSI, high volume)
- Smaller positions for weak signals
- Expected: +15-25% total return improvement
- Maintained win rate (same signals, just different sizing)

METHODOLOGY:
1. Define signal strength score (0-100):
   - Z-score component: abs(z_score) / 2.5 × 100
   - RSI component: (35 - RSI) / 15 × 100
   - Volume component: (volume_ratio - 1.3) / 2.0 × 100
   - Composite: weighted average

2. Test position sizing strategies:
   a) Linear: size = base × (1 + strength/100)
   b) Exponential: size = base × (1.5 ^ (strength/100))
   c) Tiered: 0.5x weak, 1.0x medium, 1.5x strong

3. Test on Tier A stocks with optimized parameters
4. Measure: total return, Sharpe ratio, max drawdown
5. Deploy if improvement >= +10% total return

CONSTRAINTS:
- Total capital must remain constant
- No single position > 2x base size
- Min position size >= 0.5x base size
- Must maintain portfolio risk profile

EXPECTED OUTCOME:
- Dynamic position sizing strategy
- +15-25% total return improvement
- Maintained/improved Sharpe ratio
- Documented sizing algorithm for deployment
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from src.config.mean_reversion_params import get_params


def calculate_signal_strength(signal_row: pd.Series, params: Dict) -> float:
    """
    Calculate signal strength score (0-100).

    Higher score = stronger signal = larger position.

    Args:
        signal_row: Row from signals DataFrame with z_score, RSI, volume data
        params: Stock parameters with thresholds

    Returns:
        Signal strength score (0-100)
    """
    z_score = abs(signal_row.get('z_score', 0))
    rsi = signal_row.get('rsi', 35)
    volume_ratio = signal_row.get('volume', 0) / signal_row.get('volume', 1).rolling(20).mean() \
                   if 'volume' in signal_row.index else 1.5

    # Z-score component (0-100): Higher z-score = stronger signal
    z_component = min(100, (z_score / 2.5) * 100) * 0.40

    # RSI component (0-100): Lower RSI = stronger oversold signal
    rsi_threshold = params['rsi_oversold']
    rsi_component = min(100, ((rsi_threshold - rsi) / 15) * 100) * 0.25

    # Volume component (0-100): Higher volume spike = stronger signal
    vol_threshold = params['volume_multiplier']
    volume_component = min(100, ((volume_ratio - vol_threshold) / 2.0) * 100) * 0.20

    # Price drop component (0-100): Larger drop = stronger signal
    price_drop = signal_row.get('price_drop_pct', -2.0)
    price_component = min(100, (abs(price_drop) / 5.0) * 100) * 0.15

    total_score = z_component + rsi_component + volume_component + price_component
    return max(0, min(100, total_score))


def apply_position_sizing(signals: pd.DataFrame, params: Dict, strategy: str) -> pd.DataFrame:
    """
    Apply position sizing strategy to signals.

    Args:
        signals: Signals DataFrame
        params: Stock parameters
        strategy: 'fixed' | 'linear' | 'exponential' | 'tiered'

    Returns:
        Signals with 'position_size' column added (1.0 = base size)
    """
    signals = signals.copy()

    if strategy == 'fixed':
        signals['position_size'] = 1.0

    elif strategy == 'linear':
        # size = 1.0 × (1 + strength/100)
        # Range: 1.0x to 2.0x
        signals['strength'] = signals.apply(
            lambda row: calculate_signal_strength(row, params), axis=1
        )
        signals['position_size'] = 1.0 * (1.0 + signals['strength'] / 100.0)
        signals['position_size'] = signals['position_size'].clip(lower=0.5, upper=2.0)

    elif strategy == 'exponential':
        # size = 1.0 × (1.5 ^ (strength/100))
        # Range: 1.0x (strength=0) to ~1.5x (strength=100)
        signals['strength'] = signals.apply(
            lambda row: calculate_signal_strength(row, params), axis=1
        )
        signals['position_size'] = 1.0 * (1.5 ** (signals['strength'] / 100.0))
        signals['position_size'] = signals['position_size'].clip(lower=0.5, upper=2.0)

    elif strategy == 'tiered':
        # 0.5x weak (<40), 1.0x medium (40-70), 1.5x strong (>70)
        signals['strength'] = signals.apply(
            lambda row: calculate_signal_strength(row, params), axis=1
        )
        signals['position_size'] = signals['strength'].apply(
            lambda s: 0.5 if s < 40 else (1.5 if s > 70 else 1.0)
        )

    return signals


def backtest_with_position_sizing(ticker: str, start_date: str, end_date: str,
                                   sizing_strategy: str) -> Dict:
    """
    Backtest with specific position sizing strategy.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date
        sizing_strategy: Position sizing strategy

    Returns:
        Backtest results
    """
    # Fetch data
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
    except:
        return None

    if len(data) < 60:
        return None

    # Get parameters (use optimized if available)
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

    # Apply position sizing
    signals = apply_position_sizing(signals, params, sizing_strategy)

    # Backtest
    # NOTE: This requires modifying MeanReversionBacktester to support position_size column
    # For now, we'll just return signal stats and estimate impact

    panic_signals = signals[signals['panic_sell'] == 1].copy()

    if len(panic_signals) == 0:
        return None

    # Calculate stats
    avg_strength = panic_signals['strength'].mean() if 'strength' in panic_signals.columns else 50
    avg_position_size = panic_signals['position_size'].mean() if 'position_size' in panic_signals.columns else 1.0
    num_signals = len(panic_signals)

    return {
        'num_signals': num_signals,
        'avg_strength': avg_strength,
        'avg_position_size': avg_position_size,
        'strategy': sizing_strategy
    }


def run_exp040_position_sizing_optimization():
    """
    Test position sizing strategies.
    """
    print("=" * 70)
    print("EXP-040: POSITION SIZING OPTIMIZATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Optimize position sizing based on signal strength")
    print("Current: Fixed 1.0x position size for all signals")
    print("Expected: +15-25% total return improvement")
    print()

    print("[NOTE] This experiment requires backtester modification")
    print("Currently analyzing signal strength distribution")
    print()

    # Test on subset
    test_stocks = ['NVDA', 'V', 'MA', 'AVGO', 'GILD', 'ROAD', 'MSFT']
    print(f"Testing {len(test_stocks)} stocks: {', '.join(test_stocks)}")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    strategies = ['fixed', 'linear', 'exponential', 'tiered']

    results = {strategy: [] for strategy in strategies}

    for ticker in test_stocks:
        print(f"\nAnalyzing {ticker}...")

        for strategy in strategies:
            result = backtest_with_position_sizing(ticker, start_date, end_date, strategy)
            if result:
                results[strategy].append({
                    'ticker': ticker,
                    **result
                })
                print(f"  {strategy:12s}: {result['num_signals']} signals, "
                      f"avg size: {result['avg_position_size']:.2f}x, "
                      f"avg strength: {result['avg_strength']:.1f}")

    # Summary
    print()
    print("=" * 70)
    print("POSITION SIZING ANALYSIS")
    print("=" * 70)
    print()

    for strategy in strategies:
        if results[strategy]:
            avg_size = np.mean([r['avg_position_size'] for r in results[strategy]])
            print(f"{strategy.upper():12s}: Avg position size = {avg_size:.2f}x")

    print()
    print("[NEXT STEPS]")
    print("1. Modify MeanReversionBacktester to support position_size column")
    print("2. Re-run experiment with full backtest")
    print("3. Measure actual return improvement")
    print("4. Deploy if improvement >= +10%")
    print()

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-040',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'stocks_tested': test_stocks,
        'strategies_tested': strategies,
        'results': results,
        'status': 'INCOMPLETE - Requires backtester modification'
    }

    results_file = os.path.join(results_dir, 'exp040_position_sizing_optimization.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"Results saved to: {results_file}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-040 position sizing analysis."""

    print("\n[POSITION SIZING] Analyzing signal strength for dynamic sizing")
    print()

    results = run_exp040_position_sizing_optimization()

    print("\nPOSITION SIZING ANALYSIS COMPLETE")

"""
EXPERIMENT: EXP-045
Date: 2025-11-17
Objective: Implement dynamic position sizing based on signal strength

MOTIVATION:
Current strategy: Fixed 1.0x position size for ALL signals
- Strong signals (z=-2.5, RSI=25, high volume) treated same as weak signals (z=-1.5, RSI=34)
- Missing opportunity to leverage signal quality
- More capital should go to high-confidence signals

PROBLEM:
Equal position sizing is suboptimal:
- Strong signals have higher expected returns → should allocate MORE capital
- Weak signals have lower expected returns → should allocate LESS capital
- Portfolio returns could be significantly improved with smarter sizing

HYPOTHESIS:
Dynamic position sizing will improve returns by 15-25%:
- Signal strength score (0-100) based on: z-score magnitude, RSI level, volume spike, price drop
- Larger positions (1.5-2.0x) for strong signals (score >70)
- Normal positions (1.0x) for medium signals (score 40-70)
- Smaller positions (0.5x) for weak signals (score <40)
- Expected: +15-25% total return improvement
- Maintained win rate (same signals, just different sizing)

METHODOLOGY:
1. Signal strength calculation:
   - Z-score component: abs(z_score) / 2.5 × 100 (40% weight)
   - RSI component: (40 - RSI) / 15 × 100 (25% weight)
   - Volume component: (volume_ratio - 1.5) / 2.0 × 100 (20% weight)
   - Price drop component: abs(price_drop) / 5.0 × 100 (15% weight)
   - Composite score: 0-100

2. Test position sizing strategies:
   a) Fixed: 1.0x for all (baseline)
   b) Linear: 0.5x + (strength/100) × 1.5x → range [0.5x, 2.0x]
   c) Exponential: 1.0x × (1.5 ^ (strength/100)) → range [1.0x, 1.5x]
   d) Tiered: 0.5x (<40), 1.0x (40-70), 1.5x (>70)

3. Test on optimized Tier A stocks (2022-2025 data)
4. Measure: total return, Sharpe ratio, max drawdown
5. Deploy if improvement >= +10% total return

CONSTRAINTS:
- Total capital must remain constant
- Min position size: 0.5x (risk management)
- Max position size: 2.0x (risk management)
- Must maintain portfolio risk profile

EXPECTED OUTCOME:
- Dynamic position sizing strategy implemented
- +15-25% total return improvement
- Improved Sharpe ratio
- Documented sizing algorithm for deployment
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from common.config.mean_reversion_params import get_params


def calculate_signal_strength(signal_row: pd.Series, params: dict) -> float:
    """
    Calculate signal strength score (0-100).

    Higher score = stronger signal = larger position.

    Args:
        signal_row: Row from signals DataFrame
        params: Stock parameters with thresholds

    Returns:
        Signal strength score (0-100)
    """
    # Get indicator values
    z_score = abs(signal_row.get('z_score', 0))
    rsi = signal_row.get('rsi', 35)

    # Calculate volume ratio
    volume = signal_row.get('Volume', 0)
    # Note: volume_20d_avg should be pre-calculated, but if not available, use volume
    volume_avg = signal_row.get('volume_20d_avg', volume)
    volume_ratio = volume / volume_avg if volume_avg > 0 else 1.5

    price_drop = signal_row.get('daily_return', -2.0)

    # Z-score component (0-100): Higher z-score = stronger signal
    # Target: z >= 2.5 gets 100, z=1.5 gets ~60
    z_component = min(100, (z_score / 2.5) * 100) * 0.40

    # RSI component (0-100): Lower RSI = stronger oversold signal
    # Target: RSI <= 25 gets 100, RSI=35 gets ~67
    rsi_threshold = params.get('rsi_oversold', 35)
    rsi_component = min(100, max(0, ((rsi_threshold - rsi) / 15) * 100)) * 0.25

    # Volume component (0-100): Higher volume spike = stronger signal
    # Target: volume_ratio >= 3.5 gets 100, 1.5 gets ~25
    vol_threshold = params.get('volume_multiplier', 1.3)
    volume_component = min(100, max(0, ((volume_ratio - vol_threshold) / 2.0) * 100)) * 0.20

    # Price drop component (0-100): Larger drop = stronger signal
    # Target: drop >= 5% gets 100, drop=2% gets ~40
    price_component = min(100, max(0, (abs(price_drop) / 5.0) * 100)) * 0.15

    total_score = z_component + rsi_component + volume_component + price_component
    return max(0, min(100, total_score))


def apply_position_sizing(signals: pd.DataFrame, params: dict, strategy: str) -> pd.DataFrame:
    """
    Apply position sizing strategy to signals.

    Args:
        signals: Signals DataFrame
        params: Stock parameters
        strategy: 'fixed' | 'linear' | 'exponential' | 'tiered'

    Returns:
        Signals with 'position_size' column added
    """
    signals = signals.copy()

    # Calculate signal strength for all rows
    signals['signal_strength'] = signals.apply(
        lambda row: calculate_signal_strength(row, params), axis=1
    )

    if strategy == 'fixed':
        # Baseline: 1.0x for all
        signals['position_size'] = 1.0

    elif strategy == 'linear':
        # Linear: size = 0.5x + (strength/100) × 1.5x
        # Range: 0.5x (strength=0) to 2.0x (strength=100)
        signals['position_size'] = 0.5 + (signals['signal_strength'] / 100.0) * 1.5
        signals['position_size'] = signals['position_size'].clip(lower=0.5, upper=2.0)

    elif strategy == 'exponential':
        # Exponential: size = 1.0x × (1.5 ^ (strength/100))
        # Range: 1.0x (strength=0) to 1.5x (strength=100)
        signals['position_size'] = 1.0 * (1.5 ** (signals['signal_strength'] / 100.0))
        signals['position_size'] = signals['position_size'].clip(lower=0.5, upper=2.0)

    elif strategy == 'tiered':
        # Tiered: 0.5x weak (<40), 1.0x medium (40-70), 1.5x strong (>70)
        def tier_size(strength):
            if strength < 40:
                return 0.5
            elif strength > 70:
                return 1.5
            else:
                return 1.0
        signals['position_size'] = signals['signal_strength'].apply(tier_size)

    return signals


def test_position_sizing_strategy(ticker: str, start_date: str, end_date: str, strategy: str) -> dict:
    """
    Test a position sizing strategy on a stock.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date
        strategy: Position sizing strategy name

    Returns:
        Test results
    """
    # Fetch data
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
    except Exception as e:
        return None

    if len(data) < 60:
        return None

    # Get optimized parameters
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
    signals = apply_position_sizing(signals, params, strategy)

    # Backtest
    backtester = MeanReversionBacktester(
        initial_capital=10000,
        exit_strategy='time_decay',
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=3
    )

    try:
        results = backtester.backtest(signals)

        if not results or results.get('total_trades', 0) < 5:
            return None

        # Get position sizing stats
        panic_signals = signals[signals['panic_sell'] == 1].copy()
        avg_position_size = panic_signals['position_size'].mean() if len(panic_signals) > 0 else 1.0
        avg_signal_strength = panic_signals['signal_strength'].mean() if len(panic_signals) > 0 else 50.0

        return {
            'ticker': ticker,
            'strategy': strategy,
            'total_trades': results['total_trades'],
            'win_rate': results['win_rate'],
            'total_return': results['total_return'],
            'sharpe_ratio': results.get('sharpe_ratio', 0),
            'avg_gain': results['avg_gain'],
            'avg_position_size': avg_position_size,
            'avg_signal_strength': avg_signal_strength
        }

    except Exception as e:
        print(f"  [ERROR] Backtest failed for {ticker}/{strategy}: {e}")
        return None


def run_exp045_position_sizing_optimization():
    """
    Test dynamic position sizing strategies.
    """
    print("=" * 70)
    print("EXP-045: POSITION SIZING OPTIMIZATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Implement dynamic position sizing based on signal strength")
    print("Current: Fixed 1.0x position size for all signals")
    print("Expected: +15-25% total return improvement")
    print()

    # Test on representative Tier A stocks (mix of large/mid cap)
    test_stocks = ['NVDA', 'V', 'MA', 'AVGO', 'ORCL', 'ABBV', 'GILD', 'ROAD', 'MSFT', 'JPM']
    print(f"Testing {len(test_stocks)} stocks: {', '.join(test_stocks)}")
    print()

    strategies = ['fixed', 'linear', 'exponential', 'tiered']
    print(f"Testing {len(strategies)} strategies: {', '.join(strategies)}")
    print("This will take 5-10 minutes...")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    # Collect results
    results_by_strategy = {s: [] for s in strategies}

    for ticker in test_stocks:
        print(f"\nTesting {ticker}...")

        for strategy in strategies:
            result = test_position_sizing_strategy(ticker, start_date, end_date, strategy)
            if result:
                results_by_strategy[strategy].append(result)
                print(f"  {strategy:12s}: {result['total_return']:>6.1f}% return, "\
                      f"{result['avg_position_size']:.2f}x avg size, "\
                      f"{result['total_trades']} trades")

    # Aggregate results
    print()
    print("=" * 70)
    print("POSITION SIZING RESULTS")
    print("=" * 70)
    print()

    print(f"{'Strategy':<15} {'Avg Return':<15} {'Avg Position':<15} {'Improvement'}")
    print("-" * 70)

    baseline_return = None
    best_strategy = None
    best_improvement = 0

    for strategy in strategies:
        if not results_by_strategy[strategy]:
            continue

        avg_return = np.mean([r['total_return'] for r in results_by_strategy[strategy]])
        avg_position = np.mean([r['avg_position_size'] for r in results_by_strategy[strategy]])

        if strategy == 'fixed':
            baseline_return = avg_return
            improvement_pct = 0
        else:
            improvement_pct = ((avg_return - baseline_return) / abs(baseline_return)) * 100 if baseline_return else 0
            if improvement_pct > best_improvement:
                best_improvement = improvement_pct
                best_strategy = strategy

        print(f"{strategy:<15} {avg_return:>13.2f}% {avg_position:>13.2f}x {improvement_pct:>13.1f}%")

    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print()

    if best_improvement >= 10.0:
        print(f"[SUCCESS] {best_strategy.upper()} strategy shows {best_improvement:.1f}% improvement!")
        print(f"Baseline (fixed): {baseline_return:.2f}% avg return")
        best_return = np.mean([r['total_return'] for r in results_by_strategy[best_strategy]])
        print(f"Optimized ({best_strategy}): {best_return:.2f}% avg return")
        print()
        print("DEPLOY: Implement dynamic position sizing with best strategy")
    elif best_improvement > 0:
        print(f"[MARGINAL] Best strategy ({best_strategy}) shows {best_improvement:.1f}% improvement")
        print(f"Below +10% threshold for deployment")
    else:
        print(f"[NO BENEFIT] Dynamic position sizing shows no improvement")
        print(f"Keep fixed 1.0x position sizing")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-045',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'stocks_tested': test_stocks,
        'strategies': strategies,
        'results_by_strategy': {
            strategy: [{**r} for r in results]
            for strategy, results in results_by_strategy.items()
        },
        'baseline_return': float(baseline_return) if baseline_return else 0,
        'best_strategy': best_strategy if best_strategy else 'none',
        'best_improvement_pct': float(best_improvement),
        'deploy': best_improvement >= 10.0
    }

    results_file = os.path.join(results_dir, 'exp045_position_sizing_optimization.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending position sizing report email...")
            notifier.send_experiment_report('EXP-045', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-045 position sizing optimization."""

    print("\n[POSITION SIZING] Testing dynamic position sizing strategies")
    print("This will take 5-10 minutes...")
    print()

    results = run_exp045_position_sizing_optimization()

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    if results.get('deploy', False):
        print("1. Implement position sizing in production scanner")
        print("2. Add signal strength calculation to signal generation")
        print("3. Update to v14.0-DYNAMIC-SIZING")
        print("4. Monitor performance in live trading")
    else:
        print("1. Keep fixed 1.0x position sizing")
        print("2. Consider other enhancement options")
    print()
    print("POSITION SIZING OPTIMIZATION COMPLETE")

"""
EXPERIMENT: EXP-076
Date: 2025-11-17
Objective: Implement volatility-adjusted position sizing for better risk-adjusted returns

HYPOTHESIS:
Current strategy uses fixed 100% position size for all trades.
Volatility-adjusted sizing (based on ATR) will improve risk-adjusted returns:
- Low volatility stocks (ATR < 1.5%): Size up to 1.2-1.5x
- Medium volatility (ATR 1.5-3%): Standard 1.0x
- High volatility (ATR > 3%): Size down to 0.5-0.7x

This should improve Sharpe ratio by 20-30% and reduce max drawdown.

ALGORITHM:
1. Calculate ATR (Average True Range) for each stock
2. Define volatility tiers and position size multipliers
3. Backtest with dynamic position sizing (2020-2025)
4. Compare to fixed 1.0x baseline
5. Measure impact on Sharpe ratio, max drawdown, total return

EXPECTED RESULTS:
- Sharpe ratio improvement: +20-30%
- Max drawdown reduction: -15-25%
- Total return: Similar or slightly better (+5-10%)
- Risk-adjusted returns: Significantly better

SUCCESS CRITERIA:
- Sharpe ratio improvement >= +15%
- Max drawdown reduction >= -10%
- No reduction in total return
- Win rate maintained or improved
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from src.config.mean_reversion_params import get_all_tickers


def calculate_atr_pct(data: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate ATR as percentage of price.

    Args:
        data: OHLCV DataFrame
        period: ATR period

    Returns:
        ATR as percentage of current price
    """
    if 'atr_14' in data.columns:
        atr = data['atr_14'].iloc[-1]
        price = data['Close'].iloc[-1]
        return (atr / price) * 100
    else:
        # Calculate manually if not available
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())

        true_range = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        price = data['Close'].iloc[-1]

        return (atr / price) * 100


def get_volatility_tier(atr_pct: float) -> Tuple[str, float]:
    """
    Get volatility tier and position size multiplier.

    Args:
        atr_pct: ATR as percentage

    Returns:
        Tuple of (tier_name, position_size_multiplier)
    """
    if atr_pct < 1.5:
        return ('LOW', 1.3)  # Low volatility: size up
    elif atr_pct < 2.5:
        return ('MEDIUM', 1.0)  # Medium volatility: standard size
    elif atr_pct < 3.5:
        return ('HIGH', 0.7)  # High volatility: size down
    else:
        return ('VERY_HIGH', 0.5)  # Very high volatility: size down significantly


def test_volatility_position_sizing_stock(ticker: str,
                                         start_date: str = '2020-01-01',
                                         end_date: str = '2025-11-17') -> Dict:
    """
    Test volatility-adjusted position sizing for a single stock.

    Args:
        ticker: Stock ticker
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        Test results dict
    """
    try:
        # Fetch data
        buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
        fetcher = YahooFinanceFetcher()
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)

        if len(data) < 60:
            return None

        # Engineer features
        engineer = TechnicalFeatureEngineer(fillna=True)
        enriched_data = engineer.engineer_features(data)

        # Use baseline parameters for fair comparison
        detector = MeanReversionDetector(
            z_score_threshold=1.5,
            rsi_oversold=35,
            volume_multiplier=1.3,
            price_drop_threshold=-1.5
        )

        signals = detector.detect_overcorrections(enriched_data)

        if len(signals) == 0:
            return None

        signals = detector.calculate_reversion_targets(signals)

        # Apply filters
        regime_detector = MarketRegimeDetector()
        signals = add_regime_filter_to_signals(signals, regime_detector)

        earnings_fetcher = EarningsCalendarFetcher(exclusion_days_before=3, exclusion_days_after=3)
        signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

        if len(signals) == 0:
            return None

        # Calculate current ATR for volatility tier
        atr_pct = calculate_atr_pct(enriched_data)
        vol_tier, position_multiplier = get_volatility_tier(atr_pct)

        # Baseline: Fixed 1.0x position size
        backtester_baseline = MeanReversionBacktester(
            initial_capital=10000,
            exit_strategy='trailing_stop',
            max_hold_days=3,
            position_size_multiplier=1.0
        )
        baseline_results = backtester_baseline.backtest(signals)

        # Volatility-adjusted: Dynamic position sizing
        # Add position_size column to signals
        signals['position_size'] = position_multiplier

        backtester_vol_adjusted = MeanReversionBacktester(
            initial_capital=10000,
            exit_strategy='trailing_stop',
            max_hold_days=3,
            position_size_multiplier=1.0  # Will be overridden by per-signal position_size
        )
        vol_adjusted_results = backtester_vol_adjusted.backtest(signals)

        # Calculate improvements
        sharpe_improvement = (
            (vol_adjusted_results.get('sharpe_ratio', 0) - baseline_results.get('sharpe_ratio', 0))
            / baseline_results.get('sharpe_ratio', 1)
        ) * 100 if baseline_results.get('sharpe_ratio', 0) > 0 else 0

        return_improvement = (
            vol_adjusted_results.get('total_return', 0) - baseline_results.get('total_return', 0)
        )

        return {
            'ticker': ticker,
            'atr_pct': atr_pct,
            'volatility_tier': vol_tier,
            'position_multiplier': position_multiplier,
            'total_trades': vol_adjusted_results.get('total_trades', 0),
            'baseline': {
                'total_return': baseline_results.get('total_return', 0),
                'sharpe_ratio': baseline_results.get('sharpe_ratio', 0),
                'max_drawdown': baseline_results.get('max_drawdown', 0),
                'win_rate': baseline_results.get('win_rate', 0)
            },
            'vol_adjusted': {
                'total_return': vol_adjusted_results.get('total_return', 0),
                'sharpe_ratio': vol_adjusted_results.get('sharpe_ratio', 0),
                'max_drawdown': vol_adjusted_results.get('max_drawdown', 0),
                'win_rate': vol_adjusted_results.get('win_rate', 0)
            },
            'improvement': {
                'sharpe_pct': sharpe_improvement,
                'return_delta': return_improvement,
                'drawdown_delta': vol_adjusted_results.get('max_drawdown', 0) - baseline_results.get('max_drawdown', 0)
            }
        }

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_exp076_volatility_position_sizing():
    """
    Test volatility-adjusted position sizing across portfolio.
    """
    print("="*70)
    print("EXP-076: VOLATILITY-ADJUSTED POSITION SIZING")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("OBJECTIVE: Improve risk-adjusted returns via volatility-based sizing")
    print("METHODOLOGY: ATR-based position sizing (Low: 1.3x, Med: 1.0x, High: 0.5-0.7x)")
    print("TARGET: +15% Sharpe improvement, -10% max drawdown")
    print()

    # Get all portfolio tickers
    tickers = get_all_tickers()
    print(f"Testing {len(tickers)} stocks...")
    print()

    test_results = []

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Testing {ticker}...", end=" ")

        result = test_volatility_position_sizing_stock(ticker)

        if result and result.get('total_trades', 0) > 0:
            test_results.append(result)
            print(f"ATR={result['atr_pct']:.2f}%, Tier={result['volatility_tier']}, "
                  f"Size={result['position_multiplier']:.1f}x, "
                  f"Sharpe Δ={result['improvement']['sharpe_pct']:+.1f}%")
        else:
            print("No trades")

    # Summary
    print("\n" + "="*70)
    print("VOLATILITY POSITION SIZING RESULTS")
    print("="*70)
    print()

    if not test_results:
        print("[ERROR] No test results")
        return None

    # Calculate aggregate statistics
    avg_sharpe_improvement = np.mean([r['improvement']['sharpe_pct'] for r in test_results])
    avg_return_delta = np.mean([r['improvement']['return_delta'] for r in test_results])
    avg_drawdown_delta = np.mean([r['improvement']['drawdown_delta'] for r in test_results])

    baseline_sharpe = np.mean([r['baseline']['sharpe_ratio'] for r in test_results])
    vol_adj_sharpe = np.mean([r['vol_adjusted']['sharpe_ratio'] for r in test_results])

    baseline_return = np.mean([r['baseline']['total_return'] for r in test_results])
    vol_adj_return = np.mean([r['vol_adjusted']['total_return'] for r in test_results])

    improved_count = len([r for r in test_results if r['improvement']['sharpe_pct'] > 0])

    print(f"Stocks Tested: {len(test_results)}/54")
    print(f"Stocks Improved: {improved_count}/{len(test_results)} ({improved_count/len(test_results)*100:.1f}%)")
    print()

    # Volatility tier distribution
    tier_counts = {}
    for r in test_results:
        tier = r['volatility_tier']
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    print("VOLATILITY TIER DISTRIBUTION:")
    for tier in ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']:
        count = tier_counts.get(tier, 0)
        if count > 0:
            print(f"  {tier}: {count} stocks ({count/len(test_results)*100:.1f}%)")
    print()

    print(f"{'Metric':<25} {'Baseline (1.0x)':<20} {'Vol-Adjusted':<20} {'Improvement'}")
    print("-"*70)
    print(f"{'Sharpe Ratio':<25} {baseline_sharpe:<19.2f} {vol_adj_sharpe:<19.2f} {avg_sharpe_improvement:+.1f}%")
    print(f"{'Total Return':<25} {baseline_return:<19.1f}% {vol_adj_return:<19.1f}% {avg_return_delta:+.1f}%")
    print(f"{'Max Drawdown':<25} {np.mean([r['baseline']['max_drawdown'] for r in test_results]):<19.1f}% "
          f"{np.mean([r['vol_adjusted']['max_drawdown'] for r in test_results]):<19.1f}% {avg_drawdown_delta:+.1f}%")
    print()

    # Top improvers
    print("TOP 10 STOCKS BY SHARPE IMPROVEMENT:")
    print("-"*70)
    print(f"{'Ticker':<8} {'Vol Tier':<12} {'Size':<8} {'Baseline':<12} {'Vol-Adj':<12} {'Improvement'}")
    print("-"*70)

    sorted_results = sorted(test_results,
                          key=lambda x: x['improvement']['sharpe_pct'],
                          reverse=True)

    for result in sorted_results[:10]:
        print(f"{result['ticker']:<8} "
              f"{result['volatility_tier']:<12} "
              f"{result['position_multiplier']:<7.1f}x "
              f"{result['baseline']['sharpe_ratio']:<11.2f} "
              f"{result['vol_adjusted']['sharpe_ratio']:<11.2f} "
              f"{result['improvement']['sharpe_pct']:+.1f}%")

    # Deployment recommendation
    print()
    print("="*70)
    print("DEPLOYMENT RECOMMENDATION")
    print("="*70)
    print()

    if avg_sharpe_improvement >= 15.0 and avg_drawdown_delta <= -10.0 and avg_return_delta >= 0:
        print(f"[SUCCESS] Volatility-adjusted sizing VALIDATED for deployment!")
        print()
        print(f"RESULTS: {avg_sharpe_improvement:+.1f}% Sharpe improvement, "
              f"{avg_drawdown_delta:+.1f}% drawdown reduction")
        print()
        print("NEXT STEPS:")
        print("  1. Integrate ATR-based sizing into ml_signal_scanner.py")
        print("  2. Add position_size column to signals")
        print("  3. Deploy to production")
        print()
        print(f"ESTIMATED IMPACT: {avg_sharpe_improvement:+.1f}% better risk-adjusted returns")
    elif avg_sharpe_improvement >= 10.0:
        print(f"[PARTIAL SUCCESS] {avg_sharpe_improvement:+.1f}% Sharpe improvement (target: +15%)")
        print()
        print("Volatility sizing shows promise:")
        print(f"  - {improved_count}/{len(test_results)} stocks improved")
        print(f"  - Sharpe: {baseline_sharpe:.2f} → {vol_adj_sharpe:.2f}")
        print()
        print("Consider deploying with refinements")
    else:
        print(f"[NEEDS WORK] {avg_sharpe_improvement:+.1f}% Sharpe improvement below threshold")
        print()
        print("May need different tier thresholds or multipliers")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-076',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'objective': 'Test volatility-adjusted position sizing',
        'algorithm': 'ATR-based dynamic position sizing',
        'stocks_tested': len(test_results),
        'stocks_improved': improved_count,
        'avg_sharpe_improvement_pct': float(avg_sharpe_improvement),
        'avg_return_delta': float(avg_return_delta),
        'avg_drawdown_delta': float(avg_drawdown_delta),
        'baseline_sharpe': float(baseline_sharpe),
        'vol_adjusted_sharpe': float(vol_adj_sharpe),
        'volatility_tiers': tier_counts,
        'stock_results': test_results,
        'validated': avg_sharpe_improvement >= 15.0 and avg_drawdown_delta <= -10.0,
        'next_step': 'Deploy to production' if avg_sharpe_improvement >= 15.0 else 'Refine parameters',
        'methodology': {
            'data_period': '2020-2025',
            'low_volatility': {'atr_threshold': '<1.5%', 'position_size': 1.3},
            'medium_volatility': {'atr_threshold': '1.5-2.5%', 'position_size': 1.0},
            'high_volatility': {'atr_threshold': '2.5-3.5%', 'position_size': 0.7},
            'very_high_volatility': {'atr_threshold': '>3.5%', 'position_size': 0.5}
        }
    }

    results_file = os.path.join(results_dir, 'exp076_volatility_position_sizing.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending volatility position sizing report...")
            notifier.send_experiment_report('EXP-076', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-076: Volatility-adjusted position sizing."""

    print("\n[VOLATILITY POSITION SIZING] Testing ATR-based dynamic sizing")
    print("Size up on stable stocks, size down on volatile stocks")
    print()

    results = run_exp076_volatility_position_sizing()

    print("\n" + "="*70)
    print("VOLATILITY POSITION SIZING TEST COMPLETE")
    print("="*70)

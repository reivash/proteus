"""
EXPERIMENT: EXP-047
Date: 2025-11-17
Objective: Comprehensive validation of v14.0-DYNAMIC-SIZING system

MOTIVATION:
Multiple major enhancements deployed in this session:
- EXP-043: 7 new stocks optimized (+13.2pp avg)
- EXP-044: 4 final stocks optimized (+5.8pp avg)
- EXP-045: Dynamic position sizing (+41.7% return improvement!)
- v14.0: All enhancements deployed to production

Need comprehensive validation to:
- Confirm cumulative impact of all improvements
- Validate dynamic position sizing in real portfolio context
- Measure actual vs expected performance
- Provide confidence for live trading

HYPOTHESIS:
v14.0 system will show massive improvement vs baseline:
- Baseline (v12.0): ~63.6% win rate, ~13-14% avg return per stock
- Expected v14.0: ~80%+ win rate, ~20%+ avg return per stock (with dynamic sizing)
- Cumulative improvement: +50%+ total return improvement

METHODOLOGY:
1. Run full portfolio backtest (all 31 Tier A stocks)
2. Test period: 2022-01-01 to 2025-11-15 (3+ years)
3. Use optimized parameters for each stock
4. Apply dynamic position sizing (v14.0)
5. Apply all validated filters (earnings, regime)
6. Measure: win rate, total return, Sharpe ratio, max drawdown

EXPECTED OUTCOME:
- Win rate: ~80%+ (vs 63.6% baseline)
- Avg return per stock: ~20%+ (vs 13% baseline)
- Total portfolio return: Significant improvement
- Sharpe ratio: >5.0
- Validation of v14.0 readiness for live trading
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from src.config.mean_reversion_params import get_params, get_all_tickers


def calculate_signal_strength(signal_row: pd.Series, params: dict) -> float:
    """Calculate signal strength score (0-100) - same as production scanner."""
    z_score = abs(signal_row.get('z_score', 0))
    rsi = signal_row.get('rsi', 35)

    volume = signal_row.get('Volume', 0)
    volume_avg = signal_row.get('volume_20d_avg', volume)
    volume_ratio = volume / volume_avg if volume_avg > 0 else 1.5

    price_drop = signal_row.get('daily_return', -2.0)

    z_component = min(100, (z_score / 2.5) * 100) * 0.40
    rsi_threshold = params.get('rsi_oversold', 35)
    rsi_component = min(100, max(0, ((rsi_threshold - rsi) / 15) * 100)) * 0.25
    vol_threshold = params.get('volume_multiplier', 1.3)
    volume_component = min(100, max(0, ((volume_ratio - vol_threshold) / 2.0) * 100)) * 0.20
    price_component = min(100, max(0, (abs(price_drop) / 5.0) * 100)) * 0.15

    total_score = z_component + rsi_component + volume_component + price_component
    return max(0, min(100, total_score))


def calculate_position_size(signal_strength: float) -> float:
    """Calculate position size - LINEAR strategy from EXP-045."""
    position_size = 0.5 + (signal_strength / 100.0) * 1.5
    return max(0.5, min(2.0, position_size))


def apply_dynamic_position_sizing(signals: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply dynamic position sizing to signals."""
    signals = signals.copy()
    signals['signal_strength'] = signals.apply(
        lambda row: calculate_signal_strength(row, params), axis=1
    )
    signals['position_size'] = signals['signal_strength'].apply(calculate_position_size)
    return signals


def validate_stock(ticker: str, start_date: str, end_date: str, use_dynamic_sizing: bool = True) -> dict:
    """
    Validate a single stock with v14.0 configuration.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date
        use_dynamic_sizing: Whether to use dynamic position sizing

    Returns:
        Validation results
    """
    print(f"  Validating {ticker}...")

    # Fetch data
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
    except Exception as e:
        print(f"    [ERROR] Failed to fetch: {e}")
        return None

    if len(data) < 60:
        print(f"    [SKIP] Insufficient data")
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

    # Apply dynamic position sizing (v14.0)
    if use_dynamic_sizing:
        signals = apply_dynamic_position_sizing(signals, params)

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

        if not results or results.get('total_trades', 0) < 1:
            print(f"    [SKIP] No trades")
            return None

        win_rate = results.get('win_rate', 0)
        total_return = results.get('total_return', 0)
        total_trades = results.get('total_trades', 0)
        sharpe = results.get('sharpe_ratio', 0)

        # Get position sizing stats if applicable
        panic_signals = signals[signals['panic_sell'] == 1].copy()
        avg_position_size = panic_signals['position_size'].mean() if use_dynamic_sizing and len(panic_signals) > 0 else 1.0

        print(f"    Win: {win_rate:.1f}%, Return: {total_return:+.1f}%, Trades: {total_trades}, Sharpe: {sharpe:.2f}")

        return {
            'ticker': ticker,
            'win_rate': win_rate,
            'total_return': total_return,
            'total_trades': total_trades,
            'sharpe_ratio': sharpe,
            'avg_gain': results.get('avg_gain', 0),
            'avg_position_size': avg_position_size,
            'tier': params.get('tier', 'A')
        }

    except Exception as e:
        print(f"    [ERROR] Backtest failed: {e}")
        return None


def run_exp047_v14_comprehensive_validation():
    """
    Comprehensive validation of v14.0-DYNAMIC-SIZING system.
    """
    print("=" * 70)
    print("EXP-047: v14.0 COMPREHENSIVE SYSTEM VALIDATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Validate complete v14.0-DYNAMIC-SIZING system")
    print("Configuration:")
    print("  - Optimized parameters: 28/31 stocks (90.3%)")
    print("  - Dynamic position sizing: LINEAR (EXP-045: +41.7%)")
    print("  - Earnings filter: ±3 days")
    print("  - Regime filter: BEAR market protection")
    print("  - Exit strategy: Time-decay (±2%, ±1.5%, ±1%, max 3d)")
    print()

    # Get all tickers
    all_tickers = get_all_tickers()
    print(f"Validating {len(all_tickers)} Tier A stocks")
    print("This will take 10-15 minutes...")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    print(f"Test period: {start_date} to {end_date} (3+ years)")
    print()
    print("=" * 70)

    # Validate all stocks
    validation_results = []

    for ticker in all_tickers:
        result = validate_stock(ticker, start_date, end_date, use_dynamic_sizing=True)
        if result:
            validation_results.append(result)

    # Calculate aggregate statistics
    print()
    print("=" * 70)
    print("VALIDATION RESULTS - v14.0-DYNAMIC-SIZING")
    print("=" * 70)
    print()

    if validation_results:
        total_stocks = len(validation_results)
        avg_win_rate = np.mean([r['win_rate'] for r in validation_results])
        avg_return = np.mean([r['total_return'] for r in validation_results])
        total_trades = sum([r['total_trades'] for r in validation_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in validation_results])
        avg_position_size = np.mean([r['avg_position_size'] for r in validation_results])

        # Top performers
        top_performers = sorted(validation_results, key=lambda x: x['win_rate'], reverse=True)[:10]

        print(f"Portfolio Statistics:")
        print(f"  Stocks validated: {total_stocks}")
        print(f"  Average win rate: {avg_win_rate:.1f}%")
        print(f"  Average return per stock: {avg_return:+.1f}%")
        print(f"  Total trades (all stocks): {total_trades}")
        print(f"  Average Sharpe ratio: {avg_sharpe:.2f}")
        print(f"  Average position size: {avg_position_size:.2f}x")
        print()

        print("TOP 10 PERFORMERS (by win rate):")
        print(f"{'Ticker':<8} {'Win Rate':<12} {'Return':<12} {'Sharpe':<10} {'Trades':<10}")
        print("-" * 70)
        for r in top_performers:
            print(f"{r['ticker']:<8} {r['win_rate']:>10.1f}% {r['total_return']:>10.1f}% "
                  f"{r['sharpe_ratio']:>8.2f} {r['total_trades']:>8}")

        print()
        print("=" * 70)
        print("SYSTEM VALIDATION")
        print("=" * 70)
        print()

        # Compare to baseline expectations
        baseline_wr = 63.6
        baseline_return = 13.0

        wr_improvement = avg_win_rate - baseline_wr
        return_improvement_pct = ((avg_return - baseline_return) / abs(baseline_return)) * 100

        print(f"Baseline (v12.0):")
        print(f"  Win rate: {baseline_wr}%")
        print(f"  Avg return: {baseline_return}%")
        print()

        print(f"Current (v14.0-DYNAMIC-SIZING):")
        print(f"  Win rate: {avg_win_rate:.1f}% ({wr_improvement:+.1f}pp)")
        print(f"  Avg return: {avg_return:.1f}% ({return_improvement_pct:+.1f}%)")
        print()

        if avg_win_rate >= 75.0 and avg_return >= 18.0:
            print("[SUCCESS] v14.0 system meets performance targets!")
            print(f"✓ Win rate: {avg_win_rate:.1f}% (target: 75%+)")
            print(f"✓ Avg return: {avg_return:.1f}% (target: 18%+)")
            print()
            print("READY FOR LIVE TRADING")
        else:
            print("[REVIEW] Performance below targets")
            print(f"Win rate: {avg_win_rate:.1f}% (target: 75%+)")
            print(f"Avg return: {avg_return:.1f}% (target: 18%+)")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-047',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'version': 'v14.0-DYNAMIC-SIZING',
        'stocks_validated': len(validation_results),
        'total_trades': total_trades if validation_results else 0,
        'avg_win_rate': float(avg_win_rate) if validation_results else 0,
        'avg_return': float(avg_return) if validation_results else 0,
        'avg_sharpe': float(avg_sharpe) if validation_results else 0,
        'avg_position_size': float(avg_position_size) if validation_results else 1.0,
        'stock_results': validation_results,
        'ready_for_live': (avg_win_rate >= 75.0 and avg_return >= 18.0) if validation_results else False
    }

    results_file = os.path.join(results_dir, 'exp047_v14_comprehensive_validation.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending validation report email...")
            notifier.send_experiment_report('EXP-047', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-047 comprehensive validation."""

    print("\n[SYSTEM VALIDATION] Comprehensive v14.0 validation")
    print("This will take 10-15 minutes...")
    print()

    results = run_exp047_v14_comprehensive_validation()

    print("\n" + "=" * 70)
    print("v14.0 COMPREHENSIVE VALIDATION COMPLETE")
    print("=" * 70)

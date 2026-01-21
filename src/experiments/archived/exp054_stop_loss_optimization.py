"""
EXPERIMENT: EXP-054
Date: 2025-11-17
Objective: Optimize stop-loss levels per stock based on volatility

MOTIVATION:
Current system: -2% stop-loss for ALL stocks (one-size-fits-all)
- Low volatility stocks (V, MA, JPM): -2% might be too wide, accepting unnecessary losses
- High volatility stocks (NVDA, AVGO): -2% might be too tight, causing false stops
- No adaptation to individual stock characteristics
- Missing 5-10% Sharpe improvement from optimized stops

PROBLEM:
One-size-fits-all stop-loss != optimal risk management:
- Stable stocks like V rarely move 2% on noise, could use -1.5% stop
- Volatile stocks like NVDA move 2%+ frequently, need -2.5% or -3% to avoid false stops
- Current -2% is suboptimal for most stocks
- Better stops -> fewer false exits, better risk management

HYPOTHESIS:
Per-stock stop-loss based on volatility will improve performance:
- Low volatility (ATR < 2%): Tighter stops at -1.5%
- Medium volatility (ATR 2-3%): Standard stops at -2.0%
- High volatility (ATR > 3%): Wider stops at -2.5% or -3.0%
- Expected: +5-10% Sharpe ratio, maintained win rate, better risk-adjusted returns

METHODOLOGY:
1. Calculate stock volatility (ATR - Average True Range):
   - Use 20-day ATR as volatility measure
   - Classify as low/medium/high volatility

2. Test stop-loss levels per stock:
   - Low-vol: [-1.0%, -1.5%, -2.0%]
   - Med-vol: [-1.5%, -2.0%, -2.5%]
   - High-vol: [-2.0%, -2.5%, -3.0%, -3.5%]

3. Measure impact on:
   - Win rate
   - Total return
   - Sharpe ratio
   - Max drawdown
   - Average loss per losing trade

4. Deploy if improvement >= +5% Sharpe ratio

CONSTRAINTS:
- Stop-loss <= -1.0% (avoid noise stops)
- Stop-loss >= -4.0% (limit max loss)
- Maintain >= 70% win rate

EXPECTED OUTCOME:
- Per-stock stop-loss parameters
- Volatility-based stop framework
- +5-10% Sharpe improvement
- Better risk management
- Fewer false stops on volatile stocks
- Smaller losses on stable stocks
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
from src.config.mean_reversion_params import get_params


def calculate_stock_volatility(data: pd.DataFrame) -> float:
    """
    Calculate average volatility (ATR as % of price).

    Returns:
        Average volatility as percentage
    """
    # Calculate True Range
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift(1))
    low_close = abs(data['Low'] - data['Close'].shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # ATR as percentage of close price
    atr_pct = (true_range / data['Close']) * 100

    return atr_pct.rolling(20).mean().mean()


def optimize_stop_loss(ticker: str, start_date: str, end_date: str) -> dict:
    """
    Find optimal stop-loss for a stock.

    Returns:
        Optimization results with best stop-loss
    """
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
    except:
        return None

    if len(data) < 60:
        return None

    # Calculate volatility
    volatility = calculate_stock_volatility(data)

    # Determine stop-loss candidates based on volatility
    if volatility < 2.0:
        stop_candidates = [-1.0, -1.5, -2.0]
        vol_class = 'LOW'
    elif volatility < 3.0:
        stop_candidates = [-1.5, -2.0, -2.5]
        vol_class = 'MEDIUM'
    else:
        stop_candidates = [-2.0, -2.5, -3.0, -3.5]
        vol_class = 'HIGH'

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

    # Test each stop-loss
    best_stop = -2.0
    best_sharpe = 0
    best_win_rate = 0
    best_return = 0

    results_by_stop = {}

    for stop_loss in stop_candidates:
        backtester = MeanReversionBacktester(
            initial_capital=10000,
            exit_strategy='time_decay',
            profit_target=2.0,
            stop_loss=stop_loss,
            max_hold_days=3
        )

        try:
            result = backtester.backtest(signals)

            if not result or result.get('total_trades', 0) < 5:
                continue

            win_rate = result['win_rate']
            total_return = result['total_return']
            sharpe = result.get('sharpe_ratio', 0)

            results_by_stop[stop_loss] = {
                'win_rate': win_rate,
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'total_trades': result['total_trades']
            }

            # Best = highest Sharpe ratio with >= 70% win rate
            if win_rate >= 70.0 and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_stop = stop_loss
                best_win_rate = win_rate
                best_return = total_return

        except Exception as e:
            continue

    if not results_by_stop:
        return None

    return {
        'ticker': ticker,
        'volatility': volatility,
        'volatility_class': vol_class,
        'current_stop': -2.0,
        'optimal_stop': best_stop,
        'improvement': best_sharpe - results_by_stop.get(-2.0, {}).get('sharpe_ratio', 0),
        'best_win_rate': best_win_rate,
        'best_return': best_return,
        'best_sharpe': best_sharpe,
        'results_by_stop': results_by_stop
    }


def run_exp054_stop_loss_optimization():
    """
    Optimize stop-loss levels per stock.
    """
    print("="*70)
    print("EXP-054: STOP-LOSS OPTIMIZATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Optimize stop-loss per stock based on volatility")
    print("Current: -2% stop-loss for ALL stocks (one-size-fits-all)")
    print("Expected: +5-10% Sharpe improvement from optimized stops")
    print()

    # Test on representative stocks
    test_stocks = [
        'NVDA',  # High volatility
        'V', 'MA', 'JPM',  # Low volatility
        'MSFT', 'ORCL', 'ABBV',  # Medium volatility
        'AVGO', 'GILD', 'WMT'
    ]
    print(f"Testing {len(test_stocks)} stocks")
    print("This will take 5-10 minutes...")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    results = []

    for ticker in test_stocks:
        print(f"\nOptimizing {ticker}...", end=" ")

        result = optimize_stop_loss(ticker, start_date, end_date)

        if result:
            results.append(result)
            print(f"[{result['volatility_class']}] "
                  f"Volatility: {result['volatility']:.2f}%, "
                  f"Optimal: {result['optimal_stop']:.1f}%, "
                  f"Improvement: +{result['improvement']:.2f} Sharpe")
        else:
            print("[SKIP]")

    # Analyze results
    print()
    print("="*70)
    print("STOP-LOSS OPTIMIZATION RESULTS")
    print("="*70)
    print()

    print(f"{'Ticker':<8} {'Vol%':<8} {'Class':<10} {'Current':<10} {'Optimal':<10} {'Improvement'}")
    print("-"*70)

    total_improvement = 0
    stocks_improved = 0

    for r in results:
        improvement = r['improvement']
        if improvement > 0:
            stocks_improved += 1
            total_improvement += improvement

        print(f"{r['ticker']:<8} {r['volatility']:>6.2f}% {r['volatility_class']:<10} "
              f"{r['current_stop']:>8.1f}% {r['optimal_stop']:>8.1f}% {improvement:>+13.2f}")

    avg_improvement = total_improvement / len(results) if results else 0

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    print(f"Stocks tested: {len(results)}")
    print(f"Stocks with improvement: {stocks_improved}")
    print(f"Average Sharpe improvement: +{avg_improvement:.2f}")
    print()

    # Volatility-based recommendations
    low_vol_stops = [r['optimal_stop'] for r in results if r['volatility_class'] == 'LOW']
    med_vol_stops = [r['optimal_stop'] for r in results if r['volatility_class'] == 'MEDIUM']
    high_vol_stops = [r['optimal_stop'] for r in results if r['volatility_class'] == 'HIGH']

    print("VOLATILITY-BASED STOP RECOMMENDATIONS:")
    if low_vol_stops:
        print(f"  Low volatility (<2%): {np.mean(low_vol_stops):.1f}% avg optimal")
    if med_vol_stops:
        print(f"  Medium volatility (2-3%): {np.mean(med_vol_stops):.1f}% avg optimal")
    if high_vol_stops:
        print(f"  High volatility (>3%): {np.mean(high_vol_stops):.1f}% avg optimal")

    print()
    print("="*70)
    print("RECOMMENDATION")
    print("="*70)
    print()

    if avg_improvement >= 0.5:
        print(f"[SUCCESS] Stop-loss optimization shows +{avg_improvement:.2f} avg Sharpe improvement!")
        print(f"{stocks_improved}/{len(results)} stocks improved with optimized stops")
        print()
        print("DEPLOY: Implement per-stock stop-loss based on volatility")
        print()
        print("OPTIMIZED STOPS BY STOCK:")
        for r in results:
            if r['optimal_stop'] != -2.0:
                print(f"  {r['ticker']}: {r['optimal_stop']:.1f}% (was -2.0%)")
    else:
        print(f"[MARGINAL] Stop-loss optimization shows only +{avg_improvement:.2f} improvement")
        print(f"Below +0.5 threshold for deployment")
        print("Current -2.0% stop-loss is adequate across all stocks")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-054',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'stocks_tested': test_stocks,
        'results': [{**r} for r in results],
        'avg_improvement': float(avg_improvement),
        'stocks_improved': stocks_improved,
        'deploy': avg_improvement >= 0.5
    }

    results_file = os.path.join(results_dir, 'exp054_stop_loss_optimization.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending stop-loss optimization report email...")
            notifier.send_experiment_report('EXP-054', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-054 stop-loss optimization."""

    print("\n[STOP-LOSS] Optimizing per-stock stop-loss levels")
    print("This will take 5-10 minutes...")
    print()

    results = run_exp054_stop_loss_optimization()

    print("\n" + "="*70)
    print("STOP-LOSS OPTIMIZATION COMPLETE")
    print("="*70)

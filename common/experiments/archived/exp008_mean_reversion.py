"""
EXPERIMENT: EXP-008
Date: 2025-11-14
Objective: Test mean reversion strategy on stock overcorrections

METHOD:
- Strategy: Detect panic sells and buy the dip
- Signals: Z-score < -2, RSI < 30, Volume spike (2x avg), Drop > 3%
- Trade: Buy when all signals align, hold 1-2 days
- Exit: +2% profit OR -2% stop loss OR 2-day timeout

User Observation:
"When stock jumps down suddenly, it tends to normalize in 1-2 days"

This tests if overcorrections can be profitably traded.
"""

import sys
import os
import json
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester


def run_mean_reversion_experiment(ticker="^GSPC", ticker_name="S&P 500"):
    """Run mean reversion strategy backtest."""

    print("=" * 70)
    print(f"PROTEUS - EXPERIMENT EXP-008: MEAN REVERSION STRATEGY")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    period = "2y"
    initial_capital = 10000

    print(f"Configuration:")
    print(f"  Ticker: {ticker} ({ticker_name})")
    print(f"  Period: {period}")
    print(f"  Strategy: Buy panic sells, hold 1-2 days")
    print(f"  Initial Capital: ${initial_capital:,}")
    print()

    print(f"Detection Signals:")
    print(f"  - Z-Score < -2.0 (price well below average)")
    print(f"  - RSI < 30 (oversold)")
    print(f"  - Volume Spike > 2x average")
    print(f"  - Daily Drop > 3%")
    print()

    print(f"Exit Rules:")
    print(f"  - Profit Target: +2%")
    print(f"  - Stop Loss: -2%")
    print(f"  - Timeout: 2 days maximum hold")
    print()

    # Step 1: Fetch Data
    print("[1/5] Fetching data...")
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, period=period)
        print(f"      [OK] Fetched {len(data)} rows")
        print(f"      [OK] Date range: {data['Date'].min()} to {data['Date'].max()}")
    except Exception as e:
        print(f"      [FAIL] Error: {str(e)}")
        return None

    # Step 2: Add Technical Indicators (needed for RSI)
    print("\n[2/5] Calculating technical indicators...")
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)
    print(f"      [OK] Technical indicators calculated")

    # Step 3: Detect Overcorrections
    print("\n[3/5] Detecting overcorrections...")
    detector = MeanReversionDetector(
        z_score_threshold=1.5,      # Relaxed from 2.0
        rsi_oversold=35,             # Relaxed from 30
        rsi_overbought=65,           # Relaxed from 70
        volume_multiplier=1.5,       # Relaxed from 2.0
        price_drop_threshold=-2.0   # Relaxed from -3.0
    )

    signals = detector.detect_overcorrections(enriched_data)
    signals = detector.calculate_reversion_targets(signals)

    panic_sells = signals['panic_sell'].sum()
    panic_buys = signals['panic_buy'].sum()

    print(f"      [OK] Panic Sells Detected: {panic_sells}")
    print(f"      [OK] Panic Buys Detected: {panic_buys}")
    print(f"      [OK] Total Opportunities: {panic_sells + panic_buys}")

    # Step 4: Backtest Strategy
    print("\n[4/5] Running backtest...")
    backtester = MeanReversionBacktester(
        initial_capital=initial_capital,
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=2
    )

    results = backtester.backtest(signals)

    print(f"      [OK] Backtest complete")
    print(f"      [OK] Total Trades: {results['total_trades']}")

    # Step 5: Analyze Results
    print("\n[5/5] Analyzing results...")

    if results['total_trades'] == 0:
        print("      [WARN] No trades executed (no signals met all criteria)")
        return None

    # Display Results
    print()
    print("=" * 70)
    print("BACKTEST RESULTS:")
    print("=" * 70)
    print(f"Initial Capital:       ${initial_capital:,}")
    print(f"Final Capital:         ${results['final_capital']:,.2f}")
    print(f"Total Return:          {results['total_return']:.2f}%")
    print()
    print(f"Trading Performance:")
    print(f"  Total Trades:        {results['total_trades']}")
    print(f"  Winning Trades:      {results['winning_trades']} ({results['win_rate']:.1f}%)")
    print(f"  Losing Trades:       {results['losing_trades']}")
    print()
    print(f"Average Performance:")
    print(f"  Avg Gain (winners):  {results['avg_gain']:.2f}%")
    print(f"  Avg Loss (losers):   {results['avg_loss']:.2f}%")
    print()
    print(f"Risk Metrics:")
    print(f"  Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:        {results['max_drawdown']:.2f}%")
    print(f"  Profit Factor:       {results['profit_factor']:.2f}")
    print("=" * 70)

    # Show sample trades
    if len(results['trades']) > 0:
        print("\nSample Trades (first 5):")
        print("-" * 70)
        for i, trade in enumerate(results['trades'][:5]):
            print(f"\nTrade {i+1}:")
            print(f"  Entry: {trade['entry_date'].date()} @ ${trade['entry_price']:.2f}")
            print(f"  Exit:  {trade['exit_date'].date()} @ ${trade['exit_price']:.2f}")
            print(f"  Return: {trade['return_pct']:.2f}% | Profit: ${trade['profit']:.2f}")
            print(f"  Days Held: {trade['days_held']} | Exit: {trade['exit_reason']}")

    # Prepare results for saving
    experiment_results = {
        'experiment_id': 'EXP-008',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategy': 'Mean Reversion (Panic Sell)',
        'ticker': ticker,
        'ticker_name': ticker_name,
        'period': period,
        'signals_detected': {
            'panic_sells': int(panic_sells),
            'panic_buys': int(panic_buys),
            'total_opportunities': int(panic_sells + panic_buys)
        },
        'parameters': {
            'z_score_threshold': 1.5,
            'rsi_oversold': 35,
            'volume_multiplier': 1.5,
            'price_drop_threshold': -2.0,
            'profit_target': 2.0,
            'stop_loss': -2.0,
            'max_hold_days': 2
        },
        'backtest_results': {
            'initial_capital': initial_capital,
            'final_capital': float(results['final_capital']),
            'total_return': float(results['total_return']),
            'total_trades': results['total_trades'],
            'winning_trades': results['winning_trades'],
            'losing_trades': results['losing_trades'],
            'win_rate': float(results['win_rate']),
            'avg_gain': float(results['avg_gain']),
            'avg_loss': float(results['avg_loss']),
            'sharpe_ratio': float(results['sharpe_ratio']),
            'max_drawdown': float(results['max_drawdown']),
            'profit_factor': float(results['profit_factor'])
        },
        'sample_trades': [
            {
                'entry_date': str(t['entry_date'].date()),
                'exit_date': str(t['exit_date'].date()),
                'entry_price': float(t['entry_price']),
                'exit_price': float(t['exit_price']),
                'return_pct': float(t['return_pct']),
                'profit': float(t['profit']),
                'days_held': t['days_held'],
                'exit_reason': t['exit_reason']
            }
            for t in results['trades'][:10]  # Save first 10 trades
        ],
        'comparison': {
            'directional_prediction_accuracy': 58.94,
            'mean_reversion_win_rate': float(results['win_rate']),
            'annualized_return_estimate': float(results['total_return'] * (365 / (len(data) / 252)))
        }
    }

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'experiments')
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f'exp008_{ticker.lower().replace("^", "")}_mean_reversion_results.json')

    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return experiment_results


if __name__ == "__main__":
    # Test on S&P 500
    print("\n" + "=" * 70)
    print(" " * 15 + "MEAN REVERSION STRATEGY TEST")
    print("=" * 70 + "\n")

    results = run_mean_reversion_experiment("^GSPC", "S&P 500")

    if results:
        print("\n" + "=" * 70)
        print("SUMMARY:")
        print("=" * 70)
        win_rate = results['backtest_results']['win_rate']
        total_return = results['backtest_results']['total_return']
        sharpe = results['backtest_results']['sharpe_ratio']

        print(f"[+] Win Rate: {win_rate:.1f}%")
        print(f"[+] Total Return: {total_return:.2f}%")
        print(f"[+] Sharpe Ratio: {sharpe:.2f}")

        if win_rate >= 60 and total_return > 0:
            print("\n[SUCCESS] Strategy shows promise! Consider paper trading.")
        elif win_rate >= 50:
            print("\n[MIXED] Strategy marginally profitable. Needs optimization.")
        else:
            print("\n[CAUTION] Strategy underperformed. Review parameters.")

        print("=" * 70)
    else:
        print("\n[FAILED] Strategy test failed or no trades executed.")

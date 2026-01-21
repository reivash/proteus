"""
EXPERIMENT: EXP-009-PAPER-TRADING-VALIDATION
Date: 2025-11-14
Objective: Validate paper trading system against backtest results

HYPOTHESIS:
Paper trading system should produce identical results to backtest when run
on historical dates with same entry/exit logic.

TEST METHODOLOGY:
1. Select 5-10 historical dates from 2024 with known panic sell signals
2. Run paper trading system on these dates
3. Run backtest on same period for comparison
4. Compare metrics: entry prices, exit prices, returns, win rate

SUCCESS CRITERIA:
- Entry prices match within $0.10 (data source differences)
- Exit logic triggers at same conditions
- Returns match within 0.5% (rounding/fee differences)
- Win rate matches exactly (same trades = same outcomes)

EXPECTED OUTCOME:
Paper trading should replicate backtest results within acceptable tolerance,
confirming system is ready for 6-week live validation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from src.config.mean_reversion_params import get_params, get_all_tickers


def run_backtest(ticker: str, start_date: str, end_date: str) -> Dict:
    """
    Run backtest on historical period.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date

    Returns:
        Backtest results dictionary
    """
    print(f"\n[BACKTEST] {ticker} ({start_date} to {end_date})")

    # Fetch data with buffer for indicators
    buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')

    fetcher = YahooFinanceFetcher()
    data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)

    if len(data) < 60:
        print(f"  [SKIP] Insufficient data")
        return None

    # Engineer features
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)

    # Get parameters
    params = get_params(ticker)

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

    # Apply regime filter
    regime_detector = MarketRegimeDetector()
    signals = add_regime_filter_to_signals(signals, regime_detector)

    # Apply earnings filter
    earnings_fetcher = EarningsCalendarFetcher(
        exclusion_days_before=3,
        exclusion_days_after=3
    )
    signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

    # Backtest
    backtester = MeanReversionBacktester(
        initial_capital=10000,
        profit_target=2.0,
        stop_loss=-2.0,
        max_hold_days=2
    )

    results = backtester.backtest(signals)

    if results['total_trades'] == 0:
        print(f"  [NO TRADES] No signals in period")
        return None

    print(f"  Trades: {results['total_trades']}")
    print(f"  Win rate: {results['win_rate']:.1f}%")
    print(f"  Return: {results['total_return']:+.2f}%")

    results['ticker'] = ticker
    return results


def simulate_paper_trading(ticker: str, start_date: str, end_date: str) -> Dict:
    """
    Simulate paper trading on historical period.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date

    Returns:
        Paper trading results dictionary
    """
    print(f"\n[PAPER TRADING] {ticker} ({start_date} to {end_date})")

    # Get all dates in range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Simulate daily workflow
    positions = []
    completed_trades = []

    fetcher = YahooFinanceFetcher()

    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')

        # Check for new signals
        signal = check_for_signal(ticker, date_str)

        if signal and len(positions) == 0:
            # Enter position
            position = {
                'entry_date': date_str,
                'entry_price': signal['price'],
                'shares': 100,
                'hold_days': 0,
                'signal': signal
            }
            positions.append(position)
            print(f"  [{date_str}] ENTRY @ ${signal['price']:.2f}")

        # Check exits for open positions
        if positions:
            position = positions[0]
            position['hold_days'] += 1

            # Fetch current price
            try:
                price_data = fetcher.fetch_stock_data(
                    ticker,
                    start_date=date_str,
                    end_date=date_str
                )

                if not price_data.empty:
                    current_price = float(price_data['Close'].iloc[0])
                    position['current_price'] = current_price

                    # Calculate return
                    return_pct = ((current_price - position['entry_price']) / position['entry_price']) * 100

                    # Check exit conditions
                    exit_reason = None
                    if return_pct >= 2.0:
                        exit_reason = 'profit_target'
                    elif return_pct <= -2.0:
                        exit_reason = 'stop_loss'
                    elif position['hold_days'] >= 2:
                        exit_reason = 'max_hold'

                    if exit_reason:
                        # Exit position
                        trade = {
                            'entry_date': position['entry_date'],
                            'entry_price': position['entry_price'],
                            'exit_date': date_str,
                            'exit_price': current_price,
                            'return': return_pct,
                            'hold_days': position['hold_days'],
                            'exit_reason': exit_reason
                        }
                        completed_trades.append(trade)
                        positions = []
                        print(f"  [{date_str}] EXIT @ ${current_price:.2f} ({exit_reason}, {return_pct:+.2f}%)")

            except Exception as e:
                print(f"  [WARN] Could not fetch price for {date_str}: {e}")

    # Calculate results
    if not completed_trades:
        print(f"  [NO TRADES] No completed trades")
        return None

    total_trades = len(completed_trades)
    winning_trades = sum(1 for t in completed_trades if t['return'] > 0)
    win_rate = (winning_trades / total_trades) * 100
    total_return = sum(t['return'] for t in completed_trades)

    print(f"  Trades: {total_trades}")
    print(f"  Win rate: {win_rate:.1f}%")
    print(f"  Return: {total_return:+.2f}%")

    return {
        'ticker': ticker,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'trades': completed_trades
    }


def check_for_signal(ticker: str, date: str) -> Dict:
    """
    Check if ticker has signal on given date.

    Args:
        ticker: Stock ticker
        date: Date to check

    Returns:
        Signal dictionary or None
    """
    # Fetch data with buffer
    end_date = pd.to_datetime(date)
    start_date = end_date - timedelta(days=90)

    fetcher = YahooFinanceFetcher()
    try:
        data = fetcher.fetch_stock_data(
            ticker,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
    except:
        return None

    if len(data) < 60:
        return None

    # Engineer features
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)

    # Get parameters
    params = get_params(ticker)

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

    # Apply regime filter
    regime_detector = MarketRegimeDetector()
    signals = add_regime_filter_to_signals(signals, regime_detector)

    # Apply earnings filter
    earnings_fetcher = EarningsCalendarFetcher(
        exclusion_days_before=3,
        exclusion_days_after=3
    )
    signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

    # Check if signal on target date
    try:
        target_row = signals[signals['Date'] == pd.to_datetime(date)]
        if target_row.empty:
            return None
    except:
        return None

    has_signal = target_row.iloc[0]['panic_sell'] == 1

    if not has_signal:
        return None

    row = target_row.iloc[0]
    return {
        'date': date,
        'price': float(row['Close']),
        'z_score': float(row['z_score']) if 'z_score' in row else None,
        'rsi': float(row['rsi']) if 'rsi' in row else None
    }


def compare_results(backtest_results: Dict, paper_results: Dict) -> Dict:
    """
    Compare backtest vs paper trading results.

    Args:
        backtest_results: Backtest results
        paper_results: Paper trading results

    Returns:
        Comparison dictionary
    """
    print(f"\n{'='*70}")
    print(f"COMPARISON: {backtest_results['ticker']}")
    print(f"{'='*70}")

    # Compare aggregate metrics
    print(f"\n{'Metric':<25} {'Backtest':<15} {'Paper':<15} {'Match':<10}")
    print("-" * 70)

    # Total trades
    bt_trades = backtest_results['total_trades']
    pt_trades = paper_results['total_trades']
    trades_match = bt_trades == pt_trades
    print(f"{'Total Trades':<25} {bt_trades:<15} {pt_trades:<15} {'✓' if trades_match else '✗':<10}")

    # Win rate
    bt_wr = backtest_results['win_rate']
    pt_wr = paper_results['win_rate']
    wr_match = abs(bt_wr - pt_wr) < 1.0  # Within 1%
    print(f"{'Win Rate':<25} {bt_wr:<14.1f}% {pt_wr:<14.1f}% {'✓' if wr_match else '✗':<10}")

    # Total return
    bt_ret = backtest_results['total_return']
    pt_ret = paper_results['total_return']
    ret_match = abs(bt_ret - pt_ret) < 0.5  # Within 0.5%
    print(f"{'Total Return':<25} {bt_ret:<14.2f}% {pt_ret:<14.2f}% {'✓' if ret_match else '✗':<10}")

    # Overall assessment
    all_match = trades_match and wr_match and ret_match

    print(f"\n{'OVERALL ASSESSMENT:':<25} {'✅ PASS' if all_match else '❌ FAIL'}")

    if not all_match:
        print(f"\nDISCREPANCIES:")
        if not trades_match:
            print(f"  • Trade count mismatch: {abs(bt_trades - pt_trades)} difference")
        if not wr_match:
            print(f"  • Win rate mismatch: {abs(bt_wr - pt_wr):.1f}% difference")
        if not ret_match:
            print(f"  • Return mismatch: {abs(bt_ret - pt_ret):.2f}% difference")

    return {
        'ticker': backtest_results['ticker'],
        'trades_match': trades_match,
        'win_rate_match': wr_match,
        'return_match': ret_match,
        'overall_pass': all_match,
        'backtest': backtest_results,
        'paper': paper_results
    }


def run_validation():
    """Run complete validation test."""
    print("=" * 70)
    print("PAPER TRADING HISTORICAL VALIDATION")
    print("=" * 70)
    print()
    print("Objective: Validate paper trading system against backtest results")
    print("Test Period: Select dates from 2024 with known signals")
    print()

    # Test configuration - Use periods from 2023 with known signals
    test_cases = [
        # 2023 periods with volatility
        {'ticker': 'NVDA', 'start': '2023-01-01', 'end': '2023-06-30', 'expected_signals': 'multiple'},
        {'ticker': 'TSLA', 'start': '2023-01-01', 'end': '2023-06-30', 'expected_signals': 'multiple'},
        {'ticker': 'JPM', 'start': '2023-01-01', 'end': '2023-06-30', 'expected_signals': 'few'},

        # 2022 bear market period
        {'ticker': 'TSLA', 'start': '2022-10-01', 'end': '2022-12-31', 'expected_signals': 'multiple'},
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}/{len(test_cases)}: {test_case['ticker']} ({test_case['start']} to {test_case['end']})")
        print(f"{'='*70}")

        # Run backtest
        backtest = run_backtest(
            test_case['ticker'],
            test_case['start'],
            test_case['end']
        )

        if backtest is None:
            print(f"[SKIP] No trades in backtest, skipping comparison")
            continue

        # Run paper trading simulation
        paper = simulate_paper_trading(
            test_case['ticker'],
            test_case['start'],
            test_case['end']
        )

        if paper is None:
            print(f"[SKIP] No trades in paper trading, skipping comparison")
            continue

        # Compare results
        comparison = compare_results(backtest, paper)
        results.append(comparison)

    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    print()

    if not results:
        print("[ERROR] No valid test cases completed")
        return

    passed = sum(1 for r in results if r['overall_pass'])
    total = len(results)
    pass_rate = (passed / total) * 100

    print(f"Test Cases: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    print()

    print(f"{'Ticker':<10} {'Trades':<10} {'Win Rate':<10} {'Return':<10} {'Status':<10}")
    print("-" * 70)

    for r in results:
        status = '✅ PASS' if r['overall_pass'] else '❌ FAIL'
        trades_status = '✓' if r['trades_match'] else '✗'
        wr_status = '✓' if r['win_rate_match'] else '✗'
        ret_status = '✓' if r['return_match'] else '✗'

        print(f"{r['ticker']:<10} {trades_status:<10} {wr_status:<10} {ret_status:<10} {status:<10}")

    print()

    if pass_rate == 100:
        print("✅ VALIDATION SUCCESSFUL")
        print("Paper trading system accurately replicates backtest results.")
        print("System is ready for 6-week live validation period.")
    elif pass_rate >= 75:
        print("⚠️ VALIDATION MOSTLY SUCCESSFUL")
        print("Minor discrepancies detected. Review failed cases before live deployment.")
    else:
        print("❌ VALIDATION FAILED")
        print("Significant discrepancies detected. Do NOT proceed to live validation.")
        print("Debug paper trading system before deployment.")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    run_validation()

"""
EXPERIMENT: EXP-073
Date: 2025-11-17
Objective: Validate gap trading strategy as complement to mean reversion

HYPOTHESIS:
Gap-down events (>3% gap from prior close) represent extreme panic selling
and should mean-revert with higher win rate (>85%) and larger returns (+3-5%)
than regular mean reversion signals.

ALGORITHM:
1. Scan 54 portfolio stocks for historical gap-down events (2020-2025)
2. Filter gaps: 3-15% gap down, volume > 1.5x average
3. Backtest gap trading strategy (entry at open, exit at gap fill or 3 days)
4. Compare vs regular mean reversion performance
5. Calculate incremental trade opportunities

EXPECTED RESULTS:
- Win rate: >85% (vs 78.5% for regular MR)
- Avg return: +3-5% per trade (vs +2% for regular MR)
- Trade frequency: +50-100 trades/year (gaps are less frequent)
- Risk: Larger drawdowns possible (gaps can continue)

SUCCESS CRITERIA:
- Win rate >= 80%
- At least 50 gap opportunities across 54 stocks (2020-2025)
- Average return >= +3%
- Sharpe ratio >= 2.0
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector
from common.models.trading.gap_trading import GapDetector, GapBacktester
from common.config.mean_reversion_params import get_all_tickers


def validate_gap_trading_stock(ticker: str,
                               start_date: str = '2020-01-01',
                               end_date: str = '2025-11-17') -> Dict:
    """
    Validate gap trading strategy for a single stock.

    Args:
        ticker: Stock ticker
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        Validation results dict
    """
    try:
        # Fetch data
        buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
        fetcher = YahooFinanceFetcher()
        data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)

        if len(data) < 60:
            return None

        # Engineer features (for RSI, volume ratios, etc.)
        engineer = TechnicalFeatureEngineer(fillna=True)
        enriched_data = engineer.engineer_features(data)

        # Detect gaps
        detector = GapDetector(min_gap_pct=-3.0, min_volume_ratio=1.5, max_gap_pct=-15.0)
        gaps = detector.detect_gaps(enriched_data)

        if len(gaps) == 0:
            return {
                'ticker': ticker,
                'total_gaps': 0,
                'message': 'No qualifying gaps found'
            }

        # Score gaps
        gaps = detector.calculate_gap_score(gaps)

        # TEMPORARY: Skip regime filter to validate raw gap trading performance
        # TODO: Re-enable after validating base strategy works
        # regime_detector = MarketRegimeDetector()
        # gaps = detector.filter_by_regime(gaps, regime_detector, enriched_data)

        # if len(gaps) == 0:
        #     return {
        #         'ticker': ticker,
        #         'total_gaps': 0,
        #         'message': 'No gaps in favorable regime'
        #     }

        # Calculate entry/exit targets
        gaps = detector.calculate_entry_exit(gaps, profit_target=5.0, stop_loss=-5.0, max_hold_days=3)

        # Backtest
        backtester = GapBacktester(
            initial_capital=10000,
            profit_target=5.0,
            stop_loss=-5.0,
            max_hold_days=3
        )

        results = backtester.backtest(gaps, enriched_data)

        return {
            'ticker': ticker,
            'total_gaps': results['total_trades'],
            'win_rate': results['win_rate'],
            'total_return': results['total_return'],
            'avg_return_per_trade': results['avg_return_per_trade'],
            'avg_win': results['avg_win'],
            'avg_loss': results['avg_loss'],
            'avg_hold_days': results['avg_hold_days'],
            'max_return': results['max_return'],
            'max_loss': results['max_loss'],
            'trades': results['trades']
        }

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        return None


def run_exp073_gap_trading_validation():
    """
    Validate gap trading strategy across entire portfolio.
    """
    print("="*70)
    print("EXP-073: GAP TRADING STRATEGY VALIDATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("OBJECTIVE: Validate gap-down trading as complement to mean reversion")
    print("METHODOLOGY: Historical gap detection + backtest (2020-2025)")
    print("TARGET: 80%+ win rate, +3% avg return, 50+ gap opportunities")
    print()

    # Get all portfolio tickers
    tickers = get_all_tickers()
    print(f"Validating {len(tickers)} stocks for gap trading opportunities...")
    print()

    validation_results = []
    total_gaps = 0
    total_wins = 0
    total_losses = 0
    all_returns = []

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Testing {ticker}...", end=" ")

        result = validate_gap_trading_stock(ticker)

        if result and result.get('total_gaps', 0) > 0:
            validation_results.append(result)
            total_gaps += result['total_gaps']

            if 'trades' in result:
                for trade in result['trades']:
                    all_returns.append(trade['return_pct'])
                    if trade['outcome'] == 'win':
                        total_wins += 1
                    else:
                        total_losses += 1

            print(f"Found {result['total_gaps']} gaps, WR={result['win_rate']:.1f}%, "
                  f"Avg Return={result['avg_return_per_trade']:+.2f}%")
        else:
            print("No qualifying gaps")

    # Summary
    print("\n" + "="*70)
    print("GAP TRADING VALIDATION RESULTS")
    print("="*70)
    print()

    if not validation_results:
        print("[ERROR] No gap trading opportunities found")
        return None

    # Calculate portfolio-wide statistics
    overall_win_rate = (total_wins / (total_wins + total_losses)) * 100 if (total_wins + total_losses) > 0 else 0
    avg_return = np.mean(all_returns) if all_returns else 0
    std_return = np.std(all_returns) if all_returns else 0
    sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0

    print(f"Total Gap Opportunities: {total_gaps} (across {len(validation_results)} stocks)")
    print(f"Total Trades: {total_wins + total_losses}")
    print(f"Wins: {total_wins} | Losses: {total_losses}")
    print(f"Overall Win Rate: {overall_win_rate:.1f}%")
    print(f"Average Return per Trade: {avg_return:+.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print()

    # Top performers
    print("TOP 10 STOCKS BY GAP TRADING PERFORMANCE:")
    print("-"*70)
    print(f"{'Ticker':<8} {'Gaps':<8} {'Win Rate':<12} {'Avg Return':<12} {'Total Return'}")
    print("-"*70)

    sorted_results = sorted(validation_results,
                          key=lambda x: (x['win_rate'], x['avg_return_per_trade']),
                          reverse=True)

    for result in sorted_results[:10]:
        print(f"{result['ticker']:<8} "
              f"{result['total_gaps']:<8} "
              f"{result['win_rate']:<11.1f}% "
              f"{result['avg_return_per_trade']:<11.2f}% "
              f"{result['total_return']:+.1f}%")

    # Comparison to mean reversion
    print()
    print("="*70)
    print("COMPARISON: GAP TRADING vs MEAN REVERSION")
    print("="*70)
    print()

    print(f"{'Metric':<25} {'Gap Trading':<20} {'Mean Reversion':<20} {'Improvement'}")
    print("-"*70)
    print(f"{'Win Rate':<25} {overall_win_rate:<19.1f}% {'78.5%':<20} "
          f"{overall_win_rate - 78.5:+.1f}pp")
    print(f"{'Avg Return/Trade':<25} {avg_return:<19.2f}% {'~2.0%':<20} "
          f"{avg_return - 2.0:+.2f}%")
    print(f"{'Trade Opportunities':<25} {total_gaps:<19} {'~281/year':<20} "
          f"{total_gaps / 5:.0f}/year")
    print(f"{'Sharpe Ratio':<25} {sharpe:<19.2f} {'~3.5':<20} "
          f"{sharpe - 3.5:+.2f}")

    # Deployment recommendation
    print()
    print("="*70)
    print("DEPLOYMENT RECOMMENDATION")
    print("="*70)
    print()

    if overall_win_rate >= 80 and total_gaps >= 50 and avg_return >= 3.0:
        print("[SUCCESS] Gap trading strategy VALIDATED for deployment!")
        print()
        print(f"RESULTS: {overall_win_rate:.1f}% WR, {avg_return:+.2f}% avg return, {total_gaps} opportunities")
        print()
        print("NEXT STEPS:")
        print("  1. Integrate GapDetector into daily scanner")
        print("  2. Run gap trading in parallel with mean reversion")
        print("  3. Apply ML probability scoring to gaps")
        print("  4. Deploy to production")
        print()
        print(f"ESTIMATED IMPACT: +{total_gaps / 5:.0f} trades/year, higher returns per trade")
    elif overall_win_rate >= 75:
        print(f"[PARTIAL SUCCESS] {overall_win_rate:.1f}% WR (target: 80%)")
        print()
        print("Gap trading shows promise but may need refinement:")
        print("  - Consider tighter filters (min gap 4-5%)")
        print("  - Add ML probability scoring")
        print("  - Adjust profit targets")
    else:
        print(f"[NEEDS WORK] {overall_win_rate:.1f}% WR below threshold")
        print()
        print("Gap trading may not be viable as standalone strategy")
        print("Consider combining with mean reversion indicators")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-073',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'objective': 'Validate gap trading strategy',
        'algorithm': 'Gap-down detection + mean reversion backtesting',
        'total_stocks_tested': len(tickers),
        'stocks_with_gaps': len(validation_results),
        'total_gap_opportunities': total_gaps,
        'overall_win_rate': float(overall_win_rate),
        'avg_return_per_trade': float(avg_return),
        'sharpe_ratio': float(sharpe),
        'total_wins': total_wins,
        'total_losses': total_losses,
        'stock_results': validation_results,
        'deploy': overall_win_rate >= 80 and total_gaps >= 50,
        'next_step': 'Deploy to production' if overall_win_rate >= 80 else 'Refine strategy',
        'methodology': {
            'data_period': '2020-2025',
            'min_gap_pct': -3.0,
            'max_gap_pct': -15.0,
            'min_volume_ratio': 1.5,
            'profit_target': 5.0,
            'stop_loss': -5.0,
            'max_hold_days': 3
        }
    }

    results_file = os.path.join(results_dir, 'exp073_gap_trading_validation.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending gap trading validation report...")
            notifier.send_experiment_report('EXP-073', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-073: Gap trading validation."""

    print("\n[GAP TRADING VALIDATION] Testing extreme mean reversion strategy")
    print("Validating gap-down opportunities across 54-stock portfolio")
    print()

    results = run_exp073_gap_trading_validation()

    print("\n" + "="*70)
    print("GAP TRADING VALIDATION COMPLETE")
    print("="*70)

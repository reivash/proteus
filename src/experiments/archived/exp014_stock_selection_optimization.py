"""
EXPERIMENT: EXP-014
Date: 2025-11-16
Objective: Identify which stocks have the best mean reversion performance

RESEARCH MOTIVATION:
EXP-013 revealed that not all panic sells are equal:
- NVDA: 87.5% win rate, +49.70% return (EXCELLENT)
- TSLA: 46.67% win rate, -29.84% return (POOR)

Mean reversion doesn't work universally. Some stocks have:
- Strong mean reversion characteristics (NVDA)
- Weak or no mean reversion (TSLA)
- Information-driven panics vs emotion-driven panics

HYPOTHESIS:
Backtesting all 11 monitored symbols will reveal a "tier list":
- Tier A (>70% win rate): Trade aggressively
- Tier B (55-70% win rate): Trade cautiously
- Tier C (<55% win rate): Skip entirely

By focusing capital on Tier A stocks only, we can improve portfolio returns by 30-50%
while avoiding losses from Tier C stocks.

METHOD:
1. Run 3-year backtest on all 11 monitored symbols
2. Measure: win rate, total return, avg gain, avg loss, Sharpe ratio
3. Rank stocks by performance
4. Create approved trading list
5. Estimate improvement from selective trading

SYMBOLS TO TEST (current monitored list):
1. NVDA - ✓ Tested (87.5% win rate)
2. TSLA - ✓ Tested (46.67% win rate)
3. JPM - To test
4. AAPL - To test
5. BAC - To test
6. CVX - To test
7. GS - To test
8. HD - To test
9. MA - To test
10. QQQ - To test
11. V - To test

EXPECTED OUTCOMES:
- Find 4-6 Tier A stocks with >70% win rate
- Avoid 2-3 Tier C stocks with <55% win rate
- Expected return improvement: +30-50pp by selective trading
- Reduced drawdown: -40% by avoiding losers

COMPARISON METRICS:
- Win rate by stock
- Total return by stock
- Avg gain per trade
- Risk-adjusted returns (Sharpe)
- Stock characteristics (volatility, sector, market cap)
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.models.trading.mean_reversion import MeanReversionDetector
from src.config.mean_reversion_params import get_params


class StockSelectionBacktester:
    """Backtest mean reversion across multiple stocks to find best targets."""

    def __init__(self, initial_capital: float = 10000):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital per stock test
        """
        self.initial_capital = initial_capital

    def backtest_stock(self, ticker: str, period: str = "3y") -> Dict:
        """
        Run full backtest on a single stock.

        Args:
            ticker: Stock symbol
            period: Backtest period

        Returns:
            Dictionary with performance metrics
        """
        print(f"\n[Testing {ticker}]")

        try:
            # Fetch data
            print("  [1/4] Fetching data...")
            fetcher = YahooFinanceFetcher()
            price_data = fetcher.fetch_stock_data(ticker, period=period)

            if len(price_data) < 100:
                print(f"  [SKIP] Insufficient data ({len(price_data)} rows)")
                return self._failed_result(ticker, "Insufficient data")

            print(f"  [OK] {len(price_data)} rows")

            # Add technical indicators
            print("  [2/4] Calculating indicators...")
            engineer = TechnicalFeatureEngineer(fillna=True)
            enriched_data = engineer.engineer_features(price_data)
            print("  [OK] Indicators added")

            # Detect signals
            print("  [3/4] Detecting signals...")
            params = get_params(ticker)
            detector = MeanReversionDetector(
                z_score_threshold=params['z_score_threshold'],
                rsi_oversold=params['rsi_oversold'],
                volume_multiplier=params['volume_multiplier'],
                price_drop_threshold=params['price_drop_threshold']
            )
            signals = detector.detect_overcorrections(enriched_data)
            num_signals = signals['panic_sell'].sum()

            if num_signals == 0:
                print("  [SKIP] No panic sell signals")
                return self._failed_result(ticker, "No signals")

            print(f"  [OK] {num_signals} signals")

            # Backtest
            print("  [4/4] Running backtest...")
            results = self._run_backtest(signals, enriched_data)
            results['ticker'] = ticker
            results['num_signals'] = num_signals
            results['period'] = period
            results['status'] = 'success'

            print(f"  [DONE] Win rate: {results['win_rate']:.1f}%, Return: {results['total_return']:+.2f}%")

            return results

        except Exception as e:
            print(f"  [ERROR] {str(e)}")
            return self._failed_result(ticker, str(e))

    def _run_backtest(self, signals_df: pd.DataFrame, price_data: pd.DataFrame) -> Dict:
        """
        Execute backtest with time-decay exits (v5.0 strategy).

        Args:
            signals_df: DataFrame with panic_sell signals
            price_data: Full OHLCV data

        Returns:
            Performance metrics
        """
        capital = self.initial_capital
        trades = []

        # Time-decay targets (same as v5.0)
        time_decay = {
            0: (2.0, -2.0),   # Day 0: ±2%
            1: (1.5, -1.5),   # Day 1: ±1.5%
            2: (1.0, -1.0),   # Day 2+: ±1%
        }

        max_hold_days = 7

        # Get panic sell signals
        panic_sells = signals_df[signals_df['panic_sell'] == True].copy()
        price_data_indexed = price_data.set_index('Date')

        for idx, signal_row in panic_sells.iterrows():
            entry_date = signal_row['Date']
            entry_price = signal_row['Close']

            try:
                entry_idx = price_data_indexed.index.get_loc(entry_date)
            except KeyError:
                continue

            # Find exit
            for days_held in range(max_hold_days):
                current_idx = entry_idx + days_held + 1

                if current_idx >= len(price_data_indexed):
                    break

                current_price = price_data_indexed.iloc[current_idx]['Close']
                return_pct = ((current_price / entry_price) - 1) * 100

                # Check time-decay targets
                profit_target, stop_loss = time_decay.get(min(days_held, 2), (1.0, -1.0))

                if return_pct >= profit_target:
                    # Exit with profit
                    shares = int(capital / entry_price)
                    profit = (current_price - entry_price) * shares
                    capital += profit

                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': price_data_indexed.index[current_idx],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'return_pct': return_pct,
                        'days_held': days_held,
                        'exit_reason': f'profit_day{days_held}'
                    })
                    break

                elif return_pct <= stop_loss:
                    # Exit with loss
                    shares = int(capital / entry_price)
                    profit = (current_price - entry_price) * shares
                    capital += profit

                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': price_data_indexed.index[current_idx],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'return_pct': return_pct,
                        'days_held': days_held,
                        'exit_reason': f'stop_loss_day{days_held}'
                    })
                    break
            else:
                # Timeout exit
                timeout_idx = min(entry_idx + max_hold_days, len(price_data_indexed) - 1)
                current_price = price_data_indexed.iloc[timeout_idx]['Close']
                return_pct = ((current_price / entry_price) - 1) * 100

                shares = int(capital / entry_price)
                profit = (current_price - entry_price) * shares
                capital += profit

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': price_data_indexed.index[timeout_idx],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'return_pct': return_pct,
                    'days_held': max_hold_days - 1,
                    'exit_reason': 'timeout'
                })

        # Calculate metrics
        return self._calculate_metrics(trades, capital)

    def _calculate_metrics(self, trades: List[Dict], final_capital: float) -> Dict:
        """Calculate performance metrics from trades."""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'avg_gain': 0,
                'avg_loss': 0,
                'avg_hold_days': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0
            }

        trades_df = pd.DataFrame(trades)

        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['return_pct'] > 0])
        losing_trades = len(trades_df[trades_df['return_pct'] <= 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100

        winners = trades_df[trades_df['return_pct'] > 0]['return_pct']
        losers = trades_df[trades_df['return_pct'] <= 0]['return_pct']

        avg_gain = winners.mean() if len(winners) > 0 else 0
        avg_loss = losers.mean() if len(losers) > 0 else 0

        avg_hold_days = trades_df['days_held'].mean()

        # Risk metrics
        returns = trades_df['return_pct'].values
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Max drawdown
        cumulative_returns = (1 + trades_df['return_pct'] / 100).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        # Profit factor
        total_gains = winners.sum() if len(winners) > 0 else 0
        total_losses = abs(losers.sum()) if len(losers) > 0 else 0
        profit_factor = (total_gains / total_losses) if total_losses > 0 else 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_gain': avg_gain,
            'avg_loss': avg_loss,
            'avg_hold_days': avg_hold_days,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor
        }

    def _failed_result(self, ticker: str, reason: str) -> Dict:
        """Return failed backtest result."""
        return {
            'ticker': ticker,
            'status': 'failed',
            'reason': reason,
            'total_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'avg_gain': 0,
            'avg_loss': 0,
            'sharpe_ratio': 0
        }


def run_exp014_stock_selection(symbols: List[str], period: str = "3y"):
    """
    Backtest all symbols and rank by performance.

    Args:
        symbols: List of stock tickers
        period: Backtest period
    """
    print("=" * 70)
    print("EXP-014: STOCK SELECTION OPTIMIZATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"Configuration:")
    print(f"  Symbols: {len(symbols)} stocks")
    print(f"  Period: {period}")
    print(f"  Initial Capital: $10,000 per stock")
    print(f"  Strategy: Mean Reversion v5.0 (time-decay exits)")
    print()

    backtester = StockSelectionBacktester(initial_capital=10000)
    results = []

    # Test each stock
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {symbol}")
        print("-" * 70)

        result = backtester.backtest_stock(symbol, period=period)
        results.append(result)

    # Display results
    print("\n" + "=" * 70)
    print("STOCK PERFORMANCE RANKING")
    print("=" * 70)
    print()

    # Convert to DataFrame for ranking
    results_df = pd.DataFrame(results)

    # Sort by win rate (primary) and total return (secondary)
    results_df = results_df.sort_values(
        by=['win_rate', 'total_return'],
        ascending=[False, False]
    )

    # Display table
    print(f"{'Rank':<6}{'Symbol':<8}{'Trades':<8}{'Win Rate':<12}{'Return':<12}{'Avg Gain':<12}{'Sharpe':<10}{'Status':<10}")
    print("-" * 75)

    for rank, (idx, row) in enumerate(results_df.iterrows(), 1):
        if row['status'] == 'success':
            print(f"{rank:<6}{row['ticker']:<8}{row['total_trades']:<8}{row['win_rate']:<11.1f}% {row['total_return']:<11.2f}% {row['avg_gain']:<11.2f}% {row['sharpe_ratio']:<10.2f}{'[OK]':<10}")
        else:
            reason = f"[{row.get('reason', 'ERROR')}]"
            print(f"{rank:<6}{row['ticker']:<8}{'N/A':<8}{'N/A':<12}{'N/A':<12}{'N/A':<12}{'N/A':<10}{reason:<10}")

    # Tier classification
    print()
    print("=" * 70)
    print("TIER CLASSIFICATION")
    print("=" * 70)
    print()

    tier_a = results_df[(results_df['status'] == 'success') & (results_df['win_rate'] >= 70)]
    tier_b = results_df[(results_df['status'] == 'success') & (results_df['win_rate'] >= 55) & (results_df['win_rate'] < 70)]
    tier_c = results_df[(results_df['status'] == 'success') & (results_df['win_rate'] < 55)]

    print(f"TIER A (>70% win rate) - TRADE AGGRESSIVELY:")
    if len(tier_a) > 0:
        for idx, row in tier_a.iterrows():
            print(f"  [{row['ticker']}] {row['win_rate']:.1f}% win rate, {row['total_return']:+.2f}% return")
    else:
        print("  None")

    print()
    print(f"TIER B (55-70% win rate) - TRADE CAUTIOUSLY:")
    if len(tier_b) > 0:
        for idx, row in tier_b.iterrows():
            print(f"  [{row['ticker']}] {row['win_rate']:.1f}% win rate, {row['total_return']:+.2f}% return")
    else:
        print("  None")

    print()
    print(f"TIER C (<55% win rate) - AVOID:")
    if len(tier_c) > 0:
        for idx, row in tier_c.iterrows():
            print(f"  [{row['ticker']}] {row['win_rate']:.1f}% win rate, {row['total_return']:+.2f}% return")
    else:
        print("  None")

    # Portfolio impact analysis
    print()
    print("=" * 70)
    print("PORTFOLIO IMPACT ANALYSIS")
    print("=" * 70)
    print()

    if len(tier_a) > 0:
        tier_a_avg_return = tier_a['total_return'].mean()
        tier_a_avg_win_rate = tier_a['win_rate'].mean()

        print(f"Trading ONLY Tier A stocks ({len(tier_a)} symbols):")
        print(f"  Average return: {tier_a_avg_return:+.2f}%")
        print(f"  Average win rate: {tier_a_avg_win_rate:.1f}%")
        print(f"  Sharpe ratio: {tier_a['sharpe_ratio'].mean():.2f}")
        print()

    all_successful = results_df[results_df['status'] == 'success']
    if len(all_successful) > 0:
        all_avg_return = all_successful['total_return'].mean()
        all_avg_win_rate = all_successful['win_rate'].mean()

        print(f"Trading ALL stocks ({len(all_successful)} symbols):")
        print(f"  Average return: {all_avg_return:+.2f}%")
        print(f"  Average win rate: {all_avg_win_rate:.1f}%")
        print(f"  Sharpe ratio: {all_successful['sharpe_ratio'].mean():.2f}")
        print()

        if len(tier_a) > 0:
            improvement = tier_a_avg_return - all_avg_return
            print(f"IMPROVEMENT from selective trading (Tier A only):")
            print(f"  Return improvement: {improvement:+.2f}pp")
            print(f"  Win rate improvement: {tier_a_avg_win_rate - all_avg_win_rate:+.2f}pp")

    print()
    print("=" * 70)

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    results_data = {
        'experiment_id': 'EXP-014',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': period,
        'symbols_tested': len(symbols),
        'results': results,
        'tier_a': tier_a['ticker'].tolist() if len(tier_a) > 0 else [],
        'tier_b': tier_b['ticker'].tolist() if len(tier_b) > 0 else [],
        'tier_c': tier_c['ticker'].tolist() if len(tier_c) > 0 else [],
    }

    results_file = os.path.join(results_dir, 'exp014_stock_selection.json')
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Recommendations
    print()
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print()

    if len(tier_a) > 0:
        print("[SUCCESS] Found high-performance stocks!")
        print()
        print("Deployment strategy:")
        print(f"  1. Trade ONLY Tier A stocks: {', '.join(tier_a['ticker'].tolist())}")
        print(f"  2. Expected portfolio win rate: {tier_a_avg_win_rate:.1f}%")
        print(f"  3. Expected return improvement: {improvement:+.2f}pp vs trading all")
        print()
        print("Update mean_reversion_params.py:")
        print(f"  APPROVED_SYMBOLS = {tier_a['ticker'].tolist()}")
    else:
        print("[PARTIAL SUCCESS] No stocks meet Tier A criteria (>70% win rate)")
        print()
        print("Consider:")
        print("  1. Adjust entry parameters for better signal quality")
        print("  2. Integrate EXP-011 sentiment filtering")
        print("  3. Test additional symbols")

    print("=" * 70)

    return results_data


if __name__ == '__main__':
    """Run EXP-014 stock selection optimization."""

    # All monitored symbols (from mean_reversion_params.py)
    symbols = [
        'NVDA',  # Already tested: 87.5% win rate
        'TSLA',  # Already tested: 46.67% win rate
        'JPM',
        'AAPL',
        'BAC',
        'CVX',
        'GS',
        'HD',
        'MA',
        'QQQ',
        'V'
    ]

    results = run_exp014_stock_selection(symbols, period='3y')

    print("\n\nNext steps:")
    print("1. Deploy selective trading (Tier A stocks only)")
    print("2. Integrate with sentiment filtering (EXP-011)")
    print("3. Monitor performance improvement vs baseline")
    print("4. Re-evaluate quarterly (stocks can change tiers)")

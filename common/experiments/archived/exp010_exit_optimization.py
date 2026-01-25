"""
EXPERIMENT: EXP-010-EXIT-OPTIMIZATION
Date: 2025-11-14
Objective: Optimize exit strategy to maximize returns while maintaining win rate

CURRENT EXIT STRATEGY (v4.0):
- Profit target: +2%
- Stop loss: -2%
- Max hold: 2 days
- Exit type: Fixed targets

HYPOTHESIS:
Alternative exit strategies may capture larger moves without sacrificing win rate:
1. Trailing stops: Let winners run, protect profits
2. Time-decay exits: Tighter stops as hold time increases
3. Partial profit taking: Scale out to balance risk/reward
4. Volatility-adjusted: Dynamic targets based on ATR/VIX

TEST METHODOLOGY:
Run backtest on all 10 stocks (2022-2025) with each exit strategy:
- Strategy A: Current (±2%, 2 days) - BASELINE
- Strategy B: Trailing stop (initial +2%, trail at +1%)
- Strategy C: Time-decay (Day 1: ±2%, Day 2: ±1.5%)
- Strategy D: Partial profit (50% at +2%, 50% at +4%)

EXPECTED OUTCOME:
- Trailing stops: Higher returns, similar win rate
- Time-decay: Lower max hold losses
- Partial profit: Better risk-adjusted returns

SUCCESS CRITERIA:
- Maintain 75%+ win rate
- Improve portfolio return by 5%+
- Improve Sharpe ratio by 0.5+
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from typing import Dict, List

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from common.models.trading.mean_reversion import MeanReversionDetector
from common.config.mean_reversion_params import get_params, get_all_tickers


class ExitStrategy:
    """Base class for exit strategies."""

    def __init__(self, name: str):
        self.name = name

    def check_exit(self, entry_price: float, current_price: float, hold_days: int, high_since_entry: float = None) -> tuple:
        """
        Check if position should be exited.

        Returns:
            (should_exit: bool, exit_reason: str, exit_price: float)
        """
        raise NotImplementedError


class FixedTargetExit(ExitStrategy):
    """Current strategy: Fixed ±2%, 2-day max hold."""

    def __init__(self, profit_target: float = 2.0, stop_loss: float = -2.0, max_hold: int = 2):
        super().__init__(f"Fixed (±{profit_target}%, {max_hold}d)")
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_hold = max_hold

    def check_exit(self, entry_price, current_price, hold_days, high_since_entry=None):
        return_pct = ((current_price - entry_price) / entry_price) * 100

        if return_pct >= self.profit_target:
            return (True, 'profit_target', current_price)
        elif return_pct <= self.stop_loss:
            return (True, 'stop_loss', current_price)
        elif hold_days >= self.max_hold:
            return (True, 'max_hold', current_price)

        return (False, None, None)


class TrailingStopExit(ExitStrategy):
    """Trailing stop: Initial +2% target, then trail at +1% below high."""

    def __init__(self, initial_target: float = 2.0, trail_pct: float = 1.0, stop_loss: float = -2.0, max_hold: int = 3):
        super().__init__(f"Trailing (+{initial_target}%, trail {trail_pct}%)")
        self.initial_target = initial_target
        self.trail_pct = trail_pct
        self.stop_loss = stop_loss
        self.max_hold = max_hold

    def check_exit(self, entry_price, current_price, hold_days, high_since_entry):
        return_pct = ((current_price - entry_price) / entry_price) * 100

        # Initial stop loss
        if return_pct <= self.stop_loss:
            return (True, 'stop_loss', current_price)

        # Max hold
        if hold_days >= self.max_hold:
            return (True, 'max_hold', current_price)

        # Once we hit initial target, activate trailing stop
        if high_since_entry:
            high_return = ((high_since_entry - entry_price) / entry_price) * 100

            if high_return >= self.initial_target:
                # Trail from high
                current_from_high = ((current_price - high_since_entry) / high_since_entry) * 100

                if current_from_high <= -self.trail_pct:
                    return (True, 'trailing_stop', current_price)

        return (False, None, None)


class TimeDecayExit(ExitStrategy):
    """Time-decay: Tighter stops as hold time increases."""

    def __init__(self):
        super().__init__("Time-Decay")
        # Day 0: ±2%, Day 1: ±1.5%, Day 2+: ±1%
        self.targets_by_day = {
            0: (2.0, -2.0),
            1: (1.5, -1.5),
            2: (1.0, -1.0),
        }
        self.max_hold = 3

    def check_exit(self, entry_price, current_price, hold_days, high_since_entry=None):
        return_pct = ((current_price - entry_price) / entry_price) * 100

        # Get targets for current day
        profit_target, stop_loss = self.targets_by_day.get(hold_days, (1.0, -1.0))

        if return_pct >= profit_target:
            return (True, f'profit_target_d{hold_days}', current_price)
        elif return_pct <= stop_loss:
            return (True, f'stop_loss_d{hold_days}', current_price)
        elif hold_days >= self.max_hold:
            return (True, 'max_hold', current_price)

        return (False, None, None)


class PartialProfitExit(ExitStrategy):
    """Partial profit: Scale out 50% at +2%, hold 50% for +4%."""

    def __init__(self):
        super().__init__("Partial (50%@+2%, 50%@+4%)")
        self.first_target = 2.0
        self.second_target = 4.0
        self.stop_loss = -2.0
        self.max_hold = 3
        self.partial_exited = False

    def check_exit(self, entry_price, current_price, hold_days, high_since_entry=None):
        return_pct = ((current_price - entry_price) / entry_price) * 100

        # Stop loss (full position)
        if return_pct <= self.stop_loss:
            return (True, 'stop_loss', current_price)

        # First target: Exit 50%
        if return_pct >= self.first_target and not self.partial_exited:
            self.partial_exited = True
            return (True, 'partial_profit_50pct', current_price)

        # Second target or max hold: Exit remaining 50%
        if return_pct >= self.second_target or hold_days >= self.max_hold:
            reason = 'full_profit' if return_pct >= self.second_target else 'max_hold_partial'
            return (True, reason, current_price)

        return (False, None, None)


def backtest_with_exit_strategy(ticker: str, start_date: str, end_date: str, exit_strategy: ExitStrategy) -> Dict:
    """
    Backtest ticker with specified exit strategy.

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date
        exit_strategy: Exit strategy to use

    Returns:
        Results dictionary
    """
    # Fetch data
    fetcher = YahooFinanceFetcher()
    try:
        data = fetcher.fetch_stock_data(ticker, start_date=start_date, end_date=end_date)
    except Exception as e:
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

    # Custom backtest with exit strategy
    trades = []
    position = None

    for idx, row in signals.iterrows():
        # Check for new signal
        if row['panic_sell'] == 1 and position is None:
            position = {
                'entry_date': row['Date'],
                'entry_price': row['Close'],
                'hold_days': 0,
                'high_since_entry': row['Close']
            }

        # Update existing position
        if position is not None:
            current_price = row['Close']
            position['hold_days'] += 1
            position['high_since_entry'] = max(position['high_since_entry'], current_price)

            # Check exit
            should_exit, exit_reason, exit_price = exit_strategy.check_exit(
                position['entry_price'],
                current_price,
                position['hold_days'],
                position['high_since_entry']
            )

            if should_exit:
                # Record trade
                return_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100

                trades.append({
                    'entry_date': position['entry_date'],
                    'entry_price': position['entry_price'],
                    'exit_date': row['Date'],
                    'exit_price': exit_price,
                    'return': return_pct,
                    'hold_days': position['hold_days'],
                    'exit_reason': exit_reason
                })

                position = None

    # Calculate results
    if not trades:
        return None

    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['return'] > 0)
    win_rate = (winning_trades / total_trades) * 100
    total_return = sum(t['return'] for t in trades)
    avg_return = total_return / total_trades

    # Calculate Sharpe ratio
    returns = [t['return'] for t in trades]
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 / 2) if np.std(returns) > 0 else 0

    # Max drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0

    return {
        'ticker': ticker,
        'exit_strategy': exit_strategy.name,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'avg_return': avg_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'trades': trades
    }


def compare_exit_strategies():
    """Compare all exit strategies across portfolio."""
    print("=" * 70)
    print("EXIT STRATEGY OPTIMIZATION")
    print("=" * 70)
    print()

    # Test period
    start_date = '2022-01-01'
    end_date = '2025-11-14'

    print(f"Test Period: {start_date} to {end_date}")
    print(f"Universe: {len(get_all_tickers())} stocks")
    print()

    # Define exit strategies
    strategies = [
        FixedTargetExit(profit_target=2.0, stop_loss=-2.0, max_hold=2),  # Current
        TrailingStopExit(initial_target=2.0, trail_pct=1.0, stop_loss=-2.0, max_hold=3),
        TimeDecayExit(),
        # PartialProfitExit(),  # Requires more complex position tracking
    ]

    # Test each strategy
    all_results = {}

    for strategy in strategies:
        print(f"\nTesting: {strategy.name}")
        print("-" * 70)

        strategy_results = []

        for ticker in get_all_tickers():
            print(f"  {ticker}...", end=" ")

            result = backtest_with_exit_strategy(ticker, start_date, end_date, strategy)

            if result:
                strategy_results.append(result)
                print(f"{result['total_trades']} trades, {result['win_rate']:.1f}% win, {result['total_return']:+.2f}%")
            else:
                print("No trades")

        all_results[strategy.name] = strategy_results

    # Compare results
    print("\n" + "=" * 70)
    print("PORTFOLIO-LEVEL COMPARISON")
    print("=" * 70)
    print()

    comparison = []

    for strategy_name, results in all_results.items():
        if not results:
            continue

        # Aggregate metrics
        total_trades = sum(r['total_trades'] for r in results)
        total_winning = sum(r['winning_trades'] for r in results)

        avg_win_rate = np.average(
            [r['win_rate'] for r in results],
            weights=[r['total_trades'] for r in results]
        )

        portfolio_return = sum(r['total_return'] for r in results)
        avg_return_per_stock = portfolio_return / len(results)

        avg_sharpe = np.average(
            [r['sharpe_ratio'] for r in results],
            weights=[r['total_trades'] for r in results]
        )

        avg_max_dd = np.mean([r['max_drawdown'] for r in results])

        comparison.append({
            'strategy': strategy_name,
            'total_trades': total_trades,
            'avg_win_rate': avg_win_rate,
            'portfolio_return': portfolio_return,
            'avg_return_per_stock': avg_return_per_stock,
            'avg_sharpe': avg_sharpe,
            'avg_max_dd': avg_max_dd
        })

    # Print comparison table
    baseline = comparison[0]  # First strategy is baseline

    print(f"{'Strategy':<30} {'Trades':<10} {'Win Rate':<12} {'Portfolio':<15} {'Sharpe':<10} {'vs Baseline':<15}")
    print("-" * 100)

    for comp in comparison:
        win_rate_change = comp['avg_win_rate'] - baseline['avg_win_rate']
        return_change = comp['portfolio_return'] - baseline['portfolio_return']
        sharpe_change = comp['avg_sharpe'] - baseline['avg_sharpe']

        status = ""
        if comp['strategy'] == baseline['strategy']:
            status = "(BASELINE)"
        elif return_change > 5 and win_rate_change >= -2:
            status = "[BETTER]"
        elif return_change < -5 or win_rate_change < -5:
            status = "[WORSE]"
        else:
            status = "[SIMILAR]"

        print(f"{comp['strategy']:<30} {comp['total_trades']:<10} {comp['avg_win_rate']:<11.1f}% "
              f"{comp['portfolio_return']:>+13.2f}% {comp['avg_sharpe']:<9.2f} {status:<15}")

        if comp['strategy'] != baseline['strategy']:
            print(f"{'':30} {'':10} {win_rate_change:>+10.1f}pp "
                  f"{return_change:>+13.2f}pp {sharpe_change:>+8.2f}")

    print("\n" + "=" * 100)

    # Recommendation
    print("\nRECOMMENDATION:")
    print()

    best_strategy = max(comparison[1:], key=lambda x: x['portfolio_return'])

    if best_strategy['portfolio_return'] > baseline['portfolio_return'] + 5:
        print(f"[RECOMMENDED] {best_strategy['strategy']}")
        print(f"  Portfolio return: +{best_strategy['portfolio_return'] - baseline['portfolio_return']:.2f}pp improvement")
        print(f"  Win rate: {best_strategy['avg_win_rate']:.1f}% (change: {best_strategy['avg_win_rate'] - baseline['avg_win_rate']:+.1f}pp)")
        print(f"  Sharpe ratio: {best_strategy['avg_sharpe']:.2f} (change: {best_strategy['avg_sharpe'] - baseline['avg_sharpe']:+.2f})")
    else:
        print(f"[KEEP BASELINE] {baseline['strategy']}")
        print(f"  Alternative strategies do not provide meaningful improvement (< 5% return gain)")
        print(f"  Current strategy remains optimal")

    print()


if __name__ == "__main__":
    compare_exit_strategies()

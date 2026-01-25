"""
EXPERIMENT: EXP-074
Date: 2025-11-17
Objective: Optimize exit strategy for maximum returns and win rate

HYPOTHESIS:
Current time-decay exit strategy (from EXP-010, 2020) is suboptimal.
By testing alternative exit strategies, we can improve average return per trade
by 0.5-1.0%, which compounds to +140-280% annual improvement across 281 trades.

CURRENT STRATEGY (TIME-DECAY):
- Day 0: ±2.0% profit target / stop loss
- Day 1: ±1.5% profit target / stop loss
- Day 2+: ±1.0% profit target / stop loss
- Max hold: 3 days

ALTERNATIVE STRATEGIES TO TEST:
1. ATR-BASED: Dynamic targets based on stock volatility
   - Low volatility stocks: Tighter targets (±1.5%)
   - High volatility stocks: Wider targets (±3.0%)

2. ML-CONFIDENCE: Adjust targets based on ML probability
   - High confidence (>70%): Wider targets (±3.0%, hold longer)
   - Low confidence (30-50%): Tighter targets (±1.5%, exit faster)

3. TRAILING STOP: Let winners run, cut losers quickly
   - Initial stop: -2%
   - Trail profit by 0.5% below high
   - Max hold: 5 days

4. MOMENTUM-BASED: Exit when momentum reverses
   - Exit when RSI crosses back above 50 (oversold → neutral)
   - Or when price momentum stalls

5. MEAN REVERSION COMPLETE: Exit when z-score returns to 0
   - Target: Return to mean (z-score = 0)
   - Stop: -2% or 3 days

6. HYBRID: Best elements from above
   - ATR-based initial targets
   - ML confidence for position sizing
   - Trailing stop for winners

ALGORITHM:
For each strategy:
1. Load all historical signals (2020-2025) from experiments
2. Re-backtest with each exit strategy
3. Calculate: win rate, avg return, Sharpe ratio, max drawdown
4. Compare to baseline time-decay
5. Select best performing strategy

SUCCESS CRITERIA:
- Win rate improvement: +2pp or more
- Avg return improvement: +0.5% or more per trade
- Sharpe ratio improvement: +0.3 or more
- Strategy must be robust (work across different stocks/periods)

EXPECTED RESULTS:
- Best strategy: ATR-based or Hybrid (accounts for volatility)
- Avg return improvement: +0.5-1.0% per trade
- Win rate improvement: +2-4pp
- Annual impact: +140-280% cumulative return boost
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

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.models.trading.mean_reversion import MeanReversionDetector
from common.config.mean_reversion_params import get_all_tickers


class ExitStrategyBacktester:
    """
    Backtest different exit strategies on historical signals.
    """

    def __init__(self, strategy: str = 'time_decay'):
        """
        Initialize backtester with specific exit strategy.

        Args:
            strategy: Exit strategy to use
                - 'time_decay': Current baseline (EXP-010)
                - 'atr_based': Volatility-adjusted targets
                - 'ml_confidence': ML probability-based targets
                - 'trailing_stop': Trail winners, cut losers
                - 'momentum': Exit on momentum reversal
                - 'mean_reversion': Exit at z-score = 0
                - 'hybrid': Combination of best elements
        """
        self.strategy = strategy

    def calculate_exit(self, entry_data: Dict, price_data: pd.DataFrame) -> Dict:
        """
        Calculate exit based on selected strategy.

        Args:
            entry_data: Entry signal data (price, date, ML prob, etc.)
            price_data: Future price data for calculating exit

        Returns:
            Exit details (price, date, return, reason)
        """
        entry_price = entry_data['entry_price']
        entry_date = entry_data['entry_date']

        # Get future data (up to max hold period)
        max_hold = 5 if self.strategy == 'trailing_stop' else 3
        future_data = price_data.loc[entry_date:].iloc[1:max_hold+1]

        if len(future_data) == 0:
            return None

        if self.strategy == 'time_decay':
            return self._exit_time_decay(entry_price, future_data)
        elif self.strategy == 'atr_based':
            return self._exit_atr_based(entry_price, entry_data, future_data)
        elif self.strategy == 'ml_confidence':
            return self._exit_ml_confidence(entry_price, entry_data, future_data)
        elif self.strategy == 'trailing_stop':
            return self._exit_trailing_stop(entry_price, future_data)
        elif self.strategy == 'momentum':
            return self._exit_momentum(entry_price, future_data)
        elif self.strategy == 'mean_reversion':
            return self._exit_mean_reversion(entry_price, entry_data, future_data)
        elif self.strategy == 'hybrid':
            return self._exit_hybrid(entry_price, entry_data, future_data)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _exit_time_decay(self, entry_price: float, future_data: pd.DataFrame) -> Dict:
        """Current baseline: Time-decay targets."""
        for i, (date, row) in enumerate(future_data.iterrows()):
            # Day-based targets
            if i == 0:  # Day 0
                profit_target = entry_price * 1.02
                stop_loss = entry_price * 0.98
            elif i == 1:  # Day 1
                profit_target = entry_price * 1.015
                stop_loss = entry_price * 0.985
            else:  # Day 2+
                profit_target = entry_price * 1.01
                stop_loss = entry_price * 0.99

            # Check exit conditions
            if row['High'] >= profit_target:
                return {
                    'exit_price': profit_target,
                    'exit_date': date,
                    'days_held': i + 1,
                    'return_pct': ((profit_target - entry_price) / entry_price) * 100,
                    'reason': 'PROFIT_TARGET'
                }
            elif row['Low'] <= stop_loss:
                return {
                    'exit_price': stop_loss,
                    'exit_date': date,
                    'days_held': i + 1,
                    'return_pct': ((stop_loss - entry_price) / entry_price) * 100,
                    'reason': 'STOP_LOSS'
                }

        # Max hold exit
        exit_price = future_data.iloc[-1]['Close']
        return {
            'exit_price': exit_price,
            'exit_date': future_data.index[-1],
            'days_held': len(future_data),
            'return_pct': ((exit_price - entry_price) / entry_price) * 100,
            'reason': 'MAX_HOLD'
        }

    def _exit_atr_based(self, entry_price: float, entry_data: Dict,
                        future_data: pd.DataFrame) -> Dict:
        """ATR-based dynamic targets."""
        # Calculate ATR (14-period)
        atr = entry_data.get('atr_14', 0.02)  # Default to 2% if not available
        atr_pct = (atr / entry_price) * 100

        # Adjust targets based on volatility
        if atr_pct < 1.5:  # Low volatility
            profit_pct = 1.5
            stop_pct = 1.5
        elif atr_pct > 3.0:  # High volatility
            profit_pct = 3.0
            stop_pct = 2.5
        else:  # Medium volatility
            profit_pct = 2.0
            stop_pct = 2.0

        for i, (date, row) in enumerate(future_data.iterrows()):
            profit_target = entry_price * (1 + profit_pct/100)
            stop_loss = entry_price * (1 - stop_pct/100)

            if row['High'] >= profit_target:
                return {
                    'exit_price': profit_target,
                    'exit_date': date,
                    'days_held': i + 1,
                    'return_pct': ((profit_target - entry_price) / entry_price) * 100,
                    'reason': 'PROFIT_TARGET'
                }
            elif row['Low'] <= stop_loss:
                return {
                    'exit_price': stop_loss,
                    'exit_date': date,
                    'days_held': i + 1,
                    'return_pct': ((stop_loss - entry_price) / entry_price) * 100,
                    'reason': 'STOP_LOSS'
                }

        exit_price = future_data.iloc[-1]['Close']
        return {
            'exit_price': exit_price,
            'exit_date': future_data.index[-1],
            'days_held': len(future_data),
            'return_pct': ((exit_price - entry_price) / entry_price) * 100,
            'reason': 'MAX_HOLD'
        }

    def _exit_ml_confidence(self, entry_price: float, entry_data: Dict,
                           future_data: pd.DataFrame) -> Dict:
        """ML confidence-based targets."""
        ml_prob = entry_data.get('ml_probability', 0.5)

        # Adjust targets based on ML confidence
        if ml_prob > 0.7:  # High confidence
            profit_pct = 3.0
            stop_pct = 1.5
            max_hold = 5
        elif ml_prob > 0.5:  # Medium confidence
            profit_pct = 2.0
            stop_pct = 2.0
            max_hold = 3
        else:  # Low confidence (30-50%)
            profit_pct = 1.5
            stop_pct = 2.0
            max_hold = 2

        future_data = future_data.iloc[:max_hold]

        for i, (date, row) in enumerate(future_data.iterrows()):
            profit_target = entry_price * (1 + profit_pct/100)
            stop_loss = entry_price * (1 - stop_pct/100)

            if row['High'] >= profit_target:
                return {
                    'exit_price': profit_target,
                    'exit_date': date,
                    'days_held': i + 1,
                    'return_pct': ((profit_target - entry_price) / entry_price) * 100,
                    'reason': 'PROFIT_TARGET'
                }
            elif row['Low'] <= stop_loss:
                return {
                    'exit_price': stop_loss,
                    'exit_date': date,
                    'days_held': i + 1,
                    'return_pct': ((stop_loss - entry_price) / entry_price) * 100,
                    'reason': 'STOP_LOSS'
                }

        exit_price = future_data.iloc[-1]['Close']
        return {
            'exit_price': exit_price,
            'exit_date': future_data.index[-1],
            'days_held': len(future_data),
            'return_pct': ((exit_price - entry_price) / entry_price) * 100,
            'reason': 'MAX_HOLD'
        }

    def _exit_trailing_stop(self, entry_price: float, future_data: pd.DataFrame) -> Dict:
        """Trailing stop: Let winners run."""
        stop_pct = 2.0  # Initial stop
        trailing_pct = 0.5  # Trail by 0.5% below high

        highest_price = entry_price

        for i, (date, row) in enumerate(future_data.iterrows()):
            # Update highest price
            if row['High'] > highest_price:
                highest_price = row['High']

            # Calculate trailing stop
            if highest_price > entry_price * 1.01:  # In profit
                stop_loss = highest_price * (1 - trailing_pct/100)
            else:  # Not yet profitable
                stop_loss = entry_price * (1 - stop_pct/100)

            # Check stop hit
            if row['Low'] <= stop_loss:
                exit_price = max(stop_loss, row['Low'])
                return {
                    'exit_price': exit_price,
                    'exit_date': date,
                    'days_held': i + 1,
                    'return_pct': ((exit_price - entry_price) / entry_price) * 100,
                    'reason': 'TRAILING_STOP'
                }

        exit_price = future_data.iloc[-1]['Close']
        return {
            'exit_price': exit_price,
            'exit_date': future_data.index[-1],
            'days_held': len(future_data),
            'return_pct': ((exit_price - entry_price) / entry_price) * 100,
            'reason': 'MAX_HOLD'
        }

    def _exit_momentum(self, entry_price: float, future_data: pd.DataFrame) -> Dict:
        """Exit when RSI crosses back above 50 (momentum reversal)."""
        for i, (date, row) in enumerate(future_data.iterrows()):
            # Check profit target first (cap upside)
            profit_target = entry_price * 1.03
            stop_loss = entry_price * 0.98

            if row['High'] >= profit_target:
                return {
                    'exit_price': profit_target,
                    'exit_date': date,
                    'days_held': i + 1,
                    'return_pct': ((profit_target - entry_price) / entry_price) * 100,
                    'reason': 'PROFIT_TARGET'
                }
            elif row['Low'] <= stop_loss:
                return {
                    'exit_price': stop_loss,
                    'exit_date': date,
                    'days_held': i + 1,
                    'return_pct': ((stop_loss - entry_price) / entry_price) * 100,
                    'reason': 'STOP_LOSS'
                }

            # Check RSI reversal (if available)
            if 'rsi_14' in row and row['rsi_14'] > 50:
                exit_price = row['Close']
                return {
                    'exit_price': exit_price,
                    'exit_date': date,
                    'days_held': i + 1,
                    'return_pct': ((exit_price - entry_price) / entry_price) * 100,
                    'reason': 'MOMENTUM_REVERSAL'
                }

        exit_price = future_data.iloc[-1]['Close']
        return {
            'exit_price': exit_price,
            'exit_date': future_data.index[-1],
            'days_held': len(future_data),
            'return_pct': ((exit_price - entry_price) / entry_price) * 100,
            'reason': 'MAX_HOLD'
        }

    def _exit_mean_reversion(self, entry_price: float, entry_data: Dict,
                            future_data: pd.DataFrame) -> Dict:
        """Exit when price returns to mean (z-score ≈ 0)."""
        for i, (date, row) in enumerate(future_data.iterrows()):
            # Check stop loss
            stop_loss = entry_price * 0.98
            if row['Low'] <= stop_loss:
                return {
                    'exit_price': stop_loss,
                    'exit_date': date,
                    'days_held': i + 1,
                    'return_pct': ((stop_loss - entry_price) / entry_price) * 100,
                    'reason': 'STOP_LOSS'
                }

            # Check if z-score returned to near 0 (±0.5)
            if 'z_score' in row and abs(row['z_score']) < 0.5:
                exit_price = row['Close']
                return {
                    'exit_price': exit_price,
                    'exit_date': date,
                    'days_held': i + 1,
                    'return_pct': ((exit_price - entry_price) / entry_price) * 100,
                    'reason': 'MEAN_REVERSION_COMPLETE'
                }

        exit_price = future_data.iloc[-1]['Close']
        return {
            'exit_price': exit_price,
            'exit_date': future_data.index[-1],
            'days_held': len(future_data),
            'return_pct': ((exit_price - entry_price) / entry_price) * 100,
            'reason': 'MAX_HOLD'
        }

    def _exit_hybrid(self, entry_price: float, entry_data: Dict,
                    future_data: pd.DataFrame) -> Dict:
        """Hybrid: Best of ATR + ML confidence + trailing."""
        # Get ATR and ML probability
        atr = entry_data.get('atr_14', 0.02)
        atr_pct = (atr / entry_price) * 100
        ml_prob = entry_data.get('ml_probability', 0.5)

        # ATR-based initial targets
        if atr_pct < 1.5:
            initial_profit_pct = 1.5
            initial_stop_pct = 1.5
        elif atr_pct > 3.0:
            initial_profit_pct = 3.0
            initial_stop_pct = 2.5
        else:
            initial_profit_pct = 2.0
            initial_stop_pct = 2.0

        # ML confidence adjustment
        if ml_prob > 0.7:
            initial_profit_pct *= 1.2  # 20% wider targets for high confidence
            max_hold = 5
        else:
            max_hold = 3

        future_data = future_data.iloc[:max_hold]
        highest_price = entry_price

        for i, (date, row) in enumerate(future_data.iterrows()):
            # Update highest
            if row['High'] > highest_price:
                highest_price = row['High']

            # Profit target
            profit_target = entry_price * (1 + initial_profit_pct/100)

            # Trailing stop if in profit
            if highest_price > entry_price * 1.01:
                stop_loss = highest_price * 0.995  # 0.5% trail
            else:
                stop_loss = entry_price * (1 - initial_stop_pct/100)

            if row['High'] >= profit_target:
                return {
                    'exit_price': profit_target,
                    'exit_date': date,
                    'days_held': i + 1,
                    'return_pct': ((profit_target - entry_price) / entry_price) * 100,
                    'reason': 'PROFIT_TARGET'
                }
            elif row['Low'] <= stop_loss:
                exit_price = max(stop_loss, row['Low'])
                return {
                    'exit_price': exit_price,
                    'exit_date': date,
                    'days_held': i + 1,
                    'return_pct': ((exit_price - entry_price) / entry_price) * 100,
                    'reason': 'STOP_LOSS'
                }

        exit_price = future_data.iloc[-1]['Close']
        return {
            'exit_price': exit_price,
            'exit_date': future_data.index[-1],
            'days_held': len(future_data),
            'return_pct': ((exit_price - entry_price) / entry_price) * 100,
            'reason': 'MAX_HOLD'
        }


def test_exit_strategy_stock(ticker: str, strategy: str,
                             start_date: str = '2020-01-01',
                             end_date: str = '2025-11-17') -> Dict:
    """
    Test exit strategy on a single stock's historical signals.

    Args:
        ticker: Stock ticker
        strategy: Exit strategy to test
        start_date: Backtest start
        end_date: Backtest end

    Returns:
        Performance metrics dict
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

        # Detect signals (using stock's optimized params)
        from common.config.mean_reversion_params import MEAN_REVERSION_PARAMS
        params = MEAN_REVERSION_PARAMS.get(ticker, {
            'z_score_threshold': 1.5,
            'rsi_oversold': 35,
            'volume_multiplier': 1.3
        })

        detector = MeanReversionDetector(
            z_score_threshold=params['z_score_threshold'],
            rsi_oversold=params['rsi_oversold'],
            volume_multiplier=params['volume_multiplier']
        )

        signals = detector.detect_overcorrections(enriched_data)

        if len(signals) == 0:
            return None

        # Backtest with specified exit strategy
        backtester = ExitStrategyBacktester(strategy=strategy)

        trades = []
        for signal_date in signals.index:
            entry_data = {
                'entry_price': signals.loc[signal_date, 'Open'],
                'entry_date': signal_date,
                'ml_probability': signals.loc[signal_date].get('ml_probability', 0.5),
                'atr_14': enriched_data.loc[signal_date].get('atr_14', 0.02),
                'z_score': signals.loc[signal_date].get('z_score', -2.0),
                'rsi': signals.loc[signal_date].get('rsi_14', 30)
            }

            exit_result = backtester.calculate_exit(entry_data, enriched_data)

            if exit_result:
                trades.append({
                    'entry_date': signal_date,
                    'exit_date': exit_result['exit_date'],
                    'return_pct': exit_result['return_pct'],
                    'days_held': exit_result['days_held'],
                    'reason': exit_result['reason'],
                    'outcome': 'win' if exit_result['return_pct'] > 0 else 'loss'
                })

        if not trades:
            return None

        # Calculate metrics
        wins = [t for t in trades if t['outcome'] == 'win']
        losses = [t for t in trades if t['outcome'] == 'loss']
        returns = [t['return_pct'] for t in trades]

        return {
            'ticker': ticker,
            'strategy': strategy,
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': (len(wins) / len(trades)) * 100,
            'avg_return': np.mean(returns),
            'avg_win': np.mean([t['return_pct'] for t in wins]) if wins else 0,
            'avg_loss': np.mean([t['return_pct'] for t in losses]) if losses else 0,
            'avg_hold_days': np.mean([t['days_held'] for t in trades]),
            'sharpe_ratio': (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'trades': trades
        }

    except Exception as e:
        print(f"[ERROR] {ticker} ({strategy}): {e}")
        return None


def run_exp074_exit_strategy_optimization():
    """
    Test all exit strategies across portfolio and identify best performer.
    """
    print("="*70)
    print("EXP-074: EXIT STRATEGY OPTIMIZATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("OBJECTIVE: Find optimal exit strategy for maximum ROI")
    print("CURRENT: Time-decay (±2%, ±1.5%, ±1%) from EXP-010 (2020)")
    print("TARGET: +0.5% avg return improvement per trade")
    print()

    strategies = [
        'time_decay',      # Baseline
        'atr_based',       # Volatility-adjusted
        'ml_confidence',   # ML probability-based
        'trailing_stop',   # Let winners run
        'momentum',        # RSI reversal
        'mean_reversion',  # Z-score return to 0
        'hybrid'           # Best combination
    ]

    # Test on 10 representative stocks (diverse performance profiles)
    test_stocks = ['NVDA', 'V', 'AVGO', 'TXN', 'JPM', 'ABBV', 'META', 'CMCSA', 'SHW', 'CAT']

    print(f"Testing {len(strategies)} exit strategies on {len(test_stocks)} stocks...")
    print()

    results = {strategy: [] for strategy in strategies}

    for ticker in test_stocks:
        print(f"Testing {ticker}...")
        for strategy in strategies:
            result = test_exit_strategy_stock(ticker, strategy)
            if result:
                results[strategy].append(result)
                print(f"  {strategy:20s}: WR={result['win_rate']:.1f}%, Avg={result['avg_return']:+.2f}%")

    # Calculate aggregate statistics
    print("\n" + "="*70)
    print("AGGREGATE RESULTS ACROSS ALL STOCKS")
    print("="*70)
    print()

    print(f"{'Strategy':<20} {'Trades':<8} {'Win Rate':<12} {'Avg Return':<12} {'Sharpe':<10} {'Improvement'}")
    print("-"*70)

    baseline_avg_return = None
    best_strategy = None
    best_improvement = -999

    for strategy in strategies:
        if not results[strategy]:
            continue

        total_trades = sum(r['total_trades'] for r in results[strategy])
        total_wins = sum(r['wins'] for r in results[strategy])
        all_returns = []
        for r in results[strategy]:
            all_returns.extend([t['return_pct'] for t in r['trades']])

        avg_return = np.mean(all_returns)
        win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
        sharpe = (np.mean(all_returns) / np.std(all_returns)) * np.sqrt(252) if len(all_returns) > 0 and np.std(all_returns) > 0 else 0

        if strategy == 'time_decay':
            baseline_avg_return = avg_return
            improvement = 0.0
        else:
            improvement = avg_return - baseline_avg_return if baseline_avg_return else 0
            if improvement > best_improvement:
                best_improvement = improvement
                best_strategy = strategy

        print(f"{strategy:<20} {total_trades:<8} {win_rate:<11.1f}% {avg_return:<11.2f}% {sharpe:<9.2f} {improvement:+.2f}%")

    # Recommendation
    print("\n" + "="*70)
    print("DEPLOYMENT RECOMMENDATION")
    print("="*70)
    print()

    if best_improvement >= 0.3:
        print(f"[SUCCESS] {best_strategy.upper()} shows +{best_improvement:.2f}% improvement!")
        print()
        print(f"ANNUAL IMPACT: ~281 trades/year × {best_improvement:.2f}% = +{281 * best_improvement:.1f}% cumulative boost")
        print()
        print("NEXT STEPS:")
        print(f"  1. Validate {best_strategy} on full 54-stock portfolio")
        print("  2. Update mean_reversion.py exit logic")
        print("  3. Deploy to production")
    else:
        print(f"[MARGINAL] Best strategy ({best_strategy}) shows only +{best_improvement:.2f}% improvement")
        print()
        print("Current time-decay strategy remains competitive.")
        print("Consider testing additional hybrid variations.")

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-074',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'objective': 'Optimize exit strategy for maximum returns',
        'strategies_tested': strategies,
        'test_stocks': test_stocks,
        'best_strategy': best_strategy,
        'improvement': float(best_improvement),
        'baseline_avg_return': float(baseline_avg_return),
        'results_by_strategy': {
            strategy: {
                'total_trades': sum(r['total_trades'] for r in results[strategy]),
                'win_rate': (sum(r['wins'] for r in results[strategy]) / sum(r['total_trades'] for r in results[strategy])) * 100 if results[strategy] else 0,
                'avg_return': float(np.mean([t['return_pct'] for r in results[strategy] for t in r['trades']])) if results[strategy] else 0
            }
            for strategy in strategies if results[strategy]
        }
    }

    results_file = os.path.join(results_dir, 'exp074_exit_strategy_optimization.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_file}")

    # Send email
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending exit strategy optimization report...")
            notifier.send_experiment_report('EXP-074', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-074: Exit strategy optimization."""

    print("\n[EXIT STRATEGY OPTIMIZATION] Testing 7 strategies for maximum ROI")
    print("Current: Time-decay (2020). Target: +0.5% per trade improvement")
    print()

    results = run_exp074_exit_strategy_optimization()

    print("\n" + "="*70)
    print("EXIT STRATEGY OPTIMIZATION COMPLETE")
    print("="*70)

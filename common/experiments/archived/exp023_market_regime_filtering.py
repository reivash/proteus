"""
EXPERIMENT: EXP-023
Date: 2025-11-16
Objective: Filter panic sell signals by market regime to improve win rate

RESEARCH MOTIVATION:
Current Tier A portfolio: 78.7% win rate (excellent but room for improvement)
Analysis shows ~21% of trades are losers.

Hypothesis: Many losses occur during bear markets when panic sells continue falling
instead of reversing (not true mean reversion opportunities).

Research shows:
- Mean reversion works BEST in bull markets (overshoots get corrected)
- Mean reversion works POORLY in bear markets (panic is justified, continues)
- VIX >30 indicates fear-driven selling that persists

HYPOTHESIS:
Filtering out panic sells during BEAR market regimes will:
- Improve win rate from 78.7% → 85-90%
- Reduce losing trades by 40-60%
- Slightly reduce trade frequency (~20% fewer signals)
- Net positive on risk-adjusted returns

METHOD:
MARKET REGIME CLASSIFICATION:

BULL (TRADE):
  - SPY > 50-day MA
  - VIX < 25
  - Favorable for mean reversion

NEUTRAL (TRADE):
  - Mixed signals
  - VIX 25-30
  - Still acceptable for trading

BEAR (SKIP):
  - SPY < 50-day MA OR
  - VIX > 30
  - Panic sells likely continue (not mean reversion)

COMPARISON:
- Baseline: Trade all Tier A signals (current v7.0)
- Filtered: Skip signals during BEAR regime only

EXPECTED OUTCOMES:
- Win rate: +7-12pp improvement (78.7% → 85-90%)
- Trade frequency: -20% (skip bear market signals)
- Losing trades: -40-60% reduction
- Sharpe ratio: +20-30% improvement
- Total return: Similar or better (fewer losses offset by fewer trades)

SYMBOLS TO TEST:
All 5 Tier A stocks: NVDA, AVGO, V, MA, AXP
"""

import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.models.trading.mean_reversion import MeanReversionDetector
from common.config.mean_reversion_params import get_params


class MarketRegimeDetector:
    """Detect market regime (BULL/NEUTRAL/BEAR) for filtering signals."""

    def __init__(self):
        """Initialize regime detector."""
        self.fetcher = YahooFinanceFetcher()

    def get_regime(self, date: pd.Timestamp, spy_data: pd.DataFrame, vix_data: pd.DataFrame) -> str:
        """
        Determine market regime for a given date.

        Args:
            date: Date to check
            spy_data: SPY OHLCV data (must include 50-day MA)
            vix_data: VIX data

        Returns:
            'BULL', 'NEUTRAL', or 'BEAR'
        """
        try:
            # Get SPY data for this date
            spy_row = spy_data[spy_data['Date'] == date]
            if spy_row.empty:
                return 'NEUTRAL'  # Default if no data

            spy_close = spy_row.iloc[0]['Close']
            spy_ma50 = spy_row.iloc[0]['MA_50']

            # Get VIX for this date
            vix_row = vix_data[vix_data['Date'] == date]
            if vix_row.empty:
                vix = 20  # Default moderate VIX
            else:
                vix = vix_row.iloc[0]['Close']

            # BEAR MARKET: SPY below MA50 OR VIX > 30
            if spy_close < spy_ma50 or vix > 30:
                return 'BEAR'

            # BULL MARKET: SPY above MA50 AND VIX < 25
            elif spy_close > spy_ma50 and vix < 25:
                return 'BULL'

            # NEUTRAL: Everything else
            else:
                return 'NEUTRAL'

        except Exception as e:
            print(f"  [WARNING] Regime detection error for {date}: {e}")
            return 'NEUTRAL'  # Safe default


class RegimeFilteredBacktester:
    """Backtest with optional market regime filtering."""

    def __init__(self,
                 initial_capital: float = 10000,
                 filter_bear_market: bool = True):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            filter_bear_market: Skip signals during BEAR regime (True) or trade all (False)
        """
        self.initial_capital = initial_capital
        self.filter_bear_market = filter_bear_market

        # Time-decay exit targets
        self.time_decay_targets = {
            0: (2.0, -2.0),
            1: (1.5, -1.5),
            2: (1.0, -1.0),
        }
        self.max_hold_days = 7

        self.regime_detector = MarketRegimeDetector()

    def backtest(self,
                signals_df: pd.DataFrame,
                price_data: pd.DataFrame,
                spy_data: pd.DataFrame,
                vix_data: pd.DataFrame) -> Dict:
        """
        Run backtest with optional regime filtering.

        Args:
            signals_df: DataFrame with panic_sell signals
            price_data: Stock OHLCV data
            spy_data: SPY data with 50-day MA
            vix_data: VIX data

        Returns:
            Dictionary with backtest results
        """
        capital = self.initial_capital
        trades = []
        skipped_signals = []

        # Get panic sell signals
        panic_sells = signals_df[signals_df['panic_sell'] == True].copy()
        price_data_indexed = price_data.set_index('Date')

        for idx, signal_row in panic_sells.iterrows():
            entry_date = signal_row['Date']
            entry_price = signal_row['Close']

            # Check market regime
            regime = self.regime_detector.get_regime(entry_date, spy_data, vix_data)

            # Filter out BEAR market signals if enabled
            if self.filter_bear_market and regime == 'BEAR':
                skipped_signals.append({
                    'date': entry_date,
                    'price': entry_price,
                    'regime': regime,
                    'reason': 'Bear market - skipped'
                })
                continue

            # Execute trade
            try:
                entry_idx = price_data_indexed.index.get_loc(entry_date)
            except KeyError:
                continue

            # Find exit using time-decay targets
            exit_info = None
            for days_held in range(self.max_hold_days):
                current_idx = entry_idx + days_held + 1

                if current_idx >= len(price_data_indexed):
                    break

                current_price = price_data_indexed.iloc[current_idx]['Close']
                return_pct = ((current_price / entry_price) - 1) * 100

                # Check time-decay targets
                profit_target, stop_loss = self.time_decay_targets.get(
                    min(days_held, 2),
                    (1.0, -1.0)
                )

                if return_pct >= profit_target or return_pct <= stop_loss:
                    # Exit trade
                    shares = int(capital / entry_price)
                    profit = (current_price - entry_price) * shares
                    capital += profit

                    exit_info = {
                        'entry_date': entry_date,
                        'entry_regime': regime,
                        'exit_date': price_data_indexed.index[current_idx],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'return_pct': return_pct,
                        'days_held': days_held,
                        'exit_reason': 'profit' if return_pct > 0 else 'loss'
                    }
                    trades.append(exit_info)
                    break

            if exit_info is None:
                # Timeout exit
                timeout_idx = min(entry_idx + self.max_hold_days, len(price_data_indexed) - 1)
                current_price = price_data_indexed.iloc[timeout_idx]['Close']
                return_pct = ((current_price / entry_price) - 1) * 100

                shares = int(capital / entry_price)
                profit = (current_price - entry_price) * shares
                capital += profit

                trades.append({
                    'entry_date': entry_date,
                    'entry_regime': regime,
                    'exit_date': price_data_indexed.index[timeout_idx],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'return_pct': return_pct,
                    'days_held': self.max_hold_days - 1,
                    'exit_reason': 'timeout'
                })

        # Calculate metrics
        results = self._calculate_metrics(trades, capital)
        results['skipped_signals'] = len(skipped_signals)
        results['total_signals'] = len(panic_sells)
        results['trade_rate'] = (len(trades) / len(panic_sells) * 100) if len(panic_sells) > 0 else 0

        # Regime breakdown
        if trades:
            trades_df = pd.DataFrame(trades)
            regime_counts = trades_df['entry_regime'].value_counts().to_dict()
            results['regime_breakdown'] = regime_counts
        else:
            results['regime_breakdown'] = {}

        return results

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
                'sharpe_ratio': 0,
                'max_drawdown': 0
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

        # Risk metrics
        returns = trades_df['return_pct'].values
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Max drawdown
        cumulative_returns = (1 + trades_df['return_pct'] / 100).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_gain': avg_gain,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }


def run_exp023_regime_filtering(ticker: str = "NVDA", period: str = "3y"):
    """
    Compare baseline vs regime-filtered trading.

    Args:
        ticker: Stock ticker (Tier A)
        period: Backtest period
    """
    print("=" * 70)
    print("EXP-023: MARKET REGIME FILTERING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    initial_capital = 10000
    params = get_params(ticker)

    print(f"Configuration:")
    print(f"  Ticker: {ticker}")
    print(f"  Period: {period}")
    print(f"  Initial Capital: ${initial_capital:,}")
    print(f"  Strategy: Mean Reversion v7.0 + Regime Filter")
    print()

    # Fetch price data
    print("[1/6] Fetching price data...")
    fetcher = YahooFinanceFetcher()
    price_data = fetcher.fetch_stock_data(ticker, period=period)
    print(f"      [OK] {len(price_data)} rows")

    # Fetch SPY data for regime detection
    print("\n[2/6] Fetching SPY data for regime detection...")
    spy_data = fetcher.fetch_stock_data('SPY', period=period)
    spy_data['MA_50'] = spy_data['Close'].rolling(50).mean()
    print(f"      [OK] {len(spy_data)} rows")

    # Fetch VIX data
    print("\n[3/6] Fetching VIX data...")
    vix_data = fetcher.fetch_stock_data('^VIX', period=period)
    print(f"      [OK] {len(vix_data)} rows")

    # Add technical indicators
    print("\n[4/6] Calculating technical indicators...")
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(price_data)
    print(f"      [OK] Technical indicators added")

    # Detect panic sell signals
    print("\n[5/6] Detecting panic sell signals...")
    detector = MeanReversionDetector(
        z_score_threshold=params['z_score_threshold'],
        rsi_oversold=params['rsi_oversold'],
        volume_multiplier=params['volume_multiplier'],
        price_drop_threshold=params['price_drop_threshold']
    )
    signals = detector.detect_overcorrections(enriched_data)
    num_signals = signals['panic_sell'].sum()
    print(f"      [OK] {num_signals} panic sell signals detected")

    # Run backtests
    print("\n[6/6] Running backtests...")
    print()

    # BASELINE: No regime filter
    print("  [A] BASELINE (No regime filter)...")
    baseline_backtester = RegimeFilteredBacktester(
        initial_capital=initial_capital,
        filter_bear_market=False
    )
    baseline_results = baseline_backtester.backtest(signals, enriched_data, spy_data, vix_data)
    print(f"      [OK] {baseline_results['total_trades']} trades, {baseline_results['win_rate']:.1f}% win rate")

    # REGIME-FILTERED: Skip BEAR market signals
    print("\n  [B] REGIME-FILTERED (Skip BEAR market)...")
    filtered_backtester = RegimeFilteredBacktester(
        initial_capital=initial_capital,
        filter_bear_market=True
    )
    filtered_results = filtered_backtester.backtest(signals, enriched_data, spy_data, vix_data)
    print(f"      [OK] {filtered_results['total_trades']} trades, {filtered_results['win_rate']:.1f}% win rate")
    print(f"      Skipped: {filtered_results['skipped_signals']} BEAR market signals")

    # Display comparison
    print()
    print("=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print()

    print(f"{'Metric':<30} | {'Baseline':>15} | {'Filtered':>15} | {'Improvement':>12}")
    print("-" * 75)

    metrics = [
        ('Total Signals', 'total_signals', ''),
        ('Trades Executed', 'total_trades', ''),
        ('Signals Skipped', 'skipped_signals', ''),
        ('Trade Rate', 'trade_rate', '%'),
        ('Win Rate', 'win_rate', '%'),
        ('Total Return', 'total_return', '%'),
        ('Avg Gain (Winners)', 'avg_gain', '%'),
        ('Avg Loss (Losers)', 'avg_loss', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Max Drawdown', 'max_drawdown', '%')
    ]

    for name, key, unit in metrics:
        baseline_val = baseline_results.get(key, 0)
        filtered_val = filtered_results.get(key, 0)

        if key in ['total_signals', 'total_trades', 'skipped_signals']:
            improvement = f"{filtered_val - baseline_val:+.0f}"
        elif key in ['win_rate', 'total_return', 'avg_gain', 'avg_loss', 'max_drawdown', 'trade_rate']:
            improvement = f"{filtered_val - baseline_val:+.2f}pp"
        else:
            improvement = f"{filtered_val - baseline_val:+.2f}"

        print(f"{name:<30} | {baseline_val:>15.2f}{unit} | {filtered_val:>15.2f}{unit} | {improvement:>12}")

    print()
    print("=" * 70)

    # Analysis
    win_rate_improvement = filtered_results['win_rate'] - baseline_results['win_rate']
    return_improvement = filtered_results['total_return'] - baseline_results['total_return']

    print()
    if win_rate_improvement > 5:
        print("[SUCCESS] Regime filtering significantly improved performance!")
        print(f"  Win rate improvement: +{win_rate_improvement:.2f}pp")
        print(f"  Return improvement: +{return_improvement:.2f}pp")
        print(f"  Signals skipped: {filtered_results['skipped_signals']} BEAR market signals")
        print()
        print("  RECOMMENDATION: Deploy regime filtering to production!")
    elif win_rate_improvement > 0:
        print("[PARTIAL SUCCESS] Regime filtering showed improvement")
        print(f"  Win rate improvement: +{win_rate_improvement:.2f}pp")
        print(f"  Consider deploying based on risk tolerance")
    else:
        print("[NEUTRAL] Regime filtering did not improve performance")
        print(f"  Possible reasons:")
        print(f"  - Tier A stocks work in all regimes")
        print(f"  - Bear market signals were actually profitable")
        print(f"  - Sample size too small to detect benefit")

    print("=" * 70)

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    results_data = {
        'experiment_id': 'EXP-023',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ticker': ticker,
        'period': period,
        'baseline': {k: v for k, v in baseline_results.items()},
        'filtered': {k: v for k, v in filtered_results.items()},
        'improvement': {
            'win_rate_pp': win_rate_improvement,
            'return_pp': return_improvement,
            'signals_skipped': filtered_results['skipped_signals']
        }
    }

    results_file = os.path.join(results_dir, f'exp023_{ticker.lower()}_regime_filtering.json')
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    return results_data


if __name__ == '__main__':
    """Run EXP-023 on all Tier A stocks."""

    tier_a_stocks = ['NVDA', 'AVGO', 'V', 'MA', 'AXP']

    print("Testing regime filtering on all Tier A stocks...")
    print()

    all_results = {}

    for ticker in tier_a_stocks:
        print(f"\n{'='*70}")
        print(f"Testing {ticker}")
        print(f"{'='*70}\n")

        results = run_exp023_regime_filtering(ticker=ticker, period='3y')
        all_results[ticker] = results

    # Summary across all stocks
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL TIER A STOCKS")
    print("=" * 70)
    print()

    for ticker, result in all_results.items():
        improvement = result['improvement']
        print(f"{ticker}:")
        print(f"  Win rate improvement: {improvement['win_rate_pp']:+.2f}pp")
        print(f"  Signals skipped: {improvement['signals_skipped']}")
        print()

    print("\nNext steps:")
    print("1. If improvement >5pp: Deploy regime filtering")
    print("2. Test on additional time periods for validation")
    print("3. Integrate with signal scanner for live trading")

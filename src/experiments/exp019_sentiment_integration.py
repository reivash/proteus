"""
EXPERIMENT: EXP-019
Date: 2025-11-16
Objective: Integrate real-time sentiment to filter panic sell signals

RESEARCH MOTIVATION:
EXP-014 showed some stocks have low win rates despite panic sell detection:
- TSLA: 46.7% win rate (-29.84% return)
- AAPL: 50.0% win rate (coin flip)
- Even Tier A stocks have 12-25% losing trades

Root cause: Not all panic sells are equal
- Emotion-driven panics: Overreaction, mean revert (HIGH win rate)
- Information-driven panics: Real news, continue falling (LOW win rate)

EXP-011 built sentiment analysis to detect the difference, but never integrated it.

HYPOTHESIS:
By filtering panic sell signals using real-time sentiment:
- Skip LOW confidence (information-driven, news negative)
- Trade HIGH confidence (emotion-driven, news neutral/positive)
- Win rate improvement: 78% → 85%+ on Tier A stocks
- Reduce losing trades by 40-60%

METHOD:
BASELINE (Current v6.0):
  - Panic sell detection (Z-score, RSI, volume, price drop)
  - Trade ALL panic sells on Tier A stocks
  - Win rate: 78.0% (NVDA: 87.5%, V: 75%, MA: 71.4%)

EXP-019 (Sentiment-Filtered):
  - Same panic sell detection
  - Collect real-time sentiment (News API, Twitter if available)
  - Classify: HIGH (trade), MEDIUM (trade), LOW (skip)
  - Expected win rate: 85%+

SENTIMENT CLASSIFICATION:
HIGH Confidence (TRADE):
- Social sentiment very negative (<-0.3)
- News sentiment neutral/positive (>-0.1)
- Divergence: Social panic, fundamentals OK
- Example: Elon tweets cause NVDA drop (emotion)

MEDIUM Confidence (TRADE):
- Moderate social negativity
- Mixed news sentiment
- Unclear if emotion or information

LOW Confidence (SKIP):
- Both social AND news very negative (<-0.3)
- Convergence: Real bad news
- Example: Earnings miss, guidance down (information)

EXPECTED OUTCOMES:
- Win rate: +7-10pp improvement (78% → 85-88%)
- Losing trades: -40% reduction (skip bad signals)
- Total return: +5-10pp improvement
- Trade frequency: -20% (skip LOW confidence)

COMPARISON METRICS:
- Win rate: Baseline vs Sentiment-filtered
- Return per trade
- Number of skipped trades
- False positive rate (skipped a winner)
- False negative rate (traded a loser)
"""

import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.models.trading.mean_reversion import MeanReversionDetector
from src.config.mean_reversion_params import get_params
from src.data.sentiment.news_collector import NewsCollector, aggregate_daily_news


class SentimentIntegratedBacktester:
    """Backtest mean reversion with sentiment-based signal filtering."""

    def __init__(self,
                 initial_capital: float = 10000,
                 use_sentiment_filter: bool = True):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            use_sentiment_filter: Filter signals by sentiment (True) or trade all (False)
        """
        self.initial_capital = initial_capital
        self.use_sentiment_filter = use_sentiment_filter

        # Time-decay exit targets (same as v6.0)
        self.time_decay_targets = {
            0: (2.0, -2.0),   # Day 0: ±2%
            1: (1.5, -1.5),   # Day 1: ±1.5%
            2: (1.0, -1.0),   # Day 2+: ±1%
        }
        self.max_hold_days = 7

        # Sentiment thresholds
        self.high_confidence_threshold = -0.3  # Social very negative
        self.news_neutral_threshold = -0.1     # News not too negative

    def collect_sentiment_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Collect sentiment data for date range.

        Args:
            ticker: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with date, news_sentiment
        """
        print(f"  Collecting sentiment data for {ticker}...")

        try:
            # Use News API (most reliable for historical data)
            collector = NewsCollector()
            news = collector.collect_for_date_range(ticker, start_date, end_date)

            if news.empty:
                print(f"  [WARNING] No news data for {ticker}, using neutral sentiment")
                return self._create_neutral_sentiment(start_date, end_date)

            # Aggregate to daily
            daily_news = aggregate_daily_news(news)

            sentiment_data = pd.DataFrame({
                'date': pd.to_datetime(daily_news['date']),
                'news_sentiment': daily_news['news_sentiment'],
                'news_relevance': daily_news.get('avg_relevance', 0.5),
                'article_count': daily_news.get('article_count', 0)
            })

            print(f"  [OK] {len(sentiment_data)} days of sentiment data")
            return sentiment_data

        except Exception as e:
            print(f"  [ERROR] Sentiment collection failed: {e}")
            return self._create_neutral_sentiment(start_date, end_date)

    def _create_neutral_sentiment(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Create neutral sentiment data as fallback."""
        dates = pd.date_range(start_date, end_date, freq='D')
        return pd.DataFrame({
            'date': dates,
            'news_sentiment': 0.0,  # Neutral
            'news_relevance': 0.5,
            'article_count': 0
        })

    def classify_signal_confidence(self,
                                   social_sentiment: float,
                                   news_sentiment: float) -> str:
        """
        Classify panic sell signal confidence.

        Args:
            social_sentiment: Social media sentiment (-1 to +1)
            news_sentiment: News sentiment (-1 to +1)

        Returns:
            'HIGH', 'MEDIUM', or 'LOW'
        """
        # For this experiment, we only have news sentiment (no historical social data)
        # So we'll use a simplified classification based on news sentiment

        if news_sentiment > 0.1:
            # News is positive despite panic sell = emotion-driven
            return 'HIGH'
        elif news_sentiment > -0.2:
            # News is neutral/slightly negative = unclear
            return 'MEDIUM'
        else:
            # News is very negative = information-driven
            return 'LOW'

    def backtest(self,
                signals_df: pd.DataFrame,
                price_data: pd.DataFrame,
                sentiment_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Run backtest with optional sentiment filtering.

        Args:
            signals_df: DataFrame with panic_sell signals
            price_data: Full OHLCV data
            sentiment_data: Optional sentiment data for filtering

        Returns:
            Dictionary with backtest results
        """
        capital = self.initial_capital
        trades = []
        skipped_trades = []

        # Get panic sell signals
        panic_sells = signals_df[signals_df['panic_sell'] == True].copy()
        price_data_indexed = price_data.set_index('Date')

        print(f"\n[INFO] Backtesting {len(panic_sells)} panic sell signals...")
        print(f"[INFO] Sentiment filter: {'ENABLED' if self.use_sentiment_filter else 'DISABLED'}")

        for idx, signal_row in panic_sells.iterrows():
            entry_date = signal_row['Date']
            entry_price = signal_row['Close']

            # Check sentiment if enabled
            if self.use_sentiment_filter and sentiment_data is not None:
                # Get sentiment for entry date
                sentiment_row = sentiment_data[sentiment_data['date'] == entry_date]

                if not sentiment_row.empty:
                    news_sentiment = sentiment_row.iloc[0]['news_sentiment']
                    social_sentiment = -0.5  # Assume moderate negative (we don't have historical social data)

                    confidence = self.classify_signal_confidence(social_sentiment, news_sentiment)

                    if confidence == 'LOW':
                        # Skip this trade
                        skipped_trades.append({
                            'entry_date': entry_date,
                            'entry_price': entry_price,
                            'confidence': confidence,
                            'news_sentiment': news_sentiment,
                            'reason': 'Low confidence - information-driven panic'
                        })
                        continue

            # Execute trade
            try:
                entry_idx = price_data_indexed.index.get_loc(entry_date)
            except KeyError:
                continue

            # Find exit using time-decay targets
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

                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': price_data_indexed.index[current_idx],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'return_pct': return_pct,
                        'days_held': days_held,
                        'exit_reason': 'profit' if return_pct > 0 else 'loss'
                    })
                    break
            else:
                # Timeout exit
                timeout_idx = min(entry_idx + self.max_hold_days, len(price_data_indexed) - 1)
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
                    'days_held': self.max_hold_days - 1,
                    'exit_reason': 'timeout'
                })

        # Calculate metrics
        results = self._calculate_metrics(trades, capital)
        results['skipped_trades'] = len(skipped_trades)
        results['total_signals'] = len(panic_sells)
        results['trade_rate'] = (len(trades) / len(panic_sells) * 100) if len(panic_sells) > 0 else 0

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
                'avg_hold_days': 0,
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
        avg_hold_days = trades_df['days_held'].mean()

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
            'avg_hold_days': avg_hold_days,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }


def run_exp019_sentiment_integration(ticker: str = "NVDA", period: str = "1y"):
    """
    Compare baseline vs sentiment-filtered trading.

    Args:
        ticker: Stock ticker (Tier A only)
        period: Backtest period (use 1y for News API limit)
    """
    print("=" * 70)
    print("EXP-019: SENTIMENT INTEGRATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    initial_capital = 10000
    params = get_params(ticker)

    print(f"Configuration:")
    print(f"  Ticker: {ticker}")
    print(f"  Period: {period} (News API limitation)")
    print(f"  Initial Capital: ${initial_capital:,}")
    print(f"  Strategy: Mean Reversion v6.0 + Sentiment")
    print()

    # Fetch price data
    print("[1/5] Fetching price data...")
    fetcher = YahooFinanceFetcher()
    price_data = fetcher.fetch_stock_data(ticker, period=period)
    print(f"      [OK] {len(price_data)} rows")

    # Add technical indicators
    print("\n[2/5] Calculating technical indicators...")
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(price_data)
    print(f"      [OK] Technical indicators added")

    # Detect panic sell signals
    print("\n[3/5] Detecting panic sell signals...")
    detector = MeanReversionDetector(
        z_score_threshold=params['z_score_threshold'],
        rsi_oversold=params['rsi_oversold'],
        volume_multiplier=params['volume_multiplier'],
        price_drop_threshold=params['price_drop_threshold']
    )
    signals = detector.detect_overcorrections(enriched_data)
    num_signals = signals['panic_sell'].sum()
    print(f"      [OK] {num_signals} panic sell signals detected")

    if num_signals == 0:
        print("\n[SKIP] No panic sell signals found in this period")
        return None

    # Collect sentiment data
    print("\n[4/5] Collecting sentiment data...")
    start_date = price_data['Date'].min().strftime('%Y-%m-%d')
    end_date = price_data['Date'].max().strftime('%Y-%m-%d')

    backtester = SentimentIntegratedBacktester(initial_capital=initial_capital)
    sentiment_data = backtester.collect_sentiment_data(ticker, start_date, end_date)

    # Run backtests
    print("\n[5/5] Running backtests...")
    print()

    # BASELINE: No sentiment filter
    print("  [A] BASELINE (No sentiment filter)...")
    baseline_backtester = SentimentIntegratedBacktester(
        initial_capital=initial_capital,
        use_sentiment_filter=False
    )
    baseline_results = baseline_backtester.backtest(signals, enriched_data, None)
    print(f"      [OK] {baseline_results['total_trades']} trades, {baseline_results['win_rate']:.1f}% win rate")

    # SENTIMENT-FILTERED
    print("\n  [B] SENTIMENT-FILTERED...")
    sentiment_backtester = SentimentIntegratedBacktester(
        initial_capital=initial_capital,
        use_sentiment_filter=True
    )
    sentiment_results = sentiment_backtester.backtest(signals, enriched_data, sentiment_data)
    print(f"      [OK] {sentiment_results['total_trades']} trades, {sentiment_results['win_rate']:.1f}% win rate")
    print(f"      Skipped: {sentiment_results['skipped_trades']} LOW confidence signals")

    # Display comparison
    print()
    print("=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print()

    print(f"{'Metric':<30} | {'Baseline':>15} | {'Sentiment':>15} | {'Improvement':>12}")
    print("-" * 75)

    metrics = [
        ('Total Signals', 'total_signals', ''),
        ('Trades Executed', 'total_trades', ''),
        ('Trades Skipped', 'skipped_trades', ''),
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
        sentiment_val = sentiment_results.get(key, 0)

        if key in ['total_signals', 'total_trades', 'skipped_trades']:
            improvement = f"{sentiment_val - baseline_val:+.0f}"
        elif key in ['win_rate', 'total_return', 'avg_gain', 'avg_loss', 'max_drawdown', 'trade_rate']:
            improvement = f"{sentiment_val - baseline_val:+.2f}pp"
        else:
            improvement = f"{sentiment_val - baseline_val:+.2f}"

        print(f"{name:<30} | {baseline_val:>15.2f}{unit} | {sentiment_val:>15.2f}{unit} | {improvement:>12}")

    print()
    print("=" * 70)

    # Analysis
    win_rate_improvement = sentiment_results['win_rate'] - baseline_results['win_rate']
    return_improvement = sentiment_results['total_return'] - baseline_results['total_return']

    print()
    if win_rate_improvement > 5:
        print("[SUCCESS] Sentiment filtering significantly improved performance!")
        print(f"  Win rate improvement: +{win_rate_improvement:.2f}pp")
        print(f"  Return improvement: +{return_improvement:.2f}pp")
        print(f"  Trades skipped: {sentiment_results['skipped_trades']} LOW confidence signals")
        print()
        print("  RECOMMENDATION: Deploy sentiment filtering to production!")
    elif win_rate_improvement > 0:
        print("[PARTIAL SUCCESS] Sentiment filtering showed improvement")
        print(f"  Win rate improvement: +{win_rate_improvement:.2f}pp")
        print(f"  May need more data or tuning")
    else:
        print("[NEUTRAL] Sentiment filtering did not improve performance")
        print(f"  Possible reasons:")
        print(f"  - Limited historical sentiment data (News API only)")
        print(f"  - Stock already has high win rate (sentiment filtering redundant)")
        print(f"  - Need social media data (Twitter/Reddit) for better signals")

    print("=" * 70)

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    results_data = {
        'experiment_id': 'EXP-019',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ticker': ticker,
        'period': period,
        'baseline': {k: v for k, v in baseline_results.items()},
        'sentiment_filtered': {k: v for k, v in sentiment_results.items()},
        'improvement': {
            'win_rate_pp': win_rate_improvement,
            'return_pp': return_improvement,
            'trades_skipped': sentiment_results['skipped_trades']
        }
    }

    results_file = os.path.join(results_dir, f'exp019_{ticker.lower()}_sentiment_integration.json')
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    return results_data


if __name__ == '__main__':
    """Run EXP-019 sentiment integration."""

    # Test on NVDA first (best performer, 87.5% win rate)
    # Use 1 year period due to News API free tier limitations
    results = run_exp019_sentiment_integration(ticker='NVDA', period='1y')

    print("\n\nNext steps:")
    print("1. Test on V and MA (other Tier A stocks)")
    print("2. Integrate Twitter/Reddit for better social sentiment")
    print("3. Deploy if improvement validated across all Tier A stocks")
    print("4. Monitor production performance vs backtest")

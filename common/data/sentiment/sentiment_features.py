"""
Sentiment Feature Engineering Pipeline

Combines Twitter, Reddit, and News sentiment into trading features.
Uses FinBERT to analyze text, then aggregates into daily signals.

Features created:
1. Social Sentiment: Combined Twitter + Reddit sentiment (-1 to +1)
2. News Sentiment: Professional news sentiment (-1 to +1)
3. Sentiment Divergence: |social - news| (emotion vs information)
4. Sentiment Velocity: Rate of change in sentiment
5. Signal Confidence: HIGH/MEDIUM/LOW based on divergence

Research-based classification:
- HIGH CONFIDENCE: Social negative + News neutral/positive
  → Emotion-driven panic, fundamentals OK (BUY)
- MEDIUM CONFIDENCE: Mixed or divergent signals
  → Uncertain, trade with caution
- LOW CONFIDENCE: Social negative + News negative
  → Information-driven panic, fundamental issue (AVOID)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Import sentiment modules
try:
    from common.data.sentiment.twitter_collector import TwitterSentimentCollector, aggregate_daily_tweets
    from common.data.sentiment.reddit_collector import RedditSentimentCollector, aggregate_daily_posts
    from common.data.sentiment.news_collector import NewsCollector, aggregate_daily_news
    from common.models.sentiment.finbert_analyzer import FinBERTAnalyzer, aggregate_sentiment
    SENTIMENT_MODULES_AVAILABLE = True
except ImportError:
    SENTIMENT_MODULES_AVAILABLE = False


class SentimentFeaturePipeline:
    """Build sentiment features for trading signals."""

    def __init__(self):
        """Initialize sentiment pipeline."""
        if not SENTIMENT_MODULES_AVAILABLE:
            print("[WARN] Sentiment modules not available")
            self.twitter_collector = None
            self.reddit_collector = None
            self.news_collector = None
            self.sentiment_analyzer = None
            return

        # Initialize data collectors
        self.twitter_collector = TwitterSentimentCollector()
        self.reddit_collector = RedditSentimentCollector()
        self.news_collector = NewsCollector()

        # Initialize FinBERT analyzer
        self.sentiment_analyzer = FinBERTAnalyzer()

    def is_ready(self) -> bool:
        """Check if pipeline is ready."""
        return (
            SENTIMENT_MODULES_AVAILABLE and
            self.sentiment_analyzer is not None and
            self.sentiment_analyzer.is_ready()
        )

    def collect_all_sentiment(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Collect sentiment from all sources for a date range.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with daily sentiment features
        """
        if not self.is_ready():
            print(f"[WARN] Sentiment pipeline not ready for {ticker}")
            return pd.DataFrame()

        print(f"[INFO] Collecting sentiment data for {ticker} ({start_date} to {end_date})")

        # 1. Collect Twitter data
        twitter_data = pd.DataFrame()
        if self.twitter_collector and self.twitter_collector.is_enabled():
            print(f"  [1/3] Collecting Twitter data...")
            tweets = self.twitter_collector.collect(ticker, start_date, end_date)
            if not tweets.empty:
                # Analyze sentiment with FinBERT
                tweets = self.sentiment_analyzer.analyze_dataframe(tweets, 'text', 'twitter')
                # Aggregate by day
                twitter_data = aggregate_daily_tweets(tweets)
                print(f"        [OK] {len(twitter_data)} days of Twitter data")
        else:
            print(f"  [1/3] Twitter disabled")

        # 2. Collect Reddit data
        reddit_data = pd.DataFrame()
        if self.reddit_collector and self.reddit_collector.is_enabled():
            print(f"  [2/3] Collecting Reddit data...")
            posts = self.reddit_collector.collect(ticker, start_date, end_date)
            if not posts.empty:
                # Combine title + text for sentiment analysis
                posts['combined_text'] = posts['title'] + ' ' + posts['text']
                posts = self.sentiment_analyzer.analyze_dataframe(posts, 'combined_text', 'reddit')
                # Aggregate by day
                reddit_data = aggregate_daily_posts(posts)
                print(f"        [OK] {len(reddit_data)} days of Reddit data")
        else:
            print(f"  [2/3] Reddit disabled")

        # 3. Collect News data
        news_data = pd.DataFrame()
        if self.news_collector and self.news_collector.is_enabled():
            print(f"  [3/3] Collecting News data...")
            news = self.news_collector.collect_for_date_range(ticker, start_date, end_date)
            if not news.empty:
                # News already has sentiment from Alpha Vantage API
                news_data = aggregate_daily_news(news)
                print(f"        [OK] {len(news_data)} days of News data")
        else:
            print(f"  [3/3] News disabled")

        # Combine all sources
        sentiment_features = self._combine_sentiment_sources(
            ticker, twitter_data, reddit_data, news_data
        )

        print(f"[OK] Generated sentiment features for {len(sentiment_features)} days")

        return sentiment_features

    def _combine_sentiment_sources(self, ticker: str,
                                   twitter_data: pd.DataFrame,
                                   reddit_data: pd.DataFrame,
                                   news_data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine Twitter, Reddit, News into unified sentiment features.

        Args:
            ticker: Stock ticker
            twitter_data: Aggregated Twitter sentiment
            reddit_data: Aggregated Reddit sentiment
            news_data: Aggregated news sentiment

        Returns:
            DataFrame with combined sentiment features
        """
        # Create date range
        all_dates = []

        if not twitter_data.empty:
            all_dates.extend(twitter_data['date'].tolist())
        if not reddit_data.empty:
            all_dates.extend(reddit_data['date'].tolist())
        if not news_data.empty:
            all_dates.extend(news_data['date'].tolist())

        if not all_dates:
            return pd.DataFrame()

        # Unique dates
        unique_dates = sorted(set(all_dates))

        # Initialize feature DataFrame
        features = pd.DataFrame({
            'ticker': ticker,
            'date': unique_dates
        })

        # Merge Twitter sentiment
        if not twitter_data.empty and 'twitter_value' in twitter_data.columns:
            features = features.merge(
                twitter_data[['date', 'twitter_value', 'tweet_count']],
                on='date',
                how='left'
            )
        else:
            features['twitter_value'] = 0.0
            features['tweet_count'] = 0

        # Merge Reddit sentiment
        if not reddit_data.empty and 'reddit_value' in reddit_data.columns:
            features = features.merge(
                reddit_data[['date', 'reddit_value', 'post_count']],
                on='date',
                how='left'
            )
        else:
            features['reddit_value'] = 0.0
            features['post_count'] = 0

        # Merge News sentiment
        if not news_data.empty and 'news_sentiment' in news_data.columns:
            features = features.merge(
                news_data[['date', 'news_sentiment', 'news_count', 'news_intensity']],
                on='date',
                how='left'
            )
        else:
            features['news_sentiment'] = 0.0
            features['news_count'] = 0
            features['news_intensity'] = 0.0

        # Fill NaN with 0
        features = features.fillna(0)

        # Calculate combined features
        features = self._calculate_sentiment_features(features)

        return features

    def _calculate_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived sentiment features.

        Args:
            df: DataFrame with raw sentiment data

        Returns:
            DataFrame with additional features
        """
        # 1. Social Sentiment: Combined Twitter + Reddit
        # Weight by volume if available
        twitter_weight = df['tweet_count'] / (df['tweet_count'] + df['post_count'] + 1e-6)
        reddit_weight = df['post_count'] / (df['tweet_count'] + df['post_count'] + 1e-6)

        df['social_sentiment'] = (
            df['twitter_value'] * twitter_weight +
            df['reddit_value'] * reddit_weight
        )

        # If no social data, use 0
        df.loc[df['tweet_count'] + df['post_count'] == 0, 'social_sentiment'] = 0.0

        # 2. Sentiment Divergence: |social - news|
        # High divergence = emotion vs information mismatch
        df['sentiment_divergence'] = abs(df['social_sentiment'] - df['news_sentiment'])

        # 3. Sentiment Velocity: Rate of change
        # Rapid sentiment shift indicates panic/euphoria
        df['social_velocity'] = df['social_sentiment'].diff()
        df['news_velocity'] = df['news_sentiment'].diff()

        # Fill first row
        df['social_velocity'] = df['social_velocity'].fillna(0)
        df['news_velocity'] = df['news_velocity'].fillna(0)

        # 4. Sentiment Intensity: Magnitude of sentiment
        df['social_intensity'] = abs(df['social_sentiment'])
        df['news_intensity_score'] = df.get('news_intensity', 0)

        # 5. Data availability flags
        df['has_social_data'] = (df['tweet_count'] + df['post_count']) > 0
        df['has_news_data'] = df['news_count'] > 0

        return df

    def classify_signal_confidence(self, row: pd.Series) -> str:
        """
        Classify panic sell signal confidence based on sentiment.

        Research-based logic:
        - HIGH: Social negative + News neutral/positive
          → Emotion-driven panic, fundamentals OK (BUY)
        - MEDIUM: Mixed signals or limited data
          → Uncertain, trade with caution
        - LOW: Social negative + News negative
          → Information-driven panic, fundamental issue (AVOID)

        Args:
            row: DataFrame row with sentiment features

        Returns:
            Confidence level: 'HIGH', 'MEDIUM', 'LOW', or 'BASELINE'
        """
        social_sentiment = row.get('social_sentiment', 0)
        news_sentiment = row.get('news_sentiment', 0)
        has_social = row.get('has_social_data', False)
        has_news = row.get('has_news_data', False)

        # No sentiment data → baseline strategy
        if not has_social and not has_news:
            return 'BASELINE'

        # Only social data available
        if has_social and not has_news:
            # Negative social → likely emotion-driven
            if social_sentiment < -0.3:
                return 'MEDIUM'  # Uncertain without news confirmation
            else:
                return 'BASELINE'

        # Only news data available
        if not has_social and has_news:
            # Negative news → likely information-driven
            if news_sentiment < -0.3:
                return 'LOW'  # Avoid information-driven drops
            else:
                return 'BASELINE'

        # Both social and news available (BEST CASE)
        if has_social and has_news:
            # HIGH CONFIDENCE: Emotion-driven panic
            # Social very negative BUT news neutral/positive
            if social_sentiment < -0.3 and news_sentiment > -0.1:
                return 'HIGH'

            # LOW CONFIDENCE: Information-driven panic
            # Both negative and aligned (divergence < 0.3)
            elif social_sentiment < -0.3 and news_sentiment < -0.3:
                divergence = abs(social_sentiment - news_sentiment)
                if divergence < 0.3:
                    return 'LOW'  # Aligned negative = fundamental issue
                else:
                    return 'MEDIUM'  # Divergent negative = uncertain

            # MEDIUM CONFIDENCE: Mixed or moderate signals
            else:
                return 'MEDIUM'

        return 'BASELINE'

    def add_confidence_labels(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add confidence classification to sentiment data.

        Args:
            sentiment_df: DataFrame with sentiment features

        Returns:
            DataFrame with 'confidence' column added
        """
        sentiment_df['confidence'] = sentiment_df.apply(
            self.classify_signal_confidence,
            axis=1
        )

        return sentiment_df


if __name__ == '__main__':
    """Test sentiment feature pipeline."""

    print("=" * 70)
    print("SENTIMENT FEATURE PIPELINE - TEST")
    print("=" * 70)
    print()

    pipeline = SentimentFeaturePipeline()

    if not pipeline.is_ready():
        print("[WARN] Pipeline not ready (modules not installed or configured)")
        print()
        print("To set up:")
        print("1. Install: pip install tweepy praw transformers torch")
        print("2. Configure API keys (see collector modules)")
        print()
    else:
        print("[OK] Pipeline ready!")
        print()

        # Test with recent date range
        ticker = 'NVDA'
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        print(f"Testing sentiment collection for {ticker}...")
        print(f"Date range: {start_date} to {end_date}")
        print()

        sentiment_features = pipeline.collect_all_sentiment(ticker, start_date, end_date)

        if not sentiment_features.empty:
            print()
            print("[SUCCESS] Sentiment features generated!")
            print()
            print("Sample features:")
            print("-" * 70)
            print(sentiment_features.head())

            # Add confidence labels
            sentiment_features = pipeline.add_confidence_labels(sentiment_features)

            print()
            print("Confidence distribution:")
            print(sentiment_features['confidence'].value_counts())

            # Show example of each confidence level
            for conf in ['HIGH', 'MEDIUM', 'LOW', 'BASELINE']:
                subset = sentiment_features[sentiment_features['confidence'] == conf]
                if not subset.empty:
                    print()
                    print(f"\n{conf} confidence example:")
                    row = subset.iloc[0]
                    print(f"  Date: {row['date']}")
                    print(f"  Social sentiment: {row['social_sentiment']:+.3f}")
                    print(f"  News sentiment: {row['news_sentiment']:+.3f}")
                    print(f"  Divergence: {row['sentiment_divergence']:.3f}")

        else:
            print("\n[WARN] No sentiment data collected")
            print("Check API configurations and try again")

    print()
    print("=" * 70)

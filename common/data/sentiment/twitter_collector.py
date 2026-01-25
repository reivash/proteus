"""
Twitter Sentiment Data Collector

Collects tweets about stocks for sentiment analysis.
Uses Twitter API v2 (requires Twitter Developer account).

Setup:
1. Create Twitter Developer account: https://developer.twitter.com
2. Create app and get API credentials
3. Add to config file (twitter_config.json):
   {
     "api_key": "YOUR_API_KEY",
     "api_secret": "YOUR_API_SECRET",
     "bearer_token": "YOUR_BEARER_TOKEN",
     "enabled": true
   }

Free tier limits:
- 500,000 tweets/month
- 100 tweets per request
- Good enough for daily scans of 11 stocks

Research justification:
- Twitter sentiment strongly predicts intraday trends (68.7% accuracy)
- Social media outperforms news for daily returns (67% vs 34%)
- Retail panic (negative social) + neutral news = mean reversion opportunity
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd

# Twitter API library
try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False


class TwitterSentimentCollector:
    """Collect tweets about stocks for sentiment analysis."""

    def __init__(self, config_path='twitter_config.json'):
        """
        Initialize Twitter collector.

        Args:
            config_path: Path to Twitter API config file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.client = None

        if self.is_enabled() and TWEEPY_AVAILABLE:
            self._initialize_client()

    def _load_config(self) -> Dict:
        """Load Twitter API configuration."""
        if not os.path.exists(self.config_path):
            return {'enabled': False}

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            # Check required fields
            required = ['bearer_token']
            if not all(field in config for field in required):
                return {'enabled': False}

            return config

        except Exception as e:
            print(f"[ERROR] Failed to load Twitter config: {e}")
            return {'enabled': False}

    def _initialize_client(self):
        """Initialize Twitter API client."""
        try:
            self.client = tweepy.Client(
                bearer_token=self.config['bearer_token'],
                wait_on_rate_limit=True
            )
            print("[INFO] Twitter API client initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Twitter client: {e}")
            self.client = None

    def is_enabled(self) -> bool:
        """Check if Twitter collection is enabled."""
        return (
            self.config.get('enabled', False) and
            TWEEPY_AVAILABLE and
            'bearer_token' in self.config
        )

    def collect(self, ticker: str, start_date: str, end_date: str,
                max_tweets_per_day: int = 100) -> pd.DataFrame:
        """
        Collect tweets about a stock ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'NVDA')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_tweets_per_day: Max tweets to collect per day

        Returns:
            DataFrame with tweet data (date, text, metrics)
        """
        if not self.is_enabled():
            print(f"[WARN] Twitter collection disabled for {ticker}")
            return pd.DataFrame()

        if self.client is None:
            print(f"[ERROR] Twitter client not initialized")
            return pd.DataFrame()

        print(f"[INFO] Collecting tweets for ${ticker} ({start_date} to {end_date})")

        # Build search query
        # Search for: $TICKER OR #TICKER OR "Ticker Name"
        # Exclude: retweets, replies (focus on original content)
        query = f"(${ticker} OR #{ticker}) -is:retweet -is:reply lang:en"

        all_tweets = []

        try:
            # Twitter API v2 recent search (last 7 days free, full archive paid)
            tweets = self.client.search_recent_tweets(
                query=query,
                start_time=start_date + "T00:00:00Z",
                end_time=end_date + "T23:59:59Z",
                max_results=min(max_tweets_per_day, 100),  # API limit: 100 per request
                tweet_fields=['created_at', 'public_metrics', 'text', 'lang'],
                expansions=['author_id'],
                user_fields=['username', 'verified']
            )

            if tweets.data:
                for tweet in tweets.data:
                    all_tweets.append({
                        'ticker': ticker,
                        'date': tweet.created_at.date(),
                        'text': tweet.text,
                        'likes': tweet.public_metrics['like_count'],
                        'retweets': tweet.public_metrics['retweet_count'],
                        'replies': tweet.public_metrics['reply_count'],
                        'engagement': (
                            tweet.public_metrics['like_count'] +
                            tweet.public_metrics['retweet_count'] +
                            tweet.public_metrics['reply_count']
                        )
                    })

                print(f"[OK] Collected {len(all_tweets)} tweets for ${ticker}")
            else:
                print(f"[WARN] No tweets found for ${ticker}")

        except Exception as e:
            print(f"[ERROR] Twitter API error for ${ticker}: {e}")

        return pd.DataFrame(all_tweets)

    def collect_realtime(self, ticker: str, lookback_hours: int = 24,
                        max_tweets: int = 100) -> pd.DataFrame:
        """
        Collect recent tweets for real-time sentiment.

        Args:
            ticker: Stock ticker symbol
            lookback_hours: Hours to look back
            max_tweets: Maximum tweets to collect

        Returns:
            DataFrame with recent tweet data
        """
        if not self.is_enabled():
            return pd.DataFrame()

        end_time = datetime.utcnow() - timedelta(seconds=30)  # 30 seconds before now (API requirement)
        start_time = end_time - timedelta(hours=lookback_hours)

        start_date = start_time.strftime('%Y-%m-%d')
        end_date = end_time.strftime('%Y-%m-%d')

        return self.collect(ticker, start_date, end_date, max_tweets)


def aggregate_daily_tweets(tweets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate tweets by day for sentiment analysis.

    Args:
        tweets_df: DataFrame with individual tweets

    Returns:
        DataFrame with daily aggregates (volume, engagement)
    """
    if tweets_df.empty:
        return pd.DataFrame()

    daily_agg = tweets_df.groupby(['ticker', 'date']).agg({
        'text': lambda x: ' '.join(x),  # Combine all tweets
        'likes': 'sum',
        'retweets': 'sum',
        'replies': 'sum',
        'engagement': 'sum',
        'date': 'count'  # Count tweets
    }).rename(columns={'date': 'tweet_count'})

    daily_agg = daily_agg.reset_index()

    return daily_agg


if __name__ == '__main__':
    """Test Twitter collector."""

    print("=" * 70)
    print("TWITTER SENTIMENT COLLECTOR - TEST")
    print("=" * 70)
    print()

    if not TWEEPY_AVAILABLE:
        print("[ERROR] Tweepy not installed")
        print("\nTo install:")
        print("  pip install tweepy")
        print()
        print("Then configure twitter_config.json with your API credentials")
        print("=" * 70)
    else:
        collector = TwitterSentimentCollector()

        if not collector.is_enabled():
            print("[WARN] Twitter API not configured")
            print()
            print("Setup instructions:")
            print("1. Create Twitter Developer account: https://developer.twitter.com")
            print("2. Create app and get API credentials")
            print("3. Create twitter_config.json:")
            print("""
{
  "bearer_token": "YOUR_BEARER_TOKEN_HERE",
  "enabled": true
}
            """)
        else:
            print("[OK] Twitter API configured")
            print()
            print("Testing real-time collection for NVDA...")

            # Test on recent tweets
            tweets = collector.collect_realtime('NVDA', lookback_hours=24, max_tweets=20)

            if not tweets.empty:
                print(f"\n[SUCCESS] Collected {len(tweets)} tweets")
                print("\nSample tweets:")
                print("-" * 70)
                for idx, tweet in tweets.head(3).iterrows():
                    print(f"\nDate: {tweet['date']}")
                    print(f"Text: {tweet['text'][:100]}...")
                    print(f"Engagement: {tweet['engagement']} (likes: {tweet['likes']}, RTs: {tweet['retweets']})")
            else:
                print("\n[WARN] No tweets collected")

    print()
    print("=" * 70)

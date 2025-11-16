"""
Reddit Sentiment Data Collector

Collects Reddit posts/comments about stocks for sentiment analysis.
Focuses on r/wallstreetbets, r/stocks, r/investing for retail sentiment.

Setup:
1. Create Reddit app: https://www.reddit.com/prefs/apps
2. Get client ID and secret
3. Add to config file (reddit_config.json):
   {
     "client_id": "YOUR_CLIENT_ID",
     "client_secret": "YOUR_CLIENT_SECRET",
     "user_agent": "proteus-trading-bot/1.0",
     "enabled": true
   }

Free tier:
- 60 requests/minute
- 100 posts per request
- Excellent for daily stock sentiment

Research justification:
- r/wallstreetbets strong indicator of retail panic/euphoria
- Social media outperforms news for daily returns (67% vs 34%)
- Retail panic + neutral fundamentals = mean reversion opportunity
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd

# Reddit API library
try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False


class RedditSentimentCollector:
    """Collect Reddit posts/comments about stocks for sentiment analysis."""

    def __init__(self, config_path='reddit_config.json'):
        """
        Initialize Reddit collector.

        Args:
            config_path: Path to Reddit API config file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.reddit = None

        # Subreddits to monitor (retail sentiment sources)
        self.target_subreddits = [
            'wallstreetbets',  # Retail panic/euphoria
            'stocks',          # General stock discussion
            'investing',       # Long-term investor sentiment
            'StockMarket'      # Market-wide sentiment
        ]

        if self.is_enabled() and PRAW_AVAILABLE:
            self._initialize_client()

    def _load_config(self) -> Dict:
        """Load Reddit API configuration."""
        if not os.path.exists(self.config_path):
            return {'enabled': False}

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            # Check required fields
            required = ['client_id', 'client_secret', 'user_agent']
            if not all(field in config for field in required):
                return {'enabled': False}

            return config

        except Exception as e:
            print(f"[ERROR] Failed to load Reddit config: {e}")
            return {'enabled': False}

    def _initialize_client(self):
        """Initialize Reddit API client."""
        try:
            self.reddit = praw.Reddit(
                client_id=self.config['client_id'],
                client_secret=self.config['client_secret'],
                user_agent=self.config['user_agent']
            )
            # Test connection
            self.reddit.user.me()
            print("[INFO] Reddit API client initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Reddit client: {e}")
            self.reddit = None

    def is_enabled(self) -> bool:
        """Check if Reddit collection is enabled."""
        return (
            self.config.get('enabled', False) and
            PRAW_AVAILABLE and
            all(field in self.config for field in ['client_id', 'client_secret', 'user_agent'])
        )

    def collect(self, ticker: str, start_date: str, end_date: str,
                max_posts_per_sub: int = 50) -> pd.DataFrame:
        """
        Collect Reddit posts about a stock ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'NVDA')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_posts_per_sub: Max posts per subreddit

        Returns:
            DataFrame with Reddit post data
        """
        if not self.is_enabled():
            print(f"[WARN] Reddit collection disabled for {ticker}")
            return pd.DataFrame()

        if self.reddit is None:
            print(f"[ERROR] Reddit client not initialized")
            return pd.DataFrame()

        print(f"[INFO] Collecting Reddit posts for ${ticker} ({start_date} to {end_date})")

        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        all_posts = []

        # Search each subreddit
        for subreddit_name in self.target_subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)

                # Search for ticker mentions
                # Query: ticker symbol OR $ticker
                query = f"{ticker} OR ${ticker}"

                posts = subreddit.search(
                    query,
                    sort='new',
                    time_filter='all',
                    limit=max_posts_per_sub
                )

                for post in posts:
                    post_date = datetime.fromtimestamp(post.created_utc)

                    # Filter by date range
                    if start_ts <= post.created_utc <= end_ts:
                        all_posts.append({
                            'ticker': ticker,
                            'date': post_date.date(),
                            'subreddit': subreddit_name,
                            'title': post.title,
                            'text': post.selftext if hasattr(post, 'selftext') else '',
                            'score': post.score,
                            'upvote_ratio': post.upvote_ratio,
                            'num_comments': post.num_comments,
                            'url': post.url,
                            'engagement': post.score + post.num_comments
                        })

                print(f"[OK] Collected {len([p for p in all_posts if p['subreddit'] == subreddit_name])} posts from r/{subreddit_name}")

            except Exception as e:
                print(f"[ERROR] Reddit API error for r/{subreddit_name}: {e}")

        if all_posts:
            print(f"[OK] Total collected: {len(all_posts)} posts for ${ticker}")
        else:
            print(f"[WARN] No posts found for ${ticker}")

        return pd.DataFrame(all_posts)

    def collect_realtime(self, ticker: str, lookback_hours: int = 24,
                        max_posts_per_sub: int = 20) -> pd.DataFrame:
        """
        Collect recent Reddit posts for real-time sentiment.

        Args:
            ticker: Stock ticker symbol
            lookback_hours: Hours to look back
            max_posts_per_sub: Maximum posts per subreddit

        Returns:
            DataFrame with recent post data
        """
        if not self.is_enabled():
            return pd.DataFrame()

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=lookback_hours)

        start_date = start_time.strftime('%Y-%m-%d')
        end_date = end_time.strftime('%Y-%m-%d')

        return self.collect(ticker, start_date, end_date, max_posts_per_sub)

    def collect_hot_posts(self, ticker: str, max_posts: int = 10) -> pd.DataFrame:
        """
        Collect currently HOT/trending posts about a ticker.

        Useful for detecting sudden retail panic or euphoria.

        Args:
            ticker: Stock ticker symbol
            max_posts: Maximum hot posts to collect

        Returns:
            DataFrame with hot posts
        """
        if not self.is_enabled():
            return pd.DataFrame()

        if self.reddit is None:
            return pd.DataFrame()

        all_hot_posts = []

        for subreddit_name in self.target_subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)

                # Get hot posts and filter for ticker mentions
                for post in subreddit.hot(limit=max_posts):
                    # Check if ticker mentioned in title or text
                    title_text = (post.title + ' ' + (post.selftext if hasattr(post, 'selftext') else '')).upper()

                    if ticker.upper() in title_text or f"${ticker.upper()}" in title_text:
                        all_hot_posts.append({
                            'ticker': ticker,
                            'date': datetime.fromtimestamp(post.created_utc).date(),
                            'subreddit': subreddit_name,
                            'title': post.title,
                            'text': post.selftext if hasattr(post, 'selftext') else '',
                            'score': post.score,
                            'upvote_ratio': post.upvote_ratio,
                            'num_comments': post.num_comments,
                            'is_hot': True,
                            'engagement': post.score + post.num_comments
                        })

            except Exception as e:
                print(f"[ERROR] Error collecting hot posts from r/{subreddit_name}: {e}")

        return pd.DataFrame(all_hot_posts)


def aggregate_daily_posts(posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Reddit posts by day for sentiment analysis.

    Args:
        posts_df: DataFrame with individual posts

    Returns:
        DataFrame with daily aggregates
    """
    if posts_df.empty:
        return pd.DataFrame()

    daily_agg = posts_df.groupby(['ticker', 'date']).agg({
        'title': lambda x: ' '.join(x),  # Combine titles
        'text': lambda x: ' '.join(x),   # Combine text
        'score': 'sum',
        'num_comments': 'sum',
        'engagement': 'sum',
        'upvote_ratio': 'mean',
        'date': 'count'  # Count posts
    }).rename(columns={'date': 'post_count'})

    daily_agg = daily_agg.reset_index()

    # Calculate average upvote ratio (sentiment proxy)
    # High upvote = consensus sentiment
    # Low upvote = controversial/mixed sentiment
    daily_agg['consensus_score'] = daily_agg['upvote_ratio']

    return daily_agg


if __name__ == '__main__':
    """Test Reddit collector."""

    print("=" * 70)
    print("REDDIT SENTIMENT COLLECTOR - TEST")
    print("=" * 70)
    print()

    if not PRAW_AVAILABLE:
        print("[ERROR] PRAW not installed")
        print("\nTo install:")
        print("  pip install praw")
        print()
        print("Then configure reddit_config.json with your API credentials")
        print("=" * 70)
    else:
        collector = RedditSentimentCollector()

        if not collector.is_enabled():
            print("[WARN] Reddit API not configured")
            print()
            print("Setup instructions:")
            print("1. Create Reddit app: https://www.reddit.com/prefs/apps")
            print("2. Click 'create app' (choose 'script')")
            print("3. Create reddit_config.json:")
            print("""
{
  "client_id": "YOUR_CLIENT_ID_HERE",
  "client_secret": "YOUR_CLIENT_SECRET_HERE",
  "user_agent": "proteus-trading-bot/1.0",
  "enabled": true
}
            """)
        else:
            print("[OK] Reddit API configured")
            print()
            print("Testing collection for NVDA...")

            # Test real-time collection
            posts = collector.collect_realtime('NVDA', lookback_hours=24, max_posts_per_sub=5)

            if not posts.empty:
                print(f"\n[SUCCESS] Collected {len(posts)} posts")
                print(f"Subreddits: {posts['subreddit'].unique()}")
                print("\nSample posts:")
                print("-" * 70)
                for idx, post in posts.head(3).iterrows():
                    print(f"\nSubreddit: r/{post['subreddit']}")
                    print(f"Date: {post['date']}")
                    print(f"Title: {post['title'][:80]}...")
                    print(f"Score: {post['score']}, Comments: {post['num_comments']}")
            else:
                print("\n[WARN] No posts collected")

            # Test hot posts
            print("\n" + "=" * 70)
            print("Testing HOT posts detection...")
            hot_posts = collector.collect_hot_posts('NVDA', max_posts=20)

            if not hot_posts.empty:
                print(f"\n[SUCCESS] Found {len(hot_posts)} hot posts mentioning NVDA")
                print("\nHot posts:")
                print("-" * 70)
                for idx, post in hot_posts.head(2).iterrows():
                    print(f"\nr/{post['subreddit']}")
                    print(f"Title: {post['title']}")
                    print(f"Engagement: {post['engagement']} (score: {post['score']}, comments: {post['num_comments']})")
            else:
                print("\n[INFO] No hot posts found")

    print()
    print("=" * 70)

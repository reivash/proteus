"""
News Headline Sentiment Collector

Collects financial news headlines for sentiment analysis.
Uses Alpha Vantage News API (free tier available).

Setup:
1. Get free API key: https://www.alphavantage.co/support/#api-key
2. Add to config file (news_config.json):
   {
     "alphavantage_api_key": "YOUR_API_KEY",
     "enabled": true
   }

Free tier:
- 25 requests/day (plenty for daily scans of 11 stocks)
- 50 news items per request
- Real-time news with sentiment scores

Research justification:
- News distinguishes information-driven vs emotion-driven panics
- Negative social + neutral news = emotion-driven panic (mean reversion)
- Negative social + negative news = information-driven (avoid)
- News sentiment provides fundamental context
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import requests
from time import sleep


class NewsCollector:
    """Collect financial news headlines for sentiment analysis."""

    def __init__(self, config_path='news_config.json'):
        """
        Initialize news collector.

        Args:
            config_path: Path to news API config file
        """
        self.config_path = config_path
        self.config = self._load_config()

        # Alpha Vantage endpoint
        self.base_url = "https://www.alphavantage.co/query"

    def _load_config(self) -> Dict:
        """Load news API configuration."""
        if not os.path.exists(self.config_path):
            return {'enabled': False}

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            # Check for API key
            if 'alphavantage_api_key' not in config or not config['alphavantage_api_key']:
                return {'enabled': False}

            return config

        except Exception as e:
            print(f"[ERROR] Failed to load news config: {e}")
            return {'enabled': False}

    def is_enabled(self) -> bool:
        """Check if news collection is enabled."""
        return (
            self.config.get('enabled', False) and
            'alphavantage_api_key' in self.config
        )

    def collect(self, ticker: str, start_date: str = None, end_date: str = None,
                limit: int = 50) -> pd.DataFrame:
        """
        Collect news headlines for a stock ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'NVDA')
            start_date: Start date (YYYY-MM-DD, optional)
            end_date: End date (YYYY-MM-DD, optional)
            limit: Max news items to collect

        Returns:
            DataFrame with news data
        """
        if not self.is_enabled():
            print(f"[WARN] News collection disabled for {ticker}")
            return pd.DataFrame()

        print(f"[INFO] Collecting news for {ticker}...")

        try:
            # Alpha Vantage News & Sentiment API
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'apikey': self.config['alphavantage_api_key'],
                'limit': min(limit, 1000)  # API max: 1000
            }

            # Add time range if specified
            if start_date and end_date:
                # Convert to Alpha Vantage format: YYYYMMDDTHHMM
                time_from = start_date.replace('-', '') + 'T0000'
                time_to = end_date.replace('-', '') + 'T2359'
                params['time_from'] = time_from
                params['time_to'] = time_to

            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if 'Error Message' in data:
                print(f"[ERROR] Alpha Vantage API error: {data['Error Message']}")
                return pd.DataFrame()

            if 'Note' in data:
                print(f"[WARN] API limit reached: {data['Note']}")
                return pd.DataFrame()

            # Parse news feed
            news_items = []

            if 'feed' in data:
                for article in data['feed']:
                    # Parse timestamp
                    time_published = datetime.strptime(article['time_published'], '%Y%m%dT%H%M%S')

                    # Get ticker-specific sentiment
                    ticker_sentiment = 0.0
                    ticker_relevance = 0.0

                    if 'ticker_sentiment' in article:
                        for ts in article['ticker_sentiment']:
                            if ts['ticker'] == ticker:
                                ticker_sentiment = float(ts['ticker_sentiment_score'])
                                ticker_relevance = float(ts['relevance_score'])
                                break

                    news_items.append({
                        'ticker': ticker,
                        'date': time_published.date(),
                        'datetime': time_published,
                        'title': article['title'],
                        'summary': article.get('summary', ''),
                        'source': article['source'],
                        'url': article['url'],
                        'overall_sentiment_score': float(article.get('overall_sentiment_score', 0)),
                        'overall_sentiment_label': article.get('overall_sentiment_label', 'Neutral'),
                        'ticker_sentiment_score': ticker_sentiment,
                        'ticker_relevance_score': ticker_relevance
                    })

                print(f"[OK] Collected {len(news_items)} news items for {ticker}")
            else:
                print(f"[WARN] No news found for {ticker}")

            # Rate limiting (free tier: 25 calls/day)
            sleep(1)  # Be nice to API

            return pd.DataFrame(news_items)

        except Exception as e:
            print(f"[ERROR] News collection error for {ticker}: {e}")
            return pd.DataFrame()

    def collect_realtime(self, ticker: str, limit: int = 50) -> pd.DataFrame:
        """
        Collect latest news for real-time sentiment.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum news items

        Returns:
            DataFrame with recent news
        """
        if not self.is_enabled():
            return pd.DataFrame()

        return self.collect(ticker, limit=limit)

    def collect_for_date_range(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Collect news for specific date range.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with news in date range
        """
        return self.collect(ticker, start_date, end_date, limit=200)


def aggregate_daily_news(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate news by day for sentiment analysis.

    Args:
        news_df: DataFrame with individual news items

    Returns:
        DataFrame with daily sentiment aggregates
    """
    if news_df.empty:
        return pd.DataFrame()

    # Filter for high-relevance news only (relevance > 0.3)
    relevant_news = news_df[news_df['ticker_relevance_score'] > 0.3].copy()

    if relevant_news.empty:
        return pd.DataFrame()

    # Aggregate by day
    daily_agg = relevant_news.groupby(['ticker', 'date']).agg({
        'title': lambda x: ' | '.join(x),  # Combine headlines
        'ticker_sentiment_score': 'mean',  # Average sentiment
        'ticker_relevance_score': 'mean',  # Average relevance
        'overall_sentiment_score': 'mean',
        'date': 'count'  # Count articles
    }).rename(columns={'date': 'news_count'})

    daily_agg = daily_agg.reset_index()

    # Calculate sentiment metrics
    daily_agg['news_sentiment'] = daily_agg['ticker_sentiment_score']
    daily_agg['news_volume'] = daily_agg['news_count']

    # High news volume with negative sentiment = information-driven panic
    # Low news volume with negative social = emotion-driven panic
    daily_agg['news_intensity'] = daily_agg['news_volume'] * abs(daily_agg['news_sentiment'])

    return daily_agg


if __name__ == '__main__':
    """Test news collector."""

    print("=" * 70)
    print("NEWS SENTIMENT COLLECTOR - TEST")
    print("=" * 70)
    print()

    collector = NewsCollector()

    if not collector.is_enabled():
        print("[WARN] News API not configured")
        print()
        print("Setup instructions:")
        print("1. Get free API key: https://www.alphavantage.co/support/#api-key")
        print("2. Create news_config.json:")
        print("""
{
  "alphavantage_api_key": "YOUR_API_KEY_HERE",
  "enabled": true
}
        """)
        print("\nFree tier: 25 requests/day (enough for daily scans)")
    else:
        print("[OK] News API configured")
        print()
        print("Testing news collection for NVDA...")

        # Test real-time collection
        news = collector.collect_realtime('NVDA', limit=10)

        if not news.empty:
            print(f"\n[SUCCESS] Collected {len(news)} news items")
            print("\nRecent headlines:")
            print("-" * 70)

            for idx, article in news.head(5).iterrows():
                sentiment_label = article['overall_sentiment_label']
                sentiment_score = article['ticker_sentiment_score']

                print(f"\nDate: {article['datetime']}")
                print(f"Title: {article['title'][:80]}...")
                print(f"Source: {article['source']}")
                print(f"Sentiment: {sentiment_label} (score: {sentiment_score:.3f})")
                print(f"Relevance: {article['ticker_relevance_score']:.3f}")

            # Test aggregation
            print("\n" + "=" * 70)
            print("Testing daily aggregation...")

            daily_news = aggregate_daily_news(news)
            if not daily_news.empty:
                print(f"\n[OK] Aggregated into {len(daily_news)} days")
                print("\nDaily sentiment summary:")
                print("-" * 70)
                for idx, day in daily_news.iterrows():
                    print(f"\nDate: {day['date']}")
                    print(f"News count: {day['news_count']}")
                    print(f"Average sentiment: {day['news_sentiment']:.3f}")
                    print(f"News intensity: {day['news_intensity']:.3f}")
        else:
            print("\n[WARN] No news collected")

    print()
    print("=" * 70)

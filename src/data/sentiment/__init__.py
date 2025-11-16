"""
Sentiment data collectors for EXP-011.

Modules:
- twitter_collector: Collect Twitter sentiment
- reddit_collector: Collect Reddit sentiment
- news_collector: Collect news headline sentiment
- sentiment_features: Combine all sources into trading features
"""

__all__ = [
    'TwitterSentimentCollector',
    'RedditSentimentCollector',
    'NewsCollector',
    'SentimentFeaturePipeline'
]

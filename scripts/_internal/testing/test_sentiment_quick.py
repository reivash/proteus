"""Quick test to see if sentiment pipeline works."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from common.data.sentiment.sentiment_features import SentimentFeaturePipeline

# Test if pipeline is ready
pipeline = SentimentFeaturePipeline()

print(f"Sentiment pipeline ready: {pipeline.is_ready()}")

if pipeline.is_ready():
    print("  Twitter collector: {pipeline.twitter_collector is not None}")
    print("  Reddit collector: {pipeline.reddit_collector is not None}")
    print("  News collector: {pipeline.news_collector is not None}")
    print("  FinBERT analyzer: {pipeline.sentiment_analyzer is not None}")
else:
    print("[WARNING] Sentiment pipeline NOT ready - missing modules or API keys")

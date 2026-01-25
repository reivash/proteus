"""
Unified Sentiment Scoring Pipeline
===================================

Combines multiple sentiment sources with FinBERT scoring to produce
a single sentiment score per stock for trading signal enhancement.

Pipeline:
1. Collect news headlines (Alpha Vantage)
2. Score with FinBERT (finance-specific sentiment)
3. Aggregate to daily sentiment score
4. Classify as: PANIC (-1), NEGATIVE (-0.5), NEUTRAL (0), POSITIVE (+0.5), EUPHORIC (+1)

Integration with SmartScanner:
- Panic selling + negative sentiment = BUY (mean reversion opportunity)
- Panic selling + positive sentiment = AVOID (price justified)

Jan 7, 2026 - Proteus Trading System
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from common.data.sentiment.news_collector import NewsCollector
from common.models.sentiment.finbert_analyzer import FinBERTAnalyzer


@dataclass
class SentimentResult:
    """Aggregated sentiment result for a stock."""
    ticker: str
    timestamp: str

    # Aggregate scores
    sentiment_score: float  # -1 to +1
    sentiment_label: str    # PANIC, NEGATIVE, NEUTRAL, POSITIVE, EUPHORIC
    confidence: float       # 0 to 1

    # Components
    news_count: int
    positive_count: int
    negative_count: int
    neutral_count: int

    # For signal filtering
    is_panic: bool          # High negative sentiment volume
    is_euphoric: bool       # High positive sentiment volume
    should_filter: bool     # Sentiment suggests avoiding trade
    filter_reason: str


class UnifiedSentimentScorer:
    """
    Unified sentiment scoring combining news data with FinBERT analysis.
    """

    # Sentiment thresholds
    PANIC_THRESHOLD = -0.5      # Score below this = panic
    NEGATIVE_THRESHOLD = -0.2
    POSITIVE_THRESHOLD = 0.2
    EUPHORIC_THRESHOLD = 0.5    # Score above this = euphoric

    # Volume thresholds for panic detection
    MIN_NEWS_FOR_PANIC = 5      # Need at least 5 news items to detect panic
    PANIC_NEGATIVE_RATIO = 0.7  # 70%+ negative news = panic

    def __init__(self, use_gpu: bool = True):
        """Initialize sentiment scorer with FinBERT."""
        self.news_collector = NewsCollector()
        self.finbert = None

        # Lazy load FinBERT (heavy model)
        self._finbert_loaded = False
        self._use_gpu = use_gpu

    def _ensure_finbert(self):
        """Lazy load FinBERT model."""
        if not self._finbert_loaded:
            print("[SENTIMENT] Loading FinBERT model...")
            self.finbert = FinBERTAnalyzer(use_gpu=self._use_gpu)
            self._finbert_loaded = True

    def get_sentiment(self, ticker: str, lookback_hours: int = 24) -> SentimentResult:
        """
        Get aggregated sentiment for a stock.

        Args:
            ticker: Stock symbol
            lookback_hours: How far back to look for news (default 24h)

        Returns:
            SentimentResult with aggregated sentiment
        """
        self._ensure_finbert()

        timestamp = datetime.now().isoformat()

        # Collect recent news
        if not self.news_collector.is_enabled():
            return self._empty_result(ticker, timestamp, "News API not configured")

        news_df = self.news_collector.collect(ticker, limit=50)

        if news_df is None or len(news_df) == 0:
            return self._empty_result(ticker, timestamp, "No news found")

        # Score each headline with FinBERT
        sentiments = []
        for _, row in news_df.iterrows():
            headline = row.get('title', row.get('headline', ''))
            if not headline:
                continue

            # Get FinBERT sentiment
            result = self.finbert.analyze(headline)
            if result:
                sentiments.append({
                    'text': headline[:100],
                    'label': result['label'],
                    'score': result['score'],
                    # Convert to numeric: positive=+1, negative=-1, neutral=0
                    'numeric': self._label_to_score(result['label'], result['score'])
                })

        if not sentiments:
            return self._empty_result(ticker, timestamp, "Could not score news")

        # Aggregate sentiment
        return self._aggregate_sentiments(ticker, timestamp, sentiments)

    def _label_to_score(self, label: str, confidence: float) -> float:
        """Convert FinBERT label to numeric score."""
        if label == 'positive':
            return confidence
        elif label == 'negative':
            return -confidence
        else:  # neutral
            return 0.0

    def _aggregate_sentiments(self, ticker: str, timestamp: str,
                             sentiments: List[Dict]) -> SentimentResult:
        """Aggregate individual sentiment scores."""

        # Count by label
        positive_count = sum(1 for s in sentiments if s['label'] == 'positive')
        negative_count = sum(1 for s in sentiments if s['label'] == 'negative')
        neutral_count = sum(1 for s in sentiments if s['label'] == 'neutral')
        total = len(sentiments)

        # Weighted average (weight by confidence)
        scores = [s['numeric'] for s in sentiments]
        avg_score = np.mean(scores) if scores else 0.0

        # Confidence = consistency of sentiment
        confidence = 1.0 - np.std(scores) if len(scores) > 1 else 0.5
        confidence = max(0.0, min(1.0, confidence))

        # Classify sentiment
        if avg_score <= self.PANIC_THRESHOLD:
            label = "PANIC"
        elif avg_score <= self.NEGATIVE_THRESHOLD:
            label = "NEGATIVE"
        elif avg_score >= self.EUPHORIC_THRESHOLD:
            label = "EUPHORIC"
        elif avg_score >= self.POSITIVE_THRESHOLD:
            label = "POSITIVE"
        else:
            label = "NEUTRAL"

        # Detect panic conditions
        is_panic = (
            total >= self.MIN_NEWS_FOR_PANIC and
            negative_count / total >= self.PANIC_NEGATIVE_RATIO
        )

        # Detect euphoria
        is_euphoric = (
            total >= self.MIN_NEWS_FOR_PANIC and
            positive_count / total >= self.PANIC_NEGATIVE_RATIO
        )

        # Determine filter recommendation
        # For mean reversion: We WANT panic selling (negative sentiment)
        # But we want to AVOID if news is legitimately bad (negative + high confidence)
        should_filter = False
        filter_reason = ""

        if is_euphoric:
            # Euphoric sentiment = price may be justified, risky mean reversion
            should_filter = True
            filter_reason = f"Euphoric sentiment ({positive_count}/{total} positive)"
        elif avg_score > 0.3 and confidence > 0.7:
            # Strong positive sentiment = not a panic sell
            should_filter = True
            filter_reason = f"Strong positive news (score={avg_score:.2f})"

        return SentimentResult(
            ticker=ticker,
            timestamp=timestamp,
            sentiment_score=round(avg_score, 3),
            sentiment_label=label,
            confidence=round(confidence, 3),
            news_count=total,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            is_panic=is_panic,
            is_euphoric=is_euphoric,
            should_filter=should_filter,
            filter_reason=filter_reason
        )

    def _empty_result(self, ticker: str, timestamp: str, reason: str) -> SentimentResult:
        """Create empty result when no data available."""
        return SentimentResult(
            ticker=ticker,
            timestamp=timestamp,
            sentiment_score=0.0,
            sentiment_label="UNKNOWN",
            confidence=0.0,
            news_count=0,
            positive_count=0,
            negative_count=0,
            neutral_count=0,
            is_panic=False,
            is_euphoric=False,
            should_filter=False,
            filter_reason=reason
        )

    def scan_portfolio(self, tickers: List[str]) -> Dict[str, SentimentResult]:
        """
        Get sentiment for all tickers in portfolio.

        Returns dict mapping ticker -> SentimentResult
        """
        results = {}

        print(f"[SENTIMENT] Scanning {len(tickers)} tickers...")

        for i, ticker in enumerate(tickers):
            print(f"  [{i+1}/{len(tickers)}] {ticker}...", end=" ")
            result = self.get_sentiment(ticker)
            results[ticker] = result
            print(f"{result.sentiment_label} ({result.sentiment_score:+.2f})")

        return results

    def get_filter_recommendation(self, ticker: str, signal_strength: float) -> Tuple[bool, str]:
        """
        Get filter recommendation for a trading signal.

        For mean reversion strategy:
        - Panic selling (negative sentiment) = GOOD for entry
        - Positive sentiment during selloff = BAD (price decline is justified)

        Args:
            ticker: Stock symbol
            signal_strength: Current signal strength (0-100)

        Returns:
            (should_trade, reason)
        """
        sentiment = self.get_sentiment(ticker)

        if sentiment.sentiment_label == "UNKNOWN":
            # No sentiment data - allow trade
            return True, "No sentiment data available"

        # Strong positive sentiment during what should be a mean reversion
        # opportunity suggests the selloff may be justified
        if sentiment.should_filter:
            return False, sentiment.filter_reason

        # Panic sentiment is GOOD for mean reversion
        if sentiment.is_panic:
            return True, f"Panic detected ({sentiment.negative_count} negative news) - mean reversion opportunity"

        # Negative sentiment = good for mean reversion
        if sentiment.sentiment_score < -0.2:
            return True, f"Negative sentiment ({sentiment.sentiment_score:+.2f}) supports mean reversion"

        # Neutral - allow trade
        return True, f"Sentiment neutral ({sentiment.sentiment_score:+.2f})"


def test_unified_sentiment():
    """Test the unified sentiment scorer."""
    print("=" * 70)
    print("UNIFIED SENTIMENT SCORER TEST")
    print("=" * 70)

    scorer = UnifiedSentimentScorer(use_gpu=True)

    # Test on a few stocks
    test_tickers = ['NVDA', 'AAPL', 'JPM']

    for ticker in test_tickers:
        print(f"\n{ticker}:")
        print("-" * 40)
        result = scorer.get_sentiment(ticker)

        print(f"  Score: {result.sentiment_score:+.3f}")
        print(f"  Label: {result.sentiment_label}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  News count: {result.news_count}")
        print(f"  Distribution: +{result.positive_count} / ~{result.neutral_count} / -{result.negative_count}")
        print(f"  Is panic: {result.is_panic}")
        print(f"  Should filter: {result.should_filter}")
        if result.filter_reason:
            print(f"  Reason: {result.filter_reason}")


if __name__ == '__main__':
    test_unified_sentiment()

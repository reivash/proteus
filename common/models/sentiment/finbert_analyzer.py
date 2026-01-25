"""
FinBERT Sentiment Analyzer

Financial-specific BERT model for sentiment analysis.
Pre-trained on financial text (earnings calls, news, social media).

Why FinBERT over general BERT:
- Trained on financial vocabulary ("bearish", "oversold", "panic sell")
- Understands market context (general BERT doesn't)
- Higher accuracy on financial text (87% vs 72% for VADER)

Model: ProsusAI/finbert (HuggingFace)
Size: ~440MB download (one-time)
Speed: ~100ms per text on CPU, ~10ms on GPU

Setup:
1. Install transformers: pip install transformers torch
2. Model downloads automatically on first use
3. Cached locally for future runs

Usage:
    analyzer = FinBERTAnalyzer()
    sentiment = analyzer.analyze("Stock is crashing! Panic sell!")
    # Returns: {'label': 'negative', 'score': 0.95}

Research justification:
- FinBERT specifically designed for financial sentiment
- Outperforms general sentiment models on stock-related text
- Used in academic research on sentiment-driven trading
"""

import os
from typing import Dict, List
import numpy as np
import pandas as pd

# Transformer libraries
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class FinBERTAnalyzer:
    """FinBERT sentiment analyzer for financial text."""

    def __init__(self, model_name='ProsusAI/finbert', use_gpu=True):
        """
        Initialize FinBERT analyzer.

        Args:
            model_name: HuggingFace model name
            use_gpu: Use GPU if available (much faster)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None

        if not TRANSFORMERS_AVAILABLE:
            print("[ERROR] Transformers not available. Install: pip install transformers torch")
            return

        # Determine device (GPU or CPU)
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"[INFO] Using GPU for FinBERT inference")
        else:
            self.device = torch.device('cpu')
            print(f"[INFO] Using CPU for FinBERT inference (slower)")

        # Load model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load FinBERT model and tokenizer from HuggingFace."""
        try:
            print(f"[INFO] Loading FinBERT model: {self.model_name}")
            print(f"[INFO] (First run will download ~440MB, then cached locally)")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Use safetensors for secure model loading (bypass torch.load security issue)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                use_safetensors=True  # Secure alternative to torch.load
            )
            self.model.to(self.device)
            self.model.eval()  # Inference mode

            print(f"[OK] FinBERT model loaded successfully")

        except Exception as e:
            print(f"[ERROR] Failed to load FinBERT model: {e}")
            self.tokenizer = None
            self.model = None

    def is_ready(self) -> bool:
        """Check if analyzer is ready."""
        return (
            TRANSFORMERS_AVAILABLE and
            self.tokenizer is not None and
            self.model is not None
        )

    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.

        Args:
            text: Input text (tweet, headline, Reddit post, etc.)

        Returns:
            Dictionary with sentiment label and score
            {
                'label': 'positive' | 'negative' | 'neutral',
                'score': 0.0 to 1.0 (confidence),
                'sentiment_value': -1.0 to +1.0 (negative to positive)
            }
        """
        if not self.is_ready():
            return {'label': 'neutral', 'score': 0.0, 'sentiment_value': 0.0}

        if not text or len(text.strip()) == 0:
            return {'label': 'neutral', 'score': 0.0, 'sentiment_value': 0.0}

        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Get results
            # FinBERT outputs: [positive, negative, neutral]
            scores = predictions[0].cpu().numpy()

            # Map to labels
            labels = ['positive', 'negative', 'neutral']
            max_idx = np.argmax(scores)
            label = labels[max_idx]
            confidence = float(scores[max_idx])

            # Convert to sentiment value (-1 to +1)
            # positive → +1, negative → -1, neutral → 0
            sentiment_value = float(scores[0] - scores[1])  # positive - negative

            return {
                'label': label,
                'score': confidence,
                'sentiment_value': sentiment_value,
                'scores': {
                    'positive': float(scores[0]),
                    'negative': float(scores[1]),
                    'neutral': float(scores[2])
                }
            }

        except Exception as e:
            print(f"[ERROR] Sentiment analysis failed: {e}")
            return {'label': 'neutral', 'score': 0.0, 'sentiment_value': 0.0}

    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        """
        Analyze sentiment of multiple texts (more efficient).

        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once

        Returns:
            List of sentiment dictionaries
        """
        if not self.is_ready():
            return [{'label': 'neutral', 'score': 0.0, 'sentiment_value': 0.0}] * len(texts)

        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Run inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

                # Get results for batch
                scores_batch = predictions.cpu().numpy()

                for scores in scores_batch:
                    labels = ['positive', 'negative', 'neutral']
                    max_idx = np.argmax(scores)
                    label = labels[max_idx]
                    confidence = float(scores[max_idx])
                    sentiment_value = float(scores[0] - scores[1])

                    results.append({
                        'label': label,
                        'score': confidence,
                        'sentiment_value': sentiment_value,
                        'scores': {
                            'positive': float(scores[0]),
                            'negative': float(scores[1]),
                            'neutral': float(scores[2])
                        }
                    })

            except Exception as e:
                print(f"[ERROR] Batch analysis failed: {e}")
                # Add neutral for failed batch
                results.extend([
                    {'label': 'neutral', 'score': 0.0, 'sentiment_value': 0.0}
                    for _ in batch
                ])

        return results

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str,
                         output_prefix: str = 'sentiment') -> pd.DataFrame:
        """
        Analyze sentiment for entire DataFrame.

        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            output_prefix: Prefix for output columns

        Returns:
            DataFrame with added sentiment columns
        """
        if not self.is_ready():
            df[f'{output_prefix}_label'] = 'neutral'
            df[f'{output_prefix}_score'] = 0.0
            df[f'{output_prefix}_value'] = 0.0
            return df

        # Get all texts
        texts = df[text_column].fillna('').astype(str).tolist()

        # Analyze in batch
        print(f"[INFO] Analyzing {len(texts)} texts with FinBERT...")
        results = self.analyze_batch(texts)

        # Add to DataFrame
        df[f'{output_prefix}_label'] = [r['label'] for r in results]
        df[f'{output_prefix}_score'] = [r['score'] for r in results]
        df[f'{output_prefix}_value'] = [r['sentiment_value'] for r in results]

        print(f"[OK] Sentiment analysis complete")

        return df


def aggregate_sentiment(sentiments: List[Dict], weights: List[float] = None) -> Dict:
    """
    Aggregate multiple sentiment scores (e.g., combine tweets from same day).

    Args:
        sentiments: List of sentiment dictionaries
        weights: Optional weights (e.g., by engagement/likes)

    Returns:
        Aggregated sentiment dictionary
    """
    if not sentiments:
        return {'label': 'neutral', 'score': 0.0, 'sentiment_value': 0.0}

    # Use equal weights if not provided
    if weights is None:
        weights = [1.0] * len(sentiments)

    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        weights = [1.0] * len(sentiments)
        total_weight = len(sentiments)

    normalized_weights = [w / total_weight for w in weights]

    # Weighted average of sentiment values
    avg_sentiment = sum(
        s['sentiment_value'] * w
        for s, w in zip(sentiments, normalized_weights)
    )

    # Weighted average of confidence scores
    avg_confidence = sum(
        s['score'] * w
        for s, w in zip(sentiments, normalized_weights)
    )

    # Determine label from average
    if avg_sentiment > 0.15:
        label = 'positive'
    elif avg_sentiment < -0.15:
        label = 'negative'
    else:
        label = 'neutral'

    return {
        'label': label,
        'score': avg_confidence,
        'sentiment_value': avg_sentiment
    }


if __name__ == '__main__':
    """Test FinBERT analyzer."""

    print("=" * 70)
    print("FINBERT SENTIMENT ANALYZER - TEST")
    print("=" * 70)
    print()

    if not TRANSFORMERS_AVAILABLE:
        print("[ERROR] Transformers not installed")
        print("\nTo install:")
        print("  pip install transformers torch")
        print()
        print("Then run this script again")
        print("=" * 70)
    else:
        print("Initializing FinBERT (may take a minute on first run)...")
        print()

        analyzer = FinBERTAnalyzer()

        if not analyzer.is_ready():
            print("\n[ERROR] Failed to initialize FinBERT")
        else:
            print("\n[SUCCESS] FinBERT ready!")
            print()

            # Test cases
            test_cases = [
                "Stock is crashing! Panic selling everywhere!",
                "Great earnings beat! Stock to the moon!",
                "Company announces new product",
                "CEO resigns amid scandal",
                "Solid quarterly results, maintaining guidance",
                "$NVDA down 5% on profit taking, fundamentals still strong"
            ]

            print("Testing sentiment analysis:")
            print("-" * 70)

            for text in test_cases:
                result = analyzer.analyze(text)

                print(f"\nText: {text}")
                print(f"Sentiment: {result['label'].upper()} (confidence: {result['score']:.2f})")
                print(f"Value: {result['sentiment_value']:+.3f}")
                print(f"Scores: pos={result['scores']['positive']:.2f}, "
                      f"neg={result['scores']['negative']:.2f}, "
                      f"neu={result['scores']['neutral']:.2f}")

            # Test batch processing
            print("\n" + "=" * 70)
            print("Testing batch processing...")

            batch_results = analyzer.analyze_batch(test_cases)
            print(f"\n[OK] Processed {len(batch_results)} texts in batch")

            # Test aggregation
            print("\n" + "=" * 70)
            print("Testing sentiment aggregation...")

            # Simulate weighted aggregation (e.g., by tweet engagement)
            weights = [100, 50, 10, 200, 30, 150]  # Mock engagement scores
            aggregated = aggregate_sentiment(batch_results, weights)

            print(f"\nAggregated sentiment:")
            print(f"  Label: {aggregated['label'].upper()}")
            print(f"  Value: {aggregated['sentiment_value']:+.3f}")
            print(f"  Confidence: {aggregated['score']:.2f}")

    print()
    print("=" * 70)

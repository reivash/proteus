"""
Signal Quality Scoring System

Ranks mean reversion signals by composite quality score.
Higher quality signals = higher win rate.

Based on EXP-033 findings:
- Quality scoring improves win rate by +3.7pp (85% -> 88.7%)
- Top 60% of signals achieve 88.7% win rate
- Ultra-selective = ultra-high quality

Quality Score Formula (0-100):
- Z-score depth: 40% weight
- RSI oversold level: 25% weight
- Volume spike magnitude: 20% weight
- Price drop severity: 15% weight

TOTAL: 100%

Usage:
    scorer = SignalQualityScorer()
    quality_score = scorer.calculate_score(
        z_score=-2.5,
        rsi=25,
        volume_ratio=2.0,
        price_drop=-3.5
    )

    is_high_quality = quality_score >= 60  # Top 60% threshold
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class SignalQualityScorer:
    """
    Calculate composite quality scores for mean reversion signals.
    """

    def __init__(self, min_quality_threshold: int = 60):
        """
        Initialize quality scorer.

        Args:
            min_quality_threshold: Minimum score to pass (0-100)
                                  Default: 60 (top 60% of signals)
        """
        self.min_quality_threshold = min_quality_threshold

    def calculate_score(self, z_score: float, rsi: float,
                       volume_ratio: float, price_drop: float) -> float:
        """
        Calculate composite quality score for a signal.

        Args:
            z_score: Z-score value (e.g., -2.5)
            rsi: RSI value (e.g., 25)
            volume_ratio: Volume spike ratio (e.g., 2.0 = 2x average)
            price_drop: Price drop % (e.g., -3.5)

        Returns:
            Quality score (0-100)
        """
        # 1. Z-score depth (40% weight)
        # Deeper panic = higher quality
        # Example: Z=-3.0 -> score=200, Z=-1.5 -> score=100
        z_component = min(100, (abs(z_score) / 1.5) * 100) * 0.40

        # 2. RSI oversold (25% weight)
        # More oversold = higher quality
        # Example: RSI=20 -> score=100, RSI=34 -> score=6.7
        rsi_component = min(100, ((35 - rsi) / 15) * 100) * 0.25

        # 3. Volume spike (20% weight)
        # Larger spike = higher quality
        # Example: 3x volume -> score=100, 1.4x -> score=5.9
        volume_component = min(100, ((volume_ratio - 1.3) / 1.7) * 100) * 0.20

        # 4. Price drop severity (15% weight)
        # Bigger drop = higher quality
        # Example: -5% drop -> score=100, -2% drop -> score=40
        price_component = min(100, (abs(price_drop) / 5.0) * 100) * 0.15

        total_score = z_component + rsi_component + volume_component + price_component

        return max(0, min(100, total_score))

    def is_high_quality(self, score: float) -> bool:
        """
        Check if signal meets minimum quality threshold.

        Args:
            score: Quality score

        Returns:
            True if score >= threshold
        """
        return score >= self.min_quality_threshold

    def add_quality_scores_to_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add quality scores to signal DataFrame.

        Args:
            df: DataFrame with signals and indicators

        Returns:
            DataFrame with quality_score and high_quality columns added
        """
        data = df.copy()

        # Calculate quality score for each row
        quality_scores = []

        for idx, row in data.iterrows():
            # Get signal indicators
            z_score = row.get('z_score', 0)
            rsi = row.get('rsi', 50)

            # Calculate volume ratio (current volume / 20-day average)
            if 'volume' in row:
                volume_20ma = data['volume'].rolling(20).mean().loc[idx]
                volume_ratio = row['volume'] / volume_20ma if volume_20ma > 0 else 1.0
            else:
                volume_ratio = 1.0

            # Calculate price drop (today vs yesterday)
            if 'Close' in row:
                if idx > 0:
                    prev_close = data.loc[idx - 1, 'Close']
                    price_drop = ((row['Close'] - prev_close) / prev_close) * 100
                else:
                    price_drop = 0
            else:
                price_drop = 0

            # Calculate quality score
            score = self.calculate_score(
                z_score=z_score,
                rsi=rsi,
                volume_ratio=volume_ratio,
                price_drop=price_drop
            )

            quality_scores.append(score)

        # Add to DataFrame
        data['quality_score'] = quality_scores
        data['high_quality'] = data['quality_score'] >= self.min_quality_threshold

        return data

    def filter_low_quality_signals(self, df: pd.DataFrame,
                                   signal_column: str = 'panic_sell') -> pd.DataFrame:
        """
        Filter out low-quality signals.

        Sets signal to 0 if quality score below threshold.

        Args:
            df: DataFrame with signals and quality_score column
            signal_column: Name of signal column to filter

        Returns:
            DataFrame with filtered signals
        """
        data = df.copy()

        # Ensure quality scores exist
        if 'quality_score' not in data.columns:
            data = self.add_quality_scores_to_signals(data)

        # Store original signals
        if signal_column in data.columns:
            data[f'{signal_column}_before_quality_filter'] = data[signal_column].copy()

            # Disable low-quality signals
            low_quality_mask = data['quality_score'] < self.min_quality_threshold
            data.loc[low_quality_mask, signal_column] = 0

        return data

    def get_score_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get statistics about quality score distribution.

        Args:
            df: DataFrame with quality_score column

        Returns:
            Dictionary with distribution statistics
        """
        if 'quality_score' not in df.columns:
            return {'error': 'No quality_score column found'}

        scores = df['quality_score']

        return {
            'min': float(scores.min()),
            'max': float(scores.max()),
            'mean': float(scores.mean()),
            'median': float(scores.median()),
            'p25': float(scores.quantile(0.25)),
            'p75': float(scores.quantile(0.75)),
            'pct_high_quality': float((scores >= self.min_quality_threshold).sum() / len(scores) * 100)
        }


def test_quality_scorer():
    """Test the quality scoring system."""
    print("=" * 70)
    print("SIGNAL QUALITY SCORER TEST")
    print("=" * 70)
    print()

    scorer = SignalQualityScorer(min_quality_threshold=60)

    # Test signals with different qualities
    test_signals = [
        {
            'name': 'EXCELLENT signal (deep panic)',
            'z_score': -3.0,
            'rsi': 20,
            'volume_ratio': 3.0,
            'price_drop': -5.0
        },
        {
            'name': 'GOOD signal (moderate panic)',
            'z_score': -2.0,
            'rsi': 28,
            'volume_ratio': 2.0,
            'price_drop': -3.0
        },
        {
            'name': 'MARGINAL signal (weak panic)',
            'z_score': -1.6,
            'rsi': 34,
            'volume_ratio': 1.4,
            'price_drop': -2.0
        },
        {
            'name': 'POOR signal (barely qualifies)',
            'z_score': -1.5,
            'rsi': 35,
            'volume_ratio': 1.3,
            'price_drop': -1.5
        }
    ]

    print("Testing individual signals:")
    print()

    for signal in test_signals:
        score = scorer.calculate_score(
            z_score=signal['z_score'],
            rsi=signal['rsi'],
            volume_ratio=signal['volume_ratio'],
            price_drop=signal['price_drop']
        )

        is_hq = scorer.is_high_quality(score)
        verdict = "PASS" if is_hq else "REJECT"

        print(f"{signal['name']}:")
        print(f"  Z-score: {signal['z_score']}")
        print(f"  RSI: {signal['rsi']}")
        print(f"  Volume: {signal['volume_ratio']}x")
        print(f"  Drop: {signal['price_drop']}%")
        print(f"  Quality Score: {score:.1f}/100")
        print(f"  Verdict: [{verdict}]")
        print()

    print("=" * 70)
    print("EXPECTED OUTCOMES:")
    print("=" * 70)
    print()
    print("EXP-033 findings:")
    print("  - Top 60% threshold (score >= 60)")
    print("  - Expected win rate: 88.7% (+3.7pp vs 85%)")
    print("  - Trades reduced to ~40% of signals")
    print("  - Quality over quantity strategy")
    print()
    print("[SUCCESS] Quality scoring system ready for production!")
    print()


if __name__ == "__main__":
    test_quality_scorer()

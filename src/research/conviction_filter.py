"""
Conviction Filter Integration

Integrates deep dive conviction scores with Proteus signal filtering.
Modifies position sizing based on fundamental conviction.
"""

import os
import json
import glob
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple


class ConvictionFilter:
    """
    Filters and modifies Proteus signals based on deep dive conviction scores.
    """

    def __init__(self, deep_dive_dir: str = "data/deep_dives"):
        self.deep_dive_dir = deep_dive_dir
        self._cache = {}
        self._load_all_analyses()

    def _load_all_analyses(self):
        """Load all deep dive analyses into cache."""
        pattern = os.path.join(self.deep_dive_dir, "*_2025-*.json")
        files = glob.glob(pattern)

        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    ticker = data.get('ticker')
                    if ticker:
                        # Keep most recent analysis per ticker
                        existing = self._cache.get(ticker)
                        if not existing or data.get('analysis_date', '') > existing.get('analysis_date', ''):
                            self._cache[ticker] = data
            except:
                pass

        print(f"[ConvictionFilter] Loaded {len(self._cache)} stock analyses")

    def get_conviction(self, ticker: str) -> Optional[Dict]:
        """Get conviction data for a ticker."""
        return self._cache.get(ticker)

    def filter_signal(self, signal: Dict) -> Tuple[bool, Dict]:
        """
        Filter a signal based on conviction.

        Returns:
            Tuple of (should_trade, modified_signal)
        """
        ticker = signal.get('ticker')
        conviction = self.get_conviction(ticker)

        if not conviction:
            # No conviction data - trade with default sizing
            signal['conviction_tier'] = 'UNKNOWN'
            signal['conviction_score'] = None
            signal['position_modifier'] = 1.0
            signal['conviction_note'] = 'No deep dive analysis available'
            return True, signal

        tier = conviction.get('conviction_tier', 'MEDIUM')
        score = conviction.get('conviction_score', 50)

        # Determine if we should trade
        if tier == 'AVOID':
            signal['conviction_tier'] = tier
            signal['conviction_score'] = score
            signal['position_modifier'] = 0
            signal['conviction_note'] = f'SKIPPED: Low conviction ({score}/100)'
            return False, signal

        # Modify position size based on conviction
        if tier == 'HIGH':
            modifier = 1.2 + (score - 75) / 100  # 1.2x to 1.45x
        elif tier == 'MEDIUM':
            modifier = 0.8 + (score - 50) / 100  # 0.8x to 1.05x
        else:  # LOW
            modifier = 0.5 + (score - 25) / 100  # 0.5x to 0.75x

        signal['conviction_tier'] = tier
        signal['conviction_score'] = score
        signal['position_modifier'] = round(modifier, 2)
        signal['conviction_note'] = f'{tier} conviction ({score}/100) - {modifier:.2f}x size'
        signal['bull_case'] = conviction.get('bull_case', '')
        signal['bear_case'] = conviction.get('bear_case', '')

        return True, signal

    def get_high_conviction_tickers(self) -> list:
        """Get list of HIGH conviction tickers."""
        return [
            ticker for ticker, data in self._cache.items()
            if data.get('conviction_tier') == 'HIGH'
        ]

    def get_summary(self) -> Dict:
        """Get summary of conviction database."""
        tiers = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'AVOID': 0}
        for data in self._cache.values():
            tier = data.get('conviction_tier', 'MEDIUM')
            tiers[tier] = tiers.get(tier, 0) + 1

        return {
            'total_stocks': len(self._cache),
            'tier_counts': tiers,
            'high_conviction': self.get_high_conviction_tickers()
        }


def integrate_with_scanner():
    """
    Example integration with Proteus signal scanner.

    This shows how to modify the daily workflow to use conviction filtering.
    """
    from src.trading.signal_scanner import SignalScanner

    # Initialize
    scanner = SignalScanner(lookback_days=90, min_signal_strength=50.0)
    conviction_filter = ConvictionFilter()

    # Get signals
    signals = scanner.scan_all_stocks()

    # Filter and modify based on conviction
    filtered_signals = []
    for signal in signals:
        should_trade, modified_signal = conviction_filter.filter_signal(signal)
        if should_trade:
            filtered_signals.append(modified_signal)
        else:
            print(f"[SKIP] {signal['ticker']}: {modified_signal['conviction_note']}")

    return filtered_signals


if __name__ == "__main__":
    # Test the filter
    cf = ConvictionFilter()
    summary = cf.get_summary()

    print("\nConviction Database Summary:")
    print(f"  Total stocks: {summary['total_stocks']}")
    print(f"  HIGH: {summary['tier_counts']['HIGH']}")
    print(f"  MEDIUM: {summary['tier_counts']['MEDIUM']}")
    print(f"  LOW: {summary['tier_counts']['LOW']}")
    print(f"  AVOID: {summary['tier_counts']['AVOID']}")

    print("\nHigh conviction stocks:")
    for ticker in summary['high_conviction'][:10]:
        conv = cf.get_conviction(ticker)
        print(f"  {ticker}: {conv['conviction_score']}/100")

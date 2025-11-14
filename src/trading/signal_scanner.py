"""
Real-Time Signal Scanner

Scans all stocks in the universe for mean reversion signals.
Applies regime filter, earnings filter, and stock-specific parameters.

Usage:
    scanner = SignalScanner()
    signals = scanner.scan_all_stocks()
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector
from src.config.mean_reversion_params import get_params, get_all_tickers


class SignalScanner:
    """
    Scan all stocks in universe for mean reversion signals.
    """

    def __init__(self, lookback_days=90):
        """
        Initialize signal scanner.

        Args:
            lookback_days: Days of historical data to fetch for indicator calculation
        """
        self.lookback_days = lookback_days
        self.tickers = get_all_tickers()

    def scan_all_stocks(self, date=None) -> List[Dict]:
        """
        Scan all stocks for signals.

        Args:
            date: Date to check for signals (default: today)

        Returns:
            List of signal dictionaries
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        signals = []

        print(f"Scanning {len(self.tickers)} stocks for signals on {date}...")
        print("=" * 70)

        for ticker in self.tickers:
            signal = self.scan_stock(ticker, date)
            if signal:
                signals.append(signal)
                print(f"[SIGNAL] {ticker}: {signal['signal_type']}")
            else:
                print(f"[NO SIGNAL] {ticker}")

        print("=" * 70)
        print(f"Total signals found: {len(signals)}")

        return signals

    def scan_stock(self, ticker: str, date: str = None) -> Dict:
        """
        Scan a single stock for signals.

        Args:
            ticker: Stock ticker
            date: Date to check (default: today)

        Returns:
            Signal dictionary if signal found, None otherwise
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        # Calculate start date for lookback period
        end_date = pd.to_datetime(date)
        start_date = end_date - timedelta(days=self.lookback_days)

        # Fetch data
        fetcher = YahooFinanceFetcher()
        try:
            data = fetcher.fetch_stock_data(
                ticker,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
        except Exception as e:
            print(f"[ERROR] Failed to fetch data for {ticker}: {e}")
            return None

        if len(data) < 60:
            print(f"[WARN] Insufficient data for {ticker}")
            return None

        # Engineer features
        engineer = TechnicalFeatureEngineer(fillna=True)
        enriched_data = engineer.engineer_features(data)

        # Get stock-specific parameters
        params = get_params(ticker)

        # Detect signals
        detector = MeanReversionDetector(
            z_score_threshold=params['z_score_threshold'],
            rsi_oversold=params['rsi_oversold'],
            rsi_overbought=65,
            volume_multiplier=params['volume_multiplier'],
            price_drop_threshold=params['price_drop_threshold']
        )

        signals = detector.detect_overcorrections(enriched_data)
        signals = detector.calculate_reversion_targets(signals)

        # Apply regime filter
        regime_detector = MarketRegimeDetector()
        signals = add_regime_filter_to_signals(signals, regime_detector)

        # Apply earnings filter
        earnings_fetcher = EarningsCalendarFetcher(
            exclusion_days_before=3,
            exclusion_days_after=3
        )
        signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

        # Check if signal on target date
        try:
            target_row = signals[signals['Date'] == pd.to_datetime(date)]
            if target_row.empty:
                # Try the most recent date if exact date not found
                target_row = signals.iloc[[-1]]
        except:
            target_row = signals.iloc[[-1]]

        if target_row.empty:
            return None

        has_signal = target_row.iloc[0]['panic_sell'] == 1

        if not has_signal:
            return None

        # Build signal dictionary
        row = target_row.iloc[0]

        return {
            'ticker': ticker,
            'date': row['Date'].strftime('%Y-%m-%d'),
            'signal_type': 'BUY',  # Mean reversion = buy the dip
            'price': float(row['Close']),
            'z_score': float(row['z_score']) if 'z_score' in row else None,
            'rsi': float(row['rsi']) if 'rsi' in row else None,
            'volume_ratio': float(row['volume'] / row['volume'].rolling(20).mean()) if 'volume' in row else None,
            'reversion_target': float(row['reversion_target']) if 'reversion_target' in row else None,
            'expected_return': float(row['expected_return']) if 'expected_return' in row else None,
            'parameters': params
        }

    def get_market_regime(self) -> str:
        """
        Get current market regime.

        Returns:
            'BULL', 'BEAR', or 'SIDEWAYS'
        """
        detector = MarketRegimeDetector()
        fetcher = YahooFinanceFetcher()

        # Get SPY data for regime detection
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        spy_data = fetcher.fetch_stock_data('SPY', start_date=start_date, end_date=end_date)
        regime = detector.detect_regime(spy_data)

        if not regime.empty and 'regime' in regime.columns:
            return regime['regime'].iloc[-1].upper()
        return 'UNKNOWN'


if __name__ == "__main__":
    # Example usage
    scanner = SignalScanner()

    print("Market Regime Check:")
    regime = scanner.get_market_regime()
    print(f"Current regime: {regime}")
    print()

    if regime == 'BEAR':
        print("[ALERT] Market is in BEAR regime - ALL trading disabled!")
        print("Strategy will not generate signals in bear markets.")
    else:
        print(f"[OK] Market is in {regime} regime - Trading enabled")
        print()

        # Scan for signals
        signals = scanner.scan_all_stocks()

        if signals:
            print()
            print("SIGNAL DETAILS:")
            print("=" * 70)
            for signal in signals:
                print(f"\nTicker: {signal['ticker']}")
                print(f"Date: {signal['date']}")
                print(f"Signal: {signal['signal_type']}")
                print(f"Price: ${signal['price']:.2f}")
                print(f"Z-score: {signal['z_score']:.2f}")
                print(f"RSI: {signal['rsi']:.1f}")
                print(f"Expected return: {signal['expected_return']:.2f}%")
        else:
            print()
            print("[INFO] No signals found today")

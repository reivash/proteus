"""
Real-Time Signal Scanner

Scans all stocks in the universe for mean reversion signals.
Applies validated filters for consistent performance:
- Regime filter (market conditions)
- Earnings filter (avoid unpredictable volatility, ±3 days)
- Dynamic position sizing (EXP-045: +41.7% return improvement!)

v15.0-EXPANDED (Current Version):
Based on comprehensive validation and MASSIVE portfolio expansion (EXP-047, EXP-048, EXP-050)
- MASSIVE EXPANSION: 27 → 45 stocks (+67%!) via EXP-050
- Win rate: ~79.3% maintained (all 45 stocks meet 70%+ threshold)
- Trade frequency: ~251 trades/year (+51 trades vs v14.1, +25% more opportunities!)
- Dynamic position sizing: +41.7% return improvement (EXP-045)
- Portfolio optimization: 100% (all 45 stocks optimized!)
- SEVEN stocks with 100% win rate: AVGO, TXN, JPM, ADI, NOW, ROAD, COP!
- New high performers: MS (88.9% WR), AIG (83.3% WR + 25.3% return), CAT (85.7% WR)
- Only proven, effective enhancements deployed

Position Sizing Strategy (v14.1 - EXP-045):
- Signal strength scoring based on z-score, RSI, volume, price drop
- LINEAR sizing: 0.5x + (strength/100) × 1.5x → range [0.5x, 2.0x]
- Strong signals (>70): 1.7-2.0x position
- Medium signals (40-70): 1.2-1.5x position
- Weak signals (<40): 0.5-0.9x position

Lessons from Failed Enhancements (v15.0, v16.0):
- Quality scoring (EXP-033): Theoretical only, hurt performance in practice
- VIX regime (EXP-034): No measurable benefit
- Validation is critical before deployment

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

    def __init__(self, lookback_days=90, min_signal_strength=None):
        """
        Initialize signal scanner.

        Args:
            lookback_days: Calendar days of historical data to fetch for indicator calculation
                         Default: 90 calendar days (~60-65 trading days after weekends/holidays)
            min_signal_strength: Minimum signal strength to trade (Q4 filter).
                               Set to 75th percentile value for Q4-only filtering.
                               Default: None (no filtering - trade all signals)
                               Example: 65.0 for Q4 threshold (EXP-093)

        Note: We need ~90 calendar days to reliably get 60+ trading days due to:
            - Weekends (28-30% reduction)
            - Market holidays (2-3% additional reduction)
            - This ensures indicators have sufficient data

        Q4 Filtering (EXP-093):
            - When enabled, only trade signals in top quartile (highest quality)
            - Expected impact: +14.1pp win rate (63.7% → 77.8%)
            - Reduces trade frequency by ~75% (quality over quantity)
        """
        self.lookback_days = lookback_days
        self.min_signal_strength = min_signal_strength
        self.tickers = get_all_tickers()

    def calculate_signal_strength(self, row: pd.Series, params: dict) -> float:
        """
        Calculate signal strength score (0-100) for position sizing.

        Based on EXP-045 validated formula:
        - Z-score magnitude (40% weight)
        - RSI oversold level (25% weight)
        - Volume spike (20% weight)
        - Price drop (15% weight)

        Args:
            row: Signal row with indicators
            params: Stock parameters

        Returns:
            Signal strength score (0-100)
        """
        # Get indicator values
        z_score = abs(row.get('z_score', 0))
        rsi = row.get('rsi', 35)

        # Calculate volume ratio
        volume = row.get('Volume', 0)
        volume_avg = row.get('volume_20d_avg', volume)
        volume_ratio = volume / volume_avg if volume_avg > 0 else 1.5

        price_drop = row.get('daily_return', -2.0)

        # Z-score component (0-100): Higher z-score = stronger signal
        z_component = min(100, (z_score / 2.5) * 100) * 0.40

        # RSI component (0-100): Lower RSI = stronger oversold signal
        rsi_threshold = params.get('rsi_oversold', 35)
        rsi_component = min(100, max(0, ((rsi_threshold - rsi) / 15) * 100)) * 0.25

        # Volume component (0-100): Higher volume spike = stronger signal
        vol_threshold = params.get('volume_multiplier', 1.3)
        volume_component = min(100, max(0, ((volume_ratio - vol_threshold) / 2.0) * 100)) * 0.20

        # Price drop component (0-100): Larger drop = stronger signal
        price_component = min(100, max(0, (abs(price_drop) / 5.0) * 100)) * 0.15

        total_score = z_component + rsi_component + volume_component + price_component
        return max(0, min(100, total_score))

    def calculate_position_size(self, signal_strength: float) -> float:
        """
        Calculate position size based on signal strength.

        LINEAR strategy (EXP-045 winner: +41.7% improvement):
        size = 0.5x + (strength/100) × 1.5x

        Args:
            signal_strength: Signal strength score (0-100)

        Returns:
            Position size multiplier (0.5-2.0)
        """
        position_size = 0.5 + (signal_strength / 100.0) * 1.5
        return max(0.5, min(2.0, position_size))

    def scan_all_stocks(self, date=None) -> List[Dict]:
        """
        Scan all stocks for signals.

        Args:
            date: Date to check for signals (default: today)

        Returns:
            List of signal dictionaries (filtered by min_signal_strength if enabled)
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

        # Apply Q4 filter if enabled (EXP-093)
        if self.min_signal_strength is not None and signals:
            pre_filter_count = len(signals)
            signals = [s for s in signals if s['signal_strength'] >= self.min_signal_strength]
            filtered_count = pre_filter_count - len(signals)

            print(f"\nQ4 Filter Applied (min_signal_strength >= {self.min_signal_strength:.1f}):")
            print(f"  Signals before filter: {pre_filter_count}")
            print(f"  Signals after filter: {len(signals)}")
            print(f"  Filtered out: {filtered_count} ({filtered_count/pre_filter_count*100:.1f}%)")
            print("=" * 70)

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

        # Apply earnings filter (v13.0 - VALIDATED)
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

        # Calculate signal strength and position size (v14.0 - EXP-045)
        signal_strength = self.calculate_signal_strength(row, params)
        position_size = self.calculate_position_size(signal_strength)

        # Calculate limit order price (EXP-080: 0.3% below close for better entry)
        close_price = float(row['Close'])
        limit_price = close_price * 0.997  # 0.3% discount
        low_price = float(row['Low']) if 'Low' in row else close_price

        return {
            'ticker': ticker,
            'date': row['Date'].strftime('%Y-%m-%d'),
            'signal_type': 'BUY',  # Mean reversion = buy the dip
            'price': close_price,
            'limit_price': limit_price,
            'intraday_low': low_price,
            'limit_would_fill': low_price <= limit_price,  # Whether limit order would have filled
            'z_score': float(row['z_score']) if 'z_score' in row else None,
            'rsi': float(row['rsi']) if 'rsi' in row else None,
            'volume_ratio': float(row['Volume'] / row.get('volume_20d_avg', row['Volume'])) if 'Volume' in row else None,
            'reversion_target': float(row['reversion_target']) if 'reversion_target' in row else None,
            'expected_return': float(row['expected_return']) if 'expected_return' in row else None,
            'signal_strength': float(signal_strength),
            'position_size': float(position_size),
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
                print(f"Signal strength: {signal['signal_strength']:.1f}/100")
                print(f"Position size: {signal['position_size']:.2f}x")
        else:
            print()
            print("[INFO] No signals found today")

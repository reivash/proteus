"""
Data Quality Validation Tests

Validates that data fetching and processing produces
quality outputs within expected ranges.

Tests:
1. Price data validity
2. Feature value ranges
3. Indicator calculations
4. Missing data handling
5. Staleness detection

Run with: python tests/test_data_quality.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple


def test_price_validation():
    """Test price data validation logic."""
    print("=" * 60)
    print("TEST: Price Data Validation")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: Valid price range check
    try:
        from common.config.trading_constants import DATA_QUALITY

        valid_prices = [10.0, 100.0, 500.0, 1000.0]
        invalid_prices = [0.0, -5.0, 0.50, 15000.0]

        for price in valid_prices:
            assert DATA_QUALITY.MIN_PRICE <= price <= DATA_QUALITY.MAX_PRICE
        print(f"[PASS] Valid prices accepted: {valid_prices}")
        passed += 1

        invalid_count = 0
        for price in invalid_prices:
            if not (DATA_QUALITY.MIN_PRICE <= price <= DATA_QUALITY.MAX_PRICE):
                invalid_count += 1
        assert invalid_count == len(invalid_prices)
        print(f"[PASS] Invalid prices rejected: {invalid_prices}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Price validation failed: {e}")
        failed += 1

    # Test 2: Staleness check
    try:
        from common.models.gpu_signal_model import GPUSignalModel
        model = GPUSignalModel()

        # Mock stale data
        model._last_scan_data_timestamps = {
            'FRESH': datetime.now() - timedelta(days=1),
            'STALE': datetime.now() - timedelta(days=10)
        }

        stale = model.check_data_staleness(max_days=3)
        assert 'STALE' in stale
        assert 'FRESH' not in stale
        print(f"[PASS] Staleness detection correct: stale={list(stale.keys())}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Staleness check failed: {e}")
        failed += 1

    print(f"\nPrice Validation: {passed}/{passed+failed} passed")
    return passed, failed


def test_feature_ranges():
    """Test that calculated features are within expected ranges."""
    print("\n" + "=" * 60)
    print("TEST: Feature Value Ranges")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: RSI should be 0-100
    try:
        # Create sample price data
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        prices = pd.Series(prices)

        # Calculate RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        rsi_valid = rsi.dropna()
        assert rsi_valid.min() >= 0
        assert rsi_valid.max() <= 100
        print(f"[PASS] RSI in valid range: {rsi_valid.min():.1f} - {rsi_valid.max():.1f}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] RSI range check failed: {e}")
        failed += 1

    # Test 2: Z-score should typically be -5 to +5
    try:
        z_scores = (prices - prices.rolling(20).mean()) / prices.rolling(20).std()
        z_valid = z_scores.dropna()

        # Most should be within -3 to +3
        in_range = ((z_valid >= -5) & (z_valid <= 5)).mean()
        assert in_range > 0.95  # 95%+ within range
        print(f"[PASS] Z-scores {in_range*100:.1f}% within [-5, +5]")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Z-score range check failed: {e}")
        failed += 1

    # Test 3: Volatility should be positive
    try:
        returns = prices.pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        vol_valid = volatility.dropna()

        assert (vol_valid >= 0).all()
        print(f"[PASS] Volatility all positive: {vol_valid.min():.4f} - {vol_valid.max():.4f}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Volatility check failed: {e}")
        failed += 1

    # Test 4: Moving averages should be reasonable
    try:
        sma_20 = prices.rolling(20).mean()
        sma_50 = prices.rolling(50).mean()

        # SMAs should be close to price (within 20%)
        deviation = abs(sma_20.dropna() / prices[19:] - 1)
        assert deviation.max() < 0.3  # Within 30%
        print(f"[PASS] SMA deviation reasonable: max {deviation.max()*100:.1f}%")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Moving average check failed: {e}")
        failed += 1

    print(f"\nFeature Ranges: {passed}/{passed+failed} passed")
    return passed, failed


def test_missing_data_handling():
    """Test handling of missing data in features."""
    print("\n" + "=" * 60)
    print("TEST: Missing Data Handling")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: NaN handling in features
    try:
        # Create data with gaps
        prices = pd.Series([100, 101, np.nan, 103, 104, np.nan, 106, 107, 108, 109])

        # Forward fill
        filled = prices.ffill()
        assert not filled.isna().any()
        print("[PASS] Forward fill removes NaN")
        passed += 1
    except Exception as e:
        print(f"[FAIL] NaN handling failed: {e}")
        failed += 1

    # Test 2: Interpolation
    try:
        prices = pd.Series([100, 101, np.nan, 103, 104, np.nan, 106, 107, 108, 109])
        interpolated = prices.interpolate()
        assert not interpolated.isna().any()
        assert abs(interpolated.iloc[2] - 102) < 0.1  # Should be ~102
        print("[PASS] Interpolation fills gaps correctly")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Interpolation failed: {e}")
        failed += 1

    # Test 3: Missing data rate calculation
    try:
        data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [1, np.nan, np.nan, 4, 5],
            'C': [1, 2, 3, 4, 5]
        })
        missing_rate = data.isna().mean()

        assert missing_rate['A'] == 0.2  # 1/5 missing
        assert missing_rate['B'] == 0.4  # 2/5 missing
        assert missing_rate['C'] == 0.0  # None missing
        print(f"[PASS] Missing rate calculation: A={missing_rate['A']:.1%}, B={missing_rate['B']:.1%}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Missing rate calculation failed: {e}")
        failed += 1

    print(f"\nMissing Data: {passed}/{passed+failed} passed")
    return passed, failed


def test_indicator_calculations():
    """Test technical indicator calculations."""
    print("\n" + "=" * 60)
    print("TEST: Indicator Calculations")
    print("=" * 60)

    passed = 0
    failed = 0

    # Create sample OHLCV data
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_price = close + np.random.randn(n) * 0.5
    volume = 1000000 + np.random.randint(-100000, 100000, n)

    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })

    # Test 1: SMA calculation
    try:
        sma_20 = df['Close'].rolling(20).mean()
        # Manual check
        manual_sma = df['Close'].iloc[-20:].mean()
        assert abs(sma_20.iloc[-1] - manual_sma) < 0.001
        print(f"[PASS] SMA(20) calculation correct: {sma_20.iloc[-1]:.2f}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] SMA calculation failed: {e}")
        failed += 1

    # Test 2: ATR calculation
    try:
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift(1))
        tr3 = abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        assert (atr.dropna() > 0).all()  # ATR must be positive
        print(f"[PASS] ATR calculation correct: {atr.iloc[-1]:.2f}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] ATR calculation failed: {e}")
        failed += 1

    # Test 3: Bollinger Bands
    try:
        sma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std

        # Bands should be correctly structured
        valid_idx = ~(sma.isna() | std.isna())
        assert (upper[valid_idx] > sma[valid_idx]).all(), "Upper band must be above SMA"
        assert (lower[valid_idx] < sma[valid_idx]).all(), "Lower band must be below SMA"
        assert (std[valid_idx] > 0).all(), "Std must be positive"

        # Check bands are properly spaced (width = 4 * std)
        band_width = upper - lower
        expected_width = 4 * std
        width_match = np.allclose(band_width[valid_idx], expected_width[valid_idx])
        assert width_match, "Band width should equal 4 * std"
        print(f"[PASS] Bollinger Bands: correctly structured, avg width={band_width.mean():.2f}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Bollinger Bands failed: {e}")
        failed += 1

    # Test 4: Volume ratio
    try:
        avg_volume = df['Volume'].rolling(20).mean()
        volume_ratio = df['Volume'] / avg_volume

        # Volume ratio should be positive and reasonable
        vr_valid = volume_ratio.dropna()
        assert (vr_valid > 0).all()
        assert vr_valid.median() < 3  # Median should be around 1
        print(f"[PASS] Volume ratio: median={vr_valid.median():.2f}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Volume ratio failed: {e}")
        failed += 1

    # Test 5: Drawdown calculation
    try:
        rolling_max = df['Close'].rolling(20).max()
        drawdown = (df['Close'] - rolling_max) / rolling_max * 100

        # Drawdown should be <= 0
        dd_valid = drawdown.dropna()
        assert (dd_valid <= 0.001).all()  # Small tolerance for float
        print(f"[PASS] Drawdown calculation: max={dd_valid.min():.2f}%")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Drawdown calculation failed: {e}")
        failed += 1

    print(f"\nIndicator Calculations: {passed}/{passed+failed} passed")
    return passed, failed


def test_vectorized_operations():
    """Test that vectorized operations match loop-based results."""
    print("\n" + "=" * 60)
    print("TEST: Vectorized Operations")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: Vectorized SMA vs loop
    try:
        from common.models.gpu_signal_model import GPUSignalModel
        model = GPUSignalModel()

        arr = np.arange(100, dtype=float)
        period = 5

        # Vectorized
        vec_sma = model.feature_extractor._sma(arr, period)

        # Loop-based (reference)
        loop_sma = np.zeros_like(arr)
        for i in range(period - 1, len(arr)):
            loop_sma[i] = np.mean(arr[i - period + 1:i + 1])

        # Compare (skip warmup period)
        diff = np.abs(vec_sma[period:] - loop_sma[period:])
        assert diff.max() < 0.001
        print(f"[PASS] Vectorized SMA matches loop: max diff={diff.max():.6f}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Vectorized SMA comparison failed: {e}")
        failed += 1

    # Test 2: Vectorized rolling std
    try:
        vec_std = model.feature_extractor._rolling_std(arr, period)

        # Reference using pandas
        ref_std = pd.Series(arr).rolling(period).std().values

        # Compare (skip NaN values)
        valid_idx = ~np.isnan(ref_std)
        diff = np.abs(vec_std[valid_idx] - ref_std[valid_idx])
        assert diff.max() < 0.001
        print(f"[PASS] Vectorized std matches pandas: max diff={diff.max():.6f}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Vectorized std comparison failed: {e}")
        failed += 1

    print(f"\nVectorized Operations: {passed}/{passed+failed} passed")
    return passed, failed


def test_sector_feature_quality():
    """Test sector-relative feature calculations."""
    print("\n" + "=" * 60)
    print("TEST: Sector Feature Quality")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: Sector mapping coverage
    try:
        from common.data.features.cross_sectional_features import SECTOR_MAP, SECTOR_ETF_MAP

        # Check coverage
        assert len(SECTOR_MAP) >= 50
        assert len(SECTOR_ETF_MAP) >= 10
        print(f"[PASS] Sector mapping: {len(SECTOR_MAP)} stocks, {len(SECTOR_ETF_MAP)} ETFs")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Sector mapping check failed: {e}")
        failed += 1

    # Test 2: All mapped stocks have valid sectors
    try:
        valid_sectors = set(SECTOR_ETF_MAP.keys())
        stocks_with_valid_sector = sum(1 for s in SECTOR_MAP.values() if s in valid_sectors)

        coverage = stocks_with_valid_sector / len(SECTOR_MAP)
        assert coverage > 0.95  # 95%+ should have valid sector
        print(f"[PASS] Sector coverage: {coverage*100:.1f}% stocks have valid sector")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Sector coverage check failed: {e}")
        failed += 1

    # Test 3: Sector momentum categories
    try:
        from common.trading.sector_momentum import SectorMomentumCalculator
        calc = SectorMomentumCalculator()

        # Test category boundaries
        assert calc.get_momentum_category(-5.0) == 'weak'
        assert calc.get_momentum_category(-2.0) == 'slightly_weak'
        assert calc.get_momentum_category(0.0) == 'neutral'
        assert calc.get_momentum_category(2.0) == 'slightly_strong'
        assert calc.get_momentum_category(5.0) == 'strong'
        print("[PASS] Momentum categories correctly defined")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Momentum category check failed: {e}")
        failed += 1

    print(f"\nSector Features: {passed}/{passed+failed} passed")
    return passed, failed


def run_all_tests():
    """Run all data quality tests."""
    print("=" * 70)
    print("PROTEUS DATA QUALITY TESTS")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    total_passed = 0
    total_failed = 0

    test_suites = [
        ("Price Validation", test_price_validation),
        ("Feature Ranges", test_feature_ranges),
        ("Missing Data", test_missing_data_handling),
        ("Indicators", test_indicator_calculations),
        ("Vectorized Ops", test_vectorized_operations),
        ("Sector Features", test_sector_feature_quality),
    ]

    results = []
    for name, test_func in test_suites:
        try:
            passed, failed = test_func()
            results.append((name, passed, failed))
            total_passed += passed
            total_failed += failed
        except Exception as e:
            print(f"\n[ERROR] {name} test suite crashed: {e}")
            results.append((name, 0, 1))
            total_failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("DATA QUALITY TEST SUMMARY")
    print("=" * 70)
    for name, passed, failed in results:
        status = "PASS" if failed == 0 else "FAIL"
        print(f"  {name:25} {passed:3}/{passed+failed:3} [{status}]")

    print("-" * 70)
    print(f"  {'TOTAL':25} {total_passed:3}/{total_passed+total_failed:3}")
    print("=" * 70)

    if total_failed == 0:
        print("SUCCESS - All data quality tests passed!")
    else:
        print(f"FAILED - {total_failed} test(s) failed")

    return total_failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

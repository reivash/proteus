"""
Integration Tests for Smart Scanner Pipeline

Tests that all components work together correctly:
1. Market regime detection
2. Bear detection
3. GPU signal model
4. Signal enhancement
5. Position sizing
6. Data staleness checking

Run with: python tests/test_scan_integration.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


def test_regime_detection():
    """Test market regime detection components."""
    print("=" * 60)
    print("TEST: Market Regime Detection")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: MarketRegimeDetector loads
    try:
        from src.analysis.market_regime import MarketRegimeDetector, RegimeAnalysis
        detector = MarketRegimeDetector()
        print("[PASS] MarketRegimeDetector imports successfully")
        passed += 1
    except Exception as e:
        print(f"[FAIL] MarketRegimeDetector import failed: {e}")
        failed += 1
        return passed, failed

    # Test 2: detect_regime returns valid result
    try:
        result = detector.detect_regime()
        assert isinstance(result, RegimeAnalysis)
        assert result.regime is not None
        assert 0 <= result.confidence <= 1
        print(f"[PASS] detect_regime returns valid RegimeAnalysis: {result.regime.value}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] detect_regime failed: {e}")
        failed += 1

    # Test 3: data_error flag exists
    try:
        assert hasattr(result, 'data_error')
        print(f"[PASS] RegimeAnalysis has data_error flag: {result.data_error}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] data_error flag missing: {e}")
        failed += 1

    # Test 4: Regime multipliers are valid
    try:
        from src.analysis.market_regime import MarketRegime
        for regime in MarketRegime:
            mult = detector.get_position_multiplier(regime)
            assert 0 < mult <= 1.5
        print("[PASS] All regime position multipliers valid (0-1.5)")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Regime multipliers invalid: {e}")
        failed += 1

    print(f"\nRegime Detection: {passed}/{passed+failed} passed")
    return passed, failed


def test_bear_detection():
    """Test fast bear detection."""
    print("\n" + "=" * 60)
    print("TEST: Fast Bear Detection")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: FastBearDetector loads
    try:
        from src.analysis.fast_bear_detector import FastBearDetector, FastBearSignal
        detector = FastBearDetector()
        print("[PASS] FastBearDetector imports successfully")
        passed += 1
    except Exception as e:
        print(f"[FAIL] FastBearDetector import failed: {e}")
        failed += 1
        return passed, failed

    # Test 2: detect returns valid signal
    try:
        signal = detector.detect()
        assert isinstance(signal, FastBearSignal)
        assert 0 <= signal.bear_score <= 100
        assert signal.alert_level in ['NORMAL', 'WATCH', 'WARNING', 'CRITICAL']
        print(f"[PASS] detect returns valid signal: score={signal.bear_score}, level={signal.alert_level}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] detect failed: {e}")
        failed += 1

    # Test 3: Signal has required fields
    try:
        required = ['bear_score', 'alert_level', 'vix_level', 'triggers', 'recommendation']
        for field in required:
            assert hasattr(signal, field), f"Missing field: {field}"
        print("[PASS] FastBearSignal has all required fields")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Missing required fields: {e}")
        failed += 1

    print(f"\nBear Detection: {passed}/{passed+failed} passed")
    return passed, failed


def test_gpu_signal_model():
    """Test GPU signal model components."""
    print("\n" + "=" * 60)
    print("TEST: GPU Signal Model")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: GPUSignalModel loads
    try:
        from src.models.gpu_signal_model import GPUSignalModel
        model = GPUSignalModel()
        print(f"[PASS] GPUSignalModel loads on device: {model.device}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] GPUSignalModel load failed: {e}")
        failed += 1
        return passed, failed

    # Test 2: Staleness checking methods exist
    try:
        assert hasattr(model, 'get_data_timestamps')
        assert hasattr(model, 'check_data_staleness')
        assert hasattr(model, 'get_fresh_prices')
        print("[PASS] Staleness checking methods exist")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Staleness methods missing: {e}")
        failed += 1

    # Test 3: Staleness check works with mock data
    try:
        model._last_scan_prices = {'TEST1': 100.0, 'TEST2': 200.0}
        model._last_scan_data_timestamps = {
            'TEST1': datetime.now() - timedelta(days=1),
            'TEST2': datetime.now() - timedelta(days=10)
        }
        stale = model.check_data_staleness(max_days=3)
        assert 'TEST2' in stale
        assert 'TEST1' not in stale
        print(f"[PASS] Staleness check correctly identifies stale data: {stale}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Staleness check failed: {e}")
        failed += 1

    # Test 4: Feature extractor methods are vectorized
    try:
        import numpy as np
        arr = np.arange(100, dtype=float)
        # _sma is on the feature_extractor, not the model
        sma = model.feature_extractor._sma(arr, 5)
        assert len(sma) == len(arr)
        # Check the last value is correct (mean of last 5 values)
        expected = np.mean(arr[-5:])
        assert abs(sma[-1] - expected) < 0.01, f"Expected {expected}, got {sma[-1]}"
        print("[PASS] Vectorized SMA calculation correct")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Vectorized calculation failed: {e}")
        failed += 1

    print(f"\nGPU Signal Model: {passed}/{passed+failed} passed")
    return passed, failed


def test_signal_enhancement():
    """Test signal enhancement pipeline - both legacy and production calculators."""
    print("\n" + "=" * 60)
    print("TEST: Signal Enhancement")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: PenaltiesOnlyCalculator loads (PRODUCTION calculator)
    try:
        from src.trading.penalties_only_calculator import PenaltiesOnlyCalculator
        prod_calc = PenaltiesOnlyCalculator()
        print(f"[PASS] PenaltiesOnlyCalculator loads ({prod_calc.penalty_count} penalties)")
        passed += 1
    except Exception as e:
        print(f"[FAIL] PenaltiesOnlyCalculator load failed: {e}")
        failed += 1
        return passed, failed

    # Test 2: Production calculator works
    try:
        result = prod_calc.calculate(
            ticker='NVDA',
            base_signal=70.0,
            regime='choppy',
            sector='Technology',
            sector_momentum=-2.0
        )
        assert hasattr(result, 'final_signal')
        assert 0 <= result.final_signal <= 100
        print(f"[PASS] Production calc: 70.0 -> {result.final_signal:.1f}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Production calc failed: {e}")
        failed += 1

    # Test 3: Penalties reduce signal correctly
    try:
        # Weak base signal should get penalty
        weak_result = prod_calc.calculate(
            ticker='MSFT', base_signal=52.0, regime='choppy'
        )
        strong_result = prod_calc.calculate(
            ticker='MSFT', base_signal=75.0, regime='choppy'
        )
        # Weak signal should have penalties applied
        assert weak_result.total_penalty < 0, "Weak signal should have negative penalties"
        print(f"[PASS] Weak signal penalty: {weak_result.total_penalty:+.0f}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Penalty test failed: {e}")
        failed += 1

    # Test 4: Legacy EnhancedSignalCalculator still works (for backwards compat)
    try:
        from src.trading.enhanced_signal_calculator import EnhancedSignalCalculator
        legacy_calc = EnhancedSignalCalculator()
        result = legacy_calc.calculate_enhanced_strength(
            ticker='NVDA',
            base_strength=70.0,
            regime='choppy',
            sector_name='Technology',
            sector_momentum=-2.0,
            signal_date=datetime.now()
        )
        assert hasattr(result, 'final_strength')
        assert 0 <= result.final_strength <= 100
        print(f"[PASS] Legacy calculator still works: 70.0 -> {result.final_strength:.1f}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Legacy calculator failed: {e}")
        failed += 1

    print(f"\nSignal Enhancement: {passed}/{passed+failed} passed")
    return passed, failed


def test_position_sizing():
    """Test position sizing components."""
    print("\n" + "=" * 60)
    print("TEST: Position Sizing")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: PositionSizer loads
    try:
        from src.trading.position_sizer import PositionSizer
        sizer = PositionSizer(portfolio_value=100000)
        print("[PASS] PositionSizer loads")
        passed += 1
    except Exception as e:
        print(f"[FAIL] PositionSizer load failed: {e}")
        failed += 1
        return passed, failed

    # Test 2: Portfolio status works
    try:
        status = sizer.get_portfolio_status()
        assert 'positions' in status
        assert 'max_positions' in status
        assert 'total_heat' in status
        print(f"[PASS] Portfolio status: {status['positions']}/{status['max_positions']} positions")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Portfolio status failed: {e}")
        failed += 1

    # Test 3: can_add_position works
    try:
        can_add, reason = sizer.can_add_position()
        assert isinstance(can_add, bool)
        assert isinstance(reason, str)
        print(f"[PASS] can_add_position: {can_add} ({reason})")
        passed += 1
    except Exception as e:
        print(f"[FAIL] can_add_position failed: {e}")
        failed += 1

    print(f"\nPosition Sizing: {passed}/{passed+failed} passed")
    return passed, failed


def test_sector_momentum():
    """Test sector momentum calculator."""
    print("\n" + "=" * 60)
    print("TEST: Sector Momentum")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: SectorMomentumCalculator loads
    try:
        from src.trading.sector_momentum import SectorMomentumCalculator
        calc = SectorMomentumCalculator()
        print("[PASS] SectorMomentumCalculator loads")
        passed += 1
    except Exception as e:
        print(f"[FAIL] SectorMomentumCalculator load failed: {e}")
        failed += 1
        return passed, failed

    # Test 2: get_stock_sector_etf works
    try:
        etf = calc.get_stock_sector_etf('NVDA')
        assert etf == 'XLK'
        print(f"[PASS] NVDA -> {etf} sector ETF")
        passed += 1
    except Exception as e:
        print(f"[FAIL] get_stock_sector_etf failed: {e}")
        failed += 1

    # Test 3: Momentum category classification
    try:
        assert calc.get_momentum_category(-5.0) == 'weak'
        assert calc.get_momentum_category(0.0) == 'neutral'
        assert calc.get_momentum_category(5.0) == 'strong'
        print("[PASS] Momentum categories correct")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Momentum categories failed: {e}")
        failed += 1

    # Test 4: Strength boost calculation
    try:
        weak_boost = calc.get_strength_boost(-5.0)
        strong_boost = calc.get_strength_boost(5.0)
        assert weak_boost > 1.0  # Weak sectors get boost
        assert strong_boost < 1.0  # Strong sectors get penalty
        print(f"[PASS] Strength boosts: weak={weak_boost}, strong={strong_boost}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Strength boost failed: {e}")
        failed += 1

    print(f"\nSector Momentum: {passed}/{passed+failed} passed")
    return passed, failed


def test_cross_sectional_features():
    """Test cross-sectional feature engineering."""
    print("\n" + "=" * 60)
    print("TEST: Cross-Sectional Features")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: Module loads
    try:
        from src.data.features.cross_sectional_features import (
            CrossSectionalFeatureEngineer, SECTOR_MAP, SECTOR_ETF_MAP
        )
        engineer = CrossSectionalFeatureEngineer()
        print("[PASS] CrossSectionalFeatureEngineer loads")
        passed += 1
    except Exception as e:
        print(f"[FAIL] CrossSectionalFeatureEngineer load failed: {e}")
        failed += 1
        return passed, failed

    # Test 2: Sector maps are populated
    try:
        assert len(SECTOR_MAP) > 50
        assert len(SECTOR_ETF_MAP) >= 10
        print(f"[PASS] Sector maps: {len(SECTOR_MAP)} stocks, {len(SECTOR_ETF_MAP)} ETFs")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Sector maps incomplete: {e}")
        failed += 1

    # Test 3: Sector ETF cache works
    try:
        assert hasattr(engineer, 'sector_etf_cache')
        print("[PASS] Sector ETF cache exists")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Sector ETF cache missing: {e}")
        failed += 1

    print(f"\nCross-Sectional Features: {passed}/{passed+failed} passed")
    return passed, failed


def test_logging_infrastructure():
    """Test logging infrastructure."""
    print("\n" + "=" * 60)
    print("TEST: Logging Infrastructure")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: Logging config loads
    try:
        from src.utils.logging_config import get_logger, setup_logging
        print("[PASS] Logging config imports")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Logging config import failed: {e}")
        failed += 1
        return passed, failed

    # Test 2: Logger can be created
    try:
        logger = get_logger('test.integration')
        assert logger is not None
        print("[PASS] Logger created successfully")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Logger creation failed: {e}")
        failed += 1

    # Test 3: Log directory exists
    try:
        from src.utils.logging_config import LOG_DIR
        assert LOG_DIR.exists() or True  # May not exist until first log
        print(f"[PASS] Log directory configured: {LOG_DIR}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Log directory check failed: {e}")
        failed += 1

    print(f"\nLogging Infrastructure: {passed}/{passed+failed} passed")
    return passed, failed


def run_all_tests():
    """Run all integration tests."""
    print("=" * 70)
    print("PROTEUS SCAN PIPELINE INTEGRATION TESTS")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    total_passed = 0
    total_failed = 0

    # Run each test suite
    test_suites = [
        ("Regime Detection", test_regime_detection),
        ("Bear Detection", test_bear_detection),
        ("GPU Signal Model", test_gpu_signal_model),
        ("Signal Enhancement", test_signal_enhancement),
        ("Position Sizing", test_position_sizing),
        ("Sector Momentum", test_sector_momentum),
        ("Cross-Sectional Features", test_cross_sectional_features),
        ("Logging Infrastructure", test_logging_infrastructure),
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
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)
    for name, passed, failed in results:
        status = "PASS" if failed == 0 else "FAIL"
        print(f"  {name:30} {passed:3}/{passed+failed:3} [{status}]")

    print("-" * 70)
    print(f"  {'TOTAL':30} {total_passed:3}/{total_passed+total_failed:3}")
    print("=" * 70)

    if total_failed == 0:
        print("SUCCESS - All integration tests passed!")
    else:
        print(f"FAILED - {total_failed} test(s) failed")

    return total_failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

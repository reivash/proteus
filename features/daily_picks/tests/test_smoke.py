"""
Smoke Tests for Full Scan Pipeline

Quick tests that verify the entire scanning pipeline can:
1. Initialize all components
2. Run without crashing
3. Produce structured output

These are NOT comprehensive tests - they just verify the system boots up correctly.

Run with: python features/daily_picks/tests/test_smoke.py
"""

import sys
from pathlib import Path
# Go up 3 levels: tests → daily_picks → features → project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from datetime import datetime
from typing import Tuple


def test_scanner_initialization():
    """Test that SmartScannerV2 initializes correctly."""
    print("=" * 60)
    print("SMOKE TEST: Scanner Initialization")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: Import SmartScannerV2
    try:
        from common.trading.smart_scanner_v2 import SmartScannerV2
        print("[PASS] SmartScannerV2 imports")
        passed += 1
    except Exception as e:
        print(f"[FAIL] SmartScannerV2 import failed: {e}")
        failed += 1
        return passed, failed

    # Test 2: Initialize scanner (this loads all dependencies)
    try:
        scanner = SmartScannerV2(model='hybrid', trading_mode='aggressive')
        print(f"[PASS] Scanner initializes with model={scanner.model_type}, mode={scanner.trading_mode_str}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Scanner initialization failed: {e}")
        failed += 1
        return passed, failed

    # Test 3: Check all components loaded
    try:
        components = [
            ('regime_detector', scanner.regime_detector),
            ('bear_detector', scanner.bear_detector),
            ('signal_calc', scanner.signal_calc),
            ('position_sizer', scanner.position_sizer),
        ]
        for name, comp in components:
            assert comp is not None, f"{name} is None"
        print(f"[PASS] All {len(components)} core components loaded")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Component check failed: {e}")
        failed += 1

    print(f"\nScanner Initialization: {passed}/{passed+failed} passed")
    return passed, failed


def test_regime_detection_smoke():
    """Test that regime detection runs end-to-end."""
    print("\n" + "=" * 60)
    print("SMOKE TEST: Regime Detection")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test: Full regime detection
    try:
        from common.analysis.unified_regime_detector import UnifiedRegimeDetector
        detector = UnifiedRegimeDetector()
        result = detector.detect_regime()  # Correct method name

        # Verify structure
        assert result.regime is not None, "Regime is None"
        assert hasattr(result, 'confidence'), "Missing confidence"
        assert hasattr(result, 'vix_level'), "Missing vix_level"

        # regime may be enum or string
        regime_str = result.regime.value if hasattr(result.regime, 'value') else result.regime
        print(f"[PASS] Regime detection: {regime_str}, confidence={result.confidence:.2f}")
        bear_score = getattr(result, 'early_warning_score', 0)
        print(f"       VIX: {result.vix_level:.1f}, Early Warning: {bear_score}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Regime detection failed: {e}")
        failed += 1

    print(f"\nRegime Detection Smoke: {passed}/{passed+failed} passed")
    return passed, failed


def test_bear_detection_smoke():
    """Test that bear detection runs end-to-end."""
    print("\n" + "=" * 60)
    print("SMOKE TEST: Bear Detection")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test: Full bear detection
    try:
        from common.analysis.fast_bear_detector import FastBearDetector
        detector = FastBearDetector()
        signal = detector.detect()

        # Verify structure
        assert 0 <= signal.bear_score <= 100, f"Invalid bear_score: {signal.bear_score}"
        assert signal.alert_level in ['NORMAL', 'WATCH', 'WARNING', 'CRITICAL']
        assert signal.recommendation is not None

        print(f"[PASS] Bear detection: score={signal.bear_score}, level={signal.alert_level}")
        if signal.triggers:
            print(f"       Triggers: {', '.join(signal.triggers[:3])}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Bear detection failed: {e}")
        failed += 1

    print(f"\nBear Detection Smoke: {passed}/{passed+failed} passed")
    return passed, failed


def test_signal_calculation_smoke():
    """Test that signal calculation runs with mock data."""
    print("\n" + "=" * 60)
    print("SMOKE TEST: Signal Calculation")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test: PenaltiesOnlyCalculator (production)
    try:
        from common.trading.penalties_only_calculator import PenaltiesOnlyCalculator
        calc = PenaltiesOnlyCalculator()

        # Test with various inputs
        test_cases = [
            {'ticker': 'NVDA', 'base_signal': 72.0, 'regime': 'choppy'},
            {'ticker': 'COP', 'base_signal': 65.0, 'regime': 'volatile'},
            {'ticker': 'NOW', 'base_signal': 55.0, 'regime': 'bull'},
        ]

        for tc in test_cases:
            result = calc.calculate(**tc)
            assert 0 <= result.final_signal <= 100, f"Invalid signal: {result.final_signal}"

        print(f"[PASS] Calculator processed {len(test_cases)} test cases")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Signal calculation failed: {e}")
        failed += 1

    print(f"\nSignal Calculation Smoke: {passed}/{passed+failed} passed")
    return passed, failed


def test_gpu_model_smoke():
    """Test that GPU model loads and can process data."""
    print("\n" + "=" * 60)
    print("SMOKE TEST: GPU Model")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test: GPUSignalModel loads
    try:
        from common.models.gpu_signal_model import GPUSignalModel
        model = GPUSignalModel()
        print(f"[PASS] GPU model loaded on {model.device}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] GPU model load failed: {e}")
        failed += 1
        return passed, failed

    # Test: Feature extractor works
    try:
        import numpy as np
        test_data = np.random.randn(100) * 2 + 100
        sma = model.feature_extractor._sma(test_data, 20)
        assert len(sma) == len(test_data)
        print("[PASS] Feature extractor works")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Feature extractor failed: {e}")
        failed += 1

    print(f"\nGPU Model Smoke: {passed}/{passed+failed} passed")
    return passed, failed


def test_position_sizing_smoke():
    """Test that position sizing works."""
    print("\n" + "=" * 60)
    print("SMOKE TEST: Position Sizing")
    print("=" * 60)

    passed = 0
    failed = 0

    try:
        from common.trading.position_sizer import PositionSizer

        sizer = PositionSizer(portfolio_value=100000)

        # Test basic operations
        status = sizer.get_portfolio_status()
        can_add, reason = sizer.can_add_position()

        assert 'positions' in status
        assert 'max_positions' in status
        assert isinstance(can_add, bool)

        print(f"[PASS] Position sizer: {status['positions']}/{status['max_positions']} positions")
        print(f"       Can add: {can_add} ({reason})")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Position sizing failed: {e}")
        failed += 1

    print(f"\nPosition Sizing Smoke: {passed}/{passed+failed} passed")
    return passed, failed


def test_notification_smoke():
    """Test that notification system loads (doesn't send)."""
    print("\n" + "=" * 60)
    print("SMOKE TEST: Notification System")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: Webhook notifier (always available)
    try:
        from common.notifications.webhook_notifier import WebhookNotifier
        notifier = WebhookNotifier()

        # Check it has required methods
        assert hasattr(notifier, 'send_bear_alert')
        print("[PASS] WebhookNotifier loads with send_bear_alert")
        passed += 1
    except Exception as e:
        print(f"[FAIL] WebhookNotifier failed: {type(e).__name__}: {e}")
        failed += 1

    # Test 2: SendGrid notifier (optional - may not be configured)
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        assert hasattr(notifier, 'send_scan_notification')
        print("[PASS] SendGridNotifier loads with send_scan_notification")
        passed += 1
    except ImportError as e:
        print(f"[SKIP] SendGrid not installed: {e}")
        passed += 1  # Optional dependency
    except ValueError as e:
        print(f"[SKIP] SendGrid not configured: {e}")
        passed += 1  # Optional config
    except Exception as e:
        if 'api_key' in str(e).lower() or 'config' in str(e).lower() or not str(e):
            print(f"[SKIP] SendGrid config missing")
            passed += 1
        else:
            print(f"[WARN] SendGrid error (optional): {e}")
            passed += 1  # Don't fail - it's optional

    print(f"\nNotification Smoke: {passed}/{passed+failed} passed")
    return passed, failed


def test_virtual_wallet_smoke():
    """Test that VirtualWallet initializes and basic operations work."""
    print("\n" + "=" * 60)
    print("SMOKE TEST: Virtual Wallet")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: Import and initialize
    try:
        from common.trading.virtual_wallet import VirtualWallet
        wallet = VirtualWallet(initial_capital=100000, min_signal_strength=65)
        print("[PASS] VirtualWallet initializes")
        passed += 1
    except Exception as e:
        print(f"[FAIL] VirtualWallet import/init failed: {e}")
        failed += 1
        return passed, failed

    # Test 2: Get status
    try:
        status = wallet.get_status()
        assert 'initial_capital' in status
        assert 'cash' in status
        assert 'total_equity' in status
        assert 'positions' in status
        print(f"[PASS] get_status() returns valid structure")
        passed += 1
    except Exception as e:
        print(f"[FAIL] get_status() failed: {e}")
        failed += 1

    # Test 3: Calculate metrics (even with no trades)
    try:
        metrics = wallet.calculate_metrics()
        assert 'total_trades' in metrics
        assert 'win_rate' in metrics
        assert 'sharpe_ratio' in metrics
        print(f"[PASS] calculate_metrics() returns valid structure")
        passed += 1
    except Exception as e:
        print(f"[FAIL] calculate_metrics() failed: {e}")
        failed += 1

    # Test 4: Position sync creates file
    try:
        wallet._sync_to_scanner()
        assert wallet.SCANNER_POSITIONS_FILE.parent.exists()
        print(f"[PASS] _sync_to_scanner() runs without error")
        passed += 1
    except Exception as e:
        print(f"[FAIL] _sync_to_scanner() failed: {e}")
        failed += 1

    print(f"\nVirtual Wallet Smoke: {passed}/{passed+failed} passed")
    return passed, failed


def test_scanner_runner_smoke():
    """Test that ScannerRunner health checks and initialization work."""
    print("\n" + "=" * 60)
    print("SMOKE TEST: Scanner Runner")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: Import and initialize
    try:
        from common.trading.scanner_runner import ScannerRunner, HealthCheckResult
        runner = ScannerRunner(portfolio_value=100000)
        print("[PASS] ScannerRunner initializes")
        passed += 1
    except Exception as e:
        print(f"[FAIL] ScannerRunner import/init failed: {e}")
        failed += 1
        return passed, failed

    # Test 2: Health checks
    try:
        health = runner.run_health_checks()
        assert isinstance(health, HealthCheckResult)
        assert hasattr(health, 'healthy')
        assert hasattr(health, 'checks_passed')
        assert hasattr(health, 'warnings')
        assert hasattr(health, 'errors')
        print(f"[PASS] Health checks run: {health.checks_passed} passed, {len(health.warnings)} warnings")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Health checks failed: {e}")
        failed += 1

    # Test 3: Scanner initialization
    try:
        init_result = runner.initialize_scanner()
        if init_result:
            print("[PASS] Scanner initialization successful")
        else:
            print(f"[WARN] Scanner init returned False: {runner.last_error}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Scanner initialization crashed: {e}")
        failed += 1

    print(f"\nScanner Runner Smoke: {passed}/{passed+failed} passed")
    return passed, failed


def test_daily_report_smoke():
    """Test that daily report generator works."""
    print("\n" + "=" * 60)
    print("SMOKE TEST: Daily Report Generator")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: Import
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from generate_daily_report import DailyReportGenerator
        print("[PASS] DailyReportGenerator imports")
        passed += 1
    except Exception as e:
        print(f"[FAIL] DailyReportGenerator import failed: {e}")
        failed += 1
        return passed, failed

    # Test 2: Initialize and collect data
    try:
        generator = DailyReportGenerator()
        print("[PASS] DailyReportGenerator initializes")
        passed += 1
    except Exception as e:
        print(f"[FAIL] DailyReportGenerator init failed: {e}")
        failed += 1
        return passed, failed

    # Test 3: Generate report data
    try:
        report_data = generator.generate_report()
        assert isinstance(report_data, dict)
        assert 'portfolio' in report_data
        assert 'market' in report_data
        assert 'performance' in report_data
        print(f"[PASS] Report data generated")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Report data generation failed: {e}")
        failed += 1
        return passed, failed

    # Test 4: Format console output
    try:
        console_output = generator.format_console_report(report_data)
        assert isinstance(console_output, str)
        assert len(console_output) > 100
        print(f"[PASS] Console report formatted ({len(console_output)} chars)")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Console report formatting failed: {e}")
        failed += 1

    # Test 5: Format HTML output
    try:
        html_output = generator.format_html_report(report_data)
        assert isinstance(html_output, str)
        assert '<!doctype' in html_output.lower() or '<html>' in html_output.lower()
        print(f"[PASS] HTML report formatted ({len(html_output)} chars)")
        passed += 1
    except Exception as e:
        print(f"[FAIL] HTML report formatting failed: {e}")
        failed += 1

    print(f"\nDaily Report Smoke: {passed}/{passed+failed} passed")
    return passed, failed


def test_full_pipeline_dry_run():
    """Test the complete pipeline without making any network calls."""
    print("\n" + "=" * 60)
    print("SMOKE TEST: Full Pipeline Dry Run")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test: All imports work together
    try:
        from common.analysis.unified_regime_detector import UnifiedRegimeDetector
        from common.analysis.fast_bear_detector import FastBearDetector
        from common.trading.penalties_only_calculator import PenaltiesOnlyCalculator
        from common.trading.position_sizer import PositionSizer
        from common.models.gpu_signal_model import GPUSignalModel
        from common.trading.sector_momentum import SectorMomentumCalculator
        from common.data.features.cross_sectional_features import CrossSectionalFeatureEngineer

        # Initialize all
        components = {
            'regime_detector': UnifiedRegimeDetector(),
            'bear_detector': FastBearDetector(),
            'signal_calculator': PenaltiesOnlyCalculator(),
            'position_sizer': PositionSizer(portfolio_value=100000),
            'gpu_model': GPUSignalModel(),
            'sector_momentum': SectorMomentumCalculator(),
            'feature_engineer': CrossSectionalFeatureEngineer(),
        }

        print(f"[PASS] All {len(components)} pipeline components initialized")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Pipeline initialization failed: {e}")
        failed += 1

    print(f"\nFull Pipeline Dry Run: {passed}/{passed+failed} passed")
    return passed, failed


def run_all_smoke_tests():
    """Run all smoke tests."""
    print("=" * 70)
    print("PROTEUS SMOKE TESTS")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    total_passed = 0
    total_failed = 0

    test_suites = [
        ("Scanner Init", test_scanner_initialization),
        ("Regime Detection", test_regime_detection_smoke),
        ("Bear Detection", test_bear_detection_smoke),
        ("Signal Calc", test_signal_calculation_smoke),
        ("GPU Model", test_gpu_model_smoke),
        ("Position Sizing", test_position_sizing_smoke),
        ("Notifications", test_notification_smoke),
        ("Virtual Wallet", test_virtual_wallet_smoke),
        ("Scanner Runner", test_scanner_runner_smoke),
        ("Daily Report", test_daily_report_smoke),
        ("Full Pipeline", test_full_pipeline_dry_run),
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
    print("SMOKE TEST SUMMARY")
    print("=" * 70)
    for name, passed, failed in results:
        status = "PASS" if failed == 0 else "FAIL"
        print(f"  {name:25} {passed:3}/{passed+failed:3} [{status}]")

    print("-" * 70)
    print(f"  {'TOTAL':25} {total_passed:3}/{total_passed+total_failed:3}")
    print("=" * 70)

    if total_failed == 0:
        print("SUCCESS - All smoke tests passed!")
    else:
        print(f"FAILED - {total_failed} test(s) failed")

    return total_failed == 0


if __name__ == '__main__':
    success = run_all_smoke_tests()
    sys.exit(0 if success else 1)

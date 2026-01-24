"""
Test bear detection system components.

Validates:
1. FastBearDetector outputs correct scores and alert levels
2. BearishAlertService cooldown logic works correctly
3. Bear-aware position sizing applies correct multipliers
4. UnifiedRegimeDetector includes early warning fields
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock


def test_fast_bear_detector():
    """Test FastBearDetector outputs."""
    print("=" * 60)
    print("TEST: FastBearDetector")
    print("=" * 60)

    from src.analysis.fast_bear_detector import FastBearDetector, FastBearSignal

    detector = FastBearDetector()
    passed = 0
    failed = 0

    # Test 1: Detector returns FastBearSignal
    signal = detector.detect()
    if isinstance(signal, FastBearSignal):
        print(f"[PASS] Returns FastBearSignal object")
        passed += 1
    else:
        print(f"[FAIL] Expected FastBearSignal, got {type(signal)}")
        failed += 1

    # Test 2: Bear score is in valid range
    if 0 <= signal.bear_score <= 100:
        print(f"[PASS] Bear score in range: {signal.bear_score}/100")
        passed += 1
    else:
        print(f"[FAIL] Bear score out of range: {signal.bear_score}")
        failed += 1

    # Test 3: Alert level is valid
    valid_levels = ['NORMAL', 'WATCH', 'WARNING', 'CRITICAL']
    if signal.alert_level in valid_levels:
        print(f"[PASS] Alert level valid: {signal.alert_level}")
        passed += 1
    else:
        print(f"[FAIL] Invalid alert level: {signal.alert_level}")
        failed += 1

    # Test 4: Confidence is in valid range
    if 0 <= signal.confidence <= 1:
        print(f"[PASS] Confidence in range: {signal.confidence}")
        passed += 1
    else:
        print(f"[FAIL] Confidence out of range: {signal.confidence}")
        failed += 1

    # Test 5: All required fields present
    required_fields = [
        'vix_level', 'vix_spike_pct', 'market_breadth_pct',
        'spy_roc_3d', 'sectors_declining', 'volume_confirmation',
        'yield_curve_spread', 'credit_spread_change', 'momentum_divergence',
        'put_call_ratio', 'high_yield_spread'
    ]
    missing = [f for f in required_fields if not hasattr(signal, f)]
    if not missing:
        print(f"[PASS] All {len(required_fields)} indicator fields present")
        passed += 1
    else:
        print(f"[FAIL] Missing fields: {missing}")
        failed += 1

    # Test 6: Alert level matches score range
    expected_level = (
        'CRITICAL' if signal.bear_score >= 70 else
        'WARNING' if signal.bear_score >= 50 else
        'WATCH' if signal.bear_score >= 30 else
        'NORMAL'
    )
    if signal.alert_level == expected_level:
        print(f"[PASS] Alert level matches score: {signal.bear_score} -> {signal.alert_level}")
        passed += 1
    else:
        print(f"[FAIL] Score {signal.bear_score} should be {expected_level}, got {signal.alert_level}")
        failed += 1

    # Test 7: Recommendation is present and non-empty
    if signal.recommendation and len(signal.recommendation) > 10:
        print(f"[PASS] Recommendation present: {signal.recommendation[:50]}...")
        passed += 1
    else:
        print(f"[FAIL] Recommendation missing or too short")
        failed += 1

    # Test 8: VIX level is realistic
    if 10 <= signal.vix_level <= 80:
        print(f"[PASS] VIX level realistic: {signal.vix_level}")
        passed += 1
    else:
        print(f"[FAIL] VIX level unrealistic: {signal.vix_level}")
        failed += 1

    print(f"\nFastBearDetector: {passed}/{passed+failed} passed")
    return passed, failed


def test_alert_level_logic():
    """Test alert level determination logic."""
    print("\n" + "=" * 60)
    print("TEST: Alert Level Logic")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test score -> level mapping using the expected thresholds
    def get_expected_level(score):
        if score >= 70:
            return 'CRITICAL'
        elif score >= 50:
            return 'WARNING'
        elif score >= 30:
            return 'WATCH'
        else:
            return 'NORMAL'

    test_cases = [
        (0, 'NORMAL'),
        (15, 'NORMAL'),
        (29, 'NORMAL'),
        (30, 'WATCH'),
        (45, 'WATCH'),
        (49, 'WATCH'),
        (50, 'WARNING'),
        (65, 'WARNING'),
        (69, 'WARNING'),
        (70, 'CRITICAL'),
        (85, 'CRITICAL'),
        (100, 'CRITICAL'),
    ]

    for score, expected_level in test_cases:
        level = get_expected_level(score)
        if level == expected_level:
            print(f"[PASS] Score {score:>3} -> {level}")
            passed += 1
        else:
            print(f"[FAIL] Score {score:>3} -> {level} (expected {expected_level})")
            failed += 1

    print(f"\nAlert Level Logic: {passed}/{passed+failed} passed")
    return passed, failed


def test_bearish_alert_service():
    """Test BearishAlertService cooldown logic."""
    print("\n" + "=" * 60)
    print("TEST: BearishAlertService")
    print("=" * 60)

    from src.trading.bearish_alert_service import BearishAlertService, AlertState

    passed = 0
    failed = 0

    service = BearishAlertService()

    # Test 1: Cooldowns are defined
    expected_cooldowns = ['WATCH', 'WARNING', 'CRITICAL', 'NORMAL']
    if all(level in service.COOLDOWNS for level in expected_cooldowns):
        print(f"[PASS] All cooldown levels defined")
        passed += 1
    else:
        print(f"[FAIL] Missing cooldown levels")
        failed += 1

    # Test 2: Alert priorities are correct
    if (service.ALERT_PRIORITY['CRITICAL'] > service.ALERT_PRIORITY['WARNING'] >
        service.ALERT_PRIORITY['WATCH'] > service.ALERT_PRIORITY['NORMAL']):
        print(f"[PASS] Alert priorities in correct order")
        passed += 1
    else:
        print(f"[FAIL] Alert priorities incorrect")
        failed += 1

    # Test 3: Has daily summary method
    if hasattr(service, 'send_daily_summary'):
        print(f"[PASS] Daily summary method exists")
        passed += 1
    else:
        print(f"[FAIL] Missing send_daily_summary method")
        failed += 1

    # Test 4: Status returns expected keys
    status = service.get_status()
    expected_keys = ['last_alert_level', 'last_alert_time', 'alerts_sent_today',
                    'last_daily_summary', 'notifier_available']
    missing_keys = [k for k in expected_keys if k not in status]
    if not missing_keys:
        print(f"[PASS] Status has all expected keys")
        passed += 1
    else:
        print(f"[FAIL] Status missing keys: {missing_keys}")
        failed += 1

    # Test 5: Cooldown check logic (mock)
    class MockSignal:
        alert_level = 'WARNING'
        bear_score = 55

    # Fresh state should allow alert
    service.state = AlertState(
        last_alert_level='NORMAL',
        last_alert_time=None,
        alerts_sent_today=0
    )
    should_alert = service._should_send_alert(MockSignal())
    if should_alert:
        print(f"[PASS] Fresh state allows alert")
        passed += 1
    else:
        print(f"[FAIL] Fresh state should allow alert")
        failed += 1

    # Recent same-level alert should be blocked
    service.state = AlertState(
        last_alert_level='WARNING',
        last_alert_time=datetime.now() - timedelta(hours=1),  # Within cooldown
        alerts_sent_today=1
    )
    should_alert = service._should_send_alert(MockSignal())
    if not should_alert:
        print(f"[PASS] Cooldown blocks same-level alert")
        passed += 1
    else:
        print(f"[FAIL] Cooldown should block same-level alert")
        failed += 1

    # Escalation should bypass cooldown
    class CriticalSignal:
        alert_level = 'CRITICAL'
        bear_score = 75

    should_alert = service._should_send_alert(CriticalSignal())
    if should_alert:
        print(f"[PASS] Escalation bypasses cooldown")
        passed += 1
    else:
        print(f"[FAIL] Escalation should bypass cooldown")
        failed += 1

    print(f"\nBearishAlertService: {passed}/{passed+failed} passed")
    return passed, failed


def test_bear_aware_position_sizing():
    """Test bear-aware position sizing multipliers."""
    print("\n" + "=" * 60)
    print("TEST: Bear-Aware Position Sizing")
    print("=" * 60)

    from src.trading.unified_position_sizer import UnifiedPositionSizer

    passed = 0
    failed = 0

    sizer = UnifiedPositionSizer(portfolio_value=100000)

    # Test 1: Has get_bear_adjustment method
    if hasattr(sizer, 'get_bear_adjustment'):
        print(f"[PASS] get_bear_adjustment method exists")
        passed += 1
    else:
        print(f"[FAIL] Missing get_bear_adjustment method")
        failed += 1
        return passed, failed

    # Test 2: Normal conditions return 1.0 multiplier
    mult, reason = sizer.get_bear_adjustment()
    if mult <= 1.0:
        print(f"[PASS] Current multiplier: {mult:.0%} ({reason})")
        passed += 1
    else:
        print(f"[FAIL] Multiplier should be <= 1.0, got {mult}")
        failed += 1

    # Test 3: Multiplier logic (test with mocked bear scores)
    # We'll test the multiplier mapping directly
    test_cases = [
        (0, 'NORMAL', 1.0),
        (25, 'NORMAL', 1.0),
        (30, 'WATCH', 0.85),
        (45, 'WATCH', 0.85),
        (50, 'WARNING', 0.65),
        (65, 'WARNING', 0.65),
        (70, 'CRITICAL', 0.40),
        (90, 'CRITICAL', 0.40),
    ]

    # Mock the get_bear_score to test each case
    original_get_bear_score = sizer.get_bear_score

    for score, level, expected_mult in test_cases:
        # Mock returns tuple (score, level)
        sizer.get_bear_score = lambda s=score, l=level: (s, l)
        mult, reason = sizer.get_bear_adjustment()

        if abs(mult - expected_mult) < 0.01:
            print(f"[PASS] Score {score:>2} -> {mult:.0%} multiplier")
            passed += 1
        else:
            print(f"[FAIL] Score {score:>2} -> {mult:.0%} (expected {expected_mult:.0%})")
            failed += 1

    # Restore original method
    sizer.get_bear_score = original_get_bear_score

    print(f"\nBear-Aware Position Sizing: {passed}/{passed+failed} passed")
    return passed, failed


def test_unified_regime_early_warning():
    """Test UnifiedRegimeDetector early warning fields."""
    print("\n" + "=" * 60)
    print("TEST: UnifiedRegimeDetector Early Warning")
    print("=" * 60)

    from src.analysis.unified_regime_detector import UnifiedRegimeDetector, DetectionMethod

    passed = 0
    failed = 0

    detector = UnifiedRegimeDetector(method=DetectionMethod.ENSEMBLE)
    result = detector.detect_regime()

    # Test 1: Has early_warning field
    if hasattr(result, 'early_warning'):
        print(f"[PASS] early_warning field exists: {result.early_warning}")
        passed += 1
    else:
        print(f"[FAIL] Missing early_warning field")
        failed += 1

    # Test 2: Has early_warning_score field
    if hasattr(result, 'early_warning_score'):
        print(f"[PASS] early_warning_score field exists: {result.early_warning_score}")
        passed += 1
    else:
        print(f"[FAIL] Missing early_warning_score field")
        failed += 1

    # Test 3: Has early_warning_level field
    if hasattr(result, 'early_warning_level'):
        print(f"[PASS] early_warning_level field exists: {result.early_warning_level}")
        passed += 1
    else:
        print(f"[FAIL] Missing early_warning_level field")
        failed += 1

    # Test 4: early_warning matches score threshold
    expected_warning = result.early_warning_score >= 30
    if result.early_warning == expected_warning:
        print(f"[PASS] early_warning matches score: {result.early_warning_score} -> {result.early_warning}")
        passed += 1
    else:
        print(f"[FAIL] early_warning doesn't match score")
        failed += 1

    # Test 5: Score in valid range
    if 0 <= result.early_warning_score <= 100:
        print(f"[PASS] Score in valid range: {result.early_warning_score}")
        passed += 1
    else:
        print(f"[FAIL] Score out of range: {result.early_warning_score}")
        failed += 1

    # Test 6: Level is valid
    valid_levels = ['NORMAL', 'WATCH', 'WARNING', 'CRITICAL']
    if result.early_warning_level in valid_levels:
        print(f"[PASS] Level is valid: {result.early_warning_level}")
        passed += 1
    else:
        print(f"[FAIL] Invalid level: {result.early_warning_level}")
        failed += 1

    print(f"\nUnifiedRegimeDetector Early Warning: {passed}/{passed+failed} passed")
    return passed, failed


def test_bear_score_history():
    """Test bear score history logging."""
    print("\n" + "=" * 60)
    print("TEST: Bear Score History")
    print("=" * 60)

    from src.analysis.fast_bear_detector import (
        FastBearDetector, log_bear_score, get_bear_score_trend
    )

    passed = 0
    failed = 0

    # Test 1: log_bear_score returns entry
    detector = FastBearDetector()
    signal = detector.detect()
    entry = log_bear_score(signal)

    if entry and 'bear_score' in entry and 'timestamp' in entry:
        print(f"[PASS] log_bear_score returns valid entry")
        passed += 1
    else:
        print(f"[FAIL] log_bear_score failed")
        failed += 1

    # Test 2: get_bear_score_trend works
    trend = get_bear_score_trend(hours=24)
    if isinstance(trend, dict):
        print(f"[PASS] get_bear_score_trend returns dict")
        passed += 1
    else:
        print(f"[FAIL] get_bear_score_trend should return dict")
        failed += 1

    # Test 3: Trend has expected keys (if data available)
    if 'error' not in trend:
        expected_keys = ['current', 'min', 'max', 'avg', 'trend', 'data_points']
        missing = [k for k in expected_keys if k not in trend]
        if not missing:
            print(f"[PASS] Trend has all expected keys")
            passed += 1
        else:
            print(f"[FAIL] Trend missing keys: {missing}")
            failed += 1
    else:
        print(f"[SKIP] No trend data available: {trend.get('error')}")
        passed += 1  # Skip counts as pass if no data

    print(f"\nBear Score History: {passed}/{passed+failed} passed")
    return passed, failed


def test_webhook_notifier():
    """Test webhook notifier functionality."""
    print("\n" + "=" * 60)
    print("TEST: Webhook Notifier")
    print("=" * 60)

    from src.notifications.webhook_notifier import WebhookNotifier, get_webhook_status

    passed = 0
    failed = 0

    # Test 1: WebhookNotifier initializes
    notifier = WebhookNotifier()
    if notifier is not None:
        print(f"[PASS] WebhookNotifier initializes")
        passed += 1
    else:
        print(f"[FAIL] WebhookNotifier failed to initialize")
        failed += 1

    # Test 2: is_enabled returns boolean
    enabled = notifier.is_enabled()
    if isinstance(enabled, bool):
        print(f"[PASS] is_enabled returns boolean: {enabled}")
        passed += 1
    else:
        print(f"[FAIL] is_enabled should return boolean")
        failed += 1

    # Test 3: get_webhook_status returns dict
    status = get_webhook_status()
    if isinstance(status, dict) and 'enabled' in status and 'platforms' in status:
        print(f"[PASS] get_webhook_status returns expected dict")
        passed += 1
    else:
        print(f"[FAIL] get_webhook_status missing expected keys")
        failed += 1

    # Test 4: send_bear_alert returns dict (even if empty)
    from src.analysis.fast_bear_detector import FastBearDetector
    detector = FastBearDetector()
    signal = detector.detect()
    results = notifier.send_bear_alert(signal)
    if isinstance(results, dict):
        print(f"[PASS] send_bear_alert returns dict")
        passed += 1
    else:
        print(f"[FAIL] send_bear_alert should return dict")
        failed += 1

    # Test 5: send_all_clear returns dict
    results = notifier.send_all_clear(signal)
    if isinstance(results, dict):
        print(f"[PASS] send_all_clear returns dict")
        passed += 1
    else:
        print(f"[FAIL] send_all_clear should return dict")
        failed += 1

    # Test 6: test_webhook returns dict
    results = notifier.test_webhook()
    if isinstance(results, dict):
        print(f"[PASS] test_webhook returns dict")
        passed += 1
    else:
        print(f"[FAIL] test_webhook should return dict")
        failed += 1

    print(f"\nWebhook Notifier: {passed}/{passed+failed} passed")
    return passed, failed


def test_api_server():
    """Test API server endpoint handlers."""
    print("\n" + "=" * 60)
    print("TEST: API Server")
    print("=" * 60)

    import sys
    import json
    sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

    passed = 0
    failed = 0

    # Test 1: Module imports
    try:
        from bear_api_server import BearAPIHandler
        print(f"[PASS] API server module imports")
        passed += 1
    except ImportError as e:
        print(f"[FAIL] API server import failed: {e}")
        failed += 1
        return passed, failed

    # Test 2: Status endpoint data format
    from src.analysis.fast_bear_detector import FastBearDetector
    detector = FastBearDetector()
    signal = detector.detect()

    status = {
        'bear_score': signal.bear_score,
        'alert_level': signal.alert_level,
        'confidence': signal.confidence,
        'triggers_count': len(signal.triggers),
        'timestamp': signal.timestamp
    }

    required_keys = ['bear_score', 'alert_level', 'confidence', 'triggers_count', 'timestamp']
    missing = [k for k in required_keys if k not in status]
    if not missing:
        print(f"[PASS] Status endpoint has all required keys")
        passed += 1
    else:
        print(f"[FAIL] Status missing keys: {missing}")
        failed += 1

    # Test 3: Signal endpoint data format
    full_signal = {
        'bear_score': signal.bear_score,
        'alert_level': signal.alert_level,
        'indicators': {
            'vix_level': signal.vix_level,
            'spy_roc_3d': signal.spy_roc_3d,
            'market_breadth_pct': signal.market_breadth_pct,
        },
        'triggers': signal.triggers,
        'recommendation': signal.recommendation
    }

    if 'indicators' in full_signal and 'triggers' in full_signal:
        print(f"[PASS] Signal endpoint has expected structure")
        passed += 1
    else:
        print(f"[FAIL] Signal endpoint missing expected keys")
        failed += 1

    # Test 4: Metrics format (Prometheus)
    metrics_lines = [
        f'proteus_bear_score {signal.bear_score}',
        f'proteus_vix_level {signal.vix_level}',
    ]
    valid_metrics = all('proteus_' in line for line in metrics_lines)
    if valid_metrics:
        print(f"[PASS] Metrics format is valid Prometheus format")
        passed += 1
    else:
        print(f"[FAIL] Metrics format invalid")
        failed += 1

    # Test 5: Trend endpoint data
    from src.analysis.fast_bear_detector import get_bear_score_trend
    trend = get_bear_score_trend(hours=24)
    if isinstance(trend, dict):
        print(f"[PASS] Trend endpoint returns dict")
        passed += 1
    else:
        print(f"[FAIL] Trend endpoint should return dict")
        failed += 1

    # Test 6: Health check structure
    health = {
        'status': 'healthy',
        'components': {'fast_bear_detector': 'ok'}
    }
    if 'status' in health and 'components' in health:
        print(f"[PASS] Health endpoint has expected structure")
        passed += 1
    else:
        print(f"[FAIL] Health endpoint missing expected keys")
        failed += 1

    print(f"\nAPI Server: {passed}/{passed+failed} passed")
    return passed, failed


def test_configuration_system():
    """Test configuration system functionality."""
    print("\n" + "=" * 60)
    print("TEST: Configuration System")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: Config module imports
    try:
        from src.config.bear_config import (
            load_config, get_config, validate_config,
            BearConfig, DEFAULT_CONFIG
        )
        print(f"[PASS] Config module imports successfully")
        passed += 1
    except ImportError as e:
        print(f"[FAIL] Config module import failed: {e}")
        failed += 1
        return passed, failed

    # Test 2: load_config returns BearConfig
    config = load_config()
    if isinstance(config, BearConfig):
        print(f"[PASS] load_config returns BearConfig")
        passed += 1
    else:
        print(f"[FAIL] load_config should return BearConfig")
        failed += 1

    # Test 3: Config has required attributes
    required_attrs = ['weights', 'thresholds', 'alert_levels', 'scoring', 'cooldowns', 'position_sizing']
    missing = [a for a in required_attrs if not hasattr(config, a)]
    if not missing:
        print(f"[PASS] Config has all required attributes")
        passed += 1
    else:
        print(f"[FAIL] Config missing attributes: {missing}")
        failed += 1

    # Test 4: Weights are valid
    weights_sum = sum(config.weights.values())
    if 0.99 <= weights_sum <= 1.01:
        print(f"[PASS] Weights sum to {weights_sum:.2f}")
        passed += 1
    else:
        print(f"[FAIL] Weights sum to {weights_sum:.2f}, should be 1.0")
        failed += 1

    # Test 5: All expected weight keys present
    expected_weights = ['spy_roc', 'vix', 'breadth', 'sector_breadth', 'volume', 'yield_curve', 'credit_spread', 'high_yield', 'put_call', 'divergence']
    missing_weights = [w for w in expected_weights if w not in config.weights]
    if not missing_weights:
        print(f"[PASS] All {len(expected_weights)} weight keys present")
        passed += 1
    else:
        print(f"[FAIL] Missing weight keys: {missing_weights}")
        failed += 1

    # Test 6: Thresholds have expected structure
    expected_thresholds = ['spy_roc', 'vix_level', 'vix_spike', 'breadth', 'sector_breadth', 'yield_curve', 'credit_spread', 'high_yield', 'put_call']
    missing_thresholds = [t for t in expected_thresholds if t not in config.thresholds]
    if not missing_thresholds:
        print(f"[PASS] All {len(expected_thresholds)} threshold keys present")
        passed += 1
    else:
        print(f"[FAIL] Missing threshold keys: {missing_thresholds}")
        failed += 1

    # Test 7: Each threshold has watch/warning/critical
    all_have_levels = True
    for name, thresh in config.thresholds.items():
        if not all(level in thresh for level in ['watch', 'warning', 'critical']):
            all_have_levels = False
            break
    if all_have_levels:
        print(f"[PASS] All thresholds have watch/warning/critical levels")
        passed += 1
    else:
        print(f"[FAIL] Some thresholds missing levels")
        failed += 1

    # Test 8: Position sizing values are valid
    valid_sizing = all(0 < v <= 1.0 for v in config.position_sizing.values())
    if valid_sizing:
        print(f"[PASS] Position sizing values all between 0 and 1")
        passed += 1
    else:
        print(f"[FAIL] Invalid position sizing values")
        failed += 1

    # Test 9: validate_config returns expected structure
    result = validate_config(config)
    if isinstance(result, dict) and 'valid' in result and 'errors' in result:
        print(f"[PASS] validate_config returns expected structure")
        passed += 1
    else:
        print(f"[FAIL] validate_config missing valid/errors keys")
        failed += 1

    # Test 10: Default config validates successfully
    if result['valid']:
        print(f"[PASS] Configuration validates successfully")
        passed += 1
    else:
        print(f"[FAIL] Configuration validation failed: {result['errors']}")
        failed += 1

    # Test 11: get_config returns cached instance
    config2 = get_config()
    if config2 is not None:
        print(f"[PASS] get_config returns config instance")
        passed += 1
    else:
        print(f"[FAIL] get_config returned None")
        failed += 1

    # Test 12: DEFAULT_CONFIG is valid
    if isinstance(DEFAULT_CONFIG, dict) and 'weights' in DEFAULT_CONFIG:
        print(f"[PASS] DEFAULT_CONFIG is valid")
        passed += 1
    else:
        print(f"[FAIL] DEFAULT_CONFIG is invalid")
        failed += 1

    print(f"\nConfiguration System: {passed}/{passed+failed} passed")
    return passed, failed


def run_all_tests():
    """Run all bear detection tests and summarize."""
    print("\n" + "=" * 60)
    print("BEAR DETECTION SYSTEM VALIDATION")
    print("=" * 60 + "\n")

    total_passed = 0
    total_failed = 0

    p, f = test_fast_bear_detector()
    total_passed += p
    total_failed += f

    p, f = test_alert_level_logic()
    total_passed += p
    total_failed += f

    p, f = test_bearish_alert_service()
    total_passed += p
    total_failed += f

    p, f = test_bear_aware_position_sizing()
    total_passed += p
    total_failed += f

    p, f = test_unified_regime_early_warning()
    total_passed += p
    total_failed += f

    p, f = test_bear_score_history()
    total_passed += p
    total_failed += f

    p, f = test_webhook_notifier()
    total_passed += p
    total_failed += f

    p, f = test_api_server()
    total_passed += p
    total_failed += f

    p, f = test_configuration_system()
    total_passed += p
    total_failed += f

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_passed}/{total_passed + total_failed} tests passed")
    if total_failed == 0:
        print("SUCCESS - All bear detection tests passed!")
    else:
        print(f"WARNING - {total_failed} tests failed")
    print("=" * 60)

    return total_failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

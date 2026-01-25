"""
Test unified system components match expected behavior.

Validates:
1. UnifiedPositionSizer outputs reasonable sizes
2. UnifiedSignalCalculator applies boosts correctly
3. PositionRebalancer triggers exits appropriately
"""

import sys
from pathlib import Path
# Go up 3 levels: tests → daily_picks → features → project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from datetime import datetime
from common.trading.unified_position_sizer import UnifiedPositionSizer, SignalQuality
from common.trading.unified_signal_calculator import UnifiedSignalCalculator
from common.trading.position_rebalancer import PositionRebalancer, ExitReason


def test_position_sizer():
    """Test position sizer outputs."""
    print("=" * 60)
    print("TEST: UnifiedPositionSizer")
    print("=" * 60)

    sizer = UnifiedPositionSizer(portfolio_value=100000)
    passed = 0
    failed = 0

    # Test 1: Elite stock gets larger position
    rec = sizer.calculate_size('COP', 105.0, signal_strength=80, regime='volatile')
    if rec.size_pct >= 10:
        print(f"[PASS] Elite + volatile = {rec.size_pct}% (expected >=10%)")
        passed += 1
    else:
        print(f"[FAIL] Elite + volatile = {rec.size_pct}% (expected >=10%)")
        failed += 1

    # Test 2: Avoid stock gets skipped
    rec = sizer.calculate_size('NOW', 900.0, signal_strength=70, regime='bull')
    if rec.skip_trade:
        print(f"[PASS] AVOID tier skipped as expected")
        passed += 1
    else:
        print(f"[FAIL] AVOID tier should be skipped, got {rec.size_pct}%")
        failed += 1

    # Test 3: Bull regime reduces position
    rec_volatile = sizer.calculate_size('MSFT', 420.0, signal_strength=70, regime='volatile')
    rec_bull = sizer.calculate_size('MSFT', 420.0, signal_strength=70, regime='bull')
    if rec_volatile.size_pct > rec_bull.size_pct:
        print(f"[PASS] Volatile ({rec_volatile.size_pct}%) > Bull ({rec_bull.size_pct}%)")
        passed += 1
    else:
        print(f"[FAIL] Volatile should get larger position than Bull")
        failed += 1

    # Test 4: Quality assessment
    quality = sizer.assess_quality(85, 'elite')
    if quality == SignalQuality.EXCELLENT:
        print(f"[PASS] Elite + 85 signal = EXCELLENT")
        passed += 1
    else:
        print(f"[FAIL] Expected EXCELLENT, got {quality}")
        failed += 1

    print(f"\nPosition Sizer: {passed}/{passed+failed} passed")
    return passed, failed


def test_signal_calculator():
    """Test signal calculator outputs."""
    print("\n" + "=" * 60)
    print("TEST: UnifiedSignalCalculator")
    print("=" * 60)

    calc = UnifiedSignalCalculator()
    passed = 0
    failed = 0

    # Test 1: Elite tier gets boost
    result = calc.calculate('COP', 65.0, regime='choppy')
    if result.final_signal > 65.0:
        print(f"[PASS] Elite tier boosts signal: {65.0} -> {result.final_signal}")
        passed += 1
    else:
        print(f"[FAIL] Elite should boost signal")
        failed += 1

    # Test 2: Avoid tier reduces signal
    result = calc.calculate('NOW', 65.0, regime='choppy')
    if result.final_signal < 65.0:
        print(f"[PASS] Avoid tier reduces signal: {65.0} -> {result.final_signal}")
        passed += 1
    else:
        print(f"[FAIL] Avoid should reduce signal")
        failed += 1

    # Test 3: Volatile regime boosts
    result_vol = calc.calculate('MSFT', 70.0, regime='volatile')
    result_bull = calc.calculate('MSFT', 70.0, regime='bull')
    if result_vol.final_signal > result_bull.final_signal:
        print(f"[PASS] Volatile ({result_vol.final_signal}) > Bull ({result_bull.final_signal})")
        passed += 1
    else:
        print(f"[FAIL] Volatile should boost more than Bull")
        failed += 1

    # Test 4: Boosts applied correctly
    result = calc.calculate(
        'COP', 65.0, regime='volatile',
        is_monday=True,
        consecutive_down_days=6,
        has_rsi_divergence=True,
        volume_ratio=4.0,
        is_down_day=True,
        sector_momentum=-5.0
    )
    expected_boosts = 5  # monday, rsi, down_streak, volume_exhaustion, weak_sector
    if len(result.boosts_applied) == expected_boosts:
        print(f"[PASS] All 5 boosts applied: {list(result.boosts_applied.keys())}")
        passed += 1
    else:
        print(f"[FAIL] Expected 5 boosts, got {len(result.boosts_applied)}")
        failed += 1

    # Test 5: Signal caps at 100
    if result.final_signal <= 100.0:
        print(f"[PASS] Signal capped at 100: {result.final_signal}")
        passed += 1
    else:
        print(f"[FAIL] Signal should cap at 100")
        failed += 1

    print(f"\nSignal Calculator: {passed}/{passed+failed} passed")
    return passed, failed


def test_position_rebalancer():
    """Test position rebalancer outputs."""
    print("\n" + "=" * 60)
    print("TEST: PositionRebalancer")
    print("=" * 60)

    rebalancer = PositionRebalancer()
    passed = 0
    failed = 0

    # Test 1: Stop loss triggers exit
    status = rebalancer.check_position(
        'NVDA', entry_price=140.0, current_price=137.2,  # -2%
        entry_date=datetime(2026, 1, 3)
    )
    if status.exit_reason == ExitReason.STOP_LOSS:
        print(f"[PASS] Stop loss triggered at -2%: {status.recommendation}")
        passed += 1
    else:
        print(f"[FAIL] Expected stop loss, got {status.exit_reason}")
        failed += 1

    # Test 2: Profit target partial exit
    status = rebalancer.check_position(
        'COP', entry_price=100.0, current_price=102.5,  # +2.5%
        entry_date=datetime(2026, 1, 3)
    )
    if status.exit_reason == ExitReason.PROFIT_TARGET_1 and status.exit_pct == 50:
        print(f"[PASS] Partial exit at +2.5%: {status.recommendation}")
        passed += 1
    else:
        print(f"[FAIL] Expected partial profit take, got {status.exit_reason}")
        failed += 1

    # Test 3: Full profit exit
    status = rebalancer.check_position(
        'COP', entry_price=100.0, current_price=103.6,  # +3.6% > 3.5% target_2
        entry_date=datetime(2026, 1, 3)
    )
    if status.exit_reason == ExitReason.PROFIT_TARGET_2 and status.exit_pct == 100:
        print(f"[PASS] Full exit at +3.6%: {status.recommendation}")
        passed += 1
    else:
        print(f"[FAIL] Expected full profit exit, got {status.exit_reason}")
        failed += 1

    # Test 4: Max hold days triggers exit
    status = rebalancer.check_position(
        'MSFT', entry_price=420.0, current_price=422.0,  # +0.5%, held 4 days
        entry_date=datetime(2025, 12, 31)  # 4 days ago
    )
    if status.exit_reason == ExitReason.MAX_HOLD_DAYS:
        print(f"[PASS] Max hold days exit: {status.recommendation}")
        passed += 1
    else:
        print(f"[FAIL] Expected max hold days, got {status.exit_reason} (days={status.days_held})")
        failed += 1

    # Test 5: Hold when no exit condition
    status = rebalancer.check_position(
        'COP', entry_price=100.0, current_price=101.0,  # +1%, day 1
        entry_date=datetime(2026, 1, 3)
    )
    if status.exit_reason == ExitReason.NONE:
        print(f"[PASS] Hold when no exit: {status.recommendation}")
        passed += 1
    else:
        print(f"[FAIL] Should hold, got {status.exit_reason}")
        failed += 1

    print(f"\nPosition Rebalancer: {passed}/{passed+failed} passed")
    return passed, failed


def run_all_tests():
    """Run all tests and summarize."""
    print("\n" + "=" * 60)
    print("UNIFIED SYSTEM VALIDATION")
    print("=" * 60 + "\n")

    total_passed = 0
    total_failed = 0

    p, f = test_position_sizer()
    total_passed += p
    total_failed += f

    p, f = test_signal_calculator()
    total_passed += p
    total_failed += f

    p, f = test_position_rebalancer()
    total_passed += p
    total_failed += f

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_passed}/{total_passed + total_failed} tests passed")
    if total_failed == 0:
        print("SUCCESS - All tests passed!")
    else:
        print(f"WARNING - {total_failed} tests failed")
    print("=" * 60)

    return total_failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

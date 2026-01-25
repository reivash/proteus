#!/usr/bin/env python
"""
Quick system validation - checks all components work.

Usage:
    python check_system.py
"""

import sys
from pathlib import Path


def check_imports():
    """Verify all modules import correctly."""
    print("Checking imports...")

    modules = [
        ('unified_signal_calculator', 'src.trading.unified_signal_calculator'),
        ('unified_position_sizer', 'src.trading.unified_position_sizer'),
        ('position_rebalancer', 'src.trading.position_rebalancer'),
        ('smart_scanner_v2', 'src.trading.smart_scanner_v2'),
    ]

    all_ok = True
    for name, module_path in modules:
        try:
            __import__(module_path.replace('/', '.'))
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            all_ok = False

    return all_ok


def check_config():
    """Verify config loads correctly."""
    print("\nChecking unified_config.json...")

    config_path = Path('features/daily_picks/config.json')
    if not config_path.exists():
        print("  [FAIL] Config file not found")
        return False

    import json
    with open(config_path) as f:
        config = json.load(f)

    required_sections = ['stock_tiers', 'regime_adjustments', 'signal_boosts',
                         'exit_strategy', 'portfolio_constraints']

    all_ok = True
    for section in required_sections:
        if section in config:
            print(f"  [OK] {section}")
        else:
            print(f"  [FAIL] Missing: {section}")
            all_ok = False

    # Count tickers
    total_tickers = 0
    for tier, data in config.get('stock_tiers', {}).items():
        if not tier.startswith('_'):
            total_tickers += len(data.get('tickers', []))
    print(f"  Total tickers configured: {total_tickers}")

    return all_ok


def check_unified_modules():
    """Test unified modules work correctly."""
    print("\nTesting unified modules...")

    # Signal calculator
    from common.trading.unified_signal_calculator import UnifiedSignalCalculator
    calc = UnifiedSignalCalculator()
    result = calc.calculate('COP', 65.0, 'volatile', is_monday=True)
    print(f"  [OK] Signal calculator: COP 65->  {result.final_signal:.1f}")

    # Position sizer
    from common.trading.unified_position_sizer import UnifiedPositionSizer
    sizer = UnifiedPositionSizer(100000)
    rec = sizer.calculate_size('COP', 105.0, 80, 'volatile')
    print(f"  [OK] Position sizer: COP -> {rec.size_pct:.1f}% ({rec.quality.value})")

    # Rebalancer
    from common.trading.position_rebalancer import PositionRebalancer
    from datetime import datetime
    reb = PositionRebalancer()
    status = reb.check_position('COP', 100.0, 103.5, datetime(2026, 1, 3))
    print(f"  [OK] Rebalancer: COP +3.5% -> {status.exit_reason.value}")

    return True


def check_gpu():
    """Test GPU model loads."""
    print("\nChecking GPU model...")

    try:
        from common.models.gpu_signal_model import GPUSignalModel
        model = GPUSignalModel()
        print(f"  [OK] GPU model loaded on {model.device}")
        return True
    except Exception as e:
        print(f"  [WARN] GPU model: {e}")
        return False


def main():
    print("=" * 60)
    print("PROTEUS SYSTEM CHECK")
    print("=" * 60)
    print()

    results = []

    results.append(('Imports', check_imports()))
    results.append(('Config', check_config()))
    results.append(('Unified Modules', check_unified_modules()))
    results.append(('GPU Model', check_gpu()))

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_ok = True
    for name, ok in results:
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {name}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All checks passed! System ready.")
    else:
        print("Some checks failed. Review above.")

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())

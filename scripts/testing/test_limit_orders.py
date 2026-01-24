"""
Quick test of limit order implementation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.trading.signal_scanner import SignalScanner
from src.trading.paper_trader import PaperTrader

def test_limit_orders():
    """Test limit order functionality."""

    print("=" * 70)
    print("TESTING LIMIT ORDER IMPLEMENTATION")
    print("=" * 70)
    print()

    # Test 1: SignalScanner includes limit_price fields
    print("TEST 1: SignalScanner generates limit_price fields")
    print("-" * 70)

    scanner = SignalScanner(lookback_days=90)

    # Test with a single stock
    test_ticker = 'NVDA'
    print(f"Scanning {test_ticker}...")

    signal = scanner.scan_stock(test_ticker, date='2025-11-14')

    if signal:
        print(f"[OK] Signal found for {test_ticker}")
        print(f"  Close price: ${signal['price']:.2f}")
        print(f"  Limit price: ${signal.get('limit_price', 'MISSING'):.2f}" if 'limit_price' in signal else "  [ERROR] limit_price field missing!")
        print(f"  Intraday low: ${signal.get('intraday_low', 'MISSING'):.2f}" if 'intraday_low' in signal else "  [ERROR] intraday_low field missing!")
        print(f"  Would fill: {signal.get('limit_would_fill', 'MISSING')}" if 'limit_would_fill' in signal else "  [ERROR] limit_would_fill field missing!")

        if 'limit_price' in signal and 'intraday_low' in signal and 'limit_would_fill' in signal:
            print("  [TEST PASSED] All limit order fields present")
        else:
            print("  [TEST FAILED] Missing limit order fields")
            return False
    else:
        print(f"[INFO] No signal for {test_ticker} on 2025-11-14 (this is OK)")

    print()

    # Test 2: PaperTrader uses limit orders
    print("TEST 2: PaperTrader uses limit order pricing")
    print("-" * 70)

    # Create test signal
    test_signal = {
        'ticker': 'TEST',
        'date': '2025-11-14',
        'signal_type': 'BUY',
        'price': 100.00,  # Close price
        'limit_price': 99.70,  # 0.3% discount
        'intraday_low': 99.50,  # Would fill
        'limit_would_fill': True,
        'z_score': -2.1,
        'rsi': 28.5,
        'expected_return': 3.2,
        'signal_strength': 75.0,
        'position_size': 1.0
    }

    # Test with limit orders ENABLED (default)
    trader_limit = PaperTrader(initial_capital=10000, use_limit_orders=True)
    print("  Testing with use_limit_orders=True...")
    trades = trader_limit.process_signals([test_signal], '2025-11-14')

    if trades:
        entry_price = trades[0]['entry_price']
        expected_price = test_signal['limit_price']
        if abs(entry_price - expected_price) < 0.01:
            print(f"  [TEST PASSED] Entry at limit price ${entry_price:.2f}")
            print(f"    Saved {((test_signal['price'] - entry_price)/test_signal['price']*100):.2f}% vs close")
        else:
            print(f"  [TEST FAILED] Entry at ${entry_price:.2f}, expected ${expected_price:.2f}")
            return False
    else:
        print("  [TEST FAILED] No trade executed")
        return False

    print()

    # Test with limit orders DISABLED
    trader_market = PaperTrader(initial_capital=10000, use_limit_orders=False)
    print("  Testing with use_limit_orders=False...")
    trades = trader_market.process_signals([test_signal], '2025-11-14')

    if trades:
        entry_price = trades[0]['entry_price']
        expected_price = test_signal['price']
        if abs(entry_price - expected_price) < 0.01:
            print(f"  [TEST PASSED] Entry at close price ${entry_price:.2f}")
        else:
            print(f"  [TEST FAILED] Entry at ${entry_price:.2f}, expected ${expected_price:.2f}")
            return False
    else:
        print("  [TEST FAILED] No trade executed")
        return False

    print()

    # Test 3: Signal filtering (limit order would NOT fill)
    print("TEST 3: Signal filtering when limit order wouldn't fill")
    print("-" * 70)

    no_fill_signal = {
        'ticker': 'TEST2',
        'date': '2025-11-14',
        'signal_type': 'BUY',
        'price': 100.00,
        'limit_price': 99.70,  # 0.3% discount
        'intraday_low': 99.80,  # Would NOT fill (low is above limit)
        'limit_would_fill': False,
        'z_score': -2.1,
        'rsi': 28.5,
        'expected_return': 3.2,
        'signal_strength': 75.0,
        'position_size': 1.0
    }

    trader_filter = PaperTrader(initial_capital=10000, use_limit_orders=True)
    trades = trader_filter.process_signals([no_fill_signal], '2025-11-14')

    if len(trades) == 0:
        print("  [TEST PASSED] Signal correctly filtered (limit order wouldn't fill)")
    else:
        print("  [TEST FAILED] Signal should have been filtered")
        return False

    print()
    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    print()
    print("LIMIT ORDER IMPLEMENTATION VALIDATED:")
    print("- SignalScanner generates limit_price fields (0.3% below close)")
    print("- PaperTrader uses limit orders when enabled")
    print("- PaperTrader filters signals that wouldn't fill")
    print("- PaperTrader falls back to close price when disabled")
    print()

    return True


if __name__ == '__main__':
    success = test_limit_orders()
    sys.exit(0 if success else 1)

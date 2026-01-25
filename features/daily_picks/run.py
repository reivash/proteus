#!/usr/bin/env python
"""
Run daily scan with SmartScannerV2.

Usage:
    python run_scan.py                              # Default: hybrid model + aggressive mode
    python run_scan.py --model hybrid               # LSTM V2 primary + MLP fallback
    python run_scan.py --model lstm                 # Pure LSTM V2 (most selective)
    python run_scan.py --model mlp                  # Original GPU MLP
    python run_scan.py --mode aggressive            # Skip BULL regime (+82% Sharpe) [DEFAULT]
    python run_scan.py --mode conservative          # Skip BULL, reduce CHOPPY
    python run_scan.py --mode balanced              # Trade all regimes
    python run_scan.py --legacy                     # Run old scanner
    python run_scan.py --compare                    # Compare V2 vs legacy

Model Performance (Jan 4, 2026 A/B Test):
    Hybrid: 79.2% win rate, 4.96 Sharpe (LSTM primary, MLP fallback)
    MLP:    62.2% win rate, 1.74 Sharpe (original)

Regime-Adaptive Mode Performance (Jan 4, 2026):
    AGGRESSIVE:   66.1% win rate, 2.53 Sharpe (+82% vs baseline) [RECOMMENDED]
    CONSERVATIVE: 64.9% win rate, 2.31 Sharpe (+66% vs baseline)
    BALANCED:     62.2% win rate, 1.73 Sharpe (+24% vs baseline)

    Key insight: BULL regime has only 1.0 Sharpe, VOLATILE has 4.86 Sharpe.
    By skipping BULL, we nearly DOUBLE risk-adjusted returns.
"""

import sys
import argparse
from datetime import datetime


def show_bear_warning(pre_scan: bool = False):
    """
    Check and display fast bear warning if elevated.

    Args:
        pre_scan: If True, this is a pre-scan check (shows brief summary).
                  If False, this is post-scan (shows detailed warning if needed).

    Returns:
        tuple: (bear_score, alert_level) or (0, 'UNKNOWN') if detection fails
    """
    try:
        from common.analysis.fast_bear_detector import FastBearDetector
        detector = FastBearDetector()
        signal = detector.detect()

        if pre_scan:
            # Brief pre-scan summary
            level_colors = {
                'NORMAL': '',
                'WATCH': '[!]',
                'WARNING': '[!!]',
                'CRITICAL': '[!!!]'
            }
            marker = level_colors.get(signal.alert_level, '')
            print(f"\n{marker} Bear Check: {signal.bear_score:.0f}/100 ({signal.alert_level}) | "
                  f"VIX: {signal.vix_level:.1f} | Breadth: {signal.market_breadth_pct:.0f}%")

            if signal.alert_level == 'CRITICAL':
                print("=" * 70)
                print("*** CRITICAL BEAR WARNING - Consider defensive positioning ***")
                print("=" * 70)
            elif signal.alert_level == 'WARNING':
                print("[Bear Warning] Elevated risk detected - reduce new positions")

            return (signal.bear_score, signal.alert_level)

        # Post-scan detailed warning
        if signal.bear_score >= 30 or signal.alert_level != 'NORMAL':
            print()
            print("=" * 70)
            print(f"BEAR WARNING: {signal.alert_level} (Score: {signal.bear_score}/100)")
            print("=" * 70)

            if signal.triggers:
                print("Active Triggers:")
                for trigger in signal.triggers:
                    print(f"  - {trigger}")

            print()
            print("Key Metrics:")
            print(f"  SPY 3d ROC: {signal.spy_roc_3d:+.2f}%")
            print(f"  VIX: {signal.vix_level:.1f} ({signal.vix_spike_pct:+.1f}% 2d spike)")
            print(f"  Breadth: {signal.market_breadth_pct:.1f}% above MA")
            print(f"  Sectors Down: {signal.sectors_declining}/{signal.sectors_total}")
            curve_status = "INVERTED" if signal.yield_curve_spread <= 0 else ("FLAT" if signal.yield_curve_spread < 0.25 else "OK")
            print(f"  Yield Curve: {signal.yield_curve_spread:+.2f}% ({curve_status})")
            credit_status = "STRESSED" if signal.credit_spread_change >= 10 else ("WIDENING" if signal.credit_spread_change >= 5 else "OK")
            print(f"  Credit Spread: {signal.credit_spread_change:+.2f}% ({credit_status})")
            hy_status = "STRESSED" if signal.high_yield_spread >= 5 else ("WIDENING" if signal.high_yield_spread >= 3 else "OK")
            print(f"  High-Yield Spread: {signal.high_yield_spread:+.2f}% ({hy_status})")
            pc_status = "COMPLACENT" if signal.put_call_ratio < 0.65 else ("LOW" if signal.put_call_ratio < 0.75 else "OK")
            print(f"  Put/Call Ratio: {signal.put_call_ratio:.2f} ({pc_status})")
            if signal.momentum_divergence:
                print(f"  DIVERGENCE: SPY near highs but breadth weak!")
            print()
            print(f"Recommendation: {signal.recommendation}")
            print("=" * 70)
        else:
            # Show brief status even when normal if yield curve is warning
            if signal.yield_curve_spread < 0.25:
                print()
                print(f"[Bear Watch] Score: {signal.bear_score}/100 | Yield curve: {signal.yield_curve_spread:+.2f}% (flattening)")

        return (signal.bear_score, signal.alert_level)

    except Exception as e:
        # Log error instead of silent fail - user should know if bear detection failed
        print(f"\n[Bear Detection Error] {type(e).__name__}: {e}")
        print("[Bear Detection] Unable to check market conditions - proceeding with caution")
        return (0, 'UNKNOWN')


def run_v2(model: str = 'hybrid', trading_mode: str = 'aggressive', use_sentiment: bool = False):
    """Run SmartScannerV2 (unified modules) with model and mode selection."""
    from common.trading.smart_scanner_v2 import SmartScannerV2

    # Pre-scan bear check - get early warning before running full scan
    bear_score, alert_level = show_bear_warning(pre_scan=True)

    # If CRITICAL, warn user but still allow scan (they may want to check exit positions)
    if alert_level == 'CRITICAL':
        print("\n*** Running scan in defensive mode due to CRITICAL bear warning ***")
        print("*** Consider reducing position sizes or avoiding new entries ***\n")

    scanner = SmartScannerV2(model=model, trading_mode=trading_mode, use_sentiment=use_sentiment)
    result = scanner.scan()

    # Post-scan detailed bear warning if conditions are elevated
    if bear_score >= 30 or alert_level not in ['NORMAL', 'UNKNOWN']:
        show_bear_warning(pre_scan=False)

    return result


def run_legacy():
    """Run original SmartScanner."""
    from common.trading.smart_scanner import run_smart_scan
    return run_smart_scan()


def compare():
    """Run both scanners and compare results."""
    print("=" * 70)
    print("SCANNER COMPARISON")
    print("=" * 70)
    print()

    # Run V2 first
    print(">>> Running SmartScannerV2...")
    result_v2 = run_v2()

    print()
    print(">>> Running Legacy SmartScanner...")
    result_legacy = run_legacy()

    # Compare
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    v2_tickers = set(s.ticker for s in result_v2.signals)
    legacy_tickers = set(s.ticker for s in result_legacy.signals)

    common = v2_tickers & legacy_tickers
    v2_only = v2_tickers - legacy_tickers
    legacy_only = legacy_tickers - v2_tickers

    print(f"\nV2 signals: {len(result_v2.signals)}")
    print(f"Legacy signals: {len(result_legacy.signals)}")
    print(f"Common: {len(common)}")

    if common:
        print(f"\nCommon signals: {', '.join(sorted(common))}")
    if v2_only:
        print(f"V2 only: {', '.join(sorted(v2_only))}")
    if legacy_only:
        print(f"Legacy only: {', '.join(sorted(legacy_only))}")

    # Compare top signal strengths
    if result_v2.signals and result_legacy.signals:
        print("\nTop signal comparison:")
        print(f"  V2 top: {result_v2.signals[0].ticker} @ {result_v2.signals[0].adjusted_strength:.1f}")
        print(f"  Legacy top: {result_legacy.signals[0].ticker} @ {result_legacy.signals[0].adjusted_strength:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Run Proteus trading scanner')
    parser.add_argument('--model', type=str, default='hybrid',
                       choices=['hybrid', 'lstm', 'mlp'],
                       help='Model to use: hybrid (default), lstm, or mlp')
    parser.add_argument('--mode', type=str, default='aggressive',
                       choices=['aggressive', 'balanced', 'conservative'],
                       help='Trading mode: aggressive (skip CHOPPY, +136%% Sharpe), balanced, conservative')
    parser.add_argument('--sentiment', action='store_true',
                       help='Enable sentiment filtering (Jan 7, 2026 - filters by news sentiment)')
    parser.add_argument('--legacy', action='store_true', help='Run legacy scanner')
    parser.add_argument('--compare', action='store_true', help='Compare V2 vs legacy')
    args = parser.parse_args()

    if args.compare:
        compare()
    elif args.legacy:
        run_legacy()
    else:
        run_v2(model=args.model, trading_mode=args.mode, use_sentiment=args.sentiment)


if __name__ == '__main__':
    main()

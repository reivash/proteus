"""
Proteus Full Research Suite

Master orchestrator that runs all research systems:
1. GPU Model Training/Scan
2. Earnings Watcher
3. Sector Review
4. Trade Retrospective
5. Deep Dive Analysis
6. Generate Consolidated Report

Run overnight to maximize Claude Max quota usage.
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def print_header(title: str):
    """Print section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def run_gpu_model():
    """Train and run GPU signal model."""
    print_header("GPU SIGNAL MODEL")
    try:
        from models.gpu_signal_model import GPUSignalModel

        model = GPUSignalModel()

        # Train if needed
        weights_file = "models/gpu_signal/model_weights.pt"
        if not os.path.exists(weights_file):
            print("[INFO] Training GPU model...")
            model.train_on_history(epochs=50)
        else:
            print("[INFO] GPU model already trained")

        # Run scan
        signals = model.scan_all()

        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'device': str(model.device),
            'top_signals': [
                {
                    'ticker': s.ticker,
                    'strength': s.signal_strength,
                    'prob': s.mean_reversion_prob,
                    'expected_return': s.expected_return,
                    'confidence': s.confidence
                }
                for s in signals[:10]
            ]
        }

        os.makedirs("data/gpu_signals", exist_ok=True)
        with open("data/gpu_signals/latest_scan.json", 'w') as f:
            json.dump(output, f, indent=2)

        print(f"[OK] GPU scan complete - {len(signals)} stocks analyzed")
        return output

    except Exception as e:
        print(f"[ERROR] GPU model failed: {e}")
        return None


def run_earnings_watcher():
    """Run earnings analysis."""
    print_header("EARNINGS WATCHER")
    try:
        from research.earnings_watcher import EarningsWatcher

        watcher = EarningsWatcher()

        # Get upcoming earnings
        upcoming = watcher.get_upcoming_earnings(days_ahead=14)
        print(f"[INFO] {len(upcoming)} stocks with upcoming earnings")

        # Get recent earnings
        recent = watcher.get_recent_earnings(days_back=14)
        print(f"[INFO] {len(recent)} stocks with recent earnings")

        # Note: Full analysis requires API key
        # We'll just return the calendar info
        output = {
            'timestamp': datetime.now().isoformat(),
            'upcoming': upcoming,
            'recent': recent
        }

        os.makedirs("data/earnings_analysis", exist_ok=True)
        with open("data/earnings_analysis/latest_calendar.json", 'w') as f:
            json.dump(output, f, indent=2)

        print(f"[OK] Earnings calendar updated")
        return output

    except Exception as e:
        print(f"[ERROR] Earnings watcher failed: {e}")
        return None


def run_sector_review():
    """Run sector rotation analysis."""
    print_header("SECTOR REVIEW")
    try:
        from research.sector_review import SectorReviewer

        reviewer = SectorReviewer()
        report = reviewer.run_weekly_review()

        output = {
            'timestamp': datetime.now().isoformat(),
            'market_regime': report.market_regime,
            'rotation_theme': report.rotation_theme,
            'sector_rankings': report.sector_rankings[:5],
            'insights': report.actionable_insights,
            'top_adjustments': sorted(
                report.conviction_adjustments.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]
        }

        print(f"[OK] Sector review complete - Regime: {report.market_regime}")
        return output

    except Exception as e:
        print(f"[ERROR] Sector review failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_trade_retrospective():
    """Run trade analysis."""
    print_header("TRADE RETROSPECTIVE")
    try:
        from research.trade_retrospective import TradeRetrospective

        retro = TradeRetrospective()
        report = retro.run_retrospective(days_back=90)

        output = {
            'timestamp': datetime.now().isoformat(),
            'trades_analyzed': report.total_trades_analyzed,
            'win_rate': report.win_rate,
            'profit_factor': report.profit_factor,
            'recommendations': report.recommendations,
            'parameter_suggestions': report.parameter_suggestions
        }

        print(f"[OK] Retrospective complete - {report.total_trades_analyzed} trades analyzed")
        return output

    except Exception as e:
        print(f"[ERROR] Trade retrospective failed: {e}")
        return None


def run_deep_dives():
    """Run deep dive analysis."""
    print_header("DEEP DIVE ANALYSIS")
    try:
        # Check for stale analyses
        deep_dive_dir = "data/deep_dives"
        os.makedirs(deep_dive_dir, exist_ok=True)

        from trading.signal_scanner import SignalScanner
        tickers = SignalScanner.PROTEUS_TICKERS

        # Count existing analyses
        existing = 0
        for ticker in tickers:
            analysis_file = os.path.join(deep_dive_dir, f"{ticker}_*.json")
            import glob
            if glob.glob(os.path.join(deep_dive_dir, f"{ticker}_*.json")):
                existing += 1

        print(f"[INFO] {existing}/{len(tickers)} stocks have deep dive analyses")

        # Run overnight deep dives if not done recently
        summary_file = f"data/deep_dives/overnight_summary_{datetime.now().strftime('%Y-%m-%d')}.json"
        if not os.path.exists(summary_file):
            print("[INFO] Running overnight deep dives...")
            exec(open("run_overnight_deep_dives.py").read())
        else:
            print("[INFO] Today's deep dives already complete")

        # Load summary
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                return json.load(f)

        return {'status': 'completed', 'existing_analyses': existing}

    except Exception as e:
        print(f"[ERROR] Deep dives failed: {e}")
        return None


def generate_consolidated_report(results: Dict) -> str:
    """Generate consolidated research report."""
    print_header("CONSOLIDATED REPORT")

    report_lines = [
        "=" * 70,
        "PROTEUS RESEARCH SUITE - CONSOLIDATED REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        ""
    ]

    # GPU Model Results
    if results.get('gpu_model'):
        gpu = results['gpu_model']
        report_lines.append("GPU SIGNAL MODEL")
        report_lines.append("-" * 40)
        report_lines.append(f"Device: {gpu.get('device', 'N/A')}")
        report_lines.append("Top 5 Signals:")
        for sig in gpu.get('top_signals', [])[:5]:
            report_lines.append(f"  {sig['ticker']}: Strength={sig['strength']:.1f}, "
                              f"Prob={sig['prob']:.2f}, E[R]={sig['expected_return']:+.1f}%")
        report_lines.append("")

    # Earnings Calendar
    if results.get('earnings'):
        earn = results['earnings']
        report_lines.append("EARNINGS CALENDAR")
        report_lines.append("-" * 40)
        report_lines.append(f"Upcoming (14 days): {len(earn.get('upcoming', []))} stocks")
        for e in earn.get('upcoming', [])[:5]:
            report_lines.append(f"  {e['ticker']}: {e['earnings_date']} ({e['days_until']} days)")
        report_lines.append("")

    # Sector Review
    if results.get('sector'):
        sector = results['sector']
        report_lines.append("SECTOR ANALYSIS")
        report_lines.append("-" * 40)
        report_lines.append(f"Market Regime: {sector.get('market_regime', 'N/A')}")
        report_lines.append(f"Rotation Theme: {sector.get('rotation_theme', 'N/A')}")
        report_lines.append("Insights:")
        for insight in sector.get('insights', [])[:3]:
            report_lines.append(f"  - {insight}")
        report_lines.append("")

    # Trade Retrospective
    if results.get('retrospective'):
        retro = results['retrospective']
        report_lines.append("TRADE RETROSPECTIVE")
        report_lines.append("-" * 40)
        report_lines.append(f"Trades Analyzed: {retro.get('trades_analyzed', 0)}")
        report_lines.append(f"Win Rate: {retro.get('win_rate', 0):.1f}%")
        report_lines.append(f"Profit Factor: {retro.get('profit_factor', 0):.2f}")
        if retro.get('recommendations'):
            report_lines.append("Recommendations:")
            for rec in retro.get('recommendations', [])[:3]:
                report_lines.append(f"  - {rec}")
        report_lines.append("")

    # Deep Dives
    if results.get('deep_dives'):
        dd = results['deep_dives']
        report_lines.append("DEEP DIVE ANALYSIS")
        report_lines.append("-" * 40)
        if 'tier_counts' in dd:
            report_lines.append(f"HIGH conviction: {dd['tier_counts'].get('HIGH', 0)} stocks")
            report_lines.append(f"MEDIUM conviction: {dd['tier_counts'].get('MEDIUM', 0)} stocks")
            report_lines.append(f"LOW/AVOID: {dd['tier_counts'].get('LOW', 0) + dd['tier_counts'].get('AVOID', 0)} stocks")
        report_lines.append("")

    # Summary
    report_lines.append("=" * 70)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("Research modules completed:")
    report_lines.append(f"  [{'X' if results.get('gpu_model') else ' '}] GPU Signal Model")
    report_lines.append(f"  [{'X' if results.get('earnings') else ' '}] Earnings Watcher")
    report_lines.append(f"  [{'X' if results.get('sector') else ' '}] Sector Review")
    report_lines.append(f"  [{'X' if results.get('retrospective') else ' '}] Trade Retrospective")
    report_lines.append(f"  [{'X' if results.get('deep_dives') else ' '}] Deep Dive Analysis")
    report_lines.append("")

    report = "\n".join(report_lines)

    # Save report
    report_file = f"data/research_reports/suite_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs("data/research_reports", exist_ok=True)
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"[SAVED] {report_file}")

    return report


def main():
    """Run full research suite."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "    PROTEUS FULL RESEARCH SUITE".center(68) + "#")
    print("#" + f"    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70 + "\n")

    results = {}
    start_time = time.time()

    # Run each module
    results['gpu_model'] = run_gpu_model()
    results['earnings'] = run_earnings_watcher()
    results['sector'] = run_sector_review()
    results['retrospective'] = run_trade_retrospective()
    results['deep_dives'] = run_deep_dives()

    # Generate consolidated report
    report = generate_consolidated_report(results)
    print("\n" + report)

    # Timing
    elapsed = time.time() - start_time
    print(f"\n[COMPLETE] Research suite finished in {elapsed/60:.1f} minutes")

    # Save full results
    results_file = f"data/research_reports/full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[SAVED] {results_file}")

    return results


if __name__ == "__main__":
    main()

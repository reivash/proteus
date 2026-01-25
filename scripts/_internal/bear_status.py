#!/usr/bin/env python3
"""
Bear Detection CLI - Quick status and reports

Usage:
    python scripts/bear_status.py              # Quick status
    python scripts/bear_status.py --full       # Full daily report
    python scripts/bear_status.py --dashboard  # Monitoring dashboard
    python scripts/bear_status.py --sectors    # Sector risk ranking
    python scripts/bear_status.py --validate   # Run validation
    python scripts/bear_status.py --health     # System health check
"""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description='Bear Detection CLI')
    parser.add_argument('--full', action='store_true', help='Full daily report')
    parser.add_argument('--dashboard', action='store_true', help='Monitoring dashboard')
    parser.add_argument('--sectors', action='store_true', help='Sector risk ranking')
    parser.add_argument('--quality', action='store_true', help='Signal quality report')
    parser.add_argument('--strength', action='store_true', help='Enhanced signal strength analysis')
    parser.add_argument('--adaptive', action='store_true', help='Adaptive thresholds and early warning')
    parser.add_argument('--playbook', action='store_true', help='Bear market action playbook')
    parser.add_argument('--outlook', action='store_true', help='Next-day market outlook')
    parser.add_argument('--briefing', action='store_true', help='Daily briefing (email-ready)')
    parser.add_argument('--timing', action='store_true', help='Market timing analysis')
    parser.add_argument('--master', action='store_true', help='Complete master summary report')
    parser.add_argument('--worry', action='store_true', help='Quick should-I-worry check')
    parser.add_argument('--compare', action='store_true', help='Day-over-day comparison')
    parser.add_argument('--sector-signals', action='store_true', help='Sector-specific bear signals')
    parser.add_argument('--heat-map', action='store_true', help='Risk heat map')
    parser.add_argument('--traffic-light', action='store_true', help='Simple traffic light indicator')
    parser.add_argument('--executive', action='store_true', help='Executive summary for quick decisions')
    parser.add_argument('--save-snapshot', action='store_true', help='Save current snapshot to history')
    parser.add_argument('--scenarios', action='store_true', help='Scenario stress test')
    parser.add_argument('--ultimate', action='store_true', help='Ultimate warning report')
    parser.add_argument('--health', action='store_true', help='System health check')
    parser.add_argument('--validate', action='store_true', help='Run historical validation')
    parser.add_argument('--backtest', action='store_true', help='Run backtest analysis')
    parser.add_argument('--signal-check', action='store_true', help='Validate current signal')
    parser.add_argument('--json', action='store_true', help='JSON output for API')
    parser.add_argument('--benchmark', action='store_true', help='Performance benchmark')
    parser.add_argument('--ml-weights', action='store_true', help='ML dynamic weight optimization')
    parser.add_argument('--options-flow', action='store_true', help='Options flow anomaly detection')
    parser.add_argument('--inst-flow', action='store_true', help='Institutional flow signals')
    parser.add_argument('--contagion', action='store_true', help='International contagion detection')
    parser.add_argument('--divergence', action='store_true', help='Sentiment divergence analysis')
    parser.add_argument('--advanced', action='store_true', help='Full advanced ML & flow report')
    parser.add_argument('--composite', action='store_true', help='Composite early warning')
    parser.add_argument('--vol-surface', action='store_true', help='Volatility surface analysis')
    parser.add_argument('--momentum-exhaust', action='store_true', help='Momentum exhaustion signals')
    parser.add_argument('--liquidity', action='store_true', help='Liquidity stress indicators')
    parser.add_argument('--tail-risk', action='store_true', help='Tail risk assessment')
    parser.add_argument('--corr-regime', action='store_true', help='Correlation regime detection')
    parser.add_argument('--crash-breakdown', action='store_true', help='Crash probability breakdown')
    parser.add_argument('--risk-dashboard', action='store_true', help='Full risk dashboard')

    args = parser.parse_args()

    # Import detector
    from common.analysis.fast_bear_detector import FastBearDetector

    detector = FastBearDetector()

    if args.validate:
        # Run validation script
        import subprocess
        subprocess.run([sys.executable, 'scripts/validate_bear_detection.py', '--period', '5y'])
        return

    if args.benchmark:
        print("Running performance benchmark...")
        print("-" * 40)

        # Benchmark detection
        start = time.time()
        for _ in range(5):
            detector.detect(force_refresh=True)
        detect_time = (time.time() - start) / 5
        print(f"Detection: {detect_time:.2f}s average")

        # Benchmark reports
        start = time.time()
        detector.get_daily_summary()
        summary_time = time.time() - start
        print(f"Daily Summary: {summary_time:.2f}s")

        start = time.time()
        detector.get_sector_risk_ranking()
        sector_time = time.time() - start
        print(f"Sector Ranking: {sector_time:.2f}s")

        print("-" * 40)
        print(f"Total typical cycle: {detect_time + summary_time:.2f}s")
        return

    if args.health:
        health = detector.run_health_check()
        print(f"Health Status: {health['status']}")
        print(f"Summary: {health['summary']}")
        print()
        for check, result in health['checks'].items():
            icon = '[OK]' if result['status'] == 'OK' else '[!!]'
            print(f"  {icon} {check}: {result['status']}")
        return

    if args.backtest:
        print(detector.get_backtest_report())
        return

    if args.signal_check:
        print(detector.get_validation_summary())
        return

    if args.json:
        print(detector.get_json_output())
        return

    if args.full:
        print(detector.get_daily_report())
        return

    if args.dashboard:
        print(detector.get_monitoring_dashboard())
        return

    if args.sectors:
        print(detector.get_sector_risk_report())
        return

    if args.quality:
        print(detector.get_quality_report())
        return

    if args.strength:
        print(detector.get_strength_report())
        return

    if args.adaptive:
        print(detector.get_adaptive_report())
        return

    if args.playbook:
        print(detector.get_playbook_report())
        return

    if args.outlook:
        print(detector.get_outlook_report())
        return

    if args.briefing:
        print(detector.get_daily_briefing())
        return

    if args.timing:
        print(detector.get_timing_report())
        return

    if args.master:
        print(detector.get_master_report())
        return

    if args.worry:
        print(detector.get_quick_decision())
        worry = detector.should_i_worry()
        print(f"\nRisk Score: {worry['risk_score']:.1f}/100")
        print(f"Category: {worry['risk_category']}")
        if worry['top_concern'] != 'None':
            print(f"Top Concern: {worry['top_concern']}")
        return

    if args.compare:
        print(detector.get_comparison_report())
        return

    if args.sector_signals:
        print(detector.get_sector_report())
        return

    if args.heat_map:
        heat = detector.get_risk_heat_map()
        print(f"RISK HEAT MAP: {heat['heat_level']} ({heat['avg_risk']:.0f}/100)")
        print()
        print("TIME DIMENSION:")
        for k, v in heat['dimensions']['time'].items():
            bar = '#' * int(v/10) + '-' * (10 - int(v/10))
            print(f"  {k:<12} [{bar}] {v:.0f}")
        print()
        print("TOP SECTOR RISKS:")
        for k, v in list(heat['dimensions']['sectors'].items())[:5]:
            bar = '#' * int(v/10) + '-' * (10 - int(v/10))
            print(f"  {k:<12} [{bar}] {v:.0f}")
        print()
        print(f"Hottest: {heat['hottest_sector']} | Coolest zones: {len(heat['cool_zones'])}")
        return

    if args.traffic_light:
        traffic = detector.get_traffic_light()
        print(traffic['display'])
        print(f"Risk Score: {traffic['risk_score']:.0f}/100")
        print(f"Action: {traffic['action']}")
        return

    if args.executive:
        print(detector.get_executive_report())
        return

    if args.save_snapshot:
        result = detector.save_bear_score_snapshot()
        if result['status'] == 'SUCCESS':
            print(f"Snapshot saved. Total entries: {result['entries']}")
            print(f"  Score: {result['snapshot']['adjusted_score']:.1f}")
            print(f"  Level: {result['snapshot']['alert_level']}")
        else:
            print(f"Error: {result['message']}")
        return

    if args.scenarios:
        print(detector.get_scenario_report())
        return

    if args.ultimate:
        print(detector.get_ultimate_report())
        return

    if args.ml_weights:
        ml = detector.get_ml_dynamic_weights()
        print("=" * 60)
        print("ML DYNAMIC WEIGHT OPTIMIZATION")
        print("=" * 60)
        print(f"Regime: {ml['regime']} | Shift: {ml['weight_shift']}")
        print(f"Top: {ml['top_indicator']} | Weakest: {ml['weakest_indicator']}")
        print()
        print("Dynamic Weights:")
        for ind, wt in sorted(ml['dynamic_weights'].items(), key=lambda x: -x[1]):
            eff = ml['effectiveness'].get(ind, 0)
            bar = '#' * int(wt / 5)
            print(f"  {ind:<15} {wt:5.1f}% [{bar:<20}] eff:{eff:.0f}")
        return

    if args.options_flow:
        opt = detector.get_options_flow_anomaly()
        print("=" * 60)
        print("OPTIONS FLOW ANOMALY DETECTION")
        print("=" * 60)
        print(f"Signal: {opt['flow_signal']} | Score: {opt['anomaly_score']}/100")
        print(f"Put/Call: {opt['put_call_ratio']:.2f} ({opt['pcr_deviation_pct']:+.1f}%)")
        print(f"Action: {opt['action']}")
        print()
        if opt['anomalies']:
            print("Anomalies:")
            for a in opt['anomalies']:
                print(f"  [{a['severity']}] {a['type']}")
                print(f"    {a['detail']}")
        return

    if args.inst_flow:
        inst = detector.get_institutional_flow_signals()
        print("=" * 60)
        print("INSTITUTIONAL FLOW SIGNALS")
        print("=" * 60)
        print(f"Direction: {inst['flow_direction']} | Score: {inst['institutional_score']}/100")
        print(f"{inst['interpretation']}")
        print()
        if inst['signals']:
            print("Signals:")
            for s in inst['signals']:
                print(f"  [{s['strength']}] {s['signal']}: {s['detail']}")
        return

    if args.contagion:
        cont = detector.get_international_contagion()
        print("=" * 60)
        print("INTERNATIONAL CONTAGION DETECTION")
        print("=" * 60)
        if cont.get('status') == 'ERROR':
            print(f"Error: {cont.get('message')}")
        else:
            print(f"Level: {cont['contagion_level']} | Risk: {cont['risk_score']}/100")
            print(f"{cont['warning']}")
            print()
            if cont['signals']:
                print(f"Weak Markets ({cont['markets_weak']}/{cont['markets_analyzed']}):")
                for s in cont['signals']:
                    print(f"  {s['market']} ({s['ticker']}): {s['perf_5d']:+.1f}% (corr: {s['correlation']:.2f})")
        return

    if args.divergence:
        div = detector.get_sentiment_divergence()
        print("=" * 60)
        print("SENTIMENT DIVERGENCE ANALYSIS")
        print("=" * 60)
        print(f"Status: {div['sentiment']} | Score: {div['divergence_score']:+d}")
        print(f"Net Signal: {div['net_signal']}")
        print(f"Action: {div['action']}")
        print()
        if div['divergences']:
            print(f"Divergences ({div['bearish_divergences']} bearish, {div['bullish_divergences']} bullish):")
            for d in div['divergences']:
                print(f"  [{d['strength']}] {d['type']}")
                print(f"    {d['description']}")
        return

    if args.composite:
        comp = detector.get_composite_early_warning()
        print("=" * 60)
        print("COMPOSITE EARLY WARNING")
        print("=" * 60)
        print(f"Level: {comp['warning_level']} | Score: {comp['composite_score']:.1f}/100")
        print(f"Action: {comp['action']}")
        print()
        print("Component Scores:")
        for k, v in comp['component_scores'].items():
            bar = '#' * int(v / 5)
            print(f"  {k:<25} [{bar:<20}] {v:.0f}")
        print()
        if comp['key_concerns']:
            print("Key Concerns:")
            for c in comp['key_concerns']:
                print(f"  [!] {c}")
        return

    if args.advanced:
        print(detector.get_advanced_report())
        return

    if args.vol_surface:
        vol = detector.get_volatility_surface_analysis()
        print("=" * 60)
        print("VOLATILITY SURFACE ANALYSIS")
        print("=" * 60)
        print(f"Assessment: {vol['overall_assessment']} | Score: {vol['surface_score']}/100")
        print(f"VIX: {vol['vix_level']} ({vol['vol_level']}) - {vol['vol_signal']}")
        print()
        print("Term Structure:")
        print(f"  Ratio: {vol['term_structure']['ratio']:.3f}")
        print(f"  State: {vol['term_structure']['state']} ({vol['term_structure']['signal']})")
        print(f"  {vol['term_structure']['description']}")
        print()
        print("Compression:")
        print(f"  Value: {vol['compression']['value']:.2f} ({vol['compression']['state']})")
        print(f"  {vol['compression']['warning']}")
        print()
        print(f"Action: {vol['action']}")
        return

    if args.momentum_exhaust:
        mom = detector.get_momentum_exhaustion_signals()
        print("=" * 60)
        print("MOMENTUM EXHAUSTION SIGNALS")
        print("=" * 60)
        print(f"Level: {mom['exhaustion_level']} | Score: {mom['exhaustion_score']}/100")
        print(f"Outlook: {mom['outlook']}")
        print()
        print(f"Key Metrics:")
        print(f"  Momentum Exhaustion: {mom['momentum_exhaustion']:.2f}")
        print(f"  Breadth: {mom['breadth']:.1f}%")
        print(f"  New High/Low Ratio: {mom['new_high_low_ratio']:.2f}")
        print(f"  McClellan: {mom['mcclellan']:.1f}")
        print()
        if mom['signals']:
            print(f"Signals ({mom['signal_count']}):")
            for s in mom['signals']:
                print(f"  [{s['severity']}] {s['type']}")
                print(f"    {s['detail']}")
        return

    if args.liquidity:
        liq = detector.get_liquidity_stress_indicators()
        print("=" * 60)
        print("LIQUIDITY STRESS INDICATORS")
        print("=" * 60)
        print(f"State: {liq['liquidity_state']} | Score: {liq['stress_score']}/100")
        print(f"Action: {liq['action']}")
        print()
        print("Key Metrics:")
        print(f"  Credit Spread Change: {liq['credit_spread_change']:+.3f}%")
        print(f"  High Yield Spread: {liq['high_yield_spread']:.2f}%")
        print(f"  Bond Volatility: {liq['bond_volatility']:.1f}")
        print(f"  Dollar Strength: {liq['dollar_strength']:+.2f}%")
        print()
        if liq['indicators']:
            print(f"Stress Indicators ({liq['indicator_count']}):")
            for ind in liq['indicators']:
                print(f"  [{ind['severity']}] {ind['indicator']}: {ind['value']}")
                print(f"    {ind['detail']}")
        return

    if args.tail_risk:
        tail = detector.get_tail_risk_assessment()
        print("=" * 60)
        print("TAIL RISK ASSESSMENT")
        print("=" * 60)
        print(f"Level: {tail['risk_level']} | Score: {tail['tail_score']}/100")
        print(f"Expected Tail Move: {tail['expected_tail_move']:.1f}%")
        print(f"Recommendation: {tail['recommendation']}")
        print()
        print("Key Metrics:")
        print(f"  Skew Index: {tail['skew_index']:.1f}")
        print(f"  Vol Compression: {tail['vol_compression']:.2f}")
        print(f"  Crash Probability: {tail['crash_probability']:.1f}%")
        print()
        if tail['factors']:
            print(f"Risk Factors ({tail['factor_count']}):")
            for f in tail['factors']:
                print(f"  [{f['severity']}] {f['factor']}: {f['value']}")
                print(f"    {f['detail']}")
        return

    if args.corr_regime:
        corr = detector.get_correlation_regime()
        print("=" * 60)
        print("CORRELATION REGIME DETECTION")
        print("=" * 60)
        print(f"Regime: {corr['regime']} | Score: {corr['regime_score']}/100")
        print(f"{corr['regime_description']}")
        print()
        print(f"Diversification Benefit: {corr['diversification_benefit']}")
        print(f"Correlation Spike: {corr['correlation_spike']:.2f}")
        print(f"Flight to Quality: {'YES' if corr['flight_to_quality'] else 'NO'}")
        print()
        print(f"Action: {corr['action']}")
        print()
        if corr['signals']:
            print("Correlation Signals:")
            for s in corr['signals']:
                print(f"  {s['pair']}: {s['signal']} - {s['implication']}")
        return

    if args.crash_breakdown:
        crash = detector.get_crash_probability_breakdown()
        print("=" * 60)
        print("CRASH PROBABILITY BREAKDOWN")
        print("=" * 60)
        print(f"Crash Probability: {crash['crash_probability']:.1f}% [{crash['risk_category']}]")
        print(f"Primary Driver: {crash['primary_driver']} ({crash['primary_contribution']:.1f}%)")
        print(f"Action: {crash['action']}")
        print()
        print("Factor Contributions:")
        for name, data in crash['factors'].items():
            bar = '#' * int(data['contribution'] / 2)
            print(f"  {name:<22} [{bar:<10}] {data['contribution']:5.1f}% ({data['status']})")
        print()
        print(f"Model Bear Score: {crash['model_bear_score']:.1f}")
        print(f"Model Crash Prob: {crash['model_crash_prob']:.1f}%")
        return

    if args.risk_dashboard:
        print(detector.get_risk_dashboard())
        return

    # Default: Quick status
    print("=" * 60)
    print("BEAR DETECTION - QUICK STATUS")
    print("=" * 60)
    print()
    print(detector.get_one_liner())
    print()

    signal = detector.detect()
    quality = detector.get_signal_quality()
    recovery = detector.detect_recovery()
    regime = detector.get_regime_adjusted_score()
    early = detector.get_early_warning_composite()

    print(f"Crash Probability: {signal.crash_probability:.1f}%")
    print(f"Early Warning: {early['composite_score']:.1f}/100 ({early['warning_status']})")
    print(f"Signal Quality: Grade {quality['quality_grade']} ({quality['quality_score']}/100)")
    print(f"Recovery Status: {recovery['status']}")

    # Show adaptive adjustment if it changes the level
    if regime['level_changed']:
        print()
        print(f"[!] Adaptive Alert: {regime['raw_level']} -> {regime['adjusted_level']}")
        print(f"    (Raw {regime['raw_score']:.1f} + adjustment {regime['total_adjustment']:+.0f} = {regime['adjusted_score']:.1f})")
    print()

    # Watch items
    summary = detector.get_daily_summary()
    if summary['watch_list']:
        print("Watch Items:")
        for item in summary['watch_list']:
            print(f"  [!] {item}")
        print()

    print(f"Action: {summary['recommended_action']}")
    print()
    print("=" * 60)
    print("Use --full for detailed report, --help for all options")

if __name__ == '__main__':
    main()

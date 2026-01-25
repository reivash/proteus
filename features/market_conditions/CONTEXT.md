# Regime Detection 2.0 - Working Context

> This file helps Claude maintain context across sessions.
> Update after each work session.

---

## Current Status

**Week**: 1-2 (Complete)
**Phase**: Foundation & Research
**Last Updated**: January 25, 2026

## Completed Work

### Quick Wins (Jan 25)
- [x] HMM probabilities in scan output
- [x] VIX term structure (VIX/VIX3M)
- [x] Days in regime tracking
- [x] Enhanced regime logging

### Week 1-2 Research (Jan 25)
- [x] HMM analysis → `research/hmm_analysis.md`
- [x] Transition analysis → 10.4% early detection
- [x] Misclassification analysis → vol_underweight, momentum_lag
- [x] Benchmark → HMM 46%, Rule-based 87%
- [x] Performance dashboard

## Key Metrics (Baseline)

| Metric | Current | Q2 Target | Q4 Target |
|--------|---------|-----------|-----------|
| Accuracy | 46% (HMM) | 70% | 85% |
| Early Detection | 10.4% | 40% | 60% |
| Bull→Bear Detection | 0% | 30% | 50% |
| Stability (days) | 2.7 | 5.0 | 7.0 |

## Next Up: Week 3-4

### Competitive Intelligence
- [ ] Study AQR's "Market Regime Indicators" whitepaper
- [ ] Analyze JPMorgan's "Macro Regime Framework"
- [ ] Review Goldman's "Risk Appetite Indicator"
- [ ] Document Bloomberg Terminal regime classification
- [ ] Write summary: `research/institutional_approaches.md`

## Key Files

| Purpose | Path |
|---------|------|
| Main detector | `common/analysis/unified_regime_detector.py` |
| HMM engine | `common/analysis/hmm_regime_detector.py` |
| Scanner integration | `common/trading/smart_scanner_v2.py` |
| Research outputs | `features/market_conditions/research/` |
| Analysis scripts | `features/market_conditions/scripts/` |
| Development plan | `features/market_conditions/PLAN.md` |

## Critical Insights

1. **HMM is too jittery** - transitions every 2.7 days (should be 7+)
2. **Missing features** - VIX direct, term structure, breadth, credit spreads
3. **Bull→Bear is blind** - 0% early detection on most critical transition
4. **Rule-based wins** - Simple rules beat complex HMM currently

## Improvement Priorities

1. Add VIX as direct HMM feature (Week 5-6)
2. Add VIX term structure as feature (Week 5-6)
3. Implement regime duration modeling (Week 7-8)
4. Add market breadth feature (Week 7-8)
5. Reduce jitter with smoothing (Week 9-10)

## Commands

```bash
# Run performance dashboard
python features/market_conditions/scripts/performance_dashboard.py

# Run transition analysis
python features/market_conditions/scripts/analyze_transitions.py

# Run benchmark
python features/market_conditions/scripts/benchmark_regime_detection.py

# Test regime detection
python -c "from common.analysis.unified_regime_detector import print_regime_comparison; print_regime_comparison()"
```

---

*Update this file after each work session*

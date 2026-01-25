# Position Sizing

Kelly Criterion-based position sizing with regime and tier adjustments.

## What It Does

Calculates optimal position size for each trade based on:
- Historical edge (win rate, avg win/loss by tier)
- Current market regime
- Portfolio heat (current risk exposure)
- Stock tier quality

## Sizing Formula

```
base_size = kelly_fraction * signal_strength_scale
adjusted_size = base_size * regime_multiplier * tier_limit
final_size = clamp(adjusted_size, min=2%, max=15%)
```

## Regime Multipliers

| Regime | Multiplier | Rationale |
|--------|------------|-----------|
| Volatile | 1.3x | Highest Sharpe (4.86) |
| Bear | 1.2x | High Sharpe (3.32) |
| Choppy | 1.0x | Baseline |
| Bull | 0.8x | Lowest edge |

## Tier Limits

| Tier | Max Size | Win Rate |
|------|----------|----------|
| Elite | 15% | 65% |
| Strong | 12% | 60% |
| Average | 7.5% | 55% |
| Weak | 4% | 50% |

## Risk Controls

- Max 6 concurrent positions
- Max 15% portfolio heat
- Max 2 positions per sector
- Max 2% risk per trade

## Key Files

| File | Purpose |
|------|---------|
| `common/trading/unified_position_sizer.py` | Main sizer |
| `common/trading/position_rebalancer.py` | Exit monitoring |

## See Also

- [PLAN.md](PLAN.md) - Development roadmap with checkboxes
- [results/](results/) - Sizing analysis results

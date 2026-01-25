# Recommendations

Daily buy recommendations with confidence levels and edge metrics.

## What It Does

Transforms raw scanner signals into actionable recommendations with:
- Confidence levels (HIGH, MODERATE, LOW)
- Expected edge (win rate, avg return)
- Position sizing suggestions
- Risk caveats and context

## Confidence Levels

| Level | Criteria | Recommendation |
|-------|----------|----------------|
| HIGH | Raw 70+, Elite/Strong tier, Adjusted 65+ | Strong buy |
| MODERATE | Raw 65+, Average+ tier, Adjusted 55+ | Consider buying |
| LOW | Below thresholds | Watchlist only |

## Output Example

```
#1 MPC (Marathon Petroleum)
   Signal: 78/100 | Tier: ELITE | Confidence: HIGH
   Edge: +1.2% expected (68% win rate, n=127)
   Position: 12% ($5,400) | Targets: +2%/+3.5% | Stop: -3%
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/recommendations_gpu.py` | Entry point |
| `common/trading/smart_scanner_v2.py` | Signal generation |

## Usage

```bash
python scripts/recommendations_gpu.py          # Full report
python scripts/recommendations_gpu.py --quiet  # Summary only
python scripts/recommendations_gpu.py --json   # JSON output
```

## Delivery Channels

- Console output (default)
- Email (via Mailjet/SendGrid)
- JSON file export

## See Also

- [PLAN.md](PLAN.md) - Development roadmap with checkboxes
- [results/](results/) - Recommendation accuracy tracking

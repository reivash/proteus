# 3-Hour Refactoring Plan

> **Goal**: Make the system more manageable, cohesive, and self-documenting by completing vertical feature organization.

---

## Hour 1: Organize scripts/_internal/

**Current State**: 50+ utility scripts scattered in `scripts/_internal/`

### 1.1 Map scripts to features (15 min)
| Script | Target Feature |
|--------|----------------|
| `bear_monitor.py`, `bear_status.py`, `validate_bear_*` | `features/crash_warnings/scripts/` |
| `generate_dashboard.py` | `features/reporting/scripts/` |
| `daily_report.py`, `send_daily_report.py` | `features/reporting/scripts/` |
| `optimize_threshold.py`, `validate_modifiers.py` | `features/daily_picks/scripts/` |

### 1.2 Move feature-specific scripts (30 min)
- [ ] Create `scripts/` subfolder in each feature
- [ ] Move bear-related scripts to `features/crash_warnings/scripts/`
- [ ] Move report scripts to `features/reporting/scripts/`
- [ ] Move signal tuning scripts to `features/daily_picks/scripts/`
- [ ] Update any hardcoded paths

### 1.3 Clean up scripts/_internal/ (15 min)
- [ ] Keep only truly cross-feature utilities
- [ ] Rename to `scripts/utilities/` or delete if empty
- [ ] Update batch files to use new paths

---

## Hour 2: Organize experiments and research

**Current State**: 130+ experiments in `common/experiments/`, research scattered

### 2.1 Map experiments to features (15 min)
| Experiment Range | Feature |
|------------------|---------|
| exp052, exp095 (regime) | `features/market_conditions/experiments/` |
| exp074-077 (exit strategy) | `features/trade_sizing/experiments/` |
| exp029, exp076 (position sizing) | `features/trade_sizing/experiments/` |
| exp087-091 (model training) | `features/daily_picks/experiments/` |
| Bear-related experiments | `features/crash_warnings/experiments/` |

### 2.2 Move active experiments (30 min)
- [ ] Create `experiments/` subfolder in relevant features
- [ ] Move 33 active experiments to appropriate features
- [ ] Keep `common/experiments/archived/` as historical reference
- [ ] Update experiment imports if needed

### 2.3 Consolidate research docs (15 min)
- [ ] Move `common/research/` analysis scripts to features
- [ ] `exit_strategy_*.py` → `features/trade_sizing/research/`
- [ ] `regime_*.py` → `features/market_conditions/research/`
- [ ] Update features RESEARCH.md files

---

## Hour 3: Documentation cleanup

**Current State**: Docs scattered, some outdated, SYSTEM_STATE.md needs update

### 3.1 Update feature documentation (20 min)
- [ ] Add README.md to features missing it
- [ ] Ensure each PLAN.md has accurate checkboxes
- [ ] Add usage examples to each feature's run.py docstring

### 3.2 Consolidate system docs (20 min)
- [ ] Update SYSTEM_STATE.md with new structure
- [ ] Update QUICKSTART.md with feature-based commands
- [ ] Archive outdated docs in `docs/archive/`
- [ ] Delete empty/redundant docs

### 3.3 Clean up data directories (20 min)
- [ ] Remove any remaining duplicate data folders
- [ ] Verify gitignore covers all generated data
- [ ] Clean up `data/` to only have global caches/models
- [ ] Document what each data folder contains

---

## Validation Checklist

After each hour:
- [ ] Run `python tests/test_smoke.py` - should pass 19/20
- [ ] Run `python features/daily_picks/run.py --help` - should work
- [ ] Commit changes with clear message

---

## Files to Delete (End of Session)

After validation:
- [ ] Empty directories
- [ ] Duplicate configs
- [ ] Stale batch files referencing old paths
- [ ] Old README files superseded by feature docs

---

## Success Criteria

1. **Every feature is self-contained**: run.py, PLAN.md, config.json, data/, tests/, scripts/
2. **Root level is minimal**: only global config, shared common/, integration tests
3. **Documentation is accurate**: all paths/commands work as documented
4. **No orphaned files**: everything has a clear home

---

*Estimated completion: 3 hours*
*Priority: Structure clarity over perfect organization*

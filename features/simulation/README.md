# Paper Trading

Virtual wallet for testing the system without real money.

## What It Does

Simulates real trading with:
- Virtual capital ($100,000 default)
- Position tracking with entry/exit prices
- P&L calculation and performance metrics
- Tier-based exit rules (profit targets, stop losses, time limits)

## Usage

```bash
python features/simulation/run.py --status   # View current status
python features/simulation/run.py --full     # Full daily cycle
python features/simulation/run.py --reset    # Reset to initial state
```

## Key Files

| File | Purpose |
|------|---------|
| `run.py` | Entry point |
| `config.json` | Exit strategy rules |
| `common/trading/virtual_wallet.py` | Core wallet logic |
| `data/virtual_wallet/` | State files |

## Data Files

| File | Contents |
|------|----------|
| `wallet_state.json` | Current positions, cash, equity |
| `trade_history.json` | All completed trades |
| `daily_snapshots.json` | Daily equity curve |

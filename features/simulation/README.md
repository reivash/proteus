# Paper Trading

Virtual wallet for testing the system without real money.

## What It Does

Simulates real trading with:
- Virtual capital ($100,000 default)
- Position tracking with entry/exit prices
- P&L calculation and performance metrics
- Daily snapshots and trade history

## Features

- **Position Management**: Open, track, and close positions
- **Exit Rules**: Profit targets, stop losses, time limits
- **Performance Tracking**: Win rate, Sharpe, drawdown
- **Email Summaries**: Daily portfolio updates

## Key Files

| File | Purpose |
|------|---------|
| `common/trading/virtual_wallet.py` | Core wallet logic |
| `scripts/paper_wallet.py` | Entry point |
| `data/virtual_wallet/` | State files |

## Usage

```bash
python scripts/paper_wallet.py --status      # View current status
python scripts/paper_wallet.py --full        # Full daily cycle
python scripts/paper_wallet.py --reset       # Reset to initial state
```

## Data Files

| File | Contents |
|------|----------|
| `wallet_state.json` | Current positions, cash, equity |
| `trade_history.json` | All completed trades |
| `daily_snapshots.json` | Daily equity curve |

## Performance Metrics

The wallet tracks:
- Total return %
- Win rate
- Sharpe ratio
- Max drawdown
- Profit factor
- Average hold days

## See Also

- [PLAN.md](PLAN.md) - Development roadmap
- [results/](results/) - Performance history

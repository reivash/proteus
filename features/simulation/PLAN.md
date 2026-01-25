# Paper Trading System

> **Project Status**: Production (Complete)
> **Version**: 1.0
> **Last Updated**: January 2026

---

## Overview

### Problem Statement
Before risking real money, traders need a way to validate the system with realistic simulation including position tracking, P&L calculation, and performance metrics.

### Objectives
1. **Simulate real trading** with virtual capital
2. **Track all positions** with entry/exit prices
3. **Calculate performance metrics** (win rate, Sharpe, drawdown)
4. **Provide daily summaries** via email

### Key Achievements
- Full paper trading simulation with $100k virtual capital
- Automatic position management (entries, exits, rebalancing)
- Daily email summaries with performance metrics
- Integration with signal scanner pipeline

---

## Implementation Phases

### Phase 1: Core Wallet
**Goal**: Basic position tracking

- [x] **1.1 Wallet State**
  - [x] Cash tracking
  - [x] Position tracking (ticker, shares, entry price)
  - [x] Total equity calculation
  - [x] JSON persistence

- [x] **1.2 Position Management**
  - [x] Open position (from signal)
  - [x] Update position (current price)
  - [x] Close position (exit rules)
  - [x] Partial exits (50% at target 1)

- [x] **1.3 Trade History**
  - [x] Log all completed trades
  - [x] Track outcome (win/loss)
  - [x] Track exit reason
  - [x] Track hold days

### Phase 2: Exit Rules
**Goal**: Automatic position management

- [x] **2.1 Profit Targets**
  - [x] Target 1: Partial exit (50%)
  - [x] Target 2: Full exit
  - [x] Tier-based targets

- [x] **2.2 Stop Losses**
  - [x] Fixed stop by tier
  - [x] Bear-adjusted stops (tighter)

- [x] **2.3 Time Exits**
  - [x] Max hold days by tier
  - [x] Elite: 5 days
  - [x] Strong: 5 days
  - [x] Average: 3 days
  - [x] Weak: 2 days

### Phase 3: Performance Tracking
**Goal**: Comprehensive metrics

- [x] **3.1 Daily Snapshots**
  - [x] Daily equity recording
  - [x] Equity curve generation
  - [x] Drawdown tracking

- [x] **3.2 Performance Metrics**
  - [x] Win rate
  - [x] Sharpe ratio
  - [x] Profit factor
  - [x] Max drawdown
  - [x] Average hold days

- [x] **3.3 Email Summaries**
  - [x] Daily summary email
  - [x] Positions and P&L
  - [x] Recent trades
  - [x] Performance metrics

### Phase 4: Integration
**Goal**: Connect to scanner pipeline

- [x] **4.1 Scanner Integration**
  - [x] Read latest_scan.json
  - [x] Process signals automatically
  - [x] Respect position limits

- [x] **4.2 Rebalancing**
  - [x] Check exit conditions daily
  - [x] Process exits before entries
  - [x] Update heat after exits

---

## Progress Summary

| Phase | Status | Checkboxes |
|-------|--------|------------|
| 1. Core Wallet | Complete | 10/10 |
| 2. Exit Rules | Complete | 8/8 |
| 3. Performance Tracking | Complete | 9/9 |
| 4. Integration | Complete | 5/5 |
| **TOTAL** | **Production** | **32/32** |

---

## Key Files

| File | Purpose |
|------|---------|
| `common/trading/virtual_wallet.py` | Core wallet logic |
| `scripts/paper_wallet.py` | Entry point |
| `features/simulation/data/virtual_wallet/` | State files |

---

*Document Version: 1.0*
*Last Updated: January 2026*
*Status: Production Complete*

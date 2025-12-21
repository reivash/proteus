# Proteus Trading System Architecture

## Overview

The Proteus trading system is a GPU-accelerated mean reversion scanner that generates trading signals for large-cap stocks. This document describes the key components and data flow.

## System Architecture

```
+-------------------------------------------------------------+
|                    ENTRY POINTS                              |
|  smart_scanner.py  |  scheduled_scanner.py  |  daily_runner |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                    SIGNAL GENERATION                         |
|  GPUSignalModel (PyTorch/CUDA) -> Signal Strength + E[R]    |
|  - Multi-task learning: direction + return prediction        |
|  - 14 technical features, 1-year training window             |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                    FILTERING LAYERS                          |
|  1. Market Regime -> Dynamic thresholds (VIX-adjusted)      |
|  2. Per-Stock Config -> Historical performance adjustments   |
|  3. Sector Correlation -> Max 2 per sector, 3 correlated    |
|  4. Earnings Calendar -> Exclude 3 days before/1 day after  |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                    POSITION SIZING                           |
|  VolatilitySizer (ATR-based) -> Risk-adjusted shares        |
|  - 2% max risk per trade, 15% max position                  |
|  - Beta and volatility rank adjustments                     |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                    OUTPUT                                    |
|  SmartSignal -> Tier, Shares, Price, Exit Params, Warnings  |
|  Dashboard -> HTML visualization (auto-refresh)             |
|  JSON -> data/smart_scans/latest_scan.json                  |
+-------------------------------------------------------------+
```

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| SmartScanner | `src/trading/smart_scanner.py` | Main entry point for daily scans |
| GPUSignalModel | `src/models/gpu_signal_model.py` | PyTorch model on RTX 4080 SUPER |
| MarketRegimeDetector | `src/analysis/market_regime.py` | VIX-based regime classification |
| VolatilitySizer | `src/trading/volatility_sizing.py` | ATR-based position sizing |
| EarningsCalendar | `src/data/fetchers/earnings_calendar.py` | Earnings date filtering |
| DashboardGenerator | `src/web/dashboard_generator.py` | HTML dashboard output |
| SignalQualityValidator | `src/research/signal_quality_validator.py` | Backtesting and validation |
| PaperTrader | `src/trading/paper_trader.py` | Simulated trading for validation |

## Signal Tier Classification

| Tier | Strength Range | Trading Action |
|------|----------------|----------------|
| ELITE | 80-100 | Strong buy, full position |
| STRONG | 70-80 | Buy, full position |
| GOOD | 60-70 | Buy, standard position |
| MODERATE | 50-60 | Consider, reduced position |
| WEAK | 40-50 | Avoid unless other signals support |
| POOR | 0-40 | Do not trade |

## Market Regime Classification

| Regime | VIX Range | Position Multiplier | Min Threshold |
|--------|-----------|---------------------|---------------|
| BULL | VIX < 15, SPY up | 1.0x | 55 |
| CHOPPY | VIX 15-20 | 1.0x | 50 |
| VOLATILE | VIX 20-30 | 0.8x | 60 |
| BEAR | VIX > 30 or SPY down | 0.6x | 70 |

## Directory Structure

```
proteus/
├── data/
│   ├── smart_scans/         # Scan results (latest_scan.json)
│   ├── web_dashboard/       # HTML dashboard
│   ├── earnings_cache/      # Cached earnings dates
│   └── stock_configs/       # Per-stock configuration
├── models/
│   └── gpu_signal/          # Trained PyTorch weights
├── src/
│   ├── trading/             # Core trading logic
│   ├── models/              # ML models
│   ├── analysis/            # Regime detection
│   ├── data/                # Data fetchers
│   ├── research/            # Validation tools
│   ├── web/                 # Dashboard generator
│   └── experiments/         # Research (EXP-001 to EXP-100+)
└── logs/
    └── scans/               # Scan logs
```

## Data Flow

1. **Data Fetching**: `yfinance` fetches OHLCV data for 54 stocks
2. **Feature Engineering**: Technical indicators computed (RSI, ATR, BB, etc.)
3. **GPU Inference**: Batch prediction on CUDA with PyTorch
4. **Signal Scoring**: Combine predictions into signal strength (0-100)
5. **Filtering**: Apply regime, sector, earnings filters
6. **Sizing**: Calculate position sizes based on volatility
7. **Output**: Generate JSON + HTML dashboard

## Quick Start

```python
# Run a smart scan
from src.trading.smart_scanner import run_smart_scan
result = run_smart_scan()

# View results
print(f"Found {len(result.signals)} signals")
for sig in result.signals[:5]:
    print(f"{sig.ticker}: {sig.tier} ({sig.adjusted_strength})")
```

## Recent Improvements (Dec 2024)

1. **Dynamic VIX-based thresholds**: Exit parameters scale with volatility
2. **ATR-based position sizing**: Risk-adjusted share counts
3. **Earnings calendar filter**: Avoid trading near announcements
4. **Sector correlation filter**: Prevent over-concentration
5. **GPU multi-task learning**: Joint direction + return prediction

## Validation

The system is validated through:

1. **Backtesting**: 365-day historical signal validation
2. **Paper Trading**: Simulated execution with realistic fills
3. **Signal Quality Reports**: Win rate, hit rate by tier

Run validation:
```bash
python src/research/signal_quality_validator.py
```

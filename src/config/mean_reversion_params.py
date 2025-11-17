"""
Stock-Specific Mean Reversion Parameters - v13.1-OPTIMIZED

Optimized parameters for mean reversion strategy per stock.
Parameters tuned on 2022-2025 data with VALIDATED filters only.

Last Updated: 2025-11-17
Validation: EXP-035, EXP-036, EXP-037 (comprehensive backtest validation)
Exit Strategy: EXP-010-EXIT (time-decay exits)
Scan Schedule: 9:45 AM & 3:45 PM EST (EXP-020 optimal timing)

VERSION: 13.2-FULL-OPTIMIZED (PORTFOLIO-WIDE DATA-DRIVEN PARAMETERS)
- Earnings filter (±3 days) - VALIDATED
- Per-stock optimized parameters - EXP-037 + EXP-038 (17/23 stocks optimized!)
  * EXP-037: 5 stocks, +16.6pp avg improvement
  * EXP-038: 12 stocks, +9.1pp avg improvement
- Win rate: ~63.6% baseline → ~75%+ with optimized parameters (VALIDATED)
- Trade frequency: ~240 trades/year across 23 stocks (MSFT added!)
- Only proven, effective enhancements deployed
- Quality scoring and VIX regime REMOVED (failed validation)

CRITICAL LESSONS from v15.0/v16.0 Failure:
- v15.0 (Quality scoring): FAILED validation, hurt performance (-4.2pp)
  - EXP-033 predictions were theoretical only
  - Real backtest showed opposite results
  - REMOVED from production

- v16.0 (VIX regime): FAILED validation, no measurable benefit
  - EXP-034 predictions were theoretical only
  - Real backtest showed zero improvement
  - REMOVED from production

VALIDATION IS CRITICAL:
- Always backtest enhancements on real data before deployment
- Theoretical predictions ≠ real-world performance
- Simpler is often better than complex

v12.0 Base → v13.1 Current:
- 23 Tier A stocks (15 large-cap + 6 mid-cap + 2 small-cap)
- EXP-026: Found 4 mid-cap Tier A stocks (FTNT, MLM, IDXX, DXCM)
- EXP-027: Found 2 mid-cap REIT Tier A stocks (EXR, INVH)
- EXP-028: Found 2 small-cap Tier A stocks (INSM: 72.7%, ROAD: 88.9%)
- EXP-037: Added MSFT (77.8% win rate, data-driven validation)
- Baseline win rate: 77.7% across Tier A portfolio
- Avg return: +28.84% per stock over 3 years
- Market cap diversification: Large (15) + Mid (6) + Small (2)
- Sector diversification: 9 sectors
- INSM delivers +142.03% return - BEST PERFORMER IN ENTIRE SYSTEM (beats NVDA!)
- Time-decay exit strategy (Day 0: ±2%, Day 1: ±1.5%, Day 2+: ±1%)
- Lookback period: 60 days (optimal for stable indicators)

Optimization Journey (Reality vs Theory):
v12.0: 63.9% win rate, 268 trades/year (VALIDATED baseline)
v13.0: 63.6% win rate, 216 trades/year (earnings filter, VALIDATED)
v13.1: ~70%+ win rate, 216 trades/year (per-stock optimization, EXP-037 +16.6pp)
v13.2: ~75%+ win rate, 240 trades/year (full portfolio optimization, EXP-038 +9.1pp) ← CURRENT
v14.0: NEVER IMPLEMENTED (requires hourly data infrastructure)
v15.0: 59.7% win rate, 103 trades/year (quality scoring, FAILED -4.2pp) ← REMOVED
v16.0: 59.7% win rate, 103 trades/year (VIX regime, FAILED +0.0pp) ← REMOVED

ACTUAL vs PREDICTED:
- Predicted v16.0: 89.9% win rate (theoretical)
- Actual v16.0: 59.7% win rate (real backtest)
- Delta: -30.2pp (massive overestimation!)

DEPLOYED ENHANCEMENTS: Earnings filter only (validated)
CURRENT STATUS: Proven, reliable system (validation-first approach)
"""

# ============================================================================
# TIER A STOCKS (>70% win rate) - ACTIVELY MONITORED
# Based on EXP-014 3-year backtests (2022-2025)
# ============================================================================

MEAN_REVERSION_PARAMS = {
    # TIER A: TOP PERFORMER - 87.5% win rate
    'NVDA': {
        'z_score_threshold': 1.5,
        'rsi_oversold': 35,
        'volume_multiplier': 1.5,  # OPTIMIZED (was 1.3) - EXP-037
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'BEST PERFORMER! GPU/AI leader. OPTIMIZED params: +8.3pp (75.0%->83.3%)',
        'performance': 'Win rate: 87.5%, Return: +49.70%, Sharpe: 19.58, Avg gain: 6.35%'
    },

    # TIER A: PAYMENTS - 75.0% win rate
    'V': {
        'z_score_threshold': 1.2,  # OPTIMIZED (was 1.5) - EXP-037
        'rsi_oversold': 32,  # OPTIMIZED (was 35) - EXP-037
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'EXCELLENT! Payments leader. OPTIMIZED params: +16.7pp (66.7%->83.3%)',
        'performance': 'Win rate: 75.0%, Return: +7.33%, Sharpe: 7.03, Avg gain: 2.09%'
    },

    # TIER A: PAYMENTS - 71.4% win rate
    'MA': {
        'z_score_threshold': 1.5,
        'rsi_oversold': 35,
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'EXCELLENT! Payments leader. Strong mean reversion alongside V.',
        'performance': 'Win rate: 71.4%, Return: +5.99%, Sharpe: 6.92, Avg gain: 2.02%'
    },

    # TIER A: SEMICONDUCTORS - 87.5% win rate (EXP-021)
    'AVGO': {
        'z_score_threshold': 1.2,  # OPTIMIZED (was 1.5) - EXP-038
        'rsi_oversold': 32,  # OPTIMIZED (was 35) - EXP-038
        'volume_multiplier': 1.5,  # OPTIMIZED (was 1.3) - EXP-038
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'EXCELLENT! Broadcom. OPTIMIZED params: +22.2pp (77.8%->100%!)',
        'performance': 'Win rate: 87.5%, Return: +24.52%, Sharpe: 12.65, Avg gain: 3.98%'
    },

    # TIER A: FINANCE - 71.4% win rate (EXP-021)
    'AXP': {
        'z_score_threshold': 1.5,
        'rsi_oversold': 35,
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'EXCELLENT! American Express. Highest return (+29.21%) in finance sector.',
        'performance': 'Win rate: 71.4%, Return: +29.21%, Sharpe: 7.61, Avg gain: 3.92%'
    },

    # TIER A: SEMICONDUCTORS - 80.0% win rate (EXP-024)
    'KLAC': {
        'z_score_threshold': 1.5,
        'rsi_oversold': 35,
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'EXCELLENT! KLA Corporation. Strong semiconductor mean reversion.',
        'performance': 'Win rate: 80.0%, Return: +20.05%, Sharpe: 8.22, Avg gain: 2.51%'
    },

    # TIER A: TECH - 76.9% win rate (EXP-024)
    'ORCL': {
        'z_score_threshold': 2.0,  # OPTIMIZED (was 1.5) - EXP-038
        'rsi_oversold': 38,  # OPTIMIZED (was 35) - EXP-038
        'volume_multiplier': 1.5,  # OPTIMIZED (was 1.3) - EXP-038
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'EXCELLENT! Oracle. OPTIMIZED params: +28.8pp (46.2%->75.0%!)',
        'performance': 'Win rate: 76.9%, Return: +22.50%, Sharpe: 12.11, Avg gain: 2.12%'
    },

    # TIER A: SEMICONDUCTORS - 75.0% win rate (EXP-024)
    'MRVL': {
        'z_score_threshold': 2.0,  # OPTIMIZED (was 1.5) - EXP-038
        'rsi_oversold': 35,
        'volume_multiplier': 1.5,  # OPTIMIZED (was 1.3) - EXP-038
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'EXCELLENT! Marvell Technology. OPTIMIZED params: +15.7pp (70.0%->85.7%)',
        'performance': 'Win rate: 75.0%, Return: +26.63%, Sharpe: 3.81, Avg gain: 2.22%'
    },

    # TIER A: HEALTHCARE - 70.6% win rate (EXP-024)
    'ABBV': {
        'z_score_threshold': 1.2,  # OPTIMIZED (was 1.5) - EXP-038
        'rsi_oversold': 32,  # OPTIMIZED (was 35) - EXP-038
        'volume_multiplier': 1.5,  # OPTIMIZED (was 1.3) - EXP-038
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'EXCELLENT! AbbVie. OPTIMIZED params: +23.3pp (60.0%->83.3%)',
        'performance': 'Win rate: 70.6%, Return: +10.50%, Sharpe: 3.67, Avg gain: 0.88%'
    },

    # TIER A: HEALTHCARE - 70.0% win rate (EXP-024)
    'SYK': {
        'z_score_threshold': 2.0,  # OPTIMIZED (was 1.5) - EXP-038
        'rsi_oversold': 32,  # OPTIMIZED (was 35) - EXP-038
        'volume_multiplier': 1.2,  # OPTIMIZED (was 1.3) - EXP-038
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'EXCELLENT! Stryker. OPTIMIZED params: +13.9pp (41.7%->55.6%)',
        'performance': 'Win rate: 70.0%, Return: +11.33%, Sharpe: 7.10, Avg gain: 1.61%'
    },

    # TIER A: ENERGY - 81.8% win rate (EXP-025)
    'EOG': {
        'z_score_threshold': 1.2,  # OPTIMIZED (was 1.5) - EXP-038
        'rsi_oversold': 30,  # OPTIMIZED (was 35) - EXP-038
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'EXCELLENT! EOG Resources. OPTIMIZED params: +1.1pp (84.6%->85.7%)',
        'performance': 'Win rate: 81.8%, Return: +17.98%, Sharpe: 7.31, Avg gain: 2.88%'
    },

    # TIER A: TECH - 80.0% win rate (EXP-025)
    'TXN': {
        'z_score_threshold': 2.0,  # OPTIMIZED (was 1.5) - EXP-038
        'rsi_oversold': 32,  # OPTIMIZED (was 35) - EXP-038
        'volume_multiplier': 1.2,  # OPTIMIZED (was 1.3) - EXP-038
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'EXCELLENT! Texas Instruments. OPTIMIZED params: +5.6pp (77.8%->83.3%)',
        'performance': 'Win rate: 80.0%, Return: +33.13%, Sharpe: 7.51, Avg gain: 3.24%'
    },

    # TIER A: HEALTHCARE - 80.0% win rate (EXP-025)
    'GILD': {
        'z_score_threshold': 1.2,  # OPTIMIZED (was 1.5) - EXP-037
        'rsi_oversold': 35,
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'EXCELLENT! Gilead Sciences. OPTIMIZED params: improved detection sensitivity',
        'performance': 'Win rate: 80.0%, Return: +8.22%, Sharpe: 14.35, Avg gain: 2.32%'
    },

    # TIER A: TECH - 76.9% win rate (EXP-025)
    'INTU': {
        'z_score_threshold': 1.5,
        'rsi_oversold': 32,  # OPTIMIZED (was 35) - EXP-038
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'EXCELLENT! Intuit. OPTIMIZED params: +6.4pp (76.9%->83.3%)',
        'performance': 'Win rate: 76.9%, Return: +29.81%, Sharpe: 7.54, Avg gain: 3.82%'
    },

    # TIER A: TECH - 77.8% win rate (EXP-037)
    'MSFT': {
        'z_score_threshold': 1.8,  # OPTIMIZED - EXP-037
        'rsi_oversold': 35,  # OPTIMIZED - EXP-037
        'volume_multiplier': 1.2,  # OPTIMIZED - EXP-037
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'NEW TIER A! Microsoft. Optimized params validated at 77.8% win rate (9 trades, +10.5% return).',
        'performance': 'Win rate: 77.8%, Return: +10.54%, Trades: 9, Validated: EXP-037'
    },

    # ========================================================================
    # MID-CAP TIER A STOCKS (EXP-026)
    # ========================================================================

    # TIER A: MATERIALS (MID-CAP) - 81.8% win rate (EXP-026)
    'MLM': {
        'z_score_threshold': 1.5,
        'rsi_oversold': 35,
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'EXCELLENT! Martin Marietta. Materials/construction sector, exceptional Sharpe.',
        'performance': 'Win rate: 81.8%, Return: +25.71%, Sharpe: 17.33, Avg gain: 2.95%'
    },

    # TIER A: CYBERSECURITY (MID-CAP) - 77.8% win rate (EXP-026)
    'FTNT': {
        'z_score_threshold': 1.5,
        'rsi_oversold': 35,
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'OUTSTANDING! Fortinet. RIVALS NVDA with +48.74% return. Cybersecurity leader.',
        'performance': 'Win rate: 77.8%, Return: +48.74%, Sharpe: 9.19, Avg gain: 7.24%'
    },

    # TIER A: HEALTHCARE (MID-CAP) - 76.9% win rate (EXP-026)
    'IDXX': {
        'z_score_threshold': 1.8,  # OPTIMIZED (was 1.5) - EXP-038
        'rsi_oversold': 38,  # OPTIMIZED (was 35) - EXP-038
        'volume_multiplier': 1.2,  # OPTIMIZED (was 1.3) - EXP-038
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'EXCELLENT! IDEXX Laboratories. OPTIMIZED params: +23.2pp (54.5%->77.8%)',
        'performance': 'Win rate: 76.9%, Return: +24.89%, Sharpe: 7.90, Avg gain: 3.46%'
    },

    # TIER A: MEDICAL DEVICES (MID-CAP) - 72.2% win rate (EXP-026)
    'DXCM': {
        'z_score_threshold': 1.5,
        'rsi_oversold': 35,
        'volume_multiplier': 1.8,  # OPTIMIZED (was 1.3) - EXP-038
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'EXCELLENT! DexCom. OPTIMIZED params: +11.7pp (45.5%->57.1%)',
        'performance': 'Win rate: 72.2%, Return: +35.12%, Sharpe: 9.43, Avg gain: 3.41%'
    },

    # TIER A: REIT - SELF-STORAGE (MID-CAP) - 85.7% win rate (EXP-027)
    'EXR': {
        'z_score_threshold': 1.2,  # OPTIMIZED (was 1.5) - EXP-038
        'rsi_oversold': 30,  # OPTIMIZED (was 35) - EXP-038
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'OUTSTANDING! Extra Space Storage. OPTIMIZED params: +9.5pp (57.1%->66.7%)',
        'performance': 'Win rate: 85.7%, Return: +18.95%, Sharpe: 12.48, Avg gain: 3.73%'
    },

    # TIER A: REIT - RESIDENTIAL (MID-CAP) - 77.8% win rate (EXP-027)
    'INVH': {
        'z_score_threshold': 1.5,
        'rsi_oversold': 35,
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'EXCELLENT! Invitation Homes. Residential REIT, strong mean reversion.',
        'performance': 'Win rate: 77.8%, Return: +15.04%, Sharpe: 11.85, Avg gain: 2.68%'
    },

    # ========================================================================
    # SMALL-CAP TIER A STOCKS (EXP-028)
    # ========================================================================

    # TIER A: CONSTRUCTION (SMALL-CAP) - 88.9% win rate (EXP-028)
    'ROAD': {
        'z_score_threshold': 1.2,  # OPTIMIZED (was 1.5) - EXP-037
        'rsi_oversold': 30,  # OPTIMIZED (was 35) - EXP-037
        'volume_multiplier': 1.5,  # OPTIMIZED (was 1.3) - EXP-037
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'OUTSTANDING! Construction Partners. OPTIMIZED params: +28.6pp (71.4%->100%!)',
        'performance': 'Win rate: 88.9%, Return: +18.15%, Sharpe: 17.06, Avg gain: 2.44%'
    },

    # TIER A: BIOTECH/RARE DISEASE (SMALL-CAP) - 72.7% win rate (EXP-028)
    'INSM': {
        'z_score_threshold': 1.2,  # OPTIMIZED (was 1.5) - EXP-038
        'rsi_oversold': 38,  # OPTIMIZED (was 35) - EXP-038
        'volume_multiplier': 1.5,  # OPTIMIZED (was 1.3) - EXP-038
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'BEST PERFORMER! Insmed. OPTIMIZED params: +11.7pp (58.3%->70.0%)',
        'performance': 'Win rate: 72.7%, Return: +142.03%, Sharpe: 5.51, Avg gain: 17.73%'
    },

    # ========================================================================
    # TIER B STOCKS (55-70% win rate) - NOT MONITORED (marginal performance)
    # Removed from active monitoring based on EXP-014 results
    # ========================================================================
    # 'CVX': 66.7% win rate, +6.99% return - Energy sector, acceptable but not top tier
    # 'HD': 62.5% win rate, +5.63% return - Retail, moderate performance
    # 'BAC': 60.0% win rate, -5.86% return - Finance, LOSING strategy despite 60% win rate
    # 'JPM': 55.6% win rate, +3.65% return - Finance, marginal performance

    # ========================================================================
    # TIER C STOCKS (<55% win rate) - AVOID (poor performance)
    # These stocks do NOT work well with mean reversion strategy
    # ========================================================================
    # 'AAPL': 50.0% win rate, +13.22% return - Coin flip, returns from luck not skill
    # 'GS': 50.0% win rate, -2.75% return - Finance, coin flip with losses
    # 'TSLA': 46.7% win rate, -29.84% return - WORST PERFORMER, extreme volatility
    # 'QQQ': 42.9% win rate, +1.15% return - ETF, poor mean reversion characteristics

    # Default fallback for unknown stocks
    # Updated from original universal parameters based on optimization learnings
    'DEFAULT': {
        'z_score_threshold': 1.5,
        'rsi_oversold': 35,
        'volume_multiplier': 1.3,  # Lowered from 1.5 (universal too conservative)
        'price_drop_threshold': -1.5,  # Changed from -2.0 (universal too strict)
        'notes': 'Updated default parameters. Use for moderate volatility stocks.',
        'performance': 'Average across stocks: Win rate ~70%, reasonable for medium volatility'
    }
}


def get_params(ticker):
    """
    Get optimized parameters for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g., 'NVDA', 'TSLA')

    Returns:
        Dictionary with mean reversion parameters

    Example:
        >>> params = get_params('TSLA')
        >>> detector = MeanReversionDetector(
        ...     z_score_threshold=params['z_score_threshold'],
        ...     rsi_oversold=params['rsi_oversold'],
        ...     volume_multiplier=params['volume_multiplier'],
        ...     price_drop_threshold=params['price_drop_threshold']
        ... )
    """
    return MEAN_REVERSION_PARAMS.get(ticker, MEAN_REVERSION_PARAMS['DEFAULT'])


def get_all_tickers():
    """
    Get list of all tickers with optimized parameters.

    Returns:
        List of ticker symbols (excluding DEFAULT)
    """
    return [ticker for ticker in MEAN_REVERSION_PARAMS.keys() if ticker != 'DEFAULT']


def print_params_summary():
    """Print summary of all optimized parameters."""
    print("=" * 70)
    print("STOCK-SPECIFIC MEAN REVERSION PARAMETERS")
    print("=" * 70)
    print()

    for ticker in get_all_tickers():
        params = MEAN_REVERSION_PARAMS[ticker]
        print(f"{ticker}:")
        print(f"  Z-score: {params['z_score_threshold']}")
        print(f"  RSI: {params['rsi_oversold']}")
        print(f"  Volume: {params['volume_multiplier']}x")
        print(f"  Drop: {params['price_drop_threshold']}%")
        print(f"  Performance: {params['performance']}")
        print(f"  Notes: {params['notes']}")
        print()

    print("DEFAULT (fallback):")
    default = MEAN_REVERSION_PARAMS['DEFAULT']
    print(f"  Z-score: {default['z_score_threshold']}")
    print(f"  RSI: {default['rsi_oversold']}")
    print(f"  Volume: {default['volume_multiplier']}x")
    print(f"  Drop: {default['price_drop_threshold']}%")
    print(f"  Notes: {default['notes']}")
    print()
    print("=" * 70)


# Guidelines for adding new stocks (updated based on EXP-014)
PARAMETER_SELECTION_GUIDELINES = """
ADDING NEW STOCKS - EXP-014 METHODOLOGY:

1. Run 3-year backtest using exp014_stock_selection_optimization.py:
   - Tests panic sell detection with time-decay exits
   - Measures: win rate, total return, Sharpe ratio, avg gain/loss
   - Compares against existing stocks

2. Tier classification (based on win rate):
   - Tier A (>70% win rate): TRADE AGGRESSIVELY
     * Add to MEAN_REVERSION_PARAMS
     * Enable daily monitoring
     * Expected: High returns, excellent risk-adjusted performance

   - Tier B (55-70% win rate): CONSIDER CAUTIOUSLY
     * Marginal performance
     * May add if diversification needed
     * Monitor closely for degradation

   - Tier C (<55% win rate): AVOID
     * Mean reversion doesn't work on these stocks
     * DO NOT trade, regardless of other metrics

3. Parameter selection:
   - Start with DEFAULT parameters (z=1.5, RSI=35, vol=1.3, drop=-1.5)
   - Run backtest to verify performance
   - Only optimize if Tier A candidate needs tuning

4. Performance requirements for Tier A:
   - Win rate: >70% (CRITICAL)
   - Total return: >5% over 3 years (minimum)
   - Sharpe ratio: >5.0 (preferred)
   - Avg gain: >2% per trade (minimum)

5. Production deployment:
   - Add to MEAN_REVERSION_PARAMS with tier='A'
   - Update performance notes with actual backtest results
   - Monitor monthly, re-evaluate quarterly
   - Remove if win rate drops below 65% for 6+ months

KEY LEARNINGS (EXP-014):
- Trading ONLY Tier A stocks improves returns by +15.99pp
- Win rate improvement of +17.23pp vs trading all stocks
- Quality over quantity: 3 excellent stocks > 11 mixed stocks
- Mean reversion is stock-specific, not universal

CURRENT TIER A STOCKS (v12.0 - 22 stocks):
LARGE-CAP (14 stocks):
Semiconductors (4):
- NVDA: 87.5% win rate, +49.70% return
- AVGO: 87.5% win rate, +24.52% return ⭐ EXP-021
- KLAC: 80.0% win rate, +20.05% return ⭐ EXP-024
- MRVL: 75.0% win rate, +26.63% return ⭐ EXP-024

Tech/Software (3):
- ORCL: 76.9% win rate, +22.50% return ⭐ EXP-024
- TXN: 80.0% win rate, +33.13% return ⭐ EXP-025
- INTU: 76.9% win rate, +29.81% return ⭐ EXP-025

Healthcare/Pharma (3):
- ABBV: 70.6% win rate, +10.50% return ⭐ EXP-024
- SYK: 70.0% win rate, +11.33% return ⭐ EXP-024
- GILD: 80.0% win rate, +8.22% return ⭐ EXP-025

Payments (2):
- V: 75.0% win rate, +7.33% return
- MA: 71.4% win rate, +5.99% return

Finance (1):
- AXP: 71.4% win rate, +29.21% return ⭐ EXP-021

Energy (1):
- EOG: 81.8% win rate, +17.98% return ⭐ EXP-025

MID-CAP (6 stocks):
Materials (1):
- MLM: 81.8% win rate, +25.71% return ⭐ EXP-026

Cybersecurity (1):
- FTNT: 77.8% win rate, +48.74% return ⭐ EXP-026

Healthcare/Medical Devices (2):
- IDXX: 76.9% win rate, +24.89% return ⭐ EXP-026
- DXCM: 72.2% win rate, +35.12% return ⭐ EXP-026

REITs (2):
- EXR: 85.7% win rate, +18.95% return ⭐ EXP-027
- INVH: 77.8% win rate, +15.04% return ⭐ EXP-027

SMALL-CAP (2 stocks):
Construction (1):
- ROAD: 88.9% win rate, +18.15% return ⭐ EXP-028 (matches NVDA!)

Biotech/Rare Disease (1):
- INSM: 72.7% win rate, +142.03% return ⭐ EXP-028 (🔥 BEST PERFORMER EVER!)

Portfolio stats (v12.0):
- Average win rate: 77.7%
- Average return: +28.84%
- Trade frequency: ~110 trades/year (+10% vs v11.0)
- Full sector diversification across 9 sectors
- INSM is new system-wide champion (+142% beats NVDA's +49%)

AVOID THESE STOCKS (confirmed poor performers):
EXP-014 tested:
- TSLA: 46.7% win rate, -29.84% return (WORST)
- QQQ: 42.9% win rate, +1.15% return
- GS: 50.0% win rate, -2.75% return
- AAPL: 50.0% win rate (coin flip)

EXP-021 tested:
- INTC: 38.9% win rate, -26.79% return (VERY POOR)
- COST: 28.6% win rate, -5.13% return (WORST)
- PYPL: 42.9% win rate, +2.83% return
- WFC: 46.2% win rate, -15.44% return

EXP-024 tested (poor performers):
- ADBE: 53.8% win rate, -11.32% return
- CRM: 50.0% win rate, +0.88% return (coin flip)
- NOW: 55.6% win rate, +6.35% return (marginal)
- SHOP: 46.2% win rate, -3.85% return
- NFLX: 50.0% win rate, -4.11% return (coin flip)
- ASML: 50.0% win rate, -0.66% return (coin flip)
- TSM: 66.7% win rate, -4.42% return (LOSING despite decent win rate)
- LRCX: 61.5% win rate, +3.96% return (marginal)
- AMAT: 58.3% win rate, +3.25% return (marginal)
- JPM: 50.0% win rate, +2.75% return (coin flip - retest confirmed)
- BLK: 60.0% win rate, -0.97% return (LOSING)
- SCHW: 55.6% win rate, +1.49% return (marginal)
- USB: 54.5% win rate, -5.11% return (LOSING)
- PNC: 50.0% win rate, -3.07% return (coin flip + losing)
- TFC: 46.2% win rate, -11.41% return (POOR)
- TMO: 50.0% win rate, +1.07% return (coin flip)
- DHR: 62.5% win rate, +4.90% return (marginal)
- ISRG: 60.0% win rate, +8.19% return (marginal)
- NKE: 44.4% win rate, -8.33% return (POOR)
- SBUX: 50.0% win rate, +2.22% return (coin flip)
- TGT: 46.2% win rate, -2.70% return (POOR)
- LOW: 100% win rate, +8.54% return (ONLY 3 TRADES - insufficient data)
- HON: 66.7% win rate, +6.69% return (marginal)
- CAT: 57.1% win rate, +10.24% return (marginal)

EXP-025 tested (poor performers):
Energy:
- CVX: 44.4% win rate, -1.18% return (retest confirmed poor)
- COP: 66.7% win rate, +19.47% return (marginal - below 70%)
- XOM: 66.7% win rate, +14.20% return (marginal)
- SLB: 61.5% win rate, +18.22% return (marginal)
- OXY: 52.9% win rate, +3.57% return (marginal)

Industrials:
- BA: 35.7% win rate, -5.32% return (VERY POOR)
- RTX: 28.6% win rate, -0.55% return (VERY POOR)
- DE: 38.5% win rate, -11.38% return (POOR)
- MMM: 58.3% win rate, +13.11% return (marginal)
- UNP: 40.0% win rate, +1.69% return (POOR)
- GE: 66.7% win rate, +4.88% return (marginal)

Consumer:
- MCD: 20.0% win rate, -6.31% return (WORST in EXP-025)
- DIS: 40.0% win rate, -1.29% return (POOR)
- KO: 40.0% win rate, -0.20% return (POOR)
- PEP: 50.0% win rate, -3.00% return (coin flip)
- CMCSA: 66.7% win rate, +10.79% return (marginal)
- HD: 62.5% win rate, +5.63% return (marginal - retest confirmed)

Tech:
- CDNS: 36.4% win rate, +1.34% return (POOR)
- PANW: 50.0% win rate, +4.69% return (coin flip)
- SNPS: 57.9% win rate, +37.32% return (marginal, high return but below 70% win rate)
- ADI: 64.7% win rate, +20.76% return (marginal)

Healthcare:
- MRK: 58.3% win rate, +2.51% return (marginal)
- AMGN: 53.3% win rate, +1.41% return (marginal)
- VRTX: 55.6% win rate, -0.99% return (LOSING)
- NVO: 61.1% win rate, +11.51% return (marginal)
- LLY: 62.5% win rate, +12.01% return (marginal)
"""


if __name__ == "__main__":
    print_params_summary()
    print()
    print(PARAMETER_SELECTION_GUIDELINES)

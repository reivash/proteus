"""
Stock-Specific Mean Reversion Parameters - v6.0

Optimized parameters for mean reversion strategy per stock.
Parameters tuned on 2022-2025 data with regime + earnings filters.

Last Updated: 2025-11-16
Optimization: EXP-014 (stock selection optimization)
Exit Strategy: EXP-010-EXIT (time-decay exits)

VERSION: 6.0
- Selective trading: Only Tier A stocks (>70% win rate)
- Stock selection based on 3-year backtests (EXP-014)
- Expected improvement: +15.99pp return, +17.23pp win rate vs trading all
- Time-decay exit strategy (Day 0: ±2%, Day 1: ±1.5%, Day 2+: ±1%)
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
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'BEST PERFORMER! GPU/AI leader. Exceptionally strong mean reversion characteristics.',
        'performance': 'Win rate: 87.5%, Return: +49.70%, Sharpe: 19.58, Avg gain: 6.35%'
    },

    # TIER A: PAYMENTS - 75.0% win rate
    'V': {
        'z_score_threshold': 1.5,
        'rsi_oversold': 35,
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5,
        'tier': 'A',
        'notes': 'EXCELLENT! Payments leader. Reliable mean reversion with strong consistency.',
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

CURRENT TIER A STOCKS:
- NVDA: 87.5% win rate, +49.70% return (BEST)
- V: 75.0% win rate, +7.33% return (EXCELLENT)
- MA: 71.4% win rate, +5.99% return (EXCELLENT)

AVOID THESE STOCKS (confirmed losers):
- TSLA: 46.7% win rate, -29.84% return
- QQQ: 42.9% win rate, +1.15% return
- GS: 50.0% win rate, -2.75% return
- AAPL: 50.0% win rate (coin flip)
"""


if __name__ == "__main__":
    print_params_summary()
    print()
    print(PARAMETER_SELECTION_GUIDELINES)

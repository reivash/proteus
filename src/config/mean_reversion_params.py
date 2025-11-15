"""
Stock-Specific Mean Reversion Parameters - v5.0

Optimized parameters for mean reversion strategy per stock.
Parameters tuned on 2022-2025 data with regime + earnings filters.

Last Updated: 2025-11-15
Optimization: EXP-008-PARAMS (stock parameters)
Exit Strategy: EXP-010-EXIT (time-decay exits)

VERSION: 5.0
- Time-decay exit strategy (Day 0: ±2%, Day 1: ±1.5%, Day 2+: ±1%)
- Performance improvement: +12.35pp portfolio return, +6.1pp win rate vs v4.0
"""

# Optimized parameters by stock ticker
MEAN_REVERSION_PARAMS = {
    'NVDA': {
        'z_score_threshold': 1.5,
        'rsi_oversold': 35,
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5,
        'notes': 'High volatility GPU/AI stock. Lower volume threshold catches more quality signals.',
        'performance': 'Win rate: 87.5%, Return: +45.75%, Sharpe: 12.22'
    },

    'TSLA': {
        'z_score_threshold': 1.75,  # TIGHTER than default
        'rsi_oversold': 32,  # TIGHTER than default
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5,
        'notes': 'EXTREME volatility. Tighter thresholds filter noise and catch only true panic sells.',
        'performance': 'Win rate: 87.5%, Return: +12.81%, Sharpe: 7.22'
    },

    'AAPL': {
        'z_score_threshold': 1.0,  # LOOSER than default
        'rsi_oversold': 35,
        'volume_multiplier': 1.7,
        'price_drop_threshold': -1.5,
        'notes': 'Low volatility mega-cap. Looser z-score catches signals. Performance still weak (60% win rate).',
        'performance': 'Win rate: 60.0%, Return: -0.25%, Sharpe: -0.13'
    },

    'AMZN': {
        'z_score_threshold': 1.0,  # LOOSER for catching signals
        'rsi_oversold': 32,  # Tight RSI threshold
        'volume_multiplier': 1.5,
        'price_drop_threshold': -1.5,
        'notes': 'EXCELLENT performer! Strong mean reversion with 80% win rate. Low z-score works well.',
        'performance': 'Win rate: 80.0%, Return: +10.20%, Sharpe: 4.06'
    },

    'MSFT': {
        'z_score_threshold': 1.75,  # Tighter threshold
        'rsi_oversold': 35,
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5,
        'notes': 'Moderate volatility mega-cap. Good performance with tighter z-score threshold.',
        'performance': 'Win rate: 66.7%, Return: +2.06%, Sharpe: 1.71'
    },

    # FINANCE SECTOR
    'JPM': {
        'z_score_threshold': 1.0,  # Looser for catching signals
        'rsi_oversold': 32,  # Tight RSI
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5,
        'notes': 'STAR PERFORMER! Finance sector. Excellent win rate and returns.',
        'performance': 'Win rate: 80.0%, Return: +18.63%, Sharpe: 8.19'
    },

    # HEALTHCARE SECTOR
    'JNJ': {
        'z_score_threshold': 1.0,  # Looser for catching signals
        'rsi_oversold': 32,  # Tight RSI
        'volume_multiplier': 1.3,
        'price_drop_threshold': -1.5,
        'notes': 'EXCELLENT! Healthcare sector. High win rate with outstanding Sharpe ratio.',
        'performance': 'Win rate: 83.3%, Return: +13.21%, Sharpe: 12.46'
    },

    'UNH': {
        'z_score_threshold': 1.5,  # Moderate threshold
        'rsi_oversold': 40,  # Looser RSI
        'volume_multiplier': 1.7,
        'price_drop_threshold': -3.0,  # Stricter drop requirement
        'notes': 'EXCELLENT! Healthcare sector. Very high win rate with solid returns.',
        'performance': 'Win rate: 85.7%, Return: +5.45%, Sharpe: 4.71'
    },

    # TECH - SEMICONDUCTOR
    'INTC': {
        'z_score_threshold': 1.75,  # Tighter threshold
        'rsi_oversold': 32,  # Tight RSI
        'volume_multiplier': 1.3,
        'price_drop_threshold': -2.5,  # Moderate drop requirement
        'notes': 'Good performer. Semiconductor sector adds volatility diversification.',
        'performance': 'Win rate: 66.7%, Return: +6.46%, Sharpe: 5.07'
    },

    # ENERGY SECTOR
    'CVX': {
        'z_score_threshold': 1.0,  # Looser for catching signals
        'rsi_oversold': 35,
        'volume_multiplier': 1.7,
        'price_drop_threshold': -1.5,
        'notes': 'Acceptable. Energy sector diversification. Marginal performance, monitor closely.',
        'performance': 'Win rate: 60.0%, Return: +2.50%, Sharpe: 2.68'
    },

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


# Guidelines for adding new stocks
PARAMETER_SELECTION_GUIDELINES = """
ADDING NEW STOCKS:

1. Measure historical volatility:
   - Calculate 30-day ATR / Price ratio
   - Calculate 30-day standard deviation of returns

2. Classify volatility:
   - High (>3% daily): Use TSLA parameters (z=1.75, RSI=32)
   - Medium (1.5-3% daily): Use NVDA parameters (z=1.5, RSI=35)
   - Low (<1.5% daily): Use AAPL parameters (z=1.0, RSI=35) or skip

3. Backtest with regime + earnings filters:
   - Test on 2-3 year period
   - Verify win rate > 60%
   - Check Sharpe ratio > 1.0

4. Fine-tune if needed:
   - Run parameter optimization (exp008_parameter_optimization.py)
   - Test 400 combinations
   - Select best by win rate, Sharpe, return

5. Monitor in production:
   - Track win rate, return, Sharpe
   - Re-optimize every 6-12 months
   - Remove if consistently underperforms

NOTES:
- Mean reversion works best on volatile stocks (NVDA, TSLA)
- Stable mega-caps (AAPL) may not be suitable
- Parameters should match stock's volatility profile
"""


if __name__ == "__main__":
    print_params_summary()
    print()
    print(PARAMETER_SELECTION_GUIDELINES)

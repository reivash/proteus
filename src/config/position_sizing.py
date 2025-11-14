"""
Dynamic Position Sizing Configuration

Position sizing based on stock performance tiers.
Higher performing stocks (higher win rate, better Sharpe) get larger positions.

RATIONALE:
- Maximize capital allocation to best performers
- Reduce risk exposure to marginal performers
- Better risk-adjusted portfolio returns

PERFORMANCE TIERS (based on backtested results 2022-2025):
- Tier 1 (Excellent): 80%+ win rate → Full position (1.0x)
- Tier 2 (Good): 65-75% win rate → Reduced position (0.75x)
- Tier 3 (Acceptable): 60-65% win rate → Minimal position (0.5x)

Last Updated: 2025-11-14
Based on: EXP-008 sector diversification results
"""

# Position sizing by stock ticker
POSITION_SIZES = {
    # TIER 1: EXCELLENT PERFORMERS (80%+ win rate, full position)
    'NVDA': {
        'position_size': 1.0,
        'tier': 1,
        'win_rate': 87.5,
        'return': 45.75,
        'sharpe': 12.22,
        'rationale': 'Top performer, 87.5% win rate, exceptional Sharpe ratio'
    },

    'TSLA': {
        'position_size': 1.0,
        'tier': 1,
        'win_rate': 87.5,
        'return': 12.81,
        'sharpe': 7.22,
        'rationale': 'Top performer, 87.5% win rate, strong risk-adjusted returns'
    },

    'UNH': {
        'position_size': 1.0,
        'tier': 1,
        'win_rate': 85.7,
        'return': 5.45,
        'sharpe': 4.71,
        'rationale': 'Highest win rate (85.7%), healthcare defensive'
    },

    'JNJ': {
        'position_size': 1.0,
        'tier': 1,
        'win_rate': 83.3,
        'return': 13.21,
        'sharpe': 12.46,
        'rationale': 'Excellent win rate, highest Sharpe ratio (12.46)'
    },

    'AMZN': {
        'position_size': 1.0,
        'tier': 1,
        'win_rate': 80.0,
        'return': 10.20,
        'sharpe': 4.06,
        'rationale': 'Solid 80% win rate, consistent performer'
    },

    'JPM': {
        'position_size': 1.0,
        'tier': 1,
        'win_rate': 80.0,
        'return': 18.63,
        'sharpe': 8.19,
        'rationale': 'Star discovery, highest return (18.63%), 80% win rate'
    },

    # TIER 2: GOOD PERFORMERS (65-75% win rate, reduced position)
    'MSFT': {
        'position_size': 0.75,
        'tier': 2,
        'win_rate': 66.7,
        'return': 2.06,
        'sharpe': 1.71,
        'rationale': 'Good win rate, moderate returns, reduced position'
    },

    'INTC': {
        'position_size': 0.75,
        'tier': 2,
        'win_rate': 66.7,
        'return': 6.46,
        'sharpe': 5.07,
        'rationale': 'Good win rate, semiconductor diversification'
    },

    # TIER 3: ACCEPTABLE PERFORMERS (60-65% win rate, minimal position)
    'AAPL': {
        'position_size': 0.5,
        'tier': 3,
        'win_rate': 60.0,
        'return': -0.25,
        'sharpe': -0.13,
        'rationale': 'Marginal performance, negative return, minimal position'
    },

    'CVX': {
        'position_size': 0.5,
        'tier': 3,
        'win_rate': 60.0,
        'return': 2.50,
        'sharpe': 2.68,
        'rationale': 'Marginal performance, energy diversification, minimal position'
    },

    # DEFAULT (for unknown stocks)
    'DEFAULT': {
        'position_size': 0.75,
        'tier': 2,
        'win_rate': None,
        'return': None,
        'sharpe': None,
        'rationale': 'Default position for untested stocks (medium risk)'
    }
}


def get_position_size(ticker):
    """
    Get position size for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Position size multiplier (0.5 to 1.0)
    """
    return POSITION_SIZES.get(ticker, POSITION_SIZES['DEFAULT'])['position_size']


def get_tier(ticker):
    """
    Get performance tier for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Performance tier (1, 2, or 3)
    """
    return POSITION_SIZES.get(ticker, POSITION_SIZES['DEFAULT'])['tier']


def get_position_info(ticker):
    """
    Get full position sizing information for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with position sizing details
    """
    return POSITION_SIZES.get(ticker, POSITION_SIZES['DEFAULT'])


def get_all_tickers_by_tier():
    """
    Get all tickers grouped by performance tier.

    Returns:
        Dictionary with tiers as keys, lists of tickers as values
    """
    tiers = {1: [], 2: [], 3: []}

    for ticker, info in POSITION_SIZES.items():
        if ticker != 'DEFAULT':
            tiers[info['tier']].append(ticker)

    return tiers


def print_position_sizing_summary():
    """Print summary of position sizing configuration."""
    print("=" * 70)
    print("DYNAMIC POSITION SIZING CONFIGURATION")
    print("=" * 70)
    print()

    # Group by tier
    tiers = get_all_tickers_by_tier()

    for tier_num in [1, 2, 3]:
        tier_name = {1: 'TIER 1: EXCELLENT', 2: 'TIER 2: GOOD', 3: 'TIER 3: ACCEPTABLE'}[tier_num]
        print(f"{tier_name}")
        print("-" * 70)

        for ticker in tiers[tier_num]:
            info = POSITION_SIZES[ticker]
            print(f"{ticker}:")
            print(f"  Position size: {info['position_size']:.2f}x")
            print(f"  Win rate: {info['win_rate']}%")
            print(f"  Return: {info['return']:+.2f}%")
            print(f"  Sharpe: {info['sharpe']:.2f}")
            print(f"  Rationale: {info['rationale']}")
            print()

    print("=" * 70)

    # Summary statistics
    tier1_count = len(tiers[1])
    tier2_count = len(tiers[2])
    tier3_count = len(tiers[3])

    print("\nSUMMARY:")
    print(f"  Tier 1 (1.0x position): {tier1_count} stocks")
    print(f"  Tier 2 (0.75x position): {tier2_count} stocks")
    print(f"  Tier 3 (0.5x position): {tier3_count} stocks")
    print()
    print("EXPECTED IMPACT:")
    print("  - Maximize capital allocation to best performers (Tier 1)")
    print("  - Reduce risk exposure to marginal performers (Tier 3)")
    print("  - Better risk-adjusted portfolio returns")
    print()


# Position sizing strategy guidelines
POSITION_SIZING_GUIDELINES = """
DYNAMIC POSITION SIZING STRATEGY:

TIER DEFINITIONS:
1. Tier 1 (Excellent): 80%+ win rate, 1.0x position
   - Best performers with proven track record
   - Full capital allocation
   - Examples: NVDA, TSLA, JPM, JNJ, UNH, AMZN

2. Tier 2 (Good): 65-75% win rate, 0.75x position
   - Solid performers with acceptable returns
   - Reduced position to manage risk
   - Examples: MSFT, INTC

3. Tier 3 (Acceptable): 60-65% win rate, 0.5x position
   - Marginal performers, barely meet criteria
   - Minimal position, monitor for removal
   - Examples: AAPL, CVX

POSITION SIZING CALCULATION:
- Base capital: $10,000
- Tier 1: $10,000 * 1.0 = $10,000 per trade
- Tier 2: $10,000 * 0.75 = $7,500 per trade
- Tier 3: $10,000 * 0.5 = $5,000 per trade

PORTFOLIO ALLOCATION EXAMPLE:
If signals appear simultaneously:
- NVDA signal: Allocate $10,000 (Tier 1)
- MSFT signal: Allocate $7,500 (Tier 2)
- AAPL signal: Allocate $5,000 (Tier 3)
- Total capital used: $22,500 for 3 trades

BENEFITS:
- Concentrates capital in best performers
- Reduces exposure to marginal stocks
- Improves portfolio Sharpe ratio
- Natural risk management

RE-BALANCING:
- Review tiers every 6-12 months
- Promote stocks with improving metrics
- Demote stocks with declining performance
- Remove stocks consistently underperforming

TIER UPGRADE CRITERIA:
- Tier 3 → Tier 2: Win rate reaches 65%+
- Tier 2 → Tier 1: Win rate reaches 80%+

TIER DOWNGRADE CRITERIA:
- Tier 1 → Tier 2: Win rate drops below 75%
- Tier 2 → Tier 3: Win rate drops below 60%
- Tier 3 → REMOVE: Win rate drops below 55% or negative Sharpe
"""


if __name__ == "__main__":
    print_position_sizing_summary()
    print()
    print(POSITION_SIZING_GUIDELINES)

"""
Sector Correlation Check

Prevents overconcentration in correlated positions.
If multiple signals are from the same sector, only the strongest passes.

This reduces portfolio correlation and improves risk-adjusted returns.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass

# Sector classifications for tracked stocks
SECTOR_MAP = {
    # Technology - Semiconductors
    'NVDA': 'SEMI',
    'AVGO': 'SEMI',
    'AMAT': 'SEMI',
    'KLAC': 'SEMI',
    'MRVL': 'SEMI',
    'ADI': 'SEMI',
    'TXN': 'SEMI',
    'QCOM': 'SEMI',

    # Technology - Software
    'MSFT': 'SOFTWARE',
    'ORCL': 'SOFTWARE',
    'CRM': 'SOFTWARE',
    'ADBE': 'SOFTWARE',
    'NOW': 'SOFTWARE',
    'INTU': 'SOFTWARE',

    # Technology - Internet/Social
    'META': 'INTERNET',

    # Financials - Banks
    'JPM': 'BANKS',
    'MS': 'BANKS',
    'SCHW': 'BANKS',
    'PNC': 'BANKS',
    'USB': 'BANKS',

    # Financials - Cards/Payments
    'V': 'PAYMENTS',
    'MA': 'PAYMENTS',
    'AXP': 'PAYMENTS',

    # Financials - Insurance
    'AIG': 'INSURANCE',

    # Healthcare - Pharma
    'JNJ': 'PHARMA',
    'PFE': 'PHARMA',
    'ABBV': 'PHARMA',
    'GILD': 'PHARMA',

    # Healthcare - Medical Devices
    'SYK': 'MEDICAL_DEVICES',
    'IDXX': 'MEDICAL_DEVICES',

    # Healthcare - Providers
    'HCA': 'HEALTHCARE_PROVIDERS',
    'CVS': 'HEALTHCARE_PROVIDERS',

    # Healthcare - Biotech
    'INSM': 'BIOTECH',

    # Consumer - Retail
    'WMT': 'RETAIL',
    'HD': 'RETAIL',
    'LOW': 'RETAIL',
    'TGT': 'RETAIL',

    # Consumer - Telecom
    'TMUS': 'TELECOM',
    'CMCSA': 'TELECOM',

    # Energy - Oil/Gas
    'XOM': 'OIL_GAS',
    'EOG': 'OIL_GAS',
    'COP': 'OIL_GAS',
    'SLB': 'OIL_GAS',
    'MPC': 'OIL_GAS',

    # Industrials
    'CAT': 'INDUSTRIALS',
    'ETN': 'INDUSTRIALS',
    'LMT': 'DEFENSE',

    # Materials
    'APD': 'CHEMICALS',
    'ECL': 'CHEMICALS',
    'SHW': 'CHEMICALS',
    'MLM': 'MATERIALS',

    # REITs
    'EXR': 'REIT',

    # Utilities
    'NEE': 'UTILITIES',

    # Construction
    'ROAD': 'CONSTRUCTION',
}

# Highly correlated sector pairs (trade together)
CORRELATED_SECTORS = [
    ('SEMI', 'SOFTWARE'),  # Tech moves together
    ('BANKS', 'PAYMENTS'),  # Financials correlated
    ('OIL_GAS', 'CHEMICALS'),  # Energy/commodity linked
]


@dataclass
class CorrelationResult:
    """Result of correlation filtering."""
    passed_signals: List[Dict]
    filtered_signals: List[Dict]
    sector_counts: Dict[str, int]
    warnings: List[str]


def get_sector(ticker: str) -> str:
    """Get sector for a ticker."""
    return SECTOR_MAP.get(ticker, 'OTHER')


def are_sectors_correlated(sector1: str, sector2: str) -> bool:
    """Check if two sectors are highly correlated."""
    if sector1 == sector2:
        return True

    for s1, s2 in CORRELATED_SECTORS:
        if (sector1 == s1 and sector2 == s2) or (sector1 == s2 and sector2 == s1):
            return True

    return False


def filter_correlated_signals(
    signals: List[Dict],
    max_per_sector: int = 2,
    max_correlated_total: int = 3,
    strength_key: str = 'adjusted_strength'
) -> CorrelationResult:
    """
    Filter signals to reduce sector concentration.

    Rules:
    1. Maximum `max_per_sector` signals from same sector
    2. Maximum `max_correlated_total` signals from correlated sectors combined
    3. Always keep the strongest signal in each sector

    Args:
        signals: List of signal dicts (must have 'ticker' and strength key)
        max_per_sector: Max signals per sector
        max_correlated_total: Max from correlated sector groups
        strength_key: Key for signal strength in dict

    Returns:
        CorrelationResult with passed and filtered signals
    """
    # Sort by strength descending
    sorted_signals = sorted(
        signals,
        key=lambda x: x.get(strength_key, 0),
        reverse=True
    )

    passed = []
    filtered = []
    sector_counts: Dict[str, int] = {}
    correlated_group_counts: Dict[Tuple[str, str], int] = {}
    warnings = []

    for signal in sorted_signals:
        ticker = signal.get('ticker')
        strength = signal.get(strength_key, 0)
        sector = get_sector(ticker)

        # Check sector limit
        current_sector_count = sector_counts.get(sector, 0)
        if current_sector_count >= max_per_sector:
            filtered.append(signal)
            warnings.append(
                f"{ticker} filtered: {sector} sector already has {current_sector_count} position(s)"
            )
            continue

        # Check correlated sector limit
        skip = False
        for s1, s2 in CORRELATED_SECTORS:
            if sector in (s1, s2):
                group_key = (min(s1, s2), max(s1, s2))
                current_group = correlated_group_counts.get(group_key, 0)
                if current_group >= max_correlated_total:
                    filtered.append(signal)
                    warnings.append(
                        f"{ticker} filtered: correlated group ({s1}/{s2}) has {current_group} position(s)"
                    )
                    skip = True
                    break

        if skip:
            continue

        # Signal passes - update counts
        passed.append(signal)
        sector_counts[sector] = current_sector_count + 1

        # Update correlated group counts
        for s1, s2 in CORRELATED_SECTORS:
            if sector in (s1, s2):
                group_key = (min(s1, s2), max(s1, s2))
                correlated_group_counts[group_key] = correlated_group_counts.get(group_key, 0) + 1

    return CorrelationResult(
        passed_signals=passed,
        filtered_signals=filtered,
        sector_counts=sector_counts,
        warnings=warnings
    )


def print_sector_analysis(signals: List[Dict], strength_key: str = 'adjusted_strength'):
    """Print sector breakdown of signals."""
    print("=" * 70)
    print("SECTOR ANALYSIS")
    print("=" * 70)

    # Group by sector
    by_sector: Dict[str, List[Dict]] = {}
    for sig in signals:
        sector = get_sector(sig.get('ticker'))
        if sector not in by_sector:
            by_sector[sector] = []
        by_sector[sector].append(sig)

    # Print by sector
    for sector in sorted(by_sector.keys()):
        sigs = by_sector[sector]
        print(f"\n{sector}:")
        for sig in sorted(sigs, key=lambda x: x.get(strength_key, 0), reverse=True):
            print(f"  {sig.get('ticker')}: {sig.get(strength_key, 0):.1f}")

    print()


if __name__ == "__main__":
    # Test with sample signals
    test_signals = [
        {'ticker': 'NVDA', 'adjusted_strength': 85},
        {'ticker': 'AVGO', 'adjusted_strength': 78},
        {'ticker': 'AMAT', 'adjusted_strength': 72},
        {'ticker': 'MSFT', 'adjusted_strength': 75},
        {'ticker': 'JPM', 'adjusted_strength': 68},
        {'ticker': 'V', 'adjusted_strength': 65},
        {'ticker': 'MA', 'adjusted_strength': 64},
        {'ticker': 'XOM', 'adjusted_strength': 60},
    ]

    print("Input signals:")
    for sig in test_signals:
        print(f"  {sig['ticker']}: {sig['adjusted_strength']}")

    result = filter_correlated_signals(test_signals, max_per_sector=2)

    print("\nPassed signals:")
    for sig in result.passed_signals:
        print(f"  {sig['ticker']}: {sig['adjusted_strength']} ({get_sector(sig['ticker'])})")

    print("\nFiltered signals:")
    for sig in result.filtered_signals:
        print(f"  {sig['ticker']}: {sig['adjusted_strength']} ({get_sector(sig['ticker'])})")

    print("\nWarnings:")
    for w in result.warnings:
        print(f"  {w}")

"""
Sector Momentum Calculator

Fetches sector ETF data and calculates momentum for signal adjustment.
Based on research finding: weak sectors outperform strong sectors by +0.27% per trade.
"""

import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SectorMomentum:
    """Sector momentum data."""
    etf: str
    sector_name: str
    momentum_5d: float
    momentum_category: str
    strength_boost: float


class SectorMomentumCalculator:
    """
    Calculates sector momentum for signal adjustments.

    Based on 1-year analysis:
    - Weak sectors (-3% or lower): 56% win, +0.62% avg return -> 1.10x boost
    - Slightly weak (-1% to -3%): 57.6% win, +0.24% avg -> 1.05x boost
    - Neutral (-1% to +1%): 50.8% win, +0.18% avg -> 1.0x (baseline)
    - Slightly strong (+1% to +3%): 54.5% win, +0.16% avg -> 0.95x penalty
    - Strong (+3% or higher): 49.7% win, +0.21% avg -> 0.90x penalty
    """

    SECTOR_ETFS = ['XLF', 'XLK', 'XLV', 'XLE', 'XLI', 'XLY', 'XLC', 'XLB']

    SECTOR_NAMES = {
        'XLF': 'Financials',
        'XLK': 'Technology',
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLI': 'Industrials',
        'XLY': 'Consumer',
        'XLC': 'Communication',
        'XLB': 'Materials'
    }

    STOCK_SECTOR = {
        # Financials
        'JPM': 'XLF', 'MS': 'XLF', 'SCHW': 'XLF', 'AXP': 'XLF', 'AIG': 'XLF',
        'USB': 'XLF', 'PNC': 'XLF', 'V': 'XLF', 'MA': 'XLF',

        # Technology
        'NVDA': 'XLK', 'AVGO': 'XLK', 'MSFT': 'XLK', 'ORCL': 'XLK', 'INTU': 'XLK',
        'ADBE': 'XLK', 'CRM': 'XLK', 'NOW': 'XLK', 'AMAT': 'XLK', 'KLAC': 'XLK',
        'MRVL': 'XLK', 'QCOM': 'XLK', 'TXN': 'XLK', 'ADI': 'XLK',

        # Healthcare
        'ABBV': 'XLV', 'SYK': 'XLV', 'GILD': 'XLV', 'PFE': 'XLV', 'JNJ': 'XLV',
        'CVS': 'XLV', 'HCA': 'XLV', 'IDXX': 'XLV', 'INSM': 'XLV',

        # Energy
        'COP': 'XLE', 'SLB': 'XLE', 'XOM': 'XLE', 'MPC': 'XLE', 'EOG': 'XLE',

        # Industrials
        'CAT': 'XLI', 'ETN': 'XLI', 'LMT': 'XLI', 'ROAD': 'XLI',

        # Consumer
        'HD': 'XLY', 'LOW': 'XLY', 'TGT': 'XLY', 'WMT': 'XLY',

        # Communication
        'TMUS': 'XLC', 'CMCSA': 'XLC', 'META': 'XLC',

        # Materials
        'APD': 'XLB', 'ECL': 'XLB', 'MLM': 'XLB', 'SHW': 'XLB',
    }

    def __init__(self):
        self._sector_data: Dict[str, float] = {}
        self._last_fetch: Optional[datetime] = None
        self._cache_duration = timedelta(hours=1)  # Cache for 1 hour

    def _fetch_sector_data(self) -> Dict[str, float]:
        """Fetch 5-day momentum for all sector ETFs."""
        momentum_data = {}

        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)  # Extra buffer

        for etf in self.SECTOR_ETFS:
            try:
                ticker = yf.Ticker(etf)
                df = ticker.history(start=start_date, end=end_date)

                if len(df) >= 6:  # Need 6 days for 5-day momentum
                    momentum = ((df['Close'].iloc[-1] / df['Close'].iloc[-6]) - 1) * 100
                    momentum_data[etf] = round(momentum, 2)
                else:
                    momentum_data[etf] = 0.0
            except Exception as e:
                momentum_data[etf] = 0.0

        return momentum_data

    def get_sector_momentum(self, force_refresh: bool = False) -> Dict[str, float]:
        """
        Get current sector momentum data.
        Caches data to avoid excessive API calls.
        """
        now = datetime.now()

        if (force_refresh or
            self._last_fetch is None or
            (now - self._last_fetch) > self._cache_duration):
            self._sector_data = self._fetch_sector_data()
            self._last_fetch = now

        return self._sector_data

    def get_stock_sector_etf(self, ticker: str) -> Optional[str]:
        """Get the sector ETF for a stock."""
        return self.STOCK_SECTOR.get(ticker)

    def get_momentum_category(self, momentum: float) -> str:
        """Categorize momentum value."""
        if momentum < -3:
            return 'weak'
        elif momentum < -1:
            return 'slightly_weak'
        elif momentum < 1:
            return 'neutral'
        elif momentum < 3:
            return 'slightly_strong'
        else:
            return 'strong'

    def get_strength_boost(self, momentum: float) -> float:
        """
        Get signal strength multiplier based on sector momentum.

        Weak sectors get a boost (better mean reversion).
        Strong sectors get a penalty.
        """
        if momentum < -3:
            return 1.10  # Weak: +10% boost
        elif momentum < -1:
            return 1.05  # Slightly weak: +5% boost
        elif momentum < 1:
            return 1.0   # Neutral: no adjustment
        elif momentum < 3:
            return 0.95  # Slightly strong: -5% penalty
        else:
            return 0.90  # Strong: -10% penalty

    def get_stock_momentum_adjustment(self, ticker: str) -> SectorMomentum:
        """
        Get sector momentum adjustment for a specific stock.

        Returns:
            SectorMomentum with ETF, momentum, category, and boost multiplier
        """
        sector_etf = self.get_stock_sector_etf(ticker)

        if not sector_etf:
            # Stock not in known sectors - return neutral
            return SectorMomentum(
                etf='UNKNOWN',
                sector_name='Unknown',
                momentum_5d=0.0,
                momentum_category='neutral',
                strength_boost=1.0
            )

        momentum_data = self.get_sector_momentum()
        momentum = momentum_data.get(sector_etf, 0.0)
        category = self.get_momentum_category(momentum)
        boost = self.get_strength_boost(momentum)

        return SectorMomentum(
            etf=sector_etf,
            sector_name=self.SECTOR_NAMES.get(sector_etf, sector_etf),
            momentum_5d=momentum,
            momentum_category=category,
            strength_boost=boost
        )

    def is_avoid_combo(self, ticker: str, momentum: float) -> Tuple[bool, str]:
        """
        Check if stock + momentum combo should be avoided.

        Based on analysis, Consumer sector consistently underperforms.
        """
        sector_etf = self.get_stock_sector_etf(ticker)

        if sector_etf == 'XLY':  # Consumer
            return True, "Consumer sector consistently underperforms (-0.21% avg)"

        sector_name = self.SECTOR_NAMES.get(sector_etf, '')
        category = self.get_momentum_category(momentum)

        # Industrials + slightly_strong is bad (-0.92% avg)
        if sector_etf == 'XLI' and category == 'slightly_strong':
            return True, "Industrials + slightly_strong combo underperforms"

        # Communication + strong is bad (-0.65% avg)
        if sector_etf == 'XLC' and category == 'strong':
            return True, "Communication + strong combo underperforms"

        return False, ""

    def print_sector_summary(self):
        """Print current sector momentum summary."""
        momentum_data = self.get_sector_momentum()

        print("\nSECTOR MOMENTUM (5-day)")
        print("-" * 50)

        # Sort by momentum
        sorted_sectors = sorted(momentum_data.items(), key=lambda x: x[1])

        for etf, momentum in sorted_sectors:
            category = self.get_momentum_category(momentum)
            boost = self.get_strength_boost(momentum)
            name = self.SECTOR_NAMES.get(etf, etf)

            boost_str = f"+{(boost-1)*100:.0f}%" if boost > 1 else f"{(boost-1)*100:.0f}%"
            print(f"  {name:15} ({etf}): {momentum:+.1f}% [{category:14}] -> {boost_str}")


# Singleton instance
_calculator = None


def get_sector_momentum_calculator() -> SectorMomentumCalculator:
    """Get singleton calculator instance."""
    global _calculator
    if _calculator is None:
        _calculator = SectorMomentumCalculator()
    return _calculator


if __name__ == '__main__':
    calc = SectorMomentumCalculator()
    calc.print_sector_summary()

    print("\n\nSTOCK ADJUSTMENTS:")
    for ticker in ['NVDA', 'JPM', 'CVS', 'HD', 'COP']:
        adj = calc.get_stock_momentum_adjustment(ticker)
        avoid, reason = calc.is_avoid_combo(ticker, adj.momentum_5d)

        print(f"\n{ticker}:")
        print(f"  Sector: {adj.sector_name} ({adj.etf})")
        print(f"  Momentum: {adj.momentum_5d:+.1f}% ({adj.momentum_category})")
        print(f"  Strength boost: {adj.strength_boost:.2f}x")
        if avoid:
            print(f"  WARNING: {reason}")

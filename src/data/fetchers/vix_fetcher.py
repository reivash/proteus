"""
VIX (Volatility Index) Data Fetcher

Retrieves VIX data to identify extreme market fear conditions.

PROBLEM:
During extreme volatility (VIX > 30), market structure changes:
- Panic selling becomes contagious
- Mean reversion patterns break down
- Downside momentum overwhelms technical signals
- Even "overcorrections" continue falling

SOLUTION:
Disable mean reversion trading when VIX > 30 (extreme fear threshold).

HISTORICAL VIX LEVELS:
- Normal: 10-20
- Elevated: 20-30
- Extreme: 30+ (disable trading)
- Crisis: 40+ (COVID crash hit 85+)
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional


class VIXFetcher:
    """
    Fetch VIX (Volatility Index) data for risk filtering.
    """

    def __init__(self, extreme_threshold=30):
        """
        Initialize VIX fetcher.

        Args:
            extreme_threshold: VIX level above which to disable trading (default: 30)
        """
        self.extreme_threshold = extreme_threshold
        self._vix_cache = None
        self._cache_date = None

    def fetch_vix_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch VIX historical data.

        Args:
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            DataFrame with VIX data and Date index
        """
        try:
            vix = yf.Ticker("^VIX")

            if start_date and end_date:
                data = vix.history(start=start_date, end=end_date)
            elif start_date:
                data = vix.history(start=start_date)
            else:
                # Default: last 5 years
                data = vix.history(period="5y")

            if data.empty:
                print("[WARN] No VIX data available")
                return pd.DataFrame()

            # Simplify to just Close and Date
            vix_data = pd.DataFrame({
                'Date': data.index,
                'VIX': data['Close']
            })

            # Remove timezone for consistency
            vix_data['Date'] = pd.to_datetime(vix_data['Date']).dt.tz_localize(None)
            vix_data = vix_data.reset_index(drop=True)

            return vix_data

        except Exception as e:
            print(f"[WARN] Could not fetch VIX data: {e}")
            return pd.DataFrame()

    def get_vix_level(self, date: pd.Timestamp, vix_data: pd.DataFrame = None) -> Optional[float]:
        """
        Get VIX level for a specific date.

        Args:
            date: Date to check
            vix_data: Pre-fetched VIX data (optional, will fetch if not provided)

        Returns:
            VIX level (float) or None if not available
        """
        if vix_data is None or vix_data.empty:
            # Fetch recent VIX data
            end_date = date + timedelta(days=1)
            start_date = date - timedelta(days=30)
            vix_data = self.fetch_vix_data(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

        if vix_data.empty:
            return None

        # Normalize date
        date_normalized = pd.Timestamp(date.date())

        # Find closest date
        vix_data['Date'] = pd.to_datetime(vix_data['Date']).dt.tz_localize(None)
        vix_data['Date_normalized'] = pd.to_datetime(vix_data['Date'].dt.date)

        matching = vix_data[vix_data['Date_normalized'] == date_normalized]

        if not matching.empty:
            return float(matching.iloc[0]['VIX'])

        # If no exact match, find closest date within 5 days
        vix_data['date_diff'] = abs((vix_data['Date_normalized'] - date_normalized).dt.days)
        closest = vix_data[vix_data['date_diff'] <= 5].sort_values('date_diff')

        if not closest.empty:
            return float(closest.iloc[0]['VIX'])

        return None

    def is_extreme_fear(self, date: pd.Timestamp, vix_data: pd.DataFrame = None) -> bool:
        """
        Check if date is in extreme fear period (VIX > threshold).

        Args:
            date: Date to check
            vix_data: Pre-fetched VIX data (optional)

        Returns:
            True if extreme fear (VIX > threshold), False otherwise
        """
        vix_level = self.get_vix_level(date, vix_data)

        if vix_level is None:
            # If VIX data unavailable, assume safe (don't filter)
            return False

        return vix_level > self.extreme_threshold

    def add_vix_filter_to_signals(
        self,
        df: pd.DataFrame,
        signal_column: str = 'panic_sell'
    ) -> pd.DataFrame:
        """
        Filter out signals during extreme VIX periods.

        Args:
            df: DataFrame with trading signals and 'Date' column
            signal_column: Name of signal column to filter

        Returns:
            DataFrame with filtered signals
        """
        data = df.copy()

        if 'Date' not in data.columns:
            print("[WARN] No 'Date' column found, cannot apply VIX filter")
            return data

        # Fetch VIX data for entire period
        start_date = data['Date'].min()
        end_date = data['Date'].max()

        vix_data = self.fetch_vix_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        if vix_data.empty:
            print("[WARN] No VIX data available, skipping VIX filter")
            return data

        # Add VIX column to data
        data['VIX'] = data['Date'].apply(lambda d: self.get_vix_level(d, vix_data))

        # Store original signals
        if signal_column in data.columns:
            data[f'{signal_column}_before_vix_filter'] = data[signal_column].copy()

            # Disable signals when VIX > threshold
            extreme_fear_mask = data['VIX'] > self.extreme_threshold
            data.loc[extreme_fear_mask, signal_column] = 0

        return data

    def analyze_vix_periods(self, start_date: str, end_date: str) -> dict:
        """
        Analyze VIX periods for statistics.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with VIX statistics
        """
        vix_data = self.fetch_vix_data(start_date=start_date, end_date=end_date)

        if vix_data.empty:
            return {
                'data_available': False
            }

        total_days = len(vix_data)
        extreme_days = (vix_data['VIX'] > self.extreme_threshold).sum()
        elevated_days = ((vix_data['VIX'] > 20) & (vix_data['VIX'] <= self.extreme_threshold)).sum()
        normal_days = (vix_data['VIX'] <= 20).sum()

        return {
            'data_available': True,
            'total_days': total_days,
            'extreme_days': int(extreme_days),
            'elevated_days': int(elevated_days),
            'normal_days': int(normal_days),
            'extreme_pct': float(extreme_days / total_days * 100),
            'elevated_pct': float(elevated_days / total_days * 100),
            'normal_pct': float(normal_days / total_days * 100),
            'avg_vix': float(vix_data['VIX'].mean()),
            'max_vix': float(vix_data['VIX'].max()),
            'min_vix': float(vix_data['VIX'].min()),
            'threshold': self.extreme_threshold
        }


# Convenience function for quick filtering
def filter_signals_by_vix(
    df: pd.DataFrame,
    signal_column: str = 'panic_sell',
    extreme_threshold: int = 30
) -> pd.DataFrame:
    """
    Quick function to filter signals by VIX.

    Args:
        df: DataFrame with signals
        signal_column: Signal column name
        extreme_threshold: VIX threshold for extreme fear

    Returns:
        Filtered DataFrame
    """
    fetcher = VIXFetcher(extreme_threshold=extreme_threshold)
    return fetcher.add_vix_filter_to_signals(df, signal_column)


if __name__ == "__main__":
    print("VIX Fetcher Module")
    print()
    print("Purpose: Disable trading during extreme market fear (VIX > 30)")
    print("Threshold: VIX > 30 = extreme fear, disable mean reversion")
    print()
    print("Usage:")
    print("  fetcher = VIXFetcher(extreme_threshold=30)")
    print("  filtered_data = fetcher.add_vix_filter_to_signals(data)")
    print()
    print("Historical VIX peaks:")
    print("  - 2008 Financial Crisis: VIX 80+")
    print("  - 2020 COVID Crash: VIX 85+")
    print("  - 2022 Bear Market: VIX 30-35")

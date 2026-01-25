"""
FOMC (Federal Open Market Committee) Meeting Calendar Fetcher

Retrieves FOMC meeting dates to filter trading signals.

PROBLEM:
During FOMC meetings, market moves are driven by:
- Interest rate decisions
- Fed policy changes (QE/QT)
- Forward guidance
- Fed chair commentary

These moves are FUNDAMENTAL (policy-driven), not technical overcorrections.
Mean reversion signals near FOMC meetings often fail because:
- Panic sells may be justified reactions to hawkish policy
- Rallies may be relief from dovish policy
- Volatility overwhelms technical patterns

SOLUTION:
Disable mean reversion trading around FOMC meetings (exclusion window).

FOMC MEETING SCHEDULE:
- 8 meetings per year (roughly every 6 weeks)
- Scheduled in advance on Fed calendar
- Major market-moving events
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Set, List


class FOMCCalendarFetcher:
    """
    Fetch FOMC meeting dates for signal filtering.
    """

    # FOMC meeting dates 2020-2025
    # Source: Federal Reserve official calendar
    # https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    FOMC_DATES = [
        # 2020
        '2020-01-29', '2020-03-03', '2020-03-15',  # Emergency cut
        '2020-04-29', '2020-06-10', '2020-07-29',
        '2020-09-16', '2020-11-05', '2020-12-16',

        # 2021
        '2021-01-27', '2021-03-17', '2021-04-28',
        '2021-06-16', '2021-07-28', '2021-09-22',
        '2021-11-03', '2021-12-15',

        # 2022
        '2022-01-26', '2022-03-16', '2022-05-04',
        '2022-06-15', '2022-07-27', '2022-09-21',
        '2022-11-02', '2022-12-14',

        # 2023
        '2023-02-01', '2023-03-22', '2023-05-03',
        '2023-06-14', '2023-07-26', '2023-09-20',
        '2023-11-01', '2023-12-13',

        # 2024
        '2024-01-31', '2024-03-20', '2024-05-01',
        '2024-06-12', '2024-07-31', '2024-09-18',
        '2024-11-07', '2024-12-18',

        # 2025 (scheduled)
        '2025-01-29', '2025-03-19', '2025-05-07',
        '2025-06-18', '2025-07-30', '2025-09-17',
        '2025-11-05', '2025-12-10',
    ]

    def __init__(self, exclusion_days_before=1, exclusion_days_after=1):
        """
        Initialize FOMC calendar fetcher.

        Args:
            exclusion_days_before: Days before FOMC meeting to exclude
            exclusion_days_after: Days after FOMC meeting to exclude
        """
        self.exclusion_days_before = exclusion_days_before
        self.exclusion_days_after = exclusion_days_after

    def get_fomc_dates(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get FOMC meeting dates within date range.

        Args:
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            DataFrame with FOMC meeting dates
        """
        # Convert to DataFrame
        fomc_df = pd.DataFrame({
            'fomc_date': pd.to_datetime(self.FOMC_DATES)
        })

        # Remove timezone for consistency
        fomc_df['fomc_date'] = fomc_df['fomc_date'].dt.tz_localize(None)

        # Filter by date range if provided
        if start_date:
            start_dt = pd.to_datetime(start_date).tz_localize(None)
            fomc_df = fomc_df[fomc_df['fomc_date'] >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date).tz_localize(None)
            fomc_df = fomc_df[fomc_df['fomc_date'] <= end_dt]

        return fomc_df.sort_values('fomc_date').reset_index(drop=True)

    def get_exclusion_dates(self, start_date: str = None, end_date: str = None) -> Set[pd.Timestamp]:
        """
        Get all dates to exclude (FOMC date +/- exclusion window).

        Args:
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            Set of timestamps to exclude from trading
        """
        fomc_df = self.get_fomc_dates(start_date, end_date)

        exclusion_dates = set()

        for fomc_date in fomc_df['fomc_date']:
            # Add FOMC date itself
            exclusion_dates.add(pd.Timestamp(fomc_date.date()))

            # Add days before
            for i in range(1, self.exclusion_days_before + 1):
                exclusion_dates.add(pd.Timestamp((fomc_date - timedelta(days=i)).date()))

            # Add days after
            for i in range(1, self.exclusion_days_after + 1):
                exclusion_dates.add(pd.Timestamp((fomc_date + timedelta(days=i)).date()))

        return exclusion_dates

    def add_fomc_filter_to_signals(
        self,
        df: pd.DataFrame,
        signal_column: str = 'panic_sell'
    ) -> pd.DataFrame:
        """
        Filter out signals near FOMC meetings.

        Args:
            df: DataFrame with trading signals and 'Date' column
            signal_column: Name of signal column to filter

        Returns:
            DataFrame with filtered signals
        """
        data = df.copy()

        if 'Date' not in data.columns:
            print("[WARN] No 'Date' column found, cannot apply FOMC filter")
            return data

        # Get exclusion dates for entire period
        start_date = data['Date'].min()
        end_date = data['Date'].max()

        exclusion_dates = self.get_exclusion_dates(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        if not exclusion_dates:
            print("[INFO] No FOMC meetings in date range")
            return data

        # Store original signals
        if signal_column in data.columns:
            data[f'{signal_column}_before_fomc_filter'] = data[signal_column].copy()

            # Normalize dates for comparison
            data['trade_date_normalized'] = pd.to_datetime(data['Date'].dt.date)

            # Filter out signals near FOMC meetings
            fomc_mask = data['trade_date_normalized'].isin(exclusion_dates)
            data.loc[fomc_mask, signal_column] = 0

            # Clean up temporary column
            data = data.drop(columns=['trade_date_normalized'])

            signals_blocked = fomc_mask.sum()
            if signals_blocked > 0:
                print(f"[INFO] FOMC filter blocked {signals_blocked} signals")

        return data

    def analyze_fomc_periods(self, start_date: str, end_date: str) -> dict:
        """
        Analyze FOMC meeting frequency in period.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with FOMC statistics
        """
        fomc_df = self.get_fomc_dates(start_date, end_date)

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        total_days = (end_dt - start_dt).days

        # Calculate exclusion days
        exclusion_dates = self.get_exclusion_dates(start_date, end_date)
        exclusion_days = len(exclusion_dates)

        return {
            'fomc_meetings': len(fomc_df),
            'total_days': total_days,
            'exclusion_days': exclusion_days,
            'exclusion_pct': float(exclusion_days / total_days * 100) if total_days > 0 else 0,
            'avg_days_between_meetings': float(total_days / len(fomc_df)) if len(fomc_df) > 0 else 0,
            'exclusion_window': f"{self.exclusion_days_before} days before + {self.exclusion_days_after} days after"
        }


# Convenience function for quick filtering
def filter_signals_by_fomc(
    df: pd.DataFrame,
    signal_column: str = 'panic_sell',
    exclusion_days_before: int = 1,
    exclusion_days_after: int = 1
) -> pd.DataFrame:
    """
    Quick function to filter signals by FOMC meetings.

    Args:
        df: DataFrame with signals
        signal_column: Signal column name
        exclusion_days_before: Days before FOMC to exclude
        exclusion_days_after: Days after FOMC to exclude

    Returns:
        Filtered DataFrame
    """
    fetcher = FOMCCalendarFetcher(
        exclusion_days_before=exclusion_days_before,
        exclusion_days_after=exclusion_days_after
    )
    return fetcher.add_fomc_filter_to_signals(df, signal_column)


if __name__ == "__main__":
    print("FOMC Calendar Fetcher Module")
    print()
    print("Purpose: Disable trading around FOMC meetings (policy-driven moves)")
    print("Default exclusion: 1 day before + 1 day after meeting")
    print()
    print("Usage:")
    print("  fetcher = FOMCCalendarFetcher(exclusion_days_before=1, exclusion_days_after=1)")
    print("  filtered_data = fetcher.add_fomc_filter_to_signals(data)")
    print()

    # Show FOMC meeting frequency
    fetcher = FOMCCalendarFetcher()

    test_periods = [
        ('2020-01-01', '2020-12-31', '2020'),
        ('2022-01-01', '2022-12-31', '2022'),
        ('2024-01-01', '2024-12-31', '2024'),
    ]

    print("FOMC Meeting Frequency:")
    for start, end, year in test_periods:
        stats = fetcher.analyze_fomc_periods(start, end)
        print(f"  {year}: {stats['fomc_meetings']} meetings, "
              f"{stats['exclusion_days']} exclusion days ({stats['exclusion_pct']:.1f}% of year)")

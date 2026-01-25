"""Debug script to check why no signals are found."""

import sys
import os
import pandas as pd
from datetime import timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.models.trading.mean_reversion import MeanReversionDetector

# Test with NVDA
ticker = 'NVDA'
start_date = '2020-01-01'
end_date = '2024-11-17'  # Fixed: Use actual current date

buffer_start = (pd.to_datetime(start_date) - timedelta(days=180)).strftime('%Y-%m-%d')
fetcher = YahooFinanceFetcher()
data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)

print(f"Fetched {len(data)} days of data for {ticker}")
print(f"Date range: {data.index[0]} to {data.index[-1]}")

# Engineer features
engineer = TechnicalFeatureEngineer(fillna=True)
enriched_data = engineer.engineer_features(data)

print(f"Features engineered: {len(enriched_data)} rows")

# Detect signals
detector = MeanReversionDetector(
    z_score_threshold=1.5,
    rsi_oversold=35,
    volume_multiplier=1.3,
    price_drop_threshold=-1.5
)

signals = detector.detect_overcorrections(enriched_data)

panic_signals = signals[signals['panic_sell'] == 1]

print(f"Total panic_sell signals: {len(panic_signals)}")
print(f"Panic sell dates: {panic_signals.index.tolist()[:10]}")  # First 10

if len(panic_signals) > 0:
    print("\nFirst signal:")
    print(panic_signals.iloc[0][['z_score', 'rsi', 'daily_return', 'volume_spike']])

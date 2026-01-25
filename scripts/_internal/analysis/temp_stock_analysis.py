"""
NVDA and PLTR Technical Analysis with Outlook
"""
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, 'src')

from data.fetchers.rate_limited_yahoo import RateLimitedYahooFinanceFetcher
from trading.signal_scanner import SignalScanner

# Initialize
fetcher = RateLimitedYahooFinanceFetcher(verbose=True)
scanner = SignalScanner(lookback_days=90)

# Get data for both stocks
today = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

print("Fetching NVIDIA (NVDA) data...")
nvda_data = fetcher.get_stock_data('NVDA', start_date, today)

print("\nFetching PALANTIR (PLTR) data...")
pltr_data = fetcher.get_stock_data('PLTR', start_date, today)

# Analyze NVDA
print("\n" + "="*70)
print("NVIDIA (NVDA) ANALYSIS")
print("="*70)

if nvda_data is not None and not nvda_data.empty:
    current_price = float(nvda_data['Close'].iloc[-1])
    prev_close = float(nvda_data['Close'].iloc[-2])
    daily_change = ((current_price - prev_close) / prev_close) * 100

    # Calculate moving averages
    ma_20 = nvda_data['Close'].rolling(20).mean().iloc[-1]
    ma_50 = nvda_data['Close'].rolling(50).mean().iloc[-1] if len(nvda_data) >= 50 else None

    # Calculate recent performance
    week_ago = nvda_data['Close'].iloc[-5] if len(nvda_data) >= 5 else current_price
    month_ago = nvda_data['Close'].iloc[-20] if len(nvda_data) >= 20 else current_price

    week_change = ((current_price - week_ago) / week_ago) * 100
    month_change = ((current_price - month_ago) / month_ago) * 100

    # Calculate RSI
    delta = nvda_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    print(f"Current Price: ${current_price:.2f}")
    print(f"Daily Change: {daily_change:+.2f}%")
    print(f"Weekly Change: {week_change:+.2f}%")
    print(f"Monthly Change: {month_change:+.2f}%")
    print(f"\n20-Day MA: ${ma_20:.2f} ({((current_price - ma_20) / ma_20 * 100):+.2f}%)")
    if ma_50:
        print(f"50-Day MA: ${ma_50:.2f} ({((current_price - ma_50) / ma_50 * 100):+.2f}%)")
    print(f"\nRSI (14): {current_rsi:.1f}")

    # Technical state
    if current_rsi < 30:
        state = "OVERSOLD - Mean reversion opportunity"
    elif current_rsi > 70:
        state = "OVERBOUGHT - Potential pullback"
    else:
        state = "NEUTRAL - No extreme conditions"
    print(f"Technical State: {state}")

    # Store for email
    nvda_analysis = {
        'current_price': current_price,
        'daily_change': daily_change,
        'week_change': week_change,
        'month_change': month_change,
        'ma_20': ma_20,
        'ma_50': ma_50,
        'rsi': current_rsi,
        'state': state
    }
else:
    print("Failed to fetch NVDA data")
    nvda_analysis = None

# Analyze PLTR
print("\n" + "="*70)
print("PALANTIR (PLTR) ANALYSIS")
print("="*70)

if pltr_data is not None and not pltr_data.empty:
    current_price = float(pltr_data['Close'].iloc[-1])
    prev_close = float(pltr_data['Close'].iloc[-2])
    daily_change = ((current_price - prev_close) / prev_close) * 100

    # Calculate moving averages
    ma_20 = pltr_data['Close'].rolling(20).mean().iloc[-1]
    ma_50 = pltr_data['Close'].rolling(50).mean().iloc[-1] if len(pltr_data) >= 50 else None

    # Calculate recent performance
    week_ago = pltr_data['Close'].iloc[-5] if len(pltr_data) >= 5 else current_price
    month_ago = pltr_data['Close'].iloc[-20] if len(pltr_data) >= 20 else current_price

    week_change = ((current_price - week_ago) / week_ago) * 100
    month_change = ((current_price - month_ago) / month_ago) * 100

    # Calculate RSI
    delta = pltr_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    print(f"Current Price: ${current_price:.2f}")
    print(f"Daily Change: {daily_change:+.2f}%")
    print(f"Weekly Change: {week_change:+.2f}%")
    print(f"Monthly Change: {month_change:+.2f}%")
    print(f"\n20-Day MA: ${ma_20:.2f} ({((current_price - ma_20) / ma_20 * 100):+.2f}%)")
    if ma_50:
        print(f"50-Day MA: ${ma_50:.2f} ({((current_price - ma_50) / ma_50 * 100):+.2f}%)")
    print(f"\nRSI (14): {current_rsi:.1f}")

    # Technical state
    if current_rsi < 30:
        state = "OVERSOLD - Mean reversion opportunity"
    elif current_rsi > 70:
        state = "OVERBOUGHT - Potential pullback"
    else:
        state = "NEUTRAL - No extreme conditions"
    print(f"Technical State: {state}")

    # Store for email
    pltr_analysis = {
        'current_price': current_price,
        'daily_change': daily_change,
        'week_change': week_change,
        'month_change': month_change,
        'ma_20': ma_20,
        'ma_50': ma_50,
        'rsi': current_rsi,
        'state': state
    }
else:
    print("Failed to fetch PLTR data")
    pltr_analysis = None

# Save analysis
import json
with open('temp_stock_analysis_results.json', 'w') as f:
    json.dump({
        'nvda': nvda_analysis,
        'pltr': pltr_analysis,
        'timestamp': today
    }, f, indent=2)

print("\n" + "="*70)
print("Analysis complete - saved to temp_stock_analysis_results.json")
print("="*70)

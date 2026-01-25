"""
Overnight Data Refresh Script

Pre-fetches and caches data for faster morning analysis:
1. Refresh 1-year price data for all 54 tickers
2. Update earnings calendar cache
3. Cache sector performance data
4. Generate volatility profiles per stock
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Stock universe
TICKERS = [
    'NVDA', 'AVGO', 'MSFT', 'ORCL', 'INTU', 'ADBE', 'CRM', 'NOW', 'AMAT', 'KLAC',
    'MRVL', 'QCOM', 'TXN', 'ADI', 'JPM', 'MS', 'SCHW', 'AXP', 'AIG', 'USB',
    'PNC', 'V', 'MA', 'ABBV', 'SYK', 'GILD', 'PFE', 'JNJ', 'CVS', 'HCA',
    'IDXX', 'INSM', 'COP', 'SLB', 'XOM', 'MPC', 'EOG', 'CAT', 'ETN', 'LMT',
    'ROAD', 'HD', 'LOW', 'TGT', 'WMT', 'TMUS', 'CMCSA', 'META', 'APD', 'ECL',
    'MLM', 'SHW', 'EXR', 'NEE'
]

# Sector ETFs
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE']

# Market benchmarks
BENCHMARKS = ['SPY', 'QQQ', 'IWM', 'VIX']

CACHE_DIR = Path('features/daily_picks/data/cache')
LOG_DIR = Path('features/daily_picks/logs')


@dataclass
class StockCache:
    """Cached data for a single stock."""
    ticker: str
    last_updated: str
    days_of_data: int
    latest_price: float
    latest_volume: int
    avg_volume_20d: float
    atr_14d: float
    atr_pct: float
    rsi_14d: float
    sma_20: float
    sma_50: float
    sma_200: float
    volatility_30d: float
    consecutive_down_days: int
    ytd_return: float
    mtd_return: float
    week_return: float


def setup_directories():
    """Create necessary directories."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def calculate_rsi(prices, period=14):
    """Calculate RSI."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
    """Calculate Average True Range."""
    high = df['High']
    low = df['Low']
    close = df['Close'].shift(1)

    tr = pd.concat([
        high - low,
        abs(high - close),
        abs(low - close)
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def get_consecutive_down_days(closes):
    """Count consecutive down days ending today."""
    count = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes.iloc[i] < closes.iloc[i-1]:
            count += 1
        else:
            break
    return count


def fetch_and_cache_stock(ticker: str, days: int = 400) -> StockCache:
    """Fetch and process data for a single stock."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    if len(df) < 50:
        raise ValueError(f"Insufficient data for {ticker}: only {len(df)} rows")

    # Calculate technical indicators
    df['RSI'] = calculate_rsi(df['Close'])
    df['ATR'] = calculate_atr(df)
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['Returns'] = df['Close'].pct_change()
    df['Volatility_30d'] = df['Returns'].rolling(30).std() * np.sqrt(252)

    latest = df.iloc[-1]
    close = latest['Close']

    # Calculate returns
    ytd_start = datetime(end_date.year, 1, 1)
    mtd_start = datetime(end_date.year, end_date.month, 1)
    week_start = end_date - timedelta(days=7)

    ytd_data = df[df.index >= pd.Timestamp(ytd_start)]
    mtd_data = df[df.index >= pd.Timestamp(mtd_start)]
    week_data = df[df.index >= pd.Timestamp(week_start)]

    ytd_return = ((close / ytd_data['Close'].iloc[0]) - 1) * 100 if len(ytd_data) > 0 else 0
    mtd_return = ((close / mtd_data['Close'].iloc[0]) - 1) * 100 if len(mtd_data) > 0 else 0
    week_return = ((close / week_data['Close'].iloc[0]) - 1) * 100 if len(week_data) > 0 else 0

    # Save raw price data as CSV (more compatible than parquet)
    price_file = CACHE_DIR / f"{ticker}_prices.csv"
    df.to_csv(price_file)

    return StockCache(
        ticker=ticker,
        last_updated=datetime.now().isoformat(),
        days_of_data=len(df),
        latest_price=round(close, 2),
        latest_volume=int(latest['Volume']),
        avg_volume_20d=round(df['Volume'].tail(20).mean(), 0),
        atr_14d=round(latest['ATR'], 4) if pd.notna(latest['ATR']) else 0,
        atr_pct=round((latest['ATR'] / close) * 100, 3) if pd.notna(latest['ATR']) and close > 0 else 0,
        rsi_14d=round(latest['RSI'], 2) if pd.notna(latest['RSI']) else 50,
        sma_20=round(latest['SMA_20'], 2) if pd.notna(latest['SMA_20']) else close,
        sma_50=round(latest['SMA_50'], 2) if pd.notna(latest['SMA_50']) else close,
        sma_200=round(latest['SMA_200'], 2) if pd.notna(latest['SMA_200']) else close,
        volatility_30d=round(latest['Volatility_30d'] * 100, 2) if pd.notna(latest['Volatility_30d']) else 0,
        consecutive_down_days=get_consecutive_down_days(df['Close']),
        ytd_return=round(ytd_return, 2),
        mtd_return=round(mtd_return, 2),
        week_return=round(week_return, 2)
    )


def fetch_sector_data() -> Dict:
    """Fetch and cache sector ETF data."""
    print("\nFetching sector ETFs...")
    sector_data = {}

    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)

    for etf in SECTOR_ETFS:
        try:
            df = yf.download(etf, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            if len(df) >= 20:
                latest = df['Close'].iloc[-1]
                prev_5d = df['Close'].iloc[-6] if len(df) >= 6 else latest
                prev_20d = df['Close'].iloc[-21] if len(df) >= 21 else latest

                sector_data[etf] = {
                    'latest_price': round(latest, 2),
                    'return_5d': round(((latest / prev_5d) - 1) * 100, 2),
                    'return_20d': round(((latest / prev_20d) - 1) * 100, 2),
                    'relative_strength': 'strong' if ((latest / prev_5d) - 1) * 100 > 2 else
                                        'weak' if ((latest / prev_5d) - 1) * 100 < -2 else 'neutral'
                }
                print(f"  {etf}: {sector_data[etf]['return_5d']:+.1f}% (5d)")
        except Exception as e:
            print(f"  {etf}: Failed - {e}")

    # Save sector data
    sector_file = CACHE_DIR / 'sector_performance.json'
    with open(sector_file, 'w') as f:
        json.dump({
            'last_updated': datetime.now().isoformat(),
            'sectors': sector_data
        }, f, indent=2)

    return sector_data


def fetch_benchmark_data() -> Dict:
    """Fetch market benchmark data."""
    print("\nFetching benchmarks...")
    benchmark_data = {}

    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)

    for symbol in BENCHMARKS:
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            if len(df) >= 20:
                latest = df['Close'].iloc[-1]
                prev_5d = df['Close'].iloc[-6] if len(df) >= 6 else latest

                benchmark_data[symbol] = {
                    'latest': round(latest, 2),
                    'return_5d': round(((latest / prev_5d) - 1) * 100, 2)
                }
                print(f"  {symbol}: {latest:.2f} ({benchmark_data[symbol]['return_5d']:+.1f}% 5d)")
        except Exception as e:
            print(f"  {symbol}: Failed - {e}")

    # Determine market regime based on VIX and SPY
    regime = 'BULL'
    if 'VIX' in benchmark_data and benchmark_data['VIX']['latest'] > 25:
        regime = 'VOLATILE'
    elif 'SPY' in benchmark_data and benchmark_data['SPY']['return_5d'] < -3:
        regime = 'BEAR'
    elif 'SPY' in benchmark_data and abs(benchmark_data['SPY']['return_5d']) < 0.5:
        regime = 'CHOPPY'

    benchmark_data['regime'] = regime
    print(f"  Market Regime: {regime}")

    # Save benchmark data
    benchmark_file = CACHE_DIR / 'benchmark_data.json'
    with open(benchmark_file, 'w') as f:
        json.dump({
            'last_updated': datetime.now().isoformat(),
            'benchmarks': benchmark_data
        }, f, indent=2)

    return benchmark_data


def generate_volatility_profiles(stock_caches: Dict[str, StockCache]) -> Dict:
    """Generate volatility analysis across all stocks."""
    print("\nGenerating volatility profiles...")

    atr_values = {t: c.atr_pct for t, c in stock_caches.items() if c.atr_pct > 0}
    vol_values = {t: c.volatility_30d for t, c in stock_caches.items() if c.volatility_30d > 0}

    # Categorize by volatility
    atr_list = sorted(atr_values.values())
    vol_list = sorted(vol_values.values())

    atr_q1, atr_q2, atr_q3 = np.percentile(atr_list, [25, 50, 75]) if len(atr_list) >= 4 else (1, 2, 3)
    vol_q1, vol_q2, vol_q3 = np.percentile(vol_list, [25, 50, 75]) if len(vol_list) >= 4 else (15, 25, 35)

    volatility_profiles = {
        'low_volatility': [],
        'medium_volatility': [],
        'high_volatility': []
    }

    for ticker, cache in stock_caches.items():
        if cache.atr_pct <= atr_q1:
            volatility_profiles['low_volatility'].append(ticker)
        elif cache.atr_pct >= atr_q3:
            volatility_profiles['high_volatility'].append(ticker)
        else:
            volatility_profiles['medium_volatility'].append(ticker)

    profiles = {
        'atr_quartiles': {'q1': round(atr_q1, 3), 'q2': round(atr_q2, 3), 'q3': round(atr_q3, 3)},
        'vol_quartiles': {'q1': round(vol_q1, 2), 'q2': round(vol_q2, 2), 'q3': round(vol_q3, 2)},
        'categorization': volatility_profiles,
        'stock_profiles': {t: {
            'atr_pct': c.atr_pct,
            'volatility_30d': c.volatility_30d,
            'category': 'low' if c.atr_pct <= atr_q1 else 'high' if c.atr_pct >= atr_q3 else 'medium'
        } for t, c in stock_caches.items()}
    }

    print(f"  Low vol: {len(volatility_profiles['low_volatility'])} stocks")
    print(f"  Med vol: {len(volatility_profiles['medium_volatility'])} stocks")
    print(f"  High vol: {len(volatility_profiles['high_volatility'])} stocks")

    # Save profiles
    profile_file = CACHE_DIR / 'volatility_profiles.json'
    with open(profile_file, 'w') as f:
        json.dump({
            'last_updated': datetime.now().isoformat(),
            'profiles': profiles
        }, f, indent=2)

    return profiles


def run_overnight_refresh():
    """Run complete overnight data refresh."""
    start_time = time.time()

    print("=" * 70)
    print("OVERNIGHT DATA REFRESH")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    setup_directories()

    # Fetch all stock data
    print(f"\nFetching data for {len(TICKERS)} stocks...")
    stock_caches = {}
    failed = []

    for i, ticker in enumerate(TICKERS):
        try:
            cache = fetch_and_cache_stock(ticker)
            stock_caches[ticker] = cache
        except Exception as e:
            failed.append((ticker, str(e)))

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(TICKERS)} complete")

    print(f"\n  Successfully cached: {len(stock_caches)}")
    if failed:
        print(f"  Failed: {len(failed)} - {[f[0] for f in failed]}")

    # Fetch sector data
    sector_data = fetch_sector_data()

    # Fetch benchmark data
    benchmark_data = fetch_benchmark_data()

    # Generate volatility profiles
    volatility_profiles = generate_volatility_profiles(stock_caches)

    # Save master cache index
    master_cache = {
        'last_updated': datetime.now().isoformat(),
        'stocks_cached': len(stock_caches),
        'stocks_failed': len(failed),
        'regime': benchmark_data.get('regime', 'UNKNOWN'),
        'summary': {
            'avg_rsi': round(np.mean([c.rsi_14d for c in stock_caches.values()]), 1),
            'avg_atr_pct': round(np.mean([c.atr_pct for c in stock_caches.values()]), 2),
            'avg_ytd_return': round(np.mean([c.ytd_return for c in stock_caches.values()]), 1),
            'oversold_count': sum(1 for c in stock_caches.values() if c.rsi_14d < 30),
            'overbought_count': sum(1 for c in stock_caches.values() if c.rsi_14d > 70),
        },
        'stocks': {t: asdict(c) for t, c in stock_caches.items()},
        'failed': failed
    }

    cache_file = CACHE_DIR / 'overnight_cache.json'
    with open(cache_file, 'w') as f:
        json.dump(master_cache, f, indent=2)

    # Summary
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Duration: {elapsed:.1f} seconds")
    print(f"Stocks cached: {len(stock_caches)}")
    print(f"Sector ETFs: {len(sector_data)}")
    print(f"Market Regime: {benchmark_data.get('regime', 'UNKNOWN')}")
    print(f"\nKey Stats:")
    print(f"  Avg RSI: {master_cache['summary']['avg_rsi']}")
    print(f"  Avg ATR%: {master_cache['summary']['avg_atr_pct']}%")
    print(f"  Avg YTD Return: {master_cache['summary']['avg_ytd_return']}%")
    print(f"  Oversold (RSI<30): {master_cache['summary']['oversold_count']}")
    print(f"  Overbought (RSI>70): {master_cache['summary']['overbought_count']}")
    print(f"\nCache saved to: {cache_file}")

    return master_cache


if __name__ == '__main__':
    run_overnight_refresh()

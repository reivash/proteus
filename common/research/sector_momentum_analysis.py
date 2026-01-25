"""
Sector Momentum Analysis for Mean Reversion Signals

Investigates whether sector-level momentum affects RSI<35 mean reversion signal performance.

Hypothesis: Signals in sectors with recent weakness may have higher bounce rates than
signals in sectors that are already strong (less mean-reversion potential).

Sectors analyzed via ETF proxies:
- XLF: Financials (JPM, MS, SCHW, AXP, AIG, USB, PNC)
- XLK: Technology (NVDA, AVGO, MSFT, ORCL, INTU, ADBE, CRM, NOW)
- XLV: Healthcare (ABBV, SYK, GILD, PFE, JNJ, CVS, HCA, IDXX)
- XLE: Energy (COP, SLB, XOM, MPC, EOG)
- XLI: Industrials (CAT, ETN, LMT)
- XLY: Consumer Discretionary (HD, LOW, TGT, WMT)
- XLC: Communication (TMUS, CMCSA, META)
- XLB: Materials (APD, ECL, MLM, SHW)
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
from pathlib import Path


# Sector ETF mapping
SECTOR_ETFS = {
    'XLF': 'Financials',
    'XLK': 'Technology',
    'XLV': 'Healthcare',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLY': 'Consumer',
    'XLC': 'Communication',
    'XLB': 'Materials',
}

# Stock to sector mapping
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

    # REITs / Other
    'EXR': 'XLRE', 'NEE': 'XLU',
}


def calculate_rsi(prices, period=14):
    """Calculate RSI."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_sector_momentum(sector_df, date, lookback=5):
    """Calculate sector momentum at a given date."""
    try:
        sector_slice = sector_df.loc[:date].tail(lookback + 1)
        if len(sector_slice) < lookback + 1:
            return 0, 'neutral'

        momentum = ((sector_slice['Close'].iloc[-1] / sector_slice['Close'].iloc[0]) - 1) * 100

        # Classify momentum
        if momentum < -3:
            category = 'weak'
        elif momentum < -1:
            category = 'slightly_weak'
        elif momentum < 1:
            category = 'neutral'
        elif momentum < 3:
            category = 'slightly_strong'
        else:
            category = 'strong'

        return momentum, category
    except:
        return 0, 'neutral'


def run_sector_momentum_analysis(days_back=365):
    """Analyze sector momentum effect on signal performance."""

    print('=' * 70)
    print('SECTOR MOMENTUM ANALYSIS')
    print('=' * 70)
    print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'Period: {days_back} days')
    print()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 50)

    # Fetch sector ETF data
    print('Fetching sector ETF data...')
    sector_data = {}
    for etf in SECTOR_ETFS.keys():
        try:
            sector_data[etf] = yf.Ticker(etf).history(start=start_date, end=end_date)
            print(f'  {etf}: {len(sector_data[etf])} days')
        except:
            print(f'  {etf}: FAILED')

    print()

    # Analyze stocks
    tickers = list(set(STOCK_SECTOR.keys()))

    print(f'Analyzing {len(tickers)} stocks...')
    print()

    results = []

    for ticker in tickers:
        sector_etf = STOCK_SECTOR.get(ticker)
        if not sector_etf or sector_etf not in sector_data:
            continue

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if len(df) < 50:
                continue

            # Calculate RSI
            df['RSI'] = calculate_rsi(df['Close'])

            # Find RSI < 35 signals
            signals = df[df['RSI'] < 35].copy()

            sector_df = sector_data[sector_etf]

            for idx in signals.index:
                try:
                    pos = df.index.get_loc(idx)
                    if pos + 5 > len(df):
                        continue

                    entry_price = df['Close'].iloc[pos]

                    # Get sector momentum
                    sector_mom, sector_cat = calculate_sector_momentum(sector_df, idx)

                    # Calculate forward returns
                    exit_price_2d = df['Close'].iloc[min(pos + 2, len(df)-1)]
                    ret_2d = ((exit_price_2d / entry_price) - 1) * 100

                    future = df.iloc[pos:min(pos+5, len(df))]
                    max_gain = ((future['High'].max() / entry_price) - 1) * 100
                    max_loss = ((future['Low'].min() / entry_price) - 1) * 100

                    results.append({
                        'ticker': ticker,
                        'date': idx.strftime('%Y-%m-%d'),
                        'sector': SECTOR_ETFS.get(sector_etf, sector_etf),
                        'sector_etf': sector_etf,
                        'sector_momentum': sector_mom,
                        'sector_category': sector_cat,
                        'rsi': df['RSI'].iloc[pos],
                        'ret_2d': ret_2d,
                        'max_gain': max_gain,
                        'max_loss': max_loss,
                        'win': ret_2d > 0,
                        'hit_target': max_gain >= 2.0
                    })

                except:
                    continue

        except Exception as e:
            continue

    if not results:
        print('No signals found!')
        return

    df = pd.DataFrame(results)

    # Analyze by sector momentum category
    print('=' * 70)
    print('RESULTS BY SECTOR MOMENTUM')
    print('=' * 70)
    print()

    summary_by_momentum = {}

    for category in ['weak', 'slightly_weak', 'neutral', 'slightly_strong', 'strong']:
        cat_df = df[df['sector_category'] == category]
        if len(cat_df) > 0:
            n = len(cat_df)
            win_rate = cat_df['win'].mean() * 100
            hit_rate = cat_df['hit_target'].mean() * 100
            avg_ret = cat_df['ret_2d'].mean()
            avg_sector_mom = cat_df['sector_momentum'].mean()

            summary_by_momentum[category] = {
                'signals': n,
                'win_rate': round(win_rate, 1),
                'hit_rate': round(hit_rate, 1),
                'avg_ret': round(avg_ret, 2),
                'avg_sector_momentum': round(avg_sector_mom, 2)
            }

            print(f'{category.upper()} SECTOR MOMENTUM (5d: {avg_sector_mom:+.1f}%):')
            print(f'  Signals: {n}')
            print(f'  Win rate: {win_rate:.1f}%')
            print(f'  Hit rate (2%): {hit_rate:.1f}%')
            print(f'  Avg 2-day return: {avg_ret:+.2f}%')
            print()

    # Analyze by sector
    print('=' * 70)
    print('RESULTS BY SECTOR')
    print('=' * 70)
    print()

    summary_by_sector = {}

    for sector in df['sector'].unique():
        sector_df_filtered = df[df['sector'] == sector]
        if len(sector_df_filtered) >= 10:  # Minimum sample size
            n = len(sector_df_filtered)
            win_rate = sector_df_filtered['win'].mean() * 100
            hit_rate = sector_df_filtered['hit_target'].mean() * 100
            avg_ret = sector_df_filtered['ret_2d'].mean()

            summary_by_sector[sector] = {
                'signals': n,
                'win_rate': round(win_rate, 1),
                'hit_rate': round(hit_rate, 1),
                'avg_ret': round(avg_ret, 2)
            }

            print(f'{sector}:')
            print(f'  Signals: {n}, Win: {win_rate:.1f}%, Hit: {hit_rate:.1f}%, Ret: {avg_ret:+.2f}%')

    # Cross-analysis: Best combination of sector + momentum
    print()
    print('=' * 70)
    print('SECTOR x MOMENTUM CROSS-ANALYSIS')
    print('=' * 70)
    print()

    cross_analysis = []

    for sector in df['sector'].unique():
        for cat in ['weak', 'slightly_weak', 'neutral', 'slightly_strong', 'strong']:
            subset = df[(df['sector'] == sector) & (df['sector_category'] == cat)]
            if len(subset) >= 5:
                cross_analysis.append({
                    'sector': sector,
                    'momentum': cat,
                    'signals': len(subset),
                    'win_rate': subset['win'].mean() * 100,
                    'avg_ret': subset['ret_2d'].mean()
                })

    # Sort by avg return
    cross_analysis.sort(key=lambda x: x['avg_ret'], reverse=True)

    print('TOP 10 SECTOR + MOMENTUM COMBINATIONS:')
    for i, combo in enumerate(cross_analysis[:10]):
        print(f'  {i+1}. {combo["sector"]} + {combo["momentum"]}: n={combo["signals"]}, '
              f'win={combo["win_rate"]:.0f}%, ret={combo["avg_ret"]:+.2f}%')

    print()
    print('BOTTOM 10 SECTOR + MOMENTUM COMBINATIONS:')
    for i, combo in enumerate(cross_analysis[-10:]):
        print(f'  {i+1}. {combo["sector"]} + {combo["momentum"]}: n={combo["signals"]}, '
              f'win={combo["win_rate"]:.0f}%, ret={combo["avg_ret"]:+.2f}%')

    # Key finding: Is sector weakness helpful?
    print()
    print('=' * 70)
    print('KEY FINDING: SECTOR WEAKNESS EFFECT')
    print('=' * 70)

    weak_sectors = df[df['sector_category'].isin(['weak', 'slightly_weak'])]
    strong_sectors = df[df['sector_category'].isin(['strong', 'slightly_strong'])]

    if len(weak_sectors) > 0 and len(strong_sectors) > 0:
        print()
        print(f'WEAK SECTOR SIGNALS (n={len(weak_sectors)}):')
        print(f'  Win rate: {weak_sectors["win"].mean()*100:.1f}%')
        print(f'  Avg return: {weak_sectors["ret_2d"].mean():+.2f}%')
        print()
        print(f'STRONG SECTOR SIGNALS (n={len(strong_sectors)}):')
        print(f'  Win rate: {strong_sectors["win"].mean()*100:.1f}%')
        print(f'  Avg return: {strong_sectors["ret_2d"].mean():+.2f}%')
        print()

        diff = weak_sectors["ret_2d"].mean() - strong_sectors["ret_2d"].mean()
        if diff > 0:
            print(f'>>> WEAK SECTORS OUTPERFORM BY {diff:+.2f}% per trade <<<')
        else:
            print(f'>>> STRONG SECTORS OUTPERFORM BY {-diff:+.2f}% per trade <<<')

    # Save results
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'days_analyzed': days_back,
        'total_signals': len(df),
        'by_sector_momentum': summary_by_momentum,
        'by_sector': summary_by_sector,
        'top_combinations': cross_analysis[:10],
        'bottom_combinations': cross_analysis[-10:]
    }

    output_path = Path('data/research/sector_momentum_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print()
    print(f'Results saved to {output_path}')

    return results_data


if __name__ == '__main__':
    run_sector_momentum_analysis(days_back=365)

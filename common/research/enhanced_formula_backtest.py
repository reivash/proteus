"""
Comprehensive Backtest of Enhanced Signal Strength Formula

Validates the enhanced signal calculator by comparing:
1. Baseline (raw RSI < 35 signals)
2. Enhanced formula (with all adjustments)

Measures improvement in:
- Win rate
- Average return
- Hit rate (2% target)
- Risk-adjusted returns

Uses the same trading universe and period as other research.
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.trading.enhanced_signal_calculator import EnhancedSignalCalculator


# All stocks in the trading universe
TICKERS = [
    'NVDA', 'AVGO', 'MSFT', 'ORCL', 'INTU', 'ADBE', 'CRM', 'NOW', 'AMAT', 'KLAC',
    'MRVL', 'QCOM', 'TXN', 'ADI', 'JPM', 'MS', 'SCHW', 'AXP', 'AIG', 'USB',
    'PNC', 'V', 'MA', 'ABBV', 'SYK', 'GILD', 'PFE', 'JNJ', 'CVS', 'HCA',
    'IDXX', 'INSM', 'COP', 'SLB', 'XOM', 'MPC', 'EOG', 'CAT', 'ETN', 'LMT',
    'ROAD', 'HD', 'LOW', 'TGT', 'WMT', 'TMUS', 'CMCSA', 'META', 'APD', 'ECL',
    'MLM', 'SHW', 'EXR', 'NEE'
]

# Sector mapping for sector momentum
STOCK_SECTOR = {
    'NVDA': ('XLK', 'Technology'), 'AVGO': ('XLK', 'Technology'), 'MSFT': ('XLK', 'Technology'),
    'ORCL': ('XLK', 'Technology'), 'INTU': ('XLK', 'Technology'), 'ADBE': ('XLK', 'Technology'),
    'CRM': ('XLK', 'Technology'), 'NOW': ('XLK', 'Technology'), 'AMAT': ('XLK', 'Technology'),
    'KLAC': ('XLK', 'Technology'), 'MRVL': ('XLK', 'Technology'), 'QCOM': ('XLK', 'Technology'),
    'TXN': ('XLK', 'Technology'), 'ADI': ('XLK', 'Technology'),
    'JPM': ('XLF', 'Financials'), 'MS': ('XLF', 'Financials'), 'SCHW': ('XLF', 'Financials'),
    'AXP': ('XLF', 'Financials'), 'AIG': ('XLF', 'Financials'), 'USB': ('XLF', 'Financials'),
    'PNC': ('XLF', 'Financials'), 'V': ('XLF', 'Financials'), 'MA': ('XLF', 'Financials'),
    'ABBV': ('XLV', 'Healthcare'), 'SYK': ('XLV', 'Healthcare'), 'GILD': ('XLV', 'Healthcare'),
    'PFE': ('XLV', 'Healthcare'), 'JNJ': ('XLV', 'Healthcare'), 'CVS': ('XLV', 'Healthcare'),
    'HCA': ('XLV', 'Healthcare'), 'IDXX': ('XLV', 'Healthcare'), 'INSM': ('XLV', 'Healthcare'),
    'COP': ('XLE', 'Energy'), 'SLB': ('XLE', 'Energy'), 'XOM': ('XLE', 'Energy'),
    'MPC': ('XLE', 'Energy'), 'EOG': ('XLE', 'Energy'),
    'CAT': ('XLI', 'Industrials'), 'ETN': ('XLI', 'Industrials'), 'LMT': ('XLI', 'Industrials'),
    'ROAD': ('XLI', 'Industrials'),
    'HD': ('XLY', 'Consumer'), 'LOW': ('XLY', 'Consumer'), 'TGT': ('XLY', 'Consumer'),
    'WMT': ('XLY', 'Consumer'),
    'TMUS': ('XLC', 'Communication'), 'CMCSA': ('XLC', 'Communication'), 'META': ('XLC', 'Communication'),
    'APD': ('XLB', 'Materials'), 'ECL': ('XLB', 'Materials'), 'MLM': ('XLB', 'Materials'),
    'SHW': ('XLB', 'Materials'),
    'EXR': ('XLRE', 'Real Estate'),
    'NEE': ('XLU', 'Utilities')
}


def calculate_rsi(prices, period=14):
    """Calculate RSI."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def count_consecutive_down_days(df, position):
    """Count consecutive down days before a given position."""
    count = 0
    for i in range(position - 1, -1, -1):
        if df['Close'].iloc[i] < df['Open'].iloc[i]:
            count += 1
        else:
            break
    return count


def get_volume_ratio(df, position, lookback=20):
    """Calculate volume ratio (current / 20-day average)."""
    if position < lookback:
        return 1.0
    avg_volume = df['Volume'].iloc[position-lookback:position].mean()
    if avg_volume == 0:
        return 1.0
    return df['Volume'].iloc[position] / avg_volume


def detect_regime(spy_df, position, lookback=20):
    """Simple regime detection based on SPY volatility and trend."""
    if position < lookback:
        return 'choppy'

    # Get recent returns
    returns = spy_df['Close'].pct_change().iloc[position-lookback:position]
    volatility = returns.std() * np.sqrt(252)
    trend = (spy_df['Close'].iloc[position] / spy_df['Close'].iloc[position-lookback] - 1)

    if volatility > 0.25:  # High volatility
        return 'volatile'
    elif trend < -0.05:  # Down more than 5%
        return 'bear'
    elif trend > 0.05:  # Up more than 5%
        return 'bull'
    else:
        return 'choppy'


def get_sector_momentum(sector_etf, sector_data, position, lookback=5):
    """Calculate sector momentum (5-day return)."""
    if sector_etf not in sector_data:
        return 0.0

    df = sector_data[sector_etf]
    if position >= len(df) or position < lookback:
        return 0.0

    try:
        return (df['Close'].iloc[position] / df['Close'].iloc[position-lookback] - 1) * 100
    except:
        return 0.0


def run_enhanced_formula_backtest(days_back=365):
    """Run comprehensive backtest comparing baseline vs enhanced formula."""

    print('=' * 70)
    print('ENHANCED FORMULA BACKTEST')
    print('=' * 70)
    print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'Period: {days_back} days')
    print()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 50)

    # Initialize enhanced calculator
    calc = EnhancedSignalCalculator()

    # Fetch SPY for regime detection
    print('Fetching SPY for regime detection...')
    spy = yf.Ticker('SPY')
    spy_df = spy.history(start=start_date, end=end_date)

    # Fetch sector ETFs for sector momentum
    print('Fetching sector ETFs...')
    sector_data = {}
    sector_etfs = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLC', 'XLB', 'XLRE', 'XLU']
    for etf in sector_etfs:
        try:
            data = yf.Ticker(etf).history(start=start_date, end=end_date)
            if len(data) > 20:
                sector_data[etf] = data
        except:
            pass

    # Collect all signals
    signals = []

    print(f'Analyzing {len(TICKERS)} stocks...')

    for i, ticker in enumerate(TICKERS):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if len(df) < 50:
                continue

            # Calculate RSI
            df['RSI'] = calculate_rsi(df['Close'])

            # Get stock sector
            sector_info = STOCK_SECTOR.get(ticker, ('XLK', 'Unknown'))
            sector_etf = sector_info[0]
            sector_name = sector_info[1]

            # Find RSI < 35 signals
            signal_indices = df[df['RSI'] < 35].index

            for idx in signal_indices:
                try:
                    pos = df.index.get_loc(idx)

                    # Need forward data
                    if pos + 5 > len(df) or pos < 20:
                        continue

                    # Get spy position for same date
                    try:
                        spy_pos = spy_df.index.get_loc(idx)
                    except:
                        spy_pos = 50  # Fallback

                    # Get sector momentum
                    try:
                        sector_df = sector_data.get(sector_etf)
                        if sector_df is not None:
                            sector_pos = sector_df.index.get_loc(idx)
                            sector_momentum = get_sector_momentum(sector_etf, sector_data, sector_pos)
                        else:
                            sector_momentum = 0.0
                    except:
                        sector_momentum = 0.0

                    entry_price = df['Close'].iloc[pos]
                    base_rsi = df['RSI'].iloc[pos]

                    # Calculate base signal strength (simple RSI-based)
                    base_strength = max(0, 100 - (base_rsi * 2))  # RSI 35 -> strength 30, RSI 25 -> strength 50

                    # Get all factors
                    consecutive_down = count_consecutive_down_days(df, pos)
                    volume_ratio = get_volume_ratio(df, pos)
                    regime = detect_regime(spy_df, spy_pos) if spy_pos > 20 else 'choppy'

                    # Calculate enhanced strength
                    adj = calc.calculate_enhanced_strength(
                        ticker=ticker,
                        base_strength=base_strength,
                        regime=regime,
                        sector_name=sector_name,
                        sector_momentum=sector_momentum,
                        signal_date=idx.to_pydatetime(),
                        consecutive_down_days=consecutive_down,
                        volume_ratio=volume_ratio
                    )

                    # Calculate forward returns
                    exit_price_2d = df['Close'].iloc[min(pos + 2, len(df)-1)]
                    ret_2d = ((exit_price_2d / entry_price) - 1) * 100

                    exit_price_3d = df['Close'].iloc[min(pos + 3, len(df)-1)]
                    ret_3d = ((exit_price_3d / entry_price) - 1) * 100

                    future = df.iloc[pos:min(pos+5, len(df))]
                    max_gain = ((future['High'].max() / entry_price) - 1) * 100
                    max_loss = ((future['Low'].min() / entry_price) - 1) * 100

                    signals.append({
                        'ticker': ticker,
                        'date': idx.strftime('%Y-%m-%d'),
                        'day_of_week': idx.dayofweek,
                        'rsi': base_rsi,
                        'base_strength': base_strength,
                        'enhanced_strength': adj.final_strength,
                        'regime': regime,
                        'sector': sector_name,
                        'sector_momentum': sector_momentum,
                        'consecutive_down': consecutive_down,
                        'volume_ratio': volume_ratio,
                        'stock_tier': adj.stock_tier,
                        'ret_2d': ret_2d,
                        'ret_3d': ret_3d,
                        'max_gain': max_gain,
                        'max_loss': max_loss,
                        'win_2d': ret_2d > 0,
                        'hit_target': max_gain >= 2.0
                    })

                except:
                    continue

            if (i + 1) % 10 == 0:
                print(f'  Processed {i+1}/{len(TICKERS)} stocks ({len(signals)} signals)')

        except Exception as e:
            continue

    if not signals:
        print('No signals found!')
        return None

    df = pd.DataFrame(signals)
    print()
    print(f'Total signals collected: {len(df)}')
    print()

    # Analyze baseline (all signals)
    print('=' * 70)
    print('BASELINE PERFORMANCE (ALL RSI < 35 SIGNALS)')
    print('=' * 70)
    print()

    baseline_stats = {
        'total_signals': len(df),
        'win_rate': df['win_2d'].mean() * 100,
        'hit_rate': df['hit_target'].mean() * 100,
        'avg_ret_2d': df['ret_2d'].mean(),
        'avg_ret_3d': df['ret_3d'].mean(),
        'avg_max_gain': df['max_gain'].mean(),
        'avg_max_loss': df['max_loss'].mean()
    }

    print(f"Total Signals: {baseline_stats['total_signals']}")
    print(f"Win Rate (2d): {baseline_stats['win_rate']:.1f}%")
    print(f"Hit Rate (2%): {baseline_stats['hit_rate']:.1f}%")
    print(f"Avg 2-day Return: {baseline_stats['avg_ret_2d']:+.3f}%")
    print(f"Avg Max Gain: {baseline_stats['avg_max_gain']:+.2f}%")
    print(f"Avg Max Loss: {baseline_stats['avg_max_loss']:+.2f}%")
    print()

    # Test different enhanced strength thresholds
    print('=' * 70)
    print('ENHANCED FORMULA PERFORMANCE BY THRESHOLD')
    print('=' * 70)
    print()

    threshold_results = []

    for threshold in [30, 40, 50, 60, 70, 80]:
        filtered = df[df['enhanced_strength'] >= threshold]
        if len(filtered) >= 10:
            result = {
                'threshold': threshold,
                'signals': len(filtered),
                'win_rate': filtered['win_2d'].mean() * 100,
                'hit_rate': filtered['hit_target'].mean() * 100,
                'avg_ret_2d': filtered['ret_2d'].mean(),
                'avg_ret_3d': filtered['ret_3d'].mean()
            }
            threshold_results.append(result)

            improvement_win = result['win_rate'] - baseline_stats['win_rate']
            improvement_ret = result['avg_ret_2d'] - baseline_stats['avg_ret_2d']

            print(f"Threshold >= {threshold}:")
            print(f"  Signals: {result['signals']} ({len(filtered)/len(df)*100:.0f}% of total)")
            print(f"  Win Rate: {result['win_rate']:.1f}% ({improvement_win:+.1f}% vs baseline)")
            print(f"  Hit Rate: {result['hit_rate']:.1f}%")
            print(f"  Avg Return: {result['avg_ret_2d']:+.3f}% ({improvement_ret:+.3f}% vs baseline)")
            print()

    # Find optimal threshold
    print('=' * 70)
    print('OPTIMAL THRESHOLD ANALYSIS')
    print('=' * 70)
    print()

    best_threshold = None
    best_score = -float('inf')

    for result in threshold_results:
        if result['signals'] >= 20:  # Need minimum sample size
            # Score = win_rate * 0.4 + hit_rate * 0.3 + avg_return * 10
            score = result['win_rate'] * 0.4 + result['hit_rate'] * 0.3 + result['avg_ret_2d'] * 10
            if score > best_score:
                best_score = score
                best_threshold = result

    if best_threshold:
        print(f"RECOMMENDED THRESHOLD: {best_threshold['threshold']}")
        print(f"  Expected Win Rate: {best_threshold['win_rate']:.1f}%")
        print(f"  Expected Hit Rate: {best_threshold['hit_rate']:.1f}%")
        print(f"  Expected Return per Trade: {best_threshold['avg_ret_2d']:+.3f}%")
        print(f"  Signal Frequency: {best_threshold['signals']} signals / {days_back} days = {best_threshold['signals']/days_back:.1f} per day")

    # Analyze by regime
    print()
    print('=' * 70)
    print('PERFORMANCE BY MARKET REGIME')
    print('=' * 70)
    print()

    regime_results = {}
    for regime in ['volatile', 'bear', 'choppy', 'bull']:
        regime_df = df[df['regime'] == regime]
        if len(regime_df) >= 10:
            result = {
                'signals': len(regime_df),
                'win_rate': regime_df['win_2d'].mean() * 100,
                'avg_ret_2d': regime_df['ret_2d'].mean()
            }
            regime_results[regime] = result

            # Compare high vs low enhanced strength in this regime
            high_strength = regime_df[regime_df['enhanced_strength'] >= 60]
            low_strength = regime_df[regime_df['enhanced_strength'] < 60]

            print(f"{regime.upper()} REGIME (n={result['signals']}):")
            print(f"  All signals: {result['win_rate']:.1f}% win, {result['avg_ret_2d']:+.3f}% avg")
            if len(high_strength) >= 5:
                print(f"  High strength (>=60): {high_strength['win_2d'].mean()*100:.1f}% win, {high_strength['ret_2d'].mean():+.3f}% avg")
            if len(low_strength) >= 5:
                print(f"  Low strength (<60): {low_strength['win_2d'].mean()*100:.1f}% win, {low_strength['ret_2d'].mean():+.3f}% avg")
            print()

    # Analyze by stock tier
    print('=' * 70)
    print('PERFORMANCE BY STOCK TIER')
    print('=' * 70)
    print()

    tier_results = {}
    for tier in ['elite', 'strong', 'average', 'weak', 'avoid']:
        tier_df = df[df['stock_tier'] == tier]
        if len(tier_df) >= 10:
            result = {
                'signals': len(tier_df),
                'win_rate': tier_df['win_2d'].mean() * 100,
                'avg_ret_2d': tier_df['ret_2d'].mean()
            }
            tier_results[tier] = result

            print(f"{tier.upper()} TIER (n={result['signals']}):")
            print(f"  Win Rate: {result['win_rate']:.1f}%")
            print(f"  Avg Return: {result['avg_ret_2d']:+.3f}%")

    # Calculate formula effectiveness
    print()
    print('=' * 70)
    print('FORMULA EFFECTIVENESS')
    print('=' * 70)
    print()

    # Compare top 25% enhanced strength vs bottom 25%
    strength_75 = df['enhanced_strength'].quantile(0.75)
    strength_25 = df['enhanced_strength'].quantile(0.25)

    top_quartile = df[df['enhanced_strength'] >= strength_75]
    bottom_quartile = df[df['enhanced_strength'] <= strength_25]

    if len(top_quartile) >= 10 and len(bottom_quartile) >= 10:
        top_win = top_quartile['win_2d'].mean() * 100
        top_ret = top_quartile['ret_2d'].mean()
        bottom_win = bottom_quartile['win_2d'].mean() * 100
        bottom_ret = bottom_quartile['ret_2d'].mean()

        print(f"Top 25% Enhanced Strength (>={strength_75:.0f}):")
        print(f"  Signals: {len(top_quartile)}")
        print(f"  Win Rate: {top_win:.1f}%")
        print(f"  Avg Return: {top_ret:+.3f}%")
        print()
        print(f"Bottom 25% Enhanced Strength (<={strength_25:.0f}):")
        print(f"  Signals: {len(bottom_quartile)}")
        print(f"  Win Rate: {bottom_win:.1f}%")
        print(f"  Avg Return: {bottom_ret:+.3f}%")
        print()

        edge = top_ret - bottom_ret
        print(f">>> EDGE FROM ENHANCED FORMULA: {edge:+.3f}% per trade <<<")
        print(f">>> WIN RATE IMPROVEMENT: {top_win - bottom_win:+.1f}% <<<")

    # Summary
    print()
    print('=' * 70)
    print('BACKTEST SUMMARY')
    print('=' * 70)
    print()

    if best_threshold:
        improvement = best_threshold['avg_ret_2d'] - baseline_stats['avg_ret_2d']
        print(f"Baseline avg return: {baseline_stats['avg_ret_2d']:+.3f}%")
        print(f"Enhanced avg return (threshold {best_threshold['threshold']}): {best_threshold['avg_ret_2d']:+.3f}%")
        print(f"IMPROVEMENT: {improvement:+.3f}% per trade")
        print()

        # Annualized impact estimate
        trades_per_year = best_threshold['signals']
        annual_edge = trades_per_year * improvement
        print(f"Estimated trades/year: {trades_per_year}")
        print(f"Estimated annual edge: {annual_edge:+.2f}%")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'days_analyzed': days_back,
        'total_signals': len(df),
        'baseline': baseline_stats,
        'threshold_analysis': threshold_results,
        'regime_analysis': regime_results,
        'tier_analysis': tier_results,
        'recommended_threshold': best_threshold['threshold'] if best_threshold else 50,
        'edge_from_formula': edge if 'edge' in locals() else 0
    }

    output_path = Path('data/research/enhanced_formula_backtest.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f'Results saved to {output_path}')

    return results


if __name__ == '__main__':
    run_enhanced_formula_backtest(days_back=365)

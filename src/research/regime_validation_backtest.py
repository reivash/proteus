"""
Regime-Aware Validation Backtest

Tests the regime-specific adjustments to verify they improve performance.
Compares:
1. Baseline: No regime adjustments
2. Regime-Aware: With regime-specific threshold/position adjustments

Uses the updated smart_scanner with config-based regime parameters.
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

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.stock_config_loader import get_loader


def calculate_regime(spy_df, vix_df, date):
    """Determine regime for a specific date."""
    try:
        spy_slice = spy_df.loc[:date].tail(50)
        vix_slice = vix_df.loc[:date].tail(5)

        if len(spy_slice) < 50 or len(vix_slice) < 1:
            return 'choppy', 0, 0

        vix_level = vix_slice['Close'].iloc[-1]
        spy_close = spy_slice['Close']
        trend_20d = ((spy_close.iloc[-1] / spy_close.iloc[-20]) - 1) * 100

        if vix_level > 30:
            return 'volatile', vix_level, trend_20d
        elif trend_20d > 3:
            return 'bull', vix_level, trend_20d
        elif trend_20d < -3:
            return 'bear', vix_level, trend_20d
        else:
            return 'choppy', vix_level, trend_20d
    except:
        return 'choppy', 0, 0


def calculate_rsi(prices, period=14):
    """Calculate RSI."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def run_validation_backtest(days_back=180):
    """Run validation backtest comparing baseline vs regime-aware."""

    print('=' * 70)
    print('REGIME-AWARE VALIDATION BACKTEST')
    print('=' * 70)
    print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'Period: {days_back} days')
    print()

    config_loader = get_loader()

    # Fetch market data
    print('Fetching market data...')
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 50)

    spy = yf.Ticker('SPY').history(start=start_date, end=end_date)
    vix = yf.Ticker('^VIX').history(start=start_date, end=end_date)

    # Stock universe
    tickers = ['NVDA', 'V', 'MA', 'AVGO', 'AXP', 'KLAC', 'ORCL', 'MRVL', 'ABBV', 'SYK',
               'EOG', 'TXN', 'GILD', 'INTU', 'MSFT', 'QCOM', 'JPM', 'JNJ', 'PFE', 'WMT',
               'AMAT', 'ADI', 'NOW', 'MLM', 'IDXX', 'EXR', 'ROAD', 'INSM', 'SCHW', 'AIG',
               'USB', 'CVS', 'LOW', 'LMT', 'COP', 'SLB', 'APD', 'MS', 'PNC', 'CRM',
               'ADBE', 'TGT', 'CAT', 'XOM', 'MPC', 'ECL', 'NEE', 'HCA', 'CMCSA', 'TMUS',
               'META', 'ETN', 'HD', 'SHW']

    # Results
    baseline_results = []
    regime_aware_results = []

    print(f'Analyzing {len(tickers)} stocks...')
    print()

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if len(df) < 50:
                continue

            # Calculate indicators
            df['RSI'] = calculate_rsi(df['Close'])
            df['SMA20'] = df['Close'].rolling(20).mean()
            df['Z_Score'] = (df['Close'] - df['SMA20']) / df['Close'].rolling(20).std()

            # Find signals (RSI < 35)
            signals = df[df['RSI'] < 35].copy()

            for idx in signals.index:
                regime, vix_level, trend = calculate_regime(spy, vix, idx)
                regime_params = config_loader.get_regime_params(regime)

                pos = df.index.get_loc(idx)
                if pos + 5 > len(df):
                    continue

                entry_price = df['Close'].iloc[pos]
                rsi_val = df['RSI'].iloc[pos]
                z_score = df['Z_Score'].iloc[pos]

                # Calculate raw signal strength
                rsi_strength = max(0, (35 - rsi_val) * 2)  # 0-30
                z_strength = max(0, min(40, abs(z_score) * 15))  # 0-40
                raw_strength = 50 + rsi_strength + z_strength  # 50-120
                raw_strength = min(100, raw_strength)

                # Baseline thresholds
                baseline_threshold = 50
                baseline_position_mult = 1.0

                # Regime-aware thresholds
                regime_threshold_adj = regime_params.get('threshold_adjustment', 0)
                regime_threshold = baseline_threshold + regime_threshold_adj
                regime_position_mult = regime_params.get('position_multiplier', 1.0)

                # Check tier adjustments
                tier_adj, tier_position = config_loader.get_tier_adjustment(ticker)

                # Regime-aware adjusted strength
                if regime == 'volatile':
                    adjusted_strength = raw_strength * 1.15
                elif regime == 'bear':
                    adjusted_strength = raw_strength * 1.10
                elif regime == 'bull':
                    adjusted_strength = raw_strength * 0.85
                else:
                    adjusted_strength = raw_strength

                # Elite stock boost
                elite_stocks = config_loader.get_regime_elite_stocks(regime)
                if ticker in elite_stocks:
                    adjusted_strength *= 1.05

                # Calculate outcomes
                exit_price_2d = df['Close'].iloc[min(pos + 2, len(df)-1)]
                ret_2d = ((exit_price_2d / entry_price) - 1) * 100

                future = df.iloc[pos:min(pos+5, len(df))]
                max_gain = ((future['High'].max() / entry_price) - 1) * 100
                max_loss = ((future['Low'].min() / entry_price) - 1) * 100

                # Baseline: signal passes if raw_strength >= 50
                if raw_strength >= baseline_threshold:
                    baseline_results.append({
                        'ticker': ticker,
                        'date': idx.strftime('%Y-%m-%d'),
                        'regime': regime,
                        'strength': raw_strength,
                        'ret_2d': ret_2d,
                        'max_gain': max_gain,
                        'max_loss': max_loss,
                        'win': ret_2d > 0,
                        'hit_target': max_gain >= 2.0,
                        'position_mult': baseline_position_mult
                    })

                # Regime-aware: signal passes if adjusted_strength >= regime_threshold
                if adjusted_strength >= regime_threshold:
                    regime_aware_results.append({
                        'ticker': ticker,
                        'date': idx.strftime('%Y-%m-%d'),
                        'regime': regime,
                        'raw_strength': raw_strength,
                        'adjusted_strength': adjusted_strength,
                        'ret_2d': ret_2d,
                        'max_gain': max_gain,
                        'max_loss': max_loss,
                        'win': ret_2d > 0,
                        'hit_target': max_gain >= 2.0,
                        'position_mult': regime_position_mult * tier_position
                    })

        except Exception as e:
            continue

    # Analyze results
    print('=' * 70)
    print('COMPARISON RESULTS')
    print('=' * 70)
    print()

    def analyze_results(results, name):
        if not results:
            print(f'{name}: No signals')
            return None

        df = pd.DataFrame(results)
        n = len(df)
        win_rate = df['win'].mean() * 100
        hit_rate = df['hit_target'].mean() * 100
        avg_ret = df['ret_2d'].mean()
        avg_max_gain = df['max_gain'].mean()
        avg_max_loss = df['max_loss'].mean()

        # Position-weighted return (simulate position sizing)
        if 'position_mult' in df.columns:
            weighted_ret = (df['ret_2d'] * df['position_mult']).mean()
        else:
            weighted_ret = avg_ret

        # Calculate Sharpe-like metric
        if df['ret_2d'].std() > 0:
            sharpe = avg_ret / df['ret_2d'].std()
        else:
            sharpe = 0

        # Expectancy
        wins = df[df['ret_2d'] > 0]['ret_2d']
        losses = df[df['ret_2d'] <= 0]['ret_2d']
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

        print(f'{name}:')
        print(f'  Total signals: {n}')
        print(f'  Win rate: {win_rate:.1f}%')
        print(f'  Hit rate (2% target): {hit_rate:.1f}%')
        print(f'  Avg 2-day return: {avg_ret:+.2f}%')
        print(f'  Position-weighted return: {weighted_ret:+.2f}%')
        print(f'  Avg max gain: {avg_max_gain:+.2f}%')
        print(f'  Avg max loss: {avg_max_loss:-.2f}%')
        print(f'  Sharpe ratio: {sharpe:.3f}')
        print(f'  Expectancy: {expectancy:+.3f}%')
        print()

        return {
            'signals': n,
            'win_rate': round(win_rate, 1),
            'hit_rate': round(hit_rate, 1),
            'avg_ret': round(avg_ret, 2),
            'weighted_ret': round(weighted_ret, 2),
            'sharpe': round(sharpe, 3),
            'expectancy': round(expectancy, 3)
        }

    baseline_stats = analyze_results(baseline_results, 'BASELINE (no regime adjustments)')
    regime_aware_stats = analyze_results(regime_aware_results, 'REGIME-AWARE (with adjustments)')

    # Regime breakdown for regime-aware
    if regime_aware_results:
        print('=' * 70)
        print('REGIME-AWARE BREAKDOWN BY REGIME')
        print('=' * 70)
        print()

        df = pd.DataFrame(regime_aware_results)
        regime_breakdown = {}

        for regime in ['volatile', 'bear', 'choppy', 'bull']:
            regime_df = df[df['regime'] == regime]
            if len(regime_df) > 0:
                regime_breakdown[regime] = {
                    'signals': len(regime_df),
                    'win_rate': round(regime_df['win'].mean() * 100, 1),
                    'hit_rate': round(regime_df['hit_target'].mean() * 100, 1),
                    'avg_ret': round(regime_df['ret_2d'].mean(), 2),
                    'weighted_ret': round((regime_df['ret_2d'] * regime_df['position_mult']).mean(), 2)
                }
                print(f'{regime.upper()}:')
                print(f'  Signals: {regime_breakdown[regime]["signals"]}')
                print(f'  Win rate: {regime_breakdown[regime]["win_rate"]}%')
                print(f'  Avg return: {regime_breakdown[regime]["avg_ret"]:+.2f}%')
                print(f'  Position-weighted: {regime_breakdown[regime]["weighted_ret"]:+.2f}%')
                print()

    # Improvement summary
    print('=' * 70)
    print('IMPROVEMENT SUMMARY')
    print('=' * 70)

    if baseline_stats and regime_aware_stats:
        win_rate_diff = regime_aware_stats['win_rate'] - baseline_stats['win_rate']
        ret_diff = regime_aware_stats['avg_ret'] - baseline_stats['avg_ret']
        weighted_ret_diff = regime_aware_stats['weighted_ret'] - baseline_stats['weighted_ret']
        sharpe_diff = regime_aware_stats['sharpe'] - baseline_stats['sharpe']
        expectancy_diff = regime_aware_stats['expectancy'] - baseline_stats['expectancy']

        print()
        print(f'Win rate change: {win_rate_diff:+.1f} percentage points')
        print(f'Avg return change: {ret_diff:+.2f}%')
        print(f'Position-weighted return change: {weighted_ret_diff:+.2f}%')
        print(f'Sharpe ratio change: {sharpe_diff:+.3f}')
        print(f'Expectancy change: {expectancy_diff:+.3f}%')
        print()

        if expectancy_diff > 0:
            print('>>> REGIME-AWARE STRATEGY SHOWS IMPROVEMENT <<<')
        else:
            print('>>> NO IMPROVEMENT FROM REGIME ADJUSTMENTS <<<')

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'days_analyzed': days_back,
        'baseline': baseline_stats,
        'regime_aware': regime_aware_stats,
        'improvement': {
            'win_rate': win_rate_diff if baseline_stats and regime_aware_stats else 0,
            'avg_ret': ret_diff if baseline_stats and regime_aware_stats else 0,
            'weighted_ret': weighted_ret_diff if baseline_stats and regime_aware_stats else 0,
            'sharpe': sharpe_diff if baseline_stats and regime_aware_stats else 0,
            'expectancy': expectancy_diff if baseline_stats and regime_aware_stats else 0
        }
    }

    output_path = Path('data/research/regime_validation_backtest.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f'Results saved to {output_path}')

    return results


if __name__ == '__main__':
    run_validation_backtest(days_back=180)

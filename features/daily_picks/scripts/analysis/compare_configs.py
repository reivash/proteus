"""
Compare Full vs Simplified Modifier Configs
Tests whether 15 high-impact modifiers can match 115 modifiers
"""
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import yfinance as yf

# Add src to path
import sys
sys.path.insert(0, 'src')


def get_stock_data(ticker: str, period: str = '2y') -> dict:
    """Fetch stock data for backtesting."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if len(df) < 50:
            return None
        return df
    except:
        return None


def calculate_indicators(df) -> dict:
    """Calculate technical indicators for a given date."""
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values

    # RSI
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(14)/14, mode='valid')[-1]
    avg_loss = np.convolve(loss, np.ones(14)/14, mode='valid')[-1]
    rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))

    # SMA200
    sma200 = np.mean(close[-200:]) if len(close) >= 200 else np.mean(close)
    sma200_distance = (close[-1] / sma200 - 1) * 100

    # Consecutive down days
    down_days = 0
    for i in range(len(close)-1, 0, -1):
        if close[i] < close[i-1]:
            down_days += 1
        else:
            break

    # Drawdown from 20d high
    high_20d = np.max(high[-20:])
    drawdown = (close[-1] / high_20d - 1) * 100

    # Volume ratio
    vol_avg = np.mean(volume[-20:])
    vol_ratio = volume[-1] / (vol_avg + 1)

    # ATR
    tr = np.maximum(high[-14:] - low[-14:],
                    np.abs(high[-14:] - np.roll(close[-14:], 1)))
    atr = np.mean(tr) / close[-1] * 100

    return {
        'rsi': rsi,
        'sma200_distance': sma200_distance,
        'consecutive_down': down_days,
        'drawdown': drawdown,
        'volume_ratio': vol_ratio,
        'atr': atr,
        'close': close[-1]
    }


class PenaltiesOnlyCalculator:
    """Penalties-only calculator - NO positive boosts."""

    def __init__(self, config_path: str = 'config/penalties_only_config.json'):
        with open(config_path) as f:
            self.config = json.load(f)
        self.penalties = self.config.get('signal_boosts', {})
        self.tiers = self.config.get('stock_tiers', {})

    def get_tier(self, ticker: str) -> str:
        for tier_name, tier_data in self.tiers.items():
            if tier_name.startswith('_'):
                continue
            if isinstance(tier_data, dict) and ticker in tier_data.get('tickers', []):
                return tier_name
        return 'average'

    def get_tier_multiplier(self, tier: str) -> float:
        return self.tiers.get(tier, {}).get('signal_multiplier', 1.0)

    def calculate(self, ticker: str, base_signal: float, regime: str,
                  is_monday: bool, is_friday: bool, consecutive_down: int,
                  volume_ratio: float, sma200_distance: float, drawdown: float) -> float:
        """Calculate final signal with penalties only."""

        applied = {}

        # 1. Day of week penalties
        dow = self.penalties.get('day_of_week_penalties', {})
        if is_friday and dow.get('friday_penalty', {}).get('enabled'):
            applied['friday'] = dow['friday_penalty'].get('penalty', -5)

        # 2. Base signal quality (CRITICAL)
        bsq = self.penalties.get('base_signal_quality', {})
        if base_signal < 55 and bsq.get('weak_base_penalty', {}).get('enabled'):
            applied['weak_base'] = bsq['weak_base_penalty'].get('penalty', -15)
        elif 55 <= base_signal < 60 and bsq.get('marginal_base_penalty', {}).get('enabled'):
            applied['marginal_base'] = bsq['marginal_base_penalty'].get('penalty', -8)

        # 3. Falling knife
        fk = self.penalties.get('falling_knife', {})
        if consecutive_down >= 7 and fk.get('seven_plus_down', {}).get('enabled'):
            applied['7+_down'] = fk['seven_plus_down'].get('penalty', -10)
        elif 5 <= consecutive_down <= 6 and fk.get('five_six_down', {}).get('enabled'):
            applied['5-6_down'] = fk['five_six_down'].get('penalty', -5)

        # 4. Regime penalties
        reg = self.penalties.get('regime_penalties', {})
        if regime == 'bull' and reg.get('bull_regime', {}).get('enabled'):
            applied['bull'] = reg['bull_regime'].get('penalty', -6)

        # 5. Volume penalties
        vol = self.penalties.get('volume_penalties', {})
        if volume_ratio < 0.6 and vol.get('low_volume', {}).get('enabled'):
            applied['low_vol'] = vol['low_volume'].get('penalty', -5)

        # 6. Technical penalties
        tech = self.penalties.get('technical_penalties', {})
        if abs(sma200_distance) > 15 and tech.get('far_from_sma200', {}).get('enabled'):
            applied['far_sma200'] = tech['far_from_sma200'].get('penalty', -4)
        if drawdown > -3 and tech.get('minimal_drawdown', {}).get('enabled'):
            applied['min_dd'] = tech['minimal_drawdown'].get('penalty', -5)

        # Apply tier multiplier
        tier = self.get_tier(ticker)
        tier_mult = self.get_tier_multiplier(tier)

        # Calculate final signal (ONLY penalties, no positive boosts)
        total_penalty = sum(applied.values())
        final_signal = (base_signal * tier_mult) + total_penalty

        return final_signal, applied


class SimplifiedCalculator:
    """Simplified signal calculator with only 15 modifiers."""

    def __init__(self, config_path: str = 'config/simplified_config.json'):
        with open(config_path) as f:
            self.config = json.load(f)
        self.boosts = self.config.get('signal_boosts', {})
        self.tiers = self.config.get('stock_tiers', {})

    def get_tier(self, ticker: str) -> str:
        for tier_name, tier_data in self.tiers.items():
            if tier_name.startswith('_'):
                continue
            if isinstance(tier_data, dict) and ticker in tier_data.get('tickers', []):
                return tier_name
        return 'average'

    def get_tier_multiplier(self, tier: str) -> float:
        return self.tiers.get(tier, {}).get('signal_multiplier', 1.0)

    def calculate(self, ticker: str, base_signal: float, regime: str,
                  is_monday: bool, is_friday: bool, consecutive_down: int,
                  volume_ratio: float, sma200_distance: float, drawdown: float) -> float:
        """Calculate final signal with simplified modifiers."""

        applied = {}

        # 1. Day of week
        dow = self.boosts.get('day_of_week', {})
        if is_monday and dow.get('monday', {}).get('enabled'):
            applied['monday'] = dow['monday'].get('boost', 3)
        if is_friday and dow.get('friday', {}).get('enabled'):
            applied['friday'] = dow['friday'].get('boost', -4)

        # 2. Consecutive down days
        cdd = self.boosts.get('consecutive_down_days', {})
        if 2 <= consecutive_down <= 3 and cdd.get('two_to_three_down', {}).get('enabled'):
            applied['2-3_down'] = cdd['two_to_three_down'].get('boost', 5)
        if consecutive_down >= 7 and cdd.get('seven_plus_down', {}).get('enabled'):
            applied['7+_down'] = cdd['seven_plus_down'].get('boost', -6)

        # 3. Base signal quality (CRITICAL)
        bsq = self.boosts.get('base_signal_quality', {})
        if base_signal < 55 and bsq.get('weak_base_signal_penalty', {}).get('enabled'):
            applied['weak_base'] = bsq['weak_base_signal_penalty'].get('penalty', -10)
        elif 55 <= base_signal < 60 and bsq.get('marginal_base_signal_penalty', {}).get('enabled'):
            applied['marginal_base'] = bsq['marginal_base_signal_penalty'].get('penalty', -5)

        # 4. Volume
        vol = self.boosts.get('volume', {})
        if 0.8 <= volume_ratio <= 1.2 and vol.get('normal_volume', {}).get('enabled'):
            applied['normal_vol'] = vol['normal_volume'].get('boost', 2)
        if volume_ratio >= 2.0 and vol.get('exhaustion_volume', {}).get('enabled'):
            applied['exhaustion'] = vol['exhaustion_volume'].get('boost', 3)

        # 5. Technical
        tech = self.boosts.get('technical', {})
        if abs(sma200_distance) <= 5.0 and tech.get('near_sma200', {}).get('enabled'):
            applied['near_sma200'] = tech['near_sma200'].get('boost', 3)
        if -15 <= drawdown <= -8 and tech.get('bounce_day_drawdown', {}).get('enabled'):
            applied['bounce'] = tech['bounce_day_drawdown'].get('boost', 4)

        # 6. Regime
        reg = self.boosts.get('regime', {})
        if regime == 'volatile' and reg.get('volatile_regime', {}).get('enabled'):
            applied['volatile'] = reg['volatile_regime'].get('boost', 4)
        if regime == 'bull' and reg.get('bull_regime', {}).get('enabled'):
            applied['bull'] = reg['bull_regime'].get('boost', -5)

        # 7. High conviction
        conv = self.boosts.get('conviction', {})
        if base_signal >= 70 and conv.get('high_conviction', {}).get('enabled'):
            applied['high_conv'] = conv['high_conviction'].get('boost', 3)

        # Apply tier multiplier
        tier = self.get_tier(ticker)
        tier_mult = self.get_tier_multiplier(tier)

        # Calculate final signal
        total_boost = sum(applied.values())
        final_signal = (base_signal * tier_mult) + total_boost

        return final_signal, applied


def backtest_config(tickers: list, calc, threshold: int, config_name: str):
    """Backtest a configuration."""

    all_trades = []

    for ticker in tickers:
        df = get_stock_data(ticker, '2y')
        if df is None or len(df) < 250:
            continue

        tier = calc.get_tier(ticker)

        for i in range(250, len(df) - 5):
            sub_df = df.iloc[:i+1]
            indicators = calculate_indicators(sub_df)

            date = df.index[i]
            dow = date.weekday()

            # Generate base signal
            base_signal = 50 + (30 - indicators['rsi']) * 0.5 + abs(indicators['drawdown']) * 1.5
            base_signal = max(30, min(80, base_signal))

            # Calculate with config
            final_signal, applied = calc.calculate(
                ticker=ticker,
                base_signal=base_signal,
                regime='choppy',
                is_monday=dow == 0,
                is_friday=dow == 4,
                consecutive_down=indicators['consecutive_down'],
                volume_ratio=indicators['volume_ratio'],
                sma200_distance=indicators['sma200_distance'],
                drawdown=indicators['drawdown']
            )

            if final_signal >= threshold:
                entry_price = indicators['close']
                exit_price = df['Close'].iloc[i + 2]
                return_pct = (exit_price / entry_price - 1) * 100
                win = return_pct > 0

                all_trades.append({
                    'ticker': ticker,
                    'date': date,
                    'base': base_signal,
                    'final': final_signal,
                    'boost_total': sum(applied.values()),
                    'return': return_pct,
                    'win': win
                })

    if not all_trades:
        return None

    return {
        'config': config_name,
        'trades': len(all_trades),
        'wins': sum(1 for t in all_trades if t['win']),
        'win_rate': sum(1 for t in all_trades if t['win']) / len(all_trades) * 100,
        'avg_return': np.mean([t['return'] for t in all_trades]),
        'avg_boost': np.mean([t['boost_total'] for t in all_trades]),
        'trades_list': all_trades
    }


def backtest_no_modifiers(tickers: list, threshold: int):
    """Backtest with NO modifiers (baseline)."""

    # Load tier data
    with open('config/simplified_config.json') as f:
        config = json.load(f)
    tiers = config.get('stock_tiers', {})

    def get_tier_mult(ticker):
        for tier_name, tier_data in tiers.items():
            if tier_name.startswith('_'):
                continue
            if isinstance(tier_data, dict) and ticker in tier_data.get('tickers', []):
                return tier_data.get('signal_multiplier', 1.0)
        return 1.0

    all_trades = []

    for ticker in tickers:
        df = get_stock_data(ticker, '2y')
        if df is None or len(df) < 250:
            continue

        tier_mult = get_tier_mult(ticker)

        for i in range(250, len(df) - 5):
            sub_df = df.iloc[:i+1]
            indicators = calculate_indicators(sub_df)

            # Generate base signal
            base_signal = 50 + (30 - indicators['rsi']) * 0.5 + abs(indicators['drawdown']) * 1.5
            base_signal = max(30, min(80, base_signal))

            # Only tier multiplier, no boosts
            final_signal = base_signal * tier_mult

            if final_signal >= threshold:
                entry_price = indicators['close']
                exit_price = df['Close'].iloc[i + 2]
                return_pct = (exit_price / entry_price - 1) * 100
                win = return_pct > 0

                all_trades.append({
                    'ticker': ticker,
                    'return': return_pct,
                    'win': win
                })

    if not all_trades:
        return None

    return {
        'config': 'NO_MODIFIERS',
        'trades': len(all_trades),
        'wins': sum(1 for t in all_trades if t['win']),
        'win_rate': sum(1 for t in all_trades if t['win']) / len(all_trades) * 100,
        'avg_return': np.mean([t['return'] for t in all_trades]),
        'avg_boost': 0
    }


def main():
    print("=" * 70)
    print("CONFIG COMPARISON: Full vs Simplified vs Penalties-Only vs Baseline")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    tickers = [
        'NVDA', 'AVGO', 'MSFT', 'JPM', 'JNJ', 'XOM', 'CAT',
        'ORCL', 'MRVL', 'INSM', 'SCHW', 'COP', 'ETN', 'ABBV'
    ]

    threshold = 80

    print(f"Testing {len(tickers)} stocks at threshold {threshold}")
    print()

    # 1. Test NO MODIFIERS (baseline)
    print("[1/4] Testing NO MODIFIERS (baseline)...")
    result_none = backtest_no_modifiers(tickers, threshold)
    print(f"      {result_none['trades']} trades, {result_none['win_rate']:.1f}% win rate")

    # 2. Test PENALTIES ONLY (new)
    print("[2/4] Testing PENALTIES ONLY (12 penalties, 0 boosts)...")
    calc_penalties = PenaltiesOnlyCalculator('config/penalties_only_config.json')
    result_penalties = backtest_config(tickers, calc_penalties, threshold, 'PENALTIES_ONLY')
    print(f"      {result_penalties['trades']} trades, {result_penalties['win_rate']:.1f}% win rate")

    # 3. Test SIMPLIFIED (15 modifiers)
    print("[3/4] Testing SIMPLIFIED (15 modifiers)...")
    calc_simple = SimplifiedCalculator('config/simplified_config.json')
    result_simple = backtest_config(tickers, calc_simple, threshold, 'SIMPLIFIED_15')
    print(f"      {result_simple['trades']} trades, {result_simple['win_rate']:.1f}% win rate")

    # 4. Test FULL (need to import)
    print("[4/4] Testing FULL (115 modifiers)...")
    from trading.unified_signal_calculator import UnifiedSignalCalculator

    # Wrap the full calculator to match interface
    class FullCalcWrapper:
        def __init__(self):
            self.calc = UnifiedSignalCalculator()

        def get_tier(self, ticker):
            return self.calc.get_tier(ticker)

        def get_tier_multiplier(self, tier):
            return self.calc.get_tier_multiplier(tier)

        def calculate(self, ticker, base_signal, regime, is_monday, is_friday,
                      consecutive_down, volume_ratio, sma200_distance, drawdown):
            result = self.calc.calculate(
                ticker=ticker,
                base_signal=base_signal,
                regime=regime,
                is_monday=is_monday,
                is_tuesday=False,
                is_wednesday=False,
                is_thursday=False,
                is_friday=is_friday,
                consecutive_down_days=consecutive_down,
                rsi_level=50,  # Simplified
                volume_ratio=volume_ratio,
                is_down_day=consecutive_down > 0,
                sector='Technology',
                sector_momentum=0,
                close_position=0.5,
                gap_pct=0,
                sma200_distance=sma200_distance,
                day_range_pct=2.0,
                drawdown_pct=drawdown,
                atr_pct=2.0
            )
            return result.final_signal, result.boosts_applied

    calc_full = FullCalcWrapper()
    result_full = backtest_config(tickers, calc_full, threshold, 'FULL_115')
    print(f"      {result_full['trades']} trades, {result_full['win_rate']:.1f}% win rate")

    # Results comparison
    print()
    print("=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Config':<20} | {'Trades':>8} | {'Win Rate':>10} | {'Avg Return':>12} | {'Avg Boost':>10}")
    print("-" * 70)
    print(f"{'NO MODIFIERS':<20} | {result_none['trades']:>8} | {result_none['win_rate']:>9.1f}% | {result_none['avg_return']:>11.2f}% | {result_none['avg_boost']:>10.1f}")
    print(f"{'PENALTIES ONLY':<20} | {result_penalties['trades']:>8} | {result_penalties['win_rate']:>9.1f}% | {result_penalties['avg_return']:>11.2f}% | {result_penalties['avg_boost']:>10.1f}")
    print(f"{'SIMPLIFIED (15)':<20} | {result_simple['trades']:>8} | {result_simple['win_rate']:>9.1f}% | {result_simple['avg_return']:>11.2f}% | {result_simple['avg_boost']:>10.1f}")
    print(f"{'FULL (115)':<20} | {result_full['trades']:>8} | {result_full['win_rate']:>9.1f}% | {result_full['avg_return']:>11.2f}% | {result_full['avg_boost']:>10.1f}")

    print()
    print("ANALYSIS:")

    # Compare penalties-only vs baseline
    penalties_vs_none_wr = result_penalties['win_rate'] - result_none['win_rate']
    penalties_vs_none_ret = result_penalties['avg_return'] - result_none['avg_return']

    print(f"  Penalties-Only vs Baseline:")
    print(f"    Win rate: {penalties_vs_none_wr:+.1f}pp")
    print(f"    Avg return: {penalties_vs_none_ret:+.2f}%")
    print(f"    Trade count: {result_penalties['trades']} vs {result_none['trades']}")

    # Compare simplified vs full
    simple_vs_full_wr = result_simple['win_rate'] - result_full['win_rate']
    simple_vs_full_ret = result_simple['avg_return'] - result_full['avg_return']

    print(f"  Simplified vs Full:")
    print(f"    Win rate: {simple_vs_full_wr:+.1f}pp")
    print(f"    Avg return: {simple_vs_full_ret:+.2f}%")
    print(f"    Trade count: {result_simple['trades']} vs {result_full['trades']}")

    # Compare simplified vs no modifiers
    simple_vs_none_wr = result_simple['win_rate'] - result_none['win_rate']
    simple_vs_none_ret = result_simple['avg_return'] - result_none['avg_return']

    print(f"  Simplified vs No Modifiers:")
    print(f"    Win rate: {simple_vs_none_wr:+.1f}pp")
    print(f"    Avg return: {simple_vs_none_ret:+.2f}%")

    print()
    print("RECOMMENDATION:")

    # Find best performer
    configs = [
        ('NO_MODIFIERS', result_none),
        ('PENALTIES_ONLY', result_penalties),
        ('SIMPLIFIED', result_simple),
        ('FULL', result_full)
    ]

    # Sort by win rate (quality focus)
    best = max(configs, key=lambda x: x[1]['win_rate'])
    print(f"  BEST BY WIN RATE: {best[0]} ({best[1]['win_rate']:.1f}%)")

    # Decision logic
    if result_penalties['win_rate'] >= result_none['win_rate'] - 1:
        print("  [ADOPT PENALTIES-ONLY] Filters bad trades, maintains quality")
    elif result_none['win_rate'] > result_full['win_rate']:
        print("  [REMOVE ALL MODIFIERS] Baseline outperforms modifier systems")
    else:
        print("  [INVESTIGATE FURTHER] Results inconclusive")

    # Boost analysis
    print()
    print("BOOST ANALYSIS:")
    if result_penalties.get('trades_list'):
        boosts = [t['boost_total'] for t in result_penalties['trades_list']]
        print(f"  Penalties avg: {np.mean(boosts):.1f} (range: {min(boosts):.0f} to {max(boosts):.0f})")
    if result_simple.get('trades_list'):
        boosts = [t['boost_total'] for t in result_simple['trades_list']]
        print(f"  Simplified avg: {np.mean(boosts):.1f} (range: {min(boosts):.0f} to {max(boosts):.0f})")
    if result_full.get('trades_list'):
        boosts = [t['boost_total'] for t in result_full['trades_list']]
        print(f"  Full avg: {np.mean(boosts):.1f} (range: {min(boosts):.0f} to {max(boosts):.0f})")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'threshold': threshold,
        'tickers_tested': len(tickers),
        'no_modifiers': {
            'trades': result_none['trades'],
            'win_rate': result_none['win_rate'],
            'avg_return': result_none['avg_return']
        },
        'penalties_only': {
            'trades': result_penalties['trades'],
            'win_rate': result_penalties['win_rate'],
            'avg_return': result_penalties['avg_return'],
            'avg_boost': result_penalties['avg_boost']
        },
        'simplified_15': {
            'trades': result_simple['trades'],
            'win_rate': result_simple['win_rate'],
            'avg_return': result_simple['avg_return'],
            'avg_boost': result_simple['avg_boost']
        },
        'full_115': {
            'trades': result_full['trades'],
            'win_rate': result_full['win_rate'],
            'avg_return': result_full['avg_return'],
            'avg_boost': result_full['avg_boost']
        }
    }

    Path('data/research').mkdir(parents=True, exist_ok=True)
    with open('data/research/config_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("Results saved to data/research/config_comparison.json")


if __name__ == '__main__':
    main()

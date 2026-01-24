"""
Position Sizing Research - Phase 2 Preparation

This script runs overnight to gather data needed for the PositionSizer class:
1. Kelly Criterion calculations per tier
2. Signal strength vs optimal position size correlation
3. ATR-based volatility sizing analysis
4. Portfolio heat limits and max drawdown study
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
from typing import Dict, List, Tuple
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trading.exit_optimizer import ExitOptimizer, StockTier


TICKERS = [
    'NVDA', 'AVGO', 'MSFT', 'ORCL', 'INTU', 'ADBE', 'CRM', 'NOW', 'AMAT', 'KLAC',
    'MRVL', 'QCOM', 'TXN', 'ADI', 'JPM', 'MS', 'SCHW', 'AXP', 'AIG', 'USB',
    'PNC', 'V', 'MA', 'ABBV', 'SYK', 'GILD', 'PFE', 'JNJ', 'CVS', 'HCA',
    'IDXX', 'INSM', 'COP', 'SLB', 'XOM', 'MPC', 'EOG', 'CAT', 'ETN', 'LMT',
    'ROAD', 'HD', 'LOW', 'TGT', 'WMT', 'TMUS', 'CMCSA', 'META', 'APD', 'ECL',
    'MLM', 'SHW', 'EXR', 'NEE'
]


@dataclass
class TierSizingMetrics:
    """Sizing metrics for a stock tier."""
    tier: str
    trade_count: int
    win_rate: float
    avg_win: float
    avg_loss: float
    kelly_fraction: float
    half_kelly: float
    avg_atr_pct: float  # Average ATR as % of price
    recommended_size: float  # Recommended position size (% of portfolio)


@dataclass
class SignalStrengthBucket:
    """Metrics for a signal strength range."""
    range_label: str
    min_strength: float
    max_strength: float
    trade_count: int
    win_rate: float
    avg_return: float
    kelly_fraction: float
    optimal_size_pct: float


@dataclass
class PositionSizingReport:
    """Complete position sizing research report."""
    timestamp: str
    total_trades: int
    tier_metrics: Dict[str, TierSizingMetrics]
    strength_buckets: List[SignalStrengthBucket]
    portfolio_recommendations: Dict
    volatility_analysis: Dict


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


def fetch_stock_data(days_back: int = 400) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for all tickers."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    stock_data = {}
    print(f"Fetching data for {len(TICKERS)} stocks...")

    for i, ticker in enumerate(TICKERS):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if len(df) >= 50:
                stock_data[ticker] = df
        except:
            pass

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(TICKERS)} fetched")

    print(f"  Got data for {len(stock_data)} stocks")
    return stock_data


def calculate_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate Kelly Criterion fraction.

    Kelly = W - [(1-W) / R]
    Where:
        W = Win rate (probability of winning)
        R = Win/Loss ratio (avg win / avg loss)
    """
    if avg_loss == 0 or avg_win <= 0:
        return 0.0

    W = win_rate
    R = abs(avg_win / avg_loss)

    kelly = W - ((1 - W) / R)
    return max(0, min(kelly, 0.5))  # Cap at 50%


def generate_trades(stock_data: Dict[str, pd.DataFrame],
                   optimizer: ExitOptimizer,
                   days_back: int = 365) -> List[Dict]:
    """Generate trade data from historical signals."""

    trades = []
    cutoff_date = datetime.now() - timedelta(days=days_back)
    cutoff_str = cutoff_date.strftime('%Y-%m-%d')

    print(f"\nGenerating trades from {cutoff_str}...")

    for ticker, df in stock_data.items():
        try:
            df = df.copy()
            df['RSI'] = calculate_rsi(df['Close'])
            df['ATR'] = calculate_atr(df)

            for i in range(20, len(df) - 6):
                if df['RSI'].iloc[i] < 35:
                    signal_date = df.index[i]
                    signal_str = signal_date.strftime('%Y-%m-%d')

                    if signal_str < cutoff_str:
                        continue

                    entry_price = df['Close'].iloc[i]
                    rsi = df['RSI'].iloc[i]
                    atr = df['ATR'].iloc[i]
                    atr_pct = (atr / entry_price) * 100 if entry_price > 0 else 0

                    # Calculate signal strength
                    signal_strength = max(0, min(100, (35 - rsi) * 3 + 50))

                    # Get tier
                    tier = optimizer.get_stock_tier(ticker)

                    # Calculate 2-day return
                    exit_price = df['Close'].iloc[i+2]
                    return_pct = ((exit_price / entry_price) - 1) * 100

                    # Track high and low for potential drawdown
                    max_price = df['High'].iloc[i+1:i+6].max()
                    min_price = df['Low'].iloc[i+1:i+6].min()
                    max_gain = ((max_price / entry_price) - 1) * 100
                    max_loss = ((min_price / entry_price) - 1) * 100

                    trades.append({
                        'ticker': ticker,
                        'date': signal_str,
                        'tier': tier.value,
                        'signal_strength': signal_strength,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return_pct': return_pct,
                        'max_gain': max_gain,
                        'max_loss': max_loss,
                        'atr_pct': atr_pct,
                        'is_win': return_pct > 0
                    })
        except Exception as e:
            continue

    print(f"  Generated {len(trades)} trades")
    return trades


def analyze_by_tier(trades: List[Dict]) -> Dict[str, TierSizingMetrics]:
    """Analyze position sizing metrics by tier."""

    print("\nAnalyzing by tier...")
    tier_metrics = {}

    for tier in ['elite', 'strong', 'average', 'avoid']:
        tier_trades = [t for t in trades if t['tier'] == tier]

        if not tier_trades:
            continue

        wins = [t for t in tier_trades if t['is_win']]
        losses = [t for t in tier_trades if not t['is_win']]

        win_rate = len(wins) / len(tier_trades)
        avg_win = np.mean([t['return_pct'] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t['return_pct'] for t in losses])) if losses else 0

        kelly = calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        half_kelly = kelly / 2

        avg_atr = np.mean([t['atr_pct'] for t in tier_trades])

        # Recommended size based on volatility and Kelly
        # Lower of: half-kelly, or 2% / avg_atr (risk-parity style)
        vol_based_size = min(2.0 / avg_atr, 0.15) if avg_atr > 0 else 0.05
        recommended = min(half_kelly, vol_based_size)

        tier_metrics[tier] = TierSizingMetrics(
            tier=tier,
            trade_count=len(tier_trades),
            win_rate=round(win_rate * 100, 1),
            avg_win=round(avg_win, 3),
            avg_loss=round(avg_loss, 3),
            kelly_fraction=round(kelly * 100, 2),
            half_kelly=round(half_kelly * 100, 2),
            avg_atr_pct=round(avg_atr, 3),
            recommended_size=round(recommended * 100, 2)
        )

        print(f"  {tier.upper()}: {len(tier_trades)} trades, "
              f"win={win_rate*100:.1f}%, kelly={kelly*100:.1f}%, "
              f"recommended={recommended*100:.1f}%")

    return tier_metrics


def analyze_by_signal_strength(trades: List[Dict]) -> List[SignalStrengthBucket]:
    """Analyze position sizing metrics by signal strength buckets."""

    print("\nAnalyzing by signal strength...")
    buckets = []

    ranges = [
        ('Weak (50-60)', 50, 60),
        ('Moderate (60-70)', 60, 70),
        ('Strong (70-80)', 70, 80),
        ('Very Strong (80+)', 80, 100),
    ]

    for label, min_s, max_s in ranges:
        bucket_trades = [t for t in trades
                        if min_s <= t['signal_strength'] < max_s]

        if not bucket_trades:
            continue

        wins = [t for t in bucket_trades if t['is_win']]
        losses = [t for t in bucket_trades if not t['is_win']]

        win_rate = len(wins) / len(bucket_trades)
        avg_return = np.mean([t['return_pct'] for t in bucket_trades])
        avg_win = np.mean([t['return_pct'] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t['return_pct'] for t in losses])) if losses else 0

        kelly = calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        optimal_size = kelly / 2  # Half-Kelly for safety

        buckets.append(SignalStrengthBucket(
            range_label=label,
            min_strength=min_s,
            max_strength=max_s,
            trade_count=len(bucket_trades),
            win_rate=round(win_rate * 100, 1),
            avg_return=round(avg_return, 3),
            kelly_fraction=round(kelly * 100, 2),
            optimal_size_pct=round(optimal_size * 100, 2)
        ))

        print(f"  {label}: {len(bucket_trades)} trades, "
              f"win={win_rate*100:.1f}%, kelly={kelly*100:.1f}%")

    return buckets


def analyze_volatility(trades: List[Dict]) -> Dict:
    """Analyze volatility patterns for sizing."""

    print("\nAnalyzing volatility patterns...")

    # Group trades by ATR quartiles
    atr_values = [t['atr_pct'] for t in trades]
    q1, q2, q3 = np.percentile(atr_values, [25, 50, 75])

    quartile_analysis = []

    for label, low, high in [('Low Vol', 0, q1), ('Med-Low', q1, q2),
                              ('Med-High', q2, q3), ('High Vol', q3, 100)]:
        q_trades = [t for t in trades if low <= t['atr_pct'] < high]

        if not q_trades:
            continue

        win_rate = sum(1 for t in q_trades if t['is_win']) / len(q_trades)
        avg_return = np.mean([t['return_pct'] for t in q_trades])
        avg_max_loss = np.mean([t['max_loss'] for t in q_trades])

        quartile_analysis.append({
            'quartile': label,
            'atr_range': f"{low:.2f}% - {high:.2f}%",
            'trade_count': len(q_trades),
            'win_rate': round(win_rate * 100, 1),
            'avg_return': round(avg_return, 3),
            'avg_max_loss': round(avg_max_loss, 3)
        })

        print(f"  {label}: {len(q_trades)} trades, win={win_rate*100:.1f}%, "
              f"avg_return={avg_return:.2f}%, max_loss={avg_max_loss:.2f}%")

    return {
        'atr_quartiles': {'q1': round(q1, 3), 'q2': round(q2, 3), 'q3': round(q3, 3)},
        'quartile_analysis': quartile_analysis
    }


def generate_portfolio_recommendations(tier_metrics: Dict,
                                       strength_buckets: List,
                                       trades: List[Dict]) -> Dict:
    """Generate portfolio-level recommendations."""

    print("\nGenerating portfolio recommendations...")

    # Max positions at once (based on correlation risk)
    # More positions = more diversification but less per-position
    max_positions = 6

    # Max portfolio heat (total risk)
    max_portfolio_heat = 15.0  # 15% max at risk

    # Calculate average drawdown
    drawdowns = [abs(t['max_loss']) for t in trades]
    avg_drawdown = np.mean(drawdowns)
    max_drawdown = np.percentile(drawdowns, 95)  # 95th percentile

    # Position size limits by tier
    tier_limits = {}
    for tier, metrics in tier_metrics.items():
        # Base on half-Kelly but cap based on tier quality
        base_size = metrics.half_kelly / 100

        if tier == 'elite':
            cap = 0.12  # 12% max
        elif tier == 'strong':
            cap = 0.10  # 10% max
        elif tier == 'average':
            cap = 0.07  # 7% max
        else:  # avoid
            cap = 0.04  # 4% max

        tier_limits[tier] = {
            'min_size': 0.02,  # 2% min
            'max_size': round(cap, 2),
            'default_size': round(min(base_size, cap), 2)
        }

    # Signal strength multipliers
    strength_multipliers = {
        'very_strong': 1.3,
        'strong': 1.1,
        'moderate': 1.0,
        'weak': 0.7
    }

    recommendations = {
        'max_positions': max_positions,
        'max_portfolio_heat': max_portfolio_heat,
        'avg_drawdown': round(avg_drawdown, 2),
        'max_drawdown_95pct': round(max_drawdown, 2),
        'tier_limits': tier_limits,
        'strength_multipliers': strength_multipliers,
        'risk_per_trade': 2.0,  # 2% risk per trade as default
    }

    print(f"  Max positions: {max_positions}")
    print(f"  Max portfolio heat: {max_portfolio_heat}%")
    print(f"  Avg drawdown: {avg_drawdown:.2f}%")
    print(f"  95th pct drawdown: {max_drawdown:.2f}%")

    return recommendations


def run_research():
    """Run complete position sizing research."""

    print("=" * 70)
    print("POSITION SIZING RESEARCH")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Initialize
    optimizer = ExitOptimizer()

    # Fetch data
    stock_data = fetch_stock_data(days_back=450)

    # Generate trades
    trades = generate_trades(stock_data, optimizer, days_back=400)

    if not trades:
        print("No trades generated!")
        return None

    # Analyze by tier
    tier_metrics = analyze_by_tier(trades)

    # Analyze by signal strength
    strength_buckets = analyze_by_signal_strength(trades)

    # Analyze volatility
    volatility_analysis = analyze_volatility(trades)

    # Generate recommendations
    portfolio_recommendations = generate_portfolio_recommendations(
        tier_metrics, strength_buckets, trades
    )

    # Create report
    report = PositionSizingReport(
        timestamp=datetime.now().isoformat(),
        total_trades=len(trades),
        tier_metrics={k: asdict(v) for k, v in tier_metrics.items()},
        strength_buckets=[asdict(b) for b in strength_buckets],
        portfolio_recommendations=portfolio_recommendations,
        volatility_analysis=volatility_analysis
    )

    # Save results
    output_dir = Path('data/research')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'position_sizing_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(asdict(report), f, indent=2)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"Total trades analyzed: {len(trades)}")
    print(f"\nTier Kelly Fractions:")
    for tier, metrics in tier_metrics.items():
        print(f"  {tier.upper()}: {metrics.kelly_fraction}% (half: {metrics.half_kelly}%)")

    print(f"\nStrength-Based Sizing:")
    for bucket in strength_buckets:
        print(f"  {bucket.range_label}: {bucket.optimal_size_pct}% optimal")

    print(f"\nPortfolio Limits:")
    print(f"  Max positions: {portfolio_recommendations['max_positions']}")
    print(f"  Max heat: {portfolio_recommendations['max_portfolio_heat']}%")

    print(f"\nResults saved to: {output_path}")

    return report


if __name__ == '__main__':
    run_research()

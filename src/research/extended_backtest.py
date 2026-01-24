"""
Extended Backtest - 2-Year Validation with Combined Optimizations

This script runs extended backtests to validate:
1. Exit strategy optimization over 2 years
2. Position sizing impact on portfolio returns
3. Sharpe ratio calculations
4. Stress testing across different market regimes
5. Combined system performance
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
from typing import Dict, List, Tuple, Optional
from enum import Enum
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trading.exit_optimizer import ExitOptimizer, StockTier
from src.trading.position_sizer import PositionSizer, SignalQuality


TICKERS = [
    'NVDA', 'AVGO', 'MSFT', 'ORCL', 'INTU', 'ADBE', 'CRM', 'NOW', 'AMAT', 'KLAC',
    'MRVL', 'QCOM', 'TXN', 'ADI', 'JPM', 'MS', 'SCHW', 'AXP', 'AIG', 'USB',
    'PNC', 'V', 'MA', 'ABBV', 'SYK', 'GILD', 'PFE', 'JNJ', 'CVS', 'HCA',
    'IDXX', 'INSM', 'COP', 'SLB', 'XOM', 'MPC', 'EOG', 'CAT', 'ETN', 'LMT',
    'ROAD', 'HD', 'LOW', 'TGT', 'WMT', 'TMUS', 'CMCSA', 'META', 'APD', 'ECL',
    'MLM', 'SHW', 'EXR', 'NEE'
]


class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    VOLATILE = "volatile"
    CHOPPY = "choppy"


@dataclass
class TradeResult:
    """Individual trade result."""
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    hold_days: int
    tier: str
    signal_strength: float
    position_size_pct: float
    dollar_return: float
    exit_reason: str
    market_regime: str


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_trades: int
    win_rate: float
    avg_return: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    avg_winner: float
    avg_loser: float
    best_trade: float
    worst_trade: float
    avg_hold_days: float


@dataclass
class RegimePerformance:
    """Performance breakdown by market regime."""
    regime: str
    trade_count: int
    win_rate: float
    avg_return: float
    sharpe_ratio: float


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


def get_consecutive_down_days(prices, idx):
    """Count consecutive down days ending at index."""
    count = 0
    for i in range(idx, 0, -1):
        if prices.iloc[i] < prices.iloc[i-1]:
            count += 1
        else:
            break
    return count


def detect_market_regime(spy_data: pd.DataFrame, date_idx: int) -> MarketRegime:
    """Detect market regime based on SPY behavior."""
    if date_idx < 30:
        return MarketRegime.BULL

    lookback = spy_data.iloc[max(0, date_idx-30):date_idx+1]

    if len(lookback) < 10:
        return MarketRegime.BULL

    returns = lookback['Close'].pct_change()
    volatility = returns.std() * np.sqrt(252)
    trend = (lookback['Close'].iloc[-1] / lookback['Close'].iloc[0] - 1) * 100

    if volatility > 0.30:
        return MarketRegime.VOLATILE
    elif trend < -5:
        return MarketRegime.BEAR
    elif abs(trend) < 2 and volatility < 0.15:
        return MarketRegime.CHOPPY
    else:
        return MarketRegime.BULL


def fetch_data(years_back: int = 2) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Fetch historical data for backtesting."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365 + 60)

    print(f"Fetching {years_back} years of data for {len(TICKERS)} stocks...")

    stock_data = {}
    for i, ticker in enumerate(TICKERS):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            if len(df) >= 200:
                df['RSI'] = calculate_rsi(df['Close'])
                df['ATR'] = calculate_atr(df)
                df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
                stock_data[ticker] = df
        except Exception:
            pass

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(TICKERS)} fetched")

    # Fetch SPY for regime detection
    spy_df = yf.download('SPY', start=start_date, end=end_date, progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.droplevel(1)

    print(f"  Got data for {len(stock_data)} stocks")
    return stock_data, spy_df


def simulate_trade_with_exit_optimizer(
    ticker: str,
    df: pd.DataFrame,
    entry_idx: int,
    signal_strength: float,
    optimizer: ExitOptimizer
) -> Tuple[float, int, str, float, float]:
    """
    Simulate a trade using exit optimizer rules.

    Returns: (return_pct, hold_days, exit_reason, max_gain, max_loss)
    """
    entry_price = df['Close'].iloc[entry_idx]
    strategy = optimizer.get_exit_strategy(ticker, signal_strength)

    high_since_entry = entry_price
    portion_remaining = 1.0
    total_return = 0.0
    max_gain = 0.0
    max_loss = 0.0

    for day in range(1, strategy.max_hold_days + 1):
        if entry_idx + day >= len(df):
            # Exit at end of data
            exit_price = df['Close'].iloc[-1]
            return_pct = ((exit_price / entry_price) - 1) * 100
            return return_pct, day, "END_OF_DATA", max_gain, max_loss

        current_price = df['Close'].iloc[entry_idx + day]
        high_since_entry = max(high_since_entry, df['High'].iloc[entry_idx + day])

        # Track max gain/loss
        current_return = ((current_price / entry_price) - 1) * 100
        max_gain = max(max_gain, current_return)
        max_loss = min(max_loss, current_return)

        should_exit, exit_portion, reason = optimizer.should_exit(
            ticker=ticker,
            entry_price=entry_price,
            current_price=current_price,
            high_since_entry=high_since_entry,
            days_held=day,
            portion_remaining=portion_remaining,
            signal_strength=signal_strength
        )

        if should_exit:
            return_pct = ((current_price / entry_price) - 1) * 100
            total_return += return_pct * exit_portion
            portion_remaining -= exit_portion

            if portion_remaining <= 0.01:
                return total_return, day, reason, max_gain, max_loss

    # Max hold reached
    exit_price = df['Close'].iloc[min(entry_idx + strategy.max_hold_days, len(df) - 1)]
    final_return = ((exit_price / entry_price) - 1) * 100
    total_return += final_return * portion_remaining

    return total_return, strategy.max_hold_days, "MAX_HOLD", max_gain, max_loss


def run_backtest(
    stock_data: Dict[str, pd.DataFrame],
    spy_data: pd.DataFrame,
    years: int = 2,
    use_position_sizing: bool = True,
    portfolio_value: float = 100000.0
) -> List[TradeResult]:
    """Run extended backtest with all optimizations."""

    optimizer = ExitOptimizer()
    sizer = PositionSizer(portfolio_value=portfolio_value, load_positions=False)

    trades = []
    cutoff_date = datetime.now() - timedelta(days=years * 365)
    cutoff_str = cutoff_date.strftime('%Y-%m-%d')

    print(f"\nRunning {years}-year backtest from {cutoff_str}...")

    for ticker, df in stock_data.items():
        try:
            for i in range(30, len(df) - 10):
                # RSI signal
                if df['RSI'].iloc[i] < 35:
                    signal_date = df.index[i]
                    signal_str = signal_date.strftime('%Y-%m-%d') if hasattr(signal_date, 'strftime') else str(signal_date)[:10]

                    if signal_str < cutoff_str:
                        continue

                    entry_price = df['Close'].iloc[i]
                    rsi = df['RSI'].iloc[i]
                    atr = df['ATR'].iloc[i]
                    atr_pct = (atr / entry_price) * 100 if entry_price > 0 else 2.5
                    volume_ratio = df['Volume_Ratio'].iloc[i] if pd.notna(df['Volume_Ratio'].iloc[i]) else 1.0
                    consecutive_down = get_consecutive_down_days(df['Close'], i)

                    # Calculate signal strength
                    signal_strength = max(0, min(100, (35 - rsi) * 3 + 50))

                    # Get tier
                    tier = optimizer.get_stock_tier(ticker)

                    # Detect market regime
                    spy_idx = spy_data.index.get_indexer([signal_date], method='nearest')[0]
                    regime = detect_market_regime(spy_data, spy_idx)

                    # Calculate position size
                    if use_position_sizing:
                        pos_size = sizer.calculate_position_size(
                            ticker=ticker,
                            current_price=entry_price,
                            signal_strength=signal_strength,
                            atr_pct=atr_pct,
                            volume_ratio=volume_ratio,
                            consecutive_down_days=consecutive_down
                        )

                        if pos_size.skip_trade:
                            continue

                        position_size_pct = pos_size.size_pct / 100
                    else:
                        position_size_pct = 0.05  # Default 5%

                    # Simulate trade with exit optimizer
                    return_pct, hold_days, exit_reason, max_gain, max_loss = simulate_trade_with_exit_optimizer(
                        ticker=ticker,
                        df=df,
                        entry_idx=i,
                        signal_strength=signal_strength,
                        optimizer=optimizer
                    )

                    # Calculate dollar return
                    position_dollars = portfolio_value * position_size_pct
                    dollar_return = position_dollars * (return_pct / 100)

                    exit_idx = min(i + hold_days, len(df) - 1)
                    exit_date = df.index[exit_idx]
                    exit_date_str = exit_date.strftime('%Y-%m-%d') if hasattr(exit_date, 'strftime') else str(exit_date)[:10]

                    trades.append(TradeResult(
                        ticker=ticker,
                        entry_date=signal_str,
                        exit_date=exit_date_str,
                        entry_price=round(entry_price, 2),
                        exit_price=round(df['Close'].iloc[exit_idx], 2),
                        return_pct=round(return_pct, 3),
                        hold_days=hold_days,
                        tier=tier.value,
                        signal_strength=round(signal_strength, 1),
                        position_size_pct=round(position_size_pct * 100, 2),
                        dollar_return=round(dollar_return, 2),
                        exit_reason=exit_reason,
                        market_regime=regime.value
                    ))

        except Exception as e:
            continue

    print(f"  Generated {len(trades)} trades")
    return trades


def calculate_portfolio_metrics(trades: List[TradeResult]) -> PortfolioMetrics:
    """Calculate comprehensive portfolio metrics."""
    if not trades:
        return None

    returns = [t.return_pct for t in trades]
    dollar_returns = [t.dollar_return for t in trades]

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    win_rate = len(wins) / len(returns) * 100
    avg_return = np.mean(returns)
    total_return = np.sum(dollar_returns)

    # Sharpe ratio (assuming 252 trading days, risk-free rate 4%)
    if len(returns) >= 10:
        daily_returns = np.array(returns) / 100  # Convert to decimal
        # Annualize: assume average holding period determines frequency
        avg_hold = np.mean([t.hold_days for t in trades])
        trades_per_year = 252 / avg_hold if avg_hold > 0 else 60

        annualized_return = np.mean(daily_returns) * trades_per_year
        annualized_std = np.std(daily_returns) * np.sqrt(trades_per_year)
        risk_free = 0.04  # 4% annual

        sharpe = (annualized_return - risk_free) / annualized_std if annualized_std > 0 else 0
    else:
        sharpe = 0

    # Max drawdown (based on cumulative returns)
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    max_dd = abs(np.min(drawdown)) if len(drawdown) > 0 else 0

    # Profit factor
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return PortfolioMetrics(
        total_trades=len(trades),
        win_rate=round(win_rate, 1),
        avg_return=round(avg_return, 3),
        total_return=round(total_return, 2),
        sharpe_ratio=round(sharpe, 2),
        max_drawdown=round(max_dd, 2),
        profit_factor=round(profit_factor, 2),
        avg_winner=round(np.mean(wins), 3) if wins else 0,
        avg_loser=round(np.mean(losses), 3) if losses else 0,
        best_trade=round(max(returns), 2),
        worst_trade=round(min(returns), 2),
        avg_hold_days=round(np.mean([t.hold_days for t in trades]), 1)
    )


def analyze_by_regime(trades: List[TradeResult]) -> List[RegimePerformance]:
    """Analyze performance by market regime."""
    regime_perf = []

    for regime in MarketRegime:
        regime_trades = [t for t in trades if t.market_regime == regime.value]

        if not regime_trades:
            continue

        returns = [t.return_pct for t in regime_trades]
        win_rate = sum(1 for r in returns if r > 0) / len(returns) * 100

        # Calculate regime Sharpe
        if len(returns) >= 5:
            avg_hold = np.mean([t.hold_days for t in regime_trades])
            trades_per_year = 252 / avg_hold if avg_hold > 0 else 60
            daily_returns = np.array(returns) / 100
            annualized_return = np.mean(daily_returns) * trades_per_year
            annualized_std = np.std(daily_returns) * np.sqrt(trades_per_year)
            sharpe = (annualized_return - 0.04) / annualized_std if annualized_std > 0 else 0
        else:
            sharpe = 0

        regime_perf.append(RegimePerformance(
            regime=regime.value,
            trade_count=len(regime_trades),
            win_rate=round(win_rate, 1),
            avg_return=round(np.mean(returns), 3),
            sharpe_ratio=round(sharpe, 2)
        ))

    return regime_perf


def analyze_by_tier(trades: List[TradeResult]) -> Dict:
    """Analyze performance by stock tier."""
    tier_analysis = {}

    for tier in ['elite', 'strong', 'average', 'avoid']:
        tier_trades = [t for t in trades if t.tier == tier]

        if not tier_trades:
            continue

        returns = [t.return_pct for t in tier_trades]
        dollar_returns = [t.dollar_return for t in tier_trades]

        tier_analysis[tier] = {
            'trade_count': len(tier_trades),
            'win_rate': round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1),
            'avg_return': round(np.mean(returns), 3),
            'total_dollar_return': round(sum(dollar_returns), 2),
            'avg_position_size': round(np.mean([t.position_size_pct for t in tier_trades]), 2)
        }

    return tier_analysis


def run_extended_backtest():
    """Run complete extended backtest analysis."""
    print("=" * 70)
    print("EXTENDED BACKTEST - 2 YEAR VALIDATION")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Fetch data
    stock_data, spy_data = fetch_data(years_back=2)

    # Run backtest with all optimizations
    trades = run_backtest(
        stock_data=stock_data,
        spy_data=spy_data,
        years=2,
        use_position_sizing=True,
        portfolio_value=100000.0
    )

    if not trades:
        print("No trades generated!")
        return None

    # Calculate metrics
    metrics = calculate_portfolio_metrics(trades)
    regime_perf = analyze_by_regime(trades)
    tier_perf = analyze_by_tier(trades)

    # Run baseline comparison (without optimizations)
    print("\nRunning baseline comparison (no position sizing)...")
    baseline_trades = run_backtest(
        stock_data=stock_data,
        spy_data=spy_data,
        years=2,
        use_position_sizing=False,
        portfolio_value=100000.0
    )
    baseline_metrics = calculate_portfolio_metrics(baseline_trades) if baseline_trades else None

    # Compile results
    results = {
        'timestamp': datetime.now().isoformat(),
        'backtest_period': '2 years',
        'optimized': {
            'metrics': asdict(metrics),
            'regime_performance': [asdict(r) for r in regime_perf],
            'tier_performance': tier_perf
        },
        'baseline': {
            'metrics': asdict(baseline_metrics) if baseline_metrics else None
        },
        'improvement': {
            'sharpe_improvement': round(metrics.sharpe_ratio - (baseline_metrics.sharpe_ratio if baseline_metrics else 0), 2),
            'win_rate_improvement': round(metrics.win_rate - (baseline_metrics.win_rate if baseline_metrics else 0), 1),
            'avg_return_improvement': round(metrics.avg_return - (baseline_metrics.avg_return if baseline_metrics else 0), 3)
        }
    }

    # Save results
    output_dir = Path('data/research')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'extended_backtest_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("EXTENDED BACKTEST RESULTS")
    print("=" * 70)

    print("\n[OPTIMIZED PERFORMANCE]")
    print(f"  Total Trades: {metrics.total_trades}")
    print(f"  Win Rate: {metrics.win_rate}%")
    print(f"  Avg Return: {metrics.avg_return}%")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio}")
    print(f"  Max Drawdown: {metrics.max_drawdown}%")
    print(f"  Profit Factor: {metrics.profit_factor}")
    print(f"  Avg Hold Days: {metrics.avg_hold_days}")

    if baseline_metrics:
        print("\n[BASELINE PERFORMANCE]")
        print(f"  Total Trades: {baseline_metrics.total_trades}")
        print(f"  Win Rate: {baseline_metrics.win_rate}%")
        print(f"  Avg Return: {baseline_metrics.avg_return}%")
        print(f"  Sharpe Ratio: {baseline_metrics.sharpe_ratio}")

        print("\n[IMPROVEMENT]")
        print(f"  Sharpe: +{results['improvement']['sharpe_improvement']}")
        print(f"  Win Rate: +{results['improvement']['win_rate_improvement']}%")
        print(f"  Avg Return: +{results['improvement']['avg_return_improvement']}%")

    print("\n[BY MARKET REGIME]")
    for rp in regime_perf:
        print(f"  {rp.regime.upper()}: {rp.trade_count} trades, "
              f"{rp.win_rate}% win, {rp.avg_return}% avg, Sharpe {rp.sharpe_ratio}")

    print("\n[BY TIER]")
    for tier, perf in tier_perf.items():
        print(f"  {tier.upper()}: {perf['trade_count']} trades, "
              f"{perf['win_rate']}% win, {perf['avg_return']}% avg, "
              f"${perf['total_dollar_return']:,.0f} total")

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    run_extended_backtest()

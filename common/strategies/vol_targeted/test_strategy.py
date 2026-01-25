"""
Test Vol-Targeted Strategy on Real Data

This script:
1. Fetches SPY data using yfinance
2. Runs multiple strategy variants
3. Performs honest evaluation with Sharpe deflation
4. Compares against null hypothesis (random signal)
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("Warning: yfinance not installed. Using synthetic data.")

# Import strategy modules (use absolute imports for standalone script)
from risk_manager import VolTargetingRiskManager, RiskManagerConfig
from signals import (
    MomentumSignal, MeanReversionSignal, TrendFollowingSignal,
    RandomSignal, BuyAndHoldSignal, CombinedSignal
)
from strategy_standalone import VolTargetedStrategy, StrategyConfig, StrategyFactory
from backtest_standalone import Backtester, BacktestConfig
from evaluator_standalone import HonestStrategyEvaluator, quick_evaluate


def fetch_spy_data(start_date: str = "2015-01-01", end_date: str = None) -> tuple:
    """Fetch SPY data using yfinance."""
    if not HAS_YFINANCE:
        return generate_synthetic_data()

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Fetching SPY data from {start_date} to {end_date}...")
    spy = yf.download("SPY", start=start_date, end=end_date, progress=False)

    # Handle both single and multi-column formats from yfinance
    if isinstance(spy.columns, pd.MultiIndex):
        prices = spy['Close']['SPY'].values.flatten()
    else:
        prices = spy['Close'].values.flatten()

    # Ensure we have a 1D numpy array of floats
    prices = np.array(prices, dtype=np.float64)
    dates = spy.index.tolist()

    print(f"Loaded {len(prices)} days of SPY data")
    return prices, dates


def generate_synthetic_data(num_days: int = 2520) -> tuple:
    """Generate synthetic price data for testing without yfinance."""
    print(f"Generating {num_days} days of synthetic data...")

    np.random.seed(42)

    # Simulate price with drift and volatility regimes
    returns = []
    vol = 0.15 / np.sqrt(252)  # Base vol

    for i in range(num_days):
        # Regime switches
        if i % 252 < 63:  # Q1: Higher vol
            current_vol = vol * 1.5
        elif i % 252 < 189:  # Q2-Q3: Normal vol
            current_vol = vol
        else:  # Q4: Lower vol
            current_vol = vol * 0.8

        # Add some momentum persistence
        if i > 0 and len(returns) > 0:
            momentum = 0.05 * returns[-1]
        else:
            momentum = 0.0

        daily_return = np.random.normal(0.0003 + momentum, current_vol)
        returns.append(daily_return)

    # Convert to prices
    prices = 100 * np.cumprod(1 + np.array(returns))

    # Generate dates
    start = datetime(2015, 1, 1)
    dates = [start + timedelta(days=i) for i in range(num_days)]

    return prices, dates


def run_comparison_test(prices: np.ndarray, dates: list):
    """Run comparison test across multiple strategies."""
    print("\n" + "="*70)
    print("STRATEGY COMPARISON TEST")
    print("="*70)

    # Configure backtester with realistic costs
    backtest_config = BacktestConfig(
        spread_bps=1.0,
        commission_bps=0.5,
        slippage_bps=0.5,
        initial_capital=100000.0,
    )
    backtester = Backtester(backtest_config)

    strategies = {
        "Momentum (20d)": StrategyFactory.create_momentum_strategy(lookback=20),
        "Momentum (60d)": StrategyFactory.create_momentum_strategy(lookback=60),
        "Trend Following": StrategyFactory.create_trend_following_strategy(),
        "Buy & Hold + Vol Target": StrategyFactory.create_buy_and_hold_baseline(),
        "Random Signal Null": StrategyFactory.create_random_baseline(seed=42),
    }

    results = {}

    for name, strategy in strategies.items():
        print(f"\nTesting: {name}")
        result = backtester.run(strategy, prices, dates)
        results[name] = result

        print(f"  Sharpe: {result.sharpe_ratio:.2f}")
        print(f"  Return: {result.total_return:.1%}")
        print(f"  Max DD: {result.max_drawdown:.1%}")
        print(f"  Trades: {result.num_trades}")
        print(f"  Costs: ${result.total_costs:,.0f}")

    # Summary table
    print("\n" + "-"*70)
    print("SUMMARY")
    print("-"*70)
    print(f"{'Strategy':<30} {'Sharpe':>8} {'Return':>10} {'Max DD':>10} {'Trades':>8}")
    print("-"*70)

    for name, result in sorted(results.items(), key=lambda x: -x[1].sharpe_ratio):
        print(f"{name:<30} {result.sharpe_ratio:>8.2f} {result.total_return:>9.1%} {result.max_drawdown:>9.1%} {result.num_trades:>8}")

    return results


def run_honest_evaluation(prices: np.ndarray):
    """Run honest evaluation with Sharpe deflation."""
    print("\n" + "="*70)
    print("HONEST STRATEGY EVALUATION")
    print("="*70)

    # Test the main momentum strategy
    strategy = StrategyFactory.create_momentum_strategy(lookback=20)

    print("\nRunning comprehensive evaluation (this may take a minute)...")
    evaluation = quick_evaluate(strategy, prices, verbose=True)

    return evaluation


def run_ablation_showcase(prices: np.ndarray):
    """Demonstrate ablation analysis."""
    print("\n" + "="*70)
    print("ABLATION ANALYSIS")
    print("="*70)
    print("Testing what each component contributes...")

    backtest_config = BacktestConfig(spread_bps=1.0, commission_bps=0.5, slippage_bps=0.5)
    backtester = Backtester(backtest_config)

    # Full strategy
    full_strategy = StrategyFactory.create_momentum_strategy(lookback=20)
    full_result = backtester.run(full_strategy, prices)
    print(f"\nFull Strategy Sharpe: {full_result.sharpe_ratio:.2f}")

    # No vol targeting (target vol = 100%)
    no_vol_config = StrategyConfig(target_vol=1.0, max_leverage=1.0, min_leverage=-1.0)
    no_vol_strategy = VolTargetedStrategy(MomentumSignal(20), no_vol_config)
    no_vol_result = backtester.run(no_vol_strategy, prices)
    print(f"Without Vol Targeting: {no_vol_result.sharpe_ratio:.2f}")
    print(f"  -> Vol targeting contributes: {full_result.sharpe_ratio - no_vol_result.sharpe_ratio:.2f}")

    # No deadband
    no_db_config = StrategyConfig(deadband_tau=0.0)
    no_db_strategy = VolTargetedStrategy(MomentumSignal(20), no_db_config)
    no_db_result = backtester.run(no_db_strategy, prices)
    print(f"Without Deadband: {no_db_result.sharpe_ratio:.2f}")
    print(f"  -> Deadband contributes: {full_result.sharpe_ratio - no_db_result.sharpe_ratio:.2f}")

    # Random signal (null hypothesis)
    random_strategy = StrategyFactory.create_random_baseline(seed=42)
    random_result = backtester.run(random_strategy, prices)
    print(f"Random Signal Baseline: {random_result.sharpe_ratio:.2f}")
    print(f"  -> Signal contributes: {full_result.sharpe_ratio - random_result.sharpe_ratio:.2f}")


def main():
    """Main test runner."""
    print("="*70)
    print("VOL-TARGETED STRATEGY TEST SUITE")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Fetch data
    if HAS_YFINANCE:
        prices, dates = fetch_spy_data(start_date="2015-01-01")
    else:
        prices, dates = generate_synthetic_data()

    # Run tests
    results = run_comparison_test(prices, dates)

    # Run ablation
    run_ablation_showcase(prices)

    # Run honest evaluation
    evaluation = run_honest_evaluation(prices)

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    # Key takeaways
    print("\nKEY TAKEAWAYS:")
    print("-" * 40)
    print("1. Vol targeting adds ~0.2-0.4 Sharpe mechanically")
    print("2. Random signal + vol targeting sets the null hypothesis baseline")
    print("3. Signal alpha = deflated Sharpe - vol targeting contribution")
    print("4. Always apply Sharpe deflation for realistic expectations")

    return evaluation


if __name__ == "__main__":
    main()

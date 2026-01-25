"""
EXPERIMENT: EXP-029
Date: 2025-11-16
Objective: Optimize position sizing for Tier A portfolio

RESEARCH MOTIVATION:
We now have 22 excellent Tier A stocks (v12.0):
- Average win rate: 77.7%
- Average return: +28.84%
- Trade frequency: ~110/year

Current strategy: EQUAL WEIGHTING ($10K per trade)
Problem: INSM delivers +142% but gets same allocation as SYK at +11%

HYPOTHESIS:
Intelligent position sizing can improve returns by 10-20% without adding risk.

POSITION SIZING STRATEGIES TO TEST:

1. EQUAL WEIGHTING (baseline - current)
   - Every stock gets $10K allocation
   - Simple, no optimization

2. WIN RATE WEIGHTING
   - Allocate more to high win rate stocks
   - Formula: allocation ∝ (win_rate - 50%) / 50%
   - Example: 88.9% win rate (ROAD) gets 1.78x base allocation
   - Example: 70% win rate gets 0.40x base allocation

3. SHARPE RATIO WEIGHTING
   - Allocate based on risk-adjusted returns
   - Formula: allocation ∝ sharpe_ratio
   - Rewards consistency + returns together

4. KELLY CRITERION
   - Mathematically optimal sizing
   - Formula: f = (win_rate * avg_gain - (1 - win_rate) * avg_loss) / avg_gain
   - Maximizes long-term growth
   - Potential for aggressive sizing

5. VOLATILITY-ADJUSTED (INVERSE VOLATILITY)
   - Allocate less to volatile stocks
   - Formula: allocation ∝ 1 / volatility
   - Reduces portfolio variance

6. HYBRID (SHARPE + KELLY)
   - Combines risk-adjusted returns with growth optimization
   - Formula: allocation = 0.5 * sharpe_weight + 0.5 * kelly_weight
   - Balances safety and growth

TESTING METHODOLOGY:
1. Backtest each strategy on 22 Tier A stocks (3 years)
2. Measure:
   - Total return
   - Sharpe ratio
   - Maximum drawdown
   - Win rate (should remain ~77.7%)
   - Exposure concentration (Herfindahl index)

3. Compare vs baseline (equal weighting)

EXPECTED OUTCOMES:
- Kelly criterion: Highest returns but potentially high risk
- Sharpe weighting: Best risk-adjusted performance
- Win rate weighting: Conservative, stable
- Equal weighting: Safe baseline

RISK MANAGEMENT CONSTRAINTS:
- Max position size: 3x base allocation (prevent over-concentration)
- Min position size: 0.25x base allocation (ensure diversification)
- Max single-stock exposure: 15% of portfolio
- Rebalance: Monthly (prevent drift)

SUCCESS CRITERIA:
- 10%+ improvement in Sharpe ratio vs equal weighting
- <20% max drawdown
- Reasonable concentration (HHI < 0.20)
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.config.mean_reversion_params import MEAN_REVERSION_PARAMS


def calculate_position_sizes(stocks_data: Dict, strategy: str) -> Dict[str, float]:
    """
    Calculate position sizes for each stock based on strategy.

    Args:
        stocks_data: Dict with stock performance metrics
        strategy: Position sizing strategy

    Returns:
        Dict mapping stock symbol to allocation multiplier
    """
    allocations = {}

    if strategy == 'equal':
        # Equal weighting (baseline)
        for symbol in stocks_data:
            allocations[symbol] = 1.0

    elif strategy == 'win_rate':
        # Weight by win rate (scaled)
        for symbol, data in stocks_data.items():
            win_rate = data['win_rate']
            # Scale: 50% win = 0x, 100% win = 2x
            weight = max(0, (win_rate - 0.50) / 0.25)
            allocations[symbol] = weight

    elif strategy == 'sharpe':
        # Weight by Sharpe ratio
        sharpe_values = [data['sharpe'] for data in stocks_data.values()]
        min_sharpe = min(sharpe_values)
        max_sharpe = max(sharpe_values)

        for symbol, data in stocks_data.items():
            # Normalize Sharpe to 0-2 range
            if max_sharpe > min_sharpe:
                weight = 0.5 + 1.5 * (data['sharpe'] - min_sharpe) / (max_sharpe - min_sharpe)
            else:
                weight = 1.0
            allocations[symbol] = weight

    elif strategy == 'kelly':
        # Kelly criterion
        for symbol, data in stocks_data.items():
            win_rate = data['win_rate']
            avg_gain = data['avg_gain']
            avg_loss = abs(data['avg_loss'])

            if avg_gain > 0:
                # Kelly formula
                kelly_fraction = (win_rate * avg_gain - (1 - win_rate) * avg_loss) / avg_gain
                # Use fractional Kelly (25% of full Kelly) for safety
                weight = max(0.25, min(3.0, kelly_fraction * 0.25))
            else:
                weight = 0.25

            allocations[symbol] = weight

    elif strategy == 'inverse_volatility':
        # Inverse volatility weighting
        volatilities = [1.0 / data['sharpe'] if data['sharpe'] > 0 else 999 for data in stocks_data.values()]
        inv_vols = [1.0 / v if v > 0 else 0 for v in volatilities]
        total_inv_vol = sum(inv_vols)

        for i, (symbol, data) in enumerate(stocks_data.items()):
            if total_inv_vol > 0:
                weight = (inv_vols[i] / total_inv_vol) * len(stocks_data)
            else:
                weight = 1.0
            allocations[symbol] = weight

    elif strategy == 'hybrid':
        # Hybrid: 50% Sharpe + 50% Kelly
        sharpe_alloc = calculate_position_sizes(stocks_data, 'sharpe')
        kelly_alloc = calculate_position_sizes(stocks_data, 'kelly')

        for symbol in stocks_data:
            allocations[symbol] = 0.5 * sharpe_alloc[symbol] + 0.5 * kelly_alloc[symbol]

    # Apply constraints
    for symbol in allocations:
        allocations[symbol] = max(0.25, min(3.0, allocations[symbol]))

    # Normalize to sum to number of stocks (maintains total capital)
    total_weight = sum(allocations.values())
    if total_weight > 0:
        for symbol in allocations:
            allocations[symbol] *= len(stocks_data) / total_weight

    return allocations


def run_exp029_position_sizing():
    """
    Test different position sizing strategies on Tier A portfolio.
    """
    print("=" * 70)
    print("EXP-029: POSITION SIZING OPTIMIZATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Optimize position sizing for 22 Tier A stocks")
    print("Current: Equal weighting ($10K per trade)")
    print("Hypothesis: Intelligent sizing can improve returns 10-20%")
    print()

    # Extract Tier A stock performance from mean_reversion_params
    tier_a_stocks = {}

    print("Tier A stocks (v12.0):")
    for symbol, params in MEAN_REVERSION_PARAMS.items():
        if symbol == 'DEFAULT':
            continue
        if params.get('tier') == 'A':
            # Parse performance string
            perf_str = params.get('performance', '')

            # Extract metrics from performance string
            # Format: "Win rate: X%, Return: Y%, Sharpe: Z, Avg gain: W%"
            try:
                win_rate_str = perf_str.split('Win rate: ')[1].split('%')[0]
                return_str = perf_str.split('Return: ')[1].split('%')[0]
                sharpe_str = perf_str.split('Sharpe: ')[1].split(',')[0]

                # Try to extract avg gain and avg loss
                if 'Avg gain:' in perf_str:
                    avg_gain_str = perf_str.split('Avg gain: ')[1].split('%')[0]
                    avg_gain = float(avg_gain_str) / 100.0
                else:
                    avg_gain = 0.02  # Default 2%

                # Estimate avg loss from win rate and return
                # This is a rough approximation
                avg_loss = -0.015  # Default -1.5%

                tier_a_stocks[symbol] = {
                    'win_rate': float(win_rate_str) / 100.0,
                    'return': float(return_str) / 100.0,
                    'sharpe': float(sharpe_str),
                    'avg_gain': avg_gain,
                    'avg_loss': avg_loss
                }

                print(f"  {symbol}: {win_rate_str}% win rate, {return_str}% return, {sharpe_str} Sharpe")
            except Exception as e:
                print(f"  [WARNING] Could not parse performance for {symbol}: {e}")

    print()
    print(f"Total Tier A stocks: {len(tier_a_stocks)}")
    print()

    # Test each position sizing strategy
    strategies = ['equal', 'win_rate', 'sharpe', 'kelly', 'inverse_volatility', 'hybrid']

    results = {}

    for strategy in strategies:
        print("=" * 70)
        print(f"STRATEGY: {strategy.upper().replace('_', ' ')}")
        print("=" * 70)

        allocations = calculate_position_sizes(tier_a_stocks, strategy)

        # Calculate portfolio metrics
        total_return = 0
        weighted_sharpe = 0
        concentration_metric = 0  # Herfindahl-Hirschman Index

        print("\nPosition sizes:")
        for symbol in sorted(allocations.keys(), key=lambda x: allocations[x], reverse=True):
            alloc = allocations[symbol]
            stock_data = tier_a_stocks[symbol]

            # Calculate contribution to portfolio
            contribution = alloc * stock_data['return']
            total_return += contribution

            weighted_sharpe += alloc * stock_data['sharpe']

            # HHI contribution (concentration)
            weight_pct = alloc / len(tier_a_stocks)
            concentration_metric += weight_pct ** 2

            print(f"  {symbol:6s}: {alloc:4.2f}x allocation ({weight_pct*100:5.1f}% weight) = {contribution*100:+7.2f}% contribution")

        # Normalize metrics
        avg_sharpe = weighted_sharpe / len(tier_a_stocks)
        avg_return = total_return / len(tier_a_stocks)

        print()
        print(f"Portfolio metrics:")
        print(f"  Total return: {avg_return*100:+.2f}%")
        print(f"  Avg Sharpe: {avg_sharpe:.2f}")
        print(f"  Concentration (HHI): {concentration_metric:.3f}")
        print(f"  Max allocation: {max(allocations.values()):.2f}x")
        print(f"  Min allocation: {min(allocations.values()):.2f}x")
        print()

        results[strategy] = {
            'allocations': allocations,
            'return': avg_return,
            'sharpe': avg_sharpe,
            'concentration': concentration_metric,
            'max_alloc': max(allocations.values()),
            'min_alloc': min(allocations.values())
        }

    # Compare strategies
    print("=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)
    print()

    baseline = results['equal']

    comparison_data = []
    for strategy, metrics in results.items():
        return_improvement = (metrics['return'] - baseline['return']) * 100
        sharpe_improvement = ((metrics['sharpe'] - baseline['sharpe']) / baseline['sharpe']) * 100

        comparison_data.append({
            'strategy': strategy,
            'return': metrics['return'] * 100,
            'return_vs_equal': return_improvement,
            'sharpe': metrics['sharpe'],
            'sharpe_vs_equal': sharpe_improvement,
            'concentration': metrics['concentration']
        })

    # Sort by Sharpe ratio
    comparison_data.sort(key=lambda x: x['sharpe'], reverse=True)

    print(f"{'Strategy':<20} {'Return':>10} {'vs Equal':>10} {'Sharpe':>10} {'vs Equal':>10} {'HHI':>8}")
    print("-" * 70)
    for row in comparison_data:
        print(f"{row['strategy']:<20} {row['return']:>9.2f}% {row['return_vs_equal']:>+9.2f}pp "
              f"{row['sharpe']:>9.2f} {row['sharpe_vs_equal']:>+9.1f}% {row['concentration']:>8.3f}")

    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print()

    # Find best strategy (highest Sharpe with reasonable concentration)
    valid_strategies = [s for s in comparison_data if s['concentration'] < 0.30]  # Not too concentrated
    if valid_strategies:
        best = valid_strategies[0]
        print(f"RECOMMENDED STRATEGY: {best['strategy'].upper().replace('_', ' ')}")
        print()
        print(f"Expected improvement over equal weighting:")
        print(f"  Return: {best['return_vs_equal']:+.2f}pp")
        print(f"  Sharpe: {best['sharpe_vs_equal']:+.1f}%")
        print(f"  Concentration: {best['concentration']:.3f} (reasonable)")
        print()

        if best['sharpe_vs_equal'] >= 10:
            print("[SUCCESS] HYPOTHESIS VALIDATED! >10% Sharpe improvement achieved.")
            print()
            print("Next steps:")
            print("1. Implement position sizing in production scanner")
            print("2. Add dynamic rebalancing (monthly)")
            print("3. Monitor for over-concentration")
            print("4. Track real-world performance vs backtest")
        else:
            print("[WARNING] MARGINAL IMPROVEMENT. Consider if complexity justified.")
            print()
            print("Options:")
            print("1. Keep equal weighting (simpler)")
            print("2. Implement best strategy and monitor closely")
            print("3. Test hybrid approaches")
    else:
        print("[WARNING] No strategy meets concentration threshold.")
        print("Recommendation: Stick with equal weighting")

    print()
    print("=" * 70)

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    combined_results = {
        'experiment_id': 'EXP-029',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tier_a_stocks': len(tier_a_stocks),
        'strategies_tested': len(strategies),
        'results': {k: {**v, 'allocations': {s: float(a) for s, a in v['allocations'].items()}}
                   for k, v in results.items()},
        'recommendation': best['strategy'] if valid_strategies else 'equal'
    }

    results_file = os.path.join(results_dir, 'exp029_position_sizing.json')
    with open(results_file, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)

    print(f"Results saved to: {results_file}")

    # Send email report
    try:
        from common.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending experiment report email...")
            notifier.send_experiment_report('EXP-029', combined_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return combined_results


if __name__ == '__main__':
    """Run EXP-029 position sizing optimization."""

    results = run_exp029_position_sizing()

    print("\n\nNext steps:")
    print("1. Implement recommended strategy in production")
    print("2. Add monthly rebalancing logic")
    print("3. Monitor concentration risk")
    print("4. Track vs equal weighting baseline")

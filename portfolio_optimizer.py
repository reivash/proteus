"""
Portfolio Optimization Tool
===========================

Uses Modern Portfolio Theory to optimize allocations across multiple stocks.

Features:
1. Sharpe Ratio Maximization (risk-adjusted return optimization)
2. Efficient Frontier calculation
3. Correlation analysis between stocks
4. Risk/return tradeoff visualization
5. Optimal allocation suggestions
6. Comparison vs equal-weight portfolio

Usage:
    python portfolio_optimizer.py                          # Optimize current recommendations
    python portfolio_optimizer.py --stocks MU,APP,PLTR     # Optimize specific stocks
    python portfolio_optimizer.py --risk-free 4.5          # Set risk-free rate
    python portfolio_optimizer.py --target-return 30       # Target specific return
"""

import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import os


class PortfolioOptimizer:
    """Optimizes portfolio allocations using Modern Portfolio Theory."""

    def __init__(self, risk_free_rate: float = 4.5):
        """
        Initialize optimizer.

        Args:
            risk_free_rate: Annual risk-free rate (%) - default 4.5% (current T-bill rate)
        """
        self.risk_free_rate = risk_free_rate / 100  # Convert to decimal
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)

    def get_historical_returns(self, ticker: str, period: str = '1y') -> np.ndarray:
        """Get historical daily returns for a stock."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)

            if len(hist) < 30:  # Need minimum data
                print(f"Warning: Insufficient data for {ticker}")
                return np.array([])

            # Calculate daily returns
            returns = hist['Close'].pct_change().dropna()
            return returns.values

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return np.array([])

    def calculate_portfolio_stats(self, tickers: List[str],
                                  weights: np.ndarray = None,
                                  period: str = '1y') -> Dict:
        """Calculate portfolio statistics (return, volatility, Sharpe)."""

        # Get returns for all stocks
        returns_data = {}
        for ticker in tickers:
            returns = self.get_historical_returns(ticker, period)
            if len(returns) > 0:
                returns_data[ticker] = returns

        if not returns_data:
            return {'error': 'No valid data for any ticker'}

        # Ensure all have same length (use minimum)
        min_length = min(len(r) for r in returns_data.values())
        returns_matrix = np.array([r[-min_length:] for r in returns_data.values()]).T

        # Equal weights if not provided
        if weights is None:
            weights = np.array([1.0 / len(returns_data)] * len(returns_data))

        # Calculate portfolio metrics
        portfolio_returns = returns_matrix @ weights

        # Annualized metrics (assuming 252 trading days)
        annual_return = np.mean(portfolio_returns) * 252
        annual_volatility = np.std(portfolio_returns) * np.sqrt(252)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

        # Correlation matrix
        correlation_matrix = np.corrcoef(returns_matrix.T)

        return {
            'tickers': list(returns_data.keys()),
            'weights': weights.tolist(),
            'annual_return': round(annual_return * 100, 2),  # Convert to %
            'annual_volatility': round(annual_volatility * 100, 2),  # Convert to %
            'sharpe_ratio': round(sharpe_ratio, 3),
            'correlation_matrix': correlation_matrix.tolist(),
        }

    def optimize_sharpe(self, tickers: List[str], period: str = '1y',
                       constraints: Dict = None) -> Dict:
        """
        Find portfolio weights that maximize Sharpe ratio.

        Args:
            tickers: List of stock tickers
            period: Historical period for returns calculation
            constraints: Optional constraints (e.g., {'MU': {'min': 0.2, 'max': 0.6}})
        """

        # Get returns data
        returns_data = {}
        for ticker in tickers:
            returns = self.get_historical_returns(ticker, period)
            if len(returns) > 0:
                returns_data[ticker] = returns

        if len(returns_data) < 2:
            print("Error: Need at least 2 stocks with valid data")
            return {}

        # Build returns matrix
        min_length = min(len(r) for r in returns_data.values())
        returns_matrix = np.array([r[-min_length:] for r in returns_data.values()]).T
        tickers_list = list(returns_data.keys())
        n_assets = len(tickers_list)

        # Calculate mean returns and covariance matrix
        mean_returns = np.mean(returns_matrix, axis=0) * 252  # Annualized
        cov_matrix = np.cov(returns_matrix.T) * 252  # Annualized

        # Grid search optimization (simple but effective)
        # For more sophisticated optimization, could use scipy.optimize

        best_sharpe = -np.inf
        best_weights = None

        # Generate random portfolios and find best
        n_portfolios = 10000

        for _ in range(n_portfolios):
            # Generate random weights
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)  # Normalize to sum to 1

            # Apply constraints if provided
            if constraints:
                valid = True
                for i, ticker in enumerate(tickers_list):
                    if ticker in constraints:
                        min_w = constraints[ticker].get('min', 0)
                        max_w = constraints[ticker].get('max', 1)
                        if weights[i] < min_w or weights[i] > max_w:
                            valid = False
                            break

                if not valid:
                    continue

            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = weights

        if best_weights is None:
            print("Optimization failed to find valid portfolio")
            return {}

        # Calculate final stats for optimal portfolio
        optimal_return = np.dot(best_weights, mean_returns)
        optimal_volatility = np.sqrt(np.dot(best_weights.T, np.dot(cov_matrix, best_weights)))

        # Calculate equal-weight portfolio for comparison
        equal_weights = np.array([1.0 / n_assets] * n_assets)
        equal_return = np.dot(equal_weights, mean_returns)
        equal_volatility = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))
        equal_sharpe = (equal_return - self.risk_free_rate) / equal_volatility if equal_volatility > 0 else 0

        return {
            'tickers': tickers_list,
            'optimal_weights': {ticker: round(w * 100, 2) for ticker, w in zip(tickers_list, best_weights)},
            'optimal_return': round(optimal_return * 100, 2),
            'optimal_volatility': round(optimal_volatility * 100, 2),
            'optimal_sharpe': round(best_sharpe, 3),
            'equal_weight_return': round(equal_return * 100, 2),
            'equal_weight_volatility': round(equal_volatility * 100, 2),
            'equal_weight_sharpe': round(equal_sharpe, 3),
            'improvement_sharpe': round(best_sharpe - equal_sharpe, 3),
            'correlation_matrix': cov_matrix.tolist(),
        }

    def calculate_efficient_frontier(self, tickers: List[str], period: str = '1y',
                                     n_portfolios: int = 100) -> List[Dict]:
        """Calculate efficient frontier portfolios."""

        # Get returns data
        returns_data = {}
        for ticker in tickers:
            returns = self.get_historical_returns(ticker, period)
            if len(returns) > 0:
                returns_data[ticker] = returns

        if len(returns_data) < 2:
            return []

        # Build returns matrix
        min_length = min(len(r) for r in returns_data.values())
        returns_matrix = np.array([r[-min_length:] for r in returns_data.values()]).T
        n_assets = returns_matrix.shape[1]

        # Calculate mean returns and covariance
        mean_returns = np.mean(returns_matrix, axis=0) * 252
        cov_matrix = np.cov(returns_matrix.T) * 252

        frontier = []

        # Generate random portfolios
        for _ in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)

            portfolio_return = np.dot(weights, mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0

            frontier.append({
                'return': round(portfolio_return * 100, 2),
                'volatility': round(portfolio_volatility * 100, 2),
                'sharpe': round(sharpe, 3),
                'weights': weights.tolist(),
            })

        # Sort by volatility
        frontier.sort(key=lambda x: x['volatility'])

        return frontier

    def display_optimization_results(self, results: Dict):
        """Display portfolio optimization results."""

        if not results or 'error' in results:
            print("Error: Could not optimize portfolio")
            return

        print("\n" + "="*80)
        print("PORTFOLIO OPTIMIZATION RESULTS")
        print("="*80)

        print(f"\nStocks Analyzed: {', '.join(results['tickers'])}")
        print(f"Risk-Free Rate: {self.risk_free_rate * 100:.2f}%")

        print(f"\n{'='*80}")
        print("OPTIMAL PORTFOLIO (Maximum Sharpe Ratio)")
        print("="*80)

        print(f"\nAllocations:")
        for ticker, weight in results['optimal_weights'].items():
            bar = "#" * int(weight / 2)  # Visual bar
            print(f"  {ticker:6} {weight:>6.2f}%  {bar}")

        print(f"\nExpected Performance (based on 1-year history):")
        print(f"  Annual Return:     {results['optimal_return']:>6.2f}%")
        print(f"  Annual Volatility: {results['optimal_volatility']:>6.2f}%")
        print(f"  Sharpe Ratio:      {results['optimal_sharpe']:>7.3f}")

        print(f"\n{'='*80}")
        print("EQUAL-WEIGHT PORTFOLIO (For Comparison)")
        print("="*80)

        equal_weight = 100 / len(results['tickers'])
        print(f"\nAllocations:")
        for ticker in results['tickers']:
            bar = "#" * int(equal_weight / 2)
            print(f"  {ticker:6} {equal_weight:>6.2f}%  {bar}")

        print(f"\nExpected Performance:")
        print(f"  Annual Return:     {results['equal_weight_return']:>6.2f}%")
        print(f"  Annual Volatility: {results['equal_weight_volatility']:>6.2f}%")
        print(f"  Sharpe Ratio:      {results['equal_weight_sharpe']:>7.3f}")

        print(f"\n{'='*80}")
        print("IMPROVEMENT FROM OPTIMIZATION")
        print("="*80)

        return_diff = results['optimal_return'] - results['equal_weight_return']
        vol_diff = results['optimal_volatility'] - results['equal_weight_volatility']

        print(f"\nReturn Change:     {return_diff:+.2f}%")
        print(f"Volatility Change: {vol_diff:+.2f}%")
        print(f"Sharpe Improvement: {results['improvement_sharpe']:+.3f}")

        if results['improvement_sharpe'] > 0.1:
            print(f"\n[+] Optimization provides SIGNIFICANT improvement")
            print(f"    Recommend using optimized allocation")
        elif results['improvement_sharpe'] > 0:
            print(f"\n[=] Optimization provides modest improvement")
            print(f"    Either allocation is reasonable")
        else:
            print(f"\n[-] Equal-weight performs better")
            print(f"    May indicate low correlation benefits")

        print("\n" + "="*80)


def main():
    """Main function with CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(description='Proteus Portfolio Optimizer')
    parser.add_argument('--stocks', type=str, help='Comma-separated list of tickers (e.g., MU,APP,PLTR)')
    parser.add_argument('--risk-free', type=float, default=4.5, help='Risk-free rate (%)')
    parser.add_argument('--period', type=str, default='1y', help='Historical period (e.g., 1y, 6mo)')
    parser.add_argument('--frontier', action='store_true', help='Calculate efficient frontier')

    args = parser.parse_args()

    # Default to current STRONG BUY + potential additions
    if args.stocks:
        tickers = [t.strip().upper() for t in args.stocks.split(',')]
    else:
        # Default: current recommendations that could be bought
        tickers = ['MU']  # Only MU is STRONG BUY currently
        print("Using default tickers: MU (current STRONG BUY)")
        print("Add more tickers with --stocks to optimize multi-stock portfolio")
        print("Example: --stocks MU,APP,PLTR,ALAB\n")

    optimizer = PortfolioOptimizer(risk_free_rate=args.risk_free)

    if len(tickers) == 1:
        print(f"\nNote: Only 1 stock ({tickers[0]}) - no optimization needed")
        print("Allocation: 100% {tickers[0]}")
        print("\nAdd more stocks to optimize multi-asset portfolio")
        return

    # Optimize portfolio
    print(f"\nOptimizing portfolio of {len(tickers)} stocks...")
    results = optimizer.optimize_sharpe(tickers, period=args.period)

    if results:
        optimizer.display_optimization_results(results)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"portfolio_optimization_{timestamp}.json"
        filepath = os.path.join(optimizer.results_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {filepath}")

    # Efficient frontier if requested
    if args.frontier and len(tickers) >= 2:
        print("\nCalculating efficient frontier...")
        frontier = optimizer.calculate_efficient_frontier(tickers, period=args.period)

        if frontier:
            print(f"\nEfficient Frontier ({len(frontier)} portfolios):")
            print(f"{'Return':<10} {'Volatility':<12} {'Sharpe':<10}")
            print("-" * 35)

            # Show 10 representative points
            step = len(frontier) // 10
            for i in range(0, len(frontier), step):
                p = frontier[i]
                print(f"{p['return']:>8.2f}%  {p['volatility']:>10.2f}%  {p['sharpe']:>8.3f}")


if __name__ == "__main__":
    main()

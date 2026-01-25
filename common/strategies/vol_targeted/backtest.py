"""
Backtester for Vol-Targeted Strategy

Implements realistic backtesting with:
1. Transaction costs (spread + commission)
2. Slippage modeling
3. Position limits
4. Daily mark-to-market

Key principle: Be honest about costs. Retail costs destroy most edge.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .strategy import VolTargetedStrategy


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # Transaction costs (one-way)
    spread_bps: float = 1.0          # Bid-ask spread in basis points
    commission_bps: float = 0.5      # Commission in basis points
    slippage_bps: float = 0.5        # Market impact in basis points

    # Position constraints
    initial_capital: float = 100000.0
    margin_requirement: float = 0.25  # 4x leverage max

    # Execution
    execution_delay: int = 0          # Days delay (0 = same day)

    @property
    def total_cost_bps(self) -> float:
        """Total one-way transaction cost."""
        return self.spread_bps + self.commission_bps + self.slippage_bps

    @property
    def round_trip_cost_bps(self) -> float:
        """Round-trip transaction cost."""
        return self.total_cost_bps * 2


@dataclass
class TradeRecord:
    """Record of a single trade."""
    date: datetime
    position_before: float
    position_after: float
    price: float
    trade_value: float
    cost: float
    reason: str


@dataclass
class DailyRecord:
    """Daily P&L and position record."""
    date: datetime
    price: float
    return_: float
    position: float
    gross_pnl: float
    costs: float
    net_pnl: float
    cumulative_pnl: float
    equity: float
    vol_estimate: float
    signal: float


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Summary stats
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Trade stats
    num_trades: int
    avg_trade_size: float
    total_costs: float
    turnover: float  # Annual turnover as multiple of capital

    # Detailed records
    daily_records: List[DailyRecord] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)

    # Diagnostics
    avg_position: float = 0.0
    position_std: float = 0.0
    avg_vol_estimate: float = 0.0
    pct_time_long: float = 0.0
    pct_time_short: float = 0.0
    pct_time_flat: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'num_trades': self.num_trades,
            'avg_trade_size': self.avg_trade_size,
            'total_costs': self.total_costs,
            'turnover': self.turnover,
            'avg_position': self.avg_position,
            'position_std': self.position_std,
        }

    def summary(self) -> str:
        """Generate summary string."""
        return f"""
Backtest Results
================
Performance:
  Total Return: {self.total_return:.2%}
  Annualized Return: {self.annualized_return:.2%}
  Volatility: {self.volatility:.2%}
  Sharpe Ratio: {self.sharpe_ratio:.2f}
  Max Drawdown: {self.max_drawdown:.2%}
  Calmar Ratio: {self.calmar_ratio:.2f}

Trading:
  Number of Trades: {self.num_trades}
  Avg Trade Size: {self.avg_trade_size:.2%}
  Total Costs: ${self.total_costs:,.2f}
  Annual Turnover: {self.turnover:.1f}x

Position Stats:
  Avg Position: {self.avg_position:.2f}
  Position Std: {self.position_std:.2f}
  % Time Long: {self.pct_time_long:.1%}
  % Time Short: {self.pct_time_short:.1%}
  % Time Flat: {self.pct_time_flat:.1%}
"""


class Backtester:
    """
    Realistic backtester for vol-targeted strategies.

    Key features:
    1. Transaction cost modeling (spread + commission + slippage)
    2. Daily mark-to-market
    3. Proper position sizing with capital tracking
    4. Trade record keeping for analysis
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        strategy: VolTargetedStrategy,
        prices: np.ndarray,
        dates: Optional[List[datetime]] = None
    ) -> BacktestResult:
        """
        Run backtest on price series.

        Args:
            strategy: Vol-targeted strategy to test
            prices: Array of daily prices
            dates: Optional list of dates for records

        Returns:
            BacktestResult with full statistics
        """
        if dates is None:
            dates = [datetime(2020, 1, 1) + pd.Timedelta(days=i) for i in range(len(prices))]

        # Reset strategy
        strategy.reset()

        # Initialize tracking
        daily_records: List[DailyRecord] = []
        trades: List[TradeRecord] = []
        equity = self.config.initial_capital
        cumulative_pnl = 0.0
        total_costs = 0.0
        total_turnover = 0.0
        current_position = 0.0

        # Track positions for stats
        positions = []
        vol_estimates = []
        signals = []

        for i in range(len(prices)):
            price = prices[i]
            date = dates[i]

            # Compute return
            if i > 0:
                daily_return = (price - prices[i-1]) / prices[i-1]
            else:
                daily_return = 0.0

            # Update strategy
            result = strategy.update(price, daily_return)
            new_position = result['position']

            # Track for stats
            positions.append(new_position)
            vol_estimates.append(result['vol_estimate'])
            signals.append(result['signal'])

            # Compute P&L
            if i > 0:
                # Position was set yesterday, earns today's return
                gross_pnl = current_position * daily_return * equity
            else:
                gross_pnl = 0.0

            # Handle position change
            position_change = abs(new_position - current_position)
            if position_change > 0.001:  # Meaningful trade
                # Compute transaction costs
                trade_value = position_change * equity
                cost = trade_value * self.config.total_cost_bps / 10000

                # Record trade
                trades.append(TradeRecord(
                    date=date,
                    position_before=current_position,
                    position_after=new_position,
                    price=price,
                    trade_value=trade_value,
                    cost=cost,
                    reason=result['reason']
                ))

                total_costs += cost
                total_turnover += position_change
            else:
                cost = 0.0

            # Update equity
            net_pnl = gross_pnl - cost
            cumulative_pnl += net_pnl
            equity = self.config.initial_capital + cumulative_pnl

            # Update position for next day
            current_position = new_position

            # Record daily
            daily_records.append(DailyRecord(
                date=date,
                price=price,
                return_=daily_return,
                position=current_position,
                gross_pnl=gross_pnl,
                costs=cost,
                net_pnl=net_pnl,
                cumulative_pnl=cumulative_pnl,
                equity=equity,
                vol_estimate=result['vol_estimate'],
                signal=result['signal']
            ))

        # Compute statistics
        result = self._compute_statistics(
            daily_records=daily_records,
            trades=trades,
            positions=np.array(positions),
            vol_estimates=np.array(vol_estimates),
            signals=np.array(signals),
            total_costs=total_costs,
            total_turnover=total_turnover,
        )

        return result

    def _compute_statistics(
        self,
        daily_records: List[DailyRecord],
        trades: List[TradeRecord],
        positions: np.ndarray,
        vol_estimates: np.ndarray,
        signals: np.ndarray,
        total_costs: float,
        total_turnover: float,
    ) -> BacktestResult:
        """Compute backtest statistics."""
        # Extract daily returns
        equity_series = np.array([r.equity for r in daily_records])
        equity_returns = np.diff(equity_series) / equity_series[:-1]

        # Handle edge cases
        if len(equity_returns) == 0:
            equity_returns = np.array([0.0])

        # Basic stats
        total_return = (equity_series[-1] / equity_series[0]) - 1
        trading_days = len(daily_records)
        years = trading_days / 252

        annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1
        volatility = np.std(equity_returns) * np.sqrt(252)

        if volatility > 0:
            sharpe_ratio = annualized_return / volatility
        else:
            sharpe_ratio = 0.0

        # Drawdown
        cumulative = np.cumprod(1 + equity_returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (rolling_max - cumulative) / rolling_max
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        if max_drawdown > 0:
            calmar_ratio = annualized_return / max_drawdown
        else:
            calmar_ratio = 0.0

        # Trade stats
        num_trades = len(trades)
        if num_trades > 0:
            avg_trade_size = np.mean([abs(t.position_after - t.position_before) for t in trades])
        else:
            avg_trade_size = 0.0

        annual_turnover = total_turnover / max(years, 0.01)

        # Position stats
        avg_position = np.mean(positions)
        position_std = np.std(positions)
        pct_time_long = np.mean(positions > 0.01)
        pct_time_short = np.mean(positions < -0.01)
        pct_time_flat = np.mean(np.abs(positions) <= 0.01)
        avg_vol_estimate = np.mean(vol_estimates[vol_estimates > 0])

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            num_trades=num_trades,
            avg_trade_size=avg_trade_size,
            total_costs=total_costs,
            turnover=annual_turnover,
            daily_records=daily_records,
            trades=trades,
            avg_position=avg_position,
            position_std=position_std,
            avg_vol_estimate=avg_vol_estimate,
            pct_time_long=pct_time_long,
            pct_time_short=pct_time_short,
            pct_time_flat=pct_time_flat,
        )

    def run_with_costs_comparison(
        self,
        strategy: VolTargetedStrategy,
        prices: np.ndarray,
        dates: Optional[List[datetime]] = None
    ) -> Tuple[BacktestResult, BacktestResult]:
        """
        Run backtest with and without costs for comparison.

        Returns:
            Tuple of (result_with_costs, result_without_costs)
        """
        # With costs
        result_with_costs = self.run(strategy, prices, dates)

        # Without costs
        no_cost_config = BacktestConfig(
            spread_bps=0.0,
            commission_bps=0.0,
            slippage_bps=0.0,
            initial_capital=self.config.initial_capital,
        )
        no_cost_backtester = Backtester(no_cost_config)
        result_without_costs = no_cost_backtester.run(strategy, prices, dates)

        return result_with_costs, result_without_costs


def load_spy_data(filepath: str) -> Tuple[np.ndarray, List[datetime]]:
    """
    Load SPY price data from CSV.

    Expected format: Date, Open, High, Low, Close, Volume

    Args:
        filepath: Path to CSV file

    Returns:
        Tuple of (prices, dates)
    """
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df = df.sort_values('Date')

    prices = df['Close'].values
    dates = df['Date'].tolist()

    return prices, dates

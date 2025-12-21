"""
Volatility-Based Position Sizing

Uses ATR (Average True Range) and individual stock volatility to:
1. Size positions based on dollar risk, not share count
2. Reduce position size for volatile stocks
3. Increase position size for stable stocks
4. Ensure consistent risk per trade

Key concepts:
- Risk per trade: 1-2% of portfolio
- Position size = Risk$ / (ATR * multiplier)
- Higher ATR = smaller position
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import yfinance as yf


@dataclass
class VolatilityMetrics:
    """Volatility analysis for a single stock."""
    ticker: str
    atr_14: float           # 14-day ATR
    atr_percent: float      # ATR as % of price
    volatility_20d: float   # 20-day historical volatility
    volatility_rank: float  # 0-100 percentile vs 1-year
    beta: float             # vs SPY
    current_price: float
    suggested_shares: int
    dollar_risk: float
    risk_adjusted_size: float  # Position as % of portfolio


@dataclass
class PortfolioRiskMetrics:
    """Overall portfolio risk analysis."""
    total_risk_exposure: float  # Sum of position risks
    max_position_risk: float    # Largest single position risk
    diversification_score: float  # 0-100
    sector_concentration: Dict[str, float]
    vix_level: float
    portfolio_beta: float


class VolatilitySizer:
    """
    Position sizing based on individual stock volatility.
    """

    def __init__(self, portfolio_value: float = 100000,
                 risk_per_trade: float = 0.02,
                 max_position_pct: float = 0.15):
        """
        Initialize volatility-based sizer.

        Args:
            portfolio_value: Total portfolio value
            risk_per_trade: Max risk per trade as decimal (0.02 = 2%)
            max_position_pct: Max position size as decimal (0.15 = 15%)
        """
        self.portfolio_value = portfolio_value
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.spy_data: Optional[pd.DataFrame] = None
        self._cache: Dict[str, VolatilityMetrics] = {}

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range.

        ATR = Average of True Range over period
        True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        """
        if len(data) < period + 1:
            return 0.0

        high = data['High']
        low = data['Low']
        close = data['Close']

        # Calculate true range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is the rolling mean of true range
        atr = true_range.rolling(period).mean().iloc[-1]

        return atr if not np.isnan(atr) else 0.0

    def calculate_historical_volatility(self, data: pd.DataFrame,
                                         period: int = 20) -> float:
        """
        Calculate annualized historical volatility.

        Returns volatility as a percentage.
        """
        if len(data) < period:
            return 0.0

        returns = data['Close'].pct_change().dropna()

        if len(returns) < period:
            return 0.0

        # 20-day rolling std, annualized
        daily_vol = returns.tail(period).std()
        annual_vol = daily_vol * np.sqrt(252)

        return annual_vol * 100 if not np.isnan(annual_vol) else 0.0

    def calculate_beta(self, stock_data: pd.DataFrame) -> float:
        """
        Calculate beta vs SPY.

        Beta = Covariance(stock, market) / Variance(market)
        """
        if self.spy_data is None or len(self.spy_data) < 60:
            try:
                spy = yf.Ticker("SPY")
                self.spy_data = spy.history(period="1y")
            except:
                return 1.0

        if len(stock_data) < 60 or len(self.spy_data) < 60:
            return 1.0

        # Align dates
        stock_returns = stock_data['Close'].pct_change().dropna()
        spy_returns = self.spy_data['Close'].pct_change().dropna()

        # Get common dates
        common_dates = stock_returns.index.intersection(spy_returns.index)
        if len(common_dates) < 60:
            return 1.0

        stock_ret = stock_returns.loc[common_dates].tail(60)
        spy_ret = spy_returns.loc[common_dates].tail(60)

        # Calculate beta
        covariance = np.cov(stock_ret, spy_ret)[0, 1]
        variance = np.var(spy_ret)

        if variance > 0:
            beta = covariance / variance
            return max(0.3, min(3.0, beta))  # Cap between 0.3 and 3.0

        return 1.0

    def get_volatility_rank(self, current_vol: float,
                            data: pd.DataFrame) -> float:
        """
        Calculate volatility percentile vs 1-year history.

        Returns 0-100 percentile (100 = highest volatility in past year).
        """
        if len(data) < 252:
            return 50.0

        returns = data['Close'].pct_change().dropna()

        # Calculate rolling 20-day volatility for the past year
        rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100

        # Calculate percentile
        percentile = (rolling_vol < current_vol).sum() / len(rolling_vol) * 100

        return percentile

    def calculate_position_size(self, ticker: str,
                                 current_price: float,
                                 stop_loss_pct: float = 2.0,
                                 data: Optional[pd.DataFrame] = None) -> VolatilityMetrics:
        """
        Calculate optimal position size based on volatility.

        Uses Kelly-like sizing:
        - Higher volatility = smaller position
        - Position sized to risk 1-2% of portfolio

        Args:
            ticker: Stock ticker
            current_price: Current stock price
            stop_loss_pct: Stop loss percentage (default 2%)
            data: Optional historical data (fetched if not provided)

        Returns:
            VolatilityMetrics with sizing information
        """
        # Check cache
        cache_key = f"{ticker}_{datetime.now().strftime('%Y%m%d')}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Fetch data if not provided
        if data is None:
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period="1y")
            except Exception as e:
                print(f"[WARN] Could not fetch data for {ticker}: {e}")
                # Return default sizing
                return self._default_metrics(ticker, current_price)

        if len(data) < 20:
            return self._default_metrics(ticker, current_price)

        # Calculate metrics
        atr_14 = self.calculate_atr(data, period=14)
        atr_percent = (atr_14 / current_price) * 100 if current_price > 0 else 0
        volatility_20d = self.calculate_historical_volatility(data, period=20)
        volatility_rank = self.get_volatility_rank(volatility_20d, data)
        beta = self.calculate_beta(data)

        # Calculate position size based on ATR
        # Risk amount = portfolio * risk_per_trade
        risk_amount = self.portfolio_value * self.risk_per_trade

        # Dollar risk per share = ATR * multiplier (we use 2x ATR as stop)
        # Or use explicit stop loss %
        dollar_risk_per_share = max(
            atr_14 * 2,  # 2x ATR stop
            current_price * (stop_loss_pct / 100)  # Or explicit stop
        )

        # Position size in dollars
        if dollar_risk_per_share > 0:
            position_dollars = risk_amount / dollar_risk_per_share * current_price
        else:
            position_dollars = self.portfolio_value * 0.05  # 5% default

        # Apply max position constraint
        max_position = self.portfolio_value * self.max_position_pct
        position_dollars = min(position_dollars, max_position)

        # Volatility adjustment: reduce position for high volatility stocks
        # If volatility rank > 70 (top 30%), reduce by up to 50%
        if volatility_rank > 70:
            vol_adjustment = 1.0 - (volatility_rank - 70) / 60  # 0.5 to 1.0
            position_dollars *= vol_adjustment

        # Beta adjustment: reduce position for high beta stocks
        if beta > 1.5:
            beta_adjustment = 1.0 / beta
            position_dollars *= beta_adjustment

        # Calculate shares
        suggested_shares = int(position_dollars / current_price) if current_price > 0 else 0
        suggested_shares = max(1, suggested_shares)

        # Actual dollar risk
        actual_risk = suggested_shares * dollar_risk_per_share
        risk_pct = (actual_risk / self.portfolio_value) * 100

        metrics = VolatilityMetrics(
            ticker=ticker,
            atr_14=round(atr_14, 2),
            atr_percent=round(atr_percent, 2),
            volatility_20d=round(volatility_20d, 1),
            volatility_rank=round(volatility_rank, 1),
            beta=round(beta, 2),
            current_price=round(current_price, 2),
            suggested_shares=suggested_shares,
            dollar_risk=round(actual_risk, 2),
            risk_adjusted_size=round((suggested_shares * current_price / self.portfolio_value) * 100, 2)
        )

        self._cache[cache_key] = metrics
        return metrics

    def _default_metrics(self, ticker: str, price: float) -> VolatilityMetrics:
        """Return default metrics when data unavailable."""
        default_shares = int(self.portfolio_value * 0.05 / price) if price > 0 else 1
        return VolatilityMetrics(
            ticker=ticker,
            atr_14=price * 0.02,  # Assume 2% ATR
            atr_percent=2.0,
            volatility_20d=25.0,  # Assume 25% annual vol
            volatility_rank=50.0,
            beta=1.0,
            current_price=price,
            suggested_shares=default_shares,
            dollar_risk=price * 0.02 * default_shares,
            risk_adjusted_size=5.0
        )

    def analyze_portfolio_risk(self, positions: List[Dict]) -> PortfolioRiskMetrics:
        """
        Analyze overall portfolio risk.

        Args:
            positions: List of dicts with 'ticker', 'shares', 'price', 'sector'

        Returns:
            PortfolioRiskMetrics with portfolio-level risk analysis
        """
        if not positions:
            return PortfolioRiskMetrics(
                total_risk_exposure=0,
                max_position_risk=0,
                diversification_score=0,
                sector_concentration={},
                vix_level=20,
                portfolio_beta=1.0
            )

        total_value = sum(p.get('shares', 0) * p.get('price', 0) for p in positions)
        risks = []
        betas = []
        sector_values = {}

        for pos in positions:
            ticker = pos.get('ticker', '')
            shares = pos.get('shares', 0)
            price = pos.get('price', 0)
            sector = pos.get('sector', 'Unknown')
            value = shares * price

            # Get volatility metrics
            metrics = self.calculate_position_size(ticker, price)

            # Position risk
            risk = metrics.dollar_risk if shares else 0
            risks.append(risk)

            # Weighted beta
            if total_value > 0:
                betas.append(metrics.beta * (value / total_value))

            # Sector concentration
            if sector not in sector_values:
                sector_values[sector] = 0
            sector_values[sector] += value

        # Sector percentages
        sector_pcts = {}
        for sector, value in sector_values.items():
            if total_value > 0:
                sector_pcts[sector] = round((value / total_value) * 100, 1)

        # Diversification score (based on sector concentration)
        # Perfect diversification = equal weight in all sectors
        max_concentration = max(sector_pcts.values()) if sector_pcts else 0
        diversification = max(0, 100 - max_concentration)

        # Get VIX
        try:
            vix = yf.Ticker("^VIX")
            vix_level = vix.history(period="1d")['Close'].iloc[-1]
        except:
            vix_level = 20.0

        return PortfolioRiskMetrics(
            total_risk_exposure=round(sum(risks), 2),
            max_position_risk=round(max(risks) if risks else 0, 2),
            diversification_score=round(diversification, 1),
            sector_concentration=sector_pcts,
            vix_level=round(vix_level, 1),
            portfolio_beta=round(sum(betas) if betas else 1.0, 2)
        )


def print_sizing_report(tickers: List[str], portfolio_value: float = 100000):
    """Print volatility sizing report for a list of tickers."""
    print("=" * 80)
    print("VOLATILITY-BASED POSITION SIZING")
    print("=" * 80)
    print(f"Portfolio Value: ${portfolio_value:,.0f}")
    print(f"Risk per Trade: 2%")
    print(f"Max Position: 15%")
    print()

    sizer = VolatilitySizer(portfolio_value=portfolio_value)

    print(f"{'Ticker':<8} {'Price':>10} {'ATR%':>8} {'Vol%':>8} {'Rank':>6} "
          f"{'Beta':>6} {'Shares':>8} {'Size%':>8} {'Risk$':>10}")
    print("-" * 80)

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            price = stock.history(period="1d")['Close'].iloc[-1]

            metrics = sizer.calculate_position_size(ticker, price)

            print(f"{metrics.ticker:<8} ${metrics.current_price:>9.2f} "
                  f"{metrics.atr_percent:>7.1f}% {metrics.volatility_20d:>7.1f}% "
                  f"{metrics.volatility_rank:>5.0f} {metrics.beta:>6.2f} "
                  f"{metrics.suggested_shares:>8} {metrics.risk_adjusted_size:>7.1f}% "
                  f"${metrics.dollar_risk:>9.2f}")
        except Exception as e:
            print(f"{ticker:<8} ERROR: {e}")

    print()


if __name__ == "__main__":
    # Test with sample tickers
    test_tickers = [
        "NVDA", "AVGO", "MSFT", "META", "ORCL",
        "JPM", "V", "MA", "GILD", "JNJ"
    ]
    print_sizing_report(test_tickers)

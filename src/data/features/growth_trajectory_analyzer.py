"""
Growth Trajectory Analyzer

Analyzes growth trajectories for IPO companies:
- Market cap evolution since IPO
- Revenue growth curves and acceleration
- Category TAM expansion
- "Doing things that don't scale" → Scale inflection detection

Y Combinator Principle: Identify inflection points where companies
transition from early-stage hustle to scalable growth machines.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf


class GrowthTrajectoryAnalyzer:
    """
    Analyzes growth trajectories and inflection points.
    """

    def __init__(self):
        """Initialize growth trajectory analyzer."""
        pass

    def get_price_history_since_ipo(self, ticker: str, ipo_date: datetime) -> pd.DataFrame:
        """
        Get price history since IPO.

        Args:
            ticker: Stock ticker
            ipo_date: IPO date

        Returns:
            DataFrame with price history
        """
        try:
            stock = yf.Ticker(ticker)

            # Fetch historical data from IPO date to now
            end_date = datetime.now()
            hist = stock.history(start=ipo_date, end=end_date)

            if hist.empty:
                return pd.DataFrame()

            hist.reset_index(inplace=True)
            return hist

        except Exception as e:
            print(f"[ERROR] Could not fetch price history for {ticker}: {e}")
            return pd.DataFrame()

    def calculate_market_cap_trajectory(self, ticker: str, ipo_date: datetime,
                                       current_market_cap: float) -> Dict:
        """
        Calculate market cap trajectory since IPO.

        Analyzes:
        - Total return since IPO
        - Volatility (max drawdown, highs/lows)
        - Growth consistency
        - Current valuation vs. IPO

        Args:
            ticker: Stock ticker
            ipo_date: IPO date
            current_market_cap: Current market cap

        Returns:
            Dictionary with trajectory metrics
        """
        price_hist = self.get_price_history_since_ipo(ticker, ipo_date)

        if price_hist.empty:
            return {
                'has_data': False,
                'ticker': ticker,
            }

        # IPO price (first close)
        ipo_price = price_hist['Close'].iloc[0]
        current_price = price_hist['Close'].iloc[-1]

        # Total return
        total_return = (current_price / ipo_price - 1) * 100

        # High/low analysis
        all_time_high = price_hist['Close'].max()
        all_time_low = price_hist['Close'].min()

        ath_return = (all_time_high / ipo_price - 1) * 100
        max_drawdown = (all_time_low / all_time_high - 1) * 100

        # Current position vs. ATH
        current_vs_ath = (current_price / all_time_high - 1) * 100

        # Days since IPO
        days_since_ipo = (datetime.now() - ipo_date).days
        months_since_ipo = days_since_ipo / 30.44

        # Annualized return
        years = days_since_ipo / 365.25
        if years > 0:
            annualized_return = (((current_price / ipo_price) ** (1/years)) - 1) * 100
        else:
            annualized_return = 0

        # Volatility (std dev of daily returns)
        daily_returns = price_hist['Close'].pct_change().dropna()
        volatility = daily_returns.std() * 100
        annualized_volatility = volatility * np.sqrt(252)  # 252 trading days/year

        return {
            'has_data': True,
            'ticker': ticker,
            'ipo_price': ipo_price,
            'current_price': current_price,
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'ath_price': all_time_high,
            'ath_return_pct': ath_return,
            'max_drawdown_pct': max_drawdown,
            'current_vs_ath_pct': current_vs_ath,
            'days_since_ipo': days_since_ipo,
            'months_since_ipo': months_since_ipo,
            'daily_volatility_pct': volatility,
            'annualized_volatility_pct': annualized_volatility,
            'is_above_ipo': current_price > ipo_price,
            'is_near_ath': current_vs_ath > -10,  # Within 10% of ATH
        }

    def analyze_revenue_growth_acceleration(self, ticker: str) -> Dict:
        """
        Analyze revenue growth acceleration/deceleration.

        Detects:
        - Accelerating growth (getting faster)
        - Decelerating growth (slowing down)
        - Inflection points

        Args:
            ticker: Stock ticker

        Returns:
            Growth acceleration analysis
        """
        try:
            stock = yf.Ticker(ticker)
            quarterly_financials = stock.quarterly_financials

            if quarterly_financials.empty:
                return {'has_data': False}

            # Extract revenue
            if 'Total Revenue' not in quarterly_financials.index:
                return {'has_data': False}

            revenue_series = quarterly_financials.loc['Total Revenue'].sort_index()

            if len(revenue_series) < 4:
                return {'has_data': False}

            # Calculate QoQ and YoY growth rates
            qoq_growth = revenue_series.pct_change() * 100
            yoy_growth = revenue_series.pct_change(periods=4) * 100  # vs. 4 quarters ago

            # Growth acceleration (is growth rate increasing?)
            growth_acceleration = yoy_growth.diff()

            # Recent trends (last 4 quarters)
            recent_yoy = yoy_growth.tail(4)
            recent_acceleration = growth_acceleration.tail(4)

            # Determine trend
            is_accelerating = recent_acceleration.mean() > 0
            is_high_growth = recent_yoy.mean() > 30  # >30% YoY

            # Detect inflection point (recent acceleration after deceleration)
            inflection_detected = False
            if len(recent_acceleration) >= 3:
                # Check if growth went from decelerating to accelerating
                if recent_acceleration.iloc[-2] < 0 and recent_acceleration.iloc[-1] > 0:
                    inflection_detected = True

            return {
                'has_data': True,
                'ticker': ticker,
                'latest_qoq_growth_pct': qoq_growth.iloc[-1] if len(qoq_growth) > 0 else None,
                'latest_yoy_growth_pct': yoy_growth.iloc[-1] if len(yoy_growth) > 0 else None,
                'avg_yoy_growth_pct': recent_yoy.mean(),
                'growth_acceleration_trend': recent_acceleration.mean(),
                'is_accelerating': is_accelerating,
                'is_high_growth': is_high_growth,
                'inflection_detected': inflection_detected,
                'quarters_analyzed': len(revenue_series),
            }

        except Exception as e:
            print(f"[ERROR] Could not analyze revenue growth for {ticker}: {e}")
            return {'has_data': False}

    def detect_scaling_signals(self, ticker: str, fundamentals: Dict) -> Dict:
        """
        Detect "doing things that don't scale" → scale transition.

        Signals:
        - Improving unit economics (margins expanding)
        - Accelerating growth + profitability
        - Operating leverage kicking in
        - Customer acquisition efficiency

        Args:
            ticker: Stock ticker
            fundamentals: Fundamental metrics

        Returns:
            Scaling signals dictionary
        """
        signals = []
        score = 0

        # 1. Profitability at scale
        operating_margin = fundamentals.get('operating_margin', -1) or -1
        gross_margin = fundamentals.get('gross_margin', 0) or 0

        if operating_margin > 0.15:  # 15%+ operating margin
            signals.append("Strong profitability at scale (15%+ operating margin)")
            score += 30
        elif operating_margin > 0:
            signals.append("Profitable operations (positive operating margin)")
            score += 20

        # 2. High gross margins (good unit economics)
        if gross_margin > 0.7:  # 70%+ gross margin
            signals.append(f"Excellent unit economics ({gross_margin*100:.1f}% gross margin)")
            score += 25
        elif gross_margin > 0.5:
            signals.append(f"Good unit economics ({gross_margin*100:.1f}% gross margin)")
            score += 15

        # 3. Revenue growth still strong (not sacrificing growth for profitability)
        revenue_growth = fundamentals.get('revenue_growth_yoy', 0) or 0
        if revenue_growth > 0.4 and operating_margin > 0:  # 40%+ growth + profitable
            signals.append("Scaling efficiently (high growth + profitable)")
            score += 30
        elif revenue_growth > 0.3:
            signals.append("Strong revenue growth maintained")
            score += 15

        # 4. Efficiency metrics
        # Revenue per employee (if available)
        # High revenue per employee = leverage/automation

        is_scaling = score >= 60

        return {
            'is_scaling': is_scaling,
            'scaling_score': min(100, score),
            'scaling_signals': signals,
            'operating_margin': operating_margin,
            'gross_margin': gross_margin,
            'revenue_growth': revenue_growth,
        }

    def estimate_tam_expansion(self, category: str, revenue_growth: float,
                              market_cap: float) -> Dict:
        """
        Estimate TAM (Total Addressable Market) expansion.

        Y Combinator principle: Small market today → Massive market tomorrow

        Args:
            category: Company category
            revenue_growth: Current revenue growth rate
            market_cap: Current market cap

        Returns:
            TAM expansion estimate
        """
        # Category TAM estimates (billions USD)
        category_tams = {
            'Social Media': {'current': 200, 'future': 500, 'years': 5},
            'Cloud Infrastructure': {'current': 150, 'future': 500, 'years': 5},
            'Cybersecurity': {'current': 200, 'future': 400, 'years': 5},
            'Fintech': {'current': 300, 'future': 1000, 'years': 7},
            'E-commerce': {'current': 5000, 'future': 8000, 'years': 5},
            'Gaming': {'current': 200, 'future': 400, 'years': 5},
            'Food Delivery': {'current': 150, 'future': 350, 'years': 5},
            'EV': {'current': 500, 'future': 2000, 'years': 10},
            'Biotechnology': {'current': 500, 'future': 1500, 'years': 10},
            'AI/ML': {'current': 100, 'future': 1000, 'years': 7},
            'Developer Tools': {'current': 50, 'future': 200, 'years': 5},
        }

        # Default for unknown categories
        default_tam = {'current': 100, 'future': 300, 'years': 5}

        # Find matching category
        tam_data = default_tam
        for cat_name, tam_info in category_tams.items():
            if cat_name.lower() in category.lower() or category.lower() in cat_name.lower():
                tam_data = tam_info
                break

        current_tam = tam_data['current']
        future_tam = tam_data['future']
        years_to_future = tam_data['years']

        # Calculate TAM CAGR
        tam_cagr = ((future_tam / current_tam) ** (1 / years_to_future) - 1) * 100

        # Market cap as % of current TAM
        market_cap_b = market_cap / 1e9
        penetration = (market_cap_b / current_tam) * 100 if current_tam > 0 else 0

        # Is company growing faster than market?
        company_growth_pct = revenue_growth * 100
        outpacing_market = company_growth_pct > tam_cagr

        return {
            'category': category,
            'current_tam_b': current_tam,
            'future_tam_b': future_tam,
            'years_to_future': years_to_future,
            'tam_cagr_pct': tam_cagr,
            'market_penetration_pct': penetration,
            'company_growth_pct': company_growth_pct,
            'outpacing_market_growth': outpacing_market,
            'tam_expansion_potential_x': future_tam / current_tam,
        }

    def analyze_full_trajectory(self, ticker: str, company_info: Dict,
                               fundamentals: Dict, category: str) -> Dict:
        """
        Complete growth trajectory analysis.

        Args:
            ticker: Stock ticker
            company_info: Company information
            fundamentals: Fundamental metrics
            category: Company category

        Returns:
            Comprehensive trajectory analysis
        """
        print(f"\n[GROWTH TRAJECTORY ANALYSIS] {ticker}")
        print("=" * 70)

        results = {'ticker': ticker, 'category': category}

        # 1. Market cap trajectory
        ipo_date = company_info.get('ipo_date')
        market_cap = company_info.get('market_cap', 0)

        if ipo_date:
            trajectory = self.calculate_market_cap_trajectory(ticker, ipo_date, market_cap)
            results.update(trajectory)

            if trajectory.get('has_data'):
                print(f"Market Cap Trajectory:")
                print(f"  IPO Price: ${trajectory['ipo_price']:.2f}")
                print(f"  Current Price: ${trajectory['current_price']:.2f}")
                print(f"  Total Return: {trajectory['total_return_pct']:+.1f}%")
                print(f"  Annualized Return: {trajectory['annualized_return_pct']:+.1f}%")
                print(f"  ATH Return: {trajectory['ath_return_pct']:+.1f}%")
                print(f"  Max Drawdown: {trajectory['max_drawdown_pct']:.1f}%")

        # 2. Revenue growth acceleration
        growth_accel = self.analyze_revenue_growth_acceleration(ticker)
        results.update(growth_accel)

        if growth_accel.get('has_data'):
            print(f"\nRevenue Growth:")
            print(f"  Latest YoY Growth: {growth_accel['latest_yoy_growth_pct']:.1f}%")
            print(f"  Growth Accelerating: {growth_accel['is_accelerating']}")
            print(f"  Inflection Detected: {growth_accel['inflection_detected']}")

        # 3. Scaling signals
        scaling = self.detect_scaling_signals(ticker, fundamentals)
        results.update(scaling)

        print(f"\nScaling Signals:")
        print(f"  Scaling Score: {scaling['scaling_score']:.1f}/100")
        for signal in scaling['scaling_signals']:
            print(f"    - {signal}")

        # 4. TAM expansion
        revenue_growth = fundamentals.get('revenue_growth_yoy', 0) or 0
        tam = self.estimate_tam_expansion(category, revenue_growth, market_cap)
        results.update(tam)

        print(f"\nTAM Expansion:")
        print(f"  Current TAM: ${tam['current_tam_b']:.0f}B -> Future: ${tam['future_tam_b']:.0f}B")
        print(f"  TAM CAGR: {tam['tam_cagr_pct']:.1f}%")
        print(f"  Company Growth: {tam['company_growth_pct']:.1f}% (Outpacing: {tam['outpacing_market_growth']})")
        print(f"  Market Penetration: {tam['market_penetration_pct']:.1f}%")

        print("=" * 70)

        return results


if __name__ == "__main__":
    # Test growth trajectory analyzer
    import sys
    sys.path.insert(0, '../../..')
    from src.data.fetchers.ipo_data_fetcher import IPODataFetcher

    analyzer = GrowthTrajectoryAnalyzer()
    fetcher = IPODataFetcher()

    # Test with RDDT (Reddit)
    ticker = 'RDDT'
    category = 'Social Media'

    # Get company data
    company_info = fetcher.get_company_info(ticker)
    fundamentals = fetcher.get_fundamental_metrics(ticker)

    # Full analysis
    analysis = analyzer.analyze_full_trajectory(ticker, company_info, fundamentals, category)

    print("\n" + "=" * 70)
    print("TRAJECTORY SUMMARY")
    print("=" * 70)
    print(f"Total Return Since IPO: {analysis.get('total_return_pct', 0):+.1f}%")
    print(f"Revenue Growth Accelerating: {analysis.get('is_accelerating', False)}")
    print(f"At Scaling Inflection: {analysis.get('is_scaling', False)}")
    print(f"TAM Expansion Potential: {analysis.get('tam_expansion_potential_x', 1):.1f}x")

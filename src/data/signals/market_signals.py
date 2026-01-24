"""
Enhanced Market Signals

Implements high-edge signals identified from quant research:
1. Real rates (10Y minus inflation expectations)
2. Credit spreads (HY OAS)
3. Sentiment (CNN Fear/Greed)
4. Breadth (% stocks above MA, new highs/lows)
5. Cross-asset correlations (SPY-TLT, SPY-DXY)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
import yfinance as yf
import logging

logging.getLogger('yfinance').setLevel(logging.CRITICAL)

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False


@dataclass
class MarketSignals:
    """Comprehensive market signal snapshot."""
    timestamp: str

    # Regime signals
    real_rate: float  # 10Y minus inflation expectations
    real_rate_signal: str  # 'risk_on', 'risk_off', 'neutral'

    # Credit signals
    hy_spread: float  # High yield OAS
    credit_signal: str  # 'stress', 'normal', 'complacent'

    # Sentiment signals
    fear_greed: int  # 0-100
    fear_greed_signal: str  # 'extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed'

    # Breadth signals
    breadth_pct: float  # % stocks above 20d MA
    new_high_ratio: float  # new highs / (new highs + new lows)
    breadth_signal: str  # 'strong', 'healthy', 'weak', 'deteriorating'

    # Cross-asset signals
    spy_tlt_corr: float  # 30d correlation
    dollar_trend: str  # 'rising', 'falling', 'flat'

    # Combined assessment
    overall_signal: str  # 'strong_buy', 'buy', 'neutral', 'sell', 'strong_sell'
    confidence: float  # 0-1
    factors: List[str]  # List of contributing factors


class MarketSignalGenerator:
    """Generate comprehensive market signals."""

    def __init__(self, fred_api_key: str = None):
        self.fred = None
        if FRED_AVAILABLE and fred_api_key:
            try:
                self.fred = Fred(api_key=fred_api_key)
            except:
                pass

    def get_real_rate(self) -> Tuple[float, str]:
        """
        Calculate real rate: 10Y Treasury minus inflation expectations.

        Signals:
        - Real rate < 0: Risk ON (stocks benefit)
        - Real rate > 2%: Risk OFF (headwind for stocks)
        """
        try:
            # Use yfinance for Treasury yield
            tnx = yf.Ticker("^TNX")  # 10Y yield
            data = tnx.history(period="5d")
            if len(data) == 0:
                return 0.0, 'neutral'

            ten_year = data['Close'].iloc[-1]

            # Approximate inflation expectations from TIP spread
            # TIP ETF vs nominal bonds
            tip = yf.Ticker("TIP")
            tip_data = tip.history(period="5d")

            # Simplified: assume 2.5% inflation expectation if can't calculate
            inflation_exp = 2.5

            real_rate = ten_year - inflation_exp

            if real_rate < 0:
                signal = 'risk_on'
            elif real_rate > 2.0:
                signal = 'risk_off'
            else:
                signal = 'neutral'

            return round(real_rate, 2), signal
        except Exception as e:
            return 0.0, 'neutral'

    def get_credit_spread(self) -> Tuple[float, str]:
        """
        Get high yield credit spread as stress indicator.

        Using HYG (High Yield ETF) vs LQD (Investment Grade) as proxy.
        """
        try:
            hyg = yf.Ticker("HYG")
            lqd = yf.Ticker("LQD")

            hyg_data = hyg.history(period="30d")
            lqd_data = lqd.history(period="30d")

            if len(hyg_data) < 5 or len(lqd_data) < 5:
                return 0.0, 'normal'

            # Calculate yield difference (approximate from price performance)
            hyg_ret = (hyg_data['Close'].iloc[-1] / hyg_data['Close'].iloc[0] - 1) * 100
            lqd_ret = (lqd_data['Close'].iloc[-1] / lqd_data['Close'].iloc[0] - 1) * 100

            spread_proxy = lqd_ret - hyg_ret  # Positive = stress (HY underperforming)

            # Also check absolute levels
            hyg_vol = hyg_data['Close'].pct_change().std() * np.sqrt(252) * 100

            if hyg_vol > 20 or spread_proxy > 2:
                signal = 'stress'
            elif hyg_vol < 8 and spread_proxy < -1:
                signal = 'complacent'
            else:
                signal = 'normal'

            return round(spread_proxy, 2), signal
        except:
            return 0.0, 'normal'

    def get_fear_greed(self) -> Tuple[int, str]:
        """
        Get CNN Fear & Greed Index.

        0-25: Extreme Fear (contrarian buy)
        25-45: Fear
        45-55: Neutral
        55-75: Greed
        75-100: Extreme Greed (contrarian sell)
        """
        try:
            # Try the API
            response = requests.get(
                'https://production.dataviz.cnn.io/index/fearandgreed/graphdata',
                timeout=10,
                headers={'User-Agent': 'Mozilla/5.0'}
            )

            if response.status_code == 200:
                data = response.json()
                score = int(data.get('fear_and_greed', {}).get('score', 50))
            else:
                # Fallback: estimate from VIX
                vix = yf.Ticker("^VIX")
                vix_data = vix.history(period="5d")
                if len(vix_data) > 0:
                    vix_level = vix_data['Close'].iloc[-1]
                    # VIX to Fear/Greed approximation
                    score = max(0, min(100, int(100 - (vix_level - 12) * 3)))
                else:
                    score = 50

            if score <= 25:
                signal = 'extreme_fear'
            elif score <= 45:
                signal = 'fear'
            elif score <= 55:
                signal = 'neutral'
            elif score <= 75:
                signal = 'greed'
            else:
                signal = 'extreme_greed'

            return score, signal
        except:
            return 50, 'neutral'

    def get_breadth(self, tickers: List[str] = None) -> Tuple[float, float, str]:
        """
        Calculate market breadth from stock universe.

        Returns:
        - breadth_pct: % of stocks above 20d MA
        - new_high_ratio: new 52w highs / (highs + lows)
        - signal: breadth health assessment
        """
        if tickers is None:
            # Default to major stocks
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
                      'JPM', 'V', 'JNJ', 'UNH', 'PG', 'HD', 'MA', 'XOM',
                      'CVX', 'PFE', 'ABBV', 'KO', 'PEP', 'MRK', 'COST', 'WMT']

        try:
            above_ma = 0
            new_highs = 0
            new_lows = 0
            valid = 0

            for ticker in tickers[:30]:  # Limit for speed
                try:
                    stock = yf.Ticker(ticker)
                    data = stock.history(period="1y")

                    if len(data) < 60:
                        continue

                    valid += 1
                    current = data['Close'].iloc[-1]
                    ma20 = data['Close'].rolling(20).mean().iloc[-1]
                    high_52w = data['Close'].max()
                    low_52w = data['Close'].min()

                    if current > ma20:
                        above_ma += 1

                    # Within 2% of 52w high/low
                    if current >= high_52w * 0.98:
                        new_highs += 1
                    if current <= low_52w * 1.02:
                        new_lows += 1
                except:
                    continue

            if valid == 0:
                return 50.0, 0.5, 'neutral'

            breadth_pct = (above_ma / valid) * 100

            if new_highs + new_lows > 0:
                new_high_ratio = new_highs / (new_highs + new_lows)
            else:
                new_high_ratio = 0.5

            # Determine signal
            if breadth_pct >= 70 and new_high_ratio >= 0.6:
                signal = 'strong'
            elif breadth_pct >= 50:
                signal = 'healthy'
            elif breadth_pct >= 30:
                signal = 'weak'
            else:
                signal = 'deteriorating'

            return round(breadth_pct, 1), round(new_high_ratio, 2), signal
        except:
            return 50.0, 0.5, 'neutral'

    def get_cross_asset_signals(self) -> Tuple[float, str]:
        """
        Calculate cross-asset correlations and dollar trend.

        SPY-TLT correlation:
        - High positive: macro-driven (risk-off)
        - Negative: normal (diversification working)

        Dollar trend affects exporters.
        """
        try:
            spy = yf.Ticker("SPY")
            tlt = yf.Ticker("TLT")
            uup = yf.Ticker("UUP")  # Dollar ETF

            spy_data = spy.history(period="60d")
            tlt_data = tlt.history(period="60d")
            uup_data = uup.history(period="60d")

            if len(spy_data) < 30:
                return 0.0, 'flat'

            # Calculate correlation
            spy_ret = spy_data['Close'].pct_change().dropna()
            tlt_ret = tlt_data['Close'].pct_change().dropna()

            # Align indices
            aligned = pd.concat([spy_ret, tlt_ret], axis=1).dropna()
            if len(aligned) < 20:
                return 0.0, 'flat'

            aligned.columns = ['spy', 'tlt']
            corr = aligned['spy'].rolling(30).corr(aligned['tlt']).iloc[-1]

            # Dollar trend (20d)
            if len(uup_data) >= 20:
                dollar_change = (uup_data['Close'].iloc[-1] / uup_data['Close'].iloc[-20] - 1) * 100
                if dollar_change > 1:
                    dollar_trend = 'rising'
                elif dollar_change < -1:
                    dollar_trend = 'falling'
                else:
                    dollar_trend = 'flat'
            else:
                dollar_trend = 'flat'

            return round(corr, 2), dollar_trend
        except:
            return 0.0, 'flat'

    def generate_signals(self, stock_universe: List[str] = None) -> MarketSignals:
        """Generate comprehensive market signals."""

        # Get all signals
        real_rate, real_rate_signal = self.get_real_rate()
        hy_spread, credit_signal = self.get_credit_spread()
        fear_greed, fg_signal = self.get_fear_greed()
        breadth_pct, nh_ratio, breadth_signal = self.get_breadth(stock_universe)
        spy_tlt_corr, dollar_trend = self.get_cross_asset_signals()

        # Calculate overall signal
        factors = []
        score = 0  # -100 to +100

        # Real rate contribution
        if real_rate_signal == 'risk_on':
            score += 20
            factors.append(f"Real rates negative ({real_rate:.1f}%) - supportive")
        elif real_rate_signal == 'risk_off':
            score -= 20
            factors.append(f"Real rates high ({real_rate:.1f}%) - headwind")

        # Credit contribution
        if credit_signal == 'stress':
            score -= 25
            factors.append("Credit stress detected - risk off")
        elif credit_signal == 'complacent':
            score += 10
            factors.append("Credit complacent - risk on")

        # Sentiment contribution (contrarian)
        if fg_signal == 'extreme_fear':
            score += 30
            factors.append(f"Extreme fear ({fear_greed}) - contrarian bullish")
        elif fg_signal == 'fear':
            score += 15
            factors.append(f"Fear ({fear_greed}) - mildly bullish")
        elif fg_signal == 'extreme_greed':
            score -= 25
            factors.append(f"Extreme greed ({fear_greed}) - contrarian bearish")
        elif fg_signal == 'greed':
            score -= 10
            factors.append(f"Greed ({fear_greed}) - caution")

        # Breadth contribution
        if breadth_signal == 'strong':
            score += 15
            factors.append(f"Strong breadth ({breadth_pct:.0f}% above MA)")
        elif breadth_signal == 'deteriorating':
            score -= 20
            factors.append(f"Weak breadth ({breadth_pct:.0f}% above MA) - warning")

        # Cross-asset contribution
        if spy_tlt_corr > 0.5:
            score -= 10
            factors.append(f"High stock-bond correlation ({spy_tlt_corr:.2f}) - macro risk")

        # Determine overall signal
        if score >= 40:
            overall = 'strong_buy'
        elif score >= 15:
            overall = 'buy'
        elif score >= -15:
            overall = 'neutral'
        elif score >= -40:
            overall = 'sell'
        else:
            overall = 'strong_sell'

        confidence = min(1.0, abs(score) / 50)

        return MarketSignals(
            timestamp=datetime.now().isoformat(),
            real_rate=real_rate,
            real_rate_signal=real_rate_signal,
            hy_spread=hy_spread,
            credit_signal=credit_signal,
            fear_greed=fear_greed,
            fear_greed_signal=fg_signal,
            breadth_pct=breadth_pct,
            new_high_ratio=nh_ratio,
            breadth_signal=breadth_signal,
            spy_tlt_corr=spy_tlt_corr,
            dollar_trend=dollar_trend,
            overall_signal=overall,
            confidence=confidence,
            factors=factors
        )


def print_market_signals(signals: MarketSignals):
    """Print formatted market signals."""
    print("=" * 70)
    print("ENHANCED MARKET SIGNALS")
    print("=" * 70)
    print(f"Generated: {signals.timestamp[:19]}")
    print()

    print("MACRO SIGNALS")
    print("-" * 40)
    print(f"  Real Rate: {signals.real_rate:+.2f}% ({signals.real_rate_signal})")
    print(f"  Credit: {signals.credit_signal.upper()}")
    print()

    print("SENTIMENT SIGNALS")
    print("-" * 40)
    print(f"  Fear & Greed: {signals.fear_greed}/100 ({signals.fear_greed_signal})")
    print()

    print("BREADTH SIGNALS")
    print("-" * 40)
    print(f"  Stocks Above 20d MA: {signals.breadth_pct:.0f}%")
    print(f"  New High Ratio: {signals.new_high_ratio:.0%}")
    print(f"  Breadth Health: {signals.breadth_signal.upper()}")
    print()

    print("CROSS-ASSET SIGNALS")
    print("-" * 40)
    print(f"  SPY-TLT Correlation: {signals.spy_tlt_corr:.2f}")
    print(f"  Dollar Trend: {signals.dollar_trend.upper()}")
    print()

    print("=" * 70)
    print(f"OVERALL SIGNAL: {signals.overall_signal.upper()}")
    print(f"CONFIDENCE: {signals.confidence:.0%}")
    print("=" * 70)
    print()

    if signals.factors:
        print("CONTRIBUTING FACTORS:")
        for f in signals.factors:
            print(f"  • {f}")
    print()


if __name__ == '__main__':
    generator = MarketSignalGenerator()
    signals = generator.generate_signals()
    print_market_signals(signals)

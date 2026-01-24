"""
Technical Analysis Layer
=========================

Adds chart-based entry/exit signals to complement fundamental analysis.

For our top 4 picks, analyzes:
1. Support & Resistance Levels
2. Trend Analysis (uptrend/downtrend/sideways)
3. Momentum Indicators (RSI, MACD)
4. Moving Averages (50-day, 200-day)
5. Volume Analysis
6. Entry/Exit Recommendations
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List


class TechnicalAnalyzer:
    """Performs technical analysis on stocks."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.df = None

    def fetch_data(self, period: str = "1y") -> pd.DataFrame:
        """Fetch historical price data."""
        self.df = self.stock.history(period=period)
        return self.df

    def calculate_rsi(self, period: int = 14) -> float:
        """Calculate Relative Strength Index (RSI)."""
        if self.df is None or len(self.df) < period:
            return 50.0

        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1]

    def calculate_macd(self) -> Tuple[float, float, str]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        Returns (MACD, Signal, Status)
        """
        if self.df is None or len(self.df) < 26:
            return 0.0, 0.0, "NEUTRAL"

        exp1 = self.df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()

        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]

        if current_macd > current_signal:
            status = "BULLISH"
        elif current_macd < current_signal:
            status = "BEARISH"
        else:
            status = "NEUTRAL"

        return current_macd, current_signal, status

    def calculate_moving_averages(self) -> Dict:
        """Calculate key moving averages."""
        if self.df is None:
            return {}

        current_price = self.df['Close'].iloc[-1]

        mas = {}
        for period in [20, 50, 200]:
            if len(self.df) >= period:
                ma = self.df['Close'].rolling(window=period).mean().iloc[-1]
                mas[f'MA{period}'] = ma
                mas[f'Price_vs_MA{period}'] = ((current_price - ma) / ma) * 100

        return mas

    def find_support_resistance(self, window: int = 20) -> Dict:
        """
        Find support and resistance levels using local extrema.
        """
        if self.df is None or len(self.df) < window * 2:
            return {}

        # Get recent data (last 6 months for support/resistance)
        recent_df = self.df.tail(120)  # ~6 months of trading days

        highs = recent_df['High'].values
        lows = recent_df['Low'].values
        closes = recent_df['Close'].values

        # Find local maxima (resistance)
        resistance_levels = []
        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i - window:i + window]):
                resistance_levels.append(highs[i])

        # Find local minima (support)
        support_levels = []
        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i - window:i + window]):
                support_levels.append(lows[i])

        current_price = closes[-1]

        # Get nearest support and resistance
        support_levels = [s for s in support_levels if s < current_price]
        resistance_levels = [r for r in resistance_levels if r > current_price]

        nearest_support = max(support_levels) if support_levels else current_price * 0.9
        nearest_resistance = min(resistance_levels) if resistance_levels else current_price * 1.1

        return {
            'current_price': current_price,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_distance': ((current_price - nearest_support) / current_price) * 100,
            'resistance_distance': ((nearest_resistance - current_price) / current_price) * 100,
            'all_support': sorted(support_levels, reverse=True)[:3],
            'all_resistance': sorted(resistance_levels)[:3],
        }

    def analyze_trend(self) -> Dict:
        """Determine overall trend."""
        if self.df is None or len(self.df) < 50:
            return {'trend': 'UNKNOWN'}

        current_price = self.df['Close'].iloc[-1]
        ma20 = self.df['Close'].rolling(window=20).mean().iloc[-1]
        ma50 = self.df['Close'].rolling(window=50).mean().iloc[-1]

        # Trend determination
        if len(self.df) >= 200:
            ma200 = self.df['Close'].rolling(window=200).mean().iloc[-1]
            if current_price > ma20 > ma50 > ma200:
                trend = "STRONG UPTREND"
            elif current_price > ma50 > ma200:
                trend = "UPTREND"
            elif current_price < ma20 < ma50 < ma200:
                trend = "STRONG DOWNTREND"
            elif current_price < ma50 < ma200:
                trend = "DOWNTREND"
            else:
                trend = "SIDEWAYS"
        else:
            if current_price > ma20 > ma50:
                trend = "UPTREND"
            elif current_price < ma20 < ma50:
                trend = "DOWNTREND"
            else:
                trend = "SIDEWAYS"

        # Calculate trend strength (rate of change over 20 days)
        price_20d_ago = self.df['Close'].iloc[-20]
        roc = ((current_price - price_20d_ago) / price_20d_ago) * 100

        return {
            'trend': trend,
            'roc_20d': roc,
        }

    def analyze_volume(self) -> Dict:
        """Analyze volume patterns."""
        if self.df is None:
            return {}

        recent_volume = self.df['Volume'].iloc[-1]
        avg_volume_20d = self.df['Volume'].rolling(window=20).mean().iloc[-1]
        avg_volume_50d = self.df['Volume'].rolling(window=50).mean().iloc[-1]

        volume_vs_20d = ((recent_volume - avg_volume_20d) / avg_volume_20d) * 100
        volume_vs_50d = ((recent_volume - avg_volume_50d) / avg_volume_50d) * 100

        if volume_vs_20d > 50:
            volume_status = "HIGH (Unusual activity)"
        elif volume_vs_20d > 20:
            volume_status = "ABOVE AVERAGE"
        elif volume_vs_20d < -20:
            volume_status = "BELOW AVERAGE"
        else:
            volume_status = "NORMAL"

        return {
            'recent_volume': recent_volume,
            'avg_volume_20d': avg_volume_20d,
            'volume_vs_20d': volume_vs_20d,
            'volume_status': volume_status,
        }

    def get_entry_exit_signals(self, dcf_fair_value: float = None) -> Dict:
        """
        Generate entry/exit recommendations based on technical signals.
        """
        rsi = self.calculate_rsi()
        macd, signal, macd_status = self.calculate_macd()
        mas = self.calculate_moving_averages()
        sr = self.find_support_resistance()
        trend = self.analyze_trend()
        volume = self.analyze_volume()

        current_price = self.df['Close'].iloc[-1]

        # Score signals
        bullish_signals = 0
        bearish_signals = 0

        # RSI signals
        if rsi < 30:
            bullish_signals += 2  # Oversold
        elif rsi < 40:
            bullish_signals += 1  # Getting oversold
        elif rsi > 70:
            bearish_signals += 2  # Overbought
        elif rsi > 60:
            bearish_signals += 1  # Getting overbought

        # MACD signals
        if macd_status == "BULLISH":
            bullish_signals += 1
        elif macd_status == "BEARISH":
            bearish_signals += 1

        # Trend signals
        if "UPTREND" in trend['trend']:
            bullish_signals += 1
        elif "DOWNTREND" in trend['trend']:
            bearish_signals += 1

        # Moving average signals
        if mas.get('Price_vs_MA50', 0) > 0:
            bullish_signals += 1
        else:
            bearish_signals += 1

        # Support/resistance position
        if sr['support_distance'] < 5:  # Near support
            bullish_signals += 1
        if sr['resistance_distance'] < 5:  # Near resistance
            bearish_signals += 1

        # Generate recommendation
        total_signals = bullish_signals + bearish_signals
        if total_signals == 0:
            recommendation = "HOLD"
        else:
            bullish_pct = (bullish_signals / total_signals) * 100

            if bullish_pct >= 70:
                recommendation = "STRONG BUY"
            elif bullish_pct >= 55:
                recommendation = "BUY"
            elif bullish_pct >= 45:
                recommendation = "HOLD"
            elif bullish_pct >= 30:
                recommendation = "REDUCE"
            else:
                recommendation = "SELL"

        # DCF adjustment
        dcf_adjustment = ""
        if dcf_fair_value:
            upside = ((dcf_fair_value - current_price) / current_price) * 100
            if upside > 20:
                dcf_adjustment = "DCF shows significant upside - consider buying"
            elif upside < -20:
                dcf_adjustment = "DCF shows overvaluation - wait for pullback"
            else:
                dcf_adjustment = "DCF shows fair valuation"

        return {
            'recommendation': recommendation,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'bullish_pct': bullish_pct,
            'dcf_adjustment': dcf_adjustment,
        }

    def run_full_analysis(self, dcf_fair_value: float = None) -> Dict:
        """Run complete technical analysis."""
        print(f"\n{'='*60}")
        print(f"TECHNICAL ANALYSIS: {self.ticker}")
        print(f"{'='*60}\n")

        # Fetch data
        self.fetch_data(period="1y")

        if self.df is None or len(self.df) == 0:
            print("Error: Could not fetch data")
            return {}

        current_price = self.df['Close'].iloc[-1]

        # Calculate all indicators
        rsi = self.calculate_rsi()
        macd, signal, macd_status = self.calculate_macd()
        mas = self.calculate_moving_averages()
        sr = self.find_support_resistance()
        trend = self.analyze_trend()
        volume = self.analyze_volume()
        signals = self.get_entry_exit_signals(dcf_fair_value)

        # Print results
        print(f"Current Price: ${current_price:.2f}")
        if dcf_fair_value:
            upside = ((dcf_fair_value - current_price) / current_price) * 100
            print(f"DCF Fair Value: ${dcf_fair_value:.2f} ({upside:+.1f}%)")

        print(f"\n{'='*60}")
        print("TREND ANALYSIS")
        print(f"{'='*60}")
        print(f"  Trend: {trend['trend']}")
        print(f"  20-Day ROC: {trend['roc_20d']:+.1f}%")

        print(f"\n{'='*60}")
        print("SUPPORT & RESISTANCE")
        print(f"{'='*60}")
        print(f"  Nearest Resistance: ${sr['nearest_resistance']:.2f} (+{sr['resistance_distance']:.1f}%)")
        print(f"  Current Price:      ${sr['current_price']:.2f}")
        print(f"  Nearest Support:    ${sr['nearest_support']:.2f} (-{sr['support_distance']:.1f}%)")

        if sr.get('all_resistance'):
            print(f"\n  Key Resistance Levels: ", end="")
            print(", ".join([f"${r:.2f}" for r in sr['all_resistance']]))
        if sr.get('all_support'):
            print(f"  Key Support Levels:    ", end="")
            print(", ".join([f"${s:.2f}" for s in sr['all_support']]))

        print(f"\n{'='*60}")
        print("MOMENTUM INDICATORS")
        print(f"{'='*60}")
        print(f"  RSI (14): {rsi:.1f}", end="")
        if rsi < 30:
            print(" [OVERSOLD - Buy signal]")
        elif rsi > 70:
            print(" [OVERBOUGHT - Caution]")
        elif rsi < 40:
            print(" [Getting oversold]")
        elif rsi > 60:
            print(" [Getting overbought]")
        else:
            print(" [Neutral]")

        print(f"\n  MACD: {macd:.2f}")
        print(f"  Signal: {signal:.2f}")
        print(f"  Status: {macd_status}")

        print(f"\n{'='*60}")
        print("MOVING AVERAGES")
        print(f"{'='*60}")
        for key, value in mas.items():
            if key.startswith('MA'):
                print(f"  {key}: ${value:.2f}", end="")
                pct_key = f"Price_vs_{key}"
                if pct_key in mas:
                    print(f" ({mas[pct_key]:+.1f}%)")

        print(f"\n{'='*60}")
        print("VOLUME ANALYSIS")
        print(f"{'='*60}")
        print(f"  Recent Volume: {volume['recent_volume']:,.0f}")
        print(f"  20-Day Avg:    {volume['avg_volume_20d']:,.0f}")
        print(f"  vs 20-Day Avg: {volume['volume_vs_20d']:+.1f}%")
        print(f"  Status: {volume['volume_status']}")

        print(f"\n{'='*60}")
        print("ENTRY/EXIT SIGNALS")
        print(f"{'='*60}")
        print(f"  Bullish Signals: {signals['bullish_signals']}")
        print(f"  Bearish Signals: {signals['bearish_signals']}")
        print(f"  Bullish %: {signals['bullish_pct']:.0f}%")
        print(f"\n  TECHNICAL RECOMMENDATION: {signals['recommendation']}")
        if signals['dcf_adjustment']:
            print(f"  DCF View: {signals['dcf_adjustment']}")

        # Final recommendation
        print(f"\n{'='*60}")
        print("FINAL ASSESSMENT")
        print(f"{'='*60}\n")

        # Combine technical + fundamental
        tech_rec = signals['recommendation']
        dcf_upside = ((dcf_fair_value - current_price) / current_price) * 100 if dcf_fair_value else 0

        if tech_rec in ["STRONG BUY", "BUY"] and dcf_upside > 15:
            final = "STRONG BUY - Technical and fundamental align"
        elif tech_rec in ["STRONG BUY", "BUY"]:
            final = "BUY on technicals, but check valuation"
        elif dcf_upside > 30:
            final = "BUY on fundamentals, wait for better technical entry"
        elif tech_rec in ["SELL", "REDUCE"] and dcf_upside < -15:
            final = "AVOID - Technical and fundamental both negative"
        elif tech_rec in ["SELL", "REDUCE"]:
            final = "CAUTION - Weak technicals"
        else:
            final = "HOLD - Wait for clearer signal"

        print(f"  {final}\n")

        # Compile results
        result = {
            'ticker': self.ticker,
            'current_price': current_price,
            'dcf_fair_value': dcf_fair_value,
            'trend': trend,
            'support_resistance': sr,
            'rsi': rsi,
            'macd': {'value': macd, 'signal': signal, 'status': macd_status},
            'moving_averages': mas,
            'volume': volume,
            'signals': signals,
            'final_recommendation': final,
        }

        return result


def analyze_top_picks():
    """Analyze all 4 top picks with technical analysis."""

    print("\n" + "="*60)
    print("PROTEUS TECHNICAL ANALYSIS")
    print("Top 4 Investment Recommendations")
    print("="*60)

    # DCF fair values from our previous analysis
    dcf_values = {
        'MU': 375.20,
        'APP': 326.18,
        'PLTR': 36.53,
        'ALAB': 59.06,
    }

    results = {}

    for ticker in ['MU', 'APP', 'PLTR', 'ALAB']:
        print(f"\n\n{'#'*60}")
        print(f"# {ticker}")
        print(f"{'#'*60}")

        analyzer = TechnicalAnalyzer(ticker)
        result = analyzer.run_full_analysis(dcf_fair_value=dcf_values.get(ticker))
        results[ticker] = result

    # Summary table
    print("\n\n" + "="*60)
    print("TECHNICAL SUMMARY - ALL 4 PICKS")
    print("="*60 + "\n")

    print(f"{'Ticker':<8} {'Price':<10} {'Trend':<18} {'RSI':<8} {'Tech Rec':<15} {'Final'}")
    print("-" * 90)

    for ticker in ['MU', 'APP', 'PLTR', 'ALAB']:
        if ticker in results and results[ticker]:
            r = results[ticker]
            price = r.get('current_price', 0)
            trend = r.get('trend', {}).get('trend', 'N/A')
            rsi = r.get('rsi', 0)
            tech_rec = r.get('signals', {}).get('recommendation', 'N/A')

            # Truncate trend for display
            if len(trend) > 16:
                trend = trend[:16]

            print(f"{ticker:<8} ${price:<9.2f} {trend:<18} {rsi:<7.1f} {tech_rec:<15} ", end="")

            # Final verdict icon
            final = r.get('final_recommendation', '')
            if 'STRONG BUY' in final:
                print("[***]")
            elif 'BUY' in final:
                print("[++]")
            elif 'HOLD' in final:
                print("[=]")
            elif 'AVOID' in final or 'CAUTION' in final:
                print("[-]")
            else:
                print("")

    print("\n" + "="*60)
    print("Legend:")
    print("  [***] = Strong Buy (technical + fundamental align)")
    print("  [++]  = Buy (good technical setup)")
    print("  [=]   = Hold (wait for clearer signal)")
    print("  [-]   = Caution/Avoid (weak technicals)")
    print("="*60 + "\n")

    return results


if __name__ == "__main__":
    analyze_top_picks()

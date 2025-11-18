"""
EXP-095: Multi-Factor Regime Detection

CONTEXT:
Current regime detection is crude (SPY 200-day SMA only).
Missing critical signals: VIX (fear), ADX (trend strength), volatility.

OBJECTIVE:
Enhance regime detection with multi-factor system for better trading timing.

CURRENT SYSTEM (Crude):
- Only SPY price vs 200-day SMA
- Simple BULL/BEAR/SIDEWAYS classification
- Misses market stress, volatility spikes, weak trends

MULTI-FACTOR SYSTEM (Enhanced):
1. VIX (Volatility Index) - Market fear gauge
   - VIX < 15: Low fear (safe to trade)
   - VIX 15-20: Moderate fear (normal)
   - VIX 20-30: Elevated fear (caution)
   - VIX > 30: Panic (high risk)

2. ADX (Trend Strength)
   - ADX > 25: Strong trend (good for mean reversion at extremes)
   - ADX 20-25: Moderate trend
   - ADX < 20: Weak/sideways (best for mean reversion)

3. ATR% (Volatility)
   - Low volatility: More predictable reversions
   - High volatility: Less predictable, higher risk

4. SPY vs 200-day SMA (Keep current method)
   - Price > SMA: Uptrend
   - Price < SMA: Downtrend

5. Momentum (Keep current method)
   - 60-day return for trend confirmation

ENHANCED REGIME CLASSIFICATION:
- BULL_STRONG: Price > 200 SMA + VIX < 20 + ADX > 25 + uptrend
- BULL_WEAK: Price > 200 SMA + VIX < 20 + ADX < 20
- SIDEWAYS_SAFE: VIX < 20 + weak trend + low volatility (BEST for mean reversion)
- SIDEWAYS_CHOPPY: VIX 20-25 + high volatility (trade with caution)
- FEAR: VIX > 25 (elevated fear - reduce trading)
- PANIC: VIX > 30 + negative momentum (AVOID mean reversion)
- BEAR: Price < 200 SMA + VIX > 30 + negative momentum (DISABLE trading)

EXPECTED IMPACT:
- Better timing: Avoid choppy volatility periods
- Fewer false signals: Multi-factor confirmation
- Risk reduction: Detect market stress earlier
- May reduce trade frequency but improve quality

HYPOTHESIS:
Multi-factor regime detection will improve risk-adjusted returns by:
1. Avoiding trades during high-stress periods (VIX > 30)
2. Trading more aggressively in safe periods (VIX < 20, low volatility)
3. Better identifying sideways markets (best for mean reversion)

SUCCESS CRITERIA:
- Sharpe ratio improvement: +10-20%
- Max drawdown reduction: -10-15%
- Win rate: Neutral to slight improvement
- Deploy if validated
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer


class MultiFactorRegimeDetector:
    """
    Enhanced regime detector with VIX, ADX, and volatility.
    """

    def __init__(
        self,
        vix_safe=15,
        vix_caution=20,
        vix_fear=25,
        vix_panic=30,
        adx_strong=25,
        adx_weak=20,
        atr_high=2.5,  # ATR % threshold for high volatility
    ):
        """
        Initialize multi-factor regime detector.

        Args:
            vix_safe: VIX threshold for safe trading
            vix_caution: VIX threshold for caution
            vix_fear: VIX threshold for elevated fear
            vix_panic: VIX threshold for panic
            adx_strong: ADX threshold for strong trend
            adx_weak: ADX threshold for weak trend
            atr_high: ATR% threshold for high volatility
        """
        self.vix_safe = vix_safe
        self.vix_caution = vix_caution
        self.vix_fear = vix_fear
        self.vix_panic = vix_panic
        self.adx_strong = adx_strong
        self.adx_weak = adx_weak
        self.atr_high = atr_high

    def detect_regime(self, spy_data: pd.DataFrame, vix_data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regime using multiple factors.

        Args:
            spy_data: SPY price data with Close column
            vix_data: VIX data with Close column

        Returns:
            DataFrame with regime classification
        """
        data = spy_data.copy()

        # Add VIX to SPY data (align by date)
        data['VIX'] = vix_data['Close']

        # Calculate technical indicators
        from ta.trend import ADXIndicator

        engineer = TechnicalFeatureEngineer()

        # SMA for trend
        data['sma_50'] = data['Close'].rolling(window=50).mean()
        data['sma_200'] = data['Close'].rolling(window=200).mean()
        data['price_vs_sma200'] = (data['Close'] / data['sma_200'] - 1) * 100

        # Momentum
        data['momentum_60d'] = data['Close'].pct_change(60) * 100

        # ADX for trend strength (calculate directly)
        adx_indicator = ADXIndicator(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            window=14,
            fillna=True
        )
        data['adx'] = adx_indicator.adx()

        # ATR for volatility
        data = engineer.add_volatility_indicators(data)
        # atr_pct already calculated by add_volatility_indicators()

        # Classify regime
        def classify_regime(row):
            """Multi-factor regime classification."""
            if pd.isna(row['VIX']) or pd.isna(row['sma_200']) or pd.isna(row['adx']):
                return 'UNKNOWN'

            vix = row['VIX']
            price_vs_sma = row['price_vs_sma200']
            momentum = row['momentum_60d']
            adx = row['adx']
            atr_pct = row['atr_pct']

            # PANIC: High VIX + downtrend + negative momentum
            if vix > self.vix_panic and price_vs_sma < -5 and momentum < -5:
                return 'PANIC'

            # BEAR: Price < 200 SMA + VIX > fear threshold
            if price_vs_sma < -5 and vix > self.vix_fear:
                return 'BEAR'

            # FEAR: High VIX (elevated fear regardless of trend)
            if vix > self.vix_fear:
                return 'FEAR'

            # BULL_STRONG: Price > 200 SMA + Low VIX + Strong trend
            if price_vs_sma > 0 and vix < self.vix_caution and adx > self.adx_strong:
                return 'BULL_STRONG'

            # BULL_WEAK: Price > 200 SMA + Low VIX + Weak trend
            if price_vs_sma > 0 and vix < self.vix_caution:
                return 'BULL_WEAK'

            # SIDEWAYS_CHOPPY: Moderate VIX + High volatility
            if vix >= self.vix_caution and atr_pct > self.atr_high:
                return 'SIDEWAYS_CHOPPY'

            # SIDEWAYS_SAFE: Low VIX + Weak trend + Low volatility
            # This is IDEAL for mean reversion
            if vix < self.vix_caution and adx < self.adx_weak and atr_pct < self.atr_high:
                return 'SIDEWAYS_SAFE'

            # Default: SIDEWAYS
            return 'SIDEWAYS'

        data['regime'] = data.apply(classify_regime, axis=1)

        return data

    def should_trade_mean_reversion(self, regime: str) -> tuple[bool, str]:
        """
        Determine if mean reversion trading should be enabled.

        Args:
            regime: Current regime classification

        Returns:
            Tuple of (should_trade, reason)
        """
        if regime == 'PANIC':
            return False, "PANIC: VIX > 30 + downtrend - AVOID all trading"

        elif regime == 'BEAR':
            return False, "BEAR: Downtrend + elevated fear - AVOID mean reversion"

        elif regime == 'FEAR':
            return False, "FEAR: VIX > 25 - Market stress, avoid trading"

        elif regime == 'SIDEWAYS_SAFE':
            return True, "SIDEWAYS_SAFE: IDEAL for mean reversion (low VIX, low volatility, weak trend)"

        elif regime == 'BULL_WEAK':
            return True, "BULL_WEAK: Safe for mean reversion (uptrend but weak momentum)"

        elif regime == 'BULL_STRONG':
            return True, "BULL_STRONG: Safe for mean reversion (strong uptrend, low VIX)"

        elif regime == 'SIDEWAYS_CHOPPY':
            return True, "SIDEWAYS_CHOPPY: Trade with caution (high volatility)"

        elif regime == 'SIDEWAYS':
            return True, "SIDEWAYS: Moderate conditions for mean reversion"

        else:
            return False, "UNKNOWN: Insufficient data"


def backtest_regime_detector(detector: MultiFactorRegimeDetector,
                             start_date: str, end_date: str) -> dict:
    """
    Backtest multi-factor regime detector.

    Args:
        detector: MultiFactorRegimeDetector instance
        start_date: Start date for backtest
        end_date: End date for backtest

    Returns:
        Backtest results
    """
    print("\nFetching SPY and VIX data...")
    fetcher = YahooFinanceFetcher()

    # Fetch SPY data
    spy_data = fetcher.fetch_stock_data('SPY', start_date=start_date, end_date=end_date)
    print(f"  SPY: {len(spy_data)} days")

    # Fetch VIX data
    vix_data = fetcher.fetch_stock_data('^VIX', start_date=start_date, end_date=end_date)
    print(f"  VIX: {len(vix_data)} days")

    # Align dates
    common_dates = spy_data.index.intersection(vix_data.index)
    spy_data = spy_data.loc[common_dates]
    vix_data = vix_data.loc[common_dates]

    print(f"  Aligned: {len(spy_data)} days")

    # Detect regime
    print("\nDetecting regimes...")
    regime_data = detector.detect_regime(spy_data, vix_data)

    # Analyze regime distribution
    print("\nRegime Distribution:")
    regime_counts = regime_data['regime'].value_counts()
    total_days = len(regime_data[regime_data['regime'] != 'UNKNOWN'])

    for regime, count in regime_counts.items():
        pct = (count / total_days) * 100
        print(f"  {regime:20s}: {count:4d} days ({pct:5.1f}%)")

    # Analyze returns by regime
    print("\nReturns by Regime (next-day returns):")
    regime_data['next_day_return'] = regime_data['Close'].pct_change().shift(-1) * 100

    regime_returns = {}
    for regime in regime_counts.index:
        if regime == 'UNKNOWN':
            continue

        regime_df = regime_data[regime_data['regime'] == regime]
        if len(regime_df) > 0:
            avg_return = regime_df['next_day_return'].mean()
            volatility = regime_df['next_day_return'].std()
            regime_returns[regime] = {
                'avg_return': avg_return,
                'volatility': volatility,
                'sharpe': (avg_return / volatility) if volatility > 0 else 0,
                'days': len(regime_df)
            }
            print(f"  {regime:20s}: {avg_return:+.3f}% avg, {volatility:.3f}% vol, Sharpe: {regime_returns[regime]['sharpe']:+.3f}")

    # Compare to crude detector
    print("\n" + "=" * 70)
    print("COMPARISON: Multi-Factor vs Crude (SPY 200 SMA only)")
    print("=" * 70)

    # Crude detector: Just price vs 200 SMA
    regime_data['crude_regime'] = 'SIDEWAYS'
    regime_data.loc[regime_data['price_vs_sma200'] > 0, 'crude_regime'] = 'BULL'
    regime_data.loc[regime_data['price_vs_sma200'] < -5, 'crude_regime'] = 'BEAR'

    # Trading rules
    # Multi-factor: Trade when should_trade_mean_reversion() returns True
    regime_data['multi_factor_trade'] = regime_data['regime'].apply(
        lambda r: detector.should_trade_mean_reversion(r)[0]
    )

    # Crude: Trade when not BEAR
    regime_data['crude_trade'] = regime_data['crude_regime'] != 'BEAR'

    # Calculate returns for each strategy
    multi_factor_returns = regime_data[regime_data['multi_factor_trade']]['next_day_return']
    crude_returns = regime_data[regime_data['crude_trade']]['next_day_return']

    print(f"\nMulti-Factor Strategy:")
    print(f"  Trading days: {len(multi_factor_returns)}/{total_days} ({len(multi_factor_returns)/total_days*100:.1f}%)")
    print(f"  Avg return: {multi_factor_returns.mean():+.3f}%")
    print(f"  Volatility: {multi_factor_returns.std():.3f}%")
    print(f"  Sharpe: {(multi_factor_returns.mean() / multi_factor_returns.std()):+.3f}")

    print(f"\nCrude Strategy (Current):")
    print(f"  Trading days: {len(crude_returns)}/{total_days} ({len(crude_returns)/total_days*100:.1f}%)")
    print(f"  Avg return: {crude_returns.mean():+.3f}%")
    print(f"  Volatility: {crude_returns.std():.3f}%")
    print(f"  Sharpe: {(crude_returns.mean() / crude_returns.std()):+.3f}")

    # Calculate improvement
    sharpe_improvement = (multi_factor_returns.mean() / multi_factor_returns.std()) - (crude_returns.mean() / crude_returns.std())
    sharpe_improvement_pct = (sharpe_improvement / (crude_returns.mean() / crude_returns.std())) * 100

    print(f"\nSharpe Improvement: {sharpe_improvement:+.3f} ({sharpe_improvement_pct:+.1f}%)")

    return {
        'regime_distribution': regime_counts.to_dict(),
        'regime_returns': regime_returns,
        'multi_factor': {
            'trading_days': len(multi_factor_returns),
            'avg_return': multi_factor_returns.mean(),
            'volatility': multi_factor_returns.std(),
            'sharpe': multi_factor_returns.mean() / multi_factor_returns.std()
        },
        'crude': {
            'trading_days': len(crude_returns),
            'avg_return': crude_returns.mean(),
            'volatility': crude_returns.std(),
            'sharpe': crude_returns.mean() / crude_returns.std()
        },
        'sharpe_improvement': sharpe_improvement,
        'sharpe_improvement_pct': sharpe_improvement_pct
    }


def main():
    print("=" * 70)
    print("EXP-095: MULTI-FACTOR REGIME DETECTION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("OBJECTIVE: Enhance regime detection with VIX, ADX, volatility")
    print("CURRENT: Crude (SPY 200-day SMA only)")
    print("ENHANCED: Multi-factor (VIX + ADX + ATR + SMA + Momentum)")
    print()

    # Initialize detector
    detector = MultiFactorRegimeDetector()

    # Test period: 2 years of data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    print(f"Test period: {start_date} to {end_date}")

    # Run backtest
    results = backtest_regime_detector(detector, start_date, end_date)

    # Recommendation
    print("\n" + "=" * 70)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 70)

    sharpe_improvement_pct = results['sharpe_improvement_pct']

    if sharpe_improvement_pct >= 10:
        recommendation = "DEPLOY"
        reason = f"Sharpe improvement {sharpe_improvement_pct:+.1f}% >= 10% target"
    elif sharpe_improvement_pct >= 5:
        recommendation = "CONSIDER"
        reason = f"Sharpe improvement {sharpe_improvement_pct:+.1f}% shows promise but below 10% target"
    else:
        recommendation = "REJECT"
        reason = f"Sharpe improvement {sharpe_improvement_pct:+.1f}% insufficient"

    print(f"\n[{recommendation}] {reason}")

    # Save results
    output = {
        'experiment': 'EXP-095',
        'timestamp': datetime.now().isoformat(),
        'objective': 'Multi-factor regime detection',
        'test_period': f"{start_date} to {end_date}",
        'regime_distribution': results['regime_distribution'],
        'regime_returns': results['regime_returns'],
        'multi_factor_strategy': results['multi_factor'],
        'crude_strategy': results['crude'],
        'improvement': {
            'sharpe_absolute': results['sharpe_improvement'],
            'sharpe_pct': sharpe_improvement_pct
        },
        'recommendation': recommendation,
        'reason': reason
    }

    output_path = Path('results/ml_experiments/exp095_multi_factor_regime.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70)
    print("EXP-095 COMPLETE")
    print("=" * 70)

    # Send email notification
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            notifier.send_experiment_report('EXP-095', output)
            print("\n[EMAIL] Experiment report sent via SendGrid")
    except Exception as e:
        print(f"\n[WARNING] Could not send email: {e}")


if __name__ == "__main__":
    main()

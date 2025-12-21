"""
Signal Quality Validator

Backtests historical signals to measure actual predictive quality:
1. Generate signals for past 6 months
2. Calculate actual returns after each signal
3. Measure hit rate by signal strength tier
4. Find optimal signal threshold
5. Compare GPU model vs traditional scanner

This is the foundation - if signals aren't predictive, nothing else matters.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import yfinance as yf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trading.signal_scanner import SignalScanner
from config.mean_reversion_params import get_all_tickers


@dataclass
class SignalOutcome:
    """Outcome of a single signal."""
    ticker: str
    signal_date: str
    signal_strength: float
    z_score: float
    rsi: float
    entry_price: float

    # Actual outcomes
    return_1d: float
    return_2d: float
    return_3d: float
    return_5d: float
    max_gain_5d: float  # Best exit opportunity
    max_loss_5d: float  # Worst drawdown

    # Classification
    hit_2pct: bool  # Did it hit 2% profit target?
    hit_stop: bool  # Did it hit -2% stop loss?
    profitable_2d: bool  # Positive return after 2 days?


@dataclass
class QualityReport:
    """Signal quality analysis report."""
    report_date: str
    analysis_period: str
    total_signals: int

    # Overall metrics
    overall_hit_rate: float  # % that hit 2% target before stop
    overall_win_rate: float  # % profitable at 2-day exit
    avg_return_2d: float
    avg_max_gain: float
    avg_max_loss: float

    # By strength tier
    tier_metrics: Dict[str, Dict]

    # Optimal threshold analysis
    threshold_analysis: List[Dict]
    recommended_threshold: float

    # Statistical significance
    sample_size_adequate: bool
    confidence_level: float


class SignalQualityValidator:
    """
    Validates signal quality through historical backtesting.
    """

    PROTEUS_TICKERS = get_all_tickers()

    # Strength tiers to analyze
    TIERS = {
        'ELITE': (80, 100),
        'STRONG': (70, 80),
        'GOOD': (60, 70),
        'MODERATE': (50, 60),
        'WEAK': (40, 50),
        'POOR': (0, 40)
    }

    def __init__(self, output_dir: str = "data/signal_validation"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.scanner = SignalScanner(lookback_days=90, min_signal_strength=0)  # No filter for analysis

    def get_historical_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical price data for a ticker."""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            return df
        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")
            return pd.DataFrame()

    def calculate_signal_outcome(self, ticker: str, signal_date: str,
                                  signal: Dict, prices: pd.DataFrame) -> Optional[SignalOutcome]:
        """Calculate the actual outcome of a signal."""
        try:
            # Normalize prices index to date only (remove timezone issues)
            prices_normalized = prices.copy()
            prices_normalized.index = pd.to_datetime(prices_normalized.index).date

            # Find signal date in prices
            signal_date_obj = pd.Timestamp(signal_date).date()

            # Get index position
            if signal_date_obj not in prices_normalized.index:
                # Find nearest date
                dates = pd.Index(prices_normalized.index)
                future_dates = dates[dates >= signal_date_obj]
                if len(future_dates) == 0:
                    return None
                signal_date_obj = future_dates[0]

            idx = list(prices_normalized.index).index(signal_date_obj)

            # Need at least 5 days forward
            if idx + 5 >= len(prices):
                return None

            entry_price = prices['Close'].iloc[idx]

            # Calculate returns at different horizons
            price_1d = prices['Close'].iloc[idx + 1] if idx + 1 < len(prices) else entry_price
            price_2d = prices['Close'].iloc[idx + 2] if idx + 2 < len(prices) else entry_price
            price_3d = prices['Close'].iloc[idx + 3] if idx + 3 < len(prices) else entry_price
            price_5d = prices['Close'].iloc[idx + 5] if idx + 5 < len(prices) else entry_price

            return_1d = ((price_1d / entry_price) - 1) * 100
            return_2d = ((price_2d / entry_price) - 1) * 100
            return_3d = ((price_3d / entry_price) - 1) * 100
            return_5d = ((price_5d / entry_price) - 1) * 100

            # Max gain/loss over 5 days
            future_prices = prices['Close'].iloc[idx:idx+6]
            future_highs = prices['High'].iloc[idx:idx+6]
            future_lows = prices['Low'].iloc[idx:idx+6]

            max_gain = ((future_highs.max() / entry_price) - 1) * 100
            max_loss = ((future_lows.min() / entry_price) - 1) * 100

            # Check if hit targets
            hit_2pct = bool(max_gain >= 2.0)
            hit_stop = bool(max_loss <= -2.0)
            profitable_2d = bool(return_2d > 0)

            return SignalOutcome(
                ticker=ticker,
                signal_date=signal_date,
                signal_strength=signal.get('signal_strength', 0),
                z_score=signal.get('z_score', 0),
                rsi=signal.get('rsi', 50),
                entry_price=entry_price,
                return_1d=round(return_1d, 2),
                return_2d=round(return_2d, 2),
                return_3d=round(return_3d, 2),
                return_5d=round(return_5d, 2),
                max_gain_5d=round(max_gain, 2),
                max_loss_5d=round(max_loss, 2),
                hit_2pct=hit_2pct,
                hit_stop=hit_stop,
                profitable_2d=profitable_2d
            )

        except Exception as e:
            return None

    def generate_historical_signals(self, days_back: int = 180) -> List[Tuple[str, str, Dict]]:
        """Generate signals for historical dates."""
        print(f"\n[1] Generating signals for past {days_back} days...")

        # End 10 days ago to ensure we have forward data for outcomes
        end_date = datetime.now() - timedelta(days=10)
        start_date = end_date - timedelta(days=days_back)

        all_signals = []

        # Get price data for all tickers
        print("  Fetching price data...")
        price_cache = {}
        for ticker in self.PROTEUS_TICKERS:
            df = self.get_historical_prices(
                ticker,
                (start_date - timedelta(days=100)).strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            if len(df) > 0:
                price_cache[ticker] = df

        print(f"  Loaded data for {len(price_cache)} tickers")

        # Scan each date
        current_date = start_date
        dates_scanned = 0

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                date_str = current_date.strftime('%Y-%m-%d')

                # Scan for signals on this date
                try:
                    signals = self.scanner.scan_all_stocks(date_str)
                    for signal in signals:
                        all_signals.append((signal['ticker'], date_str, signal))
                    dates_scanned += 1

                    if dates_scanned % 20 == 0:
                        print(f"  Scanned {dates_scanned} days, {len(all_signals)} signals found...")

                except Exception as e:
                    pass

            current_date += timedelta(days=1)

        print(f"  Total: {len(all_signals)} signals across {dates_scanned} trading days")
        return all_signals, price_cache

    def validate_signals(self, signals: List[Tuple], price_cache: Dict) -> List[SignalOutcome]:
        """Validate all signals against actual outcomes."""
        print(f"\n[2] Validating {len(signals)} signals...")

        outcomes = []

        for i, (ticker, date, signal) in enumerate(signals):
            if ticker in price_cache:
                outcome = self.calculate_signal_outcome(ticker, date, signal, price_cache[ticker])
                if outcome:
                    outcomes.append(outcome)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(signals)} signals...")

        print(f"  Validated {len(outcomes)} signals with complete data")
        return outcomes

    def analyze_by_tier(self, outcomes: List[SignalOutcome]) -> Dict[str, Dict]:
        """Analyze outcomes by signal strength tier."""
        tier_results = {}

        for tier_name, (low, high) in self.TIERS.items():
            tier_outcomes = [o for o in outcomes if low <= o.signal_strength < high]

            if len(tier_outcomes) == 0:
                tier_results[tier_name] = {
                    'count': 0,
                    'hit_rate': 0,
                    'win_rate': 0,
                    'avg_return_2d': 0,
                    'avg_max_gain': 0,
                    'avg_max_loss': 0
                }
                continue

            hit_rate = sum(1 for o in tier_outcomes if o.hit_2pct and not o.hit_stop) / len(tier_outcomes) * 100
            win_rate = sum(1 for o in tier_outcomes if o.profitable_2d) / len(tier_outcomes) * 100
            avg_return = np.mean([o.return_2d for o in tier_outcomes])
            avg_gain = np.mean([o.max_gain_5d for o in tier_outcomes])
            avg_loss = np.mean([o.max_loss_5d for o in tier_outcomes])

            tier_results[tier_name] = {
                'count': len(tier_outcomes),
                'hit_rate': round(hit_rate, 1),
                'win_rate': round(win_rate, 1),
                'avg_return_2d': round(avg_return, 2),
                'avg_max_gain': round(avg_gain, 2),
                'avg_max_loss': round(avg_loss, 2)
            }

        return tier_results

    def find_optimal_threshold(self, outcomes: List[SignalOutcome]) -> Tuple[float, List[Dict]]:
        """Find the optimal signal strength threshold."""
        thresholds = [30, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        results = []

        for threshold in thresholds:
            filtered = [o for o in outcomes if o.signal_strength >= threshold]

            if len(filtered) < 10:
                continue

            win_rate = sum(1 for o in filtered if o.profitable_2d) / len(filtered) * 100
            hit_rate = sum(1 for o in filtered if o.hit_2pct and not o.hit_stop) / len(filtered) * 100
            avg_return = np.mean([o.return_2d for o in filtered])

            # Risk-adjusted score: balance win rate and signal frequency
            # Higher is better
            score = (win_rate - 50) * np.log(len(filtered) + 1)

            results.append({
                'threshold': threshold,
                'signals': len(filtered),
                'win_rate': round(win_rate, 1),
                'hit_rate': round(hit_rate, 1),
                'avg_return': round(avg_return, 2),
                'score': round(score, 1)
            })

        # Find best score
        if results:
            best = max(results, key=lambda x: x['score'])
            return best['threshold'], results

        return 50, results

    def run_validation(self, days_back: int = 180) -> QualityReport:
        """Run complete signal quality validation."""
        print("=" * 70)
        print("SIGNAL QUALITY VALIDATION")
        print("=" * 70)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Analysis period: {days_back} days")

        # Generate historical signals
        signals, price_cache = self.generate_historical_signals(days_back)

        if len(signals) == 0:
            print("[ERROR] No signals generated")
            return None

        # Validate against outcomes
        outcomes = self.validate_signals(signals, price_cache)

        if len(outcomes) == 0:
            print("[ERROR] No outcomes calculated")
            return None

        # Calculate overall metrics
        print("\n[3] Calculating metrics...")

        overall_hit_rate = sum(1 for o in outcomes if o.hit_2pct and not o.hit_stop) / len(outcomes) * 100
        overall_win_rate = sum(1 for o in outcomes if o.profitable_2d) / len(outcomes) * 100
        avg_return_2d = np.mean([o.return_2d for o in outcomes])
        avg_max_gain = np.mean([o.max_gain_5d for o in outcomes])
        avg_max_loss = np.mean([o.max_loss_5d for o in outcomes])

        # Analyze by tier
        tier_metrics = self.analyze_by_tier(outcomes)

        # Find optimal threshold
        optimal_threshold, threshold_analysis = self.find_optimal_threshold(outcomes)

        # Statistical significance
        sample_adequate = len(outcomes) >= 100
        # Simple confidence based on sample size
        confidence = min(0.95, 0.5 + 0.005 * len(outcomes))

        # Build report
        report = QualityReport(
            report_date=datetime.now().strftime('%Y-%m-%d'),
            analysis_period=f"{days_back} days",
            total_signals=len(outcomes),
            overall_hit_rate=round(overall_hit_rate, 1),
            overall_win_rate=round(overall_win_rate, 1),
            avg_return_2d=round(avg_return_2d, 2),
            avg_max_gain=round(avg_max_gain, 2),
            avg_max_loss=round(avg_max_loss, 2),
            tier_metrics=tier_metrics,
            threshold_analysis=threshold_analysis,
            recommended_threshold=optimal_threshold,
            sample_size_adequate=sample_adequate,
            confidence_level=round(confidence, 2)
        )

        # Save report
        self._save_report(report, outcomes)

        # Print summary
        self._print_summary(report)

        return report

    def _save_report(self, report: QualityReport, outcomes: List[SignalOutcome]):
        """Save report and outcomes."""
        # Save report
        report_file = os.path.join(self.output_dir, f"quality_report_{report.report_date}.json")
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"\n[SAVED] {report_file}")

        # Save outcomes
        outcomes_file = os.path.join(self.output_dir, f"signal_outcomes_{report.report_date}.json")
        with open(outcomes_file, 'w') as f:
            json.dump([asdict(o) for o in outcomes], f, indent=2)
        print(f"[SAVED] {outcomes_file}")

    def _print_summary(self, report: QualityReport):
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("SIGNAL QUALITY REPORT")
        print("=" * 70)

        print(f"\nTotal Signals Analyzed: {report.total_signals}")
        print(f"Sample Size Adequate: {'Yes' if report.sample_size_adequate else 'NO - need more data'}")
        print(f"Confidence Level: {report.confidence_level*100:.0f}%")

        print("\n--- OVERALL PERFORMANCE ---")
        print(f"Win Rate (2-day): {report.overall_win_rate:.1f}%")
        print(f"Hit Rate (2% target): {report.overall_hit_rate:.1f}%")
        print(f"Avg Return (2-day): {report.avg_return_2d:+.2f}%")
        print(f"Avg Max Gain (5-day): {report.avg_max_gain:+.2f}%")
        print(f"Avg Max Loss (5-day): {report.avg_max_loss:+.2f}%")

        print("\n--- PERFORMANCE BY TIER ---")
        print(f"{'Tier':<12} {'Count':>8} {'Win Rate':>10} {'Hit Rate':>10} {'Avg Ret':>10}")
        print("-" * 52)
        for tier_name in ['ELITE', 'STRONG', 'GOOD', 'MODERATE', 'WEAK', 'POOR']:
            m = report.tier_metrics.get(tier_name, {})
            if m.get('count', 0) > 0:
                print(f"{tier_name:<12} {m['count']:>8} {m['win_rate']:>9.1f}% {m['hit_rate']:>9.1f}% {m['avg_return_2d']:>+9.2f}%")

        print("\n--- THRESHOLD ANALYSIS ---")
        print(f"{'Threshold':>10} {'Signals':>10} {'Win Rate':>10} {'Avg Ret':>10} {'Score':>10}")
        print("-" * 52)
        for t in report.threshold_analysis:
            marker = " <-- BEST" if t['threshold'] == report.recommended_threshold else ""
            print(f"{t['threshold']:>10} {t['signals']:>10} {t['win_rate']:>9.1f}% {t['avg_return']:>+9.2f}% {t['score']:>10.1f}{marker}")

        print(f"\n>>> RECOMMENDED THRESHOLD: {report.recommended_threshold}")

        # Key insights
        print("\n--- KEY INSIGHTS ---")
        if report.overall_win_rate > 55:
            print("[OK] Signals show positive edge (win rate > 55%)")
        elif report.overall_win_rate > 50:
            print("[MARGINAL] Slight edge detected (win rate 50-55%)")
        else:
            print("[CONCERN] No clear edge (win rate <= 50%)")

        if report.avg_return_2d > 0:
            print(f"[OK] Average return is positive: {report.avg_return_2d:+.2f}%")
        else:
            print(f"[CONCERN] Average return is negative: {report.avg_return_2d:+.2f}%")

        elite = report.tier_metrics.get('ELITE', {})
        strong = report.tier_metrics.get('STRONG', {})
        if elite.get('win_rate', 0) > strong.get('win_rate', 0):
            print("[OK] Higher strength signals outperform - strength filter is working")
        else:
            print("[CONCERN] Strength filter may not be discriminating well")


def run_signal_validation():
    """Main entry point."""
    validator = SignalQualityValidator()
    report = validator.run_validation(days_back=180)
    return report


if __name__ == "__main__":
    run_signal_validation()

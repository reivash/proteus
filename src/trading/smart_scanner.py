"""
Smart Scanner - Unified Trading Signal System

Combines:
1. GPU Model signals (retrained with multi-task learning)
2. Market regime detection (adjusts thresholds/sizing)
3. Trailing stop recommendations

This is the main entry point for daily scanning.
"""

import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.gpu_signal_model import GPUSignalModel, GPUSignal
from analysis.market_regime import MarketRegimeDetector, MarketRegime, RegimeAnalysis
from config.stock_config_loader import get_loader
from trading.sector_correlation import filter_correlated_signals, get_sector, print_sector_analysis
from trading.volatility_sizing import VolatilitySizer
from data.fetchers.earnings_calendar import EarningsCalendarFetcher


@dataclass
class SmartSignal:
    """Enhanced signal with regime adjustments."""
    ticker: str
    timestamp: str

    # Core signal data
    raw_strength: float
    adjusted_strength: float  # After regime adjustment
    probability: float
    expected_return: float
    confidence: float

    # Position sizing
    position_multiplier: float
    suggested_shares: int  # Based on $10k position
    current_price: float  # Real-time price from market data

    # Exit strategy
    profit_target: float
    stop_loss: float
    trailing_stop_trigger: float  # Start trailing after this gain
    trailing_stop_distance: float  # Trail by this %
    max_hold_days: int

    # Regime context
    regime: str
    regime_confidence: float

    # Tier classification
    tier: str

    # Earnings proximity
    near_earnings: bool = False
    earnings_warning: str = ""


@dataclass
class ScanResult:
    """Complete scan output."""
    timestamp: str
    regime: RegimeAnalysis
    signals: List[SmartSignal]
    filtered_count: int
    total_scanned: int


class SmartScanner:
    """
    Production-ready scanner combining all improvements.
    """

    TIERS = {
        'ELITE': (80, 100),
        'STRONG': (70, 80),
        'GOOD': (60, 70),
        'MODERATE': (50, 60),
        'WEAK': (40, 50),
        'POOR': (0, 40)
    }

    def __init__(self, portfolio_value: float = 100000):
        self.portfolio_value = portfolio_value
        self.gpu_model = GPUSignalModel()
        self.regime_detector = MarketRegimeDetector()
        self.config_loader = get_loader()
        self.volatility_sizer = VolatilitySizer(
            portfolio_value=portfolio_value,
            risk_per_trade=0.02,  # 2% risk per trade
            max_position_pct=0.15  # 15% max position
        )
        self.earnings_calendar = EarningsCalendarFetcher(
            exclusion_days_before=3,
            exclusion_days_after=1  # Less strict on after
        )

    def get_tier(self, strength: float) -> str:
        """Get tier name for signal strength."""
        for tier_name, (low, high) in self.TIERS.items():
            if low <= strength < high:
                return tier_name
        return 'POOR' if strength < 40 else 'ELITE'

    def calculate_position_size(self, signal_strength: float,
                                 position_multiplier: float,
                                 current_price: float) -> int:
        """
        Calculate suggested share count.
        Base: 10% of portfolio per position, adjusted by strength and regime.
        """
        base_position = self.portfolio_value * 0.10  # 10% per position

        # Strength adjustment (0.5x to 1.5x based on strength)
        strength_mult = 0.5 + (signal_strength / 100)

        # Apply regime multiplier
        adjusted_position = base_position * strength_mult * position_multiplier

        # Convert to shares
        if current_price > 0:
            shares = int(adjusted_position / current_price)
            return max(1, shares)
        return 0

    def get_exit_params(self, ticker: str, regime: MarketRegime) -> Dict:
        """
        Get exit parameters adjusted for regime and current volatility.
        Uses config-based regime multipliers and VIX-based adjustments.
        """
        # Get base params from config
        base_params = self.config_loader.get_default_exit_params()

        profit_target = base_params.get('profit_target', 2.0)
        stop_loss = base_params.get('stop_loss', -2.5)
        max_hold = base_params.get('max_hold_days', 3)

        # Get regime-specific adjustments from config
        regime_params = self.config_loader.get_regime_params(regime.value)
        profit_mult = regime_params.get('profit_target_multiplier', 1.0)
        stop_mult = regime_params.get('stop_loss_multiplier', 1.0)
        regime_max_hold = regime_params.get('max_hold_days', max_hold)

        # Apply regime multipliers to base values
        regime_profit = profit_target * profit_mult
        regime_stop = stop_loss * stop_mult  # stop is negative, so multiplying widens it

        # Get volatility-adjusted exits (additional layer based on current VIX)
        vol_exits = self.regime_detector.get_volatility_adjusted_exits(
            base_profit=regime_profit,
            base_stop=regime_stop
        )

        return {
            'profit_target': round(vol_exits['profit_target'], 2),
            'stop_loss': round(vol_exits['stop_loss'], 2),
            'max_hold_days': regime_max_hold,
            'trailing_trigger': round(vol_exits['trailing_trigger'], 2),
            'trailing_distance': round(vol_exits['trailing_distance'], 2),
            'volatility_mult': vol_exits['volatility_multiplier'],
            'regime_mult': profit_mult  # For debugging
        }

    def check_earnings_proximity(self, ticker: str) -> tuple:
        """
        Check if a stock is near an earnings announcement.

        Returns:
            Tuple of (near_earnings: bool, warning_message: str)
        """
        import pandas as pd
        from datetime import timedelta

        today = pd.Timestamp.now().normalize()

        # Check if today is in the exclusion window
        if not self.earnings_calendar.should_trade_on_date(ticker, today):
            # Find the nearest earnings date
            earnings_df = self.earnings_calendar.fetch_earnings_dates(ticker)
            if len(earnings_df) > 0:
                # Find closest earnings date
                future_earnings = earnings_df[earnings_df['earnings_date'] >= today - timedelta(days=5)]
                if len(future_earnings) > 0:
                    next_earnings = future_earnings['earnings_date'].min()
                    days_until = (next_earnings - today).days
                    if days_until <= 0:
                        return True, f"Earnings on {next_earnings.strftime('%Y-%m-%d')} (today/recent)"
                    else:
                        return True, f"Earnings in {days_until} days ({next_earnings.strftime('%Y-%m-%d')})"

            return True, "Near earnings announcement"

        return False, ""

    def scan(self) -> ScanResult:
        """
        Run complete smart scan.
        """
        print("=" * 70)
        print("SMART SCANNER")
        print("=" * 70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # 1. Get market regime
        print("[1] Analyzing market regime...")
        regime_analysis = self.regime_detector.detect_regime()
        print(f"    Regime: {regime_analysis.regime.value.upper()} "
              f"(confidence: {regime_analysis.confidence*100:.0f}%)")
        print(f"    VIX: {regime_analysis.vix_level:.1f} "
              f"({regime_analysis.vix_percentile:.0f}th percentile)")

        # Get regime adjustments from config (based on 1-year regime analysis)
        # VOLATILE: 62% win, +1.94% avg - BEST for mean reversion
        # BEAR: 69% win, +0.89% avg - Strong for mean reversion
        # CHOPPY: 56% win, +0.17% avg - Baseline
        # BULL: 51% win, +0.15% avg - WORST for mean reversion
        regime_params = self.config_loader.get_regime_params(regime_analysis.regime.value)

        # Use config-based adjustments (inverted from old system!)
        # In volatile/bear regimes, we LOWER threshold and INCREASE position
        base_threshold = 50.0
        regime_threshold_adj = regime_params.get('threshold_adjustment', 0)
        min_threshold = base_threshold + regime_threshold_adj  # e.g., 50 + (-15) = 35 in volatile

        position_mult = regime_params.get('position_multiplier', 1.0)

        # Get dynamic threshold adjustments based on VIX/trends (additional layer)
        dynamic_adj = self.regime_detector.get_dynamic_threshold(base_threshold=min_threshold)
        dynamic_threshold = dynamic_adj['threshold']

        # Log regime-specific elite stocks
        elite_stocks = self.config_loader.get_regime_elite_stocks(regime_analysis.regime.value)

        print(f"    Regime adjustment: {regime_threshold_adj:+.0f}")
        print(f"    Position multiplier: {position_mult:.1f}x (from config)")
        print(f"    Base threshold: {min_threshold:.0f}")
        print(f"    Dynamic threshold: {dynamic_threshold:.1f}")
        print(f"    Elite stocks for this regime: {', '.join(elite_stocks[:5])}")

        # 2. Run GPU scan
        print()
        print("[2] Running GPU model scan...")
        gpu_signals = self.gpu_model.scan_all()
        print(f"    Found {len(gpu_signals)} raw signals")

        # Get real prices from GPU model (already fetched during scan)
        stock_prices = self.gpu_model.get_last_prices()
        print(f"    Loaded {len(stock_prices)} real prices")

        # 3. Filter and enhance signals
        print()
        print("[3] Filtering and enhancing signals...")

        smart_signals = []
        filtered_count = 0

        for sig in gpu_signals:
            # Get per-stock threshold (may override regime threshold)
            stock_threshold = self.config_loader.get_signal_threshold(
                sig.ticker, base_threshold=dynamic_threshold
            )

            # Apply the higher of dynamic or per-stock threshold
            effective_threshold = max(dynamic_threshold, stock_threshold)

            if sig.signal_strength < effective_threshold:
                filtered_count += 1
                continue

            # Get exit params for this stock + regime
            exit_params = self.get_exit_params(sig.ticker, regime_analysis.regime)

            # Calculate adjusted strength based on regime analysis findings
            # VOLATILE/BEAR are BETTER for mean reversion (counterintuitive but data-driven)
            adjusted_strength = sig.signal_strength
            if regime_analysis.regime == MarketRegime.VOLATILE:
                adjusted_strength *= 1.15  # 15% boost - best regime (62% win, +1.94% avg)
            elif regime_analysis.regime == MarketRegime.BEAR:
                adjusted_strength *= 1.10  # 10% boost - strong regime (69% win, +0.89% avg)
            elif regime_analysis.regime == MarketRegime.CHOPPY:
                adjusted_strength *= 1.0   # Baseline (56% win, +0.17% avg)
            elif regime_analysis.regime == MarketRegime.BULL:
                adjusted_strength *= 0.85  # 15% penalty - worst regime (51% win, +0.15% avg)

            # Extra boost for regime-elite stocks
            if sig.ticker in elite_stocks:
                adjusted_strength *= 1.05  # Additional 5% for elite stocks in this regime

            # Boost strong performers, penalize weak ones (based on backtest)
            if self.config_loader.is_strong_performer(sig.ticker):
                adjusted_strength *= 1.05  # 5% boost for historically strong stocks
            elif self.config_loader.is_high_risk_stock(sig.ticker):
                adjusted_strength *= 0.90  # 10% penalty for historically weak stocks

            adjusted_strength = min(100, adjusted_strength)

            # Calculate position size using volatility-based sizing
            current_price = stock_prices.get(sig.ticker, 100)

            # Get volatility-based sizing
            vol_metrics = self.volatility_sizer.calculate_position_size(
                sig.ticker, current_price, stop_loss_pct=abs(exit_params['stop_loss'])
            )

            # Use volatility-adjusted shares, scaled by position multiplier
            shares = int(vol_metrics.suggested_shares * position_mult)
            shares = max(1, shares)

            smart_signal = SmartSignal(
                ticker=sig.ticker,
                timestamp=sig.timestamp,
                raw_strength=sig.signal_strength,
                adjusted_strength=round(adjusted_strength, 1),
                probability=sig.mean_reversion_prob,
                expected_return=sig.expected_return,
                confidence=sig.confidence,
                position_multiplier=position_mult,
                suggested_shares=shares,
                current_price=round(current_price, 2),
                profit_target=exit_params['profit_target'],
                stop_loss=exit_params['stop_loss'],
                trailing_stop_trigger=exit_params['trailing_trigger'],
                trailing_stop_distance=exit_params['trailing_distance'],
                max_hold_days=exit_params['max_hold_days'],
                regime=regime_analysis.regime.value,
                regime_confidence=regime_analysis.confidence,
                tier=self.get_tier(adjusted_strength)
            )
            smart_signals.append(smart_signal)

        # Sort by adjusted strength
        smart_signals.sort(key=lambda x: x.adjusted_strength, reverse=True)

        print(f"    Passed threshold filter: {len(smart_signals)}")
        print(f"    Filtered out: {filtered_count}")

        # 4. Apply sector correlation filter
        print()
        print("[4] Applying sector correlation filter...")

        # Convert SmartSignals to dicts for the filter
        signal_dicts = [asdict(s) for s in smart_signals]

        correlation_result = filter_correlated_signals(
            signal_dicts,
            max_per_sector=2,
            max_correlated_total=3,
            strength_key='adjusted_strength'
        )

        # Convert back to SmartSignals
        sector_filtered_signals = []
        passed_tickers = {s['ticker'] for s in correlation_result.passed_signals}
        for sig in smart_signals:
            if sig.ticker in passed_tickers:
                sector_filtered_signals.append(sig)

        sector_filtered_count = len(smart_signals) - len(sector_filtered_signals)

        print(f"    Passed sector filter: {len(sector_filtered_signals)}")
        print(f"    Sector-filtered: {sector_filtered_count}")

        if correlation_result.warnings:
            print("    Sector warnings:")
            for w in correlation_result.warnings[:5]:  # Show first 5
                print(f"      - {w}")

        # Use sector-filtered signals
        final_signals = sector_filtered_signals
        total_filtered = filtered_count + sector_filtered_count

        # 5. Apply earnings filter (mark but don't remove)
        print()
        print("[5] Checking earnings proximity...")
        earnings_warnings = []
        earnings_excluded = []
        for sig in final_signals:
            near_earnings, warning = self.check_earnings_proximity(sig.ticker)
            if near_earnings:
                sig.near_earnings = True
                sig.earnings_warning = warning
                earnings_warnings.append(f"{sig.ticker}: {warning}")
                earnings_excluded.append(sig.ticker)

        # Remove signals near earnings from final list
        final_signals = [s for s in final_signals if not s.near_earnings]
        earnings_filtered_count = len(earnings_excluded)
        total_filtered += earnings_filtered_count

        print(f"    Passed earnings filter: {len(final_signals)}")
        print(f"    Near earnings (excluded): {earnings_filtered_count}")
        if earnings_warnings:
            for w in earnings_warnings[:5]:
                print(f"      - {w}")

        # 6. Create result
        result = ScanResult(
            timestamp=datetime.now().isoformat(),
            regime=regime_analysis,
            signals=final_signals,
            filtered_count=total_filtered,
            total_scanned=len(gpu_signals)
        )

        # 7. Print summary
        self._print_summary(result)

        # 8. Save results
        self._save_results(result)

        return result

    def _print_summary(self, result: ScanResult):
        """Print scan summary."""
        print()
        print("=" * 70)
        print("SCAN RESULTS")
        print("=" * 70)

        print(f"\nMarket Regime: {result.regime.regime.value.upper()}")
        print(f"Recommendation: {result.regime.recommendation}")

        print(f"\nSignals: {len(result.signals)} passed / "
              f"{result.filtered_count} filtered / "
              f"{result.total_scanned} scanned")

        if result.signals:
            print("\n--- TOP SIGNALS ---")
            print(f"{'Ticker':<8} {'Price':>10} {'Shares':>7} {'Tier':<10} {'Str':>6} "
                  f"{'E[R]':>7} {'Sector':<12}")
            print("-" * 80)

            for sig in result.signals[:10]:
                sector = get_sector(sig.ticker)
                print(f"{sig.ticker:<8} ${sig.current_price:>9.2f} {sig.suggested_shares:>7} "
                      f"{sig.tier:<10} {sig.adjusted_strength:>6.1f} "
                      f"{sig.expected_return:>+6.1f}% {sector:<12}")

        print()

    def _save_results(self, result: ScanResult):
        """Save scan results to file."""
        output_dir = "data/smart_scans"
        os.makedirs(output_dir, exist_ok=True)

        # Convert to serializable format
        output = {
            'timestamp': result.timestamp,
            'regime': {
                'type': result.regime.regime.value,
                'confidence': result.regime.confidence,
                'vix': result.regime.vix_level,
                'spy_trend_20d': result.regime.spy_trend_20d,
                'recommendation': result.regime.recommendation
            },
            'summary': {
                'total_scanned': result.total_scanned,
                'passed_filter': len(result.signals),
                'filtered_out': result.filtered_count
            },
            'signals': [asdict(s) for s in result.signals[:20]]  # Top 20
        }

        # Save latest
        latest_file = os.path.join(output_dir, "latest_scan.json")
        with open(latest_file, 'w') as f:
            json.dump(output, f, indent=2)

        # Save dated
        dated_file = os.path.join(
            output_dir,
            f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(dated_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"[SAVED] {latest_file}")


def run_smart_scan():
    """Main entry point."""
    scanner = SmartScanner()
    return scanner.scan()


if __name__ == "__main__":
    run_smart_scan()

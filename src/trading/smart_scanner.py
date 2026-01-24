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
from models.hybrid_signal_model import HybridSignalModel, HybridSignal
from analysis.market_regime import (
    MarketRegimeDetector, MarketRegime, RegimeAnalysis,
    RegimeAwareFilter, TradingDecision
)
from config.stock_config_loader import get_loader
from trading.sector_correlation import filter_correlated_signals, get_sector, print_sector_analysis
from trading.volatility_sizing import VolatilitySizer
from trading.sector_momentum import get_sector_momentum_calculator, SectorMomentum
from trading.enhanced_signal_calculator import get_enhanced_calculator, SignalAdjustments
from data.fetchers.earnings_calendar import EarningsCalendarFetcher
from trading.exit_optimizer import ExitOptimizer
from trading.position_sizer import PositionSizer
from analysis.fast_bear_detector import FastBearDetector


# Bear market exit adjustments (tighter stops when elevated)
BEAR_EXIT_ADJUSTMENTS = {
    'NORMAL': {'stop_mult': 1.0, 'profit_mult': 1.0, 'hold_mult': 1.0},
    'WATCH': {'stop_mult': 0.85, 'profit_mult': 0.9, 'hold_mult': 0.85},  # Tighter stops
    'WARNING': {'stop_mult': 0.70, 'profit_mult': 0.75, 'hold_mult': 0.70},  # Much tighter
    'CRITICAL': {'stop_mult': 0.50, 'profit_mult': 0.60, 'hold_mult': 0.50},  # Aggressive protection
}


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

    # Sector momentum (from sector_momentum.py)
    sector_etf: str = ""
    sector_name: str = ""
    sector_momentum_5d: float = 0.0
    sector_momentum_category: str = "neutral"
    sector_momentum_boost: float = 1.0

    # Enhanced signal adjustments (from enhanced_signal_calculator.py)
    day_of_week: str = ""
    day_of_week_multiplier: float = 1.0
    consecutive_down_days: int = 0
    consecutive_down_multiplier: float = 1.0
    volume_ratio: float = 1.0
    volume_multiplier: float = 1.0
    stock_tier: str = "average"
    stock_tier_multiplier: float = 1.0

    # Research-based position sizing (from position_sizer.py)
    position_size_pct: float = 0.0  # % of portfolio
    position_dollar_size: float = 0.0  # Dollar amount
    position_risk_dollars: float = 0.0  # Dollar risk
    signal_quality: str = "moderate"  # poor/weak/moderate/strong/very_strong
    skip_trade: bool = False  # Whether to skip this trade

    # Ensemble info (3-model voting)
    ensemble_votes: int = 0  # 0=single model, 2-3=ensemble votes
    ensemble_source: str = ""  # ENSEMBLE-2, ENSEMBLE-3, or model name

    # Bear market protection info
    bear_level: str = "NORMAL"  # NORMAL/WATCH/WARNING/CRITICAL
    bear_exit_adjustment: float = 1.0  # Stop loss tightening factor


@dataclass
class ScanResult:
    """Complete scan output."""
    timestamp: str
    regime: RegimeAnalysis
    signals: List[SmartSignal]
    filtered_count: int
    total_scanned: int
    bear_alert_level: str = "NORMAL"
    bear_score: int = 0


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

    def __init__(self, portfolio_value: float = 100000, use_ensemble: bool = True,
                 skip_choppy: bool = False, use_regime_filter: bool = True):
        """
        Initialize the SmartScanner.

        Args:
            portfolio_value: Total portfolio value for position sizing
            use_ensemble: Use 3-model ensemble (Transformer + LSTM + MLP)
            skip_choppy: If True, skip all trades in CHOPPY markets (most conservative)
            use_regime_filter: Apply backtest-validated regime thresholds
        """
        self.portfolio_value = portfolio_value
        self.use_ensemble = use_ensemble
        self.skip_choppy = skip_choppy
        self.use_regime_filter = use_regime_filter

        # Use 3-model ensemble by default (Transformer + LSTM + MLP)
        # Backtest validated: 63.8% win rate, 0.28 Sharpe (best)
        if use_ensemble:
            self.signal_model = HybridSignalModel()
            # Keep GPU model for helper methods (prices, volume, etc.)
            self.gpu_model = self.signal_model.mlp_model  # Access underlying MLP
        else:
            self.signal_model = GPUSignalModel()
            self.gpu_model = self.signal_model

        self.regime_detector = MarketRegimeDetector()
        # Regime-aware filter (Jan 2026 backtest validated thresholds)
        self.regime_filter = RegimeAwareFilter(skip_choppy=skip_choppy)
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
        self.sector_momentum = get_sector_momentum_calculator()
        self.enhanced_calculator = get_enhanced_calculator()
        self.exit_optimizer = ExitOptimizer()
        self.position_sizer = PositionSizer(portfolio_value=portfolio_value)

        # Bear detection for risk-adaptive exits
        self.bear_detector = None
        self._bear_signal = None
        self._bear_level = 'NORMAL'
        try:
            self.bear_detector = FastBearDetector()
        except Exception as e:
            print(f"Warning: Could not initialize bear detector: {e}")

    def _update_bear_status(self):
        """Update cached bear detection status."""
        if self.bear_detector is None:
            return
        try:
            self._bear_signal = self.bear_detector.detect()
            self._bear_level = self._bear_signal.alert_level.upper()
        except Exception as e:
            print(f"Warning: Bear detection failed: {e}")
            self._bear_level = 'NORMAL'

    def get_bear_status(self) -> dict:
        """Get current bear market status."""
        if self._bear_signal is None:
            self._update_bear_status()
        return {
            'alert_level': self._bear_level,
            'score': self._bear_signal.score if self._bear_signal else 0,
            'adjustments': BEAR_EXIT_ADJUSTMENTS.get(self._bear_level, BEAR_EXIT_ADJUSTMENTS['NORMAL'])
        }

    def get_tier(self, strength: float) -> str:
        """Get tier name for signal strength."""
        for tier_name, (low, high) in self.TIERS.items():
            if low <= strength < high:
                return tier_name
        return 'POOR' if strength < 40 else 'ELITE'

    def get_exit_params(self, ticker: str, regime: MarketRegime, signal_strength: float = 70) -> Dict:
        """
        Get exit parameters adjusted for stock tier, signal strength, regime, and bear conditions.
        Uses ExitOptimizer research-backed tier strategies combined with regime adjustments.
        Now includes bear market protection with tighter stops when bear alert elevated.
        """
        # Get tier-based exit strategy from ExitOptimizer (research-backed)
        exit_strategy = self.exit_optimizer.get_exit_strategy(ticker, signal_strength)

        # Use tier-based values as base
        profit_target = exit_strategy.profit_target_1
        stop_loss = exit_strategy.stop_loss
        max_hold = exit_strategy.max_hold_days

        # Get regime-specific adjustments from config
        regime_params = self.config_loader.get_regime_params(regime.value)
        profit_mult = regime_params.get('profit_target_multiplier', 1.0)
        stop_mult = regime_params.get('stop_loss_multiplier', 1.0)

        # Apply regime multipliers to tier-based values
        regime_profit = profit_target * profit_mult
        regime_stop = stop_loss * stop_mult  # stop is negative, so multiplying widens it

        # Get volatility-adjusted exits (additional layer based on current VIX)
        vol_exits = self.regime_detector.get_volatility_adjusted_exits(
            base_profit=regime_profit,
            base_stop=regime_stop
        )

        # Apply bear market adjustments - tighter stops when elevated
        bear_status = self.get_bear_status()
        bear_adj = bear_status['adjustments']
        bear_level = bear_status['alert_level']

        # Tighten stops (multiply by <1 to reduce absolute value = tighter)
        # stop_loss is negative, so we multiply by the adjustment factor
        # E.g., -5% * 0.70 = -3.5% (tighter stop)
        final_stop = vol_exits['stop_loss'] * bear_adj['stop_mult']

        # Reduce profit targets (take profits earlier in bear conditions)
        final_profit = vol_exits['profit_target'] * bear_adj['profit_mult']

        # Reduce hold time in bear conditions
        final_hold = int(max_hold * bear_adj['hold_mult'])

        return {
            'profit_target': round(final_profit, 2),
            'stop_loss': round(final_stop, 2),
            'max_hold_days': max(1, final_hold),  # At least 1 day
            'trailing_trigger': round(exit_strategy.profit_target_1 * bear_adj['profit_mult'], 2),
            'trailing_distance': round(exit_strategy.trailing_distance, 2),
            'use_trailing': exit_strategy.use_trailing,
            'second_target': exit_strategy.profit_target_2,  # For partial exits
            'portion_at_first_target': exit_strategy.portion_at_target_1,
            'volatility_mult': vol_exits['volatility_multiplier'],
            'regime_mult': profit_mult,
            'bear_level': bear_level,
            'bear_adjustment': bear_adj['stop_mult'],
            'stock_tier': self.exit_optimizer.get_stock_tier(ticker).value
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
        if self.skip_choppy:
            print("Mode: CONSERVATIVE (skip_choppy=True)")
        if self.use_regime_filter:
            print("Filter: REGIME-AWARE (backtest-validated thresholds)")
        print()

        # 1. Get market regime and trading decision
        print("[1] Analyzing market regime...")
        regime_analysis = self.regime_detector.detect_regime()

        # Check for data fetch errors
        if hasattr(regime_analysis, 'data_error') and regime_analysis.data_error:
            print("    [ERROR] Market data unavailable - using conservative defaults!")
            print("    [WARNING] Trading signals may be less reliable without regime context")

        print(f"    Regime: {regime_analysis.regime.value.upper()} "
              f"(confidence: {regime_analysis.confidence*100:.0f}%)")
        print(f"    VIX: {regime_analysis.vix_level:.1f} "
              f"({regime_analysis.vix_percentile:.0f}th percentile)")

        # 1b. Check bear detection status
        self._update_bear_status()
        bear_status = self.get_bear_status()
        bear_level = bear_status['alert_level']
        bear_score = bear_status['score']
        print(f"    Bear Alert: {bear_level} (score: {bear_score})")

        # CRITICAL: Skip all new positions in critical bear conditions
        if bear_level == 'CRITICAL':
            print()
            print("=" * 70)
            print("CRITICAL BEAR ALERT - CAPITAL PRESERVATION MODE")
            print(f"Bear Score: {bear_score} - NOT opening new positions")
            print("=" * 70)
            return ScanResult(
                timestamp=datetime.now().isoformat(),
                regime=regime_analysis,
                signals=[],
                filtered_count=0,
                total_scanned=0,
                bear_alert_level=bear_level,
                bear_score=bear_score
            )

        # WARNING: Severely restrict new positions
        if bear_level == 'WARNING':
            print(f"    [WARNING] Bear conditions elevated - reduced sizing & tighter exits")

        # Get trading decision from regime filter
        if self.use_regime_filter:
            trading_decision = self.regime_filter.get_trading_decision()
            print(f"    Decision: {trading_decision.reason}")

            # Early exit if we shouldn't trade
            if not trading_decision.should_trade:
                print()
                print("=" * 70)
                print(f"SKIPPING SCAN: {trading_decision.reason}")
                print("=" * 70)
                return ScanResult(
                    timestamp=datetime.now().isoformat(),
                    regime=regime_analysis,
                    signals=[],
                    filtered_count=0,
                    total_scanned=0,
                    bear_alert_level=bear_level,
                    bear_score=bear_score
                )

        # Get threshold and position multiplier based on filter mode
        if self.use_regime_filter:
            # Use backtest-validated thresholds (Jan 2026 overnight analysis)
            # BULL: 50 (Sharpe 0.43)
            # BEAR: 70 (Sharpe 1.80, 100% win rate!)
            # CHOPPY: 65 (high bar to filter low quality)
            # VOLATILE: 70 (high bar due to uncertainty)
            regime_threshold = self.regime_detector.get_min_signal_threshold(regime_analysis.regime)
            position_mult = self.regime_detector.get_position_multiplier(regime_analysis.regime)
            dynamic_threshold = regime_threshold  # No additional adjustment

            print(f"    Backtest-validated threshold: {regime_threshold:.0f}")
            print(f"    Position multiplier: {position_mult:.1f}x")
        else:
            # Use config-based adjustments (legacy system)
            regime_params = self.config_loader.get_regime_params(regime_analysis.regime.value)
            base_threshold = 50.0
            regime_threshold_adj = regime_params.get('threshold_adjustment', 0)
            min_threshold = base_threshold + regime_threshold_adj

            position_mult = regime_params.get('position_multiplier', 1.0)

            # Get dynamic threshold adjustments based on VIX/trends
            dynamic_adj = self.regime_detector.get_dynamic_threshold(base_threshold=min_threshold)
            dynamic_threshold = dynamic_adj['threshold']

            print(f"    Regime adjustment: {regime_threshold_adj:+.0f}")
            print(f"    Position multiplier: {position_mult:.1f}x (from config)")
            print(f"    Base threshold: {min_threshold:.0f}")
            print(f"    Dynamic threshold: {dynamic_threshold:.1f}")

        # Log regime-specific elite stocks
        elite_stocks = self.config_loader.get_regime_elite_stocks(regime_analysis.regime.value)
        print(f"    Elite stocks for this regime: {', '.join(elite_stocks[:5])}")

        # 2. Run model scan (ensemble or single)
        print()
        if self.use_ensemble:
            print("[2] Running 3-model ensemble scan (Transformer + LSTM + MLP)...")
            # Use min_votes=2 for optimal selectivity (backtest validated)
            model_signals = self.signal_model.scan_all(threshold=0, min_votes=1)  # Get all, filter later
        else:
            print("[2] Running GPU model scan...")
            model_signals = self.signal_model.scan_all()
        print(f"    Found {len(model_signals)} raw signals")

        # Get real prices from GPU model (already fetched during scan)
        stock_prices = self.gpu_model.get_last_prices()
        print(f"    Loaded {len(stock_prices)} real prices")

        # Check for stale data (older than 3 calendar days)
        stale_data = self.gpu_model.check_data_staleness(max_days=3)
        if stale_data:
            print(f"    [WARNING] {len(stale_data)} tickers have stale data:")
            for ticker, days in sorted(stale_data.items(), key=lambda x: -x[1])[:5]:
                print(f"      {ticker}: {days} days old")
            if len(stale_data) > 5:
                print(f"      ... and {len(stale_data) - 5} more")

        # 2b. Fetch sector momentum data (weak sectors outperform by +0.27% per trade)
        print()
        print("[2b] Fetching sector momentum...")
        try:
            sector_momentum_data = self.sector_momentum.get_sector_momentum(force_refresh=True)
            if not sector_momentum_data:
                print("    [WARNING] Sector momentum data unavailable - using neutral adjustments")
            else:
                print(f"    Sector momentum loaded for {len(sector_momentum_data)} sectors")
                for etf, mom in sorted(sector_momentum_data.items(), key=lambda x: x[1]):
                    cat = self.sector_momentum.get_momentum_category(mom)
                    print(f"    {etf}: {mom:+.1f}% ({cat})")
        except Exception as e:
            print(f"    [WARNING] Sector momentum fetch failed: {e}")
            print("    Continuing with neutral sector adjustments")
            sector_momentum_data = {}

        # 3. Filter and enhance signals
        print()
        print("[3] Filtering and enhancing signals...")

        # Check portfolio status for position limits
        portfolio_status = self.position_sizer.get_portfolio_status()
        print(f"    Current positions: {portfolio_status['positions']}/{portfolio_status['max_positions']}")
        print(f"    Portfolio heat: {portfolio_status['total_heat']:.1f}% (max {portfolio_status['max_heat']:.0f}%)")

        smart_signals = []
        filtered_count = 0
        already_holding_count = 0
        missing_price_count = 0

        for sig in model_signals:
            # Skip if we already hold this position
            if self.position_sizer.is_already_holding(sig.ticker):
                already_holding_count += 1
                continue
            # Get per-stock threshold (may override regime threshold)
            stock_threshold = self.config_loader.get_signal_threshold(
                sig.ticker, base_threshold=dynamic_threshold
            )

            # Apply the higher of dynamic or per-stock threshold
            effective_threshold = max(dynamic_threshold, stock_threshold)

            if sig.signal_strength < effective_threshold:
                filtered_count += 1
                continue

            # For ensemble: require minimum votes (backtest shows 2+ is optimal)
            if self.use_ensemble and hasattr(sig, 'votes'):
                if sig.votes < 2:
                    filtered_count += 1
                    continue

            # Get exit params for this stock + regime + signal strength
            exit_params = self.get_exit_params(sig.ticker, regime_analysis.regime, sig.signal_strength)

            # Get sector momentum data
            sector_adj = self.sector_momentum.get_stock_momentum_adjustment(sig.ticker)

            # Get consecutive down days and volume ratio from GPU model's cached data
            # (These are calculated during the scan)
            consecutive_down = self.gpu_model.get_consecutive_down_days(sig.ticker)
            volume_ratio = self.gpu_model.get_volume_ratio(sig.ticker)

            # NEW: Get RSI divergence and down-day status for enhanced multipliers
            has_rsi_divergence = self.gpu_model.has_rsi_divergence(sig.ticker)
            is_down_day = self.gpu_model.is_down_day(sig.ticker)

            # Use enhanced calculator for unified signal strength adjustment
            # This combines: regime, sector momentum, day-of-week, consecutive down-days,
            # volume profile, stock tier multipliers, RSI divergence, and volume exhaustion
            enhanced_adj = self.enhanced_calculator.calculate_enhanced_strength(
                ticker=sig.ticker,
                base_strength=sig.signal_strength,
                regime=regime_analysis.regime.value,
                sector_name=sector_adj.sector_name,
                sector_momentum=sector_adj.momentum_5d,
                signal_date=datetime.now(),
                consecutive_down_days=consecutive_down,
                volume_ratio=volume_ratio,
                has_rsi_divergence=has_rsi_divergence,
                is_down_day=is_down_day
            )

            adjusted_strength = enhanced_adj.final_strength

            # Extra boost for regime-elite stocks (config-driven)
            if sig.ticker in elite_stocks:
                adjusted_strength = min(100, adjusted_strength * 1.05)

            # Additional boost/penalty from config for historically strong/weak stocks
            if self.config_loader.is_strong_performer(sig.ticker):
                adjusted_strength = min(100, adjusted_strength * 1.05)
            elif self.config_loader.is_high_risk_stock(sig.ticker):
                adjusted_strength = min(100, adjusted_strength * 0.90)

            # Check for avoid combos (e.g., Consumer sector consistently underperforms)
            is_avoid, avoid_reason = self.sector_momentum.is_avoid_combo(
                sig.ticker, sector_adj.momentum_5d
            )
            if is_avoid:
                adjusted_strength = min(100, adjusted_strength * 0.85)

            # Calculate position size using volatility-based sizing
            # Validate price data exists - don't use hardcoded defaults
            if sig.ticker not in stock_prices:
                print(f"    [WARNING] Missing price for {sig.ticker} - skipping signal")
                missing_price_count += 1
                continue
            current_price = stock_prices[sig.ticker]
            if current_price <= 0 or current_price > 10000:
                print(f"    [WARNING] Invalid price ${current_price:.2f} for {sig.ticker} - skipping")
                missing_price_count += 1
                continue

            # Get volatility-based sizing
            vol_metrics = self.volatility_sizer.calculate_position_size(
                sig.ticker, current_price, stop_loss_pct=abs(exit_params['stop_loss'])
            )

            # Use volatility-adjusted shares, scaled by position multiplier
            shares = int(vol_metrics.suggested_shares * position_mult)
            shares = max(1, shares)

            # Calculate research-based position sizing
            position_size = self.position_sizer.calculate_position_size(
                ticker=sig.ticker,
                current_price=current_price,
                signal_strength=adjusted_strength,
                atr_pct=vol_metrics.atr_percent,
                volume_ratio=enhanced_adj.volume_ratio,
                consecutive_down_days=enhanced_adj.consecutive_down_days
            )

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
                tier=self.get_tier(adjusted_strength),
                # Sector momentum fields
                sector_etf=sector_adj.etf,
                sector_name=sector_adj.sector_name,
                sector_momentum_5d=sector_adj.momentum_5d,
                sector_momentum_category=sector_adj.momentum_category,
                sector_momentum_boost=sector_adj.strength_boost,
                # Enhanced signal adjustments
                day_of_week=enhanced_adj.day_name,
                day_of_week_multiplier=enhanced_adj.day_of_week_multiplier,
                consecutive_down_days=enhanced_adj.consecutive_down_days,
                consecutive_down_multiplier=enhanced_adj.consecutive_down_multiplier,
                volume_ratio=enhanced_adj.volume_ratio,
                volume_multiplier=enhanced_adj.volume_multiplier,
                stock_tier=enhanced_adj.stock_tier,
                stock_tier_multiplier=enhanced_adj.stock_tier_multiplier,
                # Research-based position sizing (from position_sizer.py)
                position_size_pct=position_size.size_pct,
                position_dollar_size=position_size.dollar_size,
                position_risk_dollars=position_size.risk_dollars,
                signal_quality=position_size.quality.value,
                skip_trade=position_size.skip_trade,
                # Ensemble info (3-model voting)
                ensemble_votes=getattr(sig, 'votes', 0),
                ensemble_source=getattr(sig, 'source', 'MLP'),
                # Bear market protection info
                bear_level=bear_level,
                bear_exit_adjustment=exit_params.get('bear_adjustment', 1.0)
            )
            smart_signals.append(smart_signal)

        # Sort by adjusted strength
        smart_signals.sort(key=lambda x: x.adjusted_strength, reverse=True)

        print(f"    Passed threshold filter: {len(smart_signals)}")
        print(f"    Filtered out: {filtered_count}")
        if already_holding_count > 0:
            print(f"    Already holding: {already_holding_count}")
        if missing_price_count > 0:
            print(f"    [WARNING] Missing prices: {missing_price_count} signals skipped")

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

        # 6. Apply position sizing skip filter
        print()
        print("[6] Applying position sizing filter...")
        skip_trade_signals = [s for s in final_signals if s.skip_trade]
        actionable_signals = [s for s in final_signals if not s.skip_trade]

        # Enforce max positions limit
        can_add, reason = self.position_sizer.can_add_position()
        if not can_add:
            print(f"    WARNING: {reason}")
            # Still show signals but mark them as informational
            for sig in actionable_signals:
                sig.skip_trade = True
            skip_trade_signals = actionable_signals
            actionable_signals = []

        print(f"    Actionable signals: {len(actionable_signals)}")
        if skip_trade_signals:
            print(f"    Skip (low quality/limits): {len(skip_trade_signals)}")
            for sig in skip_trade_signals[:3]:
                print(f"      - {sig.ticker}: {sig.signal_quality}")

        # Use actionable signals as final, but keep skip signals for reference
        final_signals = actionable_signals
        total_filtered += len(skip_trade_signals)

        # 7. Create result
        result = ScanResult(
            timestamp=datetime.now().isoformat(),
            regime=regime_analysis,
            signals=final_signals,
            filtered_count=total_filtered,
            total_scanned=len(model_signals),
            bear_alert_level=bear_level,
            bear_score=bear_score
        )

        # 8. Print summary
        self._print_summary(result)

        # 9. Save results
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

        # Show bear alert status
        if result.bear_alert_level != 'NORMAL':
            print(f"\n*** BEAR ALERT: {result.bear_alert_level} (score: {result.bear_score}) ***")
            print("    Position sizes reduced, stops tightened")
        else:
            print(f"\nBear Status: NORMAL (score: {result.bear_score})")

        print(f"\nSignals: {len(result.signals)} passed / "
              f"{result.filtered_count} filtered / "
              f"{result.total_scanned} scanned")

        if result.signals:
            print("\n--- TOP SIGNALS (with Position Sizing) ---")
            print(f"{'Ticker':<6} {'Price':>9} {'Tier':<8} {'Str':>5} {'Quality':<10} "
                  f"{'Size%':>6} {'$Size':>9} {'$Risk':>7} {'E[R]':>6}")
            print("-" * 90)

            for sig in result.signals[:10]:
                skip_marker = " [SKIP]" if sig.skip_trade else ""
                print(f"{sig.ticker:<6} ${sig.current_price:>8.2f} {sig.tier:<8} {sig.adjusted_strength:>5.1f} "
                      f"{sig.signal_quality:<10} {sig.position_size_pct:>5.1f}% "
                      f"${sig.position_dollar_size:>8,.0f} ${sig.position_risk_dollars:>6,.0f} "
                      f"{sig.expected_return:>+5.1f}%{skip_marker}")

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
            'bear_status': {
                'alert_level': result.bear_alert_level,
                'score': result.bear_score,
                'position_multiplier': BEAR_EXIT_ADJUSTMENTS.get(
                    result.bear_alert_level, BEAR_EXIT_ADJUSTMENTS['NORMAL']
                )['stop_mult']
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

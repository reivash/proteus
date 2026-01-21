"""
SmartScannerV2 - Refactored scanner using unified modules.

Uses:
- unified_config.json for all configuration
- UnifiedSignalCalculator for signal adjustments
- UnifiedPositionSizer for position sizing
- PositionRebalancer for exit monitoring
- HybridSignalModel (Jan 2026) - LSTM V2 primary with MLP fallback

Simplified architecture with single source of truth.

Model Selection (Jan 4, 2026 A/B Test Results):
- Hybrid: LSTM V2 primary (79.2% win, 4.96 Sharpe) + MLP fallback [RECOMMENDED]
- MLP: Original GPU model (62.2% win, 1.74 Sharpe)
- LSTM: Pure LSTM V2 (more selective, higher quality)

Updated Jan 20, 2026: Added proper error handling and logging.
"""

import os
import sys
import json
import traceback
from datetime import datetime
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Setup logging
from utils.logging_config import get_logger, LogContext
logger = get_logger('proteus.scanner')

from models.gpu_signal_model import GPUSignalModel
# Jan 5, 2026: Switch to unified regime detector (HMM + rule-based ensemble)
from analysis.unified_regime_detector import UnifiedRegimeDetector, DetectionMethod, UnifiedRegimeResult
from trading.sector_correlation import filter_correlated_signals
from trading.sector_momentum import get_sector_momentum_calculator
# Jan 5, 2026: Switch to penalties-only calculator
# Research showed: penalties-only = 62.5% win, 1.72% avg return
#                  full (115 mods) = 59.2% win, 0.76% avg return
# To revert: change PenaltiesOnlyCalculator back to UnifiedSignalCalculator
from trading.penalties_only_calculator import PenaltiesOnlyCalculator as SignalCalculator
from trading.unified_position_sizer import UnifiedPositionSizer
from trading.position_rebalancer import PositionRebalancer, ExitReason
from trading.regime_adaptive_trader import RegimeAdaptiveTrader, TradingMode
from data.fetchers.earnings_calendar import EarningsCalendarFetcher
from analysis.fast_bear_detector import FastBearDetector

# Model type for selection
ModelType = Literal['hybrid', 'mlp', 'lstm']

# Trading mode for regime adaptation
TradingModeType = Literal['aggressive', 'balanced', 'conservative']


@dataclass
class SignalV2:
    """Simplified signal with unified adjustments."""
    ticker: str
    timestamp: str

    # Core signal
    raw_strength: float
    adjusted_strength: float
    tier: str
    regime: str

    # Position sizing
    size_pct: float
    dollar_size: float
    shares: int
    risk_dollars: float
    quality: str
    skip_trade: bool

    # Exit strategy
    profit_target_1: float
    profit_target_2: Optional[float]
    stop_loss: float
    max_hold_days: int

    # Context
    price: float
    sector: str
    sector_momentum: float
    boosts_applied: List[str]

    # Warnings
    near_earnings: bool = False
    earnings_warning: str = ""


@dataclass
class ScanResultV2:
    """Simplified scan result."""
    timestamp: str
    regime: str
    regime_confidence: float
    vix: float
    signals: List[SignalV2]
    rebalance_actions: List[Dict]
    stats: Dict
    # Bear detection (Jan 2026)
    bear_score: float = 0.0
    bear_alert_level: str = "NORMAL"
    bear_triggers: List[str] = None
    # Choppiness detection (Jan 2026)
    is_choppy: bool = False
    choppiness_score: float = 0.0
    adx: float = 25.0


class SmartScannerV2:
    """
    Production scanner using unified modules.

    Key simplifications:
    1. Single config source (unified_config.json)
    2. Two multipliers + additive boosts (not 9 multiplicative layers)
    3. Integrated position rebalancing
    4. Model selection: hybrid (default), mlp, lstm

    Jan 4, 2026: Added hybrid model support based on A/B test showing:
    - LSTM V2: 79.2% win rate, 4.96 Sharpe (vs MLP: 62.2%, 1.74)
    """

    def __init__(self, portfolio_value: float = 100000, model: ModelType = 'hybrid',
                 trading_mode: TradingModeType = 'aggressive', use_sentiment: bool = False):
        """
        Initialize SmartScannerV2.

        Args:
            portfolio_value: Total portfolio value for position sizing
            model: Model to use ('hybrid', 'lstm', 'mlp')
            trading_mode: Trading mode ('aggressive', 'balanced', 'conservative')
            use_sentiment: Enable sentiment filtering (Jan 7, 2026)
        """
        self.portfolio_value = portfolio_value
        self.model_type = model
        self.trading_mode_str = trading_mode
        self.use_sentiment = use_sentiment

        # Sentiment scorer (lazy loaded if enabled)
        self.sentiment_scorer = None
        if use_sentiment:
            print("[SCANNER] Sentiment filtering ENABLED")

        # Core components - model selection
        self.gpu_model = GPUSignalModel()  # Always load for helper functions
        # Jan 5, 2026: HMM + rule-based ensemble regime detection
        self.regime_detector = UnifiedRegimeDetector(method=DetectionMethod.ENSEMBLE)

        # Regime-adaptive trader (Jan 4, 2026 - +82% Sharpe improvement)
        mode_map = {
            'aggressive': TradingMode.AGGRESSIVE,
            'balanced': TradingMode.BALANCED,
            'conservative': TradingMode.CONSERVATIVE
        }
        self.regime_trader = RegimeAdaptiveTrader(mode=mode_map.get(trading_mode, TradingMode.AGGRESSIVE))
        print(f"[SCANNER] Trading mode: {trading_mode.upper()} (regime-adaptive)")

        # Load signal model based on selection
        self.signal_model = None
        self.hybrid_model = None

        if model == 'hybrid':
            try:
                from models.hybrid_signal_model import HybridSignalModel
                self.hybrid_model = HybridSignalModel()
                print(f"[SCANNER] Using HYBRID model (LSTM V2 + MLP fallback)")
            except Exception as e:
                print(f"[SCANNER] Hybrid load failed, falling back to MLP: {e}")
                self.model_type = 'mlp'
        elif model == 'lstm':
            try:
                from models.lstm_signal_model import LSTMSignalModel
                self.signal_model = LSTMSignalModel()
                print(f"[SCANNER] Using LSTM V2 model")
            except Exception as e:
                print(f"[SCANNER] LSTM load failed, falling back to MLP: {e}")
                self.model_type = 'mlp'

        if self.model_type == 'mlp':
            print(f"[SCANNER] Using MLP model (GPU)")

        # Unified modules
        self.signal_calc = SignalCalculator()  # Penalties-only (Jan 5, 2026)
        self.position_sizer = UnifiedPositionSizer(portfolio_value=portfolio_value)
        self.rebalancer = PositionRebalancer()

        # Supporting components
        self.sector_momentum = get_sector_momentum_calculator()
        self.earnings_calendar = EarningsCalendarFetcher(
            exclusion_days_before=3,
            exclusion_days_after=1
        )

        # Bear detection (Jan 2026)
        self.bear_detector = FastBearDetector()

        # Load config for thresholds
        self.config = self.signal_calc.config

    def get_base_threshold(self, regime: str) -> float:
        """Get signal threshold adjusted for regime using regime-adaptive trader."""
        # Use regime-adaptive thresholds (Jan 4, 2026 - significantly more aggressive)
        return self.regime_trader.get_threshold(regime)

    def should_trade_in_regime(self, regime: str) -> tuple:
        """Check if trading is allowed in current regime."""
        return self.regime_trader.should_trade(regime)

    def _ensure_sentiment_scorer(self):
        """Lazy load sentiment scorer when needed."""
        if self.use_sentiment and self.sentiment_scorer is None:
            try:
                from src.data.sentiment.unified_sentiment import UnifiedSentimentScorer
                self.sentiment_scorer = UnifiedSentimentScorer(use_gpu=True)
            except Exception as e:
                print(f"[SENTIMENT] Failed to load: {e}")
                self.use_sentiment = False

    def check_sentiment(self, ticker: str, regime: str) -> tuple:
        """
        Check sentiment for a signal.

        Jan 7, 2026: Sentiment filtering logic:
        - BEAR regime + negative sentiment = GOOD (panic = opportunity)
        - BULL regime + positive sentiment = AVOID (price decline justified)
        - CHOPPY = already skipped by regime filter

        Returns: (should_trade, reason)
        """
        if not self.use_sentiment:
            return True, "Sentiment filtering disabled"

        self._ensure_sentiment_scorer()
        if self.sentiment_scorer is None:
            return True, "Sentiment scorer unavailable"

        return self.sentiment_scorer.get_filter_recommendation(ticker, signal_strength=70)

    def check_earnings(self, ticker: str) -> tuple:
        """Check if near earnings."""
        import pandas as pd
        from datetime import timedelta

        today = pd.Timestamp.now().normalize()

        if not self.earnings_calendar.should_trade_on_date(ticker, today):
            earnings_df = self.earnings_calendar.fetch_earnings_dates(ticker)
            if len(earnings_df) > 0:
                future = earnings_df[earnings_df['earnings_date'] >= today - timedelta(days=5)]
                if len(future) > 0:
                    next_date = future['earnings_date'].min()
                    days = (next_date - today).days
                    if days <= 0:
                        return True, f"Earnings {next_date.strftime('%m/%d')}"
                    return True, f"Earnings in {days}d"
            return True, "Near earnings"
        return False, ""

    def check_positions(self, prices: Dict[str, float]) -> List[Dict]:
        """Check existing positions for rebalancing."""
        actions = []

        for ticker, info in self.rebalancer.positions.items():
            if ticker not in prices:
                continue

            entry_price = info.get('entry_price')
            entry_date_str = info.get('entry_date')

            if not entry_price or not entry_date_str:
                continue

            try:
                entry_date = datetime.fromisoformat(entry_date_str)
            except ValueError:
                continue

            status = self.rebalancer.check_position(
                ticker=ticker,
                entry_price=entry_price,
                current_price=prices[ticker],
                entry_date=entry_date
            )

            if status.exit_reason != ExitReason.NONE:
                actions.append({
                    'ticker': ticker,
                    'action': status.exit_reason.value,
                    'exit_pct': status.exit_pct,
                    'pnl_pct': status.pnl_pct,
                    'recommendation': status.recommendation
                })

        return actions

    def scan(self) -> ScanResultV2:
        """Run the model scan based on model type."""
        # 1. Detect market regime
        print()
        print("[1] Detecting market regime...")
        try:
            regime_analysis = self.regime_detector.detect_regime()
            regime = regime_analysis.regime.value if hasattr(regime_analysis.regime, 'value') else str(regime_analysis.regime)
            print(f"    Regime: {regime.upper()} (confidence: {regime_analysis.confidence:.0%})")
            print(f"    VIX: {regime_analysis.vix_level:.1f}")
        except Exception as e:
            print(f"    [WARNING] Regime detection failed: {e}")
            regime = 'choppy'  # Conservative default
            regime_analysis = type('RegimeResult', (), {
                'regime': regime,
                'confidence': 0.5,
                'vix_level': 20.0,
                'early_warning_score': 0
            })()

        # 2. Bear detection
        print()
        print("[2] Running bear detection...")
        try:
            bear_signal = self.bear_detector.detect()
            print(f"    Bear score: {bear_signal.bear_score:.0f}/100 ({bear_signal.alert_level})")
            if bear_signal.triggers:
                for trigger in bear_signal.triggers[:3]:
                    print(f"    - {trigger}")
        except Exception as e:
            print(f"    [WARNING] Bear detection failed: {e}")
            bear_signal = type('BearSignal', (), {
                'bear_score': 0,
                'alert_level': 'UNKNOWN',
                'triggers': [],
                'recommendation': 'Normal operations'
            })()

        # 2b. Choppiness/ADX detection
        try:
            adx = self.gpu_model.get_market_adx() if hasattr(self.gpu_model, 'get_market_adx') else 25.0
            choppiness_score = self.gpu_model.get_choppiness() if hasattr(self.gpu_model, 'get_choppiness') else 50.0
            is_choppy = adx < 20 or choppiness_score > 61.8
            self._is_choppy = is_choppy
            if is_choppy:
                print(f"    [CHOPPY MARKET] ADX={adx:.1f}, Choppiness={choppiness_score:.1f}")
        except Exception as e:
            print(f"    [WARNING] Choppiness detection failed: {e}")
            adx = 25.0
            choppiness_score = 50.0
            is_choppy = False
            self._is_choppy = False

        # Threshold based on regime
        # Note: Lowered thresholds for forward validation period (Jan 2026)
        # Will raise back once we have 30+ days of data
        # Forward validation period (Jan 2026): lowered thresholds significantly
        # to get signals flowing for 30-day paper trading test
        threshold = {
            'bull': 45, 'volatile': 45, 'choppy': 45, 'bear': 50
        }.get(regime, 45)
        print(f"    Signal threshold: {threshold}")

        # Run model scan
        if self.model_type == 'hybrid' and self.hybrid_model:
            # First, run GPU model to populate helper data (close_position, volume_ratio, etc.)
            _ = self.gpu_model.scan_all()  # Populates _last_scan_* helper data

            hybrid_signals = self.hybrid_model.scan_all(threshold=30.0)  # Low threshold, filter later
            gpu_signals = []
            for hs in hybrid_signals:
                # Create compatible signal object
                class CompatSignal:
                    def __init__(self, hs):
                        self.ticker = hs.ticker
                        self.timestamp = hs.timestamp
                        self.signal_strength = hs.signal_strength
                        self.mean_reversion_prob = hs.mean_reversion_prob
                        self.expected_return = hs.expected_return
                        self.confidence = hs.confidence
                        self.source = hs.source  # Track LSTM vs MLP
                        self.consensus = hs.consensus
                gpu_signals.append(CompatSignal(hs))
            print(f"    Hybrid signals: {len(gpu_signals)} (LSTM: {sum(1 for s in gpu_signals if s.source=='LSTM')}, MLP: {sum(1 for s in gpu_signals if s.source=='MLP')})")
        else:
            gpu_signals = self.gpu_model.scan_all()
            print(f"    Raw signals: {len(gpu_signals)}")

        prices = self.gpu_model.get_last_prices()

        # Check for stale data (older than 3 calendar days)
        stale_data = self.gpu_model.check_data_staleness(max_days=3)
        if stale_data:
            print(f"    [WARNING] {len(stale_data)} tickers have stale data:")
            for ticker, days in sorted(stale_data.items(), key=lambda x: -x[1])[:5]:
                print(f"      {ticker}: {days} days old")
            if len(stale_data) > 5:
                print(f"      ... and {len(stale_data) - 5} more")

        # 3. Check existing positions
        print()
        print("[3] Checking existing positions...")
        rebalance_actions = self.check_positions(prices)
        if rebalance_actions:
            print(f"    Actions needed: {len(rebalance_actions)}")
            for action in rebalance_actions:
                print(f"    -> {action['ticker']}: {action['recommendation']}")
        else:
            print("    No rebalancing needed")

        # 4. Get sector momentum
        print()
        print("[4] Loading sector momentum...")
        try:
            sector_data = self.sector_momentum.get_sector_momentum(force_refresh=True)
            if not sector_data:
                print("    [WARNING] Sector momentum unavailable - using neutral adjustments")
            else:
                print(f"    Loaded {len(sector_data)} sectors")
        except Exception as e:
            print(f"    [WARNING] Sector momentum failed: {e}")
            sector_data = {}

        # 5. Filter and enhance signals
        print()
        print("[5] Processing signals...")

        portfolio_status = self.position_sizer.get_status()
        print(f"    Positions: {portfolio_status['positions']}/{portfolio_status['max_positions']}")
        print(f"    Heat: {portfolio_status['heat_pct']:.1f}%")

        signals = []
        filtered = 0
        holding = 0

        for sig in gpu_signals:
            # Skip if already holding
            if self.position_sizer.is_holding(sig.ticker):
                holding += 1
                continue

            # Get context
            sector_adj = self.sector_momentum.get_stock_momentum_adjustment(sig.ticker)
            consecutive_down = self.gpu_model.get_consecutive_down_days(sig.ticker)
            volume_ratio = self.gpu_model.get_volume_ratio(sig.ticker)
            has_rsi_div = self.gpu_model.has_rsi_divergence(sig.ticker)
            is_down_day = self.gpu_model.is_down_day(sig.ticker)
            close_position = self.gpu_model.get_close_position(sig.ticker)
            gap_pct = self.gpu_model.get_gap_pct(sig.ticker)
            sma200_distance = self.gpu_model.get_sma200_distance(sig.ticker)
            day_range_pct = self.gpu_model.get_day_range_pct(sig.ticker)
            drawdown_pct = self.gpu_model.get_drawdown_pct(sig.ticker)
            weekday = datetime.now().weekday()
            is_monday = weekday == 0
            is_tuesday = weekday == 1  # Jan 5, 2026: Tuesday + very_oversold TRAP
            is_wednesday = weekday == 2  # Jan 5, 2026: Wednesday penalty
            is_thursday = weekday == 3
            is_friday = weekday == 4  # Jan 5, 2026: Best 3-day return
            rsi_level = self.gpu_model.get_rsi_level(sig.ticker)  # For day + RSI combos
            atr_pct = self.gpu_model.get_atr_pct(sig.ticker)  # Jan 5, 2026: ATR-based volatility boosts

            # Calculate enhanced signal
            breakdown = self.signal_calc.calculate(
                ticker=sig.ticker,
                base_signal=sig.signal_strength,
                regime=regime,
                is_monday=is_monday,
                is_thursday=is_thursday,
                is_wednesday=is_wednesday,  # Jan 5, 2026: Wednesday WORST day (-4)
                is_tuesday=is_tuesday,  # Jan 5, 2026: Tuesday + very_oversold = 46.5% TRAP
                is_friday=is_friday,  # Jan 5, 2026: Friday = best 3-day return (+1.005%)
                consecutive_down_days=consecutive_down,
                has_rsi_divergence=has_rsi_div,
                rsi_level=rsi_level,  # Jan 5, 2026: For day + RSI combos
                volume_ratio=volume_ratio,
                is_down_day=is_down_day,
                sector_momentum=sector_adj.momentum_5d,
                sector=sector_adj.sector_name,  # Jan 5, 2026: sector-specific boosts
                close_position=close_position,  # Jan 4, 2026: boost near-low, penalize near-high
                gap_pct=gap_pct,  # Jan 5, 2026: boost medium gap downs (-2% to -3%)
                sma200_distance=sma200_distance,  # Jan 5, 2026: boost near SMA200 (support/resistance)
                day_range_pct=day_range_pct,  # Jan 5, 2026: boost wide range, penalize low vol narrow
                drawdown_pct=drawdown_pct,  # Jan 5, 2026: drawdown-aware down streak boost/trap
                atr_pct=atr_pct  # Jan 5, 2026: ATR-based volatility boosts
            )

            # Apply threshold
            if breakdown.final_signal < threshold:
                if sig.signal_strength >= 50:  # Only log significant ones
                    print(f"    [FILTERED] {sig.ticker}: {sig.signal_strength:.1f} -> {breakdown.final_signal:.1f} (threshold {threshold})")
                filtered += 1
                continue

            # Apply sentiment filter (Jan 7, 2026)
            if self.use_sentiment:
                sentiment_ok, sentiment_reason = self.check_sentiment(sig.ticker, regime)
                if not sentiment_ok:
                    print(f"    [SENTIMENT FILTER] {sig.ticker}: {sentiment_reason}")
                    filtered += 1
                    continue

            # Get position sizing - validate price data exists
            if sig.ticker not in prices:
                print(f"    [WARNING] Missing price for {sig.ticker} - skipping")
                filtered += 1
                continue
            price = prices[sig.ticker]
            if price <= 0 or price > 10000:
                print(f"    [WARNING] Invalid price ${price:.2f} for {sig.ticker} - skipping")
                filtered += 1
                continue
            atr_pct = self.gpu_model.get_atr_pct(sig.ticker) if hasattr(self.gpu_model, 'get_atr_pct') else 2.5

            sizing = self.position_sizer.calculate_size(
                ticker=sig.ticker,
                current_price=price,
                signal_strength=breakdown.final_signal,
                regime=regime,
                atr_pct=atr_pct
            )

            # Apply choppiness adjustment (Jan 2026) - reduce size in ranging markets
            # Choppy markets have low win rate (53.4% vs 76.4% trending)
            if hasattr(self, '_is_choppy') and self._is_choppy:
                choppy_mult = 0.5  # 50% size reduction in choppy markets
                sizing.size_pct = sizing.size_pct * choppy_mult
                sizing.dollar_size = sizing.dollar_size * choppy_mult
                sizing.shares = int(sizing.shares * choppy_mult)
                sizing.risk_dollars = sizing.risk_dollars * choppy_mult

            # Apply hierarchical HMM risk multiplier (MDPI 2025 research)
            # Meta-regime aware: reduces size in high-uncertainty regimes
            if hasattr(self, '_hierarchical_risk_mult') and self._hierarchical_risk_mult < 1.0:
                h_mult = self._hierarchical_risk_mult
                sizing.size_pct = sizing.size_pct * h_mult
                sizing.dollar_size = sizing.dollar_size * h_mult
                sizing.shares = int(sizing.shares * h_mult)
                sizing.risk_dollars = sizing.risk_dollars * h_mult

            # Apply FRED macro risk multiplier (Chen 2009 research)
            # Reduces size when recession risk is elevated
            if hasattr(self, '_macro_risk_mult') and self._macro_risk_mult < 1.0:
                m_mult = self._macro_risk_mult
                sizing.size_pct = sizing.size_pct * m_mult
                sizing.dollar_size = sizing.dollar_size * m_mult
                sizing.shares = int(sizing.shares * m_mult)
                sizing.risk_dollars = sizing.risk_dollars * m_mult

            # Apply model disagreement risk multiplier
            # Reduces size when models disagree significantly
            if hasattr(self, '_disagreement_risk_mult') and self._disagreement_risk_mult < 1.0:
                d_mult = self._disagreement_risk_mult
                sizing.size_pct = sizing.size_pct * d_mult
                sizing.dollar_size = sizing.dollar_size * d_mult
                sizing.shares = int(sizing.shares * d_mult)
                sizing.risk_dollars = sizing.risk_dollars * d_mult

            # Apply correlation regime risk multiplier
            # Reduces size when correlations spike (crisis conditions)
            if hasattr(self, '_correlation_risk_mult') and self._correlation_risk_mult < 1.0:
                c_mult = self._correlation_risk_mult
                sizing.size_pct = sizing.size_pct * c_mult
                sizing.dollar_size = sizing.dollar_size * c_mult
                sizing.shares = int(sizing.shares * c_mult)
                sizing.risk_dollars = sizing.risk_dollars * c_mult

            # Get exit rules
            exit_config = self.position_sizer.config.get('exit_strategy', {}).get(
                breakdown.tier,
                self.position_sizer.config.get('exit_strategy', {}).get('average', {})
            )

            # Apply bear adjustments to exits (tighter stops in elevated conditions)
            bear_exit_mult = {
                'NORMAL': 1.0, 'WATCH': 0.85, 'WARNING': 0.70, 'CRITICAL': 0.50
            }.get(bear_signal.alert_level, 1.0)

            # Apply to exit config
            adjusted_stop = exit_config.get('stop_loss', -2.5) * bear_exit_mult
            adjusted_target = exit_config.get('profit_target_1', 2.0) * (
                0.9 if bear_signal.alert_level == 'WATCH' else
                0.75 if bear_signal.alert_level == 'WARNING' else
                0.60 if bear_signal.alert_level == 'CRITICAL' else 1.0
            )
            adjusted_hold = int(exit_config.get('max_hold_days', 3) * bear_exit_mult)

            # Check earnings
            near_earnings, earnings_warn = self.check_earnings(sig.ticker)

            signal = SignalV2(
                ticker=sig.ticker,
                timestamp=sig.timestamp,
                raw_strength=sig.signal_strength,
                adjusted_strength=breakdown.final_signal,
                tier=breakdown.tier,
                regime=regime,
                size_pct=sizing.size_pct,
                dollar_size=sizing.dollar_size,
                shares=sizing.shares,
                risk_dollars=sizing.risk_dollars,
                quality=sizing.quality.value,
                skip_trade=sizing.skip_trade or near_earnings,
                profit_target_1=round(adjusted_target, 2),  # Bear-adjusted
                profit_target_2=exit_config.get('profit_target_2'),
                stop_loss=round(adjusted_stop, 2),  # Bear-adjusted (tighter)
                max_hold_days=max(1, adjusted_hold),  # Bear-adjusted (shorter)
                price=price,
                sector=sector_adj.sector_name,
                sector_momentum=sector_adj.momentum_5d,
                boosts_applied=list(breakdown.boosts_applied.keys()),
                near_earnings=near_earnings,
                earnings_warning=earnings_warn
            )
            signals.append(signal)

        # Sort by adjusted strength
        signals.sort(key=lambda x: x.adjusted_strength, reverse=True)

        # Apply sector correlation filter
        print()
        print("[6] Applying sector filter...")
        signal_dicts = [asdict(s) for s in signals]
        # Forward validation period (Jan 2026): relaxed sector limits
        # to allow more signals through for paper trading test
        correlation_result = filter_correlated_signals(
            signal_dicts,
            max_per_sector=3,  # Was 2
            max_correlated_total=4,  # Was 3
            strength_key='adjusted_strength'
        )
        passed_tickers = {s['ticker'] for s in correlation_result.passed_signals}
        sector_filtered = [s for s in signals if s.ticker in passed_tickers]
        sector_removed = len(signals) - len(sector_filtered)

        # Final filter: remove earnings and skipped
        actionable = [s for s in sector_filtered if not s.skip_trade]
        skipped = len(sector_filtered) - len(actionable)

        print(f"    Passed: {len(actionable)}")
        print(f"    Filtered: {filtered + sector_removed + skipped + holding}")

        # Build result
        result = ScanResultV2(
            timestamp=datetime.now().isoformat(),
            regime=regime,
            regime_confidence=regime_analysis.confidence,
            vix=regime_analysis.vix_level,
            signals=actionable,
            rebalance_actions=rebalance_actions,
            stats={
                'total_scanned': len(gpu_signals),
                'threshold': threshold,
                'passed': len(actionable),
                'filtered': filtered,
                'sector_removed': sector_removed,
                'skipped': skipped,
                'already_holding': holding
            },
            bear_score=bear_signal.bear_score,
            bear_alert_level=bear_signal.alert_level,
            bear_triggers=bear_signal.triggers,
            is_choppy=is_choppy,
            choppiness_score=choppiness_score,
            adx=adx
        )

        self._print_summary(result)
        self._save_results(result)

        return result

    def _print_summary(self, result: ScanResultV2):
        """Print scan summary."""
        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)

        # Bear status with color indication
        bear_indicator = {
            'CRITICAL': '!!!',
            'WARNING': '!!',
            'WATCH': '!',
            'NORMAL': ''
        }.get(result.bear_alert_level, '')

        choppy_indicator = " [CHOPPY]" if result.is_choppy else ""
        print(f"\nRegime: {result.regime.upper()} | VIX: {result.vix:.1f} | "
              f"Bear: {result.bear_score:.0f}/100 {bear_indicator} | "
              f"ADX: {result.adx:.1f}{choppy_indicator}")
        print(f"Signals: {len(result.signals)} actionable")

        if result.rebalance_actions:
            print("\n--- REBALANCING ACTIONS ---")
            for action in result.rebalance_actions:
                print(f"  {action['ticker']}: {action['recommendation']}")

        if result.signals:
            print("\n--- TOP SIGNALS ---")
            print(f"{'Ticker':<6} {'Tier':<8} {'Str':>5} {'Size%':>6} "
                  f"{'$Size':>9} {'Quality':<10} {'Boosts'}")
            print("-" * 75)

            for sig in result.signals[:10]:
                boosts = ', '.join(sig.boosts_applied) if sig.boosts_applied else '-'
                print(f"{sig.ticker:<6} {sig.tier:<8} {sig.adjusted_strength:>5.1f} "
                      f"{sig.size_pct:>5.1f}% ${sig.dollar_size:>8,.0f} "
                      f"{sig.quality:<10} {boosts}")

        print()

    def _create_error_result(self, error_type: str, error_msg: str) -> ScanResultV2:
        """Create an error result when scan fails."""
        logger.error(f"Scan failed: {error_type} - {error_msg}")
        return ScanResultV2(
            timestamp=datetime.now().isoformat(),
            regime='unknown',
            regime_confidence=0,
            vix=0,
            signals=[],
            rebalance_actions=[],
            stats={'error': True, 'error_type': error_type, 'error_msg': error_msg},
            bear_score=0,
            bear_alert_level='UNKNOWN',
            bear_triggers=[],
            is_choppy=False,
            choppiness_score=0,
            adx=0
        )

    def _save_results(self, result: ScanResultV2):
        """Save results to file."""
        output_dir = "data/smart_scans"
        os.makedirs(output_dir, exist_ok=True)

        output = {
            'timestamp': result.timestamp,
            'regime': result.regime,
            'regime_confidence': result.regime_confidence,
            'vix': result.vix,
            'bear_score': result.bear_score,
            'bear_alert_level': result.bear_alert_level,
            'bear_triggers': result.bear_triggers or [],
            'is_choppy': bool(result.is_choppy),
            'choppiness_score': float(result.choppiness_score),
            'adx': float(result.adx),
            'stats': result.stats,
            'rebalance_actions': result.rebalance_actions,
            'signals': [asdict(s) for s in result.signals[:20]]
        }

        # Save latest
        latest = os.path.join(output_dir, "latest_scan.json")
        with open(latest, 'w') as f:
            json.dump(output, f, indent=2)

        # Save dated
        dated = os.path.join(output_dir, f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(dated, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"[SAVED] {latest}")


def run_scan_v2():
    """Entry point."""
    scanner = SmartScannerV2()
    return scanner.scan()


if __name__ == "__main__":
    run_scan_v2()

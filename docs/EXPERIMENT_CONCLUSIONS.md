# Experiment Conclusions

> Auto-generated: 2026-01-21 09:07
> Source: Extracted from experiment docstrings

---

## Summary by Phase

### Phase 1: Baseline (exp001-007)

#### exp001_baseline.py
**Objective**: Establish baseline performance using Moving Average Crossover

#### exp002_xgboost.py
**Objective**: Beat baseline using XGBoost with technical indicators

#### exp003_lstm.py
**Objective**: Leverage GPU-accelerated LSTM for temporal pattern recognition

#### exp004_tech_stocks.py
**Objective**: Test XGBoost on tech stocks (AI sector focus)

#### exp005_xgboost_tuned.py
**Objective**: Optimize XGBoost hyperparameters using Optuna

#### exp006_ensemble.py
**Objective**: Ensemble methods for robust predictions

#### exp007_multi_timeframe.py
**Objective**: Test prediction accuracy across multiple time horizons

---

### Phase 2: Mean Reversion (exp008-019)

#### exp008_bear_market_test.py
**Objective**: Test mean reversion strategy in 2022 bear market

#### exp008_earnings_filter.py
**Objective**: Test mean reversion with earnings date filter
**Results**: /guidance.

#### exp008_expand_universe.py
**Objective**: Expand stock universe with optimized parameters

#### exp008_fomc_filter.py
**Objective**: Test FOMC meeting filter for policy-driven volatility protection

#### exp008_mean_reversion.py
**Objective**: Test mean reversion strategy on stock overcorrections

#### exp008_multi_stock_test.py

#### exp008_parameter_comparison.py
**Objective**: Compare universal vs optimized parameters

#### exp008_parameter_optimization.py
**Objective**: Optimize mean reversion parameters per stock

#### exp008_position_sizing.py
**Objective**: Test dynamic position sizing vs fixed position sizing

#### exp008_regime_filtered.py
**Objective**: Test mean reversion with market regime filter

#### exp008_sector_diversification.py
**Objective**: Diversify trading universe across sectors

#### exp008_universe_optimization.py
**Objective**: Review marginal performers (AAPL, CVX) for potential removal

#### exp008_vix_filter.py
**Objective**: Test VIX filter for extreme volatility protection

#### exp009_paper_trading_validation.py
**Objective**: Validate paper trading system against backtest results
**Results**: HYPOTHESIS:

#### exp009_simple_validation.py
**Objective**: Simple validation that paper trading components work correctly

#### exp009_simple_validation_fixed.py
**Objective**: Simple validation that paper trading components work correctly

#### exp010_exit_optimization.py
**Objective**: Optimize exit strategy to maximize returns while maintaining win rate
**Results**: - Trailing stops: Higher returns, similar win rate
- Time-decay: Lower max hold losses
- Partial profit: Better risk-adjusted returns

#### exp011_sentiment_enhanced_panic.py
**Objective**: Enhance panic sell detection with sentiment analysis
**Results**: S:
- Win rate improvement: 77.3% → 85%+ (filter out false signals)
- Better Sharpe ratio: Avoid information-driven drops
- Fewer total trades but higher quality signals

#### exp012_dynamic_position_sizing.py
**Objective**: Optimize returns through dynamic position sizing
**Results**: S:
- Same win rate (87.5%) - not changing strategy
- Higher returns (+15-25%) - better capital allocation
- Lower risk-adjusted returns - Sharpe improvement
- Fewer total trades - skip LOW confidence signals

#### exp013_momentum_capture.py
**Objective**: Capture momentum after panic sell reversals for higher returns
**Results**: S:
- Win rate: Same (87.5%) - not changing entry
- Avg gain: +30-50% improvement (6.35% → 9%)
- Total return: +20-25pp improvement
- Hold time: Slightly longer (capture full bounce)
- Risk: Same or better (trailing stops protect)

#### exp014_stock_selection_optimization.py
**Objective**: Identify which stocks have the best mean reversion performance
**Results**: S:
- Find 4-6 Tier A stocks with >70% win rate
- Avoid 2-3 Tier C stocks with <55% win rate
- Expected return improvement: +30-50pp by selective trading
- Reduced drawdown: -40% by avoiding losers

#### exp019_sentiment_integration.py
**Objective**: Integrate real-time sentiment to filter panic sell signals
**Results**: S:
- Win rate: +7-10pp improvement (78% → 85-88%)
- Losing trades: -40% reduction (skip bad signals)
- Total return: +5-10pp improvement
- Trade frequency: -20% (skip LOW confidence)

---

### Phase 3: Position & Exit Optimization (exp020-045)

#### exp020_entry_timing_optimization.py
**Objective**: Optimize entry timing after panic sell detection
**Results**: S:
- Next day open: +1-2% better entry price
- Next day low: +2-4% better entry (theoretical max)
- Total return improvement: +15-25%
- Same win rate (timing doesn't affect reversal)

#### exp021_expand_tier_a_universe.py
**Objective**: Expand Tier A stock universe from 3 to 8-10 stocks
**Results**: S:
- Find 3-6 new Tier A stocks (>70% win rate)
- Expand monitoring from 3 to 8-10 stocks
- Increase trade frequency by 2-3x
- Total return improvement: +5-10pp from more opportunities
- Diversification: Reduced single-stock risk

#### exp023_market_regime_filtering.py
**Objective**: Filter panic sell signals by market regime to improve win rate
**Results**: S:
- Win rate: +7-12pp improvement (78.7% → 85-90%)
- Trade frequency: -20% (skip bear market signals)
- Losing trades: -40-60% reduction
- Sharpe ratio: +20-30% improvement
- Total return: Similar or better (fewer losses offset by fewer trades)

#### exp024_expand_tier_a_round2.py
**Objective**: Expand Tier A universe from 5 to 9-12 stocks (Round 2)
**Results**: S:
- Find 4-6 new Tier A stocks
- Expand from 5 to 9-11 total Tier A stocks
- Trade frequency: ~25 → ~45-55 trades/year
- Return improvement: +3-5pp from increased activity
- Sector diversification: Reduce tech concentration

#### exp025_expand_tier_a_round3.py
**Objective**: Expand Tier A universe from 10 to 12-15 stocks (Round 3)
**Results**: S:
- Find 3-4 new Tier A stocks
- Expand from 10 to 13-14 total Tier A stocks
- Trade frequency: ~55 → ~65-70 trades/year
- Return improvement: +2-3pp from increased activity
- Better sector diversification (add energy, industrials, consumer)

#### exp026_midcap_expansion.py
**Objective**: Test mid-cap stocks (S&P 400) for Tier A candidates
**Results**: S:
- Find 2-4 new Tier A stocks
- Expand from 14 to 16-18 total
- Trade frequency: ~70 → ~80-90 trades/year
- Better volatility exposure (mid-caps more volatile)

#### exp027_midcap_batch2.py
**Objective**: Test second batch of mid-cap stocks for additional Tier A candidates
**Results**: S:
- Find 5-6 new Tier A stocks (22% hit rate)
- Expand from 18 to 23-24 total
- Trade frequency: ~90 → ~115-120 trades/year
- Better sector diversification (REITs, more healthcare)

#### exp028_smallcap_exploration.py
**Objective**: Test small-cap stocks ($1B-$5B) for Tier A candidates
**Results**: - Batch 1: 22% hit rate (excellent)
- Batch 2: 8.3% hit rate (declining)
- Cumulative: 14.3% hit rate across 42 stocks

#### exp029_position_sizing.py
**Objective**: Optimize position sizing for Tier A portfolio
**Results**: S:
- Kelly criterion: Highest returns but potentially high risk
- Sharpe weighting: Best risk-adjusted performance
- Win rate weighting: Conservative, stable
- Equal weighting: Safe baseline

#### exp030_portfolio_correlation.py
**Objective**: Analyze portfolio correlation and sector concentration risk
**Results**: S:

#### exp031_earnings_filter.py
**Objective**: Test earnings calendar filtering to improve win rate and reduce risk
**Results**: S:

#### exp032_multitimeframe_confirmation.py
**Objective**: Test multi-timeframe signal confirmation to improve win rate
**Results**: S:

#### exp033_signal_quality_scoring.py
**Objective**: Implement signal quality scoring to push beyond 85% win rate
**Results**: S:

#### exp034_volatility_regime_adaptation.py
**Objective**: Adapt strategy to volatility regimes for 90%+ win rate
**Results**: S:
- Skip low-volatility periods (improve win rate)
- Trade more during high-volatility (capture opportunities)
- Overall win rate: 88.7% → 90%+
- Better risk-adjusted returns

#### exp035_v16_validation.py
**Objective**: Validate v16.0 enhancements with comprehensive backtest
**Results**: S:
- Validate theoretical predictions from EXP-031/033/034
- Identify any issues before live deployment
- Build confidence in 89.9% win rate claim

#### exp036_quality_threshold_optimization.py
**Objective**: Optimize quality scoring threshold after v16.0 validation failure
**Results**: - Less aggressive filtering = keep more good signals
- Test thresholds: 70, 60, 50, 40, 30, 20, 10, 0 (disabled)
- Find optimal balance: win rate vs trade frequency

#### exp037_parameter_optimization.py
**Objective**: Optimize parameters per stock using data-driven backtest
**Results**: for transparency
- VALIDATE before deployment (learned from v15/16 failure)

#### exp038_full_portfolio_optimization.py
**Objective**: Complete per-stock parameter optimization for ALL Tier A stocks
**Results**: (4 stocks tested):
- Average improvement: +16.6pp
- NVDA: +8.3pp (75.0% → 83.3%)
- V: +16.7pp (66.7% → 83.3%)
- GILD: Optimized to z=1.2
- ROAD: +28.6pp (71.4% → 100%!)

#### exp039_exit_strategy_optimization.py
**Objective**: Optimize exit strategy thresholds for maximum returns
**Results**: - Per-stock optimized exit thresholds
- +5-10% total return improvement
- Maintained or improved win rates

#### exp040_position_sizing_optimization.py
**Objective**: Optimize position sizing based on signal strength
**Results**: - Dynamic position sizing strategy
- +15-25% total return improvement
- Maintained/improved Sharpe ratio
- Documented sizing algorithm for deployment

#### exp041_signal_timing_optimization.py
**Objective**: Optimize signal entry timing for improved returns
**Results**: s:
   - Day of week signal occurred
   - Next-day price action (gap up/down)
   - Optimal entry price vs signal day close

#### exp042_tier_a_stock_screening.py
**Objective**: Screen for new Tier A stocks to expand portfolio
**Results**: - 3-5 new Tier A stocks identified
- Portfolio expansion: 23 → 26-28 stocks
- +15-25 trades/year from new stocks
- Increased diversification

#### exp043_new_stocks_optimization.py
**Objective**: Optimize parameters for 8 newly added Tier A stocks
**Results**: - 8 stocks optimized with stock-specific parameters
- Average improvement: +10pp per stock
- Portfolio optimization coverage: 17/31 → 25/31 stocks (81%)
- System win rate: ~78%+ (from ~75%)

#### exp044_final_portfolio_optimization.py
**Objective**: Complete 100% portfolio optimization (final 7 stocks)
**Results**: - All 7 stocks optimized (100% portfolio coverage)
- Average improvement: +10pp per stock
- Portfolio optimization: 100% complete (31/31 stocks)
- System win rate: ~80%+ (from ~78%)
- MILESTONE: Full portfolio optimization complete!

#### exp045_position_sizing_optimization.py
**Objective**: Implement dynamic position sizing based on signal strength
**Results**: - Dynamic position sizing strategy implemented
- +15-25% total return improvement
- Improved Sharpe ratio
- Documented sizing algorithm for deployment

---

### Phase 4: Universe Expansion (exp046-070)

#### exp046_entry_timing_v2.py
**Objective**: Optimize entry timing for better execution prices
**Results**: s:
   - Entry at signal day close (baseline)
   - Entry at next-day open (test gap behavior)
   - Entry at next-day low (test best possible entry)
   - Entry 1 day delay (test patience)

#### exp047_v14_comprehensive_validation.py
**Objective**: Comprehensive validation of v14.0-DYNAMIC-SIZING system
**Results**: - Win rate: ~80%+ (vs 63.6% baseline)
- Avg return per stock: ~20%+ (vs 13% baseline)
- Total portfolio return: Significant improvement
- Sharpe ratio: >5.0
- Validation of v14.0 readiness for live trading

#### exp048_underperformer_rescue.py
**Objective**: Re-optimize or remove underperforming stocks from portfolio
**Results**: - 2-4 stocks successfully re-optimized (70%+ win rate)
- 0-2 stocks removed from portfolio
- +2-3pp portfolio avg return improvement
- Portfolio quality > quantity

#### exp049_exit_threshold_optimization.py
**Objective**: Optimize exit thresholds per stock based on volatility and reversion speed
**Results**: - Stock-specific exit thresholds
- +3-5% total return improvement
- Improved Sharpe ratio
- Documented exit rules for each stock

#### exp050_tier_a_expansion_v2.py
**Objective**: Expand Tier A universe with 5-10 additional high-quality stocks
**Results**: - 5-10 new Tier A stocks
- All with 70%+ win rate
- +35-100 trades/year
- +10-30% portfolio return improvement
- Documented parameters for each

#### exp051_portfolio_risk_management.py
**Objective**: Implement portfolio-level risk management for 45-stock universe
**Results**: - Portfolio risk management framework
- Optimal concurrent position limit (likely 5-7)
- Sector exposure limits
- +20-30% Sharpe improvement
- -25% max drawdown reduction
- More consistent returns

#### exp052_adaptive_regime_detection.py
**Objective**: Implement adaptive strategy based on market volatility and breadth conditions
**Results**: - Market condition filters
- Adaptive position sizing rules
- +10-15% Sharpe improvement
- +3-5pp win rate improvement
- Better risk-adjusted returns

#### exp053_portfolio_position_limits.py
**Objective**: Implement and validate portfolio-level position limits
**Results**: - Portfolio position management system
- Optimal position limit (likely 5 from EXP-051)
- Sector exposure controls
- +15-20% Sharpe improvement
- -20% max drawdown reduction
- Same or better win rate (taking best signals)

#### exp054_stop_loss_optimization.py
**Objective**: Optimize stop-loss levels per stock based on volatility
**Results**: - Per-stock stop-loss parameters
- Volatility-based stop framework
- +5-10% Sharpe improvement
- Better risk management
- Fewer false stops on volatile stocks
- Smaller losses on stable stocks

#### exp055_holding_period_optimization.py
**Objective**: Optimize holding period per stock based on reversion speed
**Results**: - Per-stock optimal holding periods
- Reversion speed classification
- +5-15% return improvement
- Higher capital efficiency
- Better profit capture

#### exp056_production_scanner.py
**Objective**: Deploy production-ready daily signal scanner with automated reporting
**Results**: - Automated daily signal detection
- Email alerts within minutes of market close
- Historical performance tracking
- Production-ready trading system
- Seamless transition from backtest to live trading

#### exp057_multitimeframe_confirmation.py
**Objective**: Test multi-timeframe signal confirmation for higher quality signals
**Results**: - Multi-timeframe confirmation framework
- Higher win rate (+3-5pp)
- Fewer but stronger signals
- Better risk-adjusted returns

#### exp058_seasonality_analysis.py
**Objective**: Analyze seasonality patterns in mean reversion performance
**Results**: - Calendar pattern analysis
- Seasonality filters (if beneficial)
- +5-10pp win rate improvement
- Better timing of signal execution

#### exp059_market_breadth_filter.py
**Objective**: Filter signals based on market breadth quality
**Results**: - Market breadth filter implementation
- +5-10pp win rate improvement
- Higher quality signals
- Better risk-adjusted returns

#### exp060_gap_analysis.py
**Objective**: Test if overnight gaps enhance mean reversion signal quality
**Results**: - Either: Gap signals show superior performance (+5-10pp)
- Or: Final confirmation that current system is at theoretical max
- This is likely the LAST unexplored optimization avenue

#### exp061_gap_validation.py
**Objective**: VALIDATE gap enhancement discovery across full portfolio
**Results**: - Confirmation of gap enhancement (+5-10pp win rate)
- Deployment to v16.0-GAP-ENHANCED
- OR: Discovery that small sample was anomaly

#### exp062_sector_relative_strength.py
**Objective**: Enhance signal quality using sector relative strength
**Results**: - Confirmation of sector-relative strength filter (+3-5pp win rate)
- Deployment to v16.0-SECTOR-ENHANCED
- OR: Discovery that relative strength doesn't matter for mean reversion

#### exp062_sentiment_data_collection.py
**Objective**: Build and validate sentiment data collection pipeline
**Results**: If successful: Proceed to EXP-063 (FinBERT sentiment model)

#### exp063_finbert_sentiment_model.py
**Objective**: Integrate FinBERT sentiment model and validate accuracy
**Results**: If successful: Proceed to EXP-064 (sentiment-enhanced backtesting)

#### exp063_optimize_unoptimized_stocks.py
**Objective**: Optimize parameters for unoptimized Tier A stocks
**Results**: )
- Focus on stocks with highest trading frequency first (max impact)
- Deploy incrementally to validate improvements

#### exp064_sector_diversification_screening.py
**Objective**: Screen underrepresented sectors for Tier A stock candidates
**Results**: - 2-4 new Tier A stocks from defensive sectors
- Better sector diversification (reduce concentration risk)
- More trading opportunities in different market cycles
- Validation that mean reversion works across ALL stable sectors

#### exp066_ml_feature_engineering.py
**Objective**: Build ML feature engineering pipeline for stock selection
**Results**: If successful: Proceed to EXP-067 (XGBoost model training)

#### exp067_xgboost_stock_prediction.py
**Objective**: Train XGBoost model to predict profitable mean reversion signals

#### exp068_ml_enhanced_scanner.py
**Objective**: Integrate XGBoost ML predictions into production scanner

#### exp069_simplified_xgboost.py
**Objective**: Retrain XGBoost with 41 production-compatible features (no SPY dependencies)

#### exp070_ml_portfolio_expansion.py
**Objective**: Use ML to screen and expand portfolio from 48 to 60+ stocks

---

### Phase 5: ML Model Development (exp071-091)

#### exp071_batch_validate_ml_candidates.py
**Objective**: Batch validate 30 ML-selected candidates for immediate portfolio expansion

#### exp072_optimize_new_stocks.py
**Objective**: Optimize parameters for 6 newly deployed ML-validated stocks
**Results**: (+22.5pp for SYK, +26.7pp for MA).

#### exp073_gap_trading_validation.py
**Objective**: Validate gap trading strategy as complement to mean reversion
**Results**: - Win rate: >85% (vs 78.5% for regular MR)
- Avg return: +3-5% per trade (vs +2% for regular MR)
- Trade frequency: +50-100 trades/year (gaps are less frequent)
- Risk: Larger drawdowns possible (gaps can continue)

#### exp074_exit_strategy_optimization.py
**Objective**: Optimize exit strategy for maximum returns and win rate
**Results**: - Best strategy: ATR-based or Hybrid (accounts for volatility)
- Avg return improvement: +0.5-1.0% per trade
- Win rate improvement: +2-4pp
- Annual impact: +140-280% cumulative return boost

#### exp075_trailing_stop_validation.py
**Objective**: Validate trailing stop strategy on full 54-stock portfolio
**Results**: .

#### exp076_volatility_position_sizing.py
**Objective**: Implement volatility-adjusted position sizing for better risk-adjusted returns
**Results**: - Sharpe ratio improvement: +20-30%
- Max drawdown reduction: -15-25%
- Total return: Similar or slightly better (+5-10%)
- Risk-adjusted returns: Significantly better

#### exp077_adaptive_exit_strategy.py
**Objective**: Implement adaptive exit strategy based on stock volatility
**Results**: - Win rate improvement: +3-5pp over time_decay baseline
- Avg return improvement: +0.30-0.50% per trade
- Sharpe ratio improvement: +15-25%
- Universal benefit: 80-90% of stocks improve

#### exp078_intraday_entry_timing.py
**Objective**: Optimize intraday entry timing for mean reversion signals
**Results**: - Open entry: +0.3-0.5% better for volatile stocks (catch panic lows)
- Close entry: +0.1-0.2% better for stable stocks (avoid intraday noise)
- Optimal entry improvement: +0.2-0.4% average across all trades

#### exp079_deep_learning_signals.py
**Objective**: Deep Learning Signal Prediction using LSTM/GRU
**Results**: (win/loss) + actual return
2. Build LSTM/GRU architecture:
   - Input: (batch, 20 days, 41 features)
   - LSTM layers with dropout
   - Multi-task head: classification + regression
3. Train on GPU with early stopping
4. Validate against XGBoost baseline (0.834 AUC)
5. Measure improvement in producti

#### exp080_limit_order_entry.py
**Objective**: Implement limit order entry for better intraday pricing
**Results**: - 0.5% discount: 70-80% fill rate, +0.3-0.5% return improvement
- Acts as volatility filter (only fills on true panic)
- Win rate improvement: +1-3pp (filtered signals)
- Annual impact: +84-140% cumulative (281 trades × 0.3-0.5%)

#### exp081_signal_quality_scoring.py
**Objective**: Implement multi-factor signal quality scoring system
**Results**: - Baseline: 67.3% WR, 281 trades/year, 3.60% avg return
- Quality >= 70: 75-80% WR, 140-180 trades/year, 4.0-4.5% avg return
- Quality >= 90: 80-85% WR, 30-50 trades/year, 4.5-5.0% avg return

#### exp082_microstructure_features_phase1.py
**Objective**: Validate market microstructure features for ML enhancement

#### exp083_sentiment_features_phase3.py

#### exp084_cross_sectional_features_phase2.py

#### exp085_temporal_features_phase4.py

#### exp086_deploy_temporal_features.py
**Results**: (EXP-085):
- Average AUC improvement: +2.0pp (0.694 → 0.715)
- Success rate: 90% stocks improved (9/10)
- All validation criteria PASSED

#### exp087_production_model_retraining.py
**Objective**: Deploy validated temporal features to production ML scanner

#### exp088_ensemble_model.py
**Objective**: Improve AUC by combining multiple ML algorithms

#### exp089_hyperparameter_optimization.py
**Objective**: Optimize hyperparameters for better generalization

#### exp090_manual_hyperparameter_tuning.py

#### exp091_base_scanner_validation.py
**Results**: Based on mean reversion strategy design (z-score, RSI, volume filters),

---

### Phase 6: Production Tuning (exp092+)

#### exp092_signal_quality_tuning.py
**Objective**: Tune base scanner filters to generate MORE Q4-quality signals
**Results**: Even modest improvements could be huge:
- 70% win rate (from 63.7%) = +10% annual return boost
- More reliable signals = lower drawdowns, higher Sharpe

#### exp093_q4_only_filter.py
**Objective**: Validate Q4-only filtering improves production performance.

#### exp094_portfolio_correlation_limits.py
**Objective**: Validate if limiting portfolio correlation improves risk-adjusted returns.
**Results**: Higher Sharpe ratio (+10-20% improvement expected)

#### exp095_multi_factor_regime.py
**Objective**: Enhance regime detection with multi-factor system for better trading timing.

#### exp109_intraday_price_position_filter.py
**Objective**: Improve signal quality by filtering out stocks with intraday recovery

#### exp110_vix_position_sizing.py
**Objective**: Improve risk-adjusted returns via VIX-based position sizing

#### exp111_dynamic_exit_thresholds.py
**Objective**: Improve profit factor and risk-adjusted returns via volatility-adaptive exits

#### exp112_signal_quality_exit_thresholds.py
**Objective**: Improve profit factor by giving elite signals room to run, quick exits on weak signals

#### exp113_adaptive_max_hold_days.py
**Objective**: Optimize holding periods based on signal quality + volatility

#### exp114_gap_fade_strategy.py
**Objective**: Capture intraday bounces from overnight gap-down overreactions

#### exp115_entry_timing_optimization.py
**Objective**: Improve entry price by optimizing trade execution timing

#### exp116_partial_profit_taking.py
**Objective**: Improve win rate and consistency via partial profit taking

#### exp117_multi_day_decline_filter.py
**Objective**: Improve signal quality by requiring sustained multi-day declines

#### exp118_market_regime_adaptive.py
**Objective**: Adapt trading strategy based on market regime to improve risk-adjusted returns.

#### exp119_volume_surge_filter.py
**Objective**: Filter signals based on volume to improve signal quality and win rate.

#### exp120_fresh_signals_filter.py
**Objective**: Filter out stale signals to improve win rate and risk-adjusted returns.

#### exp121_consecutive_loss_circuit_breaker.py
**Objective**: Implement circuit breaker to reduce max drawdown and improve risk-adjusted returns.

#### exp122_position_concentration_limits.py
**Objective**: Test different position concentration limits to optimize risk-adjusted returns.

#### exp123_rate_limit_fix.py

#### exp124_signal_quality_position_sizing.py

#### exp125_adaptive_stop_loss.py
**Objective**: Reduce premature stop-outs while maintaining risk control.

#### exp126_adaptive_profit_targets.py
**Objective**: Maximize total returns while maintaining or improving win rate.

#### exp127_adaptive_max_hold_days.py
**Objective**: Maximize risk-adjusted returns via signal-quality-based hold periods.

#### exp128_vix_entry_filter.py
**Objective**: Maximize risk-adjusted returns via volatility regime filtering.

#### exp129_signal_ensemble.py

#### exp130_news_sentiment_integration.py
**Objective**: Test news sentiment integration via multiple providers with ensemble system.

#### exp131_multi_timeframe_convergence.py

#### exp132_short_interest_squeeze.py

---

## Key Findings

### What Works
- LSTM V2 model: 79% win rate, 4.96 Sharpe
- Tier-based exits: Elite stocks get longer hold periods
- Regime-adaptive thresholds: Higher bar in choppy markets
- Earnings avoidance: Skip 3 days before earnings
- Sector diversification: Max 2 per sector

### What Doesn't Work
- Simple MA crossover: Only 52% win rate
- Fixed position sizing: ATR-based is better
- Single model: Ensemble outperforms
- Ignoring regime: Choppy markets are traps

### Open Questions
- Intraday entry optimization
- Options integration
- Small cap expansion

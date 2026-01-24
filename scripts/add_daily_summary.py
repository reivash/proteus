"""Add comprehensive daily summary and market context to fast_bear_detector.py"""

DAILY_SUMMARY_METHODS = '''

    # ==================== DAILY SUMMARY & MARKET CONTEXT ====================

    def get_daily_summary(self) -> Dict:
        """
        Generate comprehensive daily summary of all bear detection signals.

        This is the primary method for daily operational use, combining
        all analysis methods into a single actionable summary.

        Returns:
            Dict with complete daily summary
        """
        # Gather all analyses
        signal = self.detect()
        ultimate = self.get_ultimate_warning()
        effectiveness = self.get_indicator_effectiveness()
        scenario = self.get_scenario_analysis()
        attribution = self.get_risk_attribution()
        pattern = self.match_historical_patterns()
        correlation = self.get_cross_asset_correlation()
        momentum = self.get_momentum_regime()
        sector = self.get_sector_leadership()
        multiframe = self.get_multiframe_analysis()

        # Build summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),

            # Primary metrics
            'bear_score': signal.bear_score,
            'alert_level': signal.alert_level,
            'ultimate_score': ultimate['ultimate_score'],
            'crash_probability': signal.crash_probability,
            'early_warning_score': signal.early_warning_score,

            # Warning assessment
            'warning_level': ultimate['warning_level'],
            'urgency': ultimate['urgency'],
            'recommended_action': ultimate['recommended_action'],

            # Key indicators
            'vix_level': signal.vix_level,
            'vix_term_structure': signal.vix_term_structure,
            'market_breadth': signal.market_breadth_pct,
            'spy_3d_change': signal.spy_roc_3d,
            'credit_spread_change': signal.credit_spread_change,
            'vol_compression': signal.vol_compression,
            'vol_regime': signal.vol_regime,

            # Analysis results
            'warning_phase': effectiveness['phase'],
            'pattern_match': pattern['status'],
            'pattern_similarity': pattern['best_score'],
            'cross_asset_status': correlation['status'],
            'momentum_regime': momentum['regime'],
            'sector_leadership': sector['leadership'],
            'timeframe_confluence': multiframe['confluence_direction'],

            # Risk breakdown
            'primary_risk_source': attribution['primary_risk_source'],
            'active_flags': ultimate['active_flags'],
            'flag_count': ultimate['flag_count'],

            # Stress test
            'worst_case_score': scenario['worst_case_score'],
            'most_vulnerable_to': scenario['most_vulnerable_to'],

            # Trend
            'trend_direction': self.get_signal_trend().get('direction', 'UNKNOWN'),

            # Actionable items
            'watch_list': [],
            'action_items': []
        }

        # Build watch list and action items
        if signal.vol_compression >= 0.8:
            summary['watch_list'].append('Vol compression elevated - crash risk')
        if signal.vix_term_structure >= 1.1:
            summary['watch_list'].append('VIX backwardation - market stress')
        if signal.market_breadth_pct <= 35:
            summary['watch_list'].append('Poor market breadth')
        if signal.credit_spread_change >= 8:
            summary['watch_list'].append('Credit spreads widening')
        if correlation.get('is_risk_off'):
            summary['watch_list'].append('Cross-asset risk-off rotation')
        if pattern.get('is_concerning'):
            summary['watch_list'].append(f"Pattern match: {pattern.get('best_match')}")

        # Action items based on warning level
        if ultimate['warning_level'] == 'CRITICAL':
            summary['action_items'].append('REDUCE EXPOSURE immediately')
            summary['action_items'].append('Review all positions for risk')
            summary['action_items'].append('Consider hedges (puts, VIX calls)')
        elif ultimate['warning_level'] == 'WARNING':
            summary['action_items'].append('Consider reducing position sizes')
            summary['action_items'].append('Tighten stop losses')
            summary['action_items'].append('Monitor closely for escalation')
        elif ultimate['warning_level'] == 'WATCH':
            summary['action_items'].append('Monitor key indicators')
            summary['action_items'].append('Prepare contingency plans')
        else:
            summary['action_items'].append('Continue normal operations')
            summary['action_items'].append('Monitor vol compression')

        return summary

    def get_daily_report(self) -> str:
        """
        Generate formatted daily summary report.

        Returns:
            Multi-line string with complete daily report
        """
        summary = self.get_daily_summary()

        nl = chr(10)
        lines = []

        # Header
        lines.append('*' * 70)
        lines.append(f"*  BEAR DETECTION DAILY SUMMARY - {summary['date']}")
        lines.append('*' * 70)
        lines.append('')

        # Main assessment
        level_visuals = {
            'CRITICAL': '!!! CRITICAL - TAKE ACTION !!!',
            'WARNING': '!! WARNING - ELEVATED RISK !!',
            'WATCH': '! WATCH - MONITORING !',
            'NORMAL': 'NORMAL - NO IMMEDIATE CONCERN'
        }
        visual = level_visuals.get(summary['warning_level'], 'UNKNOWN')

        lines.append(f"STATUS: [{visual}]")
        lines.append('')

        # Score dashboard
        lines.append('+' + '-' * 68 + '+')
        lines.append(f"| Bear Score: {summary['bear_score']:5.1f}/100  |  Ultimate Score: {summary['ultimate_score']:5.1f}/100  |  Crash Prob: {summary['crash_probability']:4.1f}% |")
        lines.append('+' + '-' * 68 + '+')
        lines.append('')

        # Key metrics
        lines.append('KEY METRICS:')
        lines.append(f"  VIX: {summary['vix_level']:.1f}  |  Breadth: {summary['market_breadth']:.1f}%  |  SPY 3d: {summary['spy_3d_change']:+.2f}%")
        lines.append(f"  Vol Regime: {summary['vol_regime']}  |  Vol Compression: {summary['vol_compression']:.2f}")
        lines.append(f"  Credit Spread Chg: {summary['credit_spread_change']:+.1f}  |  VIX Term: {summary['vix_term_structure']:.2f}")
        lines.append('')

        # Analysis summary
        lines.append('ANALYSIS SUMMARY:')
        lines.append(f"  Warning Phase: {summary['warning_phase']}")
        lines.append(f"  Pattern Match: {summary['pattern_match']} ({summary['pattern_similarity']:.0f}%)")
        lines.append(f"  Cross-Asset: {summary['cross_asset_status']}")
        lines.append(f"  Momentum: {summary['momentum_regime']}")
        lines.append(f"  Sector Leadership: {summary['sector_leadership']}")
        lines.append(f"  Timeframe: {summary['timeframe_confluence']}")
        lines.append(f"  Trend: {summary['trend_direction']}")
        lines.append('')

        # Risk attribution
        lines.append(f"PRIMARY RISK SOURCE: {summary['primary_risk_source'].upper().replace('_', ' ')}")
        lines.append('')

        # Active flags
        if summary['active_flags']:
            lines.append(f"WARNING FLAGS ({summary['flag_count']}):")
            for flag in summary['active_flags']:
                lines.append(f"  >>> {flag}")
            lines.append('')

        # Watch list
        if summary['watch_list']:
            lines.append('WATCH LIST:')
            for item in summary['watch_list']:
                lines.append(f"  [!] {item}")
            lines.append('')

        # Action items
        lines.append('RECOMMENDED ACTIONS:')
        for item in summary['action_items']:
            lines.append(f"  -> {item}")
        lines.append('')

        # Stress test summary
        lines.append(f"STRESS TEST: Worst case score {summary['worst_case_score']:.0f} if {summary['most_vulnerable_to']}")
        lines.append('')

        lines.append('*' * 70)

        return nl.join(lines)

    def get_market_context(self) -> Dict:
        """
        Analyze current market context and environment.

        Provides broader context about market conditions beyond
        just bear signals.

        Returns:
            Dict with market context analysis
        """
        signal = self.detect()

        try:
            # Fetch additional context data
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="1y")

            if len(spy_hist) < 50:
                return {
                    'context': 'UNKNOWN',
                    'description': 'Insufficient historical data'
                }

            close = spy_hist['Close']

            # Calculate market position metrics
            current_price = close.iloc[-1]
            high_52w = close.max()
            low_52w = close.min()
            ma_50 = close.iloc[-50:].mean()
            ma_200 = close.iloc[-200:].mean() if len(close) >= 200 else close.mean()

            # Position metrics
            pct_from_high = ((current_price / high_52w) - 1) * 100
            pct_from_low = ((current_price / low_52w) - 1) * 100
            pct_above_50ma = ((current_price / ma_50) - 1) * 100
            pct_above_200ma = ((current_price / ma_200) - 1) * 100

            # Determine market phase
            if pct_from_high >= -5 and pct_above_50ma > 0 and pct_above_200ma > 0:
                phase = 'BULL_TREND'
                phase_desc = 'Market in uptrend near highs'
            elif pct_from_high < -5 and pct_from_high >= -10 and pct_above_200ma > 0:
                phase = 'PULLBACK'
                phase_desc = 'Normal pullback in uptrend'
            elif pct_from_high < -10 and pct_from_high >= -20 and pct_above_200ma > 0:
                phase = 'CORRECTION'
                phase_desc = 'Correction underway but above 200MA'
            elif pct_from_high < -10 and pct_above_200ma < 0:
                phase = 'BEAR_MARKET'
                phase_desc = 'Below 200MA - potential bear market'
            elif pct_from_low > 20 and pct_above_200ma < 0:
                phase = 'BEAR_RALLY'
                phase_desc = 'Rally within bear market'
            else:
                phase = 'TRANSITION'
                phase_desc = 'Market in transition'

            # Trend strength
            if ma_50 > ma_200:
                trend = 'BULLISH'
                trend_desc = '50MA above 200MA - bullish structure'
            else:
                trend = 'BEARISH'
                trend_desc = '50MA below 200MA - bearish structure'

            # Volatility context
            vol_context = signal.vol_regime
            if signal.vol_regime == 'LOW_COMPLACENT':
                vol_desc = 'Low volatility - complacency risk'
            elif signal.vol_regime == 'ELEVATED':
                vol_desc = 'Elevated volatility - caution advised'
            elif signal.vol_regime == 'CRISIS':
                vol_desc = 'Crisis-level volatility'
            else:
                vol_desc = 'Normal volatility environment'

            # Historical comparison
            ytd_return = ((current_price / close.iloc[0]) - 1) * 100 if len(close) > 0 else 0

            return {
                'market_phase': phase,
                'phase_description': phase_desc,
                'trend_structure': trend,
                'trend_description': trend_desc,
                'volatility_context': vol_context,
                'volatility_description': vol_desc,
                'current_price': round(current_price, 2),
                'high_52w': round(high_52w, 2),
                'low_52w': round(low_52w, 2),
                'pct_from_high': round(pct_from_high, 2),
                'pct_from_low': round(pct_from_low, 2),
                'pct_above_50ma': round(pct_above_50ma, 2),
                'pct_above_200ma': round(pct_above_200ma, 2),
                'ytd_return': round(ytd_return, 2),
                'ma_50': round(ma_50, 2),
                'ma_200': round(ma_200, 2),
                'in_bull_market': phase in ['BULL_TREND', 'PULLBACK'],
                'in_bear_market': phase in ['BEAR_MARKET', 'BEAR_RALLY']
            }

        except Exception as e:
            return {
                'market_phase': 'ERROR',
                'description': f'Error: {str(e)}'
            }

    def get_context_report(self) -> str:
        """
        Generate formatted market context report.

        Returns:
            Multi-line string with market context
        """
        ctx = self.get_market_context()

        nl = chr(10)
        lines = []
        lines.append('=' * 60)
        lines.append('MARKET CONTEXT ANALYSIS')
        lines.append('=' * 60)
        lines.append('')

        # Market phase
        phase_icons = {
            'BULL_TREND': '[^] ',
            'PULLBACK': '[-] ',
            'CORRECTION': '[!] ',
            'BEAR_MARKET': '[v] ',
            'BEAR_RALLY': '[~] ',
            'TRANSITION': '[?] '
        }
        icon = phase_icons.get(ctx.get('market_phase', ''), '[?] ')

        lines.append(f"Market Phase: {icon}{ctx.get('market_phase', 'UNKNOWN')}")
        lines.append(f"  {ctx.get('phase_description', '')}")
        lines.append('')

        lines.append(f"Trend Structure: {ctx.get('trend_structure', 'UNKNOWN')}")
        lines.append(f"  {ctx.get('trend_description', '')}")
        lines.append('')

        lines.append(f"Volatility: {ctx.get('volatility_context', 'UNKNOWN')}")
        lines.append(f"  {ctx.get('volatility_description', '')}")
        lines.append('')

        # Price metrics
        if 'current_price' in ctx:
            lines.append('Price Metrics:')
            lines.append(f"  Current: ${ctx['current_price']:.2f}")
            lines.append(f"  52-Week High: ${ctx['high_52w']:.2f} ({ctx['pct_from_high']:+.1f}%)")
            lines.append(f"  52-Week Low: ${ctx['low_52w']:.2f} ({ctx['pct_from_low']:+.1f}%)")
            lines.append(f"  50-Day MA: ${ctx['ma_50']:.2f} ({ctx['pct_above_50ma']:+.1f}%)")
            lines.append(f"  200-Day MA: ${ctx['ma_200']:.2f} ({ctx['pct_above_200ma']:+.1f}%)")
            lines.append(f"  YTD Return: {ctx['ytd_return']:+.1f}%")

        return nl.join(lines)

    def get_api_output(self) -> Dict:
        """
        Get API-friendly output with all key metrics.

        Designed for easy integration with monitoring systems,
        dashboards, and alerting infrastructure.

        Returns:
            Dict with structured API output
        """
        signal = self.detect()
        summary = self.get_daily_summary()
        context = self.get_market_context()

        return {
            'version': '2.0',
            'timestamp': datetime.now().isoformat(),

            # Primary signals
            'signals': {
                'bear_score': signal.bear_score,
                'alert_level': signal.alert_level,
                'crash_probability': signal.crash_probability,
                'early_warning_score': signal.early_warning_score,
                'ultimate_score': summary['ultimate_score']
            },

            # Alert status
            'alert': {
                'level': summary['warning_level'],
                'urgency': summary['urgency'],
                'action': summary['recommended_action'],
                'flag_count': summary['flag_count'],
                'flags': summary['active_flags']
            },

            # Market state
            'market': {
                'phase': context.get('market_phase'),
                'trend': context.get('trend_structure'),
                'vol_regime': signal.vol_regime,
                'vix': signal.vix_level,
                'breadth': signal.market_breadth_pct,
                'spy_3d': signal.spy_roc_3d
            },

            # Analysis
            'analysis': {
                'warning_phase': summary['warning_phase'],
                'pattern_status': summary['pattern_match'],
                'pattern_score': summary['pattern_similarity'],
                'momentum': summary['momentum_regime'],
                'sector': summary['sector_leadership'],
                'cross_asset': summary['cross_asset_status'],
                'primary_risk': summary['primary_risk_source']
            },

            # Thresholds for alerting
            'thresholds': {
                'is_normal': signal.alert_level == 'NORMAL',
                'is_watch': signal.alert_level == 'WATCH',
                'is_warning': signal.alert_level == 'WARNING',
                'is_critical': signal.alert_level == 'CRITICAL',
                'requires_action': summary['warning_level'] in ['WARNING', 'CRITICAL']
            }
        }

    def get_json_output(self) -> str:
        """
        Get JSON-formatted output for API consumption.

        Returns:
            JSON string with all key metrics
        """
        return json.dumps(self.get_api_output(), indent=2, default=str)

    def should_send_alert(self, min_level: str = 'WATCH') -> bool:
        """
        Determine if an alert should be sent based on current conditions.

        Args:
            min_level: Minimum alert level to trigger ('WATCH', 'WARNING', 'CRITICAL')

        Returns:
            bool: True if alert should be sent
        """
        signal = self.detect()

        level_order = {'NORMAL': 0, 'WATCH': 1, 'WARNING': 2, 'CRITICAL': 3}
        current_level = level_order.get(signal.alert_level, 0)
        min_required = level_order.get(min_level, 1)

        return current_level >= min_required

    def get_alert_message(self) -> str:
        """
        Generate concise alert message for notifications.

        Returns:
            Short alert string suitable for SMS/email/Slack
        """
        signal = self.detect()
        summary = self.get_daily_summary()

        level_emoji = {
            'CRITICAL': '🚨',
            'WARNING': '⚠️',
            'WATCH': '👀',
            'NORMAL': '✅'
        }
        emoji = level_emoji.get(signal.alert_level, '❓')

        msg = f"{emoji} BEAR ALERT: {signal.alert_level}"
        msg += f" | Score: {signal.bear_score:.0f}/100"
        msg += f" | Crash Prob: {signal.crash_probability:.1f}%"

        if summary['active_flags']:
            msg += f" | Flags: {', '.join(summary['active_flags'][:2])}"

        msg += f" | Action: {summary['recommended_action']}"

        return msg

'''

def main():
    with open('src/analysis/fast_bear_detector.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Find insertion point - before the standalone functions
    insertion_point = 'def get_fast_bear_signal() -> FastBearSignal:'

    if insertion_point in content:
        content = content.replace(insertion_point, DAILY_SUMMARY_METHODS + '\n' + insertion_point)
        print("Added daily summary and market context methods")
    else:
        print("Could not find insertion point")
        return

    with open('src/analysis/fast_bear_detector.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Done")

if __name__ == '__main__':
    main()

"""Add multi-timeframe analysis methods to fast_bear_detector.py"""

MULTIFRAME_METHODS = '''

    # ==================== MULTI-TIMEFRAME ANALYSIS ====================

    def get_multiframe_analysis(self) -> Dict:
        """
        Analyze bear signals across multiple timeframes.

        Looks at short-term (1-3 day), medium-term (5-10 day),
        and long-term (20+ day) signals for a comprehensive view.

        Returns:
            Dict with timeframe analysis and confluence score
        """
        try:
            # Fetch data for different timeframes
            spy = yf.Ticker("SPY")
            hist = spy.history(period="60d")

            if hist.empty or len(hist) < 20:
                return {
                    'short_term': {'trend': 'UNKNOWN', 'score': 0},
                    'medium_term': {'trend': 'UNKNOWN', 'score': 0},
                    'long_term': {'trend': 'UNKNOWN', 'score': 0},
                    'confluence_score': 0,
                    'confluence_direction': 'UNKNOWN',
                    'description': 'Insufficient data'
                }

            close = hist['Close']

            # Short-term analysis (1-3 days)
            short_roc = ((close.iloc[-1] / close.iloc[-3]) - 1) * 100
            short_vol_ratio = hist['Volume'].iloc[-3:].mean() / hist['Volume'].iloc[-20:-3].mean()

            short_score = 0
            if short_roc < -1: short_score += 20
            if short_roc < -2: short_score += 20
            if short_roc < -3: short_score += 20
            if short_vol_ratio > 1.3: short_score += 20  # High volume on decline
            if short_vol_ratio > 1.5: short_score += 20

            short_trend = 'BULLISH' if short_roc > 1 else 'BEARISH' if short_roc < -1 else 'NEUTRAL'

            # Medium-term analysis (5-10 days)
            med_roc = ((close.iloc[-1] / close.iloc[-10]) - 1) * 100
            med_ma5 = close.iloc[-5:].mean()
            med_ma10 = close.iloc[-10:].mean()
            med_cross = med_ma5 < med_ma10  # Bearish cross

            med_score = 0
            if med_roc < -2: med_score += 20
            if med_roc < -4: med_score += 20
            if med_roc < -6: med_score += 20
            if med_cross: med_score += 20
            if close.iloc[-1] < med_ma10: med_score += 20

            med_trend = 'BULLISH' if med_roc > 2 else 'BEARISH' if med_roc < -2 else 'NEUTRAL'

            # Long-term analysis (20+ days)
            long_roc = ((close.iloc[-1] / close.iloc[-20]) - 1) * 100
            long_ma20 = close.iloc[-20:].mean()
            long_ma50 = close.iloc[-50:].mean() if len(close) >= 50 else long_ma20

            # Distance from 20d MA
            dist_from_ma20 = ((close.iloc[-1] / long_ma20) - 1) * 100

            long_score = 0
            if long_roc < -3: long_score += 20
            if long_roc < -6: long_score += 20
            if long_roc < -10: long_score += 20
            if close.iloc[-1] < long_ma20: long_score += 20
            if long_ma20 < long_ma50: long_score += 20  # Death cross setup

            long_trend = 'BULLISH' if long_roc > 3 else 'BEARISH' if long_roc < -3 else 'NEUTRAL'

            # Confluence scoring - when multiple timeframes align
            bearish_count = sum([
                short_trend == 'BEARISH',
                med_trend == 'BEARISH',
                long_trend == 'BEARISH'
            ])

            bullish_count = sum([
                short_trend == 'BULLISH',
                med_trend == 'BULLISH',
                long_trend == 'BULLISH'
            ])

            # Weighted confluence score (short-term weighted higher for early warning)
            confluence_score = (short_score * 0.5 + med_score * 0.3 + long_score * 0.2)

            if bearish_count == 3:
                confluence_direction = 'STRONGLY_BEARISH'
                desc = 'All timeframes bearish - high confidence warning'
            elif bearish_count == 2:
                confluence_direction = 'BEARISH'
                desc = 'Multiple timeframes bearish - elevated risk'
            elif bullish_count == 3:
                confluence_direction = 'STRONGLY_BULLISH'
                desc = 'All timeframes bullish - low risk'
            elif bullish_count == 2:
                confluence_direction = 'BULLISH'
                desc = 'Multiple timeframes bullish - favorable'
            else:
                confluence_direction = 'MIXED'
                desc = 'Timeframes diverging - uncertain'

            return {
                'short_term': {
                    'trend': short_trend,
                    'score': short_score,
                    'roc_3d': round(short_roc, 2),
                    'vol_ratio': round(short_vol_ratio, 2)
                },
                'medium_term': {
                    'trend': med_trend,
                    'score': med_score,
                    'roc_10d': round(med_roc, 2),
                    'below_ma10': med_cross
                },
                'long_term': {
                    'trend': long_trend,
                    'score': long_score,
                    'roc_20d': round(long_roc, 2),
                    'dist_from_ma20': round(dist_from_ma20, 2)
                },
                'confluence_score': round(confluence_score, 1),
                'confluence_direction': confluence_direction,
                'bearish_timeframes': bearish_count,
                'description': desc
            }

        except Exception as e:
            return {
                'short_term': {'trend': 'ERROR', 'score': 0},
                'medium_term': {'trend': 'ERROR', 'score': 0},
                'long_term': {'trend': 'ERROR', 'score': 0},
                'confluence_score': 0,
                'confluence_direction': 'ERROR',
                'description': f'Error: {str(e)}'
            }

    def get_multiframe_report(self) -> str:
        """
        Get formatted multi-timeframe analysis report.

        Returns:
            Multi-line string with timeframe analysis
        """
        analysis = self.get_multiframe_analysis()

        nl = chr(10)
        lines = []
        lines.append('=' * 60)
        lines.append('MULTI-TIMEFRAME ANALYSIS')
        lines.append('=' * 60)
        lines.append('')

        # Direction indicator
        direction_icons = {
            'STRONGLY_BEARISH': '[!!!] ',
            'BEARISH': '[!!] ',
            'MIXED': '[?] ',
            'BULLISH': '[+] ',
            'STRONGLY_BULLISH': '[++] '
        }
        icon = direction_icons.get(analysis['confluence_direction'], '[?] ')

        lines.append(f"Confluence: {icon}{analysis['confluence_direction']}")
        lines.append(f"Score: {analysis['confluence_score']}/100")
        lines.append(f"Description: {analysis['description']}")
        lines.append('')

        # Timeframe details
        for tf_name, tf_key in [('Short-Term (1-3d)', 'short_term'),
                                 ('Medium-Term (5-10d)', 'medium_term'),
                                 ('Long-Term (20d)', 'long_term')]:
            tf = analysis[tf_key]
            trend_icon = '[v]' if tf['trend'] == 'BEARISH' else '[^]' if tf['trend'] == 'BULLISH' else '[-]'
            lines.append(f"{tf_name}: {trend_icon} {tf['trend']} (score: {tf['score']})")

        lines.append('')

        # Key metrics
        st = analysis['short_term']
        mt = analysis['medium_term']
        lt = analysis['long_term']

        if 'roc_3d' in st:
            lines.append('Key Metrics:')
            lines.append(f"  3-day ROC: {st.get('roc_3d', 'N/A')}%")
            lines.append(f"  10-day ROC: {mt.get('roc_10d', 'N/A')}%")
            lines.append(f"  20-day ROC: {lt.get('roc_20d', 'N/A')}%")
            lines.append(f"  Distance from 20d MA: {lt.get('dist_from_ma20', 'N/A')}%")

        return nl.join(lines)

    def get_signal_persistence(self) -> Dict:
        """
        Track how long warning signals have been active.

        Persistent warnings are more significant than brief spikes.

        Returns:
            Dict with signal persistence metrics
        """
        if len(self._signal_history) < 2:
            return {
                'watch_streak': 0,
                'warning_streak': 0,
                'critical_streak': 0,
                'elevated_duration': 0,
                'peak_score': 0,
                'avg_recent_score': 0,
                'persistence_warning': False,
                'description': 'Insufficient history'
            }

        # Count consecutive elevated signals
        watch_streak = 0
        warning_streak = 0
        critical_streak = 0

        # Go backwards through history
        for signal in reversed(self._signal_history):
            if signal.alert_level == 'CRITICAL':
                critical_streak += 1
                warning_streak += 1
                watch_streak += 1
            elif signal.alert_level == 'WARNING':
                if critical_streak == 0 or signal == self._signal_history[-1]:
                    warning_streak += 1
                watch_streak += 1
            elif signal.alert_level == 'WATCH':
                if warning_streak == 0 or signal == self._signal_history[-1]:
                    watch_streak += 1
            else:
                break  # Normal - streak broken

        # Calculate elevated duration (any non-NORMAL)
        elevated_duration = 0
        for signal in reversed(self._signal_history):
            if signal.alert_level != 'NORMAL':
                elevated_duration += 1
            else:
                break

        # Peak and average scores
        recent_scores = [s.bear_score for s in self._signal_history[-10:]]
        peak_score = max(recent_scores) if recent_scores else 0
        avg_recent_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0

        # Persistence warning - elevated for extended period
        persistence_warning = elevated_duration >= 3 or warning_streak >= 2

        # Description
        if critical_streak >= 2:
            desc = f'CRITICAL for {critical_streak} signals - high conviction warning'
        elif warning_streak >= 3:
            desc = f'WARNING for {warning_streak} signals - persistent risk'
        elif watch_streak >= 5:
            desc = f'WATCH for {watch_streak} signals - sustained concern'
        elif elevated_duration > 0:
            desc = f'Elevated for {elevated_duration} signals'
        else:
            desc = 'No persistent warnings'

        return {
            'watch_streak': watch_streak,
            'warning_streak': warning_streak,
            'critical_streak': critical_streak,
            'elevated_duration': elevated_duration,
            'peak_score': round(peak_score, 1),
            'avg_recent_score': round(avg_recent_score, 1),
            'persistence_warning': persistence_warning,
            'description': desc
        }

    def get_confluence_score(self) -> Dict:
        """
        Calculate correlation-based confluence score.

        Measures when multiple independent signals are firing together,
        which provides higher confidence warnings.

        Returns:
            Dict with confluence metrics
        """
        signal = self.detect()

        # Define independent signal groups
        price_signals = []
        volatility_signals = []
        breadth_signals = []
        credit_signals = []
        flow_signals = []

        # Price-based signals
        if signal.spy_roc_3d < -2: price_signals.append('SPY 3d drop')
        if signal.overnight_gap < -0.5: price_signals.append('Negative overnight gap')
        if signal.momentum_exhaustion > 0.3: price_signals.append('Momentum exhaustion')

        # Volatility signals
        if signal.vix_level > 25: volatility_signals.append('VIX elevated')
        if signal.vix_spike_pct > 20: volatility_signals.append('VIX spike')
        if signal.vix_term_structure > 1.05: volatility_signals.append('VIX backwardation')
        if signal.vol_compression > 0.7: volatility_signals.append('Vol compression')

        # Breadth signals
        if signal.market_breadth_pct < 40: breadth_signals.append('Poor market breadth')
        if signal.sectors_declining > 6: breadth_signals.append('Sector breadth weak')
        if signal.advance_decline_ratio < 0.7: breadth_signals.append('A/D ratio declining')
        if signal.mcclellan_proxy < -20: breadth_signals.append('McClellan negative')

        # Credit/risk signals
        if signal.credit_spread_change > 5: credit_signals.append('Credit spread widening')
        if signal.high_yield_spread > 3: credit_signals.append('High yield stress')
        if signal.liquidity_stress > 0.3: credit_signals.append('Liquidity stress')

        # Flow signals
        if signal.etf_flow_signal < -0.3: flow_signals.append('ETF outflows')
        if signal.options_volume_ratio > 1.3: flow_signals.append('Put volume elevated')
        if signal.smart_money_divergence < -0.3: flow_signals.append('Smart money selling')

        # Count signals by group
        groups_firing = sum([
            len(price_signals) > 0,
            len(volatility_signals) > 0,
            len(breadth_signals) > 0,
            len(credit_signals) > 0,
            len(flow_signals) > 0
        ])

        total_signals = (len(price_signals) + len(volatility_signals) +
                        len(breadth_signals) + len(credit_signals) + len(flow_signals))

        # Confluence score (0-100)
        # Higher when multiple independent signal types are firing
        confluence = (groups_firing / 5) * 60 + min(total_signals / 10, 1) * 40

        # Determine confidence level
        if groups_firing >= 4 and total_signals >= 6:
            confidence = 'VERY_HIGH'
            desc = 'Multiple signal groups aligned - high confidence warning'
        elif groups_firing >= 3 and total_signals >= 4:
            confidence = 'HIGH'
            desc = 'Strong signal alignment - elevated risk'
        elif groups_firing >= 2:
            confidence = 'MODERATE'
            desc = 'Some signal alignment - watch closely'
        else:
            confidence = 'LOW'
            desc = 'Limited signal confluence - isolated warnings'

        return {
            'confluence_score': round(confluence, 1),
            'confidence': confidence,
            'groups_firing': groups_firing,
            'total_signals': total_signals,
            'price_signals': price_signals,
            'volatility_signals': volatility_signals,
            'breadth_signals': breadth_signals,
            'credit_signals': credit_signals,
            'flow_signals': flow_signals,
            'description': desc
        }

    def get_comprehensive_warning(self) -> Dict:
        """
        Generate comprehensive warning combining all analysis methods.

        Combines: bear score, multiframe analysis, persistence, confluence

        Returns:
            Dict with comprehensive warning assessment
        """
        signal = self.detect()
        multiframe = self.get_multiframe_analysis()
        persistence = self.get_signal_persistence()
        confluence = self.get_confluence_score()
        trend = self.get_signal_trend()

        # Composite warning score (0-100)
        # Weight different components
        composite = (
            signal.bear_score * 0.30 +  # Base bear score
            signal.early_warning_score * 0.20 +  # Early warning
            multiframe['confluence_score'] * 0.15 +  # Timeframe confluence
            confluence['confluence_score'] * 0.20 +  # Signal confluence
            (persistence['elevated_duration'] * 5) * 0.15  # Persistence bonus
        )
        composite = min(composite, 100)

        # Warning level
        if composite >= 70:
            level = 'CRITICAL'
            action = 'REDUCE EXPOSURE IMMEDIATELY'
        elif composite >= 50:
            level = 'WARNING'
            action = 'Consider reducing positions'
        elif composite >= 30:
            level = 'WATCH'
            action = 'Monitor closely, tighten stops'
        else:
            level = 'NORMAL'
            action = 'No immediate action needed'

        # Build summary
        warnings = []
        if signal.bear_score >= 50: warnings.append(f'Bear score elevated ({signal.bear_score:.0f})')
        if multiframe['bearish_timeframes'] >= 2: warnings.append('Multiple timeframes bearish')
        if persistence['persistence_warning']: warnings.append('Persistent elevated signals')
        if confluence['groups_firing'] >= 3: warnings.append('Strong signal confluence')
        if trend['direction'] in ['DETERIORATING', 'DETERIORATING_FAST']: warnings.append('Risk trend increasing')

        return {
            'composite_score': round(composite, 1),
            'warning_level': level,
            'recommended_action': action,
            'bear_score': signal.bear_score,
            'early_warning': signal.early_warning_score,
            'crash_probability': signal.crash_probability,
            'timeframe_confluence': multiframe['confluence_direction'],
            'signal_confluence': confluence['confidence'],
            'persistence': persistence['elevated_duration'],
            'trend_direction': trend['direction'],
            'active_warnings': warnings,
            'warning_count': len(warnings)
        }

    def get_comprehensive_report(self) -> str:
        """
        Generate full comprehensive warning report.

        Returns:
            Multi-line string with complete analysis
        """
        warning = self.get_comprehensive_warning()

        nl = chr(10)
        lines = []
        lines.append('#' * 60)
        lines.append('#  COMPREHENSIVE BEAR WARNING REPORT')
        lines.append('#' * 60)
        lines.append('')

        # Main assessment
        level_icons = {
            'CRITICAL': '[!!!]',
            'WARNING': '[!!]',
            'WATCH': '[!]',
            'NORMAL': '[OK]'
        }
        icon = level_icons.get(warning['warning_level'], '[?]')

        lines.append(f"ASSESSMENT: {icon} {warning['warning_level']}")
        lines.append(f"Composite Score: {warning['composite_score']}/100")
        lines.append(f"Action: {warning['recommended_action']}")
        lines.append('')

        # Component scores
        lines.append('Component Scores:')
        lines.append(f"  Bear Score: {warning['bear_score']:.1f}/100")
        lines.append(f"  Early Warning: {warning['early_warning']:.1f}/100")
        lines.append(f"  Crash Probability: {warning['crash_probability']:.1f}%")
        lines.append(f"  Timeframe Confluence: {warning['timeframe_confluence']}")
        lines.append(f"  Signal Confluence: {warning['signal_confluence']}")
        lines.append(f"  Persistence (signals): {warning['persistence']}")
        lines.append(f"  Trend Direction: {warning['trend_direction']}")
        lines.append('')

        # Active warnings
        if warning['active_warnings']:
            lines.append(f"Active Warnings ({warning['warning_count']}):')
            for w in warning['active_warnings']:
                lines.append(f"  >>> {w}")
        else:
            lines.append('No active warnings')

        lines.append('')
        lines.append('#' * 60)

        return nl.join(lines)

'''

def main():
    with open('src/analysis/fast_bear_detector.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Find insertion point - before the standalone functions
    insertion_point = 'def get_fast_bear_signal() -> FastBearSignal:'

    if insertion_point in content:
        content = content.replace(insertion_point, MULTIFRAME_METHODS + '\n' + insertion_point)
        print("Added multi-timeframe analysis methods")
    else:
        print("Could not find insertion point")
        return

    with open('src/analysis/fast_bear_detector.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Done")

if __name__ == '__main__':
    main()

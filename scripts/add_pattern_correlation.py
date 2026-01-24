"""Add historical pattern matching and cross-asset correlation to fast_bear_detector.py"""

PATTERN_METHODS = '''

    # ==================== PATTERN MATCHING & CORRELATION ====================

    # Historical pre-crash signatures (empirically derived)
    CRASH_SIGNATURES = {
        '2022_bear': {
            'description': '2022 Bear Market (Jan-Oct)',
            'bear_score_range': (55, 80),
            'vix_range': (25, 35),
            'breadth_range': (20, 40),
            'credit_spread_min': 8,
            'vol_compression_min': 0.7,
            'defensive_rotation': True,
            'weight': 1.0  # Most relevant recent example
        },
        '2020_covid': {
            'description': '2020 COVID Crash (Feb-Mar)',
            'bear_score_range': (65, 95),
            'vix_range': (30, 80),
            'breadth_range': (10, 30),
            'credit_spread_min': 15,
            'vol_compression_min': 0.85,
            'defensive_rotation': True,
            'weight': 0.7  # Extreme event
        },
        '2018_correction': {
            'description': '2018 Q4 Correction',
            'bear_score_range': (45, 65),
            'vix_range': (20, 35),
            'breadth_range': (25, 45),
            'credit_spread_min': 5,
            'vol_compression_min': 0.6,
            'defensive_rotation': False,
            'weight': 0.9
        },
        '2024_aug': {
            'description': '2024 August Correction',
            'bear_score_range': (40, 60),
            'vix_range': (20, 65),
            'breadth_range': (30, 50),
            'credit_spread_min': 4,
            'vol_compression_min': 0.65,
            'defensive_rotation': True,
            'weight': 1.0  # Recent example
        },
        '2025_tariff': {
            'description': '2025 Tariff Shock (Mar-Apr)',
            'bear_score_range': (50, 75),
            'vix_range': (22, 45),
            'breadth_range': (25, 40),
            'credit_spread_min': 6,
            'vol_compression_min': 0.7,
            'defensive_rotation': True,
            'weight': 1.0  # Most recent
        }
    }

    def match_historical_patterns(self) -> Dict:
        """
        Match current conditions against historical pre-crash patterns.

        Uses pattern recognition to identify similarity to known
        pre-crash market conditions.

        Returns:
            Dict with pattern match results and similarity scores
        """
        signal = self.detect()
        sector = self.get_sector_leadership()

        matches = []
        best_match = None
        best_score = 0

        for pattern_name, pattern in self.CRASH_SIGNATURES.items():
            score = 0
            match_details = []

            # Bear score match
            bear_min, bear_max = pattern['bear_score_range']
            if bear_min <= signal.bear_score <= bear_max:
                score += 25
                match_details.append('bear_score_in_range')
            elif signal.bear_score >= bear_min * 0.7:
                score += 10
                match_details.append('bear_score_approaching')

            # VIX match
            vix_min, vix_max = pattern['vix_range']
            if vix_min <= signal.vix_level <= vix_max:
                score += 20
                match_details.append('vix_in_range')
            elif signal.vix_level >= vix_min * 0.8:
                score += 8
                match_details.append('vix_approaching')

            # Breadth match
            breadth_min, breadth_max = pattern['breadth_range']
            if breadth_min <= signal.market_breadth_pct <= breadth_max:
                score += 20
                match_details.append('breadth_in_range')
            elif signal.market_breadth_pct <= breadth_max * 1.3:
                score += 8
                match_details.append('breadth_approaching')

            # Credit spread match
            if signal.credit_spread_change >= pattern['credit_spread_min']:
                score += 15
                match_details.append('credit_stress')
            elif signal.credit_spread_change >= pattern['credit_spread_min'] * 0.5:
                score += 6
                match_details.append('credit_approaching')

            # Vol compression match
            if signal.vol_compression >= pattern['vol_compression_min']:
                score += 10
                match_details.append('vol_compressed')

            # Defensive rotation match
            is_defensive = sector.get('is_bearish_rotation', False)
            if pattern['defensive_rotation'] == is_defensive:
                score += 10
                match_details.append('rotation_match')

            # Apply historical weight
            weighted_score = score * pattern['weight']

            matches.append({
                'pattern': pattern_name,
                'description': pattern['description'],
                'raw_score': score,
                'weighted_score': round(weighted_score, 1),
                'match_details': match_details,
                'match_count': len(match_details)
            })

            if weighted_score > best_score:
                best_score = weighted_score
                best_match = pattern_name

        # Sort by weighted score
        matches.sort(key=lambda x: x['weighted_score'], reverse=True)

        # Determine overall pattern status
        if best_score >= 70:
            status = 'STRONG_MATCH'
            desc = f'Strong similarity to {self.CRASH_SIGNATURES[best_match]["description"]}'
        elif best_score >= 50:
            status = 'MODERATE_MATCH'
            desc = f'Moderate similarity to pre-crash patterns'
        elif best_score >= 30:
            status = 'WEAK_MATCH'
            desc = 'Some pre-crash characteristics present'
        else:
            status = 'NO_MATCH'
            desc = 'Current conditions do not match historical crash patterns'

        return {
            'status': status,
            'best_match': best_match,
            'best_score': round(best_score, 1),
            'description': desc,
            'pattern_matches': matches[:3],  # Top 3 matches
            'is_concerning': best_score >= 50
        }

    def get_cross_asset_correlation(self) -> Dict:
        """
        Analyze cross-asset correlations for risk-off signals.

        Key relationships:
        - SPY vs TLT: Negative correlation normal, positive in crisis
        - SPY vs GLD: Rising gold with falling SPY = flight to safety
        - SPY vs VIX: Always inverse, but extreme moves signal panic
        - HYG vs LQD: Credit stress indicator

        Returns:
            Dict with correlation analysis
        """
        try:
            # Fetch data for key assets
            assets = {
                'SPY': yf.Ticker('SPY'),
                'TLT': yf.Ticker('TLT'),  # Long-term treasuries
                'GLD': yf.Ticker('GLD'),  # Gold
                'HYG': yf.Ticker('HYG'),  # High yield bonds
                'LQD': yf.Ticker('LQD'),  # Investment grade bonds
            }

            # Get 20-day returns
            returns = {}
            for name, ticker in assets.items():
                try:
                    hist = ticker.history(period="30d")
                    if len(hist) >= 20:
                        returns[name] = hist['Close'].pct_change().dropna().iloc[-20:]
                except:
                    pass

            if len(returns) < 4:
                return {
                    'status': 'INSUFFICIENT_DATA',
                    'risk_off_score': 0,
                    'description': 'Could not fetch sufficient asset data'
                }

            # Calculate correlations
            correlations = {}

            # SPY-TLT correlation (normally negative, positive = panic)
            if 'SPY' in returns and 'TLT' in returns:
                spy_tlt_corr = returns['SPY'].corr(returns['TLT'])
                correlations['spy_tlt'] = round(spy_tlt_corr, 3)

            # SPY-GLD correlation (gold rises in fear)
            if 'SPY' in returns and 'GLD' in returns:
                spy_gld_corr = returns['SPY'].corr(returns['GLD'])
                correlations['spy_gld'] = round(spy_gld_corr, 3)

            # HYG-LQD spread (credit stress)
            if 'HYG' in returns and 'LQD' in returns:
                hyg_lqd_corr = returns['HYG'].corr(returns['LQD'])
                correlations['hyg_lqd'] = round(hyg_lqd_corr, 3)

            # Calculate recent performance divergence
            spy_perf = ((returns['SPY'].iloc[-1] + 1).cumprod().iloc[-1] - 1) * 100 if 'SPY' in returns else 0
            tlt_perf = ((returns['TLT'].iloc[-1] + 1).cumprod().iloc[-1] - 1) * 100 if 'TLT' in returns else 0
            gld_perf = ((returns['GLD'].iloc[-1] + 1).cumprod().iloc[-1] - 1) * 100 if 'GLD' in returns else 0

            # Risk-off scoring
            risk_off_score = 0
            signals = []

            # SPY-TLT correlation turning positive (flight to safety)
            spy_tlt = correlations.get('spy_tlt', -0.3)
            if spy_tlt > 0.3:
                risk_off_score += 30
                signals.append('Strong flight to treasuries')
            elif spy_tlt > 0:
                risk_off_score += 15
                signals.append('Mild flight to safety')

            # Gold outperforming (haven demand)
            if 'GLD' in returns and 'SPY' in returns:
                gld_vs_spy = (returns['GLD'].sum() - returns['SPY'].sum()) * 100
                if gld_vs_spy > 3:
                    risk_off_score += 25
                    signals.append('Gold significantly outperforming')
                elif gld_vs_spy > 1:
                    risk_off_score += 10
                    signals.append('Gold outperforming')

            # Credit stress (HYG underperforming LQD)
            if 'HYG' in returns and 'LQD' in returns:
                hyg_vs_lqd = (returns['HYG'].sum() - returns['LQD'].sum()) * 100
                if hyg_vs_lqd < -2:
                    risk_off_score += 25
                    signals.append('Significant credit stress')
                elif hyg_vs_lqd < -1:
                    risk_off_score += 10
                    signals.append('Mild credit stress')

            # All assets correlating (correlation spike in crisis)
            avg_corr = sum(abs(c) for c in correlations.values()) / len(correlations) if correlations else 0
            if avg_corr > 0.6:
                risk_off_score += 20
                signals.append('High cross-asset correlation')

            # Determine status
            if risk_off_score >= 60:
                status = 'HIGH_RISK_OFF'
                desc = 'Strong risk-off signals across assets'
            elif risk_off_score >= 35:
                status = 'MODERATE_RISK_OFF'
                desc = 'Some risk-off rotation occurring'
            elif risk_off_score >= 15:
                status = 'MILD_RISK_OFF'
                desc = 'Minor risk-off signals'
            else:
                status = 'RISK_ON'
                desc = 'No significant risk-off signals'

            return {
                'status': status,
                'risk_off_score': risk_off_score,
                'correlations': correlations,
                'spy_tlt_correlation': correlations.get('spy_tlt', 0),
                'signals': signals,
                'is_risk_off': risk_off_score >= 35,
                'description': desc
            }

        except Exception as e:
            return {
                'status': 'ERROR',
                'risk_off_score': 0,
                'description': f'Error: {str(e)}'
            }

    def get_momentum_regime(self) -> Dict:
        """
        Detect momentum regime shifts that precede corrections.

        Key signals:
        - Momentum deceleration before reversals
        - Breadth momentum divergence
        - Rate of change exhaustion

        Returns:
            Dict with momentum regime analysis
        """
        try:
            spy = yf.Ticker("SPY")
            hist = spy.history(period="60d")

            if len(hist) < 40:
                return {
                    'regime': 'UNKNOWN',
                    'momentum_score': 0,
                    'description': 'Insufficient data'
                }

            close = hist['Close']

            # Calculate momentum metrics
            # 10-day momentum
            mom_10d = ((close.iloc[-1] / close.iloc[-10]) - 1) * 100
            mom_10d_prev = ((close.iloc[-10] / close.iloc[-20]) - 1) * 100

            # 20-day momentum
            mom_20d = ((close.iloc[-1] / close.iloc[-20]) - 1) * 100
            mom_20d_prev = ((close.iloc[-20] / close.iloc[-40]) - 1) * 100

            # Momentum acceleration/deceleration
            mom_accel_10d = mom_10d - mom_10d_prev
            mom_accel_20d = mom_20d - mom_20d_prev

            # Rate of change of rate of change (momentum of momentum)
            roc_5d = ((close.iloc[-1] / close.iloc[-5]) - 1) * 100
            roc_5d_5d_ago = ((close.iloc[-5] / close.iloc[-10]) - 1) * 100
            roc_acceleration = roc_5d - roc_5d_5d_ago

            # Higher highs / higher lows analysis
            recent_highs = [close.iloc[i:i+5].max() for i in range(-20, -5, 5)]
            making_higher_highs = all(recent_highs[i] < recent_highs[i+1] for i in range(len(recent_highs)-1))

            # Calculate momentum score (negative = bearish momentum regime)
            momentum_score = 0

            # Momentum deceleration (bearish)
            if mom_accel_10d < -2:
                momentum_score -= 25
            elif mom_accel_10d < 0:
                momentum_score -= 10

            if mom_accel_20d < -3:
                momentum_score -= 20
            elif mom_accel_20d < 0:
                momentum_score -= 8

            # ROC deceleration
            if roc_acceleration < -1:
                momentum_score -= 15
            elif roc_acceleration < 0:
                momentum_score -= 5

            # Not making higher highs (trend weakening)
            if not making_higher_highs:
                momentum_score -= 15

            # Positive momentum (add back)
            if mom_10d > 2:
                momentum_score += 15
            if mom_20d > 4:
                momentum_score += 10

            # Determine regime
            if momentum_score <= -40:
                regime = 'DETERIORATING_FAST'
                desc = 'Momentum deteriorating rapidly - high reversal risk'
            elif momentum_score <= -20:
                regime = 'DETERIORATING'
                desc = 'Momentum weakening - watch for breakdown'
            elif momentum_score >= 20:
                regime = 'ACCELERATING'
                desc = 'Momentum accelerating - bullish'
            elif momentum_score >= 0:
                regime = 'STABLE'
                desc = 'Momentum stable'
            else:
                regime = 'WEAKENING'
                desc = 'Momentum showing mild weakness'

            return {
                'regime': regime,
                'momentum_score': momentum_score,
                'mom_10d': round(mom_10d, 2),
                'mom_20d': round(mom_20d, 2),
                'mom_accel_10d': round(mom_accel_10d, 2),
                'mom_accel_20d': round(mom_accel_20d, 2),
                'roc_acceleration': round(roc_acceleration, 2),
                'making_higher_highs': making_higher_highs,
                'is_bearish': momentum_score <= -20,
                'description': desc
            }

        except Exception as e:
            return {
                'regime': 'ERROR',
                'momentum_score': 0,
                'description': f'Error: {str(e)}'
            }

    def get_ultimate_warning(self) -> Dict:
        """
        Generate ultimate bear warning combining ALL analysis methods.

        This is the most comprehensive warning signal, combining:
        - Base bear score
        - Pattern matching
        - Cross-asset correlation
        - Momentum regime
        - Sector leadership
        - Volume profile
        - Divergence analysis
        - Multi-timeframe analysis

        Returns:
            Dict with ultimate warning assessment
        """
        # Gather all analyses
        signal = self.detect()
        pattern = self.match_historical_patterns()
        correlation = self.get_cross_asset_correlation()
        momentum = self.get_momentum_regime()
        sector = self.get_sector_leadership()
        volume = self.get_volume_profile()
        divergence = self.get_divergence_analysis()
        multiframe = self.get_multiframe_analysis()
        adaptive = self.get_adaptive_thresholds()

        # Calculate ultimate score (0-100)
        ultimate_score = 0

        # Base bear score (weight: 25%)
        ultimate_score += signal.bear_score * 0.25

        # Pattern match (weight: 15%)
        ultimate_score += pattern.get('best_score', 0) * 0.15

        # Cross-asset risk-off (weight: 15%)
        ultimate_score += correlation.get('risk_off_score', 0) * 0.15

        # Momentum regime (weight: 15%) - convert negative to positive bearish score
        mom_score = max(0, -momentum.get('momentum_score', 0))
        ultimate_score += min(mom_score, 50) * 0.30  # Cap at 50, weight 15%

        # Sector rotation (weight: 10%)
        if sector.get('is_bearish_rotation', False):
            ultimate_score += 10

        # Volume distribution (weight: 10%)
        ultimate_score += volume.get('bearish_score', 0) * 0.10

        # Divergence (weight: 5%)
        ultimate_score += divergence.get('divergence_score', 0) * 0.05

        # Timeframe confluence (weight: 5%)
        if multiframe.get('bearish_timeframes', 0) >= 2:
            ultimate_score += 5

        # Apply regime sensitivity
        regime_mult = adaptive.get('multiplier', 1.0)
        if regime_mult < 1:  # Low vol - be more sensitive
            ultimate_score *= 1.15
        elif regime_mult > 1.3:  # High vol - less sensitive
            ultimate_score *= 0.90

        ultimate_score = min(100, ultimate_score)

        # Determine warning level
        if ultimate_score >= 75:
            level = 'CRITICAL'
            action = 'IMMEDIATE ACTION: Reduce exposure significantly'
            urgency = 'HIGH'
        elif ultimate_score >= 55:
            level = 'WARNING'
            action = 'CAUTION: Consider reducing positions, set tight stops'
            urgency = 'MEDIUM'
        elif ultimate_score >= 35:
            level = 'WATCH'
            action = 'MONITOR: Stay vigilant, prepare contingency plans'
            urgency = 'LOW'
        else:
            level = 'NORMAL'
            action = 'HOLD: No immediate action required'
            urgency = 'NONE'

        # Collect all warning flags
        flags = []
        if signal.bear_score >= 40: flags.append(f'Bear score: {signal.bear_score:.0f}')
        if pattern.get('is_concerning'): flags.append(f"Pattern: {pattern.get('best_match')}")
        if correlation.get('is_risk_off'): flags.append('Cross-asset risk-off')
        if momentum.get('is_bearish'): flags.append(f"Momentum: {momentum.get('regime')}")
        if sector.get('is_bearish_rotation'): flags.append('Defensive rotation')
        if volume.get('is_bearish'): flags.append(f"Volume: {volume.get('pattern')}")
        if divergence.get('is_bearish'): flags.append('Bearish divergence')

        return {
            'ultimate_score': round(ultimate_score, 1),
            'warning_level': level,
            'urgency': urgency,
            'recommended_action': action,
            'crash_probability': signal.crash_probability,
            'components': {
                'bear_score': signal.bear_score,
                'pattern_match': pattern.get('best_score', 0),
                'risk_off_score': correlation.get('risk_off_score', 0),
                'momentum_regime': momentum.get('regime'),
                'sector_leadership': sector.get('leadership'),
                'volume_pattern': volume.get('pattern'),
                'divergence': divergence.get('divergence_type'),
                'timeframe': multiframe.get('confluence_direction')
            },
            'active_flags': flags,
            'flag_count': len(flags),
            'regime': adaptive.get('vol_regime'),
            'sensitivity': adaptive.get('sensitivity')
        }

    def get_ultimate_report(self) -> str:
        """
        Generate the ultimate comprehensive bear warning report.

        Returns:
            Multi-line string with complete analysis
        """
        warning = self.get_ultimate_warning()

        nl = chr(10)
        lines = []
        lines.append('*' * 60)
        lines.append('*  ULTIMATE BEAR WARNING REPORT')
        lines.append('*' * 60)
        lines.append('')

        # Main assessment with visual indicator
        level_visuals = {
            'CRITICAL': '[!!! CRITICAL !!!]',
            'WARNING': '[!! WARNING !!]',
            'WATCH': '[! WATCH !]',
            'NORMAL': '[OK - NORMAL]'
        }
        visual = level_visuals.get(warning['warning_level'], '[?]')

        lines.append(f"STATUS: {visual}")
        lines.append(f"Ultimate Score: {warning['ultimate_score']}/100")
        lines.append(f"Crash Probability: {warning['crash_probability']:.1f}%")
        lines.append(f"Urgency: {warning['urgency']}")
        lines.append('')
        lines.append(f">>> {warning['recommended_action']}")
        lines.append('')

        # Component breakdown
        lines.append('Component Analysis:')
        comp = warning['components']
        lines.append(f"  Bear Score: {comp['bear_score']:.1f}/100")
        lines.append(f"  Pattern Match: {comp['pattern_match']:.1f}/100")
        lines.append(f"  Risk-Off Score: {comp['risk_off_score']}/100")
        lines.append(f"  Momentum: {comp['momentum_regime']}")
        lines.append(f"  Sector Leadership: {comp['sector_leadership']}")
        lines.append(f"  Volume: {comp['volume_pattern']}")
        lines.append(f"  Divergence: {comp['divergence']}")
        lines.append(f"  Timeframe: {comp['timeframe']}")
        lines.append('')

        # Active warning flags
        if warning['active_flags']:
            lines.append(f"Warning Flags ({warning['flag_count']}):")
            for flag in warning['active_flags']:
                lines.append(f"  >>> {flag}")
        else:
            lines.append('No active warning flags')

        lines.append('')
        lines.append(f"Regime: {warning['regime']} ({warning['sensitivity']} sensitivity)")
        lines.append('')
        lines.append('*' * 60)

        return nl.join(lines)

'''

def main():
    with open('src/analysis/fast_bear_detector.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Find insertion point - before the standalone functions
    insertion_point = 'def get_fast_bear_signal() -> FastBearSignal:'

    if insertion_point in content:
        content = content.replace(insertion_point, PATTERN_METHODS + '\n' + insertion_point)
        print("Added pattern matching and correlation methods")
    else:
        print("Could not find insertion point")
        return

    with open('src/analysis/fast_bear_detector.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Done")

if __name__ == '__main__':
    main()

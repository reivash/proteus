"""Add adaptive thresholds and sector leadership analysis to fast_bear_detector.py"""

ADAPTIVE_METHODS = '''

    # ==================== ADAPTIVE & SECTOR ANALYSIS ====================

    def get_adaptive_thresholds(self) -> Dict:
        """
        Get regime-adjusted thresholds for current market conditions.

        Thresholds are tighter in low-vol regimes (more sensitive)
        and looser in high-vol regimes (avoid false positives).

        Returns:
            Dict with adjusted thresholds for each indicator
        """
        signal = self.detect()
        vol_regime = signal.vol_regime

        # Base thresholds
        base = {
            'spy_roc_watch': -2.0,
            'spy_roc_warning': -3.0,
            'vix_watch': 25,
            'vix_warning': 30,
            'breadth_watch': 40,
            'breadth_warning': 30,
            'credit_watch': 5,
            'credit_warning': 10
        }

        # Regime multipliers - tighter in low vol, looser in high vol
        multipliers = {
            'LOW_COMPLACENT': 0.7,   # More sensitive - crashes start in calm
            'NORMAL': 1.0,
            'ELEVATED': 1.3,
            'CRISIS': 1.6            # Less sensitive - already in crisis
        }

        mult = multipliers.get(vol_regime, 1.0)

        # Adjust thresholds
        adjusted = {}
        for key, value in base.items():
            if 'breadth' in key or 'vix' in key:
                # For these, higher mult means higher threshold (less sensitive)
                if 'breadth' in key:
                    adjusted[key] = value * (2 - mult)  # Invert for breadth
                else:
                    adjusted[key] = value * mult
            else:
                # For negative thresholds (ROC, credit)
                adjusted[key] = value * mult

        return {
            'vol_regime': vol_regime,
            'multiplier': mult,
            'thresholds': adjusted,
            'sensitivity': 'HIGH' if mult < 1 else 'NORMAL' if mult == 1 else 'LOW',
            'description': f'{vol_regime} regime: sensitivity {"increased" if mult < 1 else "normal" if mult == 1 else "reduced"}'
        }

    def get_sector_leadership(self) -> Dict:
        """
        Analyze sector leadership patterns to detect rotation.

        Defensive sectors leading = risk-off rotation (bearish)
        Cyclical sectors leading = risk-on (bullish)

        Returns:
            Dict with sector leadership analysis
        """
        try:
            # Define sector groups
            defensive = ['XLU', 'XLP', 'XLV']  # Utilities, Staples, Healthcare
            cyclical = ['XLK', 'XLY', 'XLF', 'XLI']  # Tech, Discretionary, Financials, Industrials
            risk_sensitive = ['XLE', 'XLB', 'XLRE']  # Energy, Materials, Real Estate

            # Fetch performance data
            all_sectors = defensive + cyclical + risk_sensitive

            performances = {}
            for sector in all_sectors:
                try:
                    ticker = yf.Ticker(sector)
                    hist = ticker.history(period="20d")
                    if len(hist) >= 10:
                        perf_5d = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-5]) - 1) * 100
                        perf_10d = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-10]) - 1) * 100
                        performances[sector] = {
                            'perf_5d': round(perf_5d, 2),
                            'perf_10d': round(perf_10d, 2)
                        }
                except:
                    pass

            if len(performances) < 5:
                return {
                    'leadership': 'UNKNOWN',
                    'rotation_signal': 0,
                    'description': 'Insufficient sector data'
                }

            # Calculate group averages
            def avg_perf(sectors, period='perf_5d'):
                perfs = [performances[s][period] for s in sectors if s in performances]
                return sum(perfs) / len(perfs) if perfs else 0

            defensive_5d = avg_perf(defensive, 'perf_5d')
            cyclical_5d = avg_perf(cyclical, 'perf_5d')
            defensive_10d = avg_perf(defensive, 'perf_10d')
            cyclical_10d = avg_perf(cyclical, 'perf_10d')

            # Calculate rotation signal
            # Positive = defensive outperforming (risk-off)
            # Negative = cyclical outperforming (risk-on)
            rotation_5d = defensive_5d - cyclical_5d
            rotation_10d = defensive_10d - cyclical_10d

            # Combined rotation signal (-100 to +100)
            rotation_signal = (rotation_5d * 10 + rotation_10d * 5) / 2
            rotation_signal = max(-100, min(100, rotation_signal))

            # Determine leadership
            if rotation_signal > 20:
                leadership = 'DEFENSIVE'
                desc = 'Defensive sectors leading - risk-off rotation (bearish signal)'
            elif rotation_signal < -20:
                leadership = 'CYCLICAL'
                desc = 'Cyclical sectors leading - risk-on rotation (bullish)'
            else:
                leadership = 'NEUTRAL'
                desc = 'No clear sector leadership'

            # Find best and worst performers
            sorted_5d = sorted(performances.items(), key=lambda x: x[1]['perf_5d'], reverse=True)
            best_sector = sorted_5d[0][0] if sorted_5d else 'N/A'
            worst_sector = sorted_5d[-1][0] if sorted_5d else 'N/A'

            return {
                'leadership': leadership,
                'rotation_signal': round(rotation_signal, 1),
                'defensive_5d': round(defensive_5d, 2),
                'cyclical_5d': round(cyclical_5d, 2),
                'defensive_10d': round(defensive_10d, 2),
                'cyclical_10d': round(cyclical_10d, 2),
                'best_sector': best_sector,
                'worst_sector': worst_sector,
                'sector_performances': performances,
                'is_bearish_rotation': leadership == 'DEFENSIVE',
                'description': desc
            }

        except Exception as e:
            return {
                'leadership': 'ERROR',
                'rotation_signal': 0,
                'description': f'Error: {str(e)}'
            }

    def get_volume_profile(self) -> Dict:
        """
        Analyze volume patterns that precede market drops.

        Key patterns:
        - Distribution days: High volume down days
        - Exhaustion: Volume declining on rallies
        - Capitulation: Extreme volume spikes

        Returns:
            Dict with volume profile analysis
        """
        try:
            spy = yf.Ticker("SPY")
            hist = spy.history(period="60d")

            if len(hist) < 30:
                return {
                    'pattern': 'UNKNOWN',
                    'distribution_days': 0,
                    'description': 'Insufficient data'
                }

            close = hist['Close']
            volume = hist['Volume']

            # Calculate metrics
            avg_volume_20d = volume.iloc[-20:].mean()

            # Distribution days: Down day with above-average volume (last 20 days)
            distribution_days = 0
            for i in range(-20, 0):
                daily_return = (close.iloc[i] / close.iloc[i-1] - 1) * 100
                daily_volume = volume.iloc[i]
                if daily_return < -0.5 and daily_volume > avg_volume_20d * 1.2:
                    distribution_days += 1

            # Volume trend on up days vs down days
            up_day_volume = []
            down_day_volume = []
            for i in range(-20, 0):
                daily_return = (close.iloc[i] / close.iloc[i-1] - 1) * 100
                if daily_return > 0:
                    up_day_volume.append(volume.iloc[i])
                else:
                    down_day_volume.append(volume.iloc[i])

            avg_up_volume = sum(up_day_volume) / len(up_day_volume) if up_day_volume else 0
            avg_down_volume = sum(down_day_volume) / len(down_day_volume) if down_day_volume else 0

            # Volume ratio: >1 means more volume on down days (bearish)
            volume_ratio = avg_down_volume / avg_up_volume if avg_up_volume > 0 else 1

            # Recent volume trend
            recent_volume = volume.iloc[-5:].mean()
            prior_volume = volume.iloc[-20:-5].mean()
            volume_expansion = recent_volume / prior_volume if prior_volume > 0 else 1

            # Determine pattern
            if distribution_days >= 5 and volume_ratio > 1.2:
                pattern = 'DISTRIBUTION'
                desc = f'{distribution_days} distribution days detected - institutional selling'
                bearish_score = 80
            elif distribution_days >= 3:
                pattern = 'ACCUMULATION_STRESS'
                desc = f'{distribution_days} distribution days - watch for breakdown'
                bearish_score = 50
            elif volume_ratio > 1.3:
                pattern = 'SELLING_PRESSURE'
                desc = 'Higher volume on down days - selling pressure'
                bearish_score = 40
            elif volume_expansion > 1.5 and volume_ratio > 1:
                pattern = 'VOLATILITY_EXPANSION'
                desc = 'Volume expanding with selling bias'
                bearish_score = 60
            else:
                pattern = 'NORMAL'
                desc = 'Normal volume patterns'
                bearish_score = 10

            return {
                'pattern': pattern,
                'distribution_days': distribution_days,
                'volume_ratio': round(volume_ratio, 2),
                'volume_expansion': round(volume_expansion, 2),
                'avg_up_volume': int(avg_up_volume),
                'avg_down_volume': int(avg_down_volume),
                'bearish_score': bearish_score,
                'is_bearish': bearish_score >= 50,
                'description': desc
            }

        except Exception as e:
            return {
                'pattern': 'ERROR',
                'distribution_days': 0,
                'description': f'Error: {str(e)}'
            }

    def get_divergence_analysis(self) -> Dict:
        """
        Detect price-indicator divergences that precede reversals.

        Bearish divergence: Price making highs, indicators making lows
        This is a classic topping pattern.

        Returns:
            Dict with divergence analysis
        """
        try:
            spy = yf.Ticker("SPY")
            hist = spy.history(period="60d")

            if len(hist) < 30:
                return {
                    'divergence_type': 'UNKNOWN',
                    'divergence_score': 0,
                    'description': 'Insufficient data'
                }

            close = hist['Close']
            volume = hist['Volume']

            # Calculate RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # Get recent price action
            price_20d_high = close.iloc[-20:].max()
            price_10d_high = close.iloc[-10:].max()
            price_now = close.iloc[-1]

            # Get RSI at those points
            rsi_at_20d_high = rsi.iloc[close.iloc[-20:].idxmax()] if len(rsi) > 20 else 50
            rsi_at_10d_high = rsi.iloc[close.iloc[-10:].idxmax()] if len(rsi) > 10 else 50
            rsi_now = rsi.iloc[-1]

            # Price near highs but RSI declining = bearish divergence
            price_near_high = price_now >= price_20d_high * 0.98
            rsi_declining = rsi_now < rsi_at_20d_high - 5

            # Volume divergence: price rising but volume declining
            vol_20d_avg = volume.iloc[-20:-10].mean()
            vol_10d_avg = volume.iloc[-10:].mean()
            volume_declining = vol_10d_avg < vol_20d_avg * 0.85

            # Calculate divergence score (0-100)
            divergence_score = 0

            # RSI divergence
            if price_near_high and rsi_declining:
                divergence_score += 40

            # Volume divergence
            if price_now > close.iloc[-20:].mean() and volume_declining:
                divergence_score += 30

            # RSI overbought
            if rsi_now > 70:
                divergence_score += 15

            # Price extended from 20d MA
            ma_20 = close.iloc[-20:].mean()
            extension = ((price_now / ma_20) - 1) * 100
            if extension > 5:
                divergence_score += 15

            # Determine divergence type
            if divergence_score >= 60:
                div_type = 'STRONG_BEARISH'
                desc = 'Strong bearish divergence - high reversal probability'
            elif divergence_score >= 40:
                div_type = 'MODERATE_BEARISH'
                desc = 'Moderate bearish divergence - watch for confirmation'
            elif divergence_score >= 20:
                div_type = 'MILD_BEARISH'
                desc = 'Mild bearish signals present'
            else:
                div_type = 'NONE'
                desc = 'No significant divergence detected'

            return {
                'divergence_type': div_type,
                'divergence_score': divergence_score,
                'rsi_current': round(float(rsi_now), 1),
                'rsi_at_high': round(float(rsi_at_20d_high), 1),
                'price_near_high': price_near_high,
                'rsi_declining': rsi_declining,
                'volume_declining': volume_declining,
                'price_extension': round(extension, 2),
                'is_bearish': divergence_score >= 40,
                'description': desc
            }

        except Exception as e:
            return {
                'divergence_type': 'ERROR',
                'divergence_score': 0,
                'description': f'Error: {str(e)}'
            }

    def get_enhanced_warning(self) -> Dict:
        """
        Generate enhanced warning combining all analysis methods.

        Combines: base signal, adaptive thresholds, sector leadership,
        volume profile, divergence analysis.

        Returns:
            Dict with enhanced warning assessment
        """
        signal = self.detect()
        adaptive = self.get_adaptive_thresholds()
        sector = self.get_sector_leadership()
        volume = self.get_volume_profile()
        divergence = self.get_divergence_analysis()
        multiframe = self.get_multiframe_analysis()

        # Calculate enhanced score
        base_score = signal.bear_score

        # Add sector rotation bonus
        sector_bonus = 10 if sector.get('is_bearish_rotation', False) else 0

        # Add volume distribution bonus
        volume_bonus = volume.get('bearish_score', 0) * 0.15

        # Add divergence bonus
        divergence_bonus = divergence.get('divergence_score', 0) * 0.15

        # Add multiframe confluence bonus
        mf_bonus = 10 if multiframe.get('bearish_timeframes', 0) >= 2 else 0

        # Adjust for regime sensitivity
        regime_mult = adaptive.get('multiplier', 1.0)
        if regime_mult < 1:  # Low vol regime - be more sensitive
            sensitivity_mult = 1.2
        elif regime_mult > 1:  # High vol regime - be less sensitive
            sensitivity_mult = 0.9
        else:
            sensitivity_mult = 1.0

        enhanced_score = (base_score + sector_bonus + volume_bonus + divergence_bonus + mf_bonus) * sensitivity_mult
        enhanced_score = min(100, enhanced_score)

        # Determine warning level
        if enhanced_score >= 70:
            level = 'CRITICAL'
            action = 'REDUCE EXPOSURE - Multiple warning signals aligned'
        elif enhanced_score >= 50:
            level = 'WARNING'
            action = 'Consider reducing positions, tighten stops'
        elif enhanced_score >= 30:
            level = 'WATCH'
            action = 'Monitor closely, prepare contingency plans'
        else:
            level = 'NORMAL'
            action = 'No immediate action required'

        # Collect all warning signals
        warnings = []
        if base_score >= 30: warnings.append(f'Bear score elevated: {base_score:.0f}')
        if sector.get('is_bearish_rotation'): warnings.append('Defensive sector rotation')
        if volume.get('is_bearish'): warnings.append(f"Volume: {volume.get('pattern')}")
        if divergence.get('is_bearish'): warnings.append(f"Divergence: {divergence.get('divergence_type')}")
        if multiframe.get('bearish_timeframes', 0) >= 2: warnings.append('Multiple bearish timeframes')

        return {
            'enhanced_score': round(enhanced_score, 1),
            'base_score': round(base_score, 1),
            'warning_level': level,
            'recommended_action': action,
            'regime': adaptive.get('vol_regime'),
            'sensitivity': adaptive.get('sensitivity'),
            'sector_leadership': sector.get('leadership'),
            'volume_pattern': volume.get('pattern'),
            'divergence': divergence.get('divergence_type'),
            'timeframe_confluence': multiframe.get('confluence_direction'),
            'active_warnings': warnings,
            'warning_count': len(warnings),
            'crash_probability': signal.crash_probability
        }

    def get_enhanced_report(self) -> str:
        """
        Generate comprehensive enhanced warning report.

        Returns:
            Multi-line string with complete enhanced analysis
        """
        warning = self.get_enhanced_warning()

        nl = chr(10)
        lines = []
        lines.append('#' * 60)
        lines.append('#  ENHANCED BEAR WARNING REPORT')
        lines.append('#' * 60)
        lines.append('')

        # Level indicator
        level_icons = {
            'CRITICAL': '[!!!]',
            'WARNING': '[!!]',
            'WATCH': '[!]',
            'NORMAL': '[OK]'
        }
        icon = level_icons.get(warning['warning_level'], '[?]')

        lines.append(f"LEVEL: {icon} {warning['warning_level']}")
        lines.append(f"Enhanced Score: {warning['enhanced_score']}/100 (base: {warning['base_score']})")
        lines.append(f"Crash Probability: {warning['crash_probability']:.1f}%")
        lines.append('')
        lines.append(f"ACTION: {warning['recommended_action']}")
        lines.append('')

        # Analysis components
        lines.append('Analysis Components:')
        lines.append(f"  Volatility Regime: {warning['regime']} ({warning['sensitivity']} sensitivity)")
        lines.append(f"  Sector Leadership: {warning['sector_leadership']}")
        lines.append(f"  Volume Pattern: {warning['volume_pattern']}")
        lines.append(f"  Divergence: {warning['divergence']}")
        lines.append(f"  Timeframe: {warning['timeframe_confluence']}")
        lines.append('')

        # Active warnings
        if warning['active_warnings']:
            lines.append(f"Active Warnings ({warning['warning_count']}):")
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
        content = content.replace(insertion_point, ADAPTIVE_METHODS + '\n' + insertion_point)
        print("Added adaptive and sector analysis methods")
    else:
        print("Could not find insertion point")
        return

    with open('src/analysis/fast_bear_detector.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Done")

if __name__ == '__main__':
    main()

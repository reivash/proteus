"""Add intraday monitoring, recovery detection, and sector risk ranking to fast_bear_detector.py"""

MONITORING_METHODS = '''

    # ==================== INTRADAY MONITORING & RECOVERY ====================

    def get_intraday_monitor(self) -> Dict:
        """
        Monitor intraday signal changes for real-time alerting.

        Tracks changes since market open and flags significant moves.

        Returns:
            Dict with intraday monitoring data
        """
        signal = self.detect()
        intraday = self.get_intraday_trend()

        # Get current time context
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)

        is_market_hours = market_open <= now <= market_close
        hours_since_open = max(0, (now - market_open).total_seconds() / 3600)

        # Intraday changes
        change_today = intraday.get('change_today', 0)
        signals_today = intraday.get('signals_today', 0)

        # Determine intraday alert status
        alerts = []

        if change_today >= 10:
            alerts.append('RAPID_DETERIORATION')
        elif change_today >= 5:
            alerts.append('SCORE_RISING')

        if change_today <= -10:
            alerts.append('RAPID_IMPROVEMENT')
        elif change_today <= -5:
            alerts.append('SCORE_FALLING')

        # High of day analysis
        hod = intraday.get('high_of_day', signal.bear_score)
        if hod >= 50 and signal.bear_score < 50:
            alerts.append('RETREATED_FROM_WARNING')
        elif hod >= 30 and signal.bear_score < 30:
            alerts.append('RETREATED_FROM_WATCH')

        # Volatility of signals
        if signals_today >= 5:
            lod = intraday.get('low_of_day', signal.bear_score)
            intraday_range = hod - lod
            if intraday_range >= 15:
                alerts.append('HIGH_INTRADAY_VOLATILITY')

        # Determine monitoring status
        if 'RAPID_DETERIORATION' in alerts:
            status = 'ALERT'
            action = 'Conditions deteriorating rapidly - monitor closely'
        elif 'SCORE_RISING' in alerts:
            status = 'WATCH'
            action = 'Bear score rising today - stay vigilant'
        elif 'RAPID_IMPROVEMENT' in alerts:
            status = 'IMPROVING'
            action = 'Conditions improving rapidly'
        else:
            status = 'STABLE'
            action = 'No significant intraday changes'

        return {
            'status': status,
            'action': action,
            'current_score': signal.bear_score,
            'alert_level': signal.alert_level,
            'change_today': change_today,
            'signals_today': signals_today,
            'high_of_day': intraday.get('high_of_day', signal.bear_score),
            'low_of_day': intraday.get('low_of_day', signal.bear_score),
            'is_market_hours': is_market_hours,
            'hours_since_open': round(hours_since_open, 1),
            'intraday_alerts': alerts,
            'alert_count': len(alerts)
        }

    def detect_recovery(self) -> Dict:
        """
        Detect when bearish conditions are improving/recovering.

        Identifies:
        - Score declining from elevated levels
        - Indicators normalizing
        - Risk-on rotation beginning

        Returns:
            Dict with recovery detection results
        """
        signal = self.detect()
        trend = self.get_signal_trend()
        sector = self.get_sector_leadership()
        correlation = self.get_cross_asset_correlation()

        # Recovery indicators
        recovery_signals = []
        recovery_score = 0

        # 1. Bear score declining
        if trend.get('direction') == 'IMPROVING_FAST':
            recovery_score += 30
            recovery_signals.append('Bear score falling rapidly')
        elif trend.get('direction') == 'IMPROVING':
            recovery_score += 15
            recovery_signals.append('Bear score declining')

        # 2. Came down from elevated levels
        max_recent = trend.get('max_score', signal.bear_score)
        if max_recent >= 50 and signal.bear_score < 40:
            recovery_score += 20
            recovery_signals.append('Retreated from WARNING level')
        elif max_recent >= 30 and signal.bear_score < 25:
            recovery_score += 10
            recovery_signals.append('Retreated from WATCH level')

        # 3. Cyclical sectors leading (risk-on)
        if sector.get('leadership') == 'CYCLICAL':
            recovery_score += 15
            recovery_signals.append('Cyclical sectors leading')

        # 4. Risk-on cross-asset
        if correlation.get('status') == 'RISK_ON':
            recovery_score += 15
            recovery_signals.append('Cross-asset risk-on')

        # 5. VIX normalizing
        if signal.vix_level < 18 and signal.vix_term_structure < 1.0:
            recovery_score += 10
            recovery_signals.append('VIX normalized')

        # 6. Breadth improving
        if signal.market_breadth_pct >= 60:
            recovery_score += 10
            recovery_signals.append('Breadth healthy')

        # Determine recovery status
        if recovery_score >= 60:
            status = 'STRONG_RECOVERY'
            desc = 'Strong recovery underway - conditions normalizing'
        elif recovery_score >= 40:
            status = 'RECOVERING'
            desc = 'Recovery signs present - improving conditions'
        elif recovery_score >= 20:
            status = 'EARLY_RECOVERY'
            desc = 'Early recovery signals - still cautious'
        elif signal.bear_score >= 30:
            status = 'STILL_ELEVATED'
            desc = 'Bear signals still elevated - no recovery yet'
        else:
            status = 'NORMAL'
            desc = 'Conditions normal - no recovery needed'

        return {
            'status': status,
            'recovery_score': recovery_score,
            'description': desc,
            'recovery_signals': recovery_signals,
            'signal_count': len(recovery_signals),
            'current_bear_score': signal.bear_score,
            'recent_max_score': max_recent,
            'trend_direction': trend.get('direction'),
            'is_recovering': recovery_score >= 40
        }

    def get_sector_risk_ranking(self) -> Dict:
        """
        Rank sectors by current risk level.

        Identifies which sectors are most vulnerable to a downturn.

        Returns:
            Dict with sector risk rankings
        """
        try:
            # Define sectors with characteristics
            sectors = {
                'XLK': {'name': 'Technology', 'beta': 1.2, 'type': 'cyclical'},
                'XLY': {'name': 'Consumer Discretionary', 'beta': 1.1, 'type': 'cyclical'},
                'XLF': {'name': 'Financials', 'beta': 1.1, 'type': 'cyclical'},
                'XLI': {'name': 'Industrials', 'beta': 1.0, 'type': 'cyclical'},
                'XLC': {'name': 'Communication', 'beta': 1.0, 'type': 'cyclical'},
                'XLE': {'name': 'Energy', 'beta': 1.3, 'type': 'cyclical'},
                'XLB': {'name': 'Materials', 'beta': 1.1, 'type': 'cyclical'},
                'XLRE': {'name': 'Real Estate', 'beta': 0.9, 'type': 'interest_sensitive'},
                'XLV': {'name': 'Healthcare', 'beta': 0.8, 'type': 'defensive'},
                'XLP': {'name': 'Consumer Staples', 'beta': 0.6, 'type': 'defensive'},
                'XLU': {'name': 'Utilities', 'beta': 0.5, 'type': 'defensive'}
            }

            rankings = []

            for ticker, info in sectors.items():
                try:
                    etf = yf.Ticker(ticker)
                    hist = etf.history(period="30d")

                    if len(hist) < 20:
                        continue

                    close = hist['Close']
                    volume = hist['Volume']

                    # Calculate risk metrics
                    perf_5d = ((close.iloc[-1] / close.iloc[-5]) - 1) * 100
                    perf_20d = ((close.iloc[-1] / close.iloc[-20]) - 1) * 100
                    volatility = close.pct_change().std() * 100

                    # Distance from 20d high
                    high_20d = close.max()
                    dist_from_high = ((close.iloc[-1] / high_20d) - 1) * 100

                    # Volume trend (high volume on down days = distribution)
                    recent_vol = volume.iloc[-5:].mean()
                    prior_vol = volume.iloc[-20:-5].mean()
                    vol_ratio = recent_vol / prior_vol if prior_vol > 0 else 1

                    # Calculate risk score (higher = more risk)
                    risk_score = 0

                    # Negative performance
                    if perf_5d < -2: risk_score += 15
                    if perf_5d < -5: risk_score += 15
                    if perf_20d < -5: risk_score += 10
                    if perf_20d < -10: risk_score += 10

                    # Distance from high
                    if dist_from_high < -5: risk_score += 10
                    if dist_from_high < -10: risk_score += 10

                    # High volatility
                    if volatility > 2: risk_score += 10

                    # Volume distribution
                    if vol_ratio > 1.3 and perf_5d < 0: risk_score += 10

                    # Beta adjustment (high beta = more risk)
                    risk_score *= info['beta']

                    # Type adjustment
                    if info['type'] == 'cyclical':
                        risk_score *= 1.1  # Cyclicals more at risk
                    elif info['type'] == 'defensive':
                        risk_score *= 0.8  # Defensives less at risk

                    rankings.append({
                        'ticker': ticker,
                        'name': info['name'],
                        'type': info['type'],
                        'risk_score': round(risk_score, 1),
                        'perf_5d': round(perf_5d, 2),
                        'perf_20d': round(perf_20d, 2),
                        'dist_from_high': round(dist_from_high, 2),
                        'volatility': round(volatility, 2)
                    })

                except Exception:
                    pass

            # Sort by risk score (highest first)
            rankings.sort(key=lambda x: x['risk_score'], reverse=True)

            # Identify highest risk sectors
            high_risk = [r for r in rankings if r['risk_score'] >= 30]
            moderate_risk = [r for r in rankings if 15 <= r['risk_score'] < 30]
            low_risk = [r for r in rankings if r['risk_score'] < 15]

            return {
                'rankings': rankings,
                'highest_risk': rankings[0]['ticker'] if rankings else None,
                'lowest_risk': rankings[-1]['ticker'] if rankings else None,
                'high_risk_count': len(high_risk),
                'high_risk_sectors': [r['ticker'] for r in high_risk],
                'moderate_risk_sectors': [r['ticker'] for r in moderate_risk],
                'low_risk_sectors': [r['ticker'] for r in low_risk],
                'avg_risk_score': round(sum(r['risk_score'] for r in rankings) / len(rankings), 1) if rankings else 0
            }

        except Exception as e:
            return {
                'status': 'ERROR',
                'description': f'Error: {str(e)}'
            }

    def get_sector_risk_report(self) -> str:
        """
        Generate formatted sector risk ranking report.

        Returns:
            Multi-line string with sector rankings
        """
        ranking = self.get_sector_risk_ranking()

        nl = chr(10)
        lines = []
        lines.append('=' * 60)
        lines.append('SECTOR RISK RANKING')
        lines.append('=' * 60)
        lines.append('')

        if 'rankings' not in ranking:
            lines.append(f"Error: {ranking.get('description', 'Unknown error')}")
            return nl.join(lines)

        lines.append(f"Average Risk Score: {ranking['avg_risk_score']:.1f}")
        lines.append(f"High Risk Sectors: {ranking['high_risk_count']}")
        lines.append('')

        lines.append('Sector Rankings (highest risk first):')
        lines.append('-' * 50)
        lines.append(f"{'Sector':<25} {'Risk':>8} {'5d':>8} {'20d':>8}")
        lines.append('-' * 50)

        for r in ranking['rankings']:
            risk_indicator = '!!!' if r['risk_score'] >= 30 else '!!' if r['risk_score'] >= 15 else ''
            lines.append(f"{r['name']:<25} {r['risk_score']:>6.1f}{risk_indicator:>2} {r['perf_5d']:>+7.1f}% {r['perf_20d']:>+7.1f}%")

        lines.append('')

        if ranking['high_risk_sectors']:
            lines.append('HIGH RISK: ' + ', '.join(ranking['high_risk_sectors']))
        if ranking['low_risk_sectors']:
            lines.append('LOW RISK: ' + ', '.join(ranking['low_risk_sectors']))

        return nl.join(lines)

    def get_monitoring_dashboard(self) -> str:
        """
        Generate real-time monitoring dashboard.

        Combines intraday monitoring, recovery status, and sector risk.

        Returns:
            Multi-line string with monitoring dashboard
        """
        monitor = self.get_intraday_monitor()
        recovery = self.detect_recovery()
        signal = self.detect()

        nl = chr(10)
        lines = []

        lines.append('+' + '=' * 58 + '+')
        lines.append('|' + ' BEAR DETECTION MONITORING DASHBOARD '.center(58) + '|')
        lines.append('+' + '=' * 58 + '+')
        lines.append('')

        # Current status
        status_icon = {
            'NORMAL': '[OK]',
            'WATCH': '[!]',
            'WARNING': '[!!]',
            'CRITICAL': '[!!!]'
        }
        icon = status_icon.get(signal.alert_level, '[?]')

        lines.append(f"  Status: {icon} {signal.alert_level}  |  Bear Score: {signal.bear_score:.1f}/100")
        lines.append(f"  Crash Probability: {signal.crash_probability:.1f}%  |  Vol Regime: {signal.vol_regime}")
        lines.append('')

        # Intraday section
        lines.append('  INTRADAY:')
        intraday_icon = {'ALERT': '[!]', 'WATCH': '[*]', 'IMPROVING': '[+]', 'STABLE': '[-]'}
        lines.append(f"    Status: {intraday_icon.get(monitor['status'], '[-]')} {monitor['status']}")
        lines.append(f"    Change Today: {monitor['change_today']:+.1f}  |  Range: {monitor['low_of_day']:.1f} - {monitor['high_of_day']:.1f}")

        if monitor['intraday_alerts']:
            lines.append(f"    Alerts: {', '.join(monitor['intraday_alerts'])}")
        lines.append('')

        # Recovery section
        lines.append('  RECOVERY STATUS:')
        recovery_icon = {
            'STRONG_RECOVERY': '[++]',
            'RECOVERING': '[+]',
            'EARLY_RECOVERY': '[~]',
            'STILL_ELEVATED': '[!]',
            'NORMAL': '[-]'
        }
        lines.append(f"    Status: {recovery_icon.get(recovery['status'], '[-]')} {recovery['status']}")
        lines.append(f"    Recovery Score: {recovery['recovery_score']}/100")

        if recovery['recovery_signals']:
            lines.append(f"    Signals: {', '.join(recovery['recovery_signals'][:3])}")
        lines.append('')

        # Action
        lines.append(f"  ACTION: {monitor['action']}")
        lines.append('')
        lines.append('+' + '=' * 58 + '+')

        return nl.join(lines)

'''

def main():
    with open('src/analysis/fast_bear_detector.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Find insertion point - before the standalone functions
    insertion_point = 'def get_fast_bear_signal() -> FastBearSignal:'

    if insertion_point in content:
        content = content.replace(insertion_point, MONITORING_METHODS + '\n' + insertion_point)
        print("Added monitoring, recovery, and sector risk methods")
    else:
        print("Could not find insertion point")
        return

    with open('src/analysis/fast_bear_detector.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Done")

if __name__ == '__main__':
    main()

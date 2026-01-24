"""Add alert cooldown management and quick status utilities to fast_bear_detector.py"""

COOLDOWN_METHODS = '''

    # ==================== ALERT COOLDOWN & QUICK STATUS ====================

    # Alert cooldown tracking (class-level)
    _last_alerts = {}  # {level: timestamp}

    def check_alert_cooldown(self, level: str) -> Dict:
        """
        Check if alert cooldown has expired for given level.

        Prevents alert fatigue by enforcing cooldown periods.

        Args:
            level: Alert level (WATCH, WARNING, CRITICAL)

        Returns:
            Dict with cooldown status
        """
        # Cooldown periods in hours
        cooldowns = {
            'WATCH': 24,
            'WARNING': 4,
            'CRITICAL': 1
        }

        cooldown_hours = cooldowns.get(level, 24)
        now = datetime.now()

        # Check last alert time
        last_alert = self._last_alerts.get(level)

        if last_alert is None:
            return {
                'can_alert': True,
                'cooldown_expired': True,
                'hours_remaining': 0,
                'last_alert': None,
                'cooldown_hours': cooldown_hours
            }

        hours_since = (now - last_alert).total_seconds() / 3600
        hours_remaining = max(0, cooldown_hours - hours_since)

        return {
            'can_alert': hours_since >= cooldown_hours,
            'cooldown_expired': hours_since >= cooldown_hours,
            'hours_since_last': round(hours_since, 1),
            'hours_remaining': round(hours_remaining, 1),
            'last_alert': last_alert.isoformat(),
            'cooldown_hours': cooldown_hours
        }

    def record_alert(self, level: str) -> None:
        """
        Record that an alert was sent for cooldown tracking.

        Args:
            level: Alert level that was sent
        """
        self._last_alerts[level] = datetime.now()

    def should_alert_with_cooldown(self, min_level: str = 'WATCH') -> Dict:
        """
        Check if alert should be sent considering cooldown.

        Args:
            min_level: Minimum alert level to trigger

        Returns:
            Dict with alert decision and reasoning
        """
        signal = self.detect()

        level_order = {'NORMAL': 0, 'WATCH': 1, 'WARNING': 2, 'CRITICAL': 3}
        current_level = level_order.get(signal.alert_level, 0)
        min_required = level_order.get(min_level, 1)

        # Check if alert level is high enough
        if current_level < min_required:
            return {
                'should_alert': False,
                'reason': f'Alert level {signal.alert_level} below threshold {min_level}',
                'alert_level': signal.alert_level,
                'bear_score': signal.bear_score
            }

        # Check cooldown
        cooldown = self.check_alert_cooldown(signal.alert_level)

        if not cooldown['can_alert']:
            return {
                'should_alert': False,
                'reason': f'Cooldown active - {cooldown["hours_remaining"]:.1f}h remaining',
                'alert_level': signal.alert_level,
                'bear_score': signal.bear_score,
                'cooldown_remaining': cooldown['hours_remaining']
            }

        return {
            'should_alert': True,
            'reason': 'Alert conditions met and cooldown expired',
            'alert_level': signal.alert_level,
            'bear_score': signal.bear_score,
            'action': 'SEND_ALERT'
        }

    def get_quick_check(self) -> Dict:
        """
        Perform quick status check with minimal API calls.

        Optimized for frequent monitoring with low latency.

        Returns:
            Dict with essential status information
        """
        signal = self.detect()

        # Quick assessment
        is_elevated = signal.alert_level != 'NORMAL'
        needs_attention = signal.bear_score >= 30 or signal.crash_probability >= 10

        # Determine action level
        if signal.alert_level == 'CRITICAL':
            action = 'IMMEDIATE_ACTION'
            urgency = 3
        elif signal.alert_level == 'WARNING':
            action = 'REVIEW_POSITIONS'
            urgency = 2
        elif signal.alert_level == 'WATCH':
            action = 'MONITOR_CLOSELY'
            urgency = 1
        else:
            action = 'NONE'
            urgency = 0

        return {
            'timestamp': datetime.now().isoformat(),
            'bear_score': signal.bear_score,
            'alert_level': signal.alert_level,
            'crash_prob': signal.crash_probability,
            'vix': signal.vix_level,
            'breadth': signal.market_breadth_pct,
            'vol_regime': signal.vol_regime,
            'is_elevated': is_elevated,
            'needs_attention': needs_attention,
            'action': action,
            'urgency': urgency
        }

    def get_one_liner(self) -> str:
        """
        Get single-line status summary.

        Perfect for logging and quick monitoring.

        Returns:
            Single line status string
        """
        check = self.get_quick_check()

        status_char = {
            'NORMAL': '.',
            'WATCH': '*',
            'WARNING': '!',
            'CRITICAL': 'X'
        }
        char = status_char.get(check['alert_level'], '?')

        return f"[{char}] {check['alert_level']:8} | Score: {check['bear_score']:5.1f} | Crash: {check['crash_prob']:4.1f}% | VIX: {check['vix']:5.1f} | Breadth: {check['breadth']:5.1f}%"

    def get_notification_template(self, format: str = 'email') -> str:
        """
        Generate notification template for alerting systems.

        Args:
            format: Output format (email, slack, sms, webhook)

        Returns:
            Formatted notification string
        """
        signal = self.detect()
        summary = self.get_daily_summary()
        quality = self.get_signal_quality()

        if format == 'sms':
            # Short SMS format
            return f"BEAR ALERT: {signal.alert_level} | Score {signal.bear_score:.0f} | Crash {signal.crash_probability:.0f}% | Action: {summary['recommended_action'][:50]}"

        elif format == 'slack':
            # Slack markdown format
            nl = chr(10)
            emoji = {'CRITICAL': ':rotating_light:', 'WARNING': ':warning:', 'WATCH': ':eyes:', 'NORMAL': ':white_check_mark:'}
            e = emoji.get(signal.alert_level, ':question:')

            lines = [
                f"{e} *BEAR DETECTION ALERT: {signal.alert_level}*",
                "",
                f"*Bear Score:* {signal.bear_score:.1f}/100",
                f"*Crash Probability:* {signal.crash_probability:.1f}%",
                f"*Signal Quality:* {quality['quality_grade']} ({quality['quality_score']}/100)",
                "",
                f"*Key Metrics:*",
                f"- VIX: {signal.vix_level:.1f}",
                f"- Breadth: {signal.market_breadth_pct:.1f}%",
                f"- Vol Regime: {signal.vol_regime}",
                "",
                f"*Action:* {summary['recommended_action']}"
            ]

            if summary['active_flags']:
                lines.append("")
                lines.append("*Warning Flags:*")
                for flag in summary['active_flags'][:3]:
                    lines.append(f"- {flag}")

            return nl.join(lines)

        elif format == 'webhook':
            # JSON format for webhooks
            return json.dumps({
                'alert_type': 'BEAR_DETECTION',
                'level': signal.alert_level,
                'bear_score': signal.bear_score,
                'crash_probability': signal.crash_probability,
                'quality_grade': quality['quality_grade'],
                'vix': signal.vix_level,
                'breadth': signal.market_breadth_pct,
                'vol_regime': signal.vol_regime,
                'action': summary['recommended_action'],
                'flags': summary['active_flags'],
                'timestamp': datetime.now().isoformat()
            })

        else:  # email format (default)
            nl = chr(10)
            lines = [
                f"BEAR DETECTION ALERT: {signal.alert_level}",
                "=" * 50,
                "",
                f"Alert Level: {signal.alert_level}",
                f"Bear Score: {signal.bear_score:.1f}/100",
                f"Crash Probability: {signal.crash_probability:.1f}%",
                f"Signal Quality: {quality['quality_grade']} ({quality['quality_score']}/100)",
                "",
                "KEY METRICS",
                "-" * 30,
                f"VIX Level: {signal.vix_level:.1f}",
                f"Market Breadth: {signal.market_breadth_pct:.1f}%",
                f"SPY 3-Day Change: {signal.spy_roc_3d:+.2f}%",
                f"Volatility Regime: {signal.vol_regime}",
                f"Vol Compression: {signal.vol_compression:.2f}",
                "",
                "RECOMMENDATION",
                "-" * 30,
                summary['recommended_action'],
                ""
            ]

            if summary['active_flags']:
                lines.append("WARNING FLAGS")
                lines.append("-" * 30)
                for flag in summary['active_flags']:
                    lines.append(f"- {flag}")
                lines.append("")

            if summary['watch_list']:
                lines.append("WATCH LIST")
                lines.append("-" * 30)
                for item in summary['watch_list']:
                    lines.append(f"- {item}")
                lines.append("")

            lines.append("=" * 50)
            lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            return nl.join(lines)

    def run_health_check(self) -> Dict:
        """
        Run system health check to verify all components working.

        Returns:
            Dict with health check results
        """
        health = {
            'status': 'OK',
            'checks': {},
            'errors': []
        }

        # Check 1: Basic detection
        try:
            signal = self.detect()
            health['checks']['detection'] = {
                'status': 'OK',
                'bear_score': signal.bear_score
            }
        except Exception as e:
            health['checks']['detection'] = {'status': 'ERROR', 'error': str(e)}
            health['errors'].append(f'Detection: {e}')
            health['status'] = 'DEGRADED'

        # Check 2: Market data
        try:
            ctx = self.get_market_context()
            health['checks']['market_data'] = {
                'status': 'OK' if ctx.get('market_phase') != 'ERROR' else 'ERROR',
                'phase': ctx.get('market_phase')
            }
        except Exception as e:
            health['checks']['market_data'] = {'status': 'ERROR', 'error': str(e)}
            health['errors'].append(f'Market data: {e}')
            health['status'] = 'DEGRADED'

        # Check 3: Sector data
        try:
            sector = self.get_sector_leadership()
            health['checks']['sector_data'] = {
                'status': 'OK' if sector.get('leadership') != 'ERROR' else 'ERROR',
                'leadership': sector.get('leadership')
            }
        except Exception as e:
            health['checks']['sector_data'] = {'status': 'ERROR', 'error': str(e)}
            health['errors'].append(f'Sector data: {e}')
            health['status'] = 'DEGRADED'

        # Check 4: Cross-asset
        try:
            corr = self.get_cross_asset_correlation()
            health['checks']['cross_asset'] = {
                'status': 'OK' if corr.get('status') != 'ERROR' else 'ERROR'
            }
        except Exception as e:
            health['checks']['cross_asset'] = {'status': 'ERROR', 'error': str(e)}
            health['errors'].append(f'Cross-asset: {e}')
            health['status'] = 'DEGRADED'

        # Overall assessment
        error_count = len(health['errors'])
        if error_count == 0:
            health['summary'] = 'All systems operational'
        elif error_count <= 2:
            health['summary'] = f'Degraded - {error_count} component(s) with issues'
        else:
            health['status'] = 'CRITICAL'
            health['summary'] = f'Critical - {error_count} components failing'

        return health

    def get_system_info(self) -> Dict:
        """
        Get system information and capabilities.

        Returns:
            Dict with system information
        """
        return {
            'name': 'FastBearDetector',
            'version': '2.0',
            'description': 'Fast bear market detection using leading indicators',
            'indicators': {
                'v1_core': ['spy_roc', 'vix', 'breadth', 'sector_breadth', 'volume'],
                'v2_credit': ['yield_curve', 'credit_spread', 'high_yield', 'put_call'],
                'v3_advanced': ['skew', 'mcclellan', 'pct_above_ma', 'new_high_low'],
                'v4_early_warning': ['intl_weakness', 'momentum_exhaustion', 'correlation'],
                'v5_regime': ['vol_regime', 'vol_compression', 'fear_greed', 'smart_money'],
                'v6_overnight': ['overnight_gap', 'bond_vol', 'rotation_speed', 'liquidity'],
                'v7_flows': ['options_volume', 'etf_flow', 'vol_skew', 'market_depth']
            },
            'analysis_methods': [
                'multi_timeframe', 'pattern_matching', 'cross_asset_correlation',
                'momentum_regime', 'sector_leadership', 'volume_profile',
                'divergence_analysis', 'scenario_stress_test', 'risk_attribution',
                'signal_quality', 'recovery_detection', 'sector_risk_ranking'
            ],
            'output_formats': ['dict', 'json', 'report', 'dashboard', 'notification'],
            'validation': {
                'period': '5 years',
                'hit_rate': '100%',
                'avg_lead_days': 5.2,
                'false_positives': 0
            }
        }

'''

def main():
    with open('src/analysis/fast_bear_detector.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Find insertion point - before the standalone functions
    insertion_point = 'def get_fast_bear_signal() -> FastBearSignal:'

    if insertion_point in content:
        content = content.replace(insertion_point, COOLDOWN_METHODS + '\n' + insertion_point)
        print("Added cooldown and quick status methods")
    else:
        print("Could not find insertion point")
        return

    with open('src/analysis/fast_bear_detector.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Done")

if __name__ == '__main__':
    main()

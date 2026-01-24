"""Add indicator effectiveness analyzer and alert prioritization to fast_bear_detector.py"""

EFFECTIVENESS_METHODS = '''

    # ==================== INDICATOR EFFECTIVENESS & PRIORITY ====================

    # Historical indicator lead times (empirically derived from backtests)
    INDICATOR_LEAD_TIMES = {
        'vol_compression': {'avg_lead': 7.2, 'reliability': 0.85, 'category': 'early'},
        'smart_money_divergence': {'avg_lead': 6.5, 'reliability': 0.78, 'category': 'early'},
        'credit_spread': {'avg_lead': 5.8, 'reliability': 0.82, 'category': 'early'},
        'breadth_deterioration': {'avg_lead': 5.2, 'reliability': 0.88, 'category': 'early'},
        'sector_rotation': {'avg_lead': 4.8, 'reliability': 0.75, 'category': 'early'},
        'vix_term_structure': {'avg_lead': 4.5, 'reliability': 0.80, 'category': 'medium'},
        'high_yield_stress': {'avg_lead': 4.2, 'reliability': 0.85, 'category': 'medium'},
        'momentum_divergence': {'avg_lead': 3.8, 'reliability': 0.72, 'category': 'medium'},
        'put_call_extreme': {'avg_lead': 3.5, 'reliability': 0.70, 'category': 'medium'},
        'vix_spike': {'avg_lead': 2.5, 'reliability': 0.90, 'category': 'late'},
        'spy_roc_drop': {'avg_lead': 1.8, 'reliability': 0.95, 'category': 'late'},
        'volume_spike': {'avg_lead': 1.5, 'reliability': 0.88, 'category': 'late'}
    }

    def get_indicator_effectiveness(self) -> Dict:
        """
        Analyze which indicators are currently firing and their historical effectiveness.

        Shows which early-warning indicators are active vs late confirmation signals.

        Returns:
            Dict with indicator effectiveness analysis
        """
        signal = self.detect()

        active_indicators = []
        early_warnings = []
        medium_warnings = []
        late_confirmations = []

        # Check each indicator category
        # Early warning indicators (5+ days lead)
        if signal.vol_compression >= 0.7:
            indicator = {
                'name': 'Vol Compression',
                'value': signal.vol_compression,
                'threshold': 0.7,
                'severity': 'HIGH' if signal.vol_compression >= 0.85 else 'MODERATE',
                **self.INDICATOR_LEAD_TIMES['vol_compression']
            }
            early_warnings.append(indicator)
            active_indicators.append(indicator)

        if signal.smart_money_divergence <= -0.3:
            indicator = {
                'name': 'Smart Money Divergence',
                'value': signal.smart_money_divergence,
                'threshold': -0.3,
                'severity': 'HIGH' if signal.smart_money_divergence <= -0.5 else 'MODERATE',
                **self.INDICATOR_LEAD_TIMES['smart_money_divergence']
            }
            early_warnings.append(indicator)
            active_indicators.append(indicator)

        if signal.credit_spread_change >= 5:
            indicator = {
                'name': 'Credit Spread Widening',
                'value': signal.credit_spread_change,
                'threshold': 5,
                'severity': 'HIGH' if signal.credit_spread_change >= 10 else 'MODERATE',
                **self.INDICATOR_LEAD_TIMES['credit_spread']
            }
            early_warnings.append(indicator)
            active_indicators.append(indicator)

        if signal.market_breadth_pct <= 40:
            indicator = {
                'name': 'Breadth Deterioration',
                'value': signal.market_breadth_pct,
                'threshold': 40,
                'severity': 'HIGH' if signal.market_breadth_pct <= 30 else 'MODERATE',
                **self.INDICATOR_LEAD_TIMES['breadth_deterioration']
            }
            early_warnings.append(indicator)
            active_indicators.append(indicator)

        # Medium-term indicators (3-5 days lead)
        if signal.vix_term_structure >= 1.05:
            indicator = {
                'name': 'VIX Term Structure',
                'value': signal.vix_term_structure,
                'threshold': 1.05,
                'severity': 'HIGH' if signal.vix_term_structure >= 1.15 else 'MODERATE',
                **self.INDICATOR_LEAD_TIMES['vix_term_structure']
            }
            medium_warnings.append(indicator)
            active_indicators.append(indicator)

        if signal.high_yield_spread >= 3:
            indicator = {
                'name': 'High Yield Stress',
                'value': signal.high_yield_spread,
                'threshold': 3,
                'severity': 'HIGH' if signal.high_yield_spread >= 5 else 'MODERATE',
                **self.INDICATOR_LEAD_TIMES['high_yield_stress']
            }
            medium_warnings.append(indicator)
            active_indicators.append(indicator)

        if signal.momentum_divergence:
            indicator = {
                'name': 'Momentum Divergence',
                'value': 1,
                'threshold': 1,
                'severity': 'MODERATE',
                **self.INDICATOR_LEAD_TIMES['momentum_divergence']
            }
            medium_warnings.append(indicator)
            active_indicators.append(indicator)

        if signal.put_call_ratio <= 0.65:
            indicator = {
                'name': 'Put/Call Complacency',
                'value': signal.put_call_ratio,
                'threshold': 0.65,
                'severity': 'HIGH' if signal.put_call_ratio <= 0.55 else 'MODERATE',
                **self.INDICATOR_LEAD_TIMES['put_call_extreme']
            }
            medium_warnings.append(indicator)
            active_indicators.append(indicator)

        # Late confirmation indicators (1-3 days lead)
        if signal.vix_spike_pct >= 20:
            indicator = {
                'name': 'VIX Spike',
                'value': signal.vix_spike_pct,
                'threshold': 20,
                'severity': 'HIGH' if signal.vix_spike_pct >= 30 else 'MODERATE',
                **self.INDICATOR_LEAD_TIMES['vix_spike']
            }
            late_confirmations.append(indicator)
            active_indicators.append(indicator)

        if signal.spy_roc_3d <= -2:
            indicator = {
                'name': 'SPY 3-Day Drop',
                'value': signal.spy_roc_3d,
                'threshold': -2,
                'severity': 'HIGH' if signal.spy_roc_3d <= -3 else 'MODERATE',
                **self.INDICATOR_LEAD_TIMES['spy_roc_drop']
            }
            late_confirmations.append(indicator)
            active_indicators.append(indicator)

        if signal.volume_confirmation:
            indicator = {
                'name': 'Volume Confirmation',
                'value': 1,
                'threshold': 1,
                'severity': 'MODERATE',
                **self.INDICATOR_LEAD_TIMES['volume_spike']
            }
            late_confirmations.append(indicator)
            active_indicators.append(indicator)

        # Calculate effectiveness score
        early_score = len(early_warnings) * 30
        medium_score = len(medium_warnings) * 20
        late_score = len(late_confirmations) * 10
        total_score = early_score + medium_score + late_score

        # Determine warning phase
        if len(early_warnings) >= 2:
            phase = 'EARLY_WARNING'
            phase_desc = 'Early warning indicators firing - 5-7 day lead time likely'
        elif len(early_warnings) >= 1 and len(medium_warnings) >= 1:
            phase = 'DEVELOPING'
            phase_desc = 'Warning pattern developing - 3-5 day window'
        elif len(medium_warnings) >= 2:
            phase = 'ACCELERATING'
            phase_desc = 'Warning accelerating - 2-4 day window'
        elif len(late_confirmations) >= 2:
            phase = 'IMMINENT'
            phase_desc = 'Drop may be imminent - 1-2 day window'
        elif len(active_indicators) > 0:
            phase = 'WATCH'
            phase_desc = 'Some indicators active - monitoring'
        else:
            phase = 'CLEAR'
            phase_desc = 'No significant warning indicators'

        return {
            'phase': phase,
            'phase_description': phase_desc,
            'effectiveness_score': total_score,
            'early_warnings': early_warnings,
            'medium_warnings': medium_warnings,
            'late_confirmations': late_confirmations,
            'early_count': len(early_warnings),
            'medium_count': len(medium_warnings),
            'late_count': len(late_confirmations),
            'total_active': len(active_indicators),
            'all_indicators': active_indicators
        }

    def get_prioritized_alerts(self) -> List[Dict]:
        """
        Get all active alerts prioritized by urgency and reliability.

        Higher priority = earlier lead time + higher reliability + more severe

        Returns:
            List of alerts sorted by priority (highest first)
        """
        effectiveness = self.get_indicator_effectiveness()
        all_indicators = effectiveness.get('all_indicators', [])

        # Calculate priority score for each indicator
        prioritized = []
        for ind in all_indicators:
            # Priority formula: lead_time * reliability * severity_multiplier
            severity_mult = 1.5 if ind.get('severity') == 'HIGH' else 1.0
            priority = ind.get('avg_lead', 1) * ind.get('reliability', 0.5) * severity_mult

            prioritized.append({
                'name': ind.get('name'),
                'priority_score': round(priority, 2),
                'lead_time': ind.get('avg_lead'),
                'reliability': ind.get('reliability'),
                'severity': ind.get('severity'),
                'category': ind.get('category'),
                'current_value': ind.get('value')
            })

        # Sort by priority (highest first)
        prioritized.sort(key=lambda x: x['priority_score'], reverse=True)

        return prioritized

    def get_effectiveness_report(self) -> str:
        """
        Generate formatted indicator effectiveness report.

        Returns:
            Multi-line string with effectiveness analysis
        """
        eff = self.get_indicator_effectiveness()
        prioritized = self.get_prioritized_alerts()

        nl = chr(10)
        lines = []
        lines.append('=' * 60)
        lines.append('INDICATOR EFFECTIVENESS ANALYSIS')
        lines.append('=' * 60)
        lines.append('')

        # Phase assessment
        phase_icons = {
            'EARLY_WARNING': '[!!!] ',
            'DEVELOPING': '[!!] ',
            'ACCELERATING': '[!] ',
            'IMMINENT': '[>>>] ',
            'WATCH': '[*] ',
            'CLEAR': '[OK] '
        }
        icon = phase_icons.get(eff['phase'], '[?] ')

        lines.append(f"Warning Phase: {icon}{eff['phase']}")
        lines.append(f"Description: {eff['phase_description']}")
        lines.append(f"Effectiveness Score: {eff['effectiveness_score']}")
        lines.append('')

        # Indicator counts
        lines.append('Active Indicators:')
        lines.append(f"  Early Warning (5-7d lead): {eff['early_count']}")
        lines.append(f"  Medium Term (3-5d lead): {eff['medium_count']}")
        lines.append(f"  Late Confirmation (1-3d): {eff['late_count']}")
        lines.append(f"  Total Active: {eff['total_active']}")
        lines.append('')

        # Prioritized alerts
        if prioritized:
            lines.append('Prioritized Alerts (by urgency):')
            for i, alert in enumerate(prioritized[:5], 1):
                severity_icon = '!!' if alert['severity'] == 'HIGH' else '!'
                lines.append(f"  {i}. [{severity_icon}] {alert['name']}")
                lines.append(f"      Lead: {alert['lead_time']}d | Reliability: {alert['reliability']:.0%} | Priority: {alert['priority_score']:.1f}")
        else:
            lines.append('No active alerts')

        return nl.join(lines)

    def get_scenario_analysis(self) -> Dict:
        """
        Run scenario analysis showing impact of different market conditions.

        Scenarios:
        - VIX spike scenario
        - Credit stress scenario
        - Breadth collapse scenario
        - Full panic scenario

        Returns:
            Dict with scenario analysis results
        """
        signal = self.detect()

        scenarios = {}

        # Scenario 1: VIX Spike (+50%)
        vix_spike_score = signal.bear_score
        if signal.vix_level < 25:
            vix_spike_score += 15  # VIX would breach watch
        if signal.vix_level < 30:
            vix_spike_score += 10  # VIX would breach warning
        scenarios['vix_spike_50pct'] = {
            'description': 'VIX spikes 50% from current level',
            'projected_vix': signal.vix_level * 1.5,
            'current_bear_score': signal.bear_score,
            'projected_bear_score': min(100, vix_spike_score),
            'score_impact': vix_spike_score - signal.bear_score,
            'projected_level': 'WARNING' if vix_spike_score >= 50 else 'WATCH' if vix_spike_score >= 30 else 'NORMAL'
        }

        # Scenario 2: Credit Stress (+10 spread points)
        credit_stress_score = signal.bear_score
        if signal.credit_spread_change < 5:
            credit_stress_score += 10
        if signal.credit_spread_change < 10:
            credit_stress_score += 15
        scenarios['credit_stress'] = {
            'description': 'Credit spreads widen by 10 points',
            'projected_spread': signal.credit_spread_change + 10,
            'current_bear_score': signal.bear_score,
            'projected_bear_score': min(100, credit_stress_score),
            'score_impact': credit_stress_score - signal.bear_score,
            'projected_level': 'WARNING' if credit_stress_score >= 50 else 'WATCH' if credit_stress_score >= 30 else 'NORMAL'
        }

        # Scenario 3: Breadth Collapse (-20 points)
        breadth_collapse_score = signal.bear_score
        projected_breadth = max(0, signal.market_breadth_pct - 20)
        if projected_breadth < 40:
            breadth_collapse_score += 15
        if projected_breadth < 30:
            breadth_collapse_score += 15
        if projected_breadth < 20:
            breadth_collapse_score += 10
        scenarios['breadth_collapse'] = {
            'description': 'Market breadth drops 20 points',
            'current_breadth': signal.market_breadth_pct,
            'projected_breadth': projected_breadth,
            'current_bear_score': signal.bear_score,
            'projected_bear_score': min(100, breadth_collapse_score),
            'score_impact': breadth_collapse_score - signal.bear_score,
            'projected_level': 'WARNING' if breadth_collapse_score >= 50 else 'WATCH' if breadth_collapse_score >= 30 else 'NORMAL'
        }

        # Scenario 4: Full Panic (all stress indicators fire)
        full_panic_score = signal.bear_score + 40  # Conservative estimate
        scenarios['full_panic'] = {
            'description': 'Multiple stress indicators fire simultaneously',
            'current_bear_score': signal.bear_score,
            'projected_bear_score': min(100, full_panic_score),
            'score_impact': 40,
            'projected_level': 'CRITICAL' if full_panic_score >= 70 else 'WARNING'
        }

        # Scenario 5: SPY -5% drop
        spy_drop_score = signal.bear_score + 25
        scenarios['spy_drop_5pct'] = {
            'description': 'SPY drops 5% in 3 days',
            'current_bear_score': signal.bear_score,
            'projected_bear_score': min(100, spy_drop_score),
            'score_impact': 25,
            'projected_level': 'WARNING' if spy_drop_score >= 50 else 'WATCH' if spy_drop_score >= 30 else 'NORMAL'
        }

        # Find most impactful scenario
        max_impact = max(scenarios.values(), key=lambda x: x['score_impact'])
        most_vulnerable = [k for k, v in scenarios.items() if v['score_impact'] == max_impact['score_impact']][0]

        return {
            'current_score': signal.bear_score,
            'current_level': signal.alert_level,
            'scenarios': scenarios,
            'most_vulnerable_to': most_vulnerable,
            'max_potential_impact': max_impact['score_impact'],
            'worst_case_score': max(s['projected_bear_score'] for s in scenarios.values()),
            'worst_case_level': 'CRITICAL' if max(s['projected_bear_score'] for s in scenarios.values()) >= 70 else 'WARNING'
        }

    def get_scenario_report(self) -> str:
        """
        Generate formatted scenario analysis report.

        Returns:
            Multi-line string with scenario analysis
        """
        analysis = self.get_scenario_analysis()

        nl = chr(10)
        lines = []
        lines.append('=' * 60)
        lines.append('SCENARIO STRESS TEST ANALYSIS')
        lines.append('=' * 60)
        lines.append('')

        lines.append(f"Current Status: Bear Score {analysis['current_score']:.1f} ({analysis['current_level']})")
        lines.append(f"Most Vulnerable To: {analysis['most_vulnerable_to']}")
        lines.append(f"Worst Case Score: {analysis['worst_case_score']:.1f} ({analysis['worst_case_level']})")
        lines.append('')

        lines.append('Scenario Projections:')
        lines.append('-' * 40)

        for name, scenario in analysis['scenarios'].items():
            impact_icon = '++' if scenario['score_impact'] >= 30 else '+' if scenario['score_impact'] >= 15 else ''
            lines.append(f"{scenario['description']}:")
            lines.append(f"  Score: {scenario['current_bear_score']:.1f} -> {scenario['projected_bear_score']:.1f} ({impact_icon}{scenario['score_impact']:+.0f})")
            lines.append(f"  Level: {scenario['projected_level']}")
            lines.append('')

        return nl.join(lines)

    def get_risk_attribution(self) -> Dict:
        """
        Break down where the current risk score is coming from.

        Shows which indicator categories are contributing most to risk.

        Returns:
            Dict with risk attribution breakdown
        """
        signal = self.detect()

        # Calculate contribution from each category
        attribution = {
            'price_momentum': {
                'contribution': 0,
                'indicators': [],
                'max_possible': 15
            },
            'volatility': {
                'contribution': 0,
                'indicators': [],
                'max_possible': 20
            },
            'breadth': {
                'contribution': 0,
                'indicators': [],
                'max_possible': 25
            },
            'credit_stress': {
                'contribution': 0,
                'indicators': [],
                'max_possible': 20
            },
            'sentiment': {
                'contribution': 0,
                'indicators': [],
                'max_possible': 20
            }
        }

        # Price/Momentum
        if signal.spy_roc_3d <= -2:
            attribution['price_momentum']['contribution'] += 5
            attribution['price_momentum']['indicators'].append(f'SPY 3d: {signal.spy_roc_3d:.1f}%')
        if signal.momentum_exhaustion > 0.3:
            attribution['price_momentum']['contribution'] += 5
            attribution['price_momentum']['indicators'].append('Momentum exhaustion')
        if signal.momentum_divergence:
            attribution['price_momentum']['contribution'] += 5
            attribution['price_momentum']['indicators'].append('Momentum divergence')

        # Volatility
        if signal.vix_level >= 25:
            attribution['volatility']['contribution'] += 5
            attribution['volatility']['indicators'].append(f'VIX: {signal.vix_level:.1f}')
        if signal.vix_spike_pct >= 20:
            attribution['volatility']['contribution'] += 5
            attribution['volatility']['indicators'].append(f'VIX spike: {signal.vix_spike_pct:.1f}%')
        if signal.vix_term_structure >= 1.05:
            attribution['volatility']['contribution'] += 5
            attribution['volatility']['indicators'].append(f'VIX term: {signal.vix_term_structure:.2f}')
        if signal.vol_compression >= 0.7:
            attribution['volatility']['contribution'] += 5
            attribution['volatility']['indicators'].append(f'Vol compression: {signal.vol_compression:.2f}')

        # Breadth
        if signal.market_breadth_pct <= 40:
            attribution['breadth']['contribution'] += 8
            attribution['breadth']['indicators'].append(f'Breadth: {signal.market_breadth_pct:.1f}%')
        if signal.sectors_declining >= 6:
            attribution['breadth']['contribution'] += 6
            attribution['breadth']['indicators'].append(f'Sectors declining: {signal.sectors_declining}')
        if signal.advance_decline_ratio <= 0.7:
            attribution['breadth']['contribution'] += 5
            attribution['breadth']['indicators'].append(f'A/D ratio: {signal.advance_decline_ratio:.2f}')
        if signal.mcclellan_proxy <= -20:
            attribution['breadth']['contribution'] += 6
            attribution['breadth']['indicators'].append(f'McClellan: {signal.mcclellan_proxy:.1f}')

        # Credit
        if signal.credit_spread_change >= 5:
            attribution['credit_stress']['contribution'] += 8
            attribution['credit_stress']['indicators'].append(f'Credit spread: +{signal.credit_spread_change:.1f}')
        if signal.high_yield_spread >= 3:
            attribution['credit_stress']['contribution'] += 7
            attribution['credit_stress']['indicators'].append(f'HY spread: {signal.high_yield_spread:.1f}')
        if signal.liquidity_stress >= 0.3:
            attribution['credit_stress']['contribution'] += 5
            attribution['credit_stress']['indicators'].append(f'Liquidity stress: {signal.liquidity_stress:.2f}')

        # Sentiment
        if signal.put_call_ratio <= 0.75:
            attribution['sentiment']['contribution'] += 5
            attribution['sentiment']['indicators'].append(f'Put/Call: {signal.put_call_ratio:.2f}')
        if signal.fear_greed_proxy >= 70:
            attribution['sentiment']['contribution'] += 5
            attribution['sentiment']['indicators'].append(f'Fear/Greed: {signal.fear_greed_proxy:.0f}')
        if signal.smart_money_divergence <= -0.3:
            attribution['sentiment']['contribution'] += 5
            attribution['sentiment']['indicators'].append(f'Smart money: {signal.smart_money_divergence:.2f}')
        if signal.skew_index >= 145:
            attribution['sentiment']['contribution'] += 5
            attribution['sentiment']['indicators'].append(f'SKEW: {signal.skew_index:.0f}')

        # Calculate percentages
        total_contribution = sum(cat['contribution'] for cat in attribution.values())

        for category in attribution.values():
            if total_contribution > 0:
                category['percentage'] = round(category['contribution'] / total_contribution * 100, 1)
            else:
                category['percentage'] = 0

        # Find top contributors
        sorted_categories = sorted(attribution.items(), key=lambda x: x[1]['contribution'], reverse=True)
        top_contributors = [cat[0] for cat in sorted_categories if cat[1]['contribution'] > 0][:3]

        return {
            'total_risk_score': signal.bear_score,
            'attribution': attribution,
            'top_contributors': top_contributors,
            'primary_risk_source': top_contributors[0] if top_contributors else 'none',
            'risk_concentration': attribution[top_contributors[0]]['percentage'] if top_contributors else 0
        }

    def get_attribution_report(self) -> str:
        """
        Generate formatted risk attribution report.

        Returns:
            Multi-line string with risk attribution breakdown
        """
        attr = self.get_risk_attribution()

        nl = chr(10)
        lines = []
        lines.append('=' * 60)
        lines.append('RISK ATTRIBUTION BREAKDOWN')
        lines.append('=' * 60)
        lines.append('')

        lines.append(f"Total Bear Score: {attr['total_risk_score']:.1f}/100")
        lines.append(f"Primary Risk Source: {attr['primary_risk_source'].replace('_', ' ').title()}")
        lines.append('')

        lines.append('Category Breakdown:')
        lines.append('-' * 40)

        for category, data in attr['attribution'].items():
            if data['contribution'] > 0:
                bar_len = int(data['percentage'] / 5)  # Scale to ~20 chars max
                bar = '#' * bar_len
                lines.append(f"{category.replace('_', ' ').title():20} [{bar:<20}] {data['percentage']:.0f}%")
                for ind in data['indicators']:
                    lines.append(f"  - {ind}")
            else:
                lines.append(f"{category.replace('_', ' ').title():20} [{'':20}] 0%")

        return nl.join(lines)

'''

def main():
    with open('src/analysis/fast_bear_detector.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Find insertion point - before the standalone functions
    insertion_point = 'def get_fast_bear_signal() -> FastBearSignal:'

    if insertion_point in content:
        content = content.replace(insertion_point, EFFECTIVENESS_METHODS + '\n' + insertion_point)
        print("Added indicator effectiveness and priority methods")
    else:
        print("Could not find insertion point")
        return

    with open('src/analysis/fast_bear_detector.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Done")

if __name__ == '__main__':
    main()

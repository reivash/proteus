"""Add signal quality scoring and historical performance tracking to fast_bear_detector.py"""

QUALITY_METHODS = '''

    # ==================== SIGNAL QUALITY & PERFORMANCE ====================

    def get_signal_quality(self) -> Dict:
        """
        Score the quality and reliability of current warning signals.

        Quality factors:
        - Indicator confluence (multiple independent signals)
        - Signal persistence (how long signals have been active)
        - Historical reliability (based on indicator track record)
        - Confirmation strength (late confirmations backing early warnings)

        Returns:
            Dict with signal quality assessment
        """
        signal = self.detect()
        effectiveness = self.get_indicator_effectiveness()
        persistence = self.get_signal_persistence()
        confluence = self.get_confluence_score()

        # Quality scoring (0-100)
        quality_score = 0
        quality_factors = []

        # Factor 1: Indicator confluence (max 30 points)
        groups_firing = confluence.get('groups_firing', 0)
        if groups_firing >= 4:
            quality_score += 30
            quality_factors.append('Strong multi-sector confluence')
        elif groups_firing >= 3:
            quality_score += 20
            quality_factors.append('Good indicator confluence')
        elif groups_firing >= 2:
            quality_score += 10
            quality_factors.append('Moderate confluence')

        # Factor 2: Signal persistence (max 25 points)
        elevated_duration = persistence.get('elevated_duration', 0)
        if elevated_duration >= 5:
            quality_score += 25
            quality_factors.append('Persistent elevated signals')
        elif elevated_duration >= 3:
            quality_score += 15
            quality_factors.append('Building signal persistence')
        elif elevated_duration >= 1:
            quality_score += 5
            quality_factors.append('Recently elevated')

        # Factor 3: Early vs late indicators (max 25 points)
        early_count = effectiveness.get('early_count', 0)
        late_count = effectiveness.get('late_count', 0)

        if early_count >= 2 and late_count >= 1:
            quality_score += 25
            quality_factors.append('Early warnings with confirmation')
        elif early_count >= 2:
            quality_score += 20
            quality_factors.append('Multiple early warnings')
        elif early_count >= 1 and late_count >= 1:
            quality_score += 15
            quality_factors.append('Early and late signals')
        elif early_count >= 1:
            quality_score += 10
            quality_factors.append('Early warning present')
        elif late_count >= 2:
            quality_score += 8
            quality_factors.append('Multiple late confirmations')

        # Factor 4: High-reliability indicators firing (max 20 points)
        high_reliability_count = 0
        for ind in effectiveness.get('all_indicators', []):
            if ind.get('reliability', 0) >= 0.85:
                high_reliability_count += 1

        if high_reliability_count >= 3:
            quality_score += 20
            quality_factors.append('Multiple high-reliability signals')
        elif high_reliability_count >= 2:
            quality_score += 12
            quality_factors.append('Two high-reliability signals')
        elif high_reliability_count >= 1:
            quality_score += 6
            quality_factors.append('One high-reliability signal')

        # Determine quality grade
        if quality_score >= 80:
            grade = 'A'
            grade_desc = 'Excellent - High confidence warning'
        elif quality_score >= 60:
            grade = 'B'
            grade_desc = 'Good - Solid warning signals'
        elif quality_score >= 40:
            grade = 'C'
            grade_desc = 'Fair - Some warning signals present'
        elif quality_score >= 20:
            grade = 'D'
            grade_desc = 'Low - Weak signals, monitor closely'
        else:
            grade = 'F'
            grade_desc = 'None - No significant warning signals'

        # Calculate confidence level
        if signal.bear_score >= 50 and quality_score >= 60:
            confidence = 'HIGH'
        elif signal.bear_score >= 30 and quality_score >= 40:
            confidence = 'MEDIUM'
        elif signal.bear_score >= 20 or quality_score >= 30:
            confidence = 'LOW'
        else:
            confidence = 'NONE'

        return {
            'quality_score': quality_score,
            'quality_grade': grade,
            'grade_description': grade_desc,
            'confidence_level': confidence,
            'quality_factors': quality_factors,
            'confluence_groups': groups_firing,
            'persistence_signals': elevated_duration,
            'early_indicators': early_count,
            'high_reliability_count': high_reliability_count,
            'bear_score': signal.bear_score,
            'actionable': quality_score >= 40 and signal.bear_score >= 30
        }

    def get_quality_report(self) -> str:
        """
        Generate formatted signal quality report.

        Returns:
            Multi-line string with quality analysis
        """
        quality = self.get_signal_quality()

        nl = chr(10)
        lines = []
        lines.append('=' * 60)
        lines.append('SIGNAL QUALITY ASSESSMENT')
        lines.append('=' * 60)
        lines.append('')

        # Grade display
        grade_display = {
            'A': '[A] EXCELLENT',
            'B': '[B] GOOD',
            'C': '[C] FAIR',
            'D': '[D] LOW',
            'F': '[F] NONE'
        }
        lines.append(f"Quality Grade: {grade_display.get(quality['quality_grade'], '[?]')}")
        lines.append(f"Quality Score: {quality['quality_score']}/100")
        lines.append(f"Confidence: {quality['confidence_level']}")
        lines.append(f"Description: {quality['grade_description']}")
        lines.append('')

        # Metrics
        lines.append('Quality Metrics:')
        lines.append(f"  Confluence Groups: {quality['confluence_groups']}/5")
        lines.append(f"  Persistent Signals: {quality['persistence_signals']}")
        lines.append(f"  Early Indicators: {quality['early_indicators']}")
        lines.append(f"  High-Reliability Count: {quality['high_reliability_count']}")
        lines.append('')

        # Quality factors
        if quality['quality_factors']:
            lines.append('Quality Factors:')
            for factor in quality['quality_factors']:
                lines.append(f"  [+] {factor}")
        else:
            lines.append('No significant quality factors')

        lines.append('')
        lines.append(f"Actionable: {'YES' if quality['actionable'] else 'NO'}")

        return nl.join(lines)

    def get_historical_performance(self, lookback_days: int = 30) -> Dict:
        """
        Track historical performance of signal predictions.

        Analyzes past signals and outcomes to measure prediction accuracy.

        Args:
            lookback_days: Days of history to analyze

        Returns:
            Dict with historical performance metrics
        """
        try:
            # Load historical bear scores
            history_file = 'data/bear_score_history.json'
            if not os.path.exists(history_file):
                return {
                    'status': 'NO_HISTORY',
                    'description': 'No historical data available'
                }

            with open(history_file, 'r') as f:
                history = json.load(f)

            if not history:
                return {
                    'status': 'EMPTY_HISTORY',
                    'description': 'History file is empty'
                }

            # Get SPY price data
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period=f"{lookback_days + 10}d")

            if len(spy_hist) < 10:
                return {
                    'status': 'INSUFFICIENT_DATA',
                    'description': 'Insufficient price data'
                }

            # Analyze signal performance
            # Count warnings and subsequent outcomes
            warnings_issued = 0
            warnings_correct = 0
            warnings_false = 0
            max_drawdown_after_warning = 0

            for entry in history[-100:]:  # Last 100 entries
                score = entry.get('bear_score', 0)
                level = entry.get('alert_level', 'NORMAL')
                timestamp = entry.get('timestamp', '')

                if level in ['WARNING', 'CRITICAL']:
                    warnings_issued += 1

                    # Check what happened in next 5 days
                    try:
                        signal_date = datetime.fromisoformat(timestamp.split('.')[0])
                        # Find price on signal date
                        signal_idx = None
                        for i, (idx, row) in enumerate(spy_hist.iterrows()):
                            if idx.date() >= signal_date.date():
                                signal_idx = i
                                break

                        if signal_idx is not None and signal_idx + 5 < len(spy_hist):
                            price_at_signal = spy_hist['Close'].iloc[signal_idx]
                            min_price_5d = spy_hist['Close'].iloc[signal_idx:signal_idx+5].min()
                            drawdown = ((min_price_5d / price_at_signal) - 1) * 100

                            if drawdown < -2:  # Dropped at least 2%
                                warnings_correct += 1
                                max_drawdown_after_warning = min(max_drawdown_after_warning, drawdown)
                            else:
                                warnings_false += 1
                    except:
                        pass

            # Calculate metrics
            if warnings_issued > 0:
                accuracy = (warnings_correct / warnings_issued) * 100
            else:
                accuracy = 0

            # Recent trend analysis
            recent_scores = [e.get('bear_score', 0) for e in history[-10:]]
            if len(recent_scores) >= 2:
                score_trend = recent_scores[-1] - recent_scores[0]
            else:
                score_trend = 0

            return {
                'status': 'OK',
                'lookback_days': lookback_days,
                'total_entries': len(history),
                'warnings_issued': warnings_issued,
                'warnings_correct': warnings_correct,
                'warnings_false': warnings_false,
                'accuracy_pct': round(accuracy, 1),
                'max_drawdown_captured': round(max_drawdown_after_warning, 2),
                'recent_score_trend': round(score_trend, 1),
                'recent_avg_score': round(sum(recent_scores) / len(recent_scores), 1) if recent_scores else 0,
                'description': f'{warnings_correct}/{warnings_issued} warnings led to drops ({accuracy:.0f}% accuracy)'
            }

        except Exception as e:
            return {
                'status': 'ERROR',
                'description': f'Error: {str(e)}'
            }

    def get_performance_report(self) -> str:
        """
        Generate formatted historical performance report.

        Returns:
            Multi-line string with performance metrics
        """
        perf = self.get_historical_performance()

        nl = chr(10)
        lines = []
        lines.append('=' * 60)
        lines.append('HISTORICAL PERFORMANCE TRACKING')
        lines.append('=' * 60)
        lines.append('')

        if perf.get('status') != 'OK':
            lines.append(f"Status: {perf.get('status')}")
            lines.append(f"Note: {perf.get('description')}")
            return nl.join(lines)

        lines.append(f"Lookback Period: {perf['lookback_days']} days")
        lines.append(f"Total Entries: {perf['total_entries']}")
        lines.append('')

        lines.append('Warning Performance:')
        lines.append(f"  Warnings Issued: {perf['warnings_issued']}")
        lines.append(f"  Correct Predictions: {perf['warnings_correct']}")
        lines.append(f"  False Alarms: {perf['warnings_false']}")
        lines.append(f"  Accuracy: {perf['accuracy_pct']:.1f}%")
        lines.append('')

        if perf['max_drawdown_captured'] < 0:
            lines.append(f"  Max Drawdown Captured: {perf['max_drawdown_captured']:.1f}%")
        lines.append('')

        lines.append('Recent Trend:')
        lines.append(f"  Average Score (10 entries): {perf['recent_avg_score']:.1f}")
        lines.append(f"  Score Trend: {perf['recent_score_trend']:+.1f}")

        return nl.join(lines)

    def get_full_diagnostic(self) -> str:
        """
        Generate complete diagnostic report combining all analyses.

        This is the most comprehensive report available, useful for
        detailed analysis and debugging.

        Returns:
            Multi-line string with complete diagnostic
        """
        nl = chr(10)
        reports = []

        # Header
        reports.append('*' * 70)
        reports.append('*  BEAR DETECTION FULL DIAGNOSTIC')
        reports.append(f"*  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        reports.append('*' * 70)
        reports.append('')

        # Daily Summary
        reports.append(self.get_daily_report())
        reports.append('')

        # Market Context
        reports.append(self.get_context_report())
        reports.append('')

        # Effectiveness Analysis
        reports.append(self.get_effectiveness_report())
        reports.append('')

        # Quality Assessment
        reports.append(self.get_quality_report())
        reports.append('')

        # Risk Attribution
        reports.append(self.get_attribution_report())
        reports.append('')

        # Scenario Analysis
        reports.append(self.get_scenario_report())
        reports.append('')

        # Multi-timeframe Analysis
        reports.append(self.get_multiframe_report())
        reports.append('')

        # Historical Performance
        reports.append(self.get_performance_report())
        reports.append('')

        # Ultimate Warning
        reports.append(self.get_ultimate_report())

        return nl.join(reports)

'''

def main():
    with open('src/analysis/fast_bear_detector.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Find insertion point - before the standalone functions
    insertion_point = 'def get_fast_bear_signal() -> FastBearSignal:'

    if insertion_point in content:
        content = content.replace(insertion_point, QUALITY_METHODS + '\n' + insertion_point)
        print("Added signal quality and performance tracking methods")
    else:
        print("Could not find insertion point")
        return

    with open('src/analysis/fast_bear_detector.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Done")

if __name__ == '__main__':
    main()

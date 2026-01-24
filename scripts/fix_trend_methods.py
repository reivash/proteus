"""Fix trend tracking methods placement - move inside FastBearDetector class"""

TREND_METHODS = '''

    # ==================== SIGNAL TREND TRACKING ====================

    def _store_signal(self, signal: FastBearSignal) -> None:
        """Store signal in history for trend tracking."""
        self._signal_history.append(signal)
        # Keep only last max_history signals
        if len(self._signal_history) > self._max_history:
            self._signal_history = self._signal_history[-self._max_history:]

    def get_signal_trend(self, periods: int = 5) -> Dict:
        """
        Analyze the trend direction of bear signals.

        Args:
            periods: Number of recent signals to analyze

        Returns:
            Dict with trend direction, momentum, and rate of change
        """
        if len(self._signal_history) < 2:
            return {
                'direction': 'UNKNOWN',
                'momentum': 0.0,
                'rate_of_change': 0.0,
                'signals_analyzed': len(self._signal_history),
                'description': 'Insufficient history for trend analysis'
            }

        # Get recent signals
        recent = self._signal_history[-min(periods, len(self._signal_history)):]
        scores = [s.bear_score for s in recent]

        # Calculate trend metrics
        if len(scores) >= 2:
            # Rate of change (per signal)
            roc = (scores[-1] - scores[0]) / len(scores)

            # Simple momentum (weighted recent more)
            weights = [i + 1 for i in range(len(scores))]
            weighted_avg = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            simple_avg = sum(scores) / len(scores)
            momentum = weighted_avg - simple_avg

            # Trend direction
            if roc > 2:
                direction = 'DETERIORATING_FAST'
                desc = f'Bear score rising rapidly ({roc:+.1f}/signal)'
            elif roc > 0.5:
                direction = 'DETERIORATING'
                desc = f'Bear score gradually rising ({roc:+.1f}/signal)'
            elif roc < -2:
                direction = 'IMPROVING_FAST'
                desc = f'Bear score falling rapidly ({roc:+.1f}/signal)'
            elif roc < -0.5:
                direction = 'IMPROVING'
                desc = f'Bear score gradually falling ({roc:+.1f}/signal)'
            else:
                direction = 'STABLE'
                desc = f'Bear score stable ({roc:+.1f}/signal)'
        else:
            roc = 0.0
            momentum = 0.0
            direction = 'UNKNOWN'
            desc = 'Insufficient data'

        return {
            'direction': direction,
            'momentum': round(momentum, 2),
            'rate_of_change': round(roc, 2),
            'signals_analyzed': len(recent),
            'latest_score': scores[-1] if scores else 0,
            'oldest_score': scores[0] if scores else 0,
            'min_score': min(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'description': desc
        }

    def get_trend_analysis(self) -> str:
        """
        Get comprehensive trend analysis as formatted text.

        Returns:
            Multi-line string with trend analysis
        """
        trend = self.get_signal_trend()

        nl = chr(10)  # Newline character
        lines = []
        lines.append('=' * 60)
        lines.append('BEAR SIGNAL TREND ANALYSIS')
        lines.append('=' * 60)
        lines.append('')

        # Direction indicator
        direction_icons = {
            'DETERIORATING_FAST': '[!!!] ',
            'DETERIORATING': '[!!] ',
            'STABLE': '[--] ',
            'IMPROVING': '[+] ',
            'IMPROVING_FAST': '[++] ',
            'UNKNOWN': '[?] '
        }
        icon = direction_icons.get(trend['direction'], '[?] ')

        lines.append(f"Trend Direction: {icon}{trend['direction']}")
        lines.append(f"Description: {trend['description']}")
        lines.append('')
        lines.append(f"Metrics:")
        lines.append(f"  - Rate of Change: {trend['rate_of_change']:+.2f} points/signal")
        lines.append(f"  - Momentum: {trend['momentum']:+.2f}")
        lines.append(f"  - Signals Analyzed: {trend['signals_analyzed']}")
        lines.append('')

        if trend['signals_analyzed'] >= 2:
            lines.append(f"Score Range:")
            lines.append(f"  - Latest: {trend['latest_score']:.1f}")
            lines.append(f"  - Oldest: {trend['oldest_score']:.1f}")
            lines.append(f"  - Min: {trend['min_score']:.1f}")
            lines.append(f"  - Max: {trend['max_score']:.1f}")

            # Add warning if trend is deteriorating
            if trend['direction'] in ['DETERIORATING', 'DETERIORATING_FAST']:
                lines.append('')
                lines.append('[WARNING] Risk signals are increasing - monitor closely')
            elif trend['direction'] in ['IMPROVING', 'IMPROVING_FAST']:
                lines.append('')
                lines.append('[INFO] Risk signals are decreasing - conditions improving')

        return nl.join(lines)

    def get_momentum_direction(self) -> str:
        """
        Get simple momentum direction for monitoring.

        Returns:
            One of: UP (increasing risk), DOWN (decreasing risk), FLAT, UNKNOWN
        """
        trend = self.get_signal_trend()
        roc = trend.get('rate_of_change', 0)

        if abs(roc) < 0.5:
            return 'FLAT'
        elif roc > 0:
            return 'UP'
        else:
            return 'DOWN'

    def detect_trend_reversal(self, lookback: int = 10) -> Dict:
        """
        Detect potential trend reversals in bear signals.

        Args:
            lookback: Number of signals to analyze

        Returns:
            Dict with reversal detection results
        """
        if len(self._signal_history) < lookback:
            return {
                'reversal_detected': False,
                'reversal_type': None,
                'confidence': 0.0,
                'description': 'Insufficient history'
            }

        recent = self._signal_history[-lookback:]
        scores = [s.bear_score for s in recent]

        # Split into first and second half
        mid = len(scores) // 2
        first_half = scores[:mid]
        second_half = scores[mid:]

        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0

        # Calculate slopes
        first_slope = (first_half[-1] - first_half[0]) / len(first_half) if len(first_half) > 1 else 0
        second_slope = (second_half[-1] - second_half[0]) / len(second_half) if len(second_half) > 1 else 0

        # Detect reversal
        reversal_detected = False
        reversal_type = None
        confidence = 0.0

        # Bullish reversal (risk was rising, now falling)
        if first_slope > 1 and second_slope < -1:
            reversal_detected = True
            reversal_type = 'BULLISH'
            confidence = min(abs(first_slope - second_slope) / 4, 1.0)

        # Bearish reversal (risk was falling, now rising)
        elif first_slope < -1 and second_slope > 1:
            reversal_detected = True
            reversal_type = 'BEARISH'
            confidence = min(abs(second_slope - first_slope) / 4, 1.0)

        description = 'No reversal detected'
        if reversal_type == 'BULLISH':
            description = f'Risk trend reversing lower (conf: {confidence:.0%})'
        elif reversal_type == 'BEARISH':
            description = f'Risk trend reversing higher (conf: {confidence:.0%}) - CAUTION'

        return {
            'reversal_detected': reversal_detected,
            'reversal_type': reversal_type,
            'confidence': round(confidence, 2),
            'first_half_slope': round(first_slope, 2),
            'second_half_slope': round(second_slope, 2),
            'first_half_avg': round(first_avg, 1),
            'second_half_avg': round(second_avg, 1),
            'description': description
        }

    def get_intraday_trend(self) -> Dict:
        """
        Analyze today's intraday bear signal trend.

        Returns:
            Dict with intraday trend analysis
        """
        today = datetime.now().date()
        today_signals = [s for s in self._signal_history
                        if datetime.fromisoformat(s.timestamp.split('.')[0]).date() == today]

        if len(today_signals) < 2:
            return {
                'trend': 'UNKNOWN',
                'change_today': 0.0,
                'signals_today': len(today_signals),
                'description': 'Insufficient intraday data'
            }

        scores = [s.bear_score for s in today_signals]
        change = scores[-1] - scores[0]

        if change > 5:
            trend = 'DETERIORATING'
            desc = f'Risk increased {change:+.1f} points today'
        elif change < -5:
            trend = 'IMPROVING'
            desc = f'Risk decreased {change:+.1f} points today'
        else:
            trend = 'STABLE'
            desc = f'Risk stable ({change:+.1f} points) today'

        return {
            'trend': trend,
            'change_today': round(change, 1),
            'signals_today': len(today_signals),
            'morning_score': scores[0],
            'current_score': scores[-1],
            'high_of_day': max(scores),
            'low_of_day': min(scores),
            'description': desc
        }

'''

def main():
    with open('src/analysis/fast_bear_detector.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # First, remove the incorrectly placed methods (after print_bear_trend function)
    # Find and remove the SIGNAL TREND TRACKING section that's outside the class
    marker_start = '\n\n\n    # ==================== SIGNAL TREND TRACKING ===================='
    marker_end = 'if __name__ == "__main__":'

    if marker_start in content:
        # Find the position of the incorrectly placed section
        start_pos = content.find(marker_start)
        end_pos = content.find(marker_end)
        if start_pos > 0 and end_pos > start_pos:
            # Remove everything between the markers
            content = content[:start_pos] + '\n\n\n' + content[end_pos:]
            print("Removed incorrectly placed trend methods")

    # Now add the methods inside the class, before the standalone functions
    # Find the end of get_detailed_report method (last class method)
    insertion_point = 'def get_fast_bear_signal() -> FastBearSignal:'

    if insertion_point in content:
        content = content.replace(insertion_point, TREND_METHODS + '\n' + insertion_point)
        print("Added trend tracking methods inside class")
    else:
        print("Could not find insertion point")
        return

    with open('src/analysis/fast_bear_detector.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Done")

if __name__ == '__main__':
    main()

"""
SendGrid Email Notifier - Simpler Alternative to SMTP

Much easier than Gmail:
1. Sign up at https://sendgrid.com (free tier: 100 emails/day)
2. Get API key from Settings > API Keys
3. Add to email_config.json: "sendgrid_api_key": "YOUR_KEY"

No passwords, no app passwords, just one API key!
"""

import json
import os
from typing import List, Dict
from datetime import datetime

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Email, To, Content
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

# ML Performance Tracking
try:
    from src.monitoring.ml_performance_tracker import MLPerformanceTracker
    ML_TRACKING_AVAILABLE = True
except ImportError:
    ML_TRACKING_AVAILABLE = False

# ML Plotting Utilities
try:
    from src.utils.ml_plot_utils import (
        create_prediction_plot,
        create_roc_curve_plot,
        create_calibration_plot,
        create_feature_importance_plot,
        create_performance_timeline,
        create_experiment_comparison_plot
    )
    ML_PLOTTING_AVAILABLE = True
except ImportError:
    ML_PLOTTING_AVAILABLE = False


class SendGridNotifier:
    """Send email notifications via SendGrid API (simpler than SMTP)."""

    def __init__(self, config_path='email_config.json'):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load email configuration."""
        if not os.path.exists(self.config_path):
            return {'enabled': False}

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            # Check for SendGrid API key
            if 'sendgrid_api_key' not in config or not config['sendgrid_api_key']:
                return {'enabled': False}

            if 'recipient_email' not in config or not config['recipient_email']:
                return {'enabled': False}

            return config

        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            return {'enabled': False}

    def is_enabled(self) -> bool:
        """Check if SendGrid is enabled and configured."""
        return self.config.get('enabled', False) and SENDGRID_AVAILABLE

    def send_scan_notification(self, scan_status: str, signals: List[Dict],
                              performance: Dict = None, scanned_tickers: List[str] = None) -> bool:
        """Send email via SendGrid."""
        if not self.is_enabled():
            return False

        if not SENDGRID_AVAILABLE:
            print("[ERROR] SendGrid not installed. Run: pip install sendgrid")
            return False

        subject = self._create_subject(signals)
        html_content = self._create_body(scan_status, signals, performance, scanned_tickers)

        message = Mail(
            from_email=Email(self.config.get('sender_email', 'proteus@trading.local')),
            to_emails=To(self.config['recipient_email']),
            subject=subject,
            html_content=Content("text/html", html_content)
        )

        try:
            sg = SendGridAPIClient(self.config['sendgrid_api_key'])
            response = sg.send(message)

            print(f"[EMAIL] Sent via SendGrid to {self.config['recipient_email']}: {subject}")
            return True

        except Exception as e:
            print(f"[ERROR] SendGrid failed: {e}")
            return False

    def send_test_email(self) -> bool:
        """Send test email."""
        if not self.is_enabled():
            return False

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        subject = f"🧪 Proteus - SendGrid Test - {timestamp}"
        html = """
<html>
<body style="font-family: Arial, sans-serif;">
    <h2 style="color: #667eea;">✓ SendGrid Email Working!</h2>
    <p>Your Proteus dashboard is now configured to send email notifications via SendGrid.</p>
    <p><strong>Much simpler than SMTP!</strong> Just one API key, no passwords needed.</p>
    <p>You'll receive notifications at: {recipient}</p>
    <hr>
    <p style="color: #666; font-size: 0.9em;">Sent: {timestamp}</p>
</body>
</html>
""".format(
            recipient=self.config['recipient_email'],
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )

        message = Mail(
            from_email=Email(self.config.get('sender_email', 'proteus@trading.local')),
            to_emails=To(self.config['recipient_email']),
            subject=subject,
            html_content=Content("text/html", html)
        )

        try:
            sg = SendGridAPIClient(self.config['sendgrid_api_key'])
            sg.send(message)
            print(f"✓ Test email sent via SendGrid!")
            return True
        except Exception as e:
            print(f"✗ SendGrid test failed: {e}")
            return False

    def send_experiment_report(self, experiment_id: str, results: Dict) -> bool:
        """
        Send experiment completion report email.

        Args:
            experiment_id: Experiment identifier (e.g., 'EXP-024')
            results: Dictionary containing experiment results

        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.is_enabled():
            print("[INFO] Email not enabled, skipping experiment report")
            return False

        if not SENDGRID_AVAILABLE:
            print("[ERROR] SendGrid not installed. Run: pip install sendgrid")
            return False

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Create informative subject line with key results
        status_summary = self._extract_experiment_summary(results)
        subject = f"📊 Proteus {experiment_id} - {status_summary} - {timestamp}"

        html_content = self._create_experiment_body(experiment_id, results)

        message = Mail(
            from_email=Email(self.config.get('sender_email', 'proteus@trading.local')),
            to_emails=To(self.config['recipient_email']),
            subject=subject,
            html_content=Content("text/html", html_content)
        )

        try:
            sg = SendGridAPIClient(self.config['sendgrid_api_key'])
            response = sg.send(message)
            print(f"[EMAIL] Experiment report sent via SendGrid: {experiment_id}")
            return True
        except Exception as e:
            print(f"[ERROR] SendGrid experiment report failed: {e}")
            return False

    def _generate_conclusion(self, results: Dict) -> str:
        """
        Generate a clear conclusion statement based on experiment results.

        Returns:
            Human-readable conclusion about experiment outcome and next steps
        """
        try:
            # Check if deploy field exists
            deploy = results.get('deploy', None)
            next_step = results.get('next_step', 'Review results and determine next action')

            # Check for explicit conclusion in results
            if 'conclusion' in results:
                return results['conclusion']

            # Generate conclusion based on deployment decision
            if deploy is True or deploy == 1.0:
                # Successful deployment
                tier_a_count = 0
                if 'new_test_results' in results:
                    tier_a = results['new_test_results'].get('tier_a', [])
                    tier_a_count = len(tier_a)

                if tier_a_count > 0:
                    return f"✅ SUCCESS: Experiment validated {tier_a_count} high-performing stocks for deployment. Next: {next_step}"
                else:
                    return f"✅ SUCCESS: Experiment meets deployment criteria. Next: {next_step}"

            elif deploy is False or deploy == 0.0:
                # Failed to meet criteria
                return f"❌ INSUFFICIENT PERFORMANCE: Experiment did not meet deployment thresholds. Next: {next_step}"

            # Check for other success indicators
            if 'win_rate_improvement' in results:
                improvement = results['win_rate_improvement']
                if improvement >= 3.0:
                    return f"✅ Significant win rate improvement (+{improvement:.1f}pp). Ready for deployment."
                elif improvement > 0:
                    return f"⚠️ Marginal improvement (+{improvement:.1f}pp) below +3pp threshold. Not deployed."
                else:
                    return f"❌ No improvement found ({improvement:+.1f}pp). Exploring alternative approaches."

            # Check stocks processed metrics
            if 'successful_processing' in results or 'successful_analyses' in results:
                successful = results.get('successful_processing', results.get('successful_analyses', 0))
                tested = results.get('stocks_tested', successful)

                if successful >= tested * 0.8:  # 80%+ success rate
                    return f"✅ Successfully processed {successful}/{tested} stocks. Pipeline validated for next phase: {next_step}"
                else:
                    return f"⚠️ Only {successful}/{tested} stocks processed successfully. Improvements needed before proceeding."

            # Default: use next_step as conclusion
            return f"Experiment complete. Next step: {next_step}"

        except Exception as e:
            return "Experiment complete. See detailed results below."

    def _extract_experiment_summary(self, results: Dict) -> str:
        """
        Extract key findings from experiment results for subject line.

        Returns concise summary like:
        - "SUCCESS: 9 New Tier A Stocks"
        - "FAILED: -5.7pp Win Rate"
        - "NEUTRAL: No Improvement Found"
        """
        try:
            # Stock expansion experiments
            if 'new_test_results' in results:
                tier_a_count = len([r for r in results.get('new_test_results', [])
                                   if r.get('win_rate', 0) >= 70])
                if tier_a_count > 0:
                    return f"✅ SUCCESS: {tier_a_count} New Tier A Stocks"
                else:
                    return "❌ FAILED: No Tier A Stocks Found"

            # Win rate / performance improvement experiments
            if 'win_rate_improvement' in results:
                improvement = results['win_rate_improvement']
                if improvement >= 3.0:
                    return f"✅ SUCCESS: +{improvement:.1f}pp Win Rate"
                elif improvement >= 0:
                    return f"⚠️ MARGINAL: +{improvement:.1f}pp Win Rate"
                else:
                    return f"❌ FAILED: {improvement:+.1f}pp Win Rate"

            # Return improvement experiments
            if 'return_improvement' in results:
                improvement = results['return_improvement']
                if improvement >= 5.0:
                    return f"✅ SUCCESS: +{improvement:.1f}% Return"
                elif improvement >= 0:
                    return f"⚠️ MARGINAL: +{improvement:.1f}% Return"
                else:
                    return f"❌ FAILED: {improvement:+.1f}% Return"

            # Optimization experiments (stocks improved)
            if 'stocks_improved' in results:
                improved = results['stocks_improved']
                tested = results.get('stocks_tested', improved)
                if improved > 0:
                    return f"✅ SUCCESS: {improved}/{tested} Stocks Improved"
                else:
                    return f"❌ FAILED: No Stocks Improved"

            # Validation experiments
            if 'validation_status' in results:
                status = results['validation_status']
                if status.lower() in ['approved', 'success', 'validated']:
                    return "✅ VALIDATED"
                elif status.lower() in ['rejected', 'failed']:
                    return "❌ REJECTED"
                else:
                    return f"⚠️ {status.upper()}"

            # Default: check for general success/failure indicators
            if 'status' in results:
                if results['status'].lower() in ['success', 'approved']:
                    return "✅ SUCCESS"
                elif results['status'].lower() in ['failed', 'rejected']:
                    return "❌ FAILED"

            # Fallback
            return "Complete"

        except Exception as e:
            print(f"[WARNING] Could not extract experiment summary: {e}")
            return "Complete"

    def _create_subject(self, signals: List[Dict]) -> str:
        """Create email subject with timestamp to prevent threading."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if len(signals) == 0:
            return f"Proteus Daily Scan - No Signals - {timestamp}"
        elif len(signals) == 1:
            return f"ALERT: Proteus BUY Signal - {signals[0]['ticker']} - {timestamp}"
        else:
            return f"ALERT: Proteus - {len(signals)} BUY Signals! - {timestamp}"

    def _generate_ml_prediction_plots_html(self, ml_results: Dict) -> str:
        """
        Generate ML prediction visualization plots for email.

        Args:
            ml_results: Dictionary containing ML prediction data with keys:
                - 'y_true': Actual labels (numpy array)
                - 'y_pred': Predicted labels (numpy array)
                - 'y_pred_proba': Predicted probabilities (numpy array, optional)
                - 'auc_score': AUC score (float, optional)
                - 'feature_names': List of feature names (optional)
                - 'feature_importance': Feature importance values (optional)

        Returns:
            HTML string with embedded prediction plots or empty string if plotting unavailable
        """
        if not ML_PLOTTING_AVAILABLE:
            return ""

        try:
            import numpy as np

            y_true = ml_results.get('y_true')
            y_pred = ml_results.get('y_pred')
            y_pred_proba = ml_results.get('y_pred_proba')
            auc_score = ml_results.get('auc_score')
            feature_names = ml_results.get('feature_names')
            feature_importance = ml_results.get('feature_importance')

            # Must have at least predictions to generate plots
            if y_true is None or y_pred is None:
                return ""

            # Convert to numpy if needed
            if not isinstance(y_true, np.ndarray):
                y_true = np.array(y_true)
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred)
            if y_pred_proba is not None and not isinstance(y_pred_proba, np.ndarray):
                y_pred_proba = np.array(y_pred_proba)

            html = """
    <div class="section" style="background: #f8fafc; border-left-color: #8b5cf6;">
        <h3 style="color: #8b5cf6;">📊 ML Model Performance Visualization</h3>
        <p style="color: #666; margin-bottom: 20px;">Visual analysis of prediction accuracy and model calibration</p>
"""

            # 1. Main prediction plot (confusion matrix + probability distribution)
            try:
                prediction_plot_b64 = create_prediction_plot(
                    y_true=y_true,
                    y_pred=y_pred,
                    y_pred_proba=y_pred_proba,
                    title="Model Predictions vs Actual Outcomes"
                )
                html += f"""
        <div style="margin: 20px 0;">
            <h4 style="color: #6b21a8; margin-bottom: 10px;">Confusion Matrix & Probability Distribution</h4>
            <img src="data:image/png;base64,{prediction_plot_b64}"
                 style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px;">
        </div>
"""
            except Exception as e:
                print(f"[WARNING] Failed to generate prediction plot: {e}")

            # 2. ROC curve (if probabilities and AUC available)
            if y_pred_proba is not None and auc_score is not None:
                try:
                    roc_plot_b64 = create_roc_curve_plot(
                        y_true=y_true,
                        y_pred_proba=y_pred_proba,
                        auc_score=auc_score
                    )
                    html += f"""
        <div style="margin: 20px 0;">
            <h4 style="color: #6b21a8; margin-bottom: 10px;">ROC Curve - Model Discrimination</h4>
            <img src="data:image/png;base64,{roc_plot_b64}"
                 style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px;">
        </div>
"""
                except Exception as e:
                    print(f"[WARNING] Failed to generate ROC curve: {e}")

            # 3. Calibration plot (if probabilities available)
            if y_pred_proba is not None:
                try:
                    calibration_plot_b64 = create_calibration_plot(
                        y_true=y_true,
                        y_pred_proba=y_pred_proba,
                        n_bins=10
                    )
                    html += f"""
        <div style="margin: 20px 0;">
            <h4 style="color: #6b21a8; margin-bottom: 10px;">Calibration Curve - Probability Accuracy</h4>
            <p style="color: #666; font-size: 0.95em; margin-bottom: 10px;">
                Shows if predicted probabilities match actual win rates. Perfect calibration = diagonal line.
            </p>
            <img src="data:image/png;base64,{calibration_plot_b64}"
                 style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px;">
        </div>
"""
                except Exception as e:
                    print(f"[WARNING] Failed to generate calibration plot: {e}")

            # 4. Feature importance (if available)
            if feature_names is not None and feature_importance is not None:
                try:
                    if not isinstance(feature_importance, np.ndarray):
                        feature_importance = np.array(feature_importance)

                    importance_plot_b64 = create_feature_importance_plot(
                        feature_names=feature_names,
                        importance_values=feature_importance,
                        top_n=15
                    )
                    html += f"""
        <div style="margin: 20px 0;">
            <h4 style="color: #6b21a8; margin-bottom: 10px;">Top Feature Importance - What Drives Predictions?</h4>
            <img src="data:image/png;base64,{importance_plot_b64}"
                 style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px;">
        </div>
"""
                except Exception as e:
                    print(f"[WARNING] Failed to generate feature importance plot: {e}")

            html += """
        <p style="color: #666; font-size: 0.9em; margin-top: 20px; padding-top: 15px; border-top: 1px solid #ddd;">
            <strong>How to interpret:</strong> Confusion matrix shows true vs predicted outcomes.
            ROC curve measures model discrimination (higher AUC = better).
            Calibration curve validates if predicted probabilities are accurate (closer to diagonal = better).
        </p>
    </div>
"""
            return html

        except Exception as e:
            print(f"[WARNING] Failed to generate ML prediction plots: {e}")
            return ""

    def _generate_ml_performance_html(self, days: int = 7) -> str:
        """
        Generate ML performance metrics HTML section.

        Args:
            days: Number of days to analyze

        Returns:
            HTML string with ML performance metrics or empty string if no data
        """
        if not ML_TRACKING_AVAILABLE:
            return ""

        try:
            tracker = MLPerformanceTracker()
            metrics = tracker.calculate_metrics(days=days)

            # Skip if no completed signals
            if 'error' in metrics or metrics.get('completed_signals', 0) == 0:
                return ""

            ml_approved = metrics['ml_approved']
            ml_filtered = metrics['ml_filtered']
            improvement_factor = metrics['improvement_factor']

            # Color code based on performance
            if improvement_factor >= 2.0:
                perf_color = '#10b981'  # Green
                perf_status = 'EXCELLENT'
            elif improvement_factor >= 1.5:
                perf_color = '#3b82f6'  # Blue
                perf_status = 'GOOD'
            elif improvement_factor >= 1.0:
                perf_color = '#f59e0b'  # Orange
                perf_status = 'ACCEPTABLE'
            else:
                perf_color = '#ef4444'  # Red
                perf_status = 'DEGRADED'

            html = f"""
    <h3>ML Performance Tracker (Last {days} Days)</h3>
    <div class="ml-performance" style="background: #f8fafc; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 4px solid {perf_color};">
        <div class="metric" style="margin: 8px 0;">
            <strong>Status:</strong>
            <span style="color: {perf_color}; font-weight: bold;">{perf_status}</span>
        </div>
        <div class="metric" style="margin: 8px 0;">
            <strong>Improvement Factor:</strong>
            <span style="color: {perf_color}; font-weight: bold;">{improvement_factor:.2f}x</span>
            (Target: 2.4x)
        </div>

        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
            <strong>ML-Approved Signals (>= 30% confidence):</strong>
            <div style="margin-left: 20px;">
                <div class="metric">Count: {ml_approved['count']} ({ml_approved['wins']}W / {ml_approved['losses']}L)</div>
                <div class="metric">Precision: <span style="color: #10b981; font-weight: bold;">{ml_approved['precision']*100:.1f}%</span></div>
                <div class="metric">Avg Return: <span style="color: {'#10b981' if ml_approved['avg_return'] > 0 else '#ef4444'};">{ml_approved['avg_return']:+.1f}%</span></div>
            </div>
        </div>

        <div style="margin-top: 10px;">
            <strong>ML-Filtered Signals (< 30% confidence):</strong>
            <div style="margin-left: 20px;">
                <div class="metric">Count: {ml_filtered['count']} ({ml_filtered['wins']}W / {ml_filtered['losses']}L)</div>
                <div class="metric">Precision: {ml_filtered['precision']*100:.1f}%</div>
                <div class="metric">Avg Return: <span style="color: {'#10b981' if ml_filtered['avg_return'] > 0 else '#ef4444'};">{ml_filtered['avg_return']:+.1f}%</span></div>
            </div>
        </div>

        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd; font-size: 0.9em; color: #666;">
            <strong>Note:</strong> ML performance tracking validates the 2.4x precision improvement claim in production.
            Manual outcome updates required via ml_outcome_updater.py
        </div>
    </div>
"""
            return html

        except Exception as e:
            print(f"[WARNING] Failed to generate ML performance HTML: {e}")
            return ""

    def _create_body(self, scan_status: str, signals: List[Dict], performance: Dict = None, scanned_tickers: List[str] = None) -> str:
        """Create email body - full HTML template."""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        html = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; color: #333; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                 color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .status {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }}
        .signal {{ background: #e0f2fe; padding: 15px; margin: 10px 0;
                 border-left: 4px solid #667eea; border-radius: 3px; }}
        .signal-header {{ font-size: 1.2em; font-weight: bold; color: #667eea; margin-bottom: 10px; }}
        .signal-detail {{ margin: 5px 0; }}
        .performance {{ background: #f0fdf4; padding: 15px; border-radius: 5px; margin: 15px 0; }}
        .metric {{ margin: 8px 0; }}
        .success {{ color: #10b981; }}
        .scanned {{ background: #fef3c7; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 4px solid #f59e0b; }}
        .footer {{ color: #666; font-size: 0.85em; margin-top: 30px; padding-top: 20px;
                  border-top: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>Proteus Trading Dashboard</h2>
        <p>AI-Powered Stock Predictor</p>
        <p style="font-size: 0.9em;">{now}</p>
    </div>

    <div class="status">
        <strong>Scan Status:</strong> {scan_status}
    </div>

    <div class="scanned">
        <strong>Instruments Scanned ({len(scanned_tickers) if scanned_tickers else 0}):</strong><br>
        <span style="font-size: 0.95em;">{', '.join(scanned_tickers) if scanned_tickers else 'Not specified'}</span>
    </div>
"""

        # Add signals section
        if len(signals) > 0:
            html += f"""
    <h3>Active Buy Signals ({len(signals)})</h3>
"""
            for signal in signals:
                expected_return = signal.get('expected_return', 0)
                html += f"""
    <div class="signal">
        <div class="signal-header">{signal['ticker']}</div>
        <div class="signal-detail"><strong>Entry Price:</strong> ${signal['price']:.2f}</div>
        <div class="signal-detail"><strong>Z-Score:</strong> {signal['z_score']:.2f}</div>
        <div class="signal-detail"><strong>RSI:</strong> {signal['rsi']:.1f}</div>
        <div class="signal-detail"><strong>Expected Return:</strong> <span class="success">+{expected_return:.2f}%</span></div>
    </div>
"""
        else:
            html += """
    <h3>No Signals Detected</h3>
    <div class="status">
        <p>No panic sell opportunities found. Strategy is waiting for the right moment.</p>
    </div>
"""

        # Add performance
        if performance:
            html += f"""
    <h3>Performance Summary</h3>
    <div class="performance">
        <div class="metric"><strong>System:</strong> Multi-Strategy Stock Predictor v15.0</div>
        <div class="metric"><strong>Strategy:</strong> Panic sell detection using Z-score, RSI, volume spikes, and price drops</div>
        <div class="metric"><strong>Exit Rules:</strong> Day 0: ±2%, Day 1: ±1.5%, Day 2+: ±1% (time-decay)</div>
        <div class="metric"><strong>Filters:</strong> Market regime (no trading in bear markets) + Earnings exclusion (±3 days)</div>
        <div class="metric" style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;"><strong>Backtest Period:</strong> 2022-2025 (3 years)</div>
        <div class="metric"><strong>Optimization:</strong> EXP-008 (stock-specific parameters) + EXP-010 (time-decay exits)</div>
        <div class="metric"><strong>Tested Symbols:</strong> NVDA, TSLA, AAPL, AMZN, MSFT, JPM, JNJ, UNH, INTC, CVX, QQQ</div>
        <div class="metric" style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;"><strong>Current Performance:</strong></div>
        <div class="metric">Total Trades: <strong>{performance.get('total_trades', 0)}</strong></div>
        <div class="metric">Win Rate: <strong>{performance.get('win_rate', 0):.1f}%</strong> (Target: 77.3%)</div>
        <div class="metric">Total Return: <strong>{performance.get('total_return', 0):+.2f}%</strong></div>
    </div>
"""

        # Add ML performance metrics if available
        ml_perf_html = self._generate_ml_performance_html(days=7)
        if ml_perf_html:
            html += ml_perf_html

        html += """
    <div class="footer">
        <p><em>Automated notification from Proteus Trading Dashboard</em></p>
        <p><em>Dashboard: <a href="http://localhost:5000">http://localhost:5000</a></em></p>
    </div>
</body>
</html>
"""
        return html

    def _create_experiment_body(self, experiment_id: str, results: Dict) -> str:
        """Create COMPREHENSIVE HTML email body for experiment report."""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Extract information - handle multiple experiment formats
        symbols_tested = results.get('symbols_tested', 0)
        test_results = results.get('new_test_results', {})
        tier_a = test_results.get('tier_a', [])
        tier_b = test_results.get('tier_b', [])
        tier_c = test_results.get('tier_c', [])
        period = results.get('period', '3y')

        # Build stock data lookup from results list
        stock_data_map = {}
        all_results = test_results.get('results', [])

        # Also check for top-level detailed_results (EXP-062 format)
        if not all_results:
            all_results = results.get('detailed_results', [])

        for stock_data in all_results:
            if isinstance(stock_data, dict) and 'ticker' in stock_data:
                stock_data_map[stock_data['ticker']] = stock_data

        # Get all tested symbols from results if available
        all_stocks = tier_a + tier_b + tier_c
        # Handle both string tickers and dict format
        tested_symbols = []
        for stock in all_stocks:
            if isinstance(stock, str):
                tested_symbols.append(stock)
            elif isinstance(stock, dict) and stock.get('ticker'):
                tested_symbols.append(stock.get('ticker'))

        # If we have no tested symbols from tiers, try multiple sources
        if not tested_symbols:
            # Try stock_data_map
            if stock_data_map:
                tested_symbols = list(stock_data_map.keys())
            # Try top-level test_stocks list (EXP-062 format)
            elif 'test_stocks' in results:
                tested_symbols = results['test_stocks']

        tier_a_count = len(tier_a) if tier_a else 0
        tier_b_count = len(tier_b) if tier_b else 0
        tier_c_count = len(tier_c) if tier_c else 0

        html = f"""
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #1a1a1a; line-height: 1.7; }}
        .header {{ background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                 color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .section {{ background: #f8f9fa; padding: 25px; border-radius: 8px; margin: 20px 0;
                   border-left: 5px solid #10b981; }}
        .objective {{ background: #f0f9ff; border-left-color: #0ea5e9; }}
        .hypothesis {{ background: #fef3c7; border-left-color: #eab308; }}
        .conclusion {{ background: #f3e8ff; border-left-color: #a855f7; }}
        .methodology {{ background: #e7f3ff; border-left-color: #0078d4; }}
        .symbols {{ background: #fff4e6; border-left-color: #f59e0b; }}
        .tier-section {{ background: #e0f2fe; padding: 20px; margin: 20px 0;
                        border-left: 5px solid #0284c7; border-radius: 8px; }}
        .tier-a {{ background: #dcfce7; border-left-color: #16a34a; }}
        .tier-b {{ background: #fef3c7; border-left-color: #f59e0b; }}
        .tier-c {{ background: #fee2e2; border-left-color: #dc2626; }}
        .stock {{ margin: 15px 0; padding: 15px; background: white; border-radius: 6px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .stock-name {{ font-weight: bold; color: #0284c7; font-size: 1.15em; margin-bottom: 8px; }}
        .metric {{ margin: 7px 0; font-size: 0.95em; }}
        .metric-label {{ color: #666; font-weight: 500; }}
        .success {{ color: #16a34a; font-weight: bold; }}
        .warning {{ color: #f59e0b; font-weight: bold; }}
        .danger {{ color: #dc2626; font-weight: bold; }}
        .symbol-list {{ display: inline-block; padding: 8px 12px; background: white;
                       border-radius: 4px; margin: 4px; font-family: monospace; }}
        .footer {{ color: #666; font-size: 0.9em; margin-top: 40px; padding-top: 25px;
                  border-top: 2px solid #ddd; }}
        h3 {{ margin-top: 0; margin-bottom: 15px; }}
        .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1 style="margin: 0; font-size: 1.8em;">📊 {experiment_id} - Complete Report</h1>
        <p style="font-size: 1.15em; margin: 10px 0; opacity: 0.95;">Proteus Stock Prediction System</p>
        <p style="font-size: 0.95em; opacity: 0.85; margin: 5px 0;">Completed: {now}</p>
    </div>

    <div class="section objective">
        <h3 style="color: #0ea5e9;">🎯 Objective</h3>
        <p style="margin: 0; font-size: 1.05em;">{results.get('objective', results.get('algorithm', 'Optimize trading signals and expand high-performing stock portfolio'))}</p>
    </div>

    <div class="section hypothesis">
        <h3 style="color: #eab308;">💡 Hypothesis</h3>
        <p style="margin: 0; font-size: 1.05em;">{results.get('hypothesis', 'Testing signal optimization to improve win rate and returns')}</p>
    </div>

    <div class="section conclusion">
        <h3 style="color: #a855f7;">📋 Conclusion</h3>
        <p style="margin: 0; font-size: 1.05em; font-weight: 500;">{self._generate_conclusion(results)}</p>
    </div>
"""

        # Add ML prediction plots if available
        if 'ml_results' in results:
            ml_plots_html = self._generate_ml_prediction_plots_html(results['ml_results'])
            html += ml_plots_html

        html += """
    <div class="section methodology">
        <h3 style="color: #0078d4;">🔬 Methodology (Current System)</h3>
        <div class="metric"><span class="metric-label">System:</span> Multi-Strategy Stock Predictor (v15.0)</div>
        <div class="metric"><span class="metric-label">Approach:</span> Research-driven signal optimization across multiple strategies</div>
        <div class="metric"><span class="metric-label">Entry Signal:</span> Z-score < -1.5, RSI < 35, Volume spike > 1.3x, Price drop > -1.5%</div>
        <div class="metric"><span class="metric-label">Exit Strategy:</span> Time-decay targets (Day 0: ±2%, Day 1: ±1.5%, Day 2+: ±1%)</div>
        <div class="metric"><span class="metric-label">Backtest Period:</span> {period} ({self._get_period_dates(period)})</div>
        <div class="metric"><span class="metric-label">Data Source:</span> Yahoo Finance (historical OHLCV)</div>
        <div class="metric"><span class="metric-label">Initial Capital:</span> $10,000 per stock (simulated)</div>
    </div>

    <div class="section symbols">
        <h3 style="color: #f59e0b;">📋 Symbols Tested ({symbols_tested} total)</h3>
        <div style="margin-top: 15px;">
"""

        # Add all tested symbols
        for symbol in tested_symbols:
            html += f'<span class="symbol-list">{symbol}</span>'

        html += f"""
        </div>
    </div>

    <div class="section">
        <h3 style="color: #059669;">📊 Results Summary</h3>
        <div class="info-grid">
            <div class="metric"><span class="metric-label">✅ Tier A Found:</span> <span class="success">{tier_a_count}</span> (>70% win rate)</div>
            <div class="metric"><span class="metric-label">⚠️ Tier B Found:</span> <span class="warning">{tier_b_count}</span> (55-70% win rate)</div>
            <div class="metric"><span class="metric-label">❌ Tier C Found:</span> <span class="danger">{tier_c_count}</span> (<55% win rate)</div>
            <div class="metric"><span class="metric-label">Hit Rate:</span> {(tier_a_count / symbols_tested * 100) if symbols_tested > 0 else 0:.1f}%</div>
        </div>
    </div>
"""

        # Add Tier A results (FULL DETAILS)
        if tier_a and tier_a_count > 0:
            html += f"""
    <div class="tier-section tier-a">
        <h3 style="color: #16a34a;">✅ Tier A Stocks ({tier_a_count}) - Win Rate >70%</h3>
        <p style="margin-bottom: 20px; color: #166534;"><strong>QUALIFIED FOR TRADING</strong> - High win rate, excellent returns, strong risk-adjusted performance</p>
"""
            for stock in tier_a:
                # Handle both string tickers and dict format
                if isinstance(stock, str):
                    stock_data = stock_data_map.get(stock, {})
                    ticker = stock
                else:
                    stock_data = stock
                    ticker = stock.get('ticker', 'N/A')

                win_rate = stock_data.get('win_rate', 0)
                total_return = stock_data.get('total_return', 0)
                sharpe = stock_data.get('sharpe_ratio', 0)
                trades = stock_data.get('total_trades', 0)
                avg_gain = stock_data.get('avg_gain', 0)
                avg_loss = stock_data.get('avg_loss', 0)

                html += f"""
        <div class="stock">
            <div class="stock-name">{ticker}</div>
            <div class="info-grid">
                <div class="metric"><span class="metric-label">Win Rate:</span> <span class="success">{win_rate:.1f}%</span></div>
                <div class="metric"><span class="metric-label">Total Return:</span> <span class="{'success' if total_return > 0 else 'danger'}">{total_return:+.2f}%</span></div>
                <div class="metric"><span class="metric-label">Sharpe Ratio:</span> {sharpe:.2f}</div>
                <div class="metric"><span class="metric-label">Total Trades:</span> {trades}</div>
                <div class="metric"><span class="metric-label">Avg Gain:</span> <span class="success">+{avg_gain:.2f}%</span></div>
                <div class="metric"><span class="metric-label">Avg Loss:</span> <span class="danger">{avg_loss:.2f}%</span></div>
            </div>
        </div>
"""
            html += """
    </div>
"""
        else:
            html += """
    <div class="tier-section">
        <h3>No Tier A Stocks Found</h3>
        <p>None of the tested symbols met the Tier A criteria (>70% win rate, >5% return, >5.0 Sharpe).</p>
    </div>
"""

        # Add Tier B results (SHOW ALL)
        if tier_b and tier_b_count > 0:
            html += f"""
    <div class="tier-section tier-b">
        <h3 style="color: #f59e0b;">⚠️ Tier B Stocks ({tier_b_count}) - Win Rate 55-70%</h3>
        <p style="margin-bottom: 20px; color: #92400e;">Marginal performance - monitored but not actively traded</p>
"""
            for stock in tier_b:
                # Handle both string tickers and dict format
                if isinstance(stock, str):
                    stock_data = stock_data_map.get(stock, {})
                    ticker = stock
                else:
                    stock_data = stock
                    ticker = stock.get('ticker', 'N/A')

                win_rate = stock_data.get('win_rate', 0)
                total_return = stock_data.get('total_return', 0)
                trades = stock_data.get('total_trades', 0)

                html += f"""
        <div class="stock">
            <div class="stock-name">{ticker}</div>
            <div class="info-grid">
                <div class="metric"><span class="metric-label">Win Rate:</span> <span class="warning">{win_rate:.1f}%</span></div>
                <div class="metric"><span class="metric-label">Total Return:</span> {total_return:+.2f}%</div>
                <div class="metric"><span class="metric-label">Trades:</span> {trades}</div>
            </div>
        </div>
"""
            html += """
    </div>
"""

        # Add Tier C results (SHOW WORST PERFORMERS)
        if tier_c and tier_c_count > 0:
            html += f"""
    <div class="tier-section tier-c">
        <h3 style="color: #dc2626;">❌ Tier C Stocks ({tier_c_count}) - Win Rate <55%</h3>
        <p style="margin-bottom: 20px; color: #991b1b;"><strong>AVOID TRADING</strong> - These stocks do not meet quality thresholds</p>
"""
            # Show worst 10 performers
            for stock in tier_c[:10]:
                # Handle both string tickers and dict format
                if isinstance(stock, str):
                    stock_data = stock_data_map.get(stock, {})
                    ticker = stock
                else:
                    stock_data = stock
                    ticker = stock.get('ticker', 'N/A')

                win_rate = stock_data.get('win_rate', 0)
                total_return = stock_data.get('total_return', 0)
                trades = stock_data.get('total_trades', 0)

                html += f"""
        <div class="stock">
            <div class="stock-name">{ticker}</div>
            <div class="info-grid">
                <div class="metric"><span class="metric-label">Win Rate:</span> <span class="danger">{win_rate:.1f}%</span></div>
                <div class="metric"><span class="metric-label">Total Return:</span> <span class="danger">{total_return:+.2f}%</span></div>
                <div class="metric"><span class="metric-label">Trades:</span> {trades}</div>
            </div>
        </div>
"""
            if tier_c_count > 10:
                html += f"""
        <p style="font-size: 0.95em; margin-top: 15px; color: #666;"><em>... and {tier_c_count - 10} more poor performers (see full report in logs)</em></p>
"""
            html += """
    </div>
"""

        # Add detailed recommendation section
        recommendation = results.get('recommendation', 'See full experiment results for details')
        html += f"""
    <div class="section" style="background: #eff6ff; border-left-color: #0284c7;">
        <h3 style="color: #0284c7;">💡 Recommendation & Next Steps</h3>
        <p style="font-size: 1.05em;"><strong>{recommendation}</strong></p>

        <h4 style="margin-top: 20px; color: #0284c7;">Criteria for Tier A Classification:</h4>
        <ul style="margin: 10px 0; padding-left: 25px;">
            <li><strong>Win Rate:</strong> Must be >70% (CRITICAL)</li>
            <li><strong>Total Return:</strong> Must be >5% over backtest period</li>
            <li><strong>Sharpe Ratio:</strong> Preferably >5.0 (risk-adjusted returns)</li>
            <li><strong>Minimum Trades:</strong> At least 5 trades (statistical significance)</li>
        </ul>

        <h4 style="margin-top: 20px; color: #0284c7;">Research Basis:</h4>
        <ul style="margin: 10px 0; padding-left: 25px;">
            <li><strong>Signal Quality:</strong> High-probability trade setups identified through continuous research</li>
            <li><strong>Behavioral Finance:</strong> Panic sells often overreact due to emotional selling, creating buying opportunities</li>
            <li><strong>Time-Decay Exits:</strong> Based on empirical observation that reversal momentum fades over time</li>
            <li><strong>Stock-Specific Optimization:</strong> Each stock has unique characteristics optimized through experimentation</li>
        </ul>
    </div>

    <div class="footer">
        <h4 style="margin-top: 0; color: #1a1a1a;">📁 Full Results Available</h4>
        <p><strong>JSON Report:</strong> logs/experiments/{experiment_id.lower().replace('-', '_')}_*.json</p>
        <p><strong>Contains:</strong> Complete trade-by-trade data, entry/exit prices, signal details, statistical analysis</p>
        <hr style="margin: 20px 0; border: none; border-top: 1px solid #ddd;">
        <p style="font-size: 0.95em;"><em>Automated experiment report from Proteus Trading System v10.0</em></p>
        <p style="font-size: 0.95em;"><em>Dashboard: <a href="http://localhost:5000">http://localhost:5000</a></em></p>
        <p style="font-size: 0.9em; color: #999; margin-top: 15px;">Questions? Check the CAPABILITIES.md documentation or experiment findings files.</p>
    </div>
</body>
</html>
"""
        return html

    def _get_period_dates(self, period: str) -> str:
        """Get approximate date range for period."""
        from datetime import datetime, timedelta

        end_date = datetime.now()

        if period == '1y':
            start_date = end_date - timedelta(days=365)
        elif period == '2y':
            start_date = end_date - timedelta(days=730)
        elif period == '3y':
            start_date = end_date - timedelta(days=1095)
        elif period == '5y':
            start_date = end_date - timedelta(days=1825)
        else:
            return "variable period"

        return f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"


if __name__ == '__main__':
    print("Testing SendGrid...")
    notifier = SendGridNotifier()

    if not SENDGRID_AVAILABLE:
        print("\nSendGrid not installed. To install:")
        print("  pip install sendgrid")
    elif not notifier.is_enabled():
        print("\nSendGrid not configured. Setup:")
        print("1. Sign up at https://sendgrid.com (free)")
        print("2. Get API key from Settings > API Keys")
        print("3. Add to email_config.json:")
        print('   "sendgrid_api_key": "YOUR_API_KEY_HERE"')
    else:
        notifier.send_test_email()

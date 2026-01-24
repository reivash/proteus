"""Send bearish market prediction email."""
from src.notifications.sendgrid_notifier import SendGridNotifier, SENDGRID_AVAILABLE
from src.analysis.unified_regime_detector import UnifiedRegimeDetector, DetectionMethod
from datetime import datetime

print('Generating bearish market prediction email...')

# Check if email is configured
notifier = SendGridNotifier()
if not notifier.is_enabled():
    print('[ERROR] Email not configured. Please check email_config.json')
    print('Required: sendgrid_api_key and recipient_email')
    exit(1)

# Get regime data
detector = UnifiedRegimeDetector(method=DetectionMethod.ENSEMBLE)
result = detector.detect_regime()

# Calculate combined risk multiplier
combined_mult = (result.hierarchical_risk_mult * result.macro_risk_mult *
                 result.disagreement_risk_mult * result.correlation_risk_mult)

# Bear outlook
bear_score = result.early_warning_score
if bear_score >= 60:
    bear_outlook = 'HIGH RISK - Significant bearish pressure detected'
    bear_color = '#dc2626'
elif bear_score >= 40:
    bear_outlook = 'ELEVATED RISK - Bearish signals emerging'
    bear_color = '#f59e0b'
elif bear_score >= 25:
    bear_outlook = 'MODERATE RISK - Some bearish indicators present'
    bear_color = '#3b82f6'
else:
    bear_outlook = 'LOW RISK - No significant bearish pressure'
    bear_color = '#10b981'

# Build email content
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
subject = f'Proteus Bearish Market Prediction - {result.early_warning_level} - {timestamp}'

# Build triggers HTML
triggers_html = ""
if result.early_warning_triggers:
    for trigger in result.early_warning_triggers[:8]:
        triggers_html += f'<div class="metric">- {trigger}</div>'
else:
    triggers_html = '<div class="metric">No active triggers</div>'

html_content = f'''
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; color: #333; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                 color: white; padding: 25px; border-radius: 8px; margin-bottom: 25px; }}
        .section {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0;
                   border-left: 5px solid #667eea; }}
        .bear-warning {{ background: #fef2f2; border-left-color: {bear_color}; }}
        .metric {{ margin: 8px 0; }}
        .metric-label {{ color: #666; font-weight: 500; }}
        .success {{ color: #10b981; font-weight: bold; }}
        .warning {{ color: #f59e0b; font-weight: bold; }}
        .danger {{ color: #dc2626; font-weight: bold; }}
        .footer {{ color: #666; font-size: 0.9em; margin-top: 30px; padding-top: 20px;
                  border-top: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="header">
        <h1 style="margin: 0;">Proteus Bearish Market Prediction</h1>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">Comprehensive Market Risk Assessment</p>
        <p style="font-size: 0.9em; opacity: 0.8;">{timestamp}</p>
    </div>

    <div class="section bear-warning">
        <h2 style="color: {bear_color}; margin-top: 0;">Bear Outlook: {bear_outlook}</h2>
        <div style="font-size: 2em; font-weight: bold; color: {bear_color};">
            Bear Score: {result.early_warning_score:.0f}/100
        </div>
        <div class="metric"><span class="metric-label">Alert Level:</span> <strong>{result.early_warning_level}</strong></div>
    </div>

    <div class="section">
        <h3 style="margin-top: 0; color: #667eea;">Regime Analysis</h3>
        <div class="metric"><span class="metric-label">Final Regime:</span> <strong>{result.regime.upper()}</strong> ({result.confidence*100:.0f}% confidence)</div>
        <div class="metric"><span class="metric-label">HMM Detection:</span> {result.hmm_regime.upper()} ({result.hmm_confidence*100:.0f}%)</div>
        <div class="metric"><span class="metric-label">Rule-based:</span> {result.rule_regime.upper()} ({result.rule_confidence*100:.0f}%)</div>
        <div class="metric"><span class="metric-label">Agreement:</span> {"YES" if result.agreement else "NO"}</div>
        <div class="metric"><span class="metric-label">Transition Signal:</span> {result.transition_signal}</div>
    </div>

    <div class="section">
        <h3 style="margin-top: 0; color: #8b5cf6;">Hierarchical HMM (Meta-Regime)</h3>
        <div class="metric"><span class="metric-label">Meta-Regime:</span> <strong>{result.meta_regime.upper()}</strong></div>
        <div class="metric"><span class="metric-label">Meta Confidence:</span> {result.meta_confidence*100:.0f}%</div>
        <div class="metric"><span class="metric-label">Position Size Multiplier:</span> {result.hierarchical_risk_mult:.2f}</div>
    </div>

    <div class="section">
        <h3 style="margin-top: 0; color: #059669;">FRED Macro (Recession Risk)</h3>
        <div class="metric"><span class="metric-label">Recession Probability:</span> <strong>{result.recession_probability:.0f}%</strong></div>
        <div class="metric"><span class="metric-label">Recession Signal:</span> {result.recession_signal}</div>
        <div class="metric"><span class="metric-label">Macro Risk Multiplier:</span> {result.macro_risk_mult:.2f}</div>
    </div>

    <div class="section">
        <h3 style="margin-top: 0; color: #0284c7;">Correlation Regime</h3>
        <div class="metric"><span class="metric-label">Average Correlation:</span> {result.avg_correlation:.2f}</div>
        <div class="metric"><span class="metric-label">Correlation Regime:</span> {result.correlation_regime}</div>
        <div class="metric"><span class="metric-label">Correlation Risk Mult:</span> {result.correlation_risk_mult:.2f}</div>
    </div>

    <div class="section">
        <h3 style="margin-top: 0; color: #f59e0b;">Model Disagreement</h3>
        <div class="metric"><span class="metric-label">Disagreement Level:</span> {result.model_disagreement*100:.0f}%</div>
        <div class="metric"><span class="metric-label">Disagreement Risk Mult:</span> {result.disagreement_risk_mult:.2f}</div>
        <div class="metric"><span class="metric-label">Note:</span> {result.disagreement_note}</div>
    </div>

    <div class="section">
        <h3 style="margin-top: 0; color: #dc2626;">Active Bear Triggers</h3>
        {triggers_html}
    </div>

    <div class="section" style="background: #e0f2fe; border-left-color: #0284c7;">
        <h3 style="margin-top: 0; color: #0284c7;">Combined Risk Assessment</h3>
        <div style="font-size: 1.5em; font-weight: bold; margin: 15px 0;">
            Combined Position Multiplier: {combined_mult:.2f}
        </div>
        <div class="metric"><span class="metric-label">VIX Level:</span> {result.vix_level:.1f}</div>
        <div class="metric" style="margin-top: 15px;"><strong>Recommendation:</strong> {result.recommendation}</div>
    </div>

    <div class="footer">
        <p><em>Automated bearish market prediction from Proteus Trading System</em></p>
        <p><em>Research basis: Hierarchical HMM (MDPI 2025), FRED Macro (Chen 2009), Correlation Regime Analysis</em></p>
    </div>
</body>
</html>
'''

# Send email using SendGrid directly
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Email, To, Content

    message = Mail(
        from_email=Email(notifier.config.get('sender_email', 'proteus@trading.local')),
        to_emails=To(notifier.config['recipient_email']),
        subject=subject,
        html_content=Content('text/html', html_content)
    )

    sg = SendGridAPIClient(notifier.config['sendgrid_api_key'])
    response = sg.send(message)

    print(f'[EMAIL] Bearish market prediction sent successfully!')
    print(f'[EMAIL] Recipient: {notifier.config["recipient_email"]}')
    print(f'[EMAIL] Subject: {subject}')
    print(f'[EMAIL] Status: {response.status_code}')

except Exception as e:
    print(f'[ERROR] Failed to send email: {e}')

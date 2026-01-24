"""Run bearish market prediction and send via Mailjet."""
import sys
sys.path.insert(0, 'src')
import json
from datetime import datetime
from mailjet_rest import Client
from analysis.unified_regime_detector import UnifiedRegimeDetector, DetectionMethod

print('Running bearish market detection...')

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

print(f'Bear Score: {bear_score:.0f}/100')
print(f'Alert Level: {result.early_warning_level}')
print(f'Regime: {result.regime.upper()} ({result.confidence*100:.0f}%)')
print(f'VIX: {result.vix_level:.1f}')
print(f'Triggers: {len(result.early_warning_triggers)}')

# Build triggers HTML
triggers_html = ''
if result.early_warning_triggers:
    for trigger in result.early_warning_triggers[:10]:
        triggers_html += f'<div style="margin: 8px 0;">- {trigger}</div>'
else:
    triggers_html = '<div style="margin: 8px 0;">No active triggers</div>'

# Build email
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
subject = f'Proteus Bearish Market Prediction - {result.early_warning_level} - {timestamp}'

html_content = f'''
<html>
<body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6; max-width: 650px; margin: 0 auto;">
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 8px; margin-bottom: 25px;">
        <h1 style="margin: 0;">Proteus Bearish Market Prediction</h1>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">Comprehensive Market Risk Assessment</p>
        <p style="font-size: 0.9em; opacity: 0.8;">{timestamp}</p>
    </div>

    <div style="background: #fef2f2; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 5px solid {bear_color};">
        <h2 style="color: {bear_color}; margin-top: 0;">Bear Outlook: {bear_outlook}</h2>
        <div style="font-size: 2em; font-weight: bold; color: {bear_color};">
            Bear Score: {result.early_warning_score:.0f}/100
        </div>
        <div style="margin: 8px 0;"><span style="color: #666;">Alert Level:</span> <strong>{result.early_warning_level}</strong></div>
    </div>

    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #667eea;">
        <h3 style="margin-top: 0; color: #667eea;">Regime Analysis</h3>
        <div style="margin: 8px 0;"><span style="color: #666;">Final Regime:</span> <strong>{result.regime.upper()}</strong> ({result.confidence*100:.0f}% confidence)</div>
        <div style="margin: 8px 0;"><span style="color: #666;">HMM Detection:</span> {result.hmm_regime.upper()} ({result.hmm_confidence*100:.0f}%)</div>
        <div style="margin: 8px 0;"><span style="color: #666;">Rule-based:</span> {result.rule_regime.upper()} ({result.rule_confidence*100:.0f}%)</div>
        <div style="margin: 8px 0;"><span style="color: #666;">Agreement:</span> {"YES" if result.agreement else "NO"}</div>
        <div style="margin: 8px 0;"><span style="color: #666;">Transition Signal:</span> {result.transition_signal}</div>
    </div>

    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #8b5cf6;">
        <h3 style="margin-top: 0; color: #8b5cf6;">Hierarchical HMM (Meta-Regime)</h3>
        <div style="margin: 8px 0;"><span style="color: #666;">Meta-Regime:</span> <strong>{result.meta_regime.upper()}</strong></div>
        <div style="margin: 8px 0;"><span style="color: #666;">Meta Confidence:</span> {result.meta_confidence*100:.0f}%</div>
        <div style="margin: 8px 0;"><span style="color: #666;">Position Size Multiplier:</span> {result.hierarchical_risk_mult:.2f}</div>
    </div>

    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #059669;">
        <h3 style="margin-top: 0; color: #059669;">FRED Macro (Recession Risk)</h3>
        <div style="margin: 8px 0;"><span style="color: #666;">Recession Probability:</span> <strong>{result.recession_probability:.0f}%</strong></div>
        <div style="margin: 8px 0;"><span style="color: #666;">Recession Signal:</span> {result.recession_signal}</div>
        <div style="margin: 8px 0;"><span style="color: #666;">Macro Risk Multiplier:</span> {result.macro_risk_mult:.2f}</div>
    </div>

    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #0284c7;">
        <h3 style="margin-top: 0; color: #0284c7;">Correlation Regime</h3>
        <div style="margin: 8px 0;"><span style="color: #666;">Average Correlation:</span> {result.avg_correlation:.2f}</div>
        <div style="margin: 8px 0;"><span style="color: #666;">Correlation Regime:</span> {result.correlation_regime}</div>
        <div style="margin: 8px 0;"><span style="color: #666;">Correlation Risk Mult:</span> {result.correlation_risk_mult:.2f}</div>
    </div>

    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #f59e0b;">
        <h3 style="margin-top: 0; color: #f59e0b;">Model Disagreement</h3>
        <div style="margin: 8px 0;"><span style="color: #666;">Disagreement Level:</span> {result.model_disagreement*100:.0f}%</div>
        <div style="margin: 8px 0;"><span style="color: #666;">Disagreement Risk Mult:</span> {result.disagreement_risk_mult:.2f}</div>
        <div style="margin: 8px 0;"><span style="color: #666;">Note:</span> {result.disagreement_note}</div>
    </div>

    <div style="background: #fef2f2; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #dc2626;">
        <h3 style="margin-top: 0; color: #dc2626;">Active Bear Triggers</h3>
        {triggers_html}
    </div>

    <div style="background: #e0f2fe; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #0284c7;">
        <h3 style="margin-top: 0; color: #0284c7;">Combined Risk Assessment</h3>
        <div style="font-size: 1.5em; font-weight: bold; margin: 15px 0;">
            Combined Position Multiplier: {combined_mult:.2f}
        </div>
        <div style="margin: 8px 0;"><span style="color: #666;">VIX Level:</span> {result.vix_level:.1f}</div>
        <div style="margin: 8px 0; margin-top: 15px;"><strong>Recommendation:</strong> {result.recommendation}</div>
    </div>

    <div style="color: #666; font-size: 0.9em; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd;">
        <p><em>Automated bearish market prediction from Proteus Trading System</em></p>
        <p><em>Research basis: Hierarchical HMM, FRED Macro, Correlation Regime Analysis</em></p>
    </div>
</body>
</html>
'''

# Load config and send via Mailjet
with open('email_config.json') as f:
    config = json.load(f)

print(f'\nSending email via Mailjet...')
print(f'From: {config["sender_email"]}')
print(f'To: {config["recipient_email"]}')

mailjet = Client(auth=(config['mailjet_api_key'], config['mailjet_secret_key']), version='v3.1')

data = {
    'Messages': [
        {
            'From': {
                'Email': config['sender_email'],
                'Name': 'Proteus Trading'
            },
            'To': [
                {
                    'Email': config['recipient_email'],
                    'Name': 'Javier'
                }
            ],
            'Subject': subject,
            'HTMLPart': html_content
        }
    ]
}

result_email = mailjet.send.create(data=data)
print(f'\nMailjet Response:')
print(f'Status Code: {result_email.status_code}')
print(f'Response: {result_email.json()}')

if result_email.status_code == 200:
    print(f'\n*** Bearish market prediction sent successfully! ***')
else:
    print(f'\n*** Email failed! ***')

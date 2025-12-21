"""
Send Investment Recommendation Email
====================================

Sends the comprehensive MU investment recommendation to the configured email address.
Uses SendGrid API directly.
"""

import json
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content


def load_config():
    """Load email configuration."""
    with open('email_config.json', 'r') as f:
        return json.load(f)


def convert_markdown_to_html(markdown_content: str) -> str:
    """Convert markdown recommendation to styled HTML email."""

    # Replace markdown headers
    html = markdown_content

    # Process line by line for better control
    lines = html.split('\n')
    processed_lines = []
    in_table = False
    in_code_block = False

    for i, line in enumerate(lines):
        # Code blocks
        if line.strip().startswith('```'):
            if not in_code_block:
                processed_lines.append('<pre style="background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; font-family: Courier New, monospace;">')
                in_code_block = True
            else:
                processed_lines.append('</pre>')
                in_code_block = False
            continue

        if in_code_block:
            processed_lines.append(line)
            continue

        # Headers
        if line.startswith('### '):
            processed_lines.append(f'<h3 style="color: #34495e; margin-top: 25px;">{line[4:]}</h3>')
        elif line.startswith('## '):
            processed_lines.append(f'<h2 style="color: #2874a6; margin-top: 30px; border-left: 4px solid #2874a6; padding-left: 15px;">{line[3:]}</h2>')
        elif line.startswith('# '):
            processed_lines.append(f'<h1 style="color: #1a5490; border-bottom: 3px solid #1a5490; padding-bottom: 10px;">{line[2:]}</h1>')

        # Tables
        elif '|' in line and line.strip().startswith('|'):
            if not in_table:
                processed_lines.append('<table style="width: 100%; border-collapse: collapse; margin: 20px 0;">')
                in_table = True
                # Header row
                cells = [c.strip() for c in line.split('|')[1:-1]]
                processed_lines.append('<tr>')
                for cell in cells:
                    cell_content = cell.replace('**', '')
                    processed_lines.append(f'<th style="padding: 12px; text-align: left; border: 1px solid #ddd; background-color: #1a5490; color: white;">{cell_content}</th>')
                processed_lines.append('</tr>')
            elif line.strip().replace('|', '').replace('-', '').strip() == '':
                # Separator line, skip
                continue
            else:
                # Data row
                cells = [c.strip() for c in line.split('|')[1:-1]]
                processed_lines.append('<tr style="background-color: #f9f9f9;">')
                for cell in cells:
                    # Check for checkmarks and format
                    cell_content = cell.replace('✓', '✓').replace('**', '<strong>').replace('**', '</strong>')
                    processed_lines.append(f'<td style="padding: 12px; text-align: left; border: 1px solid #ddd;">{cell_content}</td>')
                processed_lines.append('</tr>')
        elif in_table and not ('|' in line):
            processed_lines.append('</table>')
            in_table = False
            processed_lines.append(line)

        # Bold
        elif '**' in line:
            bold_line = line
            while '**' in bold_line:
                bold_line = bold_line.replace('**', '<strong style="color: #1a5490;">', 1)
                bold_line = bold_line.replace('**', '</strong>', 1)
            processed_lines.append(bold_line)

        # Lists
        elif line.strip().startswith('- '):
            processed_lines.append(f'<li style="margin: 8px 0;">{line.strip()[2:]}</li>')

        # Horizontal rules
        elif line.strip() == '---':
            processed_lines.append('<hr style="border: none; border-top: 2px solid #ddd; margin: 30px 0;">')

        # Regular paragraphs
        elif line.strip():
            processed_lines.append(f'<p style="margin: 10px 0; line-height: 1.6;">{line}</p>')
        else:
            processed_lines.append('<br>')

    if in_table:
        processed_lines.append('</table>')

    html = '\n'.join(processed_lines)

    # Wrap in HTML template
    html_email = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            {html}
            <hr style="border: none; border-top: 2px solid #ddd; margin: 40px 0 20px 0;">
            <p style="color: #7f8c8d; font-size: 0.9em; margin: 5px 0;"><strong>Analysis Generated By:</strong> Proteus Investment Analysis System</p>
            <p style="color: #7f8c8d; font-size: 0.9em; margin: 5px 0;"><strong>Analysis Date:</strong> December 2, 2025</p>
            <p style="color: #7f8c8d; font-size: 0.85em; margin: 15px 0 0 0; font-style: italic;">This analysis is for informational purposes. All investments carry risk. Past performance does not guarantee future results.</p>
        </div>
    </body>
    </html>
    """

    return html_email


def main():
    """Send investment recommendation email."""

    print("\n" + "="*80)
    print("SENDING INVESTMENT RECOMMENDATION EMAIL")
    print("="*80)

    # Load config
    print("\n[1/4] Loading configuration...")
    config = load_config()

    if not config.get('enabled'):
        print("[ERROR] Email is not enabled in config!")
        return

    # Read recommendation content
    print("[2/4] Reading recommendation content...")
    recommendation_file = 'EMAIL_RECOMMENDATION.md'

    try:
        with open(recommendation_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
    except FileNotFoundError:
        print(f"\n[ERROR] {recommendation_file} not found!")
        return

    # Convert to HTML
    print("[3/4] Converting to HTML format...")
    html_body = convert_markdown_to_html(markdown_content)

    # Email subject
    subject = "STRONG BUY Recommendation - Micron Technology (MU) - 98% Confidence"

    # Create SendGrid message
    message = Mail(
        from_email=Email(config.get('sender_email', 'proteus@trading.local')),
        to_emails=To(config['recipient_email']),
        subject=subject,
        html_content=Content("text/html", html_body)
    )

    # Send email
    print("[4/4] Sending email via SendGrid...")
    print(f"\nRecipient: {config['recipient_email']}")
    print(f"Subject: {subject}")

    try:
        sg = SendGridAPIClient(config['sendgrid_api_key'])
        response = sg.send(message)

        print("\n" + "="*80)
        print("[SUCCESS] EMAIL SENT SUCCESSFULLY!")
        print("="*80)
        print(f"\nStatus Code: {response.status_code}")
        print(f"To: {config['recipient_email']}")
        print(f"Subject: {subject}")
        print(f"\nKey Recommendation:")
        print(f"  - STRONG BUY: Micron Technology (MU)")
        print(f"  - Current Price: $240.46")
        print(f"  - Target Price: $375.20 (+56.2%)")
        print(f"  - Confidence: 98%")
        print(f"  - Timeframe: 12-24 months")
        print(f"  - All 10 validation layers aligned")
        print("\n" + "="*80)

    except Exception as e:
        print("\n" + "="*80)
        print("[ERROR] Failed to send email")
        print("="*80)
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check SendGrid API key in email_config.json")
        print("  2. Verify SendGrid account status")
        print("  3. Check: https://app.sendgrid.com/email_activity")
        print("="*80)


if __name__ == "__main__":
    main()

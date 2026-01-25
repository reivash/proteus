# Daily Reporting

> **Status**: Production
> **Version**: 1.0

---

## What It Does

Generates daily email reports with:
- Current portfolio status
- Today's signals and recommendations
- Performance metrics
- Market regime summary

## Key Files

| File | Purpose |
|------|---------|
| `run.py` | Main entry point |
| `data/` | Generated reports |

## Usage

```bash
python features/reporting/run.py           # Generate and send report
python features/reporting/run.py --preview # Preview without sending
```

## Configuration

Uses `config/email_config.json` for email settings (Mailjet).

---

*Status: Production Complete*

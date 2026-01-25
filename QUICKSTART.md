# Proteus Quick Start Guide

## Prerequisites
- Python 3.8+ installed
- PowerShell with Claude Code CLI
- This CLI instance (where Owl will run)

## Setup Instructions

### Step 1: Open PowerShell with Claude
1. Open a new PowerShell window
2. Navigate to the proteus directory: `cd C:\Users\javie\Desktop\proteus`
3. Start Claude: `claude` (or however you normally start it)
4. Keep this window open

### Step 2: Initialize Zeus in PowerShell Claude
In the PowerShell Claude window, paste the Zeus initialization prompt:

```
Read and load: proteus/agents/zeus_starter_prompt.md
```

Then follow the instructions to initialize Zeus as the coordinator.

### Step 3: Start Owl Agent (This Window)
In this CLI window (where you're currently working), run:

```bash
python proteus/agents/owl.py
```

This will:
- Start the Owl orchestrator
- Send nudges to Zeus every 30 minutes (or your chosen interval)
- Keep the system running continuously

### Step 4: Monitor Progress
- **PowerShell Window**: Zeus will respond to nudges and coordinate agents
- **This Window**: Owl will log all nudges and system status
- **Logs**: Check `proteus/logs/` for detailed activity logs

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     THIS CLI WINDOW                          │
│                                                               │
│                    ┌──────────────┐                          │
│                    │  OWL AGENT   │                          │
│                    │ (Orchestrator)│                          │
│                    └──────┬───────┘                          │
│                           │                                   │
│                           │ Nudges every 30 min              │
│                           ▼                                   │
└───────────────────────────┼───────────────────────────────────┘
                            │
                            │
┌───────────────────────────┼───────────────────────────────────┐
│                  POWERSHELL CLAUDE WINDOW                     │
│                           │                                   │
│                    ┌──────▼───────┐                          │
│                    │  ZEUS AGENT  │                          │
│                    │ (Coordinator) │                          │
│                    └──────┬───────┘                          │
│                           │                                   │
│          ┌────────────────┼────────────────┐                │
│          │                │                 │                │
│     ┌────▼─────┐   ┌─────▼──────┐   ┌─────▼──────┐        │
│     │ HERMES   │   │  ATHENA    │   │ PROMETHEUS │        │
│     │(Research)│   │(Feasibility)│   │  (Build)   │        │
│     └──────────┘   └────────────┘   └────────────┘        │
└───────────────────────────────────────────────────────────────┘
```

## Workflow

1. **Owl** sends a nudge to PowerShell (via file: `proteus/data/current_nudge.txt`)
2. **Zeus** (in PowerShell) reads the nudge
3. **Zeus** decides which agent to activate based on current phase
4. Assigned agent (**Hermes**, **Athena**, or **Prometheus**) performs task
5. Results are logged and status is updated
6. Wait for next Owl nudge
7. Repeat

## Expected Timeline

- **Cycle 1-2**: Prometheus builds baseline predictor
- **Cycle 3-5**: Hermes researches papers, Athena analyzes feasibility
- **Cycle 6+**: Prometheus implements improvements, system iterates

## Files to Monitor

- `proteus/logs/owl_*.log` - Owl agent activity
- `proteus/logs/experiments/` - Experiment results from Prometheus
- `proteus/docs/research/` - Research summaries from Hermes
- `proteus/data/owl_state.json` - Current system state

## Stopping the System

1. In this window (Owl): Press `Ctrl+C` to stop gracefully
2. In PowerShell: Zeus will remain idle until next nudge

You can restart Owl anytime and it will resume from the last cycle.

## Customization

Edit `proteus/agents/owl.py` to change:
- Nudge interval (default: 30 minutes)
- Log directory
- Nudge message format

## Troubleshooting

**Owl can't find PowerShell Claude:**
- Owl writes nudges to `proteus/data/current_nudge.txt`
- Manually read this file in PowerShell Claude if needed

**Zeus not responding:**
- Check if Zeus is initialized in PowerShell
- Verify Zeus read the coordinator agent instructions
- Read the current nudge file manually

**No progress:**
- Check logs for errors
- Verify all agent .md files are present
- Ensure proper permissions for file writing

## Next Steps

Once the system is running:
1. Monitor the first few cycles
2. Verify Zeus is coordinating properly
3. Check that Prometheus creates the baseline
4. Watch for research → feasibility → implementation flow
5. Adjust nudge interval if needed (faster for testing, slower for production)

Good luck with Proteus!

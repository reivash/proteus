# Switching Between Claude and Codex

This guide explains how to manually switch between Claude Code and Codex for the Proteus project.

## Current Setup

Proteus is designed to work with both Claude Code and Codex. Agent Owl monitors the PowerShell window and nudges the AI agent regardless of which one is running.

## Files for Each Agent

### Claude Code Setup
- **Agent Role**: `agents/zeus_coordinator_agent.md`
- **Starter Prompt**: `agents/zeus_starter_prompt.md`
- **Owl Config**: `../agent-owl/configs/proteus.json`
- **Screenshots**: `../agent-owl/screenshots/proteus/`

### Codex Setup
- **Agent Role**: `agents/zeus_coordinator_agent_codex.md`
- **Starter Prompt**: `agents/zeus_starter_prompt_codex.md`
- **Owl Config**: `../agent-owl/configs/proteus_codex.json`
- **Screenshots**: `../agent-owl/screenshots/proteus_codex/`

## How to Switch

### Switching from Claude to Codex

1. **In this window (Claude)**: Stop the current Agent Owl
   ```bash
   # Agent Owl will be stopped when you give the command
   ```

2. **In PowerShell**: Exit Claude Code
   - Press Ctrl+C or close the PowerShell window

3. **In PowerShell**: Start Codex
   ```powershell
   cd C:\Users\javie\Documents\GitHub\proteus
   codex  # or however you start Codex
   ```

4. **In Codex**: Initialize Zeus
   - Paste the content from `agents/zeus_starter_prompt_codex.md`

5. **In this window**: Start Agent Owl with Codex config
   ```bash
   cd ../agent-owl && python agent_owl.py --config configs/proteus_codex.json
   ```

### Switching from Codex to Claude

1. **In this window**: Stop the current Agent Owl
   - The active Agent Owl monitoring Codex will be stopped

2. **In PowerShell**: Exit Codex
   - Close the Codex session

3. **In PowerShell**: Start Claude Code
   ```powershell
   cd C:\Users\javie\Documents\GitHub\proteus
   claude
   ```

4. **In Claude**: Initialize Zeus
   - Paste the content from `agents/zeus_starter_prompt.md`

5. **In this window**: Start Agent Owl with Claude config
   ```bash
   cd ../agent-owl && python agent_owl.py --config configs/proteus.json
   ```

## State Continuity

Both Claude and Codex share the same:
- **data/state.json** - Current system state
- **docs/research/** - Research papers and summaries
- **docs/feasibility/** - Feasibility analyses
- **common/** - Implementation code
- **logs/** - Activity logs

This means you can switch between them seamlessly - they'll both continue from where the other left off.

## Agent Owl Detection

Agent Owl automatically detects PowerShell windows by process, not window title. This means:
- ✅ It will find the PowerShell regardless of what Claude/Codex is doing
- ✅ It will work with either Claude Code or Codex
- ✅ You don't need to change the window detection

## Tips

1. **Before Switching**: Make sure the current agent has updated `data/state.json` so the new agent knows where things stand

2. **Check Logs**: Review `logs/` to see what was accomplished before switching

3. **Manual Coordination**: If you switch mid-task, you may need to remind Zeus of the current context

4. **Quota Management**: Switch to Codex when Claude hits quota limits, then switch back when quota resets

## Current Agent Owl Process

To see which Agent Owl is currently running, check the background tasks in this window. Only one should be active at a time.

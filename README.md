<div align="center">
  <img src="assets/proteus-logo.png" alt="Proteus Logo" width="400"/>
</div>

---

# Proteus - Stock Market Predictor

---

## ⚠️ DISCLAIMER - READ BEFORE USING

**THIS IS A FUN EXPERIMENTAL PROJECT FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY.**

- 🚫 **NOT FINANCIAL ADVICE** - This system is an experiment in multi-agent AI collaboration, not a professional trading tool
- 🎲 **USE AT YOUR OWN RISK** - Any trading decisions based on this project are entirely your responsibility
- 🧪 **EXPERIMENTAL** - This is a research project exploring AI agent coordination and stock market analysis methodologies
- 💸 **NO GUARANTEES** - Stock markets are inherently unpredictable. This project makes no claims of profitability or accuracy
- 📚 **EDUCATIONAL ONLY** - Treat this as a learning exercise in AI systems, not as investment guidance

**By using this project, you acknowledge that you understand these risks and will not hold the authors liable for any financial losses.**

---

## About

An agentic AI system for researching, analyzing, and improving stock market prediction methodologies.

## Architecture

Proteus uses a multi-agent architecture where specialized Claude agents collaborate to continuously improve prediction capabilities:

### Agent Hierarchy

1. **Owl Agent** (Orchestrator)
   - Runs locally to manage the PowerShell Claude instance
   - Nudges the coordinator at regular intervals
   - Monitors overall system health

2. **Zeus Agent** (Coordinator)
   - Manages workflow between all agents
   - Prioritizes tasks and research directions
   - Tracks progress and maintains state

3. **Hermes Agent** (Research)
   - Searches and reads academic papers on stock prediction
   - Summarizes key findings and methodologies
   - Identifies promising approaches

4. **Athena Agent** (Feasibility Analysis)
   - Evaluates research findings for implementability
   - Assesses data requirements and computational constraints
   - Recommends achievable strategies for Claude agents

5. **Prometheus Agent** (Implementation)
   - Implements feasible prediction strategies
   - Iteratively improves the predictor
   - Runs experiments and tracks performance

## Directory Structure

```
proteus/
├── agents/           # Agent configuration files
├── data/            # Market data and research papers
├── logs/            # Agent activity logs
├── docs/            # Research summaries and findings
└── src/             # Predictor implementation
```

## Workflow

1. Owl nudges Zeus at regular intervals
2. Zeus assigns tasks to Hermes (research), Athena (analysis), or Prometheus (implementation)
3. Hermes finds and summarizes relevant papers
4. Athena evaluates feasibility and creates implementation plans
5. Prometheus implements and tests improvements
6. Results feed back to Zeus for next iteration

## Running the System

Run the Owl agent from this CLI instance:
```bash
python proteus/agents/owl.py
```

The Owl will manage the PowerShell Claude instance automatically.

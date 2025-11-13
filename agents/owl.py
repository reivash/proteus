"""
Owl Agent - Orchestrator for Proteus Stock Market Predictor

This agent runs in the background and nudges the PowerShell Claude instance
at regular intervals to keep the multi-agent system running.
"""

import time
import datetime
import subprocess
import json
from pathlib import Path
from typing import Optional
import sys
import argparse

class OwlAgent:
    def __init__(
        self,
        nudge_interval_minutes: int = 30,
        log_dir: str = "proteus/logs",
        state_file: str = "proteus/data/owl_state.json"
    ):
        """
        Initialize the Owl Agent.

        Args:
            nudge_interval_minutes: How often to nudge Zeus (in minutes)
            log_dir: Directory for logs
            state_file: File to persist state
        """
        self.nudge_interval = nudge_interval_minutes * 60  # Convert to seconds
        self.log_dir = Path(log_dir)
        self.state_file = Path(state_file)
        self.cycle_count = 0
        self.start_time = datetime.datetime.now()

        # Create directories if they don't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Load or initialize state
        self.load_state()

    def load_state(self):
        """Load the agent state from file."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.cycle_count = state.get('cycle_count', 0)
                print(f"[OWL] Resumed from cycle {self.cycle_count}")
        else:
            self.save_state()

    def save_state(self):
        """Save the current state to file."""
        state = {
            'cycle_count': self.cycle_count,
            'last_nudge': datetime.datetime.now().isoformat(),
            'start_time': self.start_time.isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        print(log_message)

        # Also write to log file
        log_file = self.log_dir / f"owl_{datetime.date.today()}.log"
        with open(log_file, 'a') as f:
            f.write(log_message + '\n')

    def create_nudge_message(self) -> str:
        """Create the nudge message for Zeus."""
        elapsed = datetime.datetime.now() - self.start_time
        hours = elapsed.total_seconds() / 3600

        message = f"""
==============================================================
                    OWL NUDGE #{self.cycle_count}
==============================================================

Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
System Uptime: {hours:.1f} hours
Interval: {self.nudge_interval / 60:.0f} minutes

ZEUS: Please provide status update and take next action.

Review:
1. Current phase and progress
2. Active tasks and completions
3. Agent statuses (Hermes, Athena, Prometheus)
4. Performance metrics vs baseline
5. Next action to take

Respond with status report and execute next task in the workflow.
"""
        return message

    def send_nudge_to_powershell(self) -> bool:
        """
        Send a nudge to the PowerShell Claude instance.

        This uses Windows clipboard to send the message.
        User should have PowerShell with Claude open and ready.
        """
        try:
            nudge_message = self.create_nudge_message()

            # Copy message to clipboard
            subprocess.run(
                ['powershell', '-command', f'Set-Clipboard -Value "{nudge_message}"'],
                check=True,
                capture_output=True
            )

            self.log(f"Nudge #{self.cycle_count} copied to clipboard", "SUCCESS")
            self.log("PASTE into PowerShell Claude window now!", "ACTION")

            return True

        except Exception as e:
            self.log(f"Error sending nudge: {e}", "ERROR")
            return False

    def send_nudge_direct(self) -> bool:
        """
        Alternative: Create a file that can be read by PowerShell Claude.
        More reliable than clipboard for automation.
        """
        try:
            nudge_message = self.create_nudge_message()
            nudge_file = Path("proteus/data/current_nudge.txt")

            with open(nudge_file, 'w') as f:
                f.write(nudge_message)

            self.log(f"Nudge #{self.cycle_count} written to {nudge_file}", "SUCCESS")
            self.log("Read proteus/data/current_nudge.txt in PowerShell Claude", "ACTION")

            return True

        except Exception as e:
            self.log(f"Error creating nudge file: {e}", "ERROR")
            return False

    def run(self):
        """Main loop for the Owl agent."""
        self.log("="*60, "SYSTEM")
        self.log("OWL AGENT STARTED", "SYSTEM")
        self.log(f"Nudge interval: {self.nudge_interval / 60:.0f} minutes", "SYSTEM")
        self.log("="*60, "SYSTEM")
        self.log("")
        self.log("Press Ctrl+C to stop gracefully", "INFO")
        self.log("")

        try:
            while True:
                self.cycle_count += 1

                self.log(f"Starting cycle #{self.cycle_count}", "CYCLE")

                # Send nudge via file (more reliable than clipboard)
                if self.send_nudge_direct():
                    self.log("Waiting for next nudge interval...", "INFO")
                else:
                    self.log("Failed to send nudge, will retry next cycle", "WARN")

                # Save state after each cycle
                self.save_state()

                # Wait for next interval
                self.log(f"Next nudge in {self.nudge_interval / 60:.0f} minutes", "INFO")
                self.log("-" * 60)
                time.sleep(self.nudge_interval)

        except KeyboardInterrupt:
            self.log("", "SYSTEM")
            self.log("Received shutdown signal", "SYSTEM")
            self.shutdown()

    def shutdown(self):
        """Gracefully shutdown the Owl agent."""
        self.log("="*60, "SYSTEM")
        self.log("OWL AGENT SHUTTING DOWN", "SYSTEM")
        self.log(f"Total cycles completed: {self.cycle_count}", "SYSTEM")

        elapsed = datetime.datetime.now() - self.start_time
        self.log(f"Total runtime: {elapsed}", "SYSTEM")

        self.save_state()
        self.log("State saved successfully", "SYSTEM")
        self.log("="*60, "SYSTEM")
        sys.exit(0)


def main():
    """Main entry point for the Owl agent."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Proteus Owl Agent - Stock Market Predictor Orchestrator"
    )
    parser.add_argument(
        '--interval',
        '-i',
        type=int,
        default=30,
        help='Nudge interval in minutes (default: 30)'
    )
    args = parser.parse_args()

    print("""
    ============================================================

                      PROTEUS OWL AGENT

                 Stock Market Predictor Orchestrator

    ============================================================
    """)

    interval = args.interval
    print(f"Starting Owl with {interval}-minute intervals...")
    print("Make sure PowerShell Claude is open and ready!")
    print()

    # Start the agent
    owl = OwlAgent(nudge_interval_minutes=interval)
    owl.run()


if __name__ == "__main__":
    main()

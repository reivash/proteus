"""
Staged Experiment Runner

Manages controlled execution of experiments to prevent API rate limit exhaustion.
Runs maximum 3-5 experiments concurrently instead of 100+.

PROBLEM:
- 100+ concurrent experiments exhaust Yahoo Finance API rate limits
- 2,000 requests/hour limit
- 3.9 million API calls needed = 82 days to complete
- Many experiments failing mid-execution

SOLUTION:
- Queue-based execution with max concurrency limit
- Priority-based scheduling (critical experiments first)
- Automatic retry on failure
- Progress tracking and status reporting

Usage:
    python common/tools/experiment_runner.py --priority high --max-concurrent 3
    python common/tools/experiment_runner.py --experiments exp125,exp126,exp127
"""

import argparse
import subprocess
import time
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum


class ExperimentPriority(Enum):
    """Experiment priority levels."""
    CRITICAL = 1  # Must run (production-blocking)
    HIGH = 2      # Should run soon (high-impact)
    MEDIUM = 3    # Normal priority
    LOW = 4       # Can wait


class ExperimentStatus(Enum):
    """Experiment execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"


@dataclass
class Experiment:
    """Experiment metadata."""
    number: int
    name: str
    file_path: str
    priority: ExperimentPriority
    status: ExperimentStatus = ExperimentStatus.PENDING
    process: subprocess.Popen = None
    start_time: datetime = None
    end_time: datetime = None
    retries: int = 0


class ExperimentRunner:
    """
    Manages staged execution of experiments with controlled concurrency.
    """

    def __init__(self, max_concurrent: int = 3, max_retries: int = 2):
        """
        Initialize experiment runner.

        Args:
            max_concurrent: Maximum concurrent experiments (default: 3)
            max_retries: Maximum retries on failure (default: 2)
        """
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.repo_root = Path(__file__).parent.parent.parent
        self.experiments_dir = self.repo_root / "src" / "experiments"
        self.logs_dir = self.repo_root / "logs"
        self.queue: List[Experiment] = []
        self.running: List[Experiment] = []
        self.completed: List[Experiment] = []
        self.failed: List[Experiment] = []

    def discover_experiments(self, pattern: str = "exp*.py") -> List[str]:
        """
        Discover experiment files matching pattern.

        Args:
            pattern: Glob pattern for experiment files

        Returns:
            List of experiment file paths
        """
        experiment_files = list(self.experiments_dir.glob(pattern))
        return sorted([str(f) for f in experiment_files])

    def parse_experiment_metadata(self, file_path: str) -> Experiment:
        """
        Parse experiment metadata from file.

        Args:
            file_path: Path to experiment file

        Returns:
            Experiment object with metadata
        """
        # Extract experiment number from filename (e.g., exp125 -> 125)
        filename = Path(file_path).stem
        try:
            exp_number = int(filename.replace("exp", ""))
        except ValueError:
            exp_number = 999  # Unknown

        # Read first 50 lines for metadata
        try:
            with open(file_path, 'r') as f:
                lines = [next(f) for _ in range(50)]
                content = "".join(lines).lower()
        except:
            content = ""

        # Determine priority from keywords in docstring
        priority = ExperimentPriority.MEDIUM
        if any(kw in content for kw in ["critical", "production", "deploy"]):
            priority = ExperimentPriority.CRITICAL
        elif any(kw in content for kw in ["adaptive", "trinity", "risk management"]):
            priority = ExperimentPriority.HIGH
        elif any(kw in content for kw in ["baseline", "validation", "comprehensive"]):
            priority = ExperimentPriority.HIGH

        return Experiment(
            number=exp_number,
            name=filename,
            file_path=file_path,
            priority=priority
        )

    def load_experiments(self, experiment_names: List[str] = None):
        """
        Load experiments into queue.

        Args:
            experiment_names: Optional list of specific experiments to run
                            (e.g., ["exp125", "exp126", "exp127"])
        """
        if experiment_names:
            # Load specific experiments
            for name in experiment_names:
                file_path = self.experiments_dir / f"{name}.py"
                if file_path.exists():
                    exp = self.parse_experiment_metadata(str(file_path))
                    self.queue.append(exp)
                else:
                    print(f"[WARN] Experiment not found: {name}")
        else:
            # Load all experiments
            experiment_files = self.discover_experiments()
            for file_path in experiment_files:
                exp = self.parse_experiment_metadata(file_path)
                self.queue.append(exp)

        # Sort queue by priority (critical first) then by number (oldest first)
        self.queue.sort(key=lambda e: (e.priority.value, e.number))

        print(f"[OK] Loaded {len(self.queue)} experiments into queue")
        print(f"     Critical: {sum(1 for e in self.queue if e.priority == ExperimentPriority.CRITICAL)}")
        print(f"     High: {sum(1 for e in self.queue if e.priority == ExperimentPriority.HIGH)}")
        print(f"     Medium: {sum(1 for e in self.queue if e.priority == ExperimentPriority.MEDIUM)}")
        print(f"     Low: {sum(1 for e in self.queue if e.priority == ExperimentPriority.LOW)}")

    def start_experiment(self, experiment: Experiment) -> bool:
        """
        Start experiment execution.

        Args:
            experiment: Experiment to start

        Returns:
            True if started successfully
        """
        log_file = self.logs_dir / f"{experiment.name}.log"

        try:
            # Start experiment in background with output logging
            with open(log_file, 'w') as log_f:
                process = subprocess.Popen(
                    ["python", experiment.file_path],
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.repo_root)
                )

            experiment.process = process
            experiment.status = ExperimentStatus.RUNNING
            experiment.start_time = datetime.now()
            self.running.append(experiment)

            print(f"[START] {experiment.name} (PID: {process.pid}, Priority: {experiment.priority.name})")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to start {experiment.name}: {e}")
            experiment.status = ExperimentStatus.FAILED
            self.failed.append(experiment)
            return False

    def check_running_experiments(self):
        """
        Check status of running experiments and update queue.
        """
        still_running = []

        for exp in self.running:
            if exp.process is None:
                continue

            # Check if process is still running
            returncode = exp.process.poll()

            if returncode is None:
                # Still running
                still_running.append(exp)
            elif returncode == 0:
                # Completed successfully
                exp.status = ExperimentStatus.COMPLETED
                exp.end_time = datetime.now()
                self.completed.append(exp)
                duration = (exp.end_time - exp.start_time).total_seconds() / 60
                print(f"[DONE] {exp.name} (Duration: {duration:.1f}min)")
            else:
                # Failed
                exp.status = ExperimentStatus.FAILED
                exp.end_time = datetime.now()

                # Check if should retry
                if exp.retries < self.max_retries:
                    exp.retries += 1
                    exp.status = ExperimentStatus.PENDING
                    exp.process = None
                    exp.start_time = None
                    exp.end_time = None
                    self.queue.insert(0, exp)  # Add to front of queue
                    print(f"[RETRY] {exp.name} (Attempt {exp.retries + 1}/{self.max_retries + 1})")
                else:
                    self.failed.append(exp)
                    print(f"[FAIL] {exp.name} (Return code: {returncode})")

        self.running = still_running

    def run(self):
        """
        Main execution loop.
        Continuously starts experiments up to max_concurrent limit.
        """
        print(f"\n{'='*70}")
        print(f"STAGED EXPERIMENT RUNNER")
        print(f"{'='*70}")
        print(f"Max Concurrent: {self.max_concurrent}")
        print(f"Max Retries: {self.max_retries}")
        print(f"Queue Size: {len(self.queue)}")
        print(f"{'='*70}\n")

        while self.queue or self.running:
            # Check running experiments
            self.check_running_experiments()

            # Start new experiments if capacity available
            while len(self.running) < self.max_concurrent and self.queue:
                exp = self.queue.pop(0)
                self.start_experiment(exp)

            # Print status
            self.print_status()

            # Wait before next check
            time.sleep(10)

        # Final summary
        self.print_summary()

    def print_status(self):
        """Print current execution status."""
        print(f"\n[STATUS] Queue: {len(self.queue)} | "
              f"Running: {len(self.running)} | "
              f"Completed: {len(self.completed)} | "
              f"Failed: {len(self.failed)}")

        if self.running:
            print(f"  Currently running:")
            for exp in self.running:
                duration = (datetime.now() - exp.start_time).total_seconds() / 60
                print(f"    - {exp.name} ({duration:.1f}min)")

    def print_summary(self):
        """Print final execution summary."""
        print(f"\n{'='*70}")
        print(f"EXECUTION COMPLETE")
        print(f"{'='*70}")
        print(f"Total Experiments: {len(self.completed) + len(self.failed)}")
        print(f"Completed: {len(self.completed)}")
        print(f"Failed: {len(self.failed)}")
        print(f"{'='*70}\n")

        if self.failed:
            print("Failed experiments:")
            for exp in self.failed:
                print(f"  - {exp.name} (Retries: {exp.retries})")

        if self.completed:
            print(f"\nCompleted experiments:")
            for exp in self.completed:
                duration = (exp.end_time - exp.start_time).total_seconds() / 60
                print(f"  - {exp.name} ({duration:.1f}min)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Staged experiment runner')
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=3,
        help='Maximum concurrent experiments (default: 3)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
        help='Maximum retries on failure (default: 2)'
    )
    parser.add_argument(
        '--experiments',
        type=str,
        help='Comma-separated list of specific experiments to run (e.g., exp125,exp126)'
    )
    parser.add_argument(
        '--priority',
        type=str,
        choices=['critical', 'high', 'medium', 'low'],
        help='Only run experiments with this priority or higher'
    )

    args = parser.parse_args()

    # Initialize runner
    runner = ExperimentRunner(
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries
    )

    # Load experiments
    if args.experiments:
        exp_names = [name.strip() for name in args.experiments.split(',')]
        runner.load_experiments(exp_names)
    else:
        runner.load_experiments()

    # Filter by priority if specified
    if args.priority:
        priority_map = {
            'critical': ExperimentPriority.CRITICAL,
            'high': ExperimentPriority.HIGH,
            'medium': ExperimentPriority.MEDIUM,
            'low': ExperimentPriority.LOW
        }
        max_priority = priority_map[args.priority]
        runner.queue = [e for e in runner.queue if e.priority.value <= max_priority.value]
        print(f"[FILTER] Running only {args.priority.upper()} priority experiments ({len(runner.queue)} remaining)")

    # Run experiments
    if runner.queue:
        runner.run()
    else:
        print("[INFO] No experiments to run")


if __name__ == "__main__":
    main()

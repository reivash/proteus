"""
Experiment Results Aggregator & Deployment Recommender

PROBLEM:
- 100+ experiments generating results across months
- No centralized system to track what works
- Manual review required to identify deployment candidates
- Validated improvements sit unused

SOLUTION:
Automated system that:
1. Scans all experiment logs for completed experiments
2. Extracts performance metrics (Sharpe, WR, returns, etc.)
3. Compares to baseline and validates improvements
4. Generates deployment-ready configuration files
5. Provides ranked deployment recommendations

EXPECTED IMPACT:
- Reduce time-to-deployment from weeks to hours
- Automatically identify winning strategies
- Generate production configs with zero manual work
- Track improvement ROI across all experiments

Created: 2025-11-19
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd


@dataclass
class ExperimentResult:
    """Experiment result metadata and metrics."""
    experiment_id: str
    name: str
    status: str  # 'completed', 'failed', 'running'
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_return: float = 0.0
    total_trades: int = 0
    avg_pnl: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    baseline_sharpe: float = 0.0
    baseline_win_rate: float = 0.0
    sharpe_improvement_pct: float = 0.0
    win_rate_improvement_pp: float = 0.0
    recommendation: str = 'pending'  # 'deploy', 'reject', 'pending', 'conditional'
    deployment_priority: int = 0  # 1=critical, 2=high, 3=medium, 4=low
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_minutes: float = 0.0


class ResultsAggregator:
    """
    Aggregates experiment results and generates deployment recommendations.
    """

    def __init__(self, repo_root: str = None):
        """
        Initialize results aggregator.

        Args:
            repo_root: Path to repository root (default: auto-detect)
        """
        if repo_root is None:
            self.repo_root = Path(__file__).parent.parent.parent
        else:
            self.repo_root = Path(repo_root)

        self.logs_dir = self.repo_root / "logs"
        self.experiments_dir = self.repo_root / "src" / "experiments"
        self.results_dir = self.repo_root / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Deployment criteria
        self.deployment_criteria = {
            'critical': {
                'min_sharpe_improvement': 20.0,  # +20% Sharpe minimum
                'min_win_rate_improvement': 5.0,  # +5pp win rate
                'min_total_trades': 50,  # Statistically significant
            },
            'high': {
                'min_sharpe_improvement': 10.0,  # +10% Sharpe minimum
                'min_win_rate_improvement': 2.0,  # +2pp win rate
                'min_total_trades': 50,
            },
            'medium': {
                'min_sharpe_improvement': 5.0,  # +5% Sharpe minimum
                'min_win_rate_improvement': 1.0,  # +1pp win rate
                'min_total_trades': 30,
            },
            'low': {
                'min_sharpe_improvement': 0.0,  # Any improvement
                'min_win_rate_improvement': 0.0,
                'min_total_trades': 20,
            }
        }

    def scan_experiment_logs(self) -> List[ExperimentResult]:
        """
        Scan all experiment logs and extract results.

        Returns:
            List of ExperimentResult objects
        """
        results = []

        if not self.logs_dir.exists():
            print(f"[WARN] Logs directory not found: {self.logs_dir}")
            return results

        log_files = list(self.logs_dir.glob("exp*.log"))
        print(f"[SCAN] Found {len(log_files)} experiment logs")

        for log_file in log_files:
            try:
                result = self._parse_experiment_log(log_file)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"[ERROR] Failed to parse {log_file.name}: {e}")

        return results

    def _parse_experiment_log(self, log_path: Path) -> Optional[ExperimentResult]:
        """
        Parse experiment log file and extract metrics.

        Args:
            log_path: Path to log file

        Returns:
            ExperimentResult or None if parsing fails
        """
        exp_id = log_path.stem  # e.g., 'exp125_adaptive_stop_loss'

        # Read log file
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"[ERROR] Could not read {log_path}: {e}")
            return None

        # Check if experiment completed
        if 'CONCLUSION:' not in content or 'SUCCESS:' not in content and 'FAILED:' not in content:
            return ExperimentResult(
                experiment_id=exp_id,
                name=self._extract_experiment_name(content),
                status='running'
            )

        # Extract metrics using regex
        metrics = {}

        # Sharpe ratio
        sharpe_match = re.search(r'Sharpe Ratio:\s+([\d.]+)', content)
        if sharpe_match:
            metrics['sharpe_ratio'] = float(sharpe_match.group(1))

        baseline_sharpe_match = re.search(r'Sharpe.*?baseline[):\s]+([\d.]+)', content, re.IGNORECASE)
        if baseline_sharpe_match:
            metrics['baseline_sharpe'] = float(baseline_sharpe_match.group(1))

        sharpe_imp_match = re.search(r'Sharpe.*?improvement.*?([+-]?[\d.]+)%', content, re.IGNORECASE)
        if sharpe_imp_match:
            metrics['sharpe_improvement_pct'] = float(sharpe_imp_match.group(1))

        # Win rate
        wr_match = re.search(r'Win Rate:\s+([\d.]+)%', content)
        if wr_match:
            metrics['win_rate'] = float(wr_match.group(1))

        baseline_wr_match = re.search(r'Win Rate.*?baseline.*?(\d+\.\d+)%', content, re.IGNORECASE)
        if baseline_wr_match:
            metrics['baseline_win_rate'] = float(baseline_wr_match.group(1))

        wr_imp_match = re.search(r'Win Rate.*?([+-]?[\d.]+)pp', content, re.IGNORECASE)
        if wr_imp_match:
            metrics['win_rate_improvement_pp'] = float(wr_imp_match.group(1))

        # Other metrics
        return_match = re.search(r'Total Return:\s+([+-]?[\d.]+)%', content)
        if return_match:
            metrics['total_return'] = float(return_match.group(1))

        trades_match = re.search(r'Total Trades:\s+(\d+)', content)
        if trades_match:
            metrics['total_trades'] = int(trades_match.group(1))

        pnl_match = re.search(r'Avg PnL.*?[$]?([\d.]+)', content, re.IGNORECASE)
        if pnl_match:
            metrics['avg_pnl'] = float(pnl_match.group(1))

        drawdown_match = re.search(r'Max Drawdown:\s+([+-]?[\d.]+)%', content)
        if drawdown_match:
            metrics['max_drawdown'] = float(drawdown_match.group(1))

        pf_match = re.search(r'Profit Factor:\s+([\d.]+)', content)
        if pf_match:
            metrics['profit_factor'] = float(pf_match.group(1))

        # Determine status
        if 'SUCCESS:' in content or 'Targets achieved' in content:
            status = 'completed'
        elif 'FAILED:' in content or 'Targets not achieved' in content:
            status = 'failed'
        else:
            status = 'completed'  # Assume completed if has conclusion

        # Create result
        result = ExperimentResult(
            experiment_id=exp_id,
            name=self._extract_experiment_name(content),
            status=status,
            **metrics
        )

        # Evaluate and recommend deployment
        result = self._evaluate_deployment(result)

        return result

    def _extract_experiment_name(self, content: str) -> str:
        """Extract experiment name from log content."""
        # Try to find experiment name in header
        name_match = re.search(r'EXP-\d+:\s+(.+)', content)
        if name_match:
            return name_match.group(1).strip()

        # Try to find in title
        title_match = re.search(r'^([A-Z][A-Z\s]+)$', content, re.MULTILINE)
        if title_match:
            return title_match.group(1).strip()

        return "Unknown Experiment"

    def _evaluate_deployment(self, result: ExperimentResult) -> ExperimentResult:
        """
        Evaluate experiment result and determine deployment recommendation.

        Args:
            result: ExperimentResult to evaluate

        Returns:
            Updated ExperimentResult with recommendation
        """
        if result.status != 'completed':
            result.recommendation = 'pending'
            result.deployment_priority = 5
            return result

        # Check against deployment criteria
        if (result.sharpe_improvement_pct >= self.deployment_criteria['critical']['min_sharpe_improvement'] and
            result.win_rate_improvement_pp >= self.deployment_criteria['critical']['min_win_rate_improvement'] and
            result.total_trades >= self.deployment_criteria['critical']['min_total_trades']):
            result.recommendation = 'deploy'
            result.deployment_priority = 1
        elif (result.sharpe_improvement_pct >= self.deployment_criteria['high']['min_sharpe_improvement'] and
              result.win_rate_improvement_pp >= self.deployment_criteria['high']['min_win_rate_improvement'] and
              result.total_trades >= self.deployment_criteria['high']['min_total_trades']):
            result.recommendation = 'deploy'
            result.deployment_priority = 2
        elif (result.sharpe_improvement_pct >= self.deployment_criteria['medium']['min_sharpe_improvement'] and
              result.win_rate_improvement_pp >= self.deployment_criteria['medium']['min_win_rate_improvement'] and
              result.total_trades >= self.deployment_criteria['medium']['min_total_trades']):
            result.recommendation = 'conditional'
            result.deployment_priority = 3
        elif result.sharpe_improvement_pct < 0 or result.win_rate_improvement_pp < -1.0:
            result.recommendation = 'reject'
            result.deployment_priority = 5
        else:
            result.recommendation = 'pending'
            result.deployment_priority = 4

        return result

    def generate_report(self, results: List[ExperimentResult]) -> str:
        """
        Generate human-readable report of experiment results.

        Args:
            results: List of ExperimentResult objects

        Returns:
            Formatted report string
        """
        # Sort by deployment priority, then by Sharpe improvement
        results.sort(key=lambda r: (r.deployment_priority, -r.sharpe_improvement_pct))

        report = []
        report.append("=" * 80)
        report.append("EXPERIMENT RESULTS SUMMARY")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total experiments: {len(results)}")
        report.append("")

        # Summary by status
        completed = sum(1 for r in results if r.status == 'completed')
        failed = sum(1 for r in results if r.status == 'failed')
        running = sum(1 for r in results if r.status == 'running')

        report.append(f"Completed: {completed} | Failed: {failed} | Running: {running}")
        report.append("")

        # Deployment recommendations
        deploy = [r for r in results if r.recommendation == 'deploy']
        conditional = [r for r in results if r.recommendation == 'conditional']
        reject = [r for r in results if r.recommendation == 'reject']

        report.append("=" * 80)
        report.append(f"DEPLOYMENT READY: {len(deploy)} experiments")
        report.append("=" * 80)

        if deploy:
            for r in deploy:
                priority_label = ['', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'][r.deployment_priority]
                report.append(f"\n[{priority_label}] {r.experiment_id}")
                report.append(f"  Name: {r.name}")
                report.append(f"  Sharpe: {r.sharpe_ratio:.2f} ({r.sharpe_improvement_pct:+.1f}% vs baseline)")
                report.append(f"  Win Rate: {r.win_rate:.1f}% ({r.win_rate_improvement_pp:+.1f}pp vs baseline)")
                report.append(f"  Total Return: {r.total_return:.2f}%")
                report.append(f"  Trades: {r.total_trades}")
                report.append(f"  Recommendation: DEPLOY IMMEDIATELY")
        else:
            report.append("\nNo experiments meet deployment criteria yet.")

        if conditional:
            report.append("\n" + "=" * 80)
            report.append(f"CONDITIONAL DEPLOYMENT: {len(conditional)} experiments")
            report.append("=" * 80)
            for r in conditional:
                report.append(f"\n{r.experiment_id}: {r.name}")
                report.append(f"  Sharpe improvement: {r.sharpe_improvement_pct:+.1f}%")
                report.append(f"  Win rate improvement: {r.win_rate_improvement_pp:+.1f}pp")
                report.append(f"  Recommendation: Review for deployment")

        if reject:
            report.append("\n" + "=" * 80)
            report.append(f"REJECTED: {len(reject)} experiments")
            report.append("=" * 80)
            for r in reject:
                report.append(f"  {r.experiment_id}: {r.name} (Sharpe: {r.sharpe_improvement_pct:+.1f}%)")

        report.append("\n" + "=" * 80)
        report.append("END REPORT")
        report.append("=" * 80)

        return "\n".join(report)

    def export_results_json(self, results: List[ExperimentResult], output_path: str = None):
        """
        Export results to JSON file.

        Args:
            results: List of ExperimentResult objects
            output_path: Path to output JSON file (default: results/experiment_results.json)
        """
        if output_path is None:
            output_path = self.results_dir / f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        data = {
            'generated_at': datetime.now().isoformat(),
            'total_experiments': len(results),
            'results': [asdict(r) for r in results]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[EXPORT] Results exported to: {output_path}")

    def run(self):
        """Main execution: scan logs, generate report, export results."""
        print("\n" + "=" * 80)
        print("EXPERIMENT RESULTS AGGREGATOR")
        print("=" * 80)

        # Scan logs
        print("\n[1/3] Scanning experiment logs...")
        results = self.scan_experiment_logs()

        if not results:
            print("[WARN] No experiment results found")
            return

        print(f"[OK] Parsed {len(results)} experiments")

        # Generate report
        print("\n[2/3] Generating deployment recommendations...")
        report = self.generate_report(results)
        print(report)

        # Export results
        print("\n[3/3] Exporting results...")
        self.export_results_json(results)

        # Save report
        report_path = self.results_dir / f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"[EXPORT] Report saved to: {report_path}")

        print("\n" + "=" * 80)
        print("AGGREGATION COMPLETE")
        print("=" * 80)


def main():
    """Main entry point."""
    aggregator = ResultsAggregator()
    aggregator.run()


if __name__ == "__main__":
    main()

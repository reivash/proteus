"""
Production Deployment System

PROBLEM:
- Results aggregator identifies deployment-ready experiments
- But manual deployment is error-prone and slow
- No standardized config generation or rollback
- Validated improvements sit unused waiting for manual deployment

SOLUTION:
Automated deployment system that:
1. Reads deployment recommendations from results aggregator
2. Generates production config files automatically
3. Updates live trading parameters
4. Validates deployments before activation
5. Provides one-click rollback capability
6. Tracks deployment history and performance

EXPECTED IMPACT:
- Reduce deployment time from days to minutes
- Eliminate manual configuration errors
- Enable rapid iteration on validated improvements
- Provide safety net via automated rollback

Created: 2025-11-19
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd


@dataclass
class DeploymentConfig:
    """Production deployment configuration."""
    deployment_id: str
    experiment_id: str
    name: str
    deployment_date: str
    deployed_by: str = "AutoDeployer"

    # Trading parameters
    min_signal_strength: Optional[float] = None
    profit_target: Optional[float] = None
    stop_loss: Optional[float] = None
    max_hold_days: Optional[int] = None
    max_positions: Optional[int] = None
    position_size_pct: Optional[float] = None

    # Adaptive parameters
    use_adaptive_stop_loss: bool = False
    use_adaptive_profit_targets: bool = False
    use_adaptive_max_hold_days: bool = False

    # Filter parameters
    min_ensemble_score: Optional[int] = None
    max_portfolio_correlation: Optional[float] = None
    volume_surge_threshold: Optional[float] = None
    vix_threshold: Optional[float] = None

    # News sentiment parameters
    use_news_sentiment: bool = False
    sentiment_provider: Optional[str] = None
    sentiment_threshold: Optional[float] = None

    # Performance tracking
    expected_sharpe: float = 0.0
    expected_win_rate: float = 0.0
    expected_return: float = 0.0

    # Deployment metadata
    status: str = 'pending'  # 'pending', 'active', 'rolled_back', 'failed'
    validation_passed: bool = False
    rollback_available: bool = False


class ProductionDeployer:
    """
    Automated production deployment system.
    """

    def __init__(self, repo_root: str = None):
        """
        Initialize production deployer.

        Args:
            repo_root: Path to repository root (default: auto-detect)
        """
        if repo_root is None:
            self.repo_root = Path(__file__).parent.parent.parent
        else:
            self.repo_root = Path(repo_root)

        # Directory structure
        self.config_dir = self.repo_root / "config"
        self.config_dir.mkdir(exist_ok=True)

        self.deployments_dir = self.config_dir / "deployments"
        self.deployments_dir.mkdir(exist_ok=True)

        self.backups_dir = self.config_dir / "backups"
        self.backups_dir.mkdir(exist_ok=True)

        self.results_dir = self.repo_root / "results"

        # Deployment history
        self.history_file = self.deployments_dir / "deployment_history.json"
        self.history = self._load_deployment_history()

    def _load_deployment_history(self) -> List[Dict]:
        """Load deployment history from JSON."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []

    def _save_deployment_history(self):
        """Save deployment history to JSON."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_aggregator_results(self, results_file: str = None) -> Dict:
        """
        Load results from results aggregator.

        Args:
            results_file: Path to results JSON (default: most recent)

        Returns:
            Dictionary with experiment results
        """
        if results_file is None:
            # Find most recent results file
            results_files = sorted(self.results_dir.glob("experiment_results_*.json"))
            if not results_files:
                raise FileNotFoundError("No experiment results found. Run results_aggregator.py first.")
            results_file = results_files[-1]
        else:
            results_file = Path(results_file)

        print(f"[LOAD] Loading results from: {results_file}")

        with open(results_file, 'r') as f:
            data = json.load(f)

        return data

    def filter_deployment_candidates(self, results: Dict, min_priority: int = 3) -> List[Dict]:
        """
        Filter experiments ready for deployment.

        Args:
            results: Results dictionary from aggregator
            min_priority: Minimum deployment priority (1=critical, 2=high, 3=medium)

        Returns:
            List of deployment-ready experiments
        """
        candidates = []

        for result in results['results']:
            if (result['recommendation'] == 'deploy' and
                result['deployment_priority'] <= min_priority):
                candidates.append(result)

        # Sort by priority (1=highest)
        candidates.sort(key=lambda x: (x['deployment_priority'], -x['sharpe_improvement_pct']))

        print(f"[FILTER] Found {len(candidates)} deployment candidates (priority <= {min_priority})")

        return candidates

    def create_deployment_config(self, experiment: Dict) -> DeploymentConfig:
        """
        Create deployment configuration from experiment results.

        Args:
            experiment: Experiment result dictionary

        Returns:
            DeploymentConfig object
        """
        exp_id = experiment['experiment_id']

        # Generate deployment ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        deployment_id = f"deploy_{exp_id}_{timestamp}"

        # Base config
        config = DeploymentConfig(
            deployment_id=deployment_id,
            experiment_id=exp_id,
            name=experiment['name'],
            deployment_date=datetime.now().isoformat(),
            expected_sharpe=experiment['sharpe_ratio'],
            expected_win_rate=experiment['win_rate'],
            expected_return=experiment['total_return']
        )

        # Extract parameters based on experiment type
        if 'adaptive_stop_loss' in exp_id:
            config.use_adaptive_stop_loss = True
            config.stop_loss = -2.0  # Base value, will be adapted

        elif 'adaptive_profit_targets' in exp_id:
            config.use_adaptive_profit_targets = True
            config.profit_target = 2.0  # Base value, will be adapted

        elif 'adaptive_max_hold_days' in exp_id or 'hold_days' in exp_id:
            config.use_adaptive_max_hold_days = True
            config.max_hold_days = 2  # Base value, will be adapted

        elif 'q4' in exp_id.lower() or 'signal_strength' in exp_id.lower():
            # Q4-only filter (top 25% signals)
            config.min_signal_strength = 75.0  # Top quartile threshold

        elif 'ensemble' in exp_id.lower():
            config.min_ensemble_score = 2  # Require at least 2 signals to align

        elif 'sentiment' in exp_id.lower():
            config.use_news_sentiment = True
            config.sentiment_provider = 'mock'  # Update with real provider
            config.sentiment_threshold = 0.0

        elif 'correlation' in exp_id.lower():
            config.max_portfolio_correlation = 0.7  # Max avg correlation with existing positions

        elif 'vix' in exp_id.lower():
            config.vix_threshold = 25.0  # High volatility environment

        elif 'volume' in exp_id.lower():
            config.volume_surge_threshold = 1.5  # 1.5x average volume

        return config

    def generate_production_config_file(self, config: DeploymentConfig) -> Path:
        """
        Generate production configuration file.

        Args:
            config: DeploymentConfig object

        Returns:
            Path to generated config file
        """
        config_path = self.deployments_dir / f"{config.deployment_id}.json"

        # Convert to dictionary and remove None values
        config_dict = asdict(config)
        config_dict = {k: v for k, v in config_dict.items() if v is not None}

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"[CONFIG] Generated: {config_path}")

        return config_path

    def backup_current_production_config(self) -> Optional[Path]:
        """
        Backup current production configuration.

        Returns:
            Path to backup file or None if no current config
        """
        prod_config = self.config_dir / "production.json"

        if not prod_config.exists():
            print("[BACKUP] No existing production config to backup")
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backups_dir / f"production_backup_{timestamp}.json"

        shutil.copy2(prod_config, backup_path)
        print(f"[BACKUP] Created: {backup_path}")

        return backup_path

    def validate_deployment_config(self, config: DeploymentConfig) -> Tuple[bool, List[str]]:
        """
        Validate deployment configuration.

        Args:
            config: DeploymentConfig to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Validate Sharpe ratio improvement
        if config.expected_sharpe < 2.0:
            errors.append(f"Sharpe ratio too low: {config.expected_sharpe:.2f} (minimum 2.0)")

        # Validate win rate
        if config.expected_win_rate < 55.0:
            errors.append(f"Win rate too low: {config.expected_win_rate:.1f}% (minimum 55%)")

        # Validate parameter ranges
        if config.profit_target is not None and (config.profit_target < 1.0 or config.profit_target > 5.0):
            errors.append(f"Profit target out of range: {config.profit_target}% (valid: 1-5%)")

        if config.stop_loss is not None and (config.stop_loss > -1.0 or config.stop_loss < -5.0):
            errors.append(f"Stop loss out of range: {config.stop_loss}% (valid: -1% to -5%)")

        if config.max_hold_days is not None and (config.max_hold_days < 1 or config.max_hold_days > 10):
            errors.append(f"Max hold days out of range: {config.max_hold_days} (valid: 1-10)")

        if config.min_signal_strength is not None and (config.min_signal_strength < 50 or config.min_signal_strength > 100):
            errors.append(f"Signal strength out of range: {config.min_signal_strength} (valid: 50-100)")

        # Validate mutual exclusivity
        if config.use_adaptive_max_hold_days and config.max_hold_days:
            errors.append("Cannot use both adaptive and fixed max_hold_days")

        is_valid = len(errors) == 0

        if is_valid:
            print(f"[VALIDATE] Configuration valid: {config.deployment_id}")
        else:
            print(f"[VALIDATE] Configuration invalid: {len(errors)} errors")
            for error in errors:
                print(f"  - {error}")

        return is_valid, errors

    def deploy_to_production(self, config: DeploymentConfig, dry_run: bool = False) -> bool:
        """
        Deploy configuration to production.

        Args:
            config: DeploymentConfig to deploy
            dry_run: If True, simulate deployment without writing files

        Returns:
            True if deployment successful
        """
        print(f"\n{'='*70}")
        print(f"DEPLOYING: {config.experiment_id}")
        print(f"{'='*70}")

        # Validate first
        is_valid, errors = self.validate_deployment_config(config)
        if not is_valid:
            print("[DEPLOY] Validation failed. Aborting deployment.")
            config.status = 'failed'
            return False

        config.validation_passed = True

        if dry_run:
            print("[DEPLOY] DRY RUN - No files will be modified")

        # Backup current production config
        backup_path = None
        if not dry_run:
            backup_path = self.backup_current_production_config()
            config.rollback_available = backup_path is not None

        # Generate new production config
        config_path = self.generate_production_config_file(config)

        # Deploy to production
        prod_config_path = self.config_dir / "production.json"

        if not dry_run:
            shutil.copy2(config_path, prod_config_path)
            print(f"[DEPLOY] Deployed to: {prod_config_path}")

            config.status = 'active'
        else:
            print(f"[DEPLOY] Would deploy to: {prod_config_path}")
            config.status = 'pending'

        # Record deployment in history
        deployment_record = asdict(config)
        deployment_record['backup_path'] = str(backup_path) if backup_path else None

        if not dry_run:
            self.history.append(deployment_record)
            self._save_deployment_history()
            print(f"[HISTORY] Deployment recorded")

        print(f"\n{'='*70}")
        print(f"DEPLOYMENT {'SIMULATED' if dry_run else 'COMPLETE'}")
        print(f"{'='*70}\n")

        return True

    def rollback_deployment(self, deployment_id: str = None) -> bool:
        """
        Rollback to previous production configuration.

        Args:
            deployment_id: Specific deployment to rollback (default: most recent)

        Returns:
            True if rollback successful
        """
        print(f"\n{'='*70}")
        print("ROLLBACK DEPLOYMENT")
        print(f"{'='*70}\n")

        # Find deployment to rollback
        if deployment_id is None:
            # Rollback most recent
            if not self.history:
                print("[ERROR] No deployment history found")
                return False
            deployment = self.history[-1]
        else:
            deployment = next((d for d in self.history if d['deployment_id'] == deployment_id), None)
            if not deployment:
                print(f"[ERROR] Deployment not found: {deployment_id}")
                return False

        # Check if rollback available
        if not deployment.get('rollback_available'):
            print(f"[ERROR] No backup available for deployment: {deployment['deployment_id']}")
            return False

        backup_path = Path(deployment['backup_path'])
        if not backup_path.exists():
            print(f"[ERROR] Backup file not found: {backup_path}")
            return False

        # Restore backup
        prod_config_path = self.config_dir / "production.json"
        shutil.copy2(backup_path, prod_config_path)

        print(f"[ROLLBACK] Restored from: {backup_path}")
        print(f"[ROLLBACK] Production config restored")

        # Update deployment status
        deployment['status'] = 'rolled_back'
        self._save_deployment_history()

        print(f"\n{'='*70}")
        print("ROLLBACK COMPLETE")
        print(f"{'='*70}\n")

        return True

    def list_deployments(self, status_filter: str = None) -> pd.DataFrame:
        """
        List all deployments with optional filtering.

        Args:
            status_filter: Filter by status ('active', 'rolled_back', 'failed')

        Returns:
            DataFrame with deployment history
        """
        if not self.history:
            print("[INFO] No deployment history")
            return pd.DataFrame()

        history = self.history
        if status_filter:
            history = [d for d in history if d.get('status') == status_filter]

        if not history:
            print(f"[INFO] No deployments with status: {status_filter}")
            return pd.DataFrame()

        df = pd.DataFrame(history)

        # Select key columns
        cols = ['deployment_id', 'experiment_id', 'deployment_date', 'status',
                'expected_sharpe', 'expected_win_rate', 'expected_return']
        cols = [c for c in cols if c in df.columns]

        return df[cols]

    def generate_deployment_report(self) -> str:
        """
        Generate human-readable deployment report.

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("PRODUCTION DEPLOYMENT REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total deployments: {len(self.history)}")
        report.append("")

        if not self.history:
            report.append("No deployments yet.")
            report.append("=" * 80)
            return "\n".join(report)

        # Current production config
        prod_config_path = self.config_dir / "production.json"
        if prod_config_path.exists():
            with open(prod_config_path, 'r') as f:
                current_config = json.load(f)

            report.append("CURRENT PRODUCTION CONFIGURATION:")
            report.append("-" * 80)
            report.append(f"  Deployment ID: {current_config.get('deployment_id', 'unknown')}")
            report.append(f"  Experiment: {current_config.get('experiment_id', 'unknown')}")
            report.append(f"  Expected Sharpe: {current_config.get('expected_sharpe', 0):.2f}")
            report.append(f"  Expected Win Rate: {current_config.get('expected_win_rate', 0):.1f}%")
            report.append(f"  Expected Return: {current_config.get('expected_return', 0):.2f}%")
            report.append("")

        # Deployment summary by status
        active = sum(1 for d in self.history if d.get('status') == 'active')
        rolled_back = sum(1 for d in self.history if d.get('status') == 'rolled_back')
        failed = sum(1 for d in self.history if d.get('status') == 'failed')

        report.append("DEPLOYMENT SUMMARY:")
        report.append("-" * 80)
        report.append(f"  Active: {active}")
        report.append(f"  Rolled back: {rolled_back}")
        report.append(f"  Failed: {failed}")
        report.append("")

        # Recent deployments (last 5)
        report.append("RECENT DEPLOYMENTS (Last 5):")
        report.append("-" * 80)

        recent = self.history[-5:]
        for deployment in reversed(recent):
            report.append(f"\n  {deployment['deployment_id']}")
            report.append(f"    Experiment: {deployment['experiment_id']}")
            report.append(f"    Date: {deployment['deployment_date']}")
            report.append(f"    Status: {deployment['status']}")
            report.append(f"    Expected Sharpe: {deployment['expected_sharpe']:.2f}")

        report.append("\n" + "=" * 80)
        report.append("END REPORT")
        report.append("=" * 80)

        return "\n".join(report)

    def auto_deploy_candidates(
        self,
        results_file: str = None,
        min_priority: int = 2,
        dry_run: bool = True
    ) -> List[DeploymentConfig]:
        """
        Automatically deploy all deployment-ready experiments.

        Args:
            results_file: Path to results JSON (default: most recent)
            min_priority: Minimum priority to deploy (1=critical, 2=high)
            dry_run: If True, simulate without deploying

        Returns:
            List of deployed configurations
        """
        print("\n" + "=" * 80)
        print("AUTO-DEPLOYMENT SYSTEM")
        print("=" * 80)

        # Load aggregator results
        results = self.load_aggregator_results(results_file)

        # Filter candidates
        candidates = self.filter_deployment_candidates(results, min_priority)

        if not candidates:
            print("\n[INFO] No deployment candidates found")
            return []

        # Deploy each candidate
        deployed = []

        for i, experiment in enumerate(candidates, 1):
            print(f"\n[{i}/{len(candidates)}] Processing: {experiment['experiment_id']}")

            # Create deployment config
            config = self.create_deployment_config(experiment)

            # Deploy
            success = self.deploy_to_production(config, dry_run=dry_run)

            if success:
                deployed.append(config)

        # Summary
        print("\n" + "=" * 80)
        print(f"AUTO-DEPLOYMENT {'SIMULATION ' if dry_run else ''}COMPLETE")
        print("=" * 80)
        print(f"Candidates processed: {len(candidates)}")
        print(f"Successfully deployed: {len(deployed)}")

        if dry_run:
            print("\nRe-run with dry_run=False to deploy to production")

        print("=" * 80 + "\n")

        return deployed


def main():
    """Main entry point."""
    deployer = ProductionDeployer()

    print("\n" + "=" * 80)
    print("PRODUCTION DEPLOYMENT SYSTEM")
    print("=" * 80)
    print("\nThis tool automates deployment of validated experiments to production.")
    print("\nCommands:")
    print("  1. Auto-deploy (dry run)")
    print("  2. Auto-deploy (production)")
    print("  3. Rollback last deployment")
    print("  4. List deployments")
    print("  5. Generate report")
    print("=" * 80)

    # For now, just run auto-deploy in dry-run mode
    print("\n[RUNNING] Auto-deploy in DRY RUN mode...")
    deployed = deployer.auto_deploy_candidates(dry_run=True, min_priority=3)

    # Generate report
    print("\n" + deployer.generate_deployment_report())


if __name__ == "__main__":
    main()

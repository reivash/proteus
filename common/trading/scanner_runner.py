"""
ScannerRunner - Safe wrapper for SmartScannerV2 with error handling and logging.

Provides:
- Health checks before running scan
- Error handling with graceful degradation
- Centralized logging
- Integration with VirtualWallet

Usage:
    from trading.scanner_runner import ScannerRunner

    runner = ScannerRunner()
    result = runner.run_safe_scan()
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Setup logging
try:
    from utils.logging_config import get_logger, setup_logging
    setup_logging()
    logger = get_logger('proteus.scanner_runner')
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('scanner_runner')


@dataclass
class HealthCheckResult:
    """Result of pre-scan health checks."""
    healthy: bool
    checks_passed: int
    checks_failed: int
    warnings: list
    errors: list


class ScannerRunner:
    """
    Safe scanner runner with health checks and error handling.
    """

    def __init__(self, portfolio_value: float = 100000):
        self.portfolio_value = portfolio_value
        self.scanner = None
        self.last_error = None

    def run_health_checks(self) -> HealthCheckResult:
        """
        Run pre-scan health checks.

        Checks:
        1. Config file exists and is valid
        2. Data directories exist
        3. yfinance is available
        4. Market data is accessible
        """
        warnings = []
        errors = []
        checks_passed = 0
        checks_failed = 0

        # Check 1: Config file
        config_path = Path('config/unified_config.json')
        if config_path.exists():
            try:
                import json
                with open(config_path) as f:
                    config = json.load(f)
                if 'stock_tiers' in config:
                    checks_passed += 1
                    logger.info("Health check: Config file OK")
                else:
                    warnings.append("Config file missing stock_tiers")
                    checks_passed += 1  # Still counts as passed
            except Exception as e:
                errors.append(f"Config file invalid: {e}")
                checks_failed += 1
        else:
            errors.append("Config file not found")
            checks_failed += 1

        # Check 2: Data directories
        data_dirs = ['features/daily_picks/data/smart_scans', 'features/simulation/data/portfolio', 'features/simulation/data/virtual_wallet']
        for dir_path in data_dirs:
            path = Path(dir_path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                warnings.append(f"Created missing directory: {dir_path}")
        checks_passed += 1
        logger.info("Health check: Data directories OK")

        # Check 3: yfinance availability
        try:
            import yfinance as yf
            # Quick test fetch
            test = yf.Ticker("SPY")
            info = test.info
            if info:
                checks_passed += 1
                logger.info("Health check: yfinance OK")
            else:
                warnings.append("yfinance returned empty data")
                checks_passed += 1
        except Exception as e:
            warnings.append(f"yfinance unavailable: {e}")
            checks_passed += 1  # Non-fatal

        # Check 4: Market hours (informational)
        now = datetime.now()
        weekday = now.weekday()
        hour = now.hour
        if weekday >= 5:  # Weekend
            warnings.append("Market is closed (weekend)")
        elif hour < 9 or hour >= 16:
            warnings.append("Market may be closed (outside regular hours)")
        checks_passed += 1

        healthy = checks_failed == 0
        return HealthCheckResult(
            healthy=healthy,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            errors=errors
        )

    def initialize_scanner(self) -> bool:
        """Initialize the scanner with error handling."""
        try:
            from trading.smart_scanner_v2 import SmartScannerV2
            self.scanner = SmartScannerV2(portfolio_value=self.portfolio_value)
            logger.info("Scanner initialized successfully")
            return True
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Scanner initialization failed: {e}", exc_info=True)
            return False

    def run_safe_scan(self, skip_health_checks: bool = False) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Run scanner with full error handling.

        Args:
            skip_health_checks: Skip pre-scan health checks

        Returns:
            Tuple of (scan_result_dict, error_message)
            If successful, error_message is None
            If failed, scan_result_dict is None
        """
        logger.info("=" * 50)
        logger.info("SCANNER RUNNER - Starting safe scan")
        logger.info("=" * 50)

        # Health checks
        if not skip_health_checks:
            health = self.run_health_checks()
            if not health.healthy:
                error_msg = f"Health checks failed: {', '.join(health.errors)}"
                logger.error(error_msg)
                return None, error_msg

            if health.warnings:
                for warn in health.warnings:
                    logger.warning(f"Health check warning: {warn}")

        # Initialize scanner
        if not self.scanner:
            if not self.initialize_scanner():
                return None, f"Scanner initialization failed: {self.last_error}"

        # Run scan
        try:
            logger.info("Running scanner...")
            result = self.scanner.scan()

            # Convert to dict for easier handling
            from dataclasses import asdict
            result_dict = {
                'timestamp': result.timestamp,
                'regime': result.regime,
                'regime_confidence': result.regime_confidence,
                'vix': result.vix,
                'signals': [asdict(s) for s in result.signals],
                'rebalance_actions': result.rebalance_actions,
                'stats': result.stats,
                'bear_score': result.bear_score,
                'bear_alert_level': result.bear_alert_level,
                'bear_triggers': result.bear_triggers or [],
                'is_choppy': result.is_choppy,
                'choppiness_score': result.choppiness_score,
                'adx': result.adx
            }

            logger.info(f"Scan complete: {len(result.signals)} signals")
            return result_dict, None

        except Exception as e:
            error_msg = f"Scan failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.last_error = error_msg
            return None, error_msg

    def run_scan_with_wallet_integration(self, wallet=None) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Run scan with automatic wallet integration.

        Args:
            wallet: VirtualWallet instance (created if not provided)

        Returns:
            Tuple of (summary_dict, error_message)
        """
        logger.info("Running scan with wallet integration")

        # Create wallet if needed
        if wallet is None:
            try:
                from trading.virtual_wallet import VirtualWallet
                wallet = VirtualWallet(
                    initial_capital=self.portfolio_value,
                    min_signal_strength=65
                )
            except Exception as e:
                return None, f"Wallet initialization failed: {e}"

        # Sync positions to scanner
        try:
            wallet._sync_to_scanner()
            logger.info("Positions synced to scanner")
        except Exception as e:
            logger.warning(f"Position sync failed: {e}")

        # Run scan
        result, error = self.run_safe_scan()
        if error:
            return None, error

        # Process with wallet
        try:
            summary = wallet.process_daily_scan()
            logger.info(f"Wallet processed: {summary['num_positions']} positions")
            return summary, None
        except Exception as e:
            return None, f"Wallet processing failed: {e}"


def run_safe_scan():
    """Entry point for safe scanning."""
    runner = ScannerRunner()
    result, error = runner.run_safe_scan()

    if error:
        print(f"\nSCAN FAILED: {error}")
        return None

    print(f"\nScan completed successfully")
    print(f"Signals: {len(result.get('signals', []))}")
    print(f"Regime: {result.get('regime', 'unknown').upper()}")

    return result


if __name__ == '__main__':
    run_safe_scan()

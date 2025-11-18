"""
Stock Configuration Loader

Loads and manages per-stock optimized parameters from config/stock_configs.json.
Merges with existing mean_reversion_params.py for backward compatibility.

Deployed: 2025-11-18
Source: EXP-044 (entry parameters), EXP-049 (exit parameters)
"""

import json
from pathlib import Path
from typing import Dict, Optional


class StockConfigLoader:
    """
    Centralized loader for per-stock configurations.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to stock_configs.json (default: config/stock_configs.json)
        """
        if config_path is None:
            # Default to project root / config / stock_configs.json
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / 'config' / 'stock_configs.json'

        self.config_path = Path(config_path)
        self._config = None
        self._load_config()

    def _load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                self._config = json.load(f)
            print(f"[CONFIG] Loaded stock configs from {self.config_path}")
            print(f"[CONFIG] Entry params: {len(self._config.get('entry_parameters', {}))} stocks")
            print(f"[CONFIG] Exit params: {len(self._config.get('exit_parameters', {}))} stocks")
        except FileNotFoundError:
            print(f"[WARN] Stock config file not found: {self.config_path}")
            self._config = {'entry_parameters': {}, 'exit_parameters': {}, 'defaults': {}}
        except Exception as e:
            print(f"[ERROR] Failed to load stock configs: {e}")
            self._config = {'entry_parameters': {}, 'exit_parameters': {}, 'defaults': {}}

    def get_entry_params(self, ticker: str) -> Optional[Dict]:
        """
        Get entry parameters for a ticker.

        Args:
            ticker: Stock ticker

        Returns:
            Entry parameters dict or None if not found
        """
        return self._config.get('entry_parameters', {}).get(ticker)

    def get_exit_params(self, ticker: str) -> Optional[Dict]:
        """
        Get exit parameters for a ticker.

        Args:
            ticker: Stock ticker

        Returns:
            Exit parameters dict or None if not found
        """
        return self._config.get('exit_parameters', {}).get(ticker)

    def get_default_entry_params(self) -> Dict:
        """Get default entry parameters."""
        return self._config.get('defaults', {}).get('entry', {
            'z_score_threshold': 2.0,
            'rsi_oversold': 30,
            'volume_multiplier': 1.5
        })

    def get_default_exit_params(self) -> Dict:
        """Get default exit parameters."""
        return self._config.get('defaults', {}).get('exit', {
            'profit_target': 2.0,
            'stop_loss': -2.0,
            'max_hold_days': 2
        })

    def has_entry_override(self, ticker: str) -> bool:
        """Check if ticker has custom entry parameters."""
        return ticker in self._config.get('entry_parameters', {})

    def has_exit_override(self, ticker: str) -> bool:
        """Check if ticker has custom exit parameters."""
        return ticker in self._config.get('exit_parameters', {})


# Singleton instance
_loader = None


def get_loader() -> StockConfigLoader:
    """Get singleton config loader instance."""
    global _loader
    if _loader is None:
        _loader = StockConfigLoader()
    return _loader


def get_stock_entry_params(ticker: str) -> Optional[Dict]:
    """
    Get entry parameters for a ticker from stock_configs.json.

    Args:
        ticker: Stock ticker

    Returns:
        Entry parameters dict or None
    """
    return get_loader().get_entry_params(ticker)


def get_stock_exit_params(ticker: str) -> Optional[Dict]:
    """
    Get exit parameters for a ticker from stock_configs.json.

    Args:
        ticker: Stock ticker

    Returns:
        Exit parameters dict with profit_target, stop_loss, max_hold_days
    """
    return get_loader().get_exit_params(ticker)


def merge_entry_params(ticker: str, base_params: Dict) -> Dict:
    """
    Merge per-stock entry parameters with base parameters.

    Args:
        ticker: Stock ticker
        base_params: Base parameters from mean_reversion_params.py

    Returns:
        Merged parameters (per-stock overrides take precedence)
    """
    stock_params = get_stock_entry_params(ticker)
    if stock_params:
        # Merge: stock-specific overrides base params
        merged = base_params.copy()
        for key in ['z_score_threshold', 'rsi_oversold', 'volume_multiplier']:
            if key in stock_params:
                merged[key] = stock_params[key]
        return merged
    return base_params


if __name__ == "__main__":
    # Test loading
    loader = get_loader()

    print("\n" + "=" * 70)
    print("STOCK CONFIGURATION LOADER TEST")
    print("=" * 70)

    # Test entry params
    print("\nEntry Parameters:")
    for ticker in ['TXN', 'IDXX', 'DXCM', 'EXR', 'NVDA']:
        params = loader.get_entry_params(ticker)
        if params:
            print(f"  {ticker}: z={params.get('z_score_threshold')}, "
                  f"rsi={params.get('rsi_oversold')}, "
                  f"vol={params.get('volume_multiplier')}")
        else:
            print(f"  {ticker}: Using defaults")

    # Test exit params
    print("\nExit Parameters:")
    for ticker in ['NVDA', 'V', 'MA', 'AVGO', 'TXN']:
        params = loader.get_exit_params(ticker)
        if params:
            print(f"  {ticker}: target={params.get('profit_target')}%, "
                  f"stop={params.get('stop_loss')}%, "
                  f"hold={params.get('max_hold_days')}d")
        else:
            print(f"  {ticker}: Using defaults")

    print("\nDefaults:")
    print(f"  Entry: {loader.get_default_entry_params()}")
    print(f"  Exit: {loader.get_default_exit_params()}")

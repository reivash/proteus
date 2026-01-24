"""
Centralized Path Configuration for Proteus

Provides consistent path handling across all modules.

Usage:
    from src.utils.paths import setup_paths, get_project_root, get_data_dir

    # Setup paths at module start (adds project root to sys.path)
    setup_paths()

    # Get commonly used paths
    root = get_project_root()
    data = get_data_dir()
"""

import os
import sys
from pathlib import Path
from typing import Optional


# Cache the project root to avoid repeated computation
_PROJECT_ROOT: Optional[Path] = None


def get_project_root() -> Path:
    """
    Get the project root directory.

    The project root is identified by containing:
    - src/ directory
    - config/ directory
    - Either setup.py, pyproject.toml, or .git

    Returns:
        Path to project root directory
    """
    global _PROJECT_ROOT

    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT

    # Start from this file's location and walk up
    current = Path(__file__).resolve()

    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        # Check for project indicators
        has_src = (parent / 'src').is_dir()
        has_config = (parent / 'config').is_dir()
        has_git = (parent / '.git').exists()

        if has_src and (has_config or has_git):
            _PROJECT_ROOT = parent
            return parent

    # Fallback: assume src/utils/paths.py structure
    _PROJECT_ROOT = current.parent.parent.parent
    return _PROJECT_ROOT


def setup_paths() -> None:
    """
    Add project paths to sys.path for consistent imports.

    Should be called at the start of any script that needs to import
    from src/ without being in the project root.
    """
    root = get_project_root()
    root_str = str(root)

    # Add project root if not already present
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # Add src/ if not already present
    src_str = str(root / 'src')
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def get_src_dir() -> Path:
    """Get the src/ directory path."""
    return get_project_root() / 'src'


def get_data_dir() -> Path:
    """Get the data/ directory path."""
    return get_project_root() / 'data'


def get_config_dir() -> Path:
    """Get the config/ directory path."""
    return get_project_root() / 'config'


def get_logs_dir() -> Path:
    """Get the logs/ directory path (creates if needed)."""
    logs = get_project_root() / 'logs'
    logs.mkdir(exist_ok=True)
    return logs


def get_models_dir() -> Path:
    """Get the models/ directory path."""
    return get_project_root() / 'models'


def get_scripts_dir() -> Path:
    """Get the scripts/ directory path."""
    return get_project_root() / 'scripts'


def get_tests_dir() -> Path:
    """Get the tests/ directory path."""
    return get_project_root() / 'tests'


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        The same path for chaining
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# Convenience: setup paths when module is imported
# This allows simple usage: from src.utils.paths import *
setup_paths()


if __name__ == '__main__':
    # Test path resolution
    print("Proteus Path Configuration")
    print("=" * 50)
    print(f"Project Root: {get_project_root()}")
    print(f"Src Dir:      {get_src_dir()}")
    print(f"Data Dir:     {get_data_dir()}")
    print(f"Config Dir:   {get_config_dir()}")
    print(f"Logs Dir:     {get_logs_dir()}")
    print(f"Models Dir:   {get_models_dir()}")
    print(f"Scripts Dir:  {get_scripts_dir()}")
    print(f"Tests Dir:    {get_tests_dir()}")
    print()
    print("sys.path includes:")
    for p in sys.path[:5]:
        print(f"  {p}")

#!/usr/bin/env python
"""
Repository Cleanup Script

Organizes and cleans up the Proteus repository:
- Archives old scan files (keeps last 7 days)
- Removes temporary files
- Moves loose scripts to proper locations
- Reports on untracked files that should be gitignored

Usage:
    python scripts/cleanup_repo.py              # Dry run - show what would be done
    python scripts/cleanup_repo.py --execute   # Actually perform cleanup
    python scripts/cleanup_repo.py --scans     # Only clean scan files
"""

import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple


class RepoCleanup:
    """Clean up and organize the repository."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.scans_dir = self.project_root / 'data' / 'smart_scans'
        self.archive_dir = self.scans_dir / 'archived'

    def get_old_scans(self, days_to_keep: int = 7) -> List[Path]:
        """Find scan files older than days_to_keep."""
        cutoff = datetime.now() - timedelta(days=days_to_keep)
        old_scans = []

        for scan_file in self.scans_dir.glob('scan_*.json'):
            # Parse date from filename: scan_YYYYMMDD_HHMMSS.json
            try:
                date_str = scan_file.stem.split('_')[1]
                file_date = datetime.strptime(date_str, '%Y%m%d')
                if file_date < cutoff:
                    old_scans.append(scan_file)
            except (IndexError, ValueError):
                continue

        return sorted(old_scans)

    def archive_old_scans(self, dry_run: bool = True) -> Tuple[int, int]:
        """Archive scan files older than 7 days."""
        old_scans = self.get_old_scans(days_to_keep=7)

        if not old_scans:
            print("No old scan files to archive")
            return 0, 0

        if not dry_run:
            self.archive_dir.mkdir(exist_ok=True)

        archived = 0
        for scan_file in old_scans:
            dest = self.archive_dir / scan_file.name
            if dry_run:
                print(f"  Would archive: {scan_file.name}")
            else:
                shutil.move(str(scan_file), str(dest))
                print(f"  Archived: {scan_file.name}")
            archived += 1

        return archived, len(old_scans)

    def find_temp_files(self) -> List[Path]:
        """Find temporary files that can be deleted."""
        temp_patterns = [
            'tmpclaude-*',
            '*.tmp',
            '*.bak',
            '*~',
            '*.pyc',
            '__pycache__',
        ]

        temp_files = []
        for pattern in temp_patterns:
            temp_files.extend(self.project_root.glob(pattern))
            temp_files.extend(self.project_root.glob(f'**/{pattern}'))

        return temp_files

    def find_loose_scripts(self) -> List[Path]:
        """Find Python scripts in root that should be in scripts/."""
        loose = []
        exclude = {'setup.py', 'conftest.py'}

        for py_file in self.project_root.glob('*.py'):
            if py_file.name not in exclude:
                loose.append(py_file)

        return loose

    def find_generated_files(self) -> List[Path]:
        """Find generated files that should be gitignored."""
        patterns = [
            '*.html',           # Generated reports
            '*.log',            # Log files
            'nul',              # Windows null file
            '*.pkl',            # Pickle files
            '*.joblib',         # Joblib files
        ]

        generated = []
        for pattern in patterns:
            generated.extend(self.project_root.glob(pattern))

        return generated

    def get_gitignore_suggestions(self) -> List[str]:
        """Suggest additions to .gitignore."""
        suggestions = [
            "# Generated reports",
            "*.html",
            "daily_report_*.html",
            "",
            "# Research results (keep conclusions, ignore raw data)",
            "data/research/*.json",
            "data/research/*.txt",
            "",
            "# Scan archives",
            "data/smart_scans/archived/",
            "",
            "# Cache directories",
            "data/cache/",
            "",
            "# Model files",
            "data/models/",
            "models/",
            "",
            "# Virtual wallet data (local state)",
            "features/simulation/data/virtual_wallet/",
            "",
            "# Portfolio data",
            "features/simulation/data/portfolio/",
            "",
            "# Loose planning docs (keep in separate repo or local)",
            "MORNING_*.md",
            "MORNING_*.txt",
            "OVERNIGHT_*.md",
            "OVERNIGHT_*.txt",
            "*_RESULTS*.txt",
            "*_RESULTS*.md",
        ]
        return suggestions

    def run_cleanup(self, dry_run: bool = True, scans_only: bool = False) -> dict:
        """Run the full cleanup."""
        results = {
            'scans_archived': 0,
            'temp_deleted': 0,
            'loose_scripts': 0,
            'generated_files': 0,
        }

        print("\n" + "=" * 60)
        print("REPOSITORY CLEANUP")
        print("=" * 60)

        if dry_run:
            print("DRY RUN - use --execute to perform cleanup\n")

        # 1. Archive old scans
        print("\n[1] Old Scan Files (>7 days)")
        print("-" * 40)
        archived, total = self.archive_old_scans(dry_run)
        results['scans_archived'] = archived
        print(f"    {archived} files to archive")

        if scans_only:
            return results

        # 2. Temp files
        print("\n[2] Temporary Files")
        print("-" * 40)
        temp_files = self.find_temp_files()
        for f in temp_files[:10]:
            print(f"  Would delete: {f.name}" if dry_run else f"  Deleted: {f.name}")
        if len(temp_files) > 10:
            print(f"  ... and {len(temp_files) - 10} more")
        results['temp_deleted'] = len(temp_files)

        if not dry_run:
            for f in temp_files:
                if f.is_dir():
                    shutil.rmtree(f, ignore_errors=True)
                else:
                    f.unlink(missing_ok=True)

        # 3. Loose scripts
        print("\n[3] Loose Scripts in Root")
        print("-" * 40)
        loose = self.find_loose_scripts()
        for f in loose[:10]:
            print(f"  {f.name} -> scripts/{f.name}")
        if len(loose) > 10:
            print(f"  ... and {len(loose) - 10} more")
        results['loose_scripts'] = len(loose)
        print(f"    Consider moving {len(loose)} scripts to scripts/ directory")

        # 4. Generated files
        print("\n[4] Generated Files (should be gitignored)")
        print("-" * 40)
        generated = self.find_generated_files()
        for f in generated[:5]:
            print(f"  {f.name}")
        if len(generated) > 5:
            print(f"  ... and {len(generated) - 5} more")
        results['generated_files'] = len(generated)

        # 5. Gitignore suggestions
        print("\n[5] Suggested .gitignore Additions")
        print("-" * 40)
        suggestions = self.get_gitignore_suggestions()
        for line in suggestions[:15]:
            print(f"  {line}")
        print("  ...")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Scans to archive:    {results['scans_archived']}")
        print(f"  Temp files:          {results['temp_deleted']}")
        print(f"  Loose scripts:       {results['loose_scripts']}")
        print(f"  Generated files:     {results['generated_files']}")

        if dry_run:
            print("\nRun with --execute to perform cleanup")

        return results


def main():
    parser = argparse.ArgumentParser(description='Clean up repository')
    parser.add_argument('--execute', action='store_true',
                       help='Actually perform cleanup (default is dry run)')
    parser.add_argument('--scans', action='store_true',
                       help='Only clean scan files')

    args = parser.parse_args()

    cleanup = RepoCleanup()
    cleanup.run_cleanup(dry_run=not args.execute, scans_only=args.scans)


if __name__ == '__main__':
    main()

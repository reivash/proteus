#!/usr/bin/env python
"""
Experiment Archival System

Archives completed experiments while preserving conclusions in a searchable format.

Features:
- Extracts docstrings/conclusions from experiment files
- Moves old experiments to archived/ folder
- Creates EXPERIMENT_CONCLUSIONS.md with all findings
- Maintains manifest of archived experiments

Usage:
    python scripts/archive_experiments.py                    # Dry run - show what would be archived
    python scripts/archive_experiments.py --execute          # Actually archive
    python scripts/archive_experiments.py --rebuild-summary  # Rebuild conclusions from all experiments
    python scripts/archive_experiments.py --list             # List all experiments with status
"""

import argparse
import ast
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ExperimentArchiver:
    """Archive experiments while preserving knowledge."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.experiments_dir = self.project_root / 'src' / 'experiments'
        self.archived_dir = self.project_root / 'src' / 'experiments' / 'archived'
        self.research_data_dir = self.project_root / 'data' / 'research'
        self.conclusions_file = self.project_root / 'EXPERIMENT_CONCLUSIONS.md'
        self.manifest_file = self.archived_dir / 'manifest.json'

        # Experiments to keep active (recent or ongoing)
        self.keep_active = {
            'exp092', 'exp093', 'exp094', 'exp095',  # Recent tuning
            'exp087', 'exp088', 'exp089', 'exp090', 'exp091',  # Model comparison (reference)
        }

        # Experiment ranges that are completed and can be archived
        self.archive_ranges = [
            (1, 7),    # Baseline experiments
            (8, 19),   # Mean reversion variants
            (20, 45),  # Position sizing, exits
            (46, 70),  # Universe expansion
            (71, 86),  # Early ML experiments
        ]

    def extract_docstring(self, filepath: Path) -> Optional[str]:
        """Extract the module docstring from a Python file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse the AST to get the docstring
            tree = ast.parse(content)
            docstring = ast.get_docstring(tree)

            if docstring:
                return docstring.strip()

            # Fallback: look for triple-quoted string at top
            match = re.match(r'^["\'][\'"]{2}(.*?)["\'][\'"]{2}', content, re.DOTALL)
            if match:
                return match.group(1).strip()

            return None
        except Exception as e:
            print(f"  Warning: Could not parse {filepath.name}: {e}")
            return None

    def get_experiment_number(self, filename: str) -> Optional[int]:
        """Extract experiment number from filename."""
        match = re.match(r'exp(\d+)', filename)
        if match:
            return int(match.group(1))
        return None

    def should_archive(self, exp_num: int) -> bool:
        """Check if experiment should be archived based on ranges."""
        for start, end in self.archive_ranges:
            if start <= exp_num <= end:
                return True
        return False

    def list_experiments(self) -> List[Dict]:
        """List all experiments with their status."""
        experiments = []

        for filepath in sorted(self.experiments_dir.glob('exp*.py')):
            exp_num = self.get_experiment_number(filepath.name)
            if exp_num is None:
                continue

            docstring = self.extract_docstring(filepath)
            title = docstring.split('\n')[0] if docstring else "No description"

            # Determine status
            if filepath.name.replace('.py', '') in self.keep_active:
                status = 'ACTIVE'
            elif self.should_archive(exp_num):
                status = 'ARCHIVE'
            else:
                status = 'KEEP'

            experiments.append({
                'number': exp_num,
                'filename': filepath.name,
                'title': title[:60] + '...' if len(title) > 60 else title,
                'status': status,
                'path': filepath
            })

        return experiments

    def extract_conclusions(self, docstring: str) -> Dict:
        """Extract structured conclusions from docstring."""
        conclusions = {
            'objective': '',
            'methodology': '',
            'results': '',
            'conclusion': '',
            'raw': docstring
        }

        # Look for common section headers
        sections = {
            'objective': r'(?:OBJECTIVE|Goal|Purpose)[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)',
            'methodology': r'(?:METHOD|METHODOLOGY|Approach)[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)',
            'results': r'(?:RESULTS?|Findings|Outcome)[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)',
            'conclusion': r'(?:CONCLUSION|Summary|Takeaway)[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)',
        }

        for key, pattern in sections.items():
            match = re.search(pattern, docstring, re.IGNORECASE | re.DOTALL)
            if match:
                conclusions[key] = match.group(1).strip()

        return conclusions

    def build_conclusions_markdown(self, experiments: List[Dict]) -> str:
        """Build the conclusions markdown file."""
        lines = [
            "# Experiment Conclusions",
            "",
            f"> Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "> Source: Extracted from experiment docstrings",
            "",
            "---",
            "",
            "## Summary by Phase",
            "",
        ]

        # Group by phase
        phases = {
            'Phase 1: Baseline (exp001-007)': (1, 7),
            'Phase 2: Mean Reversion (exp008-019)': (8, 19),
            'Phase 3: Position & Exit Optimization (exp020-045)': (20, 45),
            'Phase 4: Universe Expansion (exp046-070)': (46, 70),
            'Phase 5: ML Model Development (exp071-091)': (71, 91),
            'Phase 6: Production Tuning (exp092+)': (92, 999),
        }

        for phase_name, (start, end) in phases.items():
            phase_exps = [e for e in experiments if start <= e['number'] <= end]
            if not phase_exps:
                continue

            lines.append(f"### {phase_name}")
            lines.append("")

            for exp in phase_exps:
                docstring = self.extract_docstring(exp['path'])
                if not docstring:
                    continue

                conclusions = self.extract_conclusions(docstring)

                lines.append(f"#### {exp['filename']}")
                if conclusions['objective']:
                    lines.append(f"**Objective**: {conclusions['objective'][:200]}")
                if conclusions['conclusion']:
                    lines.append(f"**Conclusion**: {conclusions['conclusion'][:300]}")
                elif conclusions['results']:
                    lines.append(f"**Results**: {conclusions['results'][:300]}")
                lines.append("")

            lines.append("---")
            lines.append("")

        # Add key findings summary
        lines.extend([
            "## Key Findings",
            "",
            "### What Works",
            "- LSTM V2 model: 79% win rate, 4.96 Sharpe",
            "- Tier-based exits: Elite stocks get longer hold periods",
            "- Regime-adaptive thresholds: Higher bar in choppy markets",
            "- Earnings avoidance: Skip 3 days before earnings",
            "- Sector diversification: Max 2 per sector",
            "",
            "### What Doesn't Work",
            "- Simple MA crossover: Only 52% win rate",
            "- Fixed position sizing: ATR-based is better",
            "- Single model: Ensemble outperforms",
            "- Ignoring regime: Choppy markets are traps",
            "",
            "### Open Questions",
            "- Intraday entry optimization",
            "- Options integration",
            "- Small cap expansion",
            "",
        ])

        return '\n'.join(lines)

    def archive_experiments(self, dry_run: bool = True) -> Tuple[int, int]:
        """Archive completed experiments."""
        self.archived_dir.mkdir(exist_ok=True)

        experiments = self.list_experiments()
        archived = 0
        skipped = 0

        # Load existing manifest
        manifest = {}
        if self.manifest_file.exists():
            with open(self.manifest_file) as f:
                manifest = json.load(f)

        for exp in experiments:
            if exp['status'] != 'ARCHIVE':
                skipped += 1
                continue

            src = exp['path']
            dst = self.archived_dir / exp['filename']

            if dry_run:
                print(f"  Would archive: {exp['filename']} -> archived/")
            else:
                # Extract docstring before moving
                docstring = self.extract_docstring(src)

                # Move file
                shutil.move(str(src), str(dst))
                print(f"  Archived: {exp['filename']}")

                # Update manifest
                manifest[exp['filename']] = {
                    'number': exp['number'],
                    'title': exp['title'],
                    'archived_date': datetime.now().isoformat(),
                    'docstring': docstring[:500] if docstring else None
                }

            archived += 1

        if not dry_run and manifest:
            with open(self.manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)

        return archived, skipped

    def rebuild_conclusions(self) -> None:
        """Rebuild conclusions from all experiments (active + archived)."""
        all_experiments = []

        # Get active experiments
        for filepath in sorted(self.experiments_dir.glob('exp*.py')):
            if filepath.parent.name == 'archived':
                continue
            exp_num = self.get_experiment_number(filepath.name)
            if exp_num:
                all_experiments.append({
                    'number': exp_num,
                    'filename': filepath.name,
                    'title': '',
                    'status': 'ACTIVE',
                    'path': filepath
                })

        # Get archived experiments
        if self.archived_dir.exists():
            for filepath in sorted(self.archived_dir.glob('exp*.py')):
                exp_num = self.get_experiment_number(filepath.name)
                if exp_num:
                    all_experiments.append({
                        'number': exp_num,
                        'filename': filepath.name,
                        'title': '',
                        'status': 'ARCHIVED',
                        'path': filepath
                    })

        # Sort by number
        all_experiments.sort(key=lambda x: x['number'])

        # Build and save conclusions
        markdown = self.build_conclusions_markdown(all_experiments)
        with open(self.conclusions_file, 'w', encoding='utf-8') as f:
            f.write(markdown)

        print(f"Rebuilt conclusions: {self.conclusions_file}")
        print(f"  Total experiments: {len(all_experiments)}")


def main():
    parser = argparse.ArgumentParser(description='Archive completed experiments')
    parser.add_argument('--execute', action='store_true',
                       help='Actually archive (default is dry run)')
    parser.add_argument('--rebuild-summary', action='store_true',
                       help='Rebuild conclusions markdown from all experiments')
    parser.add_argument('--list', action='store_true',
                       help='List all experiments with status')

    args = parser.parse_args()

    archiver = ExperimentArchiver()

    if args.list:
        print("\nExperiment Status:")
        print("=" * 80)
        experiments = archiver.list_experiments()
        for exp in experiments:
            status_icon = {'ACTIVE': '*', 'ARCHIVE': '-', 'KEEP': ' '}[exp['status']]
            print(f"[{status_icon}] exp{exp['number']:03d}: {exp['title']}")
        print("\n* = Active, - = Archive candidate, (space) = Keep")
        print(f"\nTotal: {len(experiments)} experiments")
        return

    if args.rebuild_summary:
        archiver.rebuild_conclusions()
        return

    # Default: archive
    print("\nExperiment Archival")
    print("=" * 60)

    if not args.execute:
        print("DRY RUN - use --execute to actually archive\n")

    archived, skipped = archiver.archive_experiments(dry_run=not args.execute)

    print(f"\nSummary:")
    print(f"  Archived: {archived}")
    print(f"  Skipped: {skipped}")

    if not args.execute:
        print("\nRun with --execute to perform archival")
    else:
        # Rebuild conclusions after archiving
        print("\nRebuilding conclusions...")
        archiver.rebuild_conclusions()


if __name__ == '__main__':
    main()

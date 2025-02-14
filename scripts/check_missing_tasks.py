#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Set


def get_project_root() -> Path:
    """Get the project root directory."""
    # Assuming this script is in the scripts directory
    return Path(__file__).parent.parent


def get_available_tasks() -> Set[str]:
    """Get all available tasks from the conf/task directory."""
    task_dir = get_project_root() / 'ml_dev_bench' / 'conf' / 'task'
    return {p.stem for p in task_dir.glob('*.yaml')}


def get_completed_tasks(results_dir: str) -> Set[str]:
    """Get all completed tasks from the specified results directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return set()
    return {p.name for p in results_path.iterdir() if p.is_dir()}


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Check missing tasks from results directory'
    )
    parser.add_argument(
        '--results-dir',
        default='results',
        help='Directory containing results (default: results)',
    )

    args = parser.parse_args()

    # Get tasks
    available_tasks = get_available_tasks()
    completed_tasks = get_completed_tasks(args.results_dir)

    # Calculate missing tasks
    missing_tasks = available_tasks - completed_tasks

    # Print results
    print('=== Task Execution Status ===\n')
    print(f'Results directory: {args.results_dir}\n')

    print('Tasks not yet executed:')
    for task in sorted(missing_tasks):
        print(f'- {task}')

    print('\n=== Summary ===')
    print(f'Total available tasks: {len(available_tasks)}')
    print(f'Completed tasks: {len(completed_tasks)}')
    print(f'Remaining tasks: {len(missing_tasks)}')

    print('\nCompleted tasks:')
    for task in sorted(completed_tasks):
        print(f'- {task}')

    # Add comma-separated list of remaining tasks
    print('\nRemaining tasks (comma-separated):')
    print(','.join(sorted(missing_tasks)))


if __name__ == '__main__':
    main()

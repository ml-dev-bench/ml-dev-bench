import argparse
import asyncio
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv

import calipers
from calipers.framework import EvaluationFramework, EvaluationResult
from calipers.logger import logger, setup_logger


def save_results(
    results: List[EvaluationResult],
    output_dir: Union[str, Path],
    config: Dict[str, Any],
    commit_hash: str,
) -> None:
    """Save evaluation results to JSON file"""
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'eval_results_{timestamp}.json'

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    # Convert results to serializable format
    results_dict = {
        'metadata': {
            'timestamp': timestamp,
            'commit_hash': commit_hash,
            'config': config,
        },
        'results': [
            {
                'task_id': result.task_id,
                'agent_id': result.agent_id,
                'categories': list(result.categories),
                'success_rate': result.success_rate,
                'avg_duration': result.avg_duration,
                'aggregated_metrics': result.aggregated_metrics,
                'runs': [
                    {
                        'success': run.success,
                        'metrics': run.metrics,
                        'validation_details': run.validation_details,
                        'agent_output': run.agent_output,
                        'start_time': run.start_time.isoformat(),
                        'end_time': run.end_time.isoformat(),
                        'error': run.error,
                    }
                    for run in result.runs
                ],
            }
            for result in results
        ],
    }

    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2, cls=NumpyEncoder)

    logger.info('Results saved to %s', output_file)


def print_results(results: List[EvaluationResult]) -> None:
    """Print evaluation results in a formatted way"""
    for result in results:
        logger.info('\nEvaluation Results for %s:', result.task_id)
        logger.info('Categories: %s', ', '.join(result.categories))
        logger.info(
            'Success Rate: %.2f (%d/%d runs)',
            result.success_rate,
            sum(1 for run in result.runs if run.success),
            len(result.runs),
        )

        # Print aggregated metrics
        logger.info('Aggregated Metrics:')
        for metric, stats in result.aggregated_metrics.items():
            logger.info('  %s:', metric)
            for stat_name, value in stats.items():
                # Skip non-numeric and metadata fields
                if stat_name in ('unit', 'description'):
                    continue
                if isinstance(value, (int, float)):
                    logger.info('    %s: %.2f', stat_name, value)
                else:
                    logger.info('    %s: %s', stat_name, value)


async def run_evaluation(
    config_path: str,
    tasks: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    commit_hash: Optional[str] = None,
    langchain_project: Optional[str] = None,
    debug_mode: bool = False,
) -> List[EvaluationResult]:
    """Run evaluation with given config"""

    logger.info('Loading config from %s', config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load environment variables
    default_env_file = Path(calipers.__path__[0]).parent / '.env.runtime'
    env_file = config.get('env_file', default_env_file)
    if os.path.exists(env_file):
        logger.info(f'Loading environment variables from {env_file}')
        load_dotenv(env_file)
    else:
        logger.info(f'No environment file found at {env_file}')

    # Set LANGCHAIN_PROJECT if provided
    langchain_project = langchain_project or config.get('langchain_project')
    if langchain_project:
        os.environ['LANGCHAIN_PROJECT'] = langchain_project
        logger.info(f'Set LANGCHAIN_PROJECT to {langchain_project}')

    # Setup logging
    setup_logger(level=getattr(logging, config.get('log_level', 'INFO')))

    # Import task packages if specified
    if config.get('task_packages'):
        for package in config['task_packages']:
            try:
                logger.info(f'Importing task package: {package}')
                __import__(package)
            except ImportError as e:
                logger.warning(f'Failed to import task package {package}: {e}')

    # Import agent packages if specified
    if config.get('agent_packages'):
        for package in config['agent_packages']:
            try:
                logger.info(f'Importing agent package: {package}')
                __import__(package)
            except ImportError as e:
                logger.warning(f'Failed to import agent package {package}: {e}')

    # Get commit hash if not provided
    if not commit_hash:
        import git

        try:
            repo = git.Repo(search_parent_directories=True)
            commit_hash = repo.head.object.hexsha
        except (ImportError, git.InvalidGitRepositoryError):
            commit_hash = 'unknown'
            logger.warning('Could not determine git commit hash')

    # Set output_dir
    output_dir_path = Path(output_dir) if output_dir else Path('./results')

    root_dir = Path(calipers.__path__[0]).parent
    if not output_dir_path.is_absolute():
        output_dir_path = root_dir / output_dir_path

    # Validate config
    if config.get('category_filters') and (config.get('tasks') or tasks):
        raise ValueError(
            'Config cannot specify both category_filters and tasks. '
            'Please use only one method to select tasks.'
        )

    # Handle workspace cloning
    workspace_temp_dir = None
    if 'workspace_dir' in config:
        if not os.path.isabs(config['workspace_dir']):
            config['workspace_dir'] = str(root_dir / config['workspace_dir'])

        # Clone if debug mode is enabled or clone_workspace is specified
        if debug_mode or config.get('clone_workspace'):
            if config.get('clone_workspace_to'):
                workspace_temp_dir = config['clone_workspace_to']
                if not os.path.isabs(workspace_temp_dir):
                    workspace_temp_dir = str(root_dir / workspace_temp_dir)
                    logger.warning(
                        'Relative path provided for clone_workspace_to, using absolute path: %s',
                        workspace_temp_dir,
                    )
                # Remove contents of directory without deleting the directory itself
                if os.path.exists(workspace_temp_dir):
                    for item in os.listdir(workspace_temp_dir):
                        item_path = os.path.join(workspace_temp_dir, item)
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                else:
                    os.makedirs(workspace_temp_dir)
            else:
                workspace_temp_dir = tempfile.mkdtemp(prefix='workspace_clone_')
            logger.info(
                f'Cloning workspace to temporary directory: {workspace_temp_dir}'
            )
            shutil.copytree(
                config['workspace_dir'], workspace_temp_dir, dirs_exist_ok=True
            )
            config['workspace_dir'] = workspace_temp_dir
            if debug_mode:
                logger.info(
                    'Debug mode: Workspace clone will be preserved at '
                    f'{workspace_temp_dir}'
                )
        else:
            if config.get('clear_workspace', False):
                shutil.rmtree(config['workspace_dir'], ignore_errors=True)
                os.makedirs(config['workspace_dir'], exist_ok=True)

            # Check if workspace directory is not empty
            # if len(os.listdir(config['workspace_dir'])) > 0:
            #     raise ValueError(
            #         'Workspace directory is not empty. Set clear_workspace=True or enable debug / clone_workspace mode.'
            #     )

        logger.info(f'Workspace dir: {config["workspace_dir"]}')

    try:
        framework = EvaluationFramework(config)
        # Get tasks to evaluate
        task_ids = set()
        if tasks:
            task_ids.update(tasks)
        elif config.get('category_filters'):
            task_ids.update(
                framework.get_tasks_by_category_filter(config['category_filters'])
            )
        elif config.get('tasks'):
            task_ids.update(t['id'] for t in config['tasks'])
        else:
            logger.warning('No tasks selected for evaluation')
            return []

        logger.info('Selected tasks: %s', task_ids)

        # Run evaluations
        results = []
        for task_id in task_ids:
            try:
                result = await framework.evaluate(
                    task_id=task_id,
                    agent_id=config['agent']['id'],
                    num_runs=config.get('num_runs', 1),
                )
                results.append(result)
            except Exception as e:
                logger.error('Error evaluating task %s: %s', task_id, e)
                if config.get('fail_fast', False):
                    raise

        # Print and save results
        logger.info('Running %d tasks', len(results))
        print_results(results)
        save_results(results, str(output_dir_path), config, commit_hash or 'unknown')

        return results

    finally:
        # Clean up workspace if not in debug mode
        if workspace_temp_dir and not debug_mode:
            logger.info(f'Cleaning up temporary workspace: {workspace_temp_dir}')
            shutil.rmtree(workspace_temp_dir, ignore_errors=True)
        elif workspace_temp_dir:
            logger.info(
                f'Debug mode: Preserving workspace clone at {workspace_temp_dir}'
            )


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='*',
        help='Specific task IDs to run',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save evaluation results',
    )
    parser.add_argument(
        '--commit-hash', type=str, help='Git commit hash for version tracking'
    )
    parser.add_argument(
        '--langchain-project', type=str, help='LangChain project name for tracking'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Enable debug mode - preserves cloned workspace',
    )
    args = parser.parse_args()

    # Run evaluation
    results = await run_evaluation(
        args.config,
        args.tasks,
        args.output_dir,
        args.commit_hash,
        args.langchain_project,
        debug_mode=args.debug,
    )

    # Exit with error if any test had 0% success rate
    if any(result.success_rate == 0 for result in results):
        logger.error('One or more tests had 0% success rate')
        exit(1)


if __name__ == '__main__':
    asyncio.run(main())

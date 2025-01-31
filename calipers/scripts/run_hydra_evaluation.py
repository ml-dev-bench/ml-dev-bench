import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import json

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import calipers

from calipers.framework import EvaluationFramework, EvaluationResult
from calipers.logger import logger, setup_logger

# Get the project root directory
PROJECT_ROOT = Path(calipers.__file__).parent.parent

cs = ConfigStore.instance()
cs.store(name="config", node=DictConfig)


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


@hydra.main(
    version_base=None,
    config_path=str(PROJECT_ROOT / "ml_dev_bench/conf"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Run evaluation with Hydra config"""

    # Convert to regular dict for compatibility with existing code
    config = OmegaConf.to_container(cfg, resolve=True)

    # Setup logging
    setup_logger(level=getattr(logging, config.get('log_level', 'INFO')))

    # Import task packages
    if config.get('task_packages'):
        for package in config['task_packages']:
            try:
                logger.info(f'Importing task package: {package}')
                __import__(package)
            except ImportError as e:
                logger.warning(f'Failed to import task package {package}: {e}')

    # Import agent packages
    if config.get('agent_packages'):
        for package in config['agent_packages']:
            try:
                logger.info(f'Importing agent package: {package}')
                __import__(package)
            except ImportError as e:
                logger.warning(f'Failed to import agent package {package}: {e}')

    # Set LANGCHAIN_PROJECT if provided
    if config.get('langchain_project'):
        os.environ['LANGCHAIN_PROJECT'] = config['langchain_project']
        logger.info(f'Set LANGCHAIN_PROJECT to {config["langchain_project"]}')

    # Create framework
    framework = EvaluationFramework(config)

    # Get tasks to evaluate
    task_ids = set()
    if config.get('tasks'):
        task_ids.update(t['id'] for t in config['tasks'])
    else:
        logger.warning('No tasks selected for evaluation')
        return

    logger.info('Selected tasks: %s', task_ids)

    # Get commit hash
    try:
        import git

        repo = git.Repo(search_parent_directories=True)
        commit_hash = repo.head.object.hexsha
    except (ImportError, git.InvalidGitRepositoryError):
        commit_hash = 'unknown'
        logger.warning('Could not determine git commit hash')

    # Run evaluations
    results = []
    for task_id in task_ids:
        try:
            result = framework.evaluate(
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
    save_results(results, config['output_dir'], config, commit_hash)

    # Exit with error if any test had 0% success rate
    if any(result.success_rate == 0 for result in results):
        logger.error('One or more tests had 0% success rate')
        exit(1)


if __name__ == '__main__':
    main()

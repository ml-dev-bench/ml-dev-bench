import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import json
import yaml
import tempfile
import shutil

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import calipers

from calipers.framework import EvaluationFramework, EvaluationResult
from calipers.logger import logger, setup_logger
from calipers.scripts.run_evaluation import run_evaluation

# Get the project root directory
PROJECT_ROOT = Path(calipers.__file__).parent.parent


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
    """Entry point for Hydra"""
    print(OmegaConf.to_yaml(cfg))

    # Convert to regular dict for compatibility with existing code
    config = OmegaConf.to_container(cfg, resolve=True)

    # Get output directory from hydra config
    output_dir = config.get('output_dir', './results')

    # Get langchain project from config
    langchain_project = config.get('langchain_project')

    # Get debug mode from config
    debug_mode = config.get('debug', False)

    # Get commit hash if possible
    try:
        import git

        repo = git.Repo(search_parent_directories=True)
        commit_hash = repo.head.object.hexsha
    except (ImportError, git.InvalidGitRepositoryError):
        commit_hash = None
        logger.warning('Could not determine git commit hash')

    # Create temporary directory for config
    temp_dir = tempfile.mkdtemp(prefix='hydra_config_')
    try:
        # Save config to temporary file
        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        import asyncio

        asyncio.run(
            run_evaluation(
                config_path=str(config_path),
                output_dir=output_dir,
                commit_hash=commit_hash,
                langchain_project=langchain_project,
                debug_mode=debug_mode,
            )
        )
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()

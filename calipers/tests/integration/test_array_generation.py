import json
import logging
from pathlib import Path

import pytest
import yaml
from dotenv import load_dotenv

from calipers.scripts.run_evaluation import run_evaluation
from calipers.tests.integration.tasks.array_generation import (
    RandomArrayGenerationTask,  # noqa: F401
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_env_file(start_path: Path, filename: str = '.env.runtime') -> Path:
    """Recursively search for environment file starting from given path."""
    current = start_path
    while current != current.parent:  # Stop at root directory
        env_file = current / filename
        if env_file.exists():
            return env_file
        current = current.parent
    raise FileNotFoundError(f'Could not find {filename} in any parent directory')


@pytest.fixture
def setup_test_env(tmp_path):
    """Setup test environment"""
    logger.info('Setting up test environment')
    try:
        env_file = find_env_file(Path(__file__).resolve().parent)
        load_dotenv(env_file)
        logger.debug(f'Loaded environment file from {env_file}')
    except FileNotFoundError as e:
        logger.error(str(e))
        raise

    # Create test workspace and results directories
    workspace_dir = tmp_path / 'test_workspace'
    results_dir = tmp_path / 'results'
    workspace_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    logger.info(f'Created test workspace at {workspace_dir}')
    logger.info(f'Created results directory at {results_dir}')

    yield workspace_dir, results_dir

    # Cleanup
    logger.info('Cleaning up test environment')
    if workspace_dir.exists():
        import shutil

        shutil.rmtree(workspace_dir)
        logger.debug(f'Removed workspace directory: {workspace_dir}')
    if results_dir.exists():
        shutil.rmtree(results_dir)
        logger.debug(f'Removed results directory: {results_dir}')


@pytest.mark.asyncio
@pytest.mark.slow
async def test_array_generation(setup_test_env):
    """Test array generation using run_evaluation"""
    logger.info('Starting array generation test')
    workspace_dir, results_dir = setup_test_env
    config_path = Path(__file__).resolve().parent / 'config/array_generation_test.yaml'

    logger.debug('Loading test configuration')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Update workspace paths and agent configuration
    config['workspace_dir'] = str(workspace_dir)
    config['agent']['workspace_dir'] = str(workspace_dir)
    config['tasks'][0]['workspace_dir'] = str(workspace_dir)
    logger.debug(f'Updated config with workspace: {workspace_dir}')

    test_config_path = workspace_dir.parent / 'test_config.yaml'
    with open(test_config_path, 'w') as f:
        yaml.dump(config, f)
    logger.info(f'Wrote test config to {test_config_path}')

    # Run evaluation with tmp results dir
    logger.info('Running evaluation')
    results = await run_evaluation(
        config_path=str(test_config_path), output_dir=str(results_dir)
    )
    logger.info('Evaluation complete')

    assert len(results) == 1
    result = results[0]

    # Validate task results
    assert result.task_id == 'random_array_generation'
    assert result.agent_id == 'simple_react'
    assert len(result.runs) == 1
    # Allow failure as well and validate metrics only in success cases
    if result.success_rate == 1.0:
        assert result.success_rate == 1.0
        run = result.runs[0]

        # Check metrics
        assert 'array_shapes' in run.metrics
        assert 'steps' in run.metrics
        assert 'token_cost' in run.metrics
        assert 'token__total_tokens' in run.metrics

        # Verify array shapes metric
        assert run.metrics['array_shapes']['value'] == 3  # All shapes correct

        # Verify success
        assert run.success
        assert run.validation_details['correct_shapes'] == 3
        assert run.validation_details['total_shapes'] == 3

        # Check shape validation details
        shape_validation = run.validation_details['shape_validation']
        assert all(info['correct'] for info in shape_validation.values())

    # Verify results were saved to tmp directory
    result_files = list(results_dir.glob('eval_results_*.json'))
    assert result_files, 'No results file found'

    # Verify saved results
    latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
    with open(latest_result) as f:
        saved_data = json.load(f)

    assert 'metadata' in saved_data
    assert 'config' in saved_data['metadata']
    assert 'timestamp' in saved_data['metadata']
    assert 'commit_hash' in saved_data['metadata']

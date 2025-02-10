import hashlib
import json
import os
from typing import Any, Dict

from composio import Action

from calipers.framework.base import (
    BaseAgent,
    BaseEvaluationTask,
    BaseRuntime,
)
from calipers.framework.registry import EvalRegistry
from calipers.runtime.ml_dev_bench_runtime import MLDevBenchRuntime


@EvalRegistry.register_task
class CIFAR100PerformanceTask(BaseEvaluationTask):
    task_id = 'cifar100_performance'
    description = 'Train a model to achieve high accuracy on CIFAR-100'
    categories = {'ml', 'training', 'computer_vision', 'classification'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = self.workspace_dir / 'checkpoints'
        self.best_checkpoint_dir = self.workspace_dir / 'best_checkpoint'

    def _compute_file_hash(self, filepath) -> str:
        """Compute SHA256 hash of a file.

        Args:
            filepath: Path to the file to hash

        Returns:
            str: Hex digest of file hash
        """
        with open(filepath, 'rb') as f:
            content = f.read()
            return hashlib.sha256(content).hexdigest()

    def initialize(self) -> None:
        # Calculate and store hash of evaluate.py
        evaluate_script = self.workspace_dir / 'evaluate.py'
        self.evaluate_script_hash = self._compute_file_hash(evaluate_script)

    async def run(self, agent: BaseAgent) -> Dict[str, Any]:
        task_path = os.path.join(os.path.dirname(__file__), 'task.txt')
        with open(task_path, 'r') as f:
            task_prompt = f.read()
        return await agent.run(task_prompt)

    async def validate(
        self,
        agent_output: Dict[str, Any],
        runtime: BaseRuntime,
    ) -> Dict[str, Any]:
        """
        Validate the agent output for the CIFAR-100 training task.
        Expected Directory Structure:
            workspace/
            ├── evaluate.py
            ├── checkpoints/
            ├── best_checkpoint/
            ├── wandb_info.json
            └── eval_metrics.json
        """
        validation_results = {
            'training_setup': {'status': 'invalid'},
            'model_checkpoints': {'status': 'invalid'},
            'wandb_logging': {'status': 'invalid'},
            'model_performance': {'status': 'invalid'},
        }

        try:
            # Check training directory structure
            if not self._validate_training_setup():
                return {
                    'success': False,
                    'validation_results': validation_results,
                    'error': 'Training setup validation failed',
                }
            validation_results['training_setup']['status'] = 'valid'

            # Validate model checkpoints
            checkpoints_valid, checkpoints_error = self._validate_checkpoints()
            if not checkpoints_valid:
                validation_results['model_checkpoints']['error'] = checkpoints_error
                return {
                    'success': False,
                    'validation_results': validation_results,
                    'error': (
                        f'Model checkpoint validation failed: {checkpoints_error}'
                    ),
                }
            validation_results['model_checkpoints']['status'] = 'valid'

            # Validate wandb logging
            wandb_valid, wandb_error = self._validate_wandb_logging()
            if not wandb_valid:
                validation_results['wandb_logging']['error'] = wandb_error
                return {
                    'success': False,
                    'validation_results': validation_results,
                    'error': f'W&B logging validation failed: {wandb_error}',
                }
            validation_results['wandb_logging']['status'] = 'valid'

            # Run evaluation and validate performance
            perf_valid, perf_error, metrics = await self._validate_model_performance(
                runtime
            )
            if not perf_valid:
                validation_results['model_performance']['error'] = perf_error
                return {
                    'success': False,
                    'validation_results': validation_results,
                    'error': (f'Model performance validation failed: {perf_error}'),
                }
            validation_results['model_performance'].update(
                {
                    'status': 'valid',
                    'metrics': metrics,
                }
            )

            return {
                'success': True,
                'validation_results': validation_results,
            }

        except Exception as e:
            return {
                'success': False,
                'validation_results': validation_results,
                'error': f'Validation error: {str(e)}',
            }

    def _validate_training_setup(self) -> bool:
        """Validate training directory structure."""
        # Check if required directories exist
        if not self.checkpoint_dir.exists():
            return False
        if not self.best_checkpoint_dir.exists():
            return False

        # Check if required files exist
        required_files = [
            'evaluate.py',
            'wandb_info.json',
        ]
        for file in required_files:
            if not (self.workspace_dir / file).exists():
                return False
        return True

    def _validate_checkpoints(self) -> tuple[bool, str]:
        """Validate model checkpoints."""
        try:
            # Check if best_checkpoint directory exists and is not empty
            if not self.best_checkpoint_dir.exists():
                return False, 'best_checkpoint directory not found'

            if not any(self.best_checkpoint_dir.iterdir()):
                return False, 'best_checkpoint directory is empty'

            return True, ''
        except Exception as e:
            return False, f'Error during checkpoint validation: {str(e)}'

    def _validate_wandb_logging(self) -> tuple[bool, str]:
        """Validate W&B logging info."""
        try:
            wandb_info_file = self.workspace_dir / 'wandb_info.json'
            if not wandb_info_file.exists():
                return False, 'wandb_info.json not found'

            with open(wandb_info_file) as f:
                wandb_info = json.load(f)

            required_fields = ['project_id', 'run_id']
            for field in required_fields:
                if field not in wandb_info:
                    return False, f'Missing required field: {field}'

            return True, ''
        except json.JSONDecodeError:
            return False, 'Invalid JSON format in wandb_info.json'
        except Exception as e:
            return False, f'Error validating W&B info: {str(e)}'

    async def _validate_model_performance(
        self, runtime: BaseRuntime
    ) -> tuple[bool, str, Dict[str, Any]]:
        """Run evaluation script and validate performance."""
        try:
            # Run evaluation script using MLDevBenchRuntime
            eval_script = self.workspace_dir / 'evaluate.py'
            current_hash = self._compute_file_hash(eval_script)
            if current_hash != self.evaluate_script_hash:
                return (
                    False,
                    'evaluate.py has been modified!',
                    {},
                )

            if isinstance(runtime, MLDevBenchRuntime):
                result = runtime.execute_action(
                    action=Action.ML_DEV_BENCH_SHELL_TOOL_EXEC_COMMAND,
                    request_data={
                        'cmd': f'python {eval_script}',
                        'timeout': 600,  # 10 min timeout for classification
                    },
                    metadata={},
                )
                if result['data']['exit_code'] != 0:
                    return (
                        False,
                        'Evaluation script failed with error: '
                        f'{result["data"]["stderr"]}',
                        {},
                    )

                # Check for W&B metrics error

                # Read evaluation metrics
                metrics_file = self.workspace_dir / 'eval_metrics.json'
                if not metrics_file.exists():
                    return (
                        False,
                        'No evaluation metrics found',
                        {},
                    )

                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

                # Validate accuracy threshold
                if 'accuracy' not in metrics:
                    return (
                        False,
                        'accuracy not found in evaluation metrics',
                        metrics,
                    )

                if metrics['accuracy'] < 0.88:  # 88% accuracy threshold
                    return (
                        False,
                        f'Model accuracy ({metrics["accuracy"]:.1%}) below '
                        'target threshold (88%)',
                        metrics,
                    )

                # Validate parameter count
                if 'num_parameters' not in metrics:
                    return (
                        False,
                        'num_parameters not found in evaluation metrics',
                        metrics,
                    )

                if metrics['num_parameters'] > 50_000_000:  # 50M parameter limit
                    return (
                        False,
                        'Model has too many parameters: '
                        f'{metrics["num_parameters"]:,} > 50,000,000',
                        metrics,
                    )

                return True, '', metrics

            return False, 'Invalid runtime type', {}

        except Exception as e:
            return False, f'Error during performance validation: {str(e)}', {}

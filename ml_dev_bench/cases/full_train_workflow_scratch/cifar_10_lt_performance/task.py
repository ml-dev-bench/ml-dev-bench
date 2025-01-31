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
class CIFAR10LTPerformanceTask(BaseEvaluationTask):
    task_id = 'cifar_10_lt_performance'
    description = (
        'Train a model on Long-tailed CIFAR-10 to achieve target balanced accuracy'
    )
    categories = {'ml', 'training', 'long_tailed', 'classification'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = self.workspace_dir / 'checkpoints'

    async def run(self, agent: BaseAgent) -> Dict[str, Any]:
        task_path = os.path.join(os.path.dirname(__file__), 'task.txt')
        with open(task_path, 'r') as f:
            task_prompt = f.read()
        return await agent.run(task_prompt)

    async def validate(
        self, agent_output: Dict[str, Any], runtime: BaseRuntime
    ) -> Dict[str, Any]:
        """
        Validate the agent output for the CIFAR-10 long-tailed training.
        Expected Directory Structure:
            workspace/
            ├── dataset.py
            ├── evaluate.py
            ├── utils.py
            ├── checkpoints/
            │   └── *.pt (checkpoint files)
            └── training_metrics.json
        """
        validation_results = {
            'training_setup': {'status': 'invalid'},
            'model_checkpoints': {'status': 'invalid'},
            'training_metrics': {'status': 'invalid'},
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
                    'error': f'Model checkpoint validation failed: {checkpoints_error}',
                }
            validation_results['model_checkpoints']['status'] = 'valid'

            # Validate training metrics
            metrics_valid, metrics_error = self._validate_training_metrics()
            if not metrics_valid:
                validation_results['training_metrics']['error'] = metrics_error
                return {
                    'success': False,
                    'validation_results': validation_results,
                    'error': f'Training metrics validation failed: {metrics_error}',
                }
            validation_results['training_metrics']['status'] = 'valid'

            # Run evaluation and validate performance
            (
                performance_valid,
                performance_error,
            ) = await self._validate_model_performance(runtime)
            if not performance_valid:
                validation_results['model_performance']['error'] = performance_error
                return {
                    'success': False,
                    'validation_results': validation_results,
                    'error': f'Model performance validation failed: {performance_error}',
                }
            validation_results['model_performance']['status'] = 'valid'

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
        # Check if checkpoints directory exists
        if not self.checkpoint_dir.exists():
            return False

        # Check if required files exist
        required_files = ['dataset.py', 'evaluate.py', 'utils.py']
        for file in required_files:
            if not (self.workspace_dir / file).exists():
                return False
        return True

    def _validate_checkpoints(self) -> tuple[bool, str]:
        """Validate model checkpoints."""
        try:
            # check for ckpt or pt files
            checkpoint_files = list(self.checkpoint_dir.glob('*.ckpt')) + list(
                self.checkpoint_dir.glob('*.pt')
            )
            if not checkpoint_files:
                return False, 'No checkpoint files found'

            # Load last checkpoint and verify parameters
            checkpoints = sorted(checkpoint_files)
            last_checkpoint = checkpoints[-1]
            # check size is less than 500MB
            if last_checkpoint.stat().st_size > 500 * 1024 * 1024:
                return False, 'Last checkpoint file size exceeds 500MB'

            return True, ''
        except Exception as e:
            return False, f'Error during checkpoint validation: {str(e)}'

    def _validate_training_metrics(self) -> tuple[bool, str]:
        """Validate training metrics file."""
        metrics_file = self.workspace_dir / 'training_metrics.json'

        if not metrics_file.exists():
            return False, 'training_metrics.json file not found'

        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                if not isinstance(metrics, dict):
                    return False, 'training_metrics.json is not a valid JSON object'

                if 'best_loss' not in metrics:
                    return False, 'best_loss not found in metrics'
                if 'best_balanced_acc' not in metrics:
                    return False, 'best_balanced_acc not found in metrics'

            return True, ''
        except json.JSONDecodeError:
            return False, 'Invalid JSON format in training_metrics.json'
        except ValueError:
            return False, 'best_balanced_acc could not be converted to float'
        except KeyError:
            return False, 'Required metrics keys not found'

    async def _validate_model_performance(
        self, runtime: BaseRuntime
    ) -> tuple[bool, str]:
        """Run evaluation script and validate performance."""
        try:
            # Run evaluation script using MLDevBenchRuntime
            eval_script = self.workspace_dir / 'evaluate.py'

            if isinstance(runtime, MLDevBenchRuntime):
                result = runtime.execute_action(
                    action=Action.ML_DEV_BENCH_SHELL_TOOL_EXEC_COMMAND,
                    request_data={'cmd': f'python {eval_script}'},
                    metadata={},
                )
                if result['data']['exit_code'] != 0:
                    return (
                        False,
                        f'Evaluation script failed with error: {result["data"]["stderr"]}',
                    )

                # Read and validate the results
                result_file = self.workspace_dir / 'balanced_accuracy.txt'
                if not result_file.exists():
                    return False, 'Evaluation did not produce balanced_accuracy.txt'

                with open(result_file, 'r') as f:
                    content = f.read()
                    try:
                        acc = float(content.split(':')[1].strip())
                        if acc < 75.0:
                            return (
                                False,
                                f'Model achieved {acc:.2f}% balanced accuracy, below required 75%',
                            )
                    except (IndexError, ValueError):
                        return False, 'Could not parse balanced accuracy value'

                return True, ''
            else:
                raise ValueError('Invalid runtime')

        except Exception as e:
            return False, f'Error during performance validation: {str(e)}'

    def cleanup_task(self) -> None:
        """Cleanup task resources."""
        pass

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
class SmallDatasetOverfitEvaluationTask(BaseEvaluationTask):
    task_id = 'small_dataset_overfit'
    description = 'Evaluate model overfitting on small MNIST subset'
    categories = {'ml_workflow', 'training', 'overfitting', 'mnist'}

    def initialize(self) -> None:
        """Initialize task and prepare environment."""
        super().initialize()  # Create workspace directory

        # Create training directory
        training_dir = self.workspace_dir / 'training'
        training_dir.mkdir(exist_ok=True)

    async def run(self, agent: BaseAgent) -> Dict[str, Any]:
        # Load task description from file
        task_path = os.path.join(os.path.dirname(__file__), 'task/task.txt')
        with open(task_path, 'r') as f:
            task_prompt = f.read()

        return await agent.run(task_prompt)

    async def validate(
        self, agent_output: Dict[str, Any], runtime: BaseRuntime
    ) -> Dict[str, Any]:
        try:
            # Check if the training script exists
            script_path = os.path.join(self.workspace_dir, 'training_script_mnist.py')
            if not os.path.exists(script_path):
                return {'success': False, 'error': 'Training script not found'}

            # Run the training script
            try:
                if isinstance(runtime, MLDevBenchRuntime):
                    result = runtime.execute_action(
                        action=Action.ML_DEV_BENCH_SHELL_TOOL_EXEC_COMMAND,
                        request_data={'cmd': f'python {script_path}'},
                        metadata={},
                    )
                    exec_result = {
                        'success': result['data']['exit_code'] == 0,
                        'error': result['data']['stderr']
                        if result['data']['exit_code'] != 0
                        else None,
                    }
                else:
                    raise ValueError('Invalid runtime')
                if not exec_result['success']:
                    return {
                        'success': False,
                        'error': f'Error running script: {exec_result["error"]}',
                    }

                # Check for results JSON file
                results_path = os.path.join(
                    self.workspace_dir, 'training', 'training_results.json'
                )
                if not os.path.exists(results_path):
                    return {
                        'success': False,
                        'error': 'Training results JSON file not found',
                    }

                # Load and validate results
                with open(results_path, 'r') as f:
                    results = json.load(f)

                validation_result = self._validate_results(results)
                return validation_result

            except Exception as e:
                return {'success': False, 'error': f'Error executing script: {str(e)}'}

        except Exception as e:
            return {'success': False, 'error': f'Error validating script: {str(e)}'}

    def _validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the training results from the JSON file."""
        # Check if all required fields are present
        required_fields = {
            'train_accuracy',
            'test_accuracy',
            'unique_train_labels',
            'unique_test_labels',
            'num_train_samples',
            'num_test_samples',
        }

        missing_fields = required_fields - set(results.keys())
        if missing_fields:
            return {
                'success': False,
                'error': f'Missing required fields in results: {missing_fields}',
            }

        # Validate dataset composition
        expected_labels = {0, 1, 2}
        train_labels = set(results['unique_train_labels'])
        test_labels = set(results['unique_test_labels'])

        if train_labels != expected_labels:
            return {
                'success': False,
                'error': f'Training set contains incorrect labels. Expected {expected_labels}, got {train_labels}',
            }

        if test_labels != expected_labels:
            return {
                'success': False,
                'error': f'Test set contains incorrect labels. Expected {expected_labels}, got {test_labels}',
            }

        # Check dataset sizes and distribution
        if (
            'samples_per_class_train' not in results
            or 'samples_per_class_test' not in results
        ):
            return {
                'success': False,
                'error': 'Missing samples_per_class_train or samples_per_class_test in results',
            }

        # Check for rough uniformity in class distribution
        train_samples = results['samples_per_class_train']
        test_samples = results['samples_per_class_test']

        # For each class, check if it has roughly similar number of samples
        # Allow up to 20% deviation from the mean
        train_mean = sum(train_samples.values()) / len(train_samples)
        test_mean = sum(test_samples.values()) / len(test_samples)

        for digit in expected_labels:
            if abs(train_samples[str(digit)] - train_mean) > 0.2 * train_mean:
                return {
                    'success': False,
                    'error': f'Training set not roughly uniform. Digit {digit} has {train_samples[str(digit)]} samples, mean is {train_mean:.0f}',
                }
            if abs(test_samples[str(digit)] - test_mean) > 0.2 * test_mean:
                return {
                    'success': False,
                    'error': f'Test set not roughly uniform. Digit {digit} has {test_samples[str(digit)]} samples, mean is {test_mean:.0f}',
                }

        # Check for overfitting
        train_acc = results['train_accuracy']

        if train_acc < 0.95:
            return {
                'success': False,
                'error': f'Training accuracy too low ({train_acc:.3f}), model did not achieve high accuracy',
            }

        return {
            'success': True,
            'reason': (
                f'Model successfully trained with train_acc={train_acc:.3f}, '
                f'test_acc={results["test_accuracy"]:.3f}, using only digits 0,1,2'
            ),
            'metrics': {
                'train_accuracy': train_acc,
                'test_accuracy': results['test_accuracy'],
                'train_samples': results['num_train_samples'],
                'test_samples': results['num_test_samples'],
            },
        }

import json
import os
from typing import Any, Dict

from calipers.framework.base import (
    BaseAgent,
    BaseEvaluationTask,
    BaseRuntime,
)
from calipers.framework.registry import EvalRegistry


@EvalRegistry.register_task
class MultiGPUTrainingEvaluationTask(BaseEvaluationTask):
    task_id = 'multi_gpu_training'
    description = 'Evaluate multi-GPU training implementation'
    categories = {'ml_workflow', 'training', 'multi_gpu'}

    def initialize(self) -> None:
        """Initialize task and prepare environment."""
        super().initialize()  # Create workspace directory

        # Set multiple CUDA devices to be available
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
        self.expected_devices = {'cuda:0', 'cuda:1', 'cuda:2'}

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
            script_path = os.path.join(
                self.workspace_dir, 'training', 'training_script_mnist.py'
            )
            if not os.path.exists(script_path):
                return {'success': False, 'error': 'Training script not found'}

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
            return {'success': False, 'error': f'Error validating script: {str(e)}'}

    def _validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the training results from the JSON file."""
        # Check if all required fields are present
        required_fields = {
            'Dataset',
            'Devices',
            'Loss Function',
        }

        missing_fields = required_fields - set(results.keys())
        if missing_fields:
            return {
                'success': False,
                'error': f'Missing required fields in results: {missing_fields}',
            }

        # Validate dataset is MNIST
        if results['Dataset'].lower() != 'mnist':
            return {
                'success': False,
                'error': f'Incorrect dataset. Expected MNIST, got {results["Dataset"]}',
            }

        # Validate devices list matches exactly with expected devices
        devices = set(str(device).lower() for device in results['Devices'])

        if devices != self.expected_devices:
            return {
                'success': False,
                'error': f'Incorrect devices used. Expected {self.expected_devices}, got {devices}',
            }

        # Validate loss function is specified
        if not results['Loss Function']:
            return {
                'success': False,
                'error': 'Loss function not specified',
            }

        return {
            'success': True,
            'reason': f'Script successfully implements multi-GPU training using devices {self.expected_devices}',
            'metrics': {
                'dataset': results['Dataset'],
                'devices': list(devices),
                'loss_function': results['Loss Function'],
            },
        }

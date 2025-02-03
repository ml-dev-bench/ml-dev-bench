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
class CIFAR10ImprovementTask(BaseEvaluationTask):
    task_id = 'improve_cifar10_baseline'
    description = 'Improve CIFAR10 model performance without architecture changes'
    categories = {'ml', 'pytorch', 'model_improvement'}

    TARGET_ACCURACY = 84.69

    def initialize(self) -> None:
        """No initialization needed"""
        pass

    async def run(self, agent: BaseAgent) -> Dict[str, Any]:
        task_path = os.path.join(os.path.dirname(__file__), 'task.txt')
        with open(task_path, 'r') as f:
            task_prompt = f.read()

        return await agent.run(task_prompt)

    async def validate(
        self, agent_output: Dict[str, Any], runtime: BaseRuntime
    ) -> Dict[str, Any]:
        try:
            # Path to checkpoint from improved training
            checkpoint_path = os.path.join(self.workspace_dir, 'lightning_checkpoints')

            # Find the latest checkpoint
            checkpoints = [
                f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')
            ]
            if not checkpoints:
                return {
                    'success': False,
                    'error': 'No checkpoint files found after training',
                }

            latest_checkpoint = os.path.join(checkpoint_path, sorted(checkpoints)[-1])

            # Run test_model.py to evaluate the model
            test_script = os.path.join(self.workspace_dir, 'test_model.py')

            if isinstance(runtime, MLDevBenchRuntime):
                result = runtime.execute_action(
                    action=Action.ML_DEV_BENCH_SHELL_TOOL_EXEC_COMMAND,
                    request_data={
                        'cmd': f'python {test_script} --checkpoint {latest_checkpoint}'
                    },
                    metadata={},
                )
                exit_code = result['data']['exit_code']
                stderr = result['data']['stderr']
            else:
                raise ValueError('Invalid runtime')
            if exit_code != 0:
                return {
                    'success': False,
                    'error': f'Test script failed with error: {stderr}',
                }

            # Read metrics from the generated JSON file
            metrics_path = os.path.join(self.workspace_dir, 'metrics.json')
            if not os.path.exists(metrics_path):
                return {
                    'success': False,
                    'error': 'metrics.json not found after testing',
                }

            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            val_accuracy = metrics['validation_accuracy']

            # Check if validation accuracy improved
            if val_accuracy > self.TARGET_ACCURACY + 1.0:
                return {
                    'success': True,
                    'validation_results': {
                        'improved_accuracy': val_accuracy,
                        'baseline_accuracy': self.TARGET_ACCURACY + 1.0,
                        'improvement': val_accuracy - self.TARGET_ACCURACY - 1.0,
                    },
                }
            else:
                return {
                    'success': False,
                    'validation_results': {
                        'error': f'Validation accuracy ({val_accuracy:.2f}%) did not improve over baseline ({self.TARGET_ACCURACY}%)'
                    },
                }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error validating CIFAR10 improvement task: {str(e)}',
            }

    def cleanup_task(self) -> None:
        """No cleanup needed"""
        pass

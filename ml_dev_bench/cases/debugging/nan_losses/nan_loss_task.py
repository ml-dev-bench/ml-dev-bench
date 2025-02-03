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
class NaNLossDebugTask(BaseEvaluationTask):
    task_id = 'nan_loss_debug'
    description = 'Debug and fix model training with NaN losses'
    categories = {'debugging', 'ml', 'pytorch'}

    def _setup_metrics(self) -> None:
        """Setup metrics for nan loss debug task"""
        super()._setup_metrics()

    async def run(self, agent: BaseAgent) -> Dict[str, Any]:
        task_path = os.path.join(os.path.dirname(__file__), 'task.txt')
        with open(task_path, 'r') as f:
            task_prompt = f.read()

        return await agent.run(task_prompt)

    async def validate(
        self, agent_output: Dict[str, Any], runtime: BaseRuntime
    ) -> Dict[str, Any]:
        try:
            # Run the test script to check if losses are no longer NaN
            test_script = self.workspace_dir / 'test_model.py'
            if isinstance(runtime, MLDevBenchRuntime):
                result = runtime.execute_action(
                    action=Action.ML_DEV_BENCH_SHELL_TOOL_EXEC_COMMAND,
                    request_data={'cmd': f'python {test_script}'},
                    metadata={},
                )
                exit_code = result['data']['exit_code']
                stdout = result['data']['stdout']
                if exit_code == 0 and 'Training successful' in stdout:
                    return {
                        'success': True,
                        'validation_results': {
                            'model_training': {
                                'status': 'valid',
                                'output': result['data'],
                            }
                        },
                    }
                else:
                    return {
                        'success': False,
                        'validation_results': {
                            'model_training': {
                                'status': 'invalid',
                                'error': result['data']['stderr'],
                                'output': result['data']['stdout'],
                            }
                        },
                    }
            else:
                raise ValueError('Invalid runtime')

        except Exception as e:
            return {
                'success': False,
                'error': f'Error validating nan loss debug task: {str(e)}',
            }

    def cleanup_task(self) -> None:
        """
        No cleanup needed for this task as we're only modifying existing files
        """
        pass

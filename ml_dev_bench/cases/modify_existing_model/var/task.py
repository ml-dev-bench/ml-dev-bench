import hashlib
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
class VARImplementationTask(BaseEvaluationTask):
    task_id = 'var_implementation'
    description = 'Implement Visual AutoRegressive (VAR) model methods'
    categories = {'ml', 'model-architecture', 'extend_existing_model'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_script_hash = None

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
        # Calculate and store hash of test_var.py
        test_script = self.workspace_dir / 'test_var.py'
        self.test_script_hash = self._compute_file_hash(test_script)

    async def run(self, agent: BaseAgent) -> Dict[str, Any]:
        task_path = os.path.join(os.path.dirname(__file__), 'task.txt')
        with open(task_path, 'r') as f:
            task_prompt = f.read()

        return await agent.run(task_prompt)

    async def validate(
        self, agent_output: Dict[str, Any], runtime: BaseRuntime
    ) -> Dict[str, Any]:
        validation_results = {
            'implementation': {'status': 'invalid'},
            'tests': {'status': 'invalid'},
        }

        try:
            # Verify test script hasn't been modified
            test_script = self.workspace_dir / 'test_var.py'
            assert os.path.exists(test_script), 'test_var.py does not exist'

            current_hash = self._compute_file_hash(test_script)
            if current_hash != self.test_script_hash:
                return {
                    'success': False,
                    'validation_results': validation_results,
                    'error': 'test_var.py has been modified',
                }

            # Run the test script to validate implementation
            if isinstance(runtime, MLDevBenchRuntime):
                result = runtime.execute_action(
                    action=Action.ML_DEV_BENCH_SHELL_TOOL_EXEC_COMMAND,
                    request_data={'cmd': f'python {test_script}'},
                    metadata={},
                )
                exit_code = result['data']['exit_code']
                stdout = result['data']['stdout']
                stderr = result['data']['stderr']

                validation_results['implementation']['status'] = (
                    'valid' if exit_code == 0 else 'invalid'
                )
                validation_results['tests']['status'] = (
                    'valid' if exit_code == 0 else 'invalid'
                )

                if exit_code == 0:
                    return {
                        'success': True,
                        'validation_results': validation_results,
                    }
                else:
                    return {
                        'success': False,
                        'validation_results': validation_results,
                        'error': f'Test failures: {stderr}\n{stdout}',
                    }

            else:
                raise ValueError('Invalid runtime')
        except Exception as e:
            return {
                'success': False,
                'validation_results': validation_results,
                'error': f'Error validating VAR implementation: {str(e)}',
            }

    def cleanup_task(self) -> None:
        """No cleanup needed for this task."""
        pass

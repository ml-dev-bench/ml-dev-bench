import hashlib
import os
import shutil
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
class MLAImplementTask(BaseEvaluationTask):
    task_id = 'mla_implementation_hidden_tests'
    description = 'Implement Multi-head Latent Attention (MLA) based on MHA'
    categories = {'ml', 'model-architecture', 'extend_existing_model', 'hidden_tests'}

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
        # Calculate and store hash of test_mla.py
        test_script = self.workspace_dir / 'test_mla.py'
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
            test_script = self.workspace_dir / 'test_mla.py'
            assert os.path.exists(test_script), 'test_mla.py does not exist'

            current_hash = self._compute_file_hash(test_script)
            if current_hash != self.test_script_hash:
                return {
                    'success': False,
                    'validation_results': validation_results,
                    'error': 'test_mla.py has been modified',
                }

            # copy hidden test script to workspace
            current_file_dir = os.path.dirname(__file__)
            test_script = os.path.join(current_file_dir, 'test_mla_hidden.py')
            shutil.copy(test_script, self.workspace_dir)
            test_script = os.path.join(self.workspace_dir, 'test_mla_hidden.py')

            # Run the test script to validate implementation
            if isinstance(runtime, MLDevBenchRuntime):
                result = runtime.execute_action(
                    action=Action.ML_DEV_BENCH_SHELL_TOOL_EXEC_COMMAND,
                    request_data={'cmd': f'pytest {test_script}'},
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
                'error': f'Error validating MLA implementation: {str(e)}',
            }

    def cleanup_task(self) -> None:
        """No cleanup needed for this task."""
        pass
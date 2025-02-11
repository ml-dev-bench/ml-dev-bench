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
class ViTDebugTask(BaseEvaluationTask):
    task_id = 'vit_debugging'
    description = 'Debug and fix Vision Transformer implementation'
    categories = {'debugging', 'ml', 'pytorch', 'vision', 'transformers'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_file_hash = None

    def _compute_file_hash(self, filepath) -> str:
        """Compute SHA256 hash of a file.

        Args:
            filepath: Path to the file to hash

        Returns:
            str: Hex digest of file hash
        """
        with open(filepath, 'rb') as f:
            content = f.read().replace(b'\r\n', b'\n')
            return hashlib.sha256(content).hexdigest()

    def initialize(self) -> None:
        # Calculate and store hash of test file
        test_file = self.workspace_dir / 'test_vision_transformer.py'
        self.test_file_hash = self._compute_file_hash(test_file)

    async def run(self, agent: BaseAgent) -> Dict[str, Any]:
        task_path = os.path.join(os.path.dirname(__file__), 'task.txt')
        with open(task_path, 'r') as f:
            task_prompt = f.read()

        return await agent.run(task_prompt)

    async def validate(
        self, agent_output: Dict[str, Any], runtime: BaseRuntime
    ) -> Dict[str, Any]:
        try:
            # Verify test file hasn't been modified
            test_file = self.workspace_dir / 'test_vision_transformer.py'
            current_hash = self._compute_file_hash(test_file)

            if current_hash != self.test_file_hash:
                return {
                    'success': False,
                    'error': (
                        'test_vision_transformer.py has been modified. '
                        'Only files under dino_v2/ should be changed.'
                    ),
                }

            # Run the test file to check if all tests pass
            if isinstance(runtime, MLDevBenchRuntime):
                cmd = 'python -m pytest test_vision_transformer.py -v'
                result = runtime.execute_action(
                    action=Action.ML_DEV_BENCH_SHELL_TOOL_EXEC_COMMAND,
                    request_data={'cmd': cmd},
                    metadata={},
                )
                exit_code = result['data']['exit_code']
                if exit_code == 0:
                    return {
                        'success': True,
                        'validation_results': {
                            'test_execution': {
                                'status': 'valid',
                                'output': result['data'],
                            }
                        },
                    }
                else:
                    return {
                        'success': False,
                        'validation_results': {
                            'test_execution': {
                                'status': 'invalid',
                                'error': result['data']['stderr'],
                                'output': result['data']['stdout'],
                            }
                        },
                    }

            else:
                raise ValueError('Invalid runtime')
        except Exception as e:
            err_msg = 'Error validating Vision Transformer debug task: '
            return {
                'success': False,
                'error': f'{err_msg}{str(e)}',
            }

    def cleanup_task(self) -> None:
        """
        No cleanup needed for this task as we're only modifying existing files
        """
        pass

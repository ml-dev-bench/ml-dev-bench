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
class TrainShapeDebugTask(BaseEvaluationTask):
    task_id = 'shape_mismatch_train'
    description = 'Debug and fix model training shape mismatch error'
    categories = {'debugging', 'ml', 'pytorch'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_script_hash = None

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
        # Calculate and store hash of run_training.py
        training_script = self.workspace_dir / 'run_training.py'
        self.training_script_hash = self._compute_file_hash(training_script)

    async def run(self, agent: BaseAgent) -> Dict[str, Any]:
        task_path = os.path.join(os.path.dirname(__file__), 'task.txt')
        with open(task_path, 'r') as f:
            task_prompt = f.read()

        return await agent.run(task_prompt)

    async def validate(
        self, agent_output: Dict[str, Any], runtime: BaseRuntime
    ) -> Dict[str, Any]:
        try:
            # Verify run_training.py hasn't been modified
            training_script = self.workspace_dir / 'run_training.py'
            current_hash = self._compute_file_hash(training_script)

            if current_hash != self.training_script_hash:
                return {
                    'success': False,
                    'error': 'run_training.py has been modified. Only train.py should be changed.',
                }

            # Run the training script to check if the shape error is fixed
            run_script = self.workspace_dir / 'run_training.py'
            if isinstance(runtime, MLDevBenchRuntime):
                result = runtime.execute_action(
                    action=Action.ML_DEV_BENCH_SHELL_TOOL_EXEC_COMMAND,
                    request_data={'cmd': f'python {run_script}'},
                    metadata={},
                )
                exit_code = result['data']['exit_code']
                stdout = result['data']['stdout']
                if exit_code == 0 and 'Training ran successfully' in stdout:
                    return {
                        'success': True,
                        'validation_results': {
                            'training_execution': {
                                'status': 'valid',
                                'output': result['data'],
                            }
                        },
                    }
                else:
                    return {
                        'success': False,
                        'validation_results': {
                            'training_execution': {
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
                'error': f'Error validating train shape debug task: {str(e)}',
            }

    def cleanup_task(self) -> None:
        """
        No cleanup needed for this task as we're only modifying existing files
        """
        pass

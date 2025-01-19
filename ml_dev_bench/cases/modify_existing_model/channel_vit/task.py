import os
import subprocess
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
class ChannelViTImplementTask(BaseEvaluationTask):
    task_id = 'channel_vit_implementation'
    description = 'Implement ChannelViT based on existing ViT implementation'
    categories = {'ml', 'model-architecture', 'extend_existing_model'}

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
            test_script = self.workspace_dir / 'test_channel_vit.py'
            assert os.path.exists(test_script), 'test_channel_vit.py does not exist'
            assert (
                'def test_channel_vit_sequence_lengths_and_pos_embeddings'
                in open(test_script).read()
            ), (
                'test_channel_vit.py does not contain the test_channel_vit_sequence_lengths_and_pos_embeddings function'
            )
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

                if exit_code == 0 and 'All tests passed' in stdout:
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
                result = subprocess.run(
                    ['python', str(test_script)],
                    capture_output=True,
                    text=True,
                    cwd=self.workspace_dir,
                )
                exit_code = result.returncode

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
                        'error': f'Test failures:\n{result.stderr}\n{result.stdout}',
                    }

        except Exception as e:
            return {
                'success': False,
                'validation_results': validation_results,
                'error': f'Error validating ChannelViT implementation: {str(e)}',
            }

    def cleanup_task(self) -> None:
        """No cleanup needed for this task."""
        pass

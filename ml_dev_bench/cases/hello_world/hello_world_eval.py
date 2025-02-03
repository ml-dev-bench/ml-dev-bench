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
class HelloWorldTask(BaseEvaluationTask):
    task_id = 'hello_world'
    description = 'Create a simple Hello World script'
    categories = {'basic', 'hello_world'}

    async def run(self, agent: BaseAgent) -> Dict[str, Any]:
        # Load task description from file
        task_path = os.path.join(os.path.dirname(__file__), 'task.txt')
        with open(task_path, 'r') as f:
            task_prompt = f.read()

        return await agent.run(task_prompt)

    async def validate(
        self, agent_output: Dict[str, Any], runtime: BaseRuntime
    ) -> Dict[str, Any]:
        try:
            # Check hello.py exists in workspace directory
            hello_path = os.path.join(self.workspace_dir, 'hello.py')
            if not os.path.exists(hello_path):
                return {
                    'success': False,
                    'error': 'hello.py not found in workspace directory',
                }

            # Run the script using runtime and capture output
            if isinstance(runtime, MLDevBenchRuntime):
                result = runtime.execute_action(
                    action=Action.ML_DEV_BENCH_SHELL_TOOL_EXEC_COMMAND,
                    request_data={'cmd': f'python {hello_path}'},
                    metadata={},
                )
                exit_code = result['data']['exit_code']
                stdout = result['data']['stdout']
                stderr = result['data']['stderr']
            else:
                raise ValueError('Invalid runtime')

            if exit_code != 0:
                return {
                    'success': False,
                    'error': f'Script failed with error: {stderr}',
                }

            # Check output is exactly "Hello, World!"
            output = stdout.strip()
            if output != 'Hello, World!':
                return {
                    'success': False,
                    'error': f'Expected "Hello, World!", got "{output}"',
                }

            return {
                'success': True,
                'output': output,
            }

        except Exception as e:
            return {
                'success': False,
                'error': (f'Error validating hello world: {str(e)}'),
            }

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
class BertEvalDebugTask(BaseEvaluationTask):
    task_id = 'bert_eval_debug'
    description = (
        'Debug and fix discrepancy between training and evaluation validation accuracy'
    )
    categories = {'debugging', 'ml', 'pytorch', 'transformers'}

    def _setup_metrics(self) -> None:
        """Setup metrics for BERT evaluation debug task"""
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
            # Run the evaluation script to check if accuracy matches training
            eval_script = self.workspace_dir / 'setup_workspace/evaluate.py'
            if isinstance(runtime, MLDevBenchRuntime):
                result = runtime.execute_action(
                    action=Action.ML_DEV_BENCH_SHELL_TOOL_EXEC_COMMAND,
                    request_data={'cmd': f'python {eval_script}', 'timeout': 1000},
                    metadata={},
                )
                exit_code = result['data']['exit_code']

                # Read the accuracy from the file
                accuracy_file = self.workspace_dir / 'accuracy.txt'
                with open(accuracy_file, 'r') as f:
                    accuracy_str = f.read()
                    accuracy = float(accuracy_str.split(':')[1].strip())

                # Check if accuracy matches training accuracy (within 0.5%)
                if exit_code == 0 and abs(accuracy - 72.23) <= 0.5:
                    return {
                        'success': True,
                        'validation_results': {
                            'model_evaluation': {
                                'status': 'valid',
                                'output': result['data'],
                                'accuracy': accuracy,
                            }
                        },
                    }
                else:
                    return {
                        'success': False,
                        'validation_results': {
                            'model_evaluation': {
                                'status': 'invalid',
                                'error': result['data']['stderr'],
                                'output': result['data']['stdout'],
                                'accuracy': accuracy,
                                'expected_accuracy': 72.23,
                            }
                        },
                    }
            else:
                raise ValueError('Invalid runtime')

        except Exception as e:
            error_msg = f'Error validating BERT evaluation debug task: {str(e)}'
            return {
                'success': False,
                'error': error_msg,
            }

    def cleanup_task(self) -> None:
        """
        No cleanup needed for this task as we're only modifying existing files
        """
        pass

import os
from typing import Any, Dict

from calipers.framework.base import (
    BaseAgent,
    BaseEvaluationTask,
    BaseRuntime,
)
from calipers.framework.registry import EvalRegistry

from .metrics import ErrorHandlingMetric


@EvalRegistry.register_task
class DatasetNotAvailableTask(BaseEvaluationTask):
    task_id = 'dataset_not_available_download'
    description = 'Test graceful error handling when dataset URL is incorrect'
    categories = {'ml_workflow', 'dataset', 'download', 'error_handling'}

    # Keywords that should be present in a good error message
    EXPECTED_KEYWORDS = [
        'url',
        'download',
        'error',
        'failed',
        'not found',
        'invalid',
        'incorrect',
        'unavailable',
        '404',
        'access',
    ]

    def _setup_metrics(self) -> None:
        """Setup metrics for dataset error handling task"""
        super()._setup_metrics()
        self.add_metric(ErrorHandlingMetric())

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
            # Check if dataset_info.json exists (it shouldn't)
            info_path = self.workspace_dir / 'dataset_info.json'
            if info_path.exists():
                return {
                    'success': False,
                    'error': 'Expected error handling, but dataset_info.json was created',
                }

            # Check if dataset_not_available.txt exists (it should)
            error_file_path = self.workspace_dir / 'dataset_not_available.txt'
            if not error_file_path.exists():
                return {
                    'success': False,
                    'error': 'dataset_not_available.txt file was not created',
                }

            # Read error message from file
            try:
                with open(error_file_path, 'r') as f:
                    error_message = f.read().lower()
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Failed to read dataset_not_available.txt: {str(e)}',
                }

            # Calculate error handling score based on at least keyword in error message
            handling_score = (
                1.0
                if any(kw in error_message for kw in self.EXPECTED_KEYWORDS)
                else 0.0
            )
            self.update_metric('error_handling', handling_score)

            return {
                'success': True,  # Success if file exists and we can read it
                'error_message': error_message,
                'handling_score': handling_score,
                'total_keywords': len(self.EXPECTED_KEYWORDS),
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error validating error handling: {str(e)}',
            }

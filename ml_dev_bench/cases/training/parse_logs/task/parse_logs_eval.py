import json
import os
from typing import Any, Dict

from calipers.framework.base import (
    BaseAgent,
    BaseEvaluationTask,
    BaseRuntime,
)
from calipers.framework.registry import EvalRegistry

from ..generate_logs import generate_training_logs


@EvalRegistry.register_task
class ParseLogsTask(BaseEvaluationTask):
    task_id = 'parse_logs'
    description = 'Parse and analyze training logs'
    categories = {'ml_workflow', 'training', 'logs', 'analysis'}

    # Expected values from our sample log file
    EXPECTED_FINAL_METRICS = {'accuracy': 0.930, 'loss': 0.300}
    EXPECTED_ERROR = 'CUDA out of memory'

    def _setup_metrics(self) -> None:
        """Setup metrics for log parsing task"""
        super()._setup_metrics()

    def initialize(self) -> None:
        """Initialize task and generate training logs."""
        super().initialize()  # Create workspace directory

        # Create training directory
        training_dir = self.workspace_dir / 'training'
        training_dir.mkdir(exist_ok=True)

        # Change to workspace directory temporarily to generate logs
        original_cwd = os.getcwd()
        os.chdir(self.workspace_dir)
        try:
            generate_training_logs()
        finally:
            os.chdir(original_cwd)

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
            # Check training_analysis.json exists
            analysis_path = self.workspace_dir / 'training' / 'training_analysis.json'
            if not analysis_path.exists():
                return {'success': False, 'error': 'training_analysis.json not found'}

            # Load and validate analysis results
            with open(analysis_path) as f:
                analysis = json.load(f)

            # Check required fields
            required_fields = ['status', 'final_metrics', 'error_message']
            if not all(field in analysis for field in required_fields):
                return {
                    'success': False,
                    'error': f'Missing required fields in analysis. Required: {required_fields}',
                }

            # Validate status
            if analysis['status'] != 'failed':
                return {'success': False, 'error': 'Failed to detect training failure'}

            # Validate metrics
            metrics = analysis['final_metrics']
            if (
                not isinstance(metrics, dict)
                or 'accuracy' not in metrics
                or 'loss' not in metrics
            ):
                return {'success': False, 'error': 'Invalid metrics format'}

            # Check metric values with some tolerance
            tolerance = 0.00001
            if (
                abs(metrics['accuracy'] - self.EXPECTED_FINAL_METRICS['accuracy'])
                > tolerance
                or abs(metrics['loss'] - self.EXPECTED_FINAL_METRICS['loss'])
                > tolerance
            ):
                return {'success': False, 'error': 'Incorrect final metrics values'}

            # Validate error message
            if self.EXPECTED_ERROR not in analysis['error_message']:
                return {
                    'success': False,
                    'error': 'Error message does not contain expected CUDA error',
                }

            return {'success': True, 'analysis': analysis}

        except json.JSONDecodeError:
            return {'success': False, 'error': 'Invalid JSON in training_analysis.json'}
        except Exception as e:
            return {'success': False, 'error': f'Error validating analysis: {str(e)}'}

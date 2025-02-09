import json
import os
from typing import Any, Dict

from calipers.framework.base import (
    BaseAgent,
    BaseEvaluationTask,
    BaseRuntime,
)
from calipers.framework.registry import EvalRegistry
from ml_dev_bench.cases.api_integration.wandb_logging.metrics import WandBFileMetric


@EvalRegistry.register_task
class WandBLoggingTask(BaseEvaluationTask):
    task_id = 'wandb_logging'
    description = 'Initialize WandB and log metrics'
    categories = {'wandb', 'logging', 'ml'}

    def _setup_metrics(self) -> None:
        """Setup metrics for WandB logging task"""
        super()._setup_metrics()
        self.add_metric(WandBFileMetric())

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
            correct_files = 0
            file_validation = {}

            # Check wandb_info.json
            info_path = self.workspace_dir / 'wandb_info.json'
            if not info_path.exists():
                file_validation['wandb_info.json'] = 'File not found'
            else:
                with open(info_path) as f:
                    wandb_info = json.load(f)
                if 'run_id' in wandb_info:
                    correct_files += 1
                    file_validation['wandb_info.json'] = {
                        'status': 'valid',
                        'run_id': wandb_info['run_id'],
                    }
                else:
                    file_validation['wandb_info.json'] = 'Missing run_id'

            # Check wandb_metrics.json
            metrics_path = self.workspace_dir / 'wandb_metrics.json'
            if not metrics_path.exists():
                file_validation['wandb_metrics.json'] = 'File not found'
            else:
                with open(metrics_path) as f:
                    metrics = json.load(f)

                expected_metrics = {
                    'training_loss': 0.5,
                    'validation_loss': 0.4,
                    'epoch': 0,
                }

                if all(
                    metric in metrics and metrics[metric] == value
                    for metric, value in expected_metrics.items()
                ):
                    correct_files += 1
                    file_validation['wandb_metrics.json'] = {
                        'status': 'valid',
                        'metrics': metrics,
                    }
                else:
                    file_validation['wandb_metrics.json'] = {
                        'status': 'invalid',
                        'expected': expected_metrics,
                        'actual': metrics,
                    }

            self.update_metric('wandb_files', correct_files)

            return {
                'success': correct_files == 2,
                'file_validation': file_validation,
                'correct_files': correct_files,
                'total_files': 2,
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error validating WandB files: {str(e)}',
            }

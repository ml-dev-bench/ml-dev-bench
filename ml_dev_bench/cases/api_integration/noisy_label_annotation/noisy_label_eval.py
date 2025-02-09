import json
import os
from typing import Any, Dict

from calipers.framework.base import (
    BaseAgent,
    BaseEvaluationTask,
    BaseRuntime,
)
from calipers.framework.registry import EvalRegistry
from ml_dev_bench.cases.api_integration.noisy_label_annotation.metrics import (
    LabelboxFileMetric,
)


@EvalRegistry.register_task
class NoisyLabelTask(BaseEvaluationTask):
    task_id = 'noisy_label_annotation'
    description = 'Create Labelbox tasks for high-confidence misclassifications'
    categories = {'labelbox', 'annotation', 'ml'}

    def _setup_metrics(self) -> None:
        """Setup metrics for Labelbox annotation task"""
        super()._setup_metrics()
        self.add_metric(LabelboxFileMetric())

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

            # Check labelbox_run_id.json
            info_path = self.workspace_dir / 'labelbox_run_id.json'
            if not info_path.exists():
                file_validation['labelbox_run_id.json'] = 'File not found'
            else:
                with open(info_path) as f:
                    labelbox_info = json.load(f)
                required_fields = [
                    'project_id',
                    'image_ids',
                    'confidence_scores',
                    'predicted_labels',
                    'true_labels',
                    'task_id',
                ]

                if all(field in labelbox_info for field in required_fields):
                    if len(labelbox_info['image_ids']) == 5:  # Verify 5 images
                        correct_files += 1
                        file_validation['labelbox_run_id.json'] = {
                            'status': 'valid',
                            'project_id': labelbox_info['project_id'],
                        }
                    else:
                        file_validation['labelbox_run_id.json'] = (
                            'Incorrect number of images'
                        )
                else:
                    file_validation['labelbox_run_id.json'] = 'Missing required fields'

            # Check labelbox_metrics.json
            metrics_path = self.workspace_dir / 'labelbox_metrics.json'
            if not metrics_path.exists():
                file_validation['labelbox_metrics.json'] = 'File not found'
            else:
                with open(metrics_path) as f:
                    metrics = json.load(f)

                required_metrics = {'num_images_uploaded': 5, 'task_status': 'created'}

                if (
                    all(metric in metrics for metric in required_metrics)
                    and 'avg_confidence' in metrics
                ):
                    correct_files += 1
                    file_validation['labelbox_metrics.json'] = {
                        'status': 'valid',
                        'metrics': metrics,
                    }
                else:
                    file_validation['labelbox_metrics.json'] = {
                        'status': 'invalid',
                        'expected_fields': list(required_metrics.keys())
                        + ['avg_confidence'],
                        'actual': list(metrics.keys()),
                    }

            self.update_metric('labelbox_files', correct_files)

            return {
                'success': correct_files == 2,
                'file_validation': file_validation,
                'correct_files': correct_files,
                'total_files': 2,
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error validating Labelbox files: {str(e)}',
            }

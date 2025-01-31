import json
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
class VOCSegmentationTask(BaseEvaluationTask):
    task_id = 'improve_segmentation_baseline'
    description = 'Achieve high performance on Pascal VOC segmentation using FCN ResNet50'
    categories = {'ml', 'pytorch', 'semantic_segmentation', 'pascal_voc'}
    
    # Validation thresholds and requirements
    MIN_MIOU = 0.37  
    REQUIRED_YEAR = "2012"
    REQUIRED_IMAGE_SET = "val"
    NUM_CLASSES = 21  # 20 classes + background

    def initialize(self) -> None:
        """No initialization needed"""
        pass

    async def run(self, agent: BaseAgent) -> Dict[str, Any]:
        task_path = os.path.join(os.path.dirname(__file__), 'task.txt')
        with open(task_path, 'r') as f:
            task_prompt = f.read()

        return await agent.run(task_prompt)

    async def validate(
        self, agent_output: Dict[str, Any], runtime: BaseRuntime
    ) -> Dict[str, Any]:
        try:

            # Read output information if it exists
            output_path = os.path.join(self.workspace_dir, 'output_info.json')
            if not os.path.exists(output_path):
                return {
                    'success': False,
                    'error': 'output_info.json not found',
                }

            with open(output_path, 'r') as f:
                output_info = json.load(f)

            # Validate required fields
            required_fields = [
                'model_name', 'mean_iou', 'training_time', 'dataset_info'
            ]
            missing_fields = [f for f in required_fields if f not in output_info]
            if missing_fields:
                return {
                    'success': False,
                    'error': f'Missing required fields in output_info.json: {missing_fields}',
                }

            # Validate dataset info
            dataset_info = output_info.get('dataset_info', {})
            required_dataset_fields = ['year', 'image_set', 'num_classes']
            missing_dataset_fields = [f for f in required_dataset_fields if f not in dataset_info]
            if missing_dataset_fields:
                return {
                    'success': False,
                    'error': f'Missing dataset fields in output_info.json: {missing_dataset_fields}',
                }

            # Verify dataset requirements
            if dataset_info['year'] != self.REQUIRED_YEAR:
                return {
                    'success': False,
                    'error': f'Must use VOC {self.REQUIRED_YEAR}, got {dataset_info["year"]}',
                }

            if dataset_info['image_set'] != self.REQUIRED_IMAGE_SET:
                return {
                    'success': False,
                    'error': f'Must evaluate on {self.REQUIRED_IMAGE_SET} set, got {dataset_info["image_set"]}',
                }

            if dataset_info['num_classes'] != self.NUM_CLASSES:
                return {
                    'success': False,
                    'error': f'Expected {self.NUM_CLASSES} classes, got {dataset_info["num_classes"]}',
                }

            # Verify correct model is used
            expected_model = 'torchvision.models.segmentation.fcn_resnet50'
            if output_info['model_name'] != expected_model:
                return {
                    'success': False,
                    'error': f'Must use {expected_model}, got {output_info["model_name"]}',
                }

            mean_iou = output_info['mean_iou']

            # Check if mean IoU meets threshold
            if mean_iou >= self.MIN_MIOU:
                return {
                    'success': True,
                    'validation_results': {
                        'mean_iou': mean_iou,
                        'training_time': output_info['training_time'],
                        'model_name': output_info['model_name'],
                        'dataset_info': dataset_info
                    },
                }
            else:
                return {
                    'success': False,
                    'validation_results': {
                        'error': f'Mean IoU ({mean_iou:.2%}) did not meet minimum threshold ({self.MIN_MIOU:.2%})'
                    },
                }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error validating VOC segmentation task: {str(e)}',
            }

    def cleanup_task(self) -> None:
        """No cleanup needed"""
        pass

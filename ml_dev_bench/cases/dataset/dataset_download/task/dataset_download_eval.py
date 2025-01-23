import json
import os
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from calipers.framework.base import (
    BaseAgent,
    BaseEvaluationTask,
    BaseRuntime,
)
from calipers.framework.registry import EvalRegistry


@EvalRegistry.register_task
class DatasetDownloadTask(BaseEvaluationTask):
    task_id = 'dataset_download'
    description = 'Download and verify an image dataset'
    categories = {'ml_workflow', 'dataset', 'download', 'images'}

    IMAGENETTE_CLASSES = {
        'n01440764': 'tench',
        'n02102040': 'English springer',
        'n02979186': 'cassette player',
        'n03000684': 'chain saw',
        'n03028079': 'church',
        'n03394916': 'French horn',
        'n03417042': 'garbage truck',
        'n03425413': 'gas pump',
        'n03445777': 'golf ball',
        'n03888257': 'parachute',
    }

    def _setup_metrics(self) -> None:
        """Setup metrics for dataset download task"""
        super()._setup_metrics()

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
            # Check dataset_info.json exists
            info_path = self.workspace_dir / 'dataset_info.json'
            if not info_path.exists():
                return {
                    'success': False,
                    'error': 'dataset_info.json not found',
                }

            # Load dataset path
            with open(info_path) as f:
                dataset_info = json.load(f)

            if 'dataset_path' not in dataset_info:
                return {
                    'success': False,
                    'error': 'dataset_path not found in dataset_info.json',
                }

            dataset_path = dataset_info['dataset_path']
            if not os.path.exists(dataset_path):
                return {
                    'success': False,
                    'error': f'Dataset path {dataset_path} does not exist',
                }

            # Verify dataset structure and contents
            train_dir = os.path.join(dataset_path, 'train')
            if not os.path.exists(train_dir):
                return {
                    'success': False,
                    'error': 'Training directory not found in dataset',
                }

            # Check all classes exist and validate
            correct_classes = 0
            class_validation = {}
            total_classes = len(self.IMAGENETTE_CLASSES)

            for class_id, _ in self.IMAGENETTE_CLASSES.items():
                class_dir = os.path.join(train_dir, class_id)
                if not os.path.exists(class_dir):
                    class_validation[class_id] = {
                        'status': 'missing',
                        'error': 'Class directory not found',
                    }
                    continue

                # Check at least one image in each class and verify size
                image_files = list(Path(class_dir).glob('*.JPEG'))
                if not image_files:
                    class_validation[class_id] = {
                        'status': 'invalid',
                        'error': 'No images found',
                    }
                    continue

                # Check first image dimensions
                with Image.open(image_files[0]) as img:
                    width, height = img.size
                    if width != 160 and height != 160:
                        class_validation[class_id] = {
                            'status': 'invalid',
                            'error': f'Wrong image size: {width}x{height}',
                        }
                        continue

                correct_classes += 1
                class_validation[class_id] = {
                    'status': 'valid',
                    'images_found': len(image_files),
                }

            return {
                'success': correct_classes == total_classes,
                'class_validation': class_validation,
                'correct_classes': correct_classes,
                'total_classes': total_classes,
                'dataset_path': dataset_path,
            }

        except Exception as e:
            return {'success': False, 'error': f'Error validating dataset: {str(e)}'}

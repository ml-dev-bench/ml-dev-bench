import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from PIL import Image

from calipers.framework.base import (
    BaseAgent,
    BaseEvaluationTask,
    BaseRuntime,
)
from calipers.framework.registry import EvalRegistry


@EvalRegistry.register_task
class NoisyDatasetDownloadTask(BaseEvaluationTask):
    task_id = 'noisy_dataset_download'
    description = 'Download and verify an image dataset'
    categories = {'ml_workflow', 'dataset_download', 'images'}

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

    def validate_noisy_labels_csv(self) -> Tuple[bool, str]:
        noisy_labels_file = self.workspace_dir / 'all_noisy_labels_50.csv'
        if not noisy_labels_file.exists():
            return False, 'all_noisy_labels_50.csv file not found'

        # Load the CSV and check for specific path-label pairs
        try:
            import pandas as pd

            df = pd.read_csv(noisy_labels_file)
            if not {'path', 'label'}.issubset(df.columns):
                return False, 'CSV does not contain required columns: path, label'

            # Check a random set of path-label pairs
            path_label_checks = {
                'train/n02979186/n02979186_11957.JPEG': 'n03000684',
                'train/n02979186/n02979186_10756.JPEG': 'n03445777',
                'train/n03445777/n03445777_13093.JPEG': 'n03000684',
                'train/n03028079/n03028079_6923.JPEG': 'n02102040',
            }

            for path, expected_label in path_label_checks.items():
                actual_label = df.loc[df['path'] == path, 'label'].values
                if not actual_label or actual_label[0] != expected_label:
                    return (
                        False,
                        f'Label mismatch for {path}: expected {expected_label}, found {actual_label[0] if actual_label else "None"}',
                    )
            return True, 'Noisy labels csv is valid'
        except Exception as e:
            return False, f'Error processing CSV: {str(e)}'

    def validate_class_counts(
        self, file_name: str, class_counts: Dict[str, int]
    ) -> Tuple[bool, str]:
        class_counts_file = self.workspace_dir / file_name
        if not class_counts_file.exists():
            return False, f'{file_name} file not found'

        # Load the CSV and check for specific path-label pairs
        try:
            import pandas as pd

            df = pd.read_csv(class_counts_file)
            if not {'class', 'count'}.issubset(df.columns):
                return False, 'CSV does not contain required columns: class, count'

            # Check if all classes are present in the CSV
            if not set(class_counts.keys()).issubset(set(df['class'].unique())):
                return False, 'CSV does not contain all required classes'

            for class_id, expected_count in class_counts.items():
                actual_count = df.loc[df['class'] == class_id, 'count'].values
                if not actual_count or actual_count[0] != expected_count:
                    return (
                        False,
                        f'Class count mismatch for {class_id}: expected {expected_count}, found {actual_count[0] if actual_count else "None"}',
                    )
            return True, 'Class counts are valid'
        except Exception as e:
            return False, f'Error processing CSV: {str(e)}'

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
            # if dataset_path is relative, convert to absolute path by joining with workspace_dir
            if not os.path.isabs(dataset_path):
                dataset_path = os.path.join(self.workspace_dir, dataset_path)

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

            # validate noisy labels csv
            noisy_labels_csv_valid, noisy_labels_csv_error = (
                self.validate_noisy_labels_csv()
            )
            if not noisy_labels_csv_valid:
                return {
                    'success': False,
                    'error': f'Noisy labels csv validation failed: {noisy_labels_csv_error}',
                }

            # validate train and validation class counts
            train_class_counts_valid, train_class_counts_error = (
                self.validate_class_counts(
                    'train_class_counts.csv',
                    {
                        'n01440764': 945,
                        'n02102040': 986,
                        'n02979186': 942,
                        'n03000684': 880,
                        'n03028079': 954,
                        'n03394916': 996,
                        'n03417042': 948,
                        'n03425413': 950,
                        'n03445777': 941,
                        'n03888257': 927,
                    },
                )
            )
            if not train_class_counts_valid:
                return {
                    'success': False,
                    'error': f'Train class counts validation failed: {train_class_counts_error}',
                }

            validation_class_counts_valid, validation_class_counts_error = (
                self.validate_class_counts(
                    'validation_class_counts.csv',
                    {
                        'n01440764': 387,
                        'n02102040': 395,
                        'n02979186': 357,
                        'n03000684': 386,
                        'n03028079': 409,
                        'n03394916': 394,
                        'n03417042': 389,
                        'n03425413': 419,
                        'n03445777': 399,
                        'n03888257': 390,
                    },
                )
            )
            if not validation_class_counts_valid:
                return {
                    'success': False,
                    'error': f'Validation class counts validation failed: {validation_class_counts_error}',
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

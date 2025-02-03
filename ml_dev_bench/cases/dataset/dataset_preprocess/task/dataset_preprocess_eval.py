import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from calipers.framework.base import BaseAgent, BaseEvaluationTask, BaseRuntime
from calipers.framework.config import TaskConfig
from calipers.framework.registry import EvalRegistry

from .metrics import (
    AugmentationVarianceMetric,
    PreprocessingShapeMetric,
)


@EvalRegistry.register_task
class DatasetPreprocessTask(BaseEvaluationTask):
    task_id = 'dataset_preprocess'
    description = 'Load and preprocess CIFAR10 dataset'
    categories = {'ml_workflow', 'dataset', 'preprocessing', 'images'}

    # Default parameters (can be overridden by config)
    EXPECTED_SHAPE = (3, 32, 32)  # CIFAR10 shape
    NUM_SAMPLES_TO_CHECK = 10
    VALUE_RANGE = (-1, 1)  # Expected range after normalization
    VARIANCE_THRESHOLD = 0.01  # Minimum pixel-wise variance for augmentation check
    AUGMENTATION_SUCCESS_THRESHOLD = 1  # Required success rate for augmentations

    def __init__(self, config: TaskConfig):
        super().__init__(config)

        # Override defaults with config values if provided
        if 'preprocessing' in config.config:
            preproc_config = config.config['preprocessing']
            if 'expected_shape' in preproc_config:
                self.EXPECTED_SHAPE = tuple(preproc_config['expected_shape'])
            if 'value_range' in preproc_config:
                self.VALUE_RANGE = tuple(preproc_config['value_range'])

        if 'augmentation' in config.config:
            aug_config = config.config['augmentation']
            if 'num_samples' in aug_config:
                self.AUGMENTATION_SAMPLES = aug_config['num_samples']
            if 'variance_threshold' in aug_config:
                self.VARIANCE_THRESHOLD = aug_config['variance_threshold']
            if 'success_threshold' in aug_config:
                self.AUGMENTATION_SUCCESS_THRESHOLD = aug_config['success_threshold']

    def _setup_metrics(self) -> None:
        """Setup metrics for dataset preprocessing task"""
        super()._setup_metrics()
        self.add_metric(PreprocessingShapeMetric())
        self.add_metric(AugmentationVarianceMetric())

    async def run(self, agent: BaseAgent) -> Dict[str, Any]:
        # Load task description from file
        task_path = os.path.join(os.path.dirname(__file__), 'task.txt')
        with open(task_path, 'r') as f:
            task_prompt = f.read()

        return await agent.run(task_prompt)

    def _check_augmentation_variance(self, samples: List[np.ndarray]) -> bool:
        """Check if augmented samples show sufficient variation.

        Args:
            samples: List of augmented versions of the same image

        Returns:
            bool: True if samples show sufficient variation
        """
        # Stack samples and compute variance across augmented versions
        stacked = np.stack(samples)
        pixel_variance = np.var(stacked, axis=0).mean()
        return pixel_variance >= self.VARIANCE_THRESHOLD

    def _validate_preprocessing(
        self, dataset_info: Dict[str, Any]
    ) -> Tuple[Dict, int, int, int]:
        """Validate the preprocessing by checking shapes, value ranges, and augmentations.

        Args:
            dataset_info: Dictionary containing dataset and preprocessing information

        Returns:
            Tuple containing:
            - Dictionary with validation details
            - Number of samples with correct shape
            - Number of samples with correct value range
            - Number of samples with sufficient augmentation variance
        """
        try:
            # Get paths from dataset_info
            if 'preprocessed_data_path' not in dataset_info:
                raise ValueError('preprocessed_data_path not found in dataset_info')
            if 'augmented_data_path' not in dataset_info:
                raise ValueError('augmented_data_path not found in dataset_info')

            data_path = Path(
                self.workspace_dir / dataset_info['preprocessed_data_path']
            )
            aug_path = Path(self.workspace_dir / dataset_info['augmented_data_path'])

            if not data_path.exists() or not aug_path.exists():
                raise ValueError('Data paths do not exist')

            # Check if we have enough preprocessed samples
            data_files = list(data_path.glob('*.npy'))
            if len(data_files) != self.NUM_SAMPLES_TO_CHECK:
                raise ValueError(
                    f'Not enough preprocessed samples. Found {len(data_files)} files, '
                    f'expected {self.NUM_SAMPLES_TO_CHECK}'
                )

            # Check augmented files
            aug_files = list(aug_path.glob('*.npy'))
            if len(aug_files) % self.NUM_SAMPLES_TO_CHECK != 0:
                raise ValueError(
                    f'Number of augmented files ({len(aug_files)}) is not a multiple of '
                    f'the number of samples to check ({self.NUM_SAMPLES_TO_CHECK}). '
                    'Each sample should have the same number of augmented versions.'
                )

            # Get sorted lists of files
            data_files = sorted(data_files)
            aug_files = sorted(aug_files)

            # Load and validate samples
            correct_shapes = 0
            correct_ranges = 0
            correct_augmentations = 0
            samples_checked = 0
            validation_details = {'samples': []}

            for data_file in data_files:
                # Load preprocessed sample
                sample = np.load(data_file)

                # Check shape
                shape_correct = sample.shape == self.EXPECTED_SHAPE
                if shape_correct:
                    correct_shapes += 1

                # Check value range
                min_val, max_val = sample.min(), sample.max()
                range_correct = (
                    self.VALUE_RANGE[0] <= min_val <= self.VALUE_RANGE[1]
                    and self.VALUE_RANGE[0] <= max_val <= self.VALUE_RANGE[1]
                )
                if range_correct:
                    correct_ranges += 1

                # Check augmentation variance
                aug_samples = []
                # Get all augmented versions for this sample
                aug_pattern = f'{data_file.stem}_v*.npy'
                aug_versions = list(aug_path.glob(aug_pattern))

                # Load base augmented file (without _v suffix)
                base_aug_file = aug_path / f'{data_file.stem}.npy'
                if base_aug_file.exists():
                    aug_samples.append(np.load(base_aug_file))

                # Load additional augmented versions
                for aug_version in aug_versions:
                    aug_samples.append(np.load(aug_version))

                if len(aug_samples) != 3:
                    raise ValueError(
                        f'Num augementations should be three found {len(aug_samples)}'
                    )
                aug_variance_sufficient = False
                aug_variance_sufficient = self._check_augmentation_variance(aug_samples)
                if aug_variance_sufficient:
                    correct_augmentations += 1

                validation_details['samples'].append(
                    {
                        'shape': tuple(sample.shape),
                        'shape_correct': shape_correct,
                        'min_value': float(min_val),
                        'max_value': float(max_val),
                        'range_correct': range_correct,
                        'augmentation_variance': aug_variance_sufficient,
                    }
                )

                samples_checked += 1

            return (
                validation_details,
                correct_shapes,
                correct_ranges,
                correct_augmentations,
            )

        except Exception as e:
            raise RuntimeError(f'Error validating preprocessing: {str(e)}')

    async def validate(
        self,
        agent_output: Dict[str, Any],
        runtime: BaseRuntime,
    ) -> Dict[str, Any]:
        validation_details: Dict[str, Any] = {}
        try:
            # Check preprocessing_info.json exists
            info_path = self.workspace_dir / 'preprocessing_info.json'
            if not info_path.exists():
                return {
                    'success': False,
                    'error': 'preprocessing_info.json not found',
                }

            # Load preprocessing info
            with open(info_path) as f:
                preprocessing_info = json.load(f)

            # Validate preprocessing
            (
                validation_details,
                correct_shapes,
                correct_ranges,
                correct_augmentations,
            ) = self._validate_preprocessing(preprocessing_info)

            # Update metrics
            self.update_metric(
                'preprocessing_shape', (correct_shapes, self.NUM_SAMPLES_TO_CHECK)
            )
            self.update_metric(
                'preprocessing_range', (correct_ranges, self.NUM_SAMPLES_TO_CHECK)
            )
            self.update_metric(
                'augmentation_variance',
                (correct_augmentations, self.NUM_SAMPLES_TO_CHECK),
            )

            # Calculate success based on all metrics
            shape_success_rate = correct_shapes / self.NUM_SAMPLES_TO_CHECK
            range_success_rate = correct_ranges / self.NUM_SAMPLES_TO_CHECK
            aug_success_rate = correct_augmentations / self.NUM_SAMPLES_TO_CHECK
            success = (
                shape_success_rate == 1.0  # Must be 100% for shape
                and range_success_rate == 1.0  # Must be 100% for range
                and aug_success_rate >= self.AUGMENTATION_SUCCESS_THRESHOLD
            )

            return {
                'success': success,
                'validation_details': validation_details,
                'correct_shapes': correct_shapes,
                'correct_ranges': correct_ranges,
                'correct_augmentations': correct_augmentations,
                'total_samples': self.NUM_SAMPLES_TO_CHECK,
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error validating preprocessing: {str(e)}',
            }

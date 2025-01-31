import json
import os
from typing import Any, Dict

from calipers.framework.base import (
    BaseAgent,
    BaseEvaluationTask,
    BaseRuntime,
)
from calipers.framework.registry import EvalRegistry


@EvalRegistry.register_task
class FullTrainWorkflowPerformanceTestTask(BaseEvaluationTask):
    task_id = 'full_train_workflow_performance_test'
    description = 'Run a training pipeline on noisy imagenette dataset to achieve a target performance'
    categories = {'ml', 'training', 'workflow', 'performance_test'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_dir = self.workspace_dir / 'dataset'
        self.checkpoint_dir = self.workspace_dir / 'checkpoints'

    async def run(self, agent: BaseAgent) -> Dict[str, Any]:
        task_path = os.path.join(os.path.dirname(__file__), 'task.txt')
        with open(task_path, 'r') as f:
            task_prompt = f.read()
        return await agent.run(task_prompt)

    async def validate(
        self, agent_output: Dict[str, Any], runtime: BaseRuntime
    ) -> Dict[str, Any]:
        """
        Validate the agent output for the full training workflow.
        Expected Directory Structure:
            workspace/
            ├── dataset/
            │   ├── train/
            │   └── val/
            ├── checkpoints/
            │   └── *.pt (checkpoint files)
            └── training_metrics.json

        """
        validation_results = {
            'dataset_setup': {'status': 'invalid'},
            'training_setup': {'status': 'invalid'},
            'model_checkpoints': {'status': 'invalid'},
            'training_metrics': {'status': 'invalid'},
        }

        try:
            # Check dataset download and structure
            dataset_valid, dataset_error = self._validate_dataset_setup()
            if not dataset_valid:
                validation_results['dataset_setup']['error'] = dataset_error
                return {
                    'success': False,
                    'validation_results': validation_results,
                    'error': f'Dataset setup validation failed: {dataset_error}',
                }
            validation_results['dataset_setup']['status'] = 'valid'

            # Check training directory structure
            if not self._validate_training_setup():
                return {
                    'success': False,
                    'validation_results': validation_results,
                    'error': 'Training setup validation failed',
                }
            validation_results['training_setup']['status'] = 'valid'

            # Validate model checkpoints
            checkpoints_valid, checkpoints_error = self._validate_checkpoints()
            if not checkpoints_valid:
                validation_results['model_checkpoints']['error'] = checkpoints_error
                return {
                    'success': False,
                    'validation_results': validation_results,
                    'error': f'Model checkpoint validation failed: {checkpoints_error}',
                }
            validation_results['model_checkpoints']['status'] = 'valid'

            # Validate training metrics
            success, error_msg = self._validate_training_metrics()
            if not success:
                validation_results['training_metrics']['error'] = error_msg
                return {
                    'success': False,
                    'validation_results': validation_results,
                    'error': f'Training metrics validation failed: {error_msg}',
                }
            validation_results['training_metrics']['status'] = 'valid'

            return {
                'success': True,
                'validation_results': validation_results,
            }

        except Exception as e:
            return {
                'success': False,
                'validation_results': validation_results,
                'error': f'Validation error: {str(e)}',
            }

    def _validate_dataset_setup(self) -> tuple[bool, str]:
        """Validate dataset download and structure."""
        # Check if dataset directory exists
        if not self.dataset_dir.exists():
            return False, 'Dataset directory does not exist'

        # Check for noisy_labels_50.csv file
        noisy_labels_file = self.dataset_dir / 'noisy_labels_50.csv'
        if not noisy_labels_file.exists():
            return False, 'noisy_labels_50.csv file not found'

        # Load the CSV and check for specific path-label pairs
        try:
            import pandas as pd

            df = pd.read_csv(noisy_labels_file)
            if not {'path', 'label'}.issubset(df.columns):
                return False, 'CSV does not contain required columns: path, label'

            # Check specific path-label pairs
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

        except Exception as e:
            return False, f'Error processing CSV: {str(e)}'

        # Check train and val directories
        train_dir = self.dataset_dir / 'train'
        val_dir = self.dataset_dir / 'val'

        if not (train_dir.exists() and val_dir.exists()):
            return False, 'Train or validation directory does not exist'

        # Check if directories are non-empty
        if not (any(train_dir.iterdir()) and any(val_dir.iterdir())):
            return False, 'Train or validation directory is empty'

        return True, ''

    def _validate_training_setup(self) -> bool:
        """Validate training directory structure."""
        # Check if checkpoints directory exists
        if not self.checkpoint_dir.exists():
            return False
        return True

    def _validate_checkpoints(self) -> tuple[bool, str]:
        """Validate model checkpoints."""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob('*.pt'))
            if len(checkpoint_files) <= 20:
                return (
                    False,
                    f'Expected at least 20 checkpoint files, found {len(checkpoint_files)}',
                )

            # Load last checkpoint and verify parameters
            checkpoints = sorted(checkpoint_files)
            last_checkpoint = checkpoints[-1]

            try:
                import torch

                # Try loading with torch first
                model_state = torch.load(last_checkpoint, map_location='cpu')
                total_params = 0
                for param in model_state['model_state_dict'].values():
                    total_params += param.numel()
                if total_params > 30_000_000:  # 30M parameters
                    return False, 'Total parameters in model exceed 30M'

            except ImportError:
                # If torch not available, check file size as a rough proxy
                # Most >22M param models will be >80MB
                min_checkpoint_size = 100 * 1024 * 1024  # 100MB in bytes
                if last_checkpoint.stat().st_size > min_checkpoint_size:
                    return False, 'Last checkpoint file size exceeds 100MB'
                return True, ''

            return True, ''
        except Exception as e:
            return False, f'Error during checkpoint validation: {str(e)}'

    def _validate_training_metrics(self) -> tuple[bool, str]:
        """Validate training metrics file.

        Checks:
        1. training_metrics.json exists
        2. File contains valid JSON
        3. Contains best_val_acc key
        4. best_val_acc is > 20%

        Returns:
            tuple[bool, str]: (success, error_message)
        """
        metrics_file = self.workspace_dir / 'training_metrics.json'

        if not metrics_file.exists():
            return False, 'training_metrics.json file not found'

        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                # Basic validation that it's a valid JSON
                if not isinstance(metrics, dict):
                    return False, 'training_metrics.json is not a valid JSON object'

                # Check for best_val_acc
                if 'best_val_acc' not in metrics:
                    return False, 'best_val_acc not found in metrics'

                # Validate accuracy value
                best_val_acc = float(metrics['best_val_acc'])
                if best_val_acc > 80:  # 20%
                    return (
                        False,
                        f'best_val_acc ({best_val_acc:.2%}) is below required 80%',
                    )

            return True, ''
        except json.JSONDecodeError:
            return False, 'Invalid JSON format in training_metrics.json'
        except ValueError:
            return False, 'best_val_acc could not be converted to float'
        except KeyError:
            return False, 'Required metrics keys not found'

    def cleanup_task(self) -> None:
        """Cleanup task resources."""
        pass

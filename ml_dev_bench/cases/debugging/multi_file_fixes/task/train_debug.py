import json
import os
import shutil
from typing import Any, Dict

from calipers.framework.base import (
    BaseAgent,
    BaseEvaluationTask,
    BaseRuntime,
)
from calipers.framework.registry import EvalRegistry


@EvalRegistry.register_task
class TrainingDebugTask(BaseEvaluationTask):
    task_id = 'training_files_debug'
    description = 'Debug and fix CIFAR-10 training pipeline'
    categories = {'debugging', 'training', 'ml'}

    def _setup_metrics(self) -> None:
        """Setup metrics for training debug task"""
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
            correct_items = 0
            validation_results = {}

            # Check wandb_info.json
            info_path = self.workspace_dir / 'wandb_info.json'
            if not info_path.exists():
                validation_results['wandb_info.json'] = {
                    'status': 'invalid',
                    'error': 'File not found',
                }
            else:
                with open(info_path) as f:
                    wandb_info = json.load(f)
                if all(
                    key in wandb_info for key in ['project_id', 'run_id', 'run_url']
                ):
                    if wandb_info['project_id'] == 'cifar10-training':
                        correct_items += 1
                        validation_results['wandb_info.json'] = {
                            'status': 'valid',
                            'info': wandb_info,
                        }
                    else:
                        validation_results['wandb_info.json'] = {
                            'status': 'invalid',
                            'error': 'Invalid project_id',
                            'expected': 'cifar10-training',
                            'actual': wandb_info['project_id'],
                        }
                else:
                    validation_results['wandb_info.json'] = {
                        'status': 'invalid',
                        'error': 'Missing required fields',
                        'required': ['project_id', 'run_id', 'run_url'],
                        'found': list(wandb_info.keys()),
                    }

            # Check data directory and CIFAR-10 files
            data_dir = self.workspace_dir / 'data'
            if not data_dir.exists():
                validation_results['data'] = {
                    'status': 'invalid',
                    'error': 'Data directory not found',
                }
            else:
                tarfile_path = data_dir / 'cifar-10-python.tar.gz'
                extracted_dir = data_dir / 'cifar-10-batches-py'

                if tarfile_path.exists() and extracted_dir.is_dir():
                    correct_items += 1
                    validation_results['data'] = {
                        'status': 'valid',
                        'files_present': [
                            'cifar-10-python.tar.gz',
                            'cifar-10-batches-py/',
                        ],
                    }
                else:
                    missing_files = []
                    if not tarfile_path.exists():
                        missing_files.append('cifar-10-python.tar.gz')
                    if not extracted_dir.is_dir():
                        missing_files.append('cifar-10-batches-py/')

                    validation_results['data'] = {
                        'status': 'invalid',
                        'error': 'Missing CIFAR-10 files',
                        'missing_files': missing_files,
                    }

            # Check checkpoints directory and files
            ckpt_dir = self.workspace_dir / 'checkpoints'
            if not ckpt_dir.exists():
                validation_results['checkpoints'] = {
                    'status': 'invalid',
                    'error': 'Checkpoints directory not found',
                }
            else:
                if not any(ckpt_dir.iterdir()):
                    validation_results['checkpoints'] = {
                        'status': 'invalid',
                        'error': 'Checkpoints directory is empty',
                    }
                else:
                    checkpoint_files = list(
                        ckpt_dir.glob('cifar10-epoch=01-val_loss*.ckpt')
                    )
                    if checkpoint_files:
                        correct_items += 1
                        validation_results['checkpoints'] = {
                            'status': 'valid',
                            'checkpoint_found': checkpoint_files[0].name,
                        }
                    else:
                        validation_results['checkpoints'] = {
                            'status': 'invalid',
                            'error': 'Missing required checkpoint file',
                            'pattern': 'cifar10-epoch=01-val_loss*.ckpt',
                            'found_files': [f.name for f in ckpt_dir.iterdir()],
                        }

            return {
                'success': correct_items == 3,
                'validation_results': validation_results,
                'correct_items': correct_items,
                'total_items': 3,
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error validating training debug files: {str(e)}',
            }

    def cleanup_task(self) -> None:
        """
        Cleans up only the new created files in the workspace
        """
        CREATED_DIRS = ['checkpoints', 'data', 'wandb']
        CREATED_FILES = [
            'wandb_info.json',
        ]

        for dir in CREATED_DIRS:
            dir_path = self.workspace_dir / dir
            if dir_path.exists():
                shutil.rmtree(dir_path)

        for file in CREATED_FILES:
            file_path = self.workspace_dir / file
            if file_path.exists():
                file_path.unlink()

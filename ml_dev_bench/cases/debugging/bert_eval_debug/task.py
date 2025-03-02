import hashlib
import json
import os
from typing import Any, Dict

import requests
from composio import Action

from calipers.framework.base import (
    BaseAgent,
    BaseEvaluationTask,
    BaseRuntime,
)
from calipers.framework.registry import EvalRegistry
from calipers.runtime.ml_dev_bench_runtime import MLDevBenchRuntime


@EvalRegistry.register_task
class BertEvalDebugTask(BaseEvaluationTask):
    task_id = 'bert_eval_debug'
    description = (
        'Debug and fix discrepancy between training and evaluation validation accuracy'
    )
    categories = {'debugging', 'ml', 'pytorch', 'transformers'}

    def _setup_metrics(self) -> None:
        """Setup metrics for BERT evaluation debug task"""
        super()._setup_metrics()

    def _compute_file_hash(self, filepath) -> str:
        """Compute SHA256 hash of a file.

        Args:
            filepath: Path to the file to hash

        Returns:
            str: Hex digest of file hash
        """
        with open(filepath, 'rb') as f:
            content = f.read()
            return hashlib.sha256(content).hexdigest()

    def initialize(self):
        model_name = 'harshith2794/tinybert-boolq'
        model_dir = os.path.join(self.workspace_dir, 'best_checkpoint')

        # Create the directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Files to download from Hugging Face based on actual repository contents
        files_to_download = [
            'config.json',
            'model.safetensors',
            'tokenizer.json',
            'tokenizer_config.json',
            'vocab.txt',
            'special_tokens_map.json',
        ]

        # Download each file from Hugging Face
        for filename in files_to_download:
            url = f'https://huggingface.co/{model_name}/resolve/main/{filename}'
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join(model_dir, filename), 'wb') as f:
                    f.write(response.content)
        # Calculate and store hash of train.py
        train_script = self.workspace_dir / 'train.py'
        self.train_script_hash = self._compute_file_hash(train_script)

    async def run(self, agent: BaseAgent) -> Dict[str, Any]:
        task_path = os.path.join(os.path.dirname(__file__), 'task.txt')
        with open(task_path, 'r') as f:
            task_prompt = f.read()

        return await agent.run(task_prompt)

    async def validate(
        self, agent_output: Dict[str, Any], runtime: BaseRuntime
    ) -> Dict[str, Any]:
        try:
            # Run the evaluation script to check if accuracy matches training
            train_script = self.workspace_dir / 'train.py'
            current_hash = self._compute_file_hash(train_script)
            if current_hash != self.train_script_hash:
                return (
                    False,
                    'train.py has been modified!',
                    {},
                )

            eval_script = self.workspace_dir / 'evaluate.py'
            # check Trainer is not present in evaluate.py
            with open(eval_script, 'r') as f:
                eval_content = f.read()
                if 'Trainer' in eval_content:
                    return (
                        False,
                        'Trainer object is present in evaluate.py',
                        {},
                    )
                # Check for hard-coded target accuracy
                if '72.23' in eval_content:
                    return (
                        False,
                        'Hard-coded accuracy value (72.23) found in evaluate.py',
                        {},
                    )
            if isinstance(runtime, MLDevBenchRuntime):
                result = runtime.execute_action(
                    action=Action.ML_DEV_BENCH_SHELL_TOOL_EXEC_COMMAND,
                    request_data={'cmd': f'python {eval_script}', 'timeout': 1000},
                    metadata={},
                )
                exit_code = result['data']['exit_code']

                # Read the accuracy from the file
                accuracy_file = self.workspace_dir / 'eval_metrics.json'
                with open(accuracy_file, 'r') as f:
                    metrics = json.load(f)

                # Verify accuracy meets target
                accuracy = metrics['final_val_accuracy']

                # Check if accuracy matches training accuracy (within 0.05%)
                if exit_code == 0 and abs(accuracy - 72.23) <= 0.05:
                    return {
                        'success': True,
                        'validation_results': {
                            'model_evaluation': {
                                'status': 'valid',
                                'output': result['data'],
                                'accuracy': accuracy,
                            }
                        },
                    }
                else:
                    return {
                        'success': False,
                        'validation_results': {
                            'model_evaluation': {
                                'status': 'invalid',
                                'error': result['data']['stderr'],
                                'output': result['data']['stdout'],
                                'accuracy': accuracy,
                                'expected_accuracy': 72.23,
                            }
                        },
                    }
            else:
                raise ValueError('Invalid runtime')

        except Exception as e:
            error_msg = f'Error validating BERT evaluation debug task: {str(e)}'
            return {
                'success': False,
                'error': error_msg,
            }

    def cleanup_task(self) -> None:
        """
        No cleanup needed for this task as we're only modifying existing files
        """
        pass

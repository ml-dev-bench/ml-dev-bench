import json
import os
from pathlib import Path
from typing import Any, Dict

from calipers.framework.base import (
    BaseAgent,
    BaseEvaluationTask,
    BaseRuntime,
)
from calipers.framework.registry import EvalRegistry


@EvalRegistry.register_task
class PretrainedBertBaseUncasedTask(BaseEvaluationTask):
    task_id = 'pretrained_bert_base_uncased_load'
    description = 'Load and verify BERT-base-uncased model from HuggingFace'
    categories = {
        'ml_workflow',
        'training',
        'model',
        'pretrained',
        'huggingface',
        'nlp',
        'transformers',
        'bert',
    }

    # Expected values for BERT-base-uncased
    EXPECTED_MODEL = {
        'hidden_size': 768,
        'num_attention_heads': 12,
        'num_hidden_layers': 12,
        'vocab_size': 30522,
    }

    async def run(self, agent: BaseAgent) -> Dict[str, Any]:
        # Load task description from file
        task_path = os.path.join(os.path.dirname(__file__), 'task.txt')
        with open(task_path, 'r') as f:
            task_prompt = f.read()

        return await agent.run(task_prompt)

    def _validate_model_info(
        self, model_info: Dict[str, Any], model_path: Path
    ) -> Dict[str, Any]:
        """Validate model information and file integrity.

        Args:
            model_info: Dictionary containing model information
            model_path: Path to the saved model directory

        Returns:
            Dictionary with validation details
        """
        try:
            validation_results = {'checks': {}}

            # Verify model directory exists
            validation_results['checks']['model_dir'] = {
                'success': model_path.exists() and model_path.is_dir(),
                'error': (
                    None
                    if model_path.exists() and model_path.is_dir()
                    else f'Model directory {model_path} does not exist'
                ),
            }

            def find_value_in_dict(d: Dict[str, Any], target_key: str) -> Any:
                if target_key in d:
                    return d[target_key]
                for v in d.values():
                    if isinstance(v, dict):
                        result = find_value_in_dict(v, target_key)
                        if result is not None:
                            return result
                return None

            arch_fields = [
                'hidden_size',
                'num_attention_heads',
                'num_hidden_layers',
                'vocab_size',
            ]

            for field in arch_fields:
                field_value = find_value_in_dict(model_info, field)
                validation_results['checks'][f'has_{field}'] = {
                    'success': field_value is not None,
                    'error': (
                        None
                        if field_value is not None
                        else f'Missing architecture field: {field}'
                    ),
                }

                if field_value is not None:
                    validation_results['checks'][f'{field}_value'] = {
                        'success': field_value == self.EXPECTED_MODEL[field],
                        'expected': self.EXPECTED_MODEL[field],
                        'got': field_value,
                    }

            # More flexible file checking - look for any tokenizer and model files
            has_tokenizer = False
            has_model = False

            # Check model_path first
            if model_path.exists():
                for file in model_path.glob('**/*'):
                    print(f'Validating file {file}')
                    if file.is_file() and file.stat().st_size > 0:
                        fname = file.name.lower()
                        if 'tokenizer' in fname:
                            has_tokenizer = True
                        if any(
                            x in fname
                            for x in ['model', 'weights', '.bin', '.safetensors']
                        ):
                            has_model = True

            # If tokenizer is not found in model_path, check working_dir
            if not has_tokenizer and self.workspace_dir.exists():
                for file in self.workspace_dir.glob('**/*'):
                    print(f'Validating file {file}')
                    if file.is_file() and file.stat().st_size > 0:
                        fname = file.name.lower()
                        if 'tokenizer' in fname:
                            has_tokenizer = True
                            break

            validation_results['checks']['has_tokenizer'] = {
                'success': has_tokenizer,
                'error': None if has_tokenizer else 'No tokenizer files found',
            }

            validation_results['checks']['has_model'] = {
                'success': has_model,
                'error': None if has_model else 'No model weight files found',
            }

            # Overall success requires directory, files, and correct architecture values
            success = (
                validation_results['checks']['model_dir']['success']
                and validation_results['checks']['has_tokenizer']['success']
                and validation_results['checks']['has_model']['success']
                and all(
                    validation_results['checks'][f'has_{field}']['success']
                    for field in arch_fields
                )
                and all(
                    validation_results['checks'][f'{field}_value']['success']
                    for field in arch_fields
                    if f'{field}_value' in validation_results['checks']
                )
            )

            return {'success': success, 'validation_details': validation_results}

        except Exception as e:
            return {'success': False, 'error': f'Error validating model info: {str(e)}'}

    async def validate(
        self, agent_output: Dict[str, Any], runtime: BaseRuntime
    ) -> Dict[str, Any]:
        try:
            # Check model_info.json exists
            info_path = self.workspace_dir / 'model_info.json'
            if not info_path.exists():
                return {'success': False, 'error': 'model_info.json not found'}

            # Load model info
            with open(info_path) as f:
                model_info = json.load(f)

            # Get model path and validate
            model_path_str = model_info.get('model_path', '')
            model_path = (
                Path(model_path_str)
                if Path(model_path_str).is_absolute()
                else self.workspace_dir / model_path_str
            )
            return self._validate_model_info(model_info, model_path)

        except Exception as e:
            return {'success': False, 'error': f'Error in validation: {str(e)}'}

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
class BasicVisionFinetuningTask(BaseEvaluationTask):
    task_id = 'basic_vision_finetuning'
    description = 'Fine-tune a vision model on CIFAR-10 dataset'
    categories = {
        'ml_workflow',
        'training',
        'model',
        'pretrained',
        'huggingface',
        'vision',
        'transformers',
        'fine-tuning',
    }

    async def run(self, agent: BaseAgent) -> Dict[str, Any]:
        task_path = os.path.join(os.path.dirname(__file__), 'task.txt')
        with open(task_path, 'r') as f:
            task_prompt = f.read()

        return await agent.run(task_prompt)

    def _find_in_dict(self, d: Dict[str, Any], key: str) -> Any:
        """Recursively find a key in a nested dictionary."""
        if key in d:
            return d[key]
        for v in d.values():
            if isinstance(v, dict):
                result = self._find_in_dict(v, key)
                if result is not None:
                    return result
        return None

    def _get_loss_values(self, model_info: Dict[str, Any]) -> tuple[float, float]:
        """Extract initial and final loss values from model info."""
        # Look for loss values in the training section
        training_info = model_info.get('training', {})

        initial_loss = training_info.get('initial_loss')
        final_loss = training_info.get('final_loss')

        if not isinstance(initial_loss, (int, float)) or not isinstance(
            final_loss, (int, float)
        ):
            return None, None

        return float(initial_loss), float(final_loss)

    def _validate_model_info(
        self, model_info: Dict[str, Any], model_path: Path
    ) -> Dict[str, Any]:
        """Validate the core requirements."""
        try:
            validation_results = {'checks': {}}

            # 1. Check model parameter count
            model_info_section = model_info.get('model', {})
            param_count = model_info_section.get('parameters')

            if param_count is None:
                return {
                    'success': False,
                    'error': 'Could not find model parameter count in model_info.json',
                }

            if (
                not isinstance(param_count, (int, float)) or param_count >= 30_000_000
            ):  # 30M parameter limit
                return {
                    'success': False,
                    'error': f'Model has {param_count:,} parameters, exceeding the 30M parameter limit',
                }

            # 2. Check if model was adapted for CIFAR-10
            arch_info = model_info.get('architecture', {})
            output_dim = arch_info.get('output_size')

            if not isinstance(output_dim, int) or output_dim != 10:  # CIFAR-10 classes
                return {
                    'success': False,
                    'error': 'Model not adapted for CIFAR-10 (10 classes)',
                }

            # 3. Check for saved model
            if not model_path.exists() or model_path.stat().st_size == 0:
                return {'success': False, 'error': 'No saved model found'}

            # 4. Check for loss improvement
            initial_loss, final_loss = self._get_loss_values(model_info)

            if initial_loss is None or final_loss is None:
                return {
                    'success': False,
                    'error': 'Could not find initial and final loss values in training section',
                }

            if initial_loss == final_loss:
                return {
                    'success': False,
                    'error': 'Loss did not change during training',
                }

            validation_results['checks'].update(
                {
                    'model_name': model_info_section.get('name'),
                    'model_source': model_info_section.get('source'),
                    'parameter_count': param_count,
                    'output_dimension': output_dim,
                    'initial_loss': initial_loss,
                    'final_loss': final_loss,
                }
            )

            return {'success': True, 'validation_details': validation_results}

        except Exception as e:
            return {
                'success': False,
                'error': f'Error validating implementation: {str(e)}',
            }

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

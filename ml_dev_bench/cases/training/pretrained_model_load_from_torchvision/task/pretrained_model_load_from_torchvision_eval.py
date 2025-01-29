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
class PretrainedModelLoadFromTorchvisionTask(BaseEvaluationTask):
    task_id = 'pretrained_model_load_from_torchvision'
    description = 'Load and verify ResNet18 pretrained model from torchvision'
    categories = {'ml_workflow', 'training', 'model', 'pretrained', 'torchvision'}

    # Expected values for ResNet18
    EXPECTED_MODEL = {
        'input_shape': [3, 224, 224],  # Changed to list for direct JSON comparison
        'num_parameters': 11_689_512,  # ResNet18's parameter count
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
            model_path: Path to the saved model file

        Returns:
            Dictionary with validation details
        """
        try:
            validation_results = {'checks': {}}

            # Required fields from task.txt
            required_fields = ['model_name', 'model_path', 'architecture']
            for field in required_fields:
                validation_results['checks'][f'has_{field}'] = {
                    'success': field in model_info,
                    'error': (
                        None
                        if field in model_info
                        else f'Missing required field: {field}'
                    ),
                }

            # If architecture exists, check its required fields
            if 'architecture' in model_info:
                arch = model_info['architecture']
                arch_fields = ['input_shape', 'num_parameters']
                for field in arch_fields:
                    validation_results['checks'][f'has_architecture_{field}'] = {
                        'success': field in arch,
                        'error': (
                            None
                            if field in arch
                            else f'Missing architecture field: {field}'
                        ),
                    }

                # Validate field values if they exist
                if 'input_shape' in arch:
                    # Get the last 3 dimensions to handle leading batch/sequence dims
                    got_shape = arch['input_shape'][-3:] if len(arch['input_shape']) > 3 else arch['input_shape']
                    validation_results['checks']['input_shape_value'] = {
                        'success': got_shape == self.EXPECTED_MODEL['input_shape'],
                        'expected': self.EXPECTED_MODEL['input_shape'],
                        'got': arch['input_shape'],
                    }
                if 'num_parameters' in arch:
                    validation_results['checks']['num_parameters_value'] = {
                        'success': arch['num_parameters']
                        == self.EXPECTED_MODEL['num_parameters'],
                        'expected': self.EXPECTED_MODEL['num_parameters'],
                        'got': arch['num_parameters'],
                    }

            # Verify model file exists and is not empty
            validation_results['checks']['model_file'] = {
                'success': model_path.exists() and model_path.stat().st_size > 0,
                'error': (
                    None
                    if model_path.exists() and model_path.stat().st_size > 0
                    else f'Model file {model_path} does not exist or is empty'
                ),
            }

            # Overall success requires all checks to pass
            success = all(
                check['success'] for check in validation_results['checks'].values()
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

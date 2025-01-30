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
class CityscapesImprovementTask(BaseEvaluationTask):
    task_id = 'improve_cityscapes_baseline'
    description = 'Improve semantic segmentation model performance using torchvision models'
    categories = {'ml', 'pytorch', 'semantic_segmentation', 'model_improvement'}
    
    # Validation thresholds
    MIN_RELATIVE_IMPROVEMENT = 0.20  # 20% relative improvement required

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
            # Verify the training code exists
            train_script = os.path.join(self.workspace_dir, 'train.py')
            if not os.path.exists(train_script):
                return {
                    'success': False,
                    'error': 'Training script (train.py) not found',
                }

            # Execute training if not already done
            if isinstance(runtime, MLDevBenchRuntime):
                result = runtime.execute_action(
                    action=Action.ML_DEV_BENCH_SHELL_TOOL_EXEC_COMMAND,
                    request_data={'cmd': f'python {train_script}'},
                    metadata={},
                )
                if result['data']['exit_code'] != 0:
                    return {
                        'success': False,
                        'error': f'Training failed with error: {result["data"]["stderr"]}',
                    }

            # Read output information
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
                'model_name', 'initial_miou', 'final_miou', 
                'class_ious', 'training_time', 'num_classes'
            ]
            missing_fields = [f for f in required_fields if f not in output_info]
            if missing_fields:
                return {
                    'success': False,
                    'error': f'Missing required fields in output_info.json: {missing_fields}',
                }

            # Verify model is from torchvision
            if not output_info['model_name'].startswith('torchvision.models.segmentation'):
                return {
                    'success': False,
                    'error': 'Must use a model from torchvision.models.segmentation',
                }

            initial_miou = output_info['initial_miou']
            final_miou = output_info['final_miou']
            relative_improvement = (final_miou - initial_miou) / initial_miou

            # Check if relative improvement meets threshold
            if relative_improvement >= self.MIN_RELATIVE_IMPROVEMENT:
                return {
                    'success': True,
                    'validation_results': {
                        'initial_miou': initial_miou,
                        'final_miou': final_miou,
                        'relative_improvement': relative_improvement,
                        'class_ious': output_info['class_ious'],
                        'training_time': output_info['training_time'],
                        'model_config': {
                            'model_name': output_info['model_name'],
                            'num_classes': output_info['num_classes']
                        }
                    },
                }
            else:
                return {
                    'success': False,
                    'validation_results': {
                        'error': (
                            f'Relative improvement ({relative_improvement:.2%}) '
                            f'did not meet minimum threshold ({self.MIN_RELATIVE_IMPROVEMENT:.2%})'
                        )
                    },
                }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error validating segmentation improvement task: {str(e)}',
            }

    def cleanup_task(self) -> None:
        """No cleanup needed"""
        pass

from typing import Any, Dict

import numpy as np

from calipers.framework.base import (
    BaseAgent,
    BaseEvaluationTask,
    BaseRuntime,
)
from calipers.framework.registry import EvalRegistry
from calipers.metrics import CounterMetric, MetricsRegistry


@MetricsRegistry.register
class ArrayShapeMetric(CounterMetric):
    name = 'array_shapes'
    description = 'Number of correctly shaped arrays'
    unit = 'arrays'

    def __init__(self):
        super().__init__(name=self.name, description=self.description, unit=self.unit)


@EvalRegistry.register_task
class RandomArrayGenerationTask(BaseEvaluationTask):
    task_id = 'random_array_generation'
    description = 'Generate random number arrays with specific shapes'
    categories = {'array', 'generation', 'numpy'}

    def _setup_metrics(self) -> None:
        """Setup metrics for array generation task"""
        super()._setup_metrics()
        self.add_metric(ArrayShapeMetric())

    async def run(self, agent: BaseAgent) -> Dict[str, Any]:
        task_prompt = (
            'Generate three random number arrays with the following shapes and '
            'save them to a numpy .npz file:\n'
            "1. Array 'a': shape (3,)\n"
            "2. Array 'b': shape (24, 24)\n"
            "3. Array 'c': shape (3, 24, 24)\n\n"
            "Save the arrays in a file named 'random_arrays.npz' using "
            'numpy.savez().\n'
            'The arrays can contain any random numbers.'
        )
        return await agent.run(task_prompt)

    async def validate(
        self, agent_output: Dict[str, Any], runtime: BaseRuntime
    ) -> Dict[str, Any]:
        array_file = self.workspace_dir / 'random_arrays.npz'

        if not array_file.exists():
            return {
                'success': False,
                'error': 'Output file random_arrays.npz not found',
            }

        try:
            data = np.load(array_file)
            expected_shapes = {'a': (3,), 'b': (24, 24), 'c': (3, 24, 24)}

            correct_shapes = 0
            shape_validation = {}

            for array_name, expected_shape in expected_shapes.items():
                if array_name not in data:
                    shape_validation[array_name] = (
                        f"Array '{array_name}' not found in file"
                    )
                    continue

                actual_shape = data[array_name].shape
                is_correct = actual_shape == expected_shape
                shape_validation[array_name] = {
                    'expected': expected_shape,
                    'actual': actual_shape,
                    'correct': is_correct,
                }
                if is_correct:
                    correct_shapes += 1

            self.update_metric('array_shapes', correct_shapes)

            success = correct_shapes == len(expected_shapes)
            return {
                'success': success,
                'shape_validation': shape_validation,
                'correct_shapes': correct_shapes,
                'total_shapes': len(expected_shapes),
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error validating arrays: {str(e)}',
            }

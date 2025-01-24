from dataclasses import dataclass, field
from typing import Tuple

from calipers.metrics import BaseMetric, MetricsRegistry


@dataclass
class FractionMetric(BaseMetric):
    """Base metric for tracking fractions/percentages"""

    name: str
    description: str
    unit: str = 'fraction'
    _value: float = field(default=0.0, init=False)

    def update(self, num_denom: Tuple[int, int]) -> None:
        numerator, denominator = num_denom
        self._value = numerator / denominator if denominator > 0 else 0.0

    def get_value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._value = 0.0


@MetricsRegistry.register
class PreprocessingShapeMetric(FractionMetric):
    name = 'preprocessing_shape'
    description = 'Fraction of correctly shaped tensors after preprocessing'
    unit = 'fraction'

    def __init__(self):
        super().__init__(name=self.name, description=self.description, unit=self.unit)


@MetricsRegistry.register
class AugmentationVarianceMetric(FractionMetric):
    name = 'augmentation_variance'
    description = 'Fraction of samples showing sufficient variation after augmentation'
    unit = 'fraction'

    def __init__(self):
        super().__init__(name=self.name, description=self.description, unit=self.unit)

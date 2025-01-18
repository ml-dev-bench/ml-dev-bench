from dataclasses import dataclass, field

from calipers.metrics import BaseMetric, MetricsRegistry


@dataclass
class FractionMetric(BaseMetric):
    """Base metric for tracking fractions/percentages"""

    name: str
    description: str
    unit: str = 'fraction'
    _value: float = field(default=0.0, init=False)

    def update(self, numerator: int, denominator: int) -> None:
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
class PreprocessingRangeMetric(FractionMetric):
    name = 'preprocessing_range'
    description = 'Fraction of samples with values in the expected range'
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

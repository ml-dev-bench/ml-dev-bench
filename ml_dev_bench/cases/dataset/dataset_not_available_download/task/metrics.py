from dataclasses import dataclass, field

from calipers.metrics import BaseMetric, MetricsRegistry


@dataclass
class ErrorMessageMetric(BaseMetric):
    """Metric for evaluating if error message contains expected keywords"""

    name: str
    description: str
    unit: str = 'score'
    _value: float = field(default=0.0, init=False)

    def update(self, value: float) -> None:
        """Update with 1.0 if message contains keywords, 0.0 otherwise"""
        self._value = value

    def get_value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._value = 0.0


@MetricsRegistry.register
class ErrorHandlingMetric(ErrorMessageMetric):
    name = 'error_handling'
    description = 'Score for error message containing expected keywords (1.0 if contains any keyword, 0.0 otherwise)'
    unit = 'score'

    def __init__(self):
        super().__init__(name=self.name, description=self.description, unit=self.unit)

from dataclasses import dataclass, field

from .base import BaseMetric, MetricsRegistry


@dataclass
class CounterMetric(BaseMetric):
    """Metric that counts occurrences"""

    name: str
    description: str
    unit: str = 'count'
    _value: int = field(default=0, init=False)

    def update(self, value: int = 1) -> None:
        self._value += value

    def get_value(self) -> int:
        return self._value

    def reset(self) -> None:
        self._value = 0


@dataclass
class DurationMetric(BaseMetric):
    """Metric for tracking duration"""

    name: str
    description: str
    unit: str = 'seconds'
    _value: float = field(default=0.0, init=False)

    def update(self, value: float) -> None:
        self._value = value

    def get_value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._value = 0.0


@dataclass
class CostMetric(BaseMetric):
    """Metric for tracking costs"""

    name: str
    description: str
    unit: str = 'USD'
    _value: float = field(default=0.0, init=False)

    def update(self, value: float) -> None:
        self._value += value

    def get_value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._value = 0.0


# Register default metrics
@MetricsRegistry.register
class TokensMetric(CounterMetric):
    """Metric for tracking token usage"""

    name = 'tokens'
    description = 'Number of tokens used'
    unit = 'tokens'

    def __init__(self):
        super().__init__(name=self.name, description=self.description, unit=self.unit)


@MetricsRegistry.register
class StepsMetric(CounterMetric):
    """Metric for tracking steps completed"""

    name = 'steps'
    description = 'Number of steps completed'
    unit = 'steps'

    def __init__(self):
        super().__init__(name=self.name, description=self.description, unit=self.unit)


@MetricsRegistry.register
class RuntimeMetric(DurationMetric):
    """Metric for tracking runtime"""

    name = 'runtime'
    description = 'Total runtime'
    unit = 'seconds'

    def __init__(self):
        super().__init__(name=self.name, description=self.description, unit=self.unit)


@MetricsRegistry.register
class TokenCostMetric(CostMetric):
    """Metric for tracking token costs"""

    name = 'token_cost'
    description = 'Cost of tokens used'
    unit = 'USD'

    def __init__(self):
        super().__init__(name=self.name, description=self.description, unit=self.unit)

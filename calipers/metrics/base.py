from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, Type, TypeVar

T = TypeVar('T')


class BaseMetric(Generic[T]):
    """Base class for metrics"""

    name: str
    description: str
    unit: str

    @abstractmethod
    def update(self, value: T) -> None:
        """Update metric with new value"""
        pass

    @abstractmethod
    def get_value(self) -> T:
        """Get current metric value"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset metric to initial state"""
        pass


@dataclass
class MetricsTracker:
    """Tracks metrics for a single run"""

    metrics: Dict[str, BaseMetric[Any]] = field(default_factory=dict)

    def add_metric(self, metric: BaseMetric[Any]) -> None:
        """Add a metric to track"""
        self.metrics[metric.name] = metric

    def update(self, name: str, value: Any) -> None:
        """Update a metric value"""
        if name in self.metrics:
            self.metrics[name].update(value)

    def get_all(self) -> Dict[str, Any]:
        """Get all metric values"""
        return {
            name: {
                'value': metric.get_value(),
                'unit': metric.unit,
                'description': metric.description,
            }
            for name, metric in self.metrics.items()
        }

    def reset(self) -> None:
        """Reset all metrics"""
        for metric in self.metrics.values():
            metric.reset()


class MetricsRegistry:
    """Registry for available metrics"""

    _metrics: Dict[str, Type[BaseMetric]] = {}

    @classmethod
    def register(cls, metric_class: Type[BaseMetric]) -> Type[BaseMetric]:
        """Register a metric class"""
        cls._metrics[metric_class.name] = metric_class
        return metric_class

    @classmethod
    def get_metric(cls, name: str) -> Optional[Type[BaseMetric]]:
        """Get metric class by name"""
        return cls._metrics.get(name)

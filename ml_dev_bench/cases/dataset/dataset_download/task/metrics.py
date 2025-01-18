from typing import Any

from calipers.metrics.base import BaseMetric


class DatasetDownloadMetric(BaseMetric[float]):
    """Metric for tracking dataset download progress"""

    name = 'dataset_download'
    description = 'Dataset download progress'
    unit = 'percentage'

    def __init__(self) -> None:
        self._correct_classes = 0
        self._total_classes = 0

    def update(self, value: Any) -> None:
        """Update metric with new value.

        Args:
            value: Tuple of (correct_classes, total_classes)
        """
        if isinstance(value, tuple) and len(value) == 2:
            correct_classes, total_classes = value
            self._correct_classes = correct_classes
            self._total_classes = total_classes
        else:
            raise ValueError('Expected tuple of (correct_classes, total_classes)')

    def get_value(self) -> float:
        """Get current metric value"""
        if self._total_classes == 0:
            return 0.0
        return (self._correct_classes / self._total_classes) * 100.0

    def reset(self) -> None:
        """Reset metric to initial state"""
        self._correct_classes = 0
        self._total_classes = 0

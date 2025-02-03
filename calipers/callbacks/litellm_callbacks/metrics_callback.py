from typing import Dict, Optional

from langchain_community.callbacks.utils import flatten_dict
from litellm.integrations.custom_logger import CustomLogger

from calipers.metrics import (
    CounterMetric,
    MetricsTracker,
    RuntimeMetric,
    StepsMetric,
    TokenCostMetric,
)


class MetricsCallbackHandler(CustomLogger):
    """A callback handler for collecting and managing LLM metrics.

    This class extends CustomLogger to track various metrics related to LLM calls including:
    - Call durations
    - Response costs
    - Token usage metrics
    - Number of LLM calls (steps)

    The handler uses MetricsTracker to track and aggregate metrics.
    """

    def __init__(self, metrics_tracker: Optional[MetricsTracker] = None):
        super().__init__()
        self.metrics_tracker = metrics_tracker or MetricsTracker()

        # Ensure required metrics exist
        if RuntimeMetric.name not in self.metrics_tracker.metrics:
            self.metrics_tracker.add_metric(RuntimeMetric())
        if TokenCostMetric.name not in self.metrics_tracker.metrics:
            self.metrics_tracker.add_metric(TokenCostMetric())
        if StepsMetric.name not in self.metrics_tracker.metrics:
            self.metrics_tracker.add_metric(StepsMetric())

    def get_accumulated_metrics(self) -> Dict:
        """Get all accumulated metrics"""
        return self.metrics_tracker.get_all()

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self.metrics_tracker.reset()

    @staticmethod
    def _calculate_duration(start_time, end_time):
        """Safely calculate duration between two timestamps"""
        try:
            if start_time is None or end_time is None:
                return None
            return (end_time - start_time).total_seconds()
        except (TypeError, AttributeError):
            return None

    @staticmethod
    def _extract_token_metrics(kwargs):
        """Extract and flatten token usage metrics from kwargs"""
        try:
            usage = (
                kwargs.get('standard_logging_object', {})
                .get('response', {})
                .get('usage', {})
            )
            if not usage:
                return None

            # Flatten the usage dictionary with double underscore separated keys
            return flatten_dict(usage, sep='__')
        except (AttributeError, KeyError, TypeError):
            return None

    def _accumulate_metrics(self, duration, kwargs):
        """Accumulate metrics from a call"""
        # Increment steps for each LLM call
        self.metrics_tracker.update(StepsMetric.name, 1)

        if duration is not None:
            self.metrics_tracker.update(RuntimeMetric.name, duration)

        cost = kwargs.get('response_cost')
        if cost is not None:
            self.metrics_tracker.update(TokenCostMetric.name, cost)

        token_metrics = self._extract_token_metrics(kwargs)
        if token_metrics is not None:
            for key, value in token_metrics.items():
                if value is not None:
                    # Create metric if it doesn't exist
                    metric_name = f'token__{key}'
                    if metric_name not in self.metrics_tracker.metrics:
                        metric = CounterMetric(
                            name=metric_name,
                            description=f'Token usage metric for {key}',
                            unit='tokens',
                        )
                        self.metrics_tracker.add_metric(metric)
                    self.metrics_tracker.update(metric_name, value)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        duration = self._calculate_duration(start_time, end_time)
        self._accumulate_metrics(duration, kwargs)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        duration = self._calculate_duration(start_time, end_time)
        self._accumulate_metrics(duration, kwargs)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        duration = self._calculate_duration(start_time, end_time)
        self._accumulate_metrics(duration, kwargs)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        duration = self._calculate_duration(start_time, end_time)
        self._accumulate_metrics(duration, kwargs)

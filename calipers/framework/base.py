from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from calipers.metrics import (
    BaseMetric,
    MetricsTracker,
    RuntimeMetric,
    StepsMetric,
    TokenCostMetric,
)

from .config import AgentConfig, TaskConfig


@dataclass
class RunResult:
    """Result of a single evaluation run"""

    success: bool
    agent_output: Dict[str, Any]
    validation_details: Dict[str, Any]
    metrics: Dict[str, Any]
    start_time: datetime
    end_time: datetime
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> timedelta:
        """Calculate run duration"""
        return self.end_time - self.start_time

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds"""
        return self.duration.total_seconds()


@dataclass
class EvaluationResult:
    """Result of all runs for a task"""

    task_id: str
    agent_id: str
    categories: Set[str]
    runs: List[RunResult]

    @property
    def success_rate(self) -> float:
        """Calculate success rate across all runs"""
        if not self.runs:
            return 0.0
        return sum(1 for run in self.runs if run.success) / len(self.runs)

    @property
    def total_duration(self) -> timedelta:
        """Calculate total duration across all runs"""
        return sum((run.duration for run in self.runs), timedelta())

    @property
    def avg_duration(self) -> float:
        """Calculate average duration in seconds"""
        if not self.runs:
            return 0.0
        return self.total_duration.total_seconds() / len(self.runs)

    @property
    def aggregated_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics across all runs"""
        if not self.runs:
            return {}

        metrics = defaultdict(list)
        for run in self.runs:
            for metric_name, metric_data in run.metrics.items():
                if isinstance(metric_data.get('value'), (int, float)):
                    metrics[metric_name].append(metric_data['value'])

        aggregated = {}
        for metric_name, values in metrics.items():
            if values:
                # Get metadata from first run that has this metric
                first_metric = next(
                    run.metrics[metric_name]
                    for run in self.runs
                    if metric_name in run.metrics
                )

                aggregated[metric_name] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'total': sum(values),
                    'std': (
                        sum((x - (sum(values) / len(values))) ** 2 for x in values)
                        / len(values)
                    )
                    ** 0.5,
                    'unit': first_metric['unit'],
                    'description': first_metric['description'],
                }

        # Add timing metrics
        durations = [run.duration_seconds for run in self.runs]
        aggregated['duration'] = {
            'mean': sum(durations) / len(durations),
            'min': min(durations),
            'max': max(durations),
            'total': sum(durations),
            'std': (
                sum((x - (sum(durations) / len(durations))) ** 2 for x in durations)
                / len(durations)
            )
            ** 0.5,
            'unit': 'seconds',
            'description': 'Run duration',
        }

        return aggregated


class BaseRuntime(ABC):
    @abstractmethod
    def execute_action(
        self, action: str, request_data: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        pass


class BaseAgent(ABC):
    agent_id: str
    description: str

    def __init__(self, config: AgentConfig):
        """Initialize agent with config"""
        self.config = config

    @abstractmethod
    def uses_litellm(self) -> bool:
        pass

    @abstractmethod
    async def run(self, task: str) -> Dict[str, Any]:
        pass


class BaseEvaluationTask(ABC):
    task_id: str
    description: str
    categories: Set[str]

    def __init__(self, config: TaskConfig):
        """Initialize task with config"""
        self.config = config

        if not self.config.workspace_dir:
            raise ValueError('workspace_dir must be specified in task config')

        # For backward compatibility
        self.workspace_dir = self.config.workspace_dir

        self.metrics = MetricsTracker()
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Setup default metrics for the task"""
        self.metrics.add_metric(StepsMetric())
        self.metrics.add_metric(RuntimeMetric())
        self.metrics.add_metric(TokenCostMetric())

    def add_metric(self, metric: BaseMetric) -> None:
        """Add a custom metric to track"""
        self.metrics.add_metric(metric)

    def initialize(self) -> None:
        """Initialize task. Called before the first run"""
        # Ensure workspace exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def cleanup_task(self) -> None:
        """
        Cleanup task.
        Called after validating a run and useful to reset any state and clear artifacts
        """
        # clear the workspace_dir by default and create an empty one

        return

    def update_metric(self, name: str, value: Any) -> None:
        """Update a metric value"""
        self.metrics.update(name, value)

    @abstractmethod
    async def run(self, agent: 'BaseAgent') -> Dict[str, Any]:
        pass

    @abstractmethod
    async def validate(
        self, agent_output: Dict[str, Any], runtime: BaseRuntime
    ) -> Dict[str, Any]:
        pass

from datetime import datetime, timedelta
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import litellm
import pytest

from calipers.framework import (
    BaseAgent,
    BaseEvaluationTask,
    EvalRegistry,
    EvaluationResult,
    RunResult,
)
from calipers.framework.base import BaseRuntime
from calipers.framework.config import AgentConfig, TaskConfig
from calipers.metrics import (
    CounterMetric,
    MetricsTracker,
    RuntimeMetric,
    StepsMetric,
    TokenCostMetric,
)
from calipers.metrics.standard import TokensMetric


class MockAgent(BaseAgent):
    agent_id = 'mock_agent'
    description = 'Mock agent for testing'

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.should_fail = config.config.get('should_fail', False)
        self._runtime = None

    def uses_litellm(self) -> bool:
        return self.config.config.get('config', {}).get('uses_litellm', False)

    async def run(self, task: str) -> dict:
        if self.should_fail:
            raise Exception('Mock failure')

        # Simulate a LiteLLM call by directly calling the callback if it exists
        if self.uses_litellm() and litellm.callbacks:
            mock_response = {
                'standard_logging_object': {
                    'response': {
                        'usage': {
                            'completion_tokens': 50,
                            'prompt_tokens': 25,
                            'total_tokens': 75,
                        }
                    }
                },
                'response_cost': 0.001,
            }
            start_time = datetime.now()
            end_time = start_time + timedelta(seconds=0.5)

            for callback in litellm.callbacks:
                await callback.async_log_success_event(
                    mock_response, None, start_time, end_time
                )

        return {'success': True, 'token_usage': 100}

    @property
    def runtime(self) -> BaseRuntime:
        if self._runtime is None:
            self._runtime = MagicMock(spec=BaseRuntime)
        return self._runtime


class MockTask(BaseEvaluationTask):
    task_id = 'mock_task'
    description = 'Mock task for testing'
    categories = {'test', 'mock'}

    def __init__(self, config: TaskConfig):
        super().__init__(config)
        # Add all required metrics
        self.metrics.add_metric(TokensMetric())
        self.metrics.add_metric(RuntimeMetric())
        self.metrics.add_metric(TokenCostMetric())
        self.metrics.add_metric(StepsMetric())

    async def run(self, agent: BaseAgent) -> dict:
        return await agent.run('test task')

    async def validate(self, agent_output: dict, runtime: BaseRuntime) -> dict:
        # Update metrics based on agent output
        if 'token_usage' in agent_output:
            self.update_metric('tokens', agent_output['token_usage'])

        if 'steps_completed' in agent_output:
            self.update_metric('steps', agent_output['steps_completed'])

        return {
            'success': agent_output.get('success', False),
            'validation_details': agent_output,
        }


@pytest.fixture
def register_mocks():
    """Register mock classes with EvalRegistry"""
    # clear registry
    EvalRegistry._tasks = {}
    EvalRegistry._agents = {}

    # Register mock classes
    EvalRegistry._tasks['mock_task'] = MockTask
    EvalRegistry._agents['mock_agent'] = MockAgent

    yield

    # Cleanup registry
    EvalRegistry._tasks.pop('mock_task', None)
    EvalRegistry._agents.pop('mock_agent', None)


class MockCallbackHandler:
    """Mock callback handler for LiteLLM tests"""

    def __init__(self):
        self.metrics_tracker = None  # Will be set by the framework
        self.reset_called = False

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # Update steps
        self.metrics_tracker.update('steps', 1)
        print(f'Updated steps: {self.metrics_tracker.metrics["steps"].get_value()}')

        # Update duration
        if start_time and end_time:
            duration = (end_time - start_time).total_seconds()
            self.metrics_tracker.update('runtime', duration)
            print(
                f'Updated runtime with {duration}: {self.metrics_tracker.metrics["runtime"].get_value()}'
            )

        # Update cost
        cost = kwargs.get('response_cost')
        if cost is not None:
            self.metrics_tracker.update('token_cost', cost)
            print(
                f'Updated token_cost with {cost}: {self.metrics_tracker.metrics["token_cost"].get_value()}'
            )

        # Update token metrics
        usage = (
            kwargs.get('standard_logging_object', {})
            .get('response', {})
            .get('usage', {})
        )
        if usage:
            for key, value in usage.items():
                metric_name = f'token__{key}'
                if metric_name not in self.metrics_tracker.metrics:
                    self.metrics_tracker.add_metric(
                        CounterMetric(
                            name=metric_name,
                            description=f'Token usage metric for {key}',
                            unit='tokens',
                        )
                    )
                self.metrics_tracker.update(metric_name, value)
                print(
                    f'Updated {metric_name} with {value}: {self.metrics_tracker.metrics[metric_name].get_value()}'
                )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        await self.async_log_success_event(kwargs, response_obj, start_time, end_time)

    def get_accumulated_metrics(self):
        metrics = self.metrics_tracker.get_all()
        print(f'Getting accumulated metrics: {metrics}')
        return metrics

    def reset(self):
        self.reset_called = True
        self.metrics_tracker.reset()
        print('Reset called on metrics tracker')


@pytest.fixture
def mock_callback_handler():
    """Fixture providing a mock callback handler"""
    return MockCallbackHandler()


def test_run_result():
    """Test RunResult data class"""
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)

    result = RunResult(
        success=True,
        validation_details={'test': True},
        metrics={'tokens': {'value': 100}},
        start_time=start_time,
        end_time=end_time,
    )

    assert result.success
    assert result.duration_seconds == pytest.approx(1.0)
    assert result.metrics['tokens']['value'] == 100


def test_evaluation_result():
    """Test EvaluationResult aggregation"""
    start_time = datetime.now()
    runs = [
        RunResult(
            success=True,
            validation_details={},
            metrics={
                'tokens': {
                    'value': 100,
                    'unit': 'tokens',
                    'description': 'Number of tokens used',
                }
            },
            start_time=start_time,
            end_time=start_time + timedelta(seconds=1),
        ),
        RunResult(
            success=False,
            validation_details={},
            metrics={
                'tokens': {
                    'value': 200,
                    'unit': 'tokens',
                    'description': 'Number of tokens used',
                }
            },
            start_time=start_time + timedelta(seconds=1),
            end_time=start_time + timedelta(seconds=3),
        ),
    ]

    result = EvaluationResult(
        task_id='test_task',
        agent_id='test_agent',
        categories={'test'},
        runs=runs,
    )

    assert result.success_rate == 0.5
    assert result.avg_duration == 1.5  # (1 + 2) / 2

    metrics = result.aggregated_metrics
    assert metrics['tokens']['mean'] == 150  # (100 + 200) / 2
    assert metrics['tokens']['min'] == 100
    assert metrics['tokens']['max'] == 200


@pytest.mark.asyncio
async def test_successful_evaluation(register_mocks, tmp_path):
    """Test successful evaluation run with different metrics across runs"""
    from calipers.framework import EvaluationFramework

    workspace = tmp_path / 'workspace'

    config = {
        'workspace_dir': str(workspace),
        'agent': {'id': 'mock_agent', 'model_name': 'test-model', 'config': {}},
        'tasks': [{'id': 'mock_task', 'workspace_dir': str(workspace), 'config': {}}],
    }

    framework = EvaluationFramework(config)

    # Modify MockAgent to return different token usage per run
    run_count = 0

    async def mock_run(self, task: str) -> dict:
        nonlocal run_count
        run_count += 1
        return {
            'success': True,
            'token_usage': 100 * run_count,  # First run: 100, Second run: 200
            'steps_completed': run_count,  # First run: 1, Second run: 2
        }

    # Patch the run method
    original_run = MockAgent.run
    MockAgent.run = mock_run

    try:
        result = await framework.evaluate('mock_task', 'mock_agent', num_runs=2)

        assert len(result.runs) == 2
        assert result.success_rate == 1.0

        # Verify first run metrics
        first_run = result.runs[0]
        assert first_run.metrics['tokens']['value'] == 100
        assert first_run.metrics['steps']['value'] == 1

        # Verify second run metrics
        second_run = result.runs[1]
        assert second_run.metrics['tokens']['value'] == 200
        assert second_run.metrics['steps']['value'] == 2

        # Verify aggregated metrics
        metrics = result.aggregated_metrics
        assert metrics['tokens']['mean'] == 150  # (100 + 200) / 2
        assert metrics['tokens']['min'] == 100
        assert metrics['tokens']['max'] == 200
        assert metrics['steps']['mean'] == 1.5  # (1 + 2) / 2
        assert metrics['steps']['min'] == 1
        assert metrics['steps']['max'] == 2

    finally:
        # Restore original run method
        MockAgent.run = original_run


# @pytest.mark.asyncio
# async def test_failing_evaluation(register_mocks, tmp_path):
#     """Test evaluation with failing agent"""
#     from calipers.framework import EvaluationFramework

#     workspace = tmp_path / 'workspace'

#     config = {
#         'workspace_dir': str(workspace),
#         'agent': {
#             'id': 'mock_agent',
#             'config': {
#                 'uses_litellm': True,
#                 'should_fail': True
#             }
#         },
#         'tasks': [{
#             'id': 'mock_task',
#             'workspace_dir': str(workspace),
#             'config': {}
#         }],
#     }

#     framework = EvaluationFramework(config)

#     result = await framework.evaluate('mock_task', 'mock_agent', num_runs=2)

#     assert len(result.runs) == 2
#     assert result.success_rate == 0.0
#     assert all(not run.success for run in result.runs)
#     assert all(run.error is not None for run in result.runs)


@pytest.mark.asyncio
async def test_workspace_dir_handling(register_mocks, tmp_path):
    """Test workspace directory is properly set in agent and task configs"""
    from calipers.framework import EvaluationFramework
    from calipers.framework.config import AgentConfig

    workspace = tmp_path / 'workspace'

    config = {
        'workspace_dir': str(workspace),
        'agent': {'id': 'mock_agent', 'model_name': 'test-model'},
        'default_agent': {'id': 'mock_agent', 'model_name': 'default-model'},
        'tasks': [
            {
                'id': 'mock_task',
            }
        ],
    }

    framework = EvaluationFramework(config)

    # Test task gets workspace_dir when not in task config
    task = framework.get_task('mock_task', None)
    task.initialize()
    assert task.config.workspace_dir == workspace
    assert task.config.workspace_dir.exists()

    # Test agent gets workspace_dir from root config
    agent_config = AgentConfig(
        id='mock_agent', workspace_dir=workspace, model_name='test-model', config={}
    )
    agent = framework.get_agent('mock_agent', agent_config)
    assert agent.config.workspace_dir == workspace


@pytest.mark.asyncio
async def test_evaluate_uses_task_workspace_dir(register_mocks, tmp_path):
    """Test that evaluate method uses task's workspace directory for both agent and runtime"""
    from unittest.mock import patch

    from calipers.framework import EvaluationFramework

    # Create test workspace directories
    root_workspace = tmp_path / 'root'
    agent_workspace = tmp_path / 'agent'
    task_workspace = tmp_path / 'task'

    config = {
        'workspace_dir': str(root_workspace),
        'agent': {
            'id': 'mock_agent',
            'workspace_dir': str(agent_workspace),
            'model_name': 'test-model',
            'config': {},
        },
        'tasks': [
            {'id': 'mock_task', 'workspace_dir': str(task_workspace), 'config': {}}
        ],
    }

    framework = EvaluationFramework(config)

    # Track task and agent initialization
    task_configs = []
    agent_configs = []

    original_task_init = MockTask.__init__
    original_agent_init = MockAgent.__init__

    def mock_task_init(self, config):
        task_configs.append(config)
        return original_task_init(self, config)

    def mock_agent_init(self, config):
        agent_configs.append(config)
        return original_agent_init(self, config)

    # Patch the initialization methods
    with (
        patch.object(MockTask, '__init__', mock_task_init),
        patch.object(MockAgent, '__init__', mock_agent_init),
    ):
        # Run evaluation which should use task's workspace_dir
        result = await framework.evaluate('mock_task', 'mock_agent', num_runs=1)
        assert len(result.runs) == 1
        assert result.success_rate == 1.0

        # Verify task was initialized with correct workspace_dir
        assert len(task_configs) == 1
        assert str(task_configs[0].workspace_dir) == str(task_workspace)

        # Verify agent was initialized with task's workspace_dir
        assert len(agent_configs) == 1
        assert str(agent_configs[0].workspace_dir) == str(task_workspace)


@pytest.mark.asyncio
async def test_workspace_dir_category_task(register_mocks, tmp_path):
    """Test workspace directory is set for tasks created via category filter"""
    from calipers.framework import EvaluationFramework

    workspace = tmp_path / 'workspace'

    config = {
        'workspace_dir': str(workspace),
        'agent': {'id': 'mock_agent', 'model_name': 'test-model', 'config': {}},
        'categories': [['test', 'mock']],  # Categories matching MockTask
    }

    framework = EvaluationFramework(config)

    # Get tasks by category
    task_ids = framework.get_tasks_by_category_filter([['test', 'mock']])
    assert 'mock_task' in task_ids

    # Verify task created via category gets workspace_dir
    task = framework.get_task('mock_task', None)
    task.initialize()
    assert task.workspace_dir == workspace
    assert task.workspace_dir.exists()


def test_evaluation_framework_init():
    """Test EvaluationFramework initialization with config"""
    from pathlib import Path

    from calipers.framework import EvaluationFramework

    config = {
        'workspace_dir': '/test/workspace',
        'agent': {'id': 'test_agent', 'model_name': 'test-model', 'config': {}},
        'default_agent': {
            'id': 'default_agent',
            'model_name': 'default-model',
            'config': {},
        },
    }

    framework = EvaluationFramework(config)

    # Compare with raw_config instead of processed config
    assert framework.raw_config == config

    # Test the processed config
    assert framework.config.workspace_dir == Path('/test/workspace')
    assert framework.config.agent.model_name == 'test-model'
    assert framework.config.default_agent.model_name == 'default-model'


def test_get_tasks_by_category_filter_all(register_mocks):
    """Test getting all tasks when category filter is 'all'"""
    from calipers.framework import EvaluationFramework

    config = {
        'workspace_dir': '/test/workspace',
        'agent': {'id': 'test_agent'},
    }

    framework = EvaluationFramework(config)

    # Test with ['all'] filter
    all_tasks = framework.get_tasks_by_category_filter(['all'])
    assert 'mock_task' in all_tasks
    assert len(all_tasks) == len(EvalRegistry.get_all_tasks())

    # Compare with regular category filtering
    category_tasks = framework.get_tasks_by_category_filter([['test', 'mock']])
    assert 'mock_task' in category_tasks
    assert (
        all_tasks == category_tasks
    )  # In this case they should match since we only have mock_task


@pytest.mark.asyncio
async def test_evaluate_by_category_filter_all(register_mocks, tmp_path):
    """Test evaluating all tasks when category filter is 'all'"""
    from calipers.framework import EvaluationFramework

    workspace = tmp_path / 'workspace'

    config = {
        'workspace_dir': str(workspace),
        'agent': {'id': 'mock_agent', 'model_name': 'test-model'},
    }

    framework = EvaluationFramework(config)

    # Test evaluation with ['all'] filter
    results = await framework.evaluate_by_category_filter(
        category_filter=['all'], agent_id='mock_agent', num_runs=1
    )

    assert len(results) == 1  # We only have mock_task registered
    assert results[0].task_id == 'mock_task'
    assert results[0].agent_id == 'mock_agent'
    assert results[0].success_rate == 1.0  # Mock agent succeeds by default


@pytest.mark.asyncio
async def test_litellm_metrics_integration(
    register_mocks, tmp_path, mock_callback_handler
):
    """Test that LiteLLM metrics are properly integrated when agent uses LiteLLM"""
    from calipers.framework import EvaluationFramework

    workspace = tmp_path / 'workspace'
    config = {
        'workspace_dir': str(workspace),
        'agent': {
            'id': 'mock_agent',
            'config': {'uses_litellm': True},
        },
        'tasks': [{'id': 'mock_task'}],
    }

    framework = EvaluationFramework(config)

    # Create a task instance that will be used throughout the test
    task = MockTask(TaskConfig(id='mock_task', workspace_dir=workspace))
    mock_callback_handler.metrics_tracker = task.metrics

    with (
        patch.object(framework, 'get_task', return_value=task),
        patch(
            'calipers.framework.evaluation.MetricsCallbackHandler',
            return_value=mock_callback_handler,
        ),
    ):
        result = await framework.evaluate('mock_task', 'mock_agent')

        # Verify metrics were properly transferred to task metrics
        assert len(result.runs) == 1
        run_metrics = result.runs[0].metrics

        # Verify the metrics from the agent's callback were captured
        assert run_metrics['runtime']['value'] == pytest.approx(
            0.5
        )  # From agent's mock duration
        assert run_metrics['token_cost']['value'] == 0.001  # From agent's mock cost
        assert run_metrics['steps']['value'] == 1  # One callback call

        # Verify token metrics from agent's callback
        assert run_metrics['token__completion_tokens']['value'] == 50
        assert run_metrics['token__prompt_tokens']['value'] == 25
        assert run_metrics['token__total_tokens']['value'] == 75

        # Verify original task metrics still work
        assert run_metrics['tokens']['value'] == 100  # From MockTask's validate method
        assert mock_callback_handler.reset_called  # Verify reset was called


@pytest.mark.asyncio
async def test_litellm_metrics_not_used(register_mocks, tmp_path):
    """Test that LiteLLM metrics are not used when agent doesn't use LiteLLM"""
    from calipers.framework import EvaluationFramework

    workspace = tmp_path / 'workspace'
    config = {
        'workspace_dir': str(workspace),
        'agent': {
            'id': 'mock_agent',
            'config': {'uses_litellm': False},
        },
        'tasks': [{'id': 'mock_task'}],
    }

    framework = EvaluationFramework(config)

    with patch('calipers.framework.evaluation.MetricsCallbackHandler') as mock_handler:
        result = await framework.evaluate('mock_task', 'mock_agent')

        # Verify callback was never created
        mock_handler.assert_not_called()

        # Verify basic metrics still work
        assert len(result.runs) == 1
        run_metrics = result.runs[0].metrics
        assert 'tokens' in run_metrics  # From MockTask's validate method


@pytest.mark.asyncio
async def test_litellm_metrics_error_handling(
    register_mocks, tmp_path, mock_callback_handler
):
    """Test error handling when LiteLLM metrics collection fails"""
    from calipers.framework import EvaluationFramework

    workspace = tmp_path / 'workspace'
    config = {
        'workspace_dir': str(workspace),
        'agent': {
            'id': 'mock_agent',
            'config': {'uses_litellm': True},
        },
        'tasks': [{'id': 'mock_task'}],
    }

    framework = EvaluationFramework(config)

    # Create a task instance that will be used throughout the test
    task = MockTask(TaskConfig(id='mock_task', workspace_dir=workspace))
    mock_callback_handler.metrics_tracker = task.metrics

    # Make the mock handler raise an exception
    mock_callback_handler.get_accumulated_metrics = MagicMock(
        side_effect=Exception('Failed to get metrics')
    )

    with (
        patch.object(framework, 'get_task', return_value=task),
        patch(
            'calipers.framework.evaluation.MetricsCallbackHandler',
            return_value=mock_callback_handler,
        ),
    ):
        result = await framework.evaluate('mock_task', 'mock_agent')
        assert len(result.runs) == 1
        assert result.runs[0].success  # Task should still succeed

        # Basic metrics should still be present
        run_metrics = result.runs[0].metrics
        assert 'tokens' in run_metrics  # From MockTask's validate method


@pytest.mark.asyncio
async def test_litellm_metrics_callback_integration(register_mocks, tmp_path):
    """Test that LiteLLM metrics are properly integrated into task metrics through the callback"""
    import litellm

    from calipers.framework import EvaluationFramework

    workspace = tmp_path / 'workspace'
    config = {
        'workspace_dir': str(workspace),
        'agent': {
            'id': 'mock_agent',
            'config': {'uses_litellm': True},
        },
        'tasks': [{'id': 'mock_task'}],
    }

    framework = EvaluationFramework(config)

    # Create a task instance that will be used throughout the test
    task = MockTask(TaskConfig(id='mock_task', workspace_dir=workspace))

    # Clear any existing callbacks
    litellm.callbacks = []

    with patch.object(framework, 'get_task', return_value=task):
        result = await framework.evaluate('mock_task', 'mock_agent')

        # Verify the metrics were properly captured
        assert len(result.runs) == 1
        run_metrics = result.runs[0].metrics

        # Verify LiteLLM metrics were captured
        assert run_metrics['runtime']['value'] == pytest.approx(0.5)
        assert run_metrics['token_cost']['value'] == 0.001
        assert run_metrics['steps']['value'] == 1

        # Verify token metrics
        assert run_metrics['token__completion_tokens']['value'] == 50
        assert run_metrics['token__prompt_tokens']['value'] == 25
        assert run_metrics['token__total_tokens']['value'] == 75

        # Verify original task metrics still work
        assert run_metrics['tokens']['value'] == 100  # From MockTask's validate method


@pytest.fixture
def mock_runtime() -> BaseRuntime:
    runtime = MagicMock(spec=BaseRuntime)
    runtime.execute_action.return_value = {'success': True}
    return cast(BaseRuntime, runtime)


@pytest.fixture
def mock_task(tmp_path: Path) -> BaseEvaluationTask:
    task = MagicMock(spec=BaseEvaluationTask)
    metrics = MetricsTracker()  # Fix: Create actual MetricsTracker
    task.metrics = metrics
    task.workspace_dir = tmp_path
    return cast(BaseEvaluationTask, task)


@pytest.fixture
def mock_agent() -> BaseAgent:
    agent = MagicMock(spec=BaseAgent)
    config = AgentConfig(
        id='test_agent', workspace_dir=Path('test'), model_name='test_model', config={}
    )
    agent.config = config
    return cast(BaseAgent, agent)

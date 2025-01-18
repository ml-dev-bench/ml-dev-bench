from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import litellm

from calipers.callbacks.litellm_callbacks.metrics_callback import (
    MetricsCallbackHandler,
)
from calipers.framework.base import BaseRuntime
from calipers.logger import logger
from runtime.runtime import (
    MLDevBenchRuntimeManager,
    RuntimeBackendType,
    RuntimeEnvironmentType,
)

from .base import BaseAgent, BaseEvaluationTask, EvaluationResult, RunResult
from .config import AgentConfig, EvaluationConfig, TaskConfig
from .registry import EvalRegistry


class EvaluationFramework:
    """
    Evaluation framework for running evaluations on tasks and agents.

    Supports running evaluations on tasks for agents.
        1. use provided config
        2. load tasks and agents from EvalRegistry based on config
        3. evaluate tasks by category filter
        4. run evaluations on tasks and agents
        5. return evaluation results
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize framework with config dictionary"""
        logger.debug('Initializing EvaluationFramework with config: %s', config)
        self.raw_config: Dict[str, Any] = config
        self.config = EvaluationConfig.from_dict(config)

    def get_task(
        self, task_id: str, task_config: Dict[str, Any]
    ) -> Optional[BaseEvaluationTask]:
        """Get task instance from registry with config"""
        task_class = EvalRegistry.get_task(task_id)
        if not task_class:
            logger.error('Task %s not found in registry', task_id)
            return None

        # Create TaskConfig from dict
        config = TaskConfig(
            id=task_id,
            workspace_dir=Path(
                task_config.get('workspace_dir', self.config.workspace_dir)
            ),
            categories=set(task_config.get('categories', [])),
            config=task_config,
        )

        return task_class(config)

    def get_agent(
        self, agent_id: str, agent_config: Dict[str, Any]
    ) -> Optional[BaseAgent]:
        """Get agent instance from registry with config"""
        agent_class = EvalRegistry.get_agent(agent_id)
        if not agent_class:
            logger.error('Agent %s not found in registry', agent_id)
            return None

        # Create AgentConfig from dict
        config = AgentConfig(
            id=agent_id,
            workspace_dir=Path(
                agent_config.get('workspace_dir', self.config.workspace_dir)
            ),
            model_name=agent_config.get('model_name'),
            config=agent_config,
        )

        return agent_class(config)

    def get_tasks_by_category(self, category: str) -> Set[str]:
        """Get task IDs that belong to a category"""
        tasks: Set[str] = set()
        for task_id, task_class in EvalRegistry.get_all_tasks().items():
            if category in task_class.categories:
                tasks.add(task_id)
        return tasks

    def get_eval_runtime(self) -> BaseRuntime:
        """Get MLDevBench runtime for evaluation"""
        runtime_manager = MLDevBenchRuntimeManager(
            backend_type=RuntimeBackendType.COMPOSIO
        )
        runtime_context = runtime_manager.get_runtime(
            runtime_type=RuntimeEnvironmentType.LOCAL, config=None
        )
        return runtime_context.runtime

    def get_tasks_by_category_filter(
        self, category_filter: List[List[str]]
    ) -> set[str]:
        """Get task IDs that match the category filter"""
        matching_tasks = set()
        # if category_filter is all, return all tasks
        if category_filter == ['all']:
            return set(EvalRegistry.get_all_tasks().keys())

        for category_group in category_filter:
            tasks_in_group = None

            for category in category_group:
                tasks_with_category = self.get_tasks_by_category(category)

                if tasks_in_group is None:
                    tasks_in_group = tasks_with_category
                else:
                    # Intersect with previous categories (AND condition)
                    tasks_in_group &= tasks_with_category

            if tasks_in_group:
                # Union with other groups (OR condition)
                matching_tasks.update(tasks_in_group)

        return matching_tasks

    def get_all_categories(self) -> set[str]:
        """Get all available categories"""
        categories = set()
        for task_class in EvalRegistry.get_all_tasks().values():
            categories.update(task_class.categories)
        return categories

    async def evaluate(
        self, task_id: str, agent_id: str, num_runs: int = 1
    ) -> EvaluationResult:
        """Run evaluation multiple times and collect results"""
        task = self.get_task(task_id, {})  # Empty dict as fallback config
        agent = self.get_agent(agent_id, {})  # Empty dict as fallback config
        metrics_callback = None

        if not task or not agent:
            logger.error('Task %s or Agent %s not found', task_id, agent_id)
            raise ValueError(f'Task {task_id} or Agent {agent_id} not found')

        if agent.uses_litellm():
            # Setup MetricsCallback if agent uses LiteLLM
            try:
                metrics_callback = MetricsCallbackHandler(metrics_tracker=task.metrics)
                litellm.callbacks = [metrics_callback]
            except Exception as e:
                logger.warning(f'Failed to create MetricsCallbackHandler: {e}')
                # Continue without metrics callback
                metrics_callback = None

        runs = []
        for run_idx in range(num_runs):
            logger.info(
                'Starting run %d/%d for task %s with agent %s',
                run_idx + 1,
                num_runs,
                task_id,
                agent_id,
            )

            if run_idx == 0:
                task.initialize()

            try:
                evaluation_runtime = self.get_eval_runtime()
                start_time = datetime.now()
                logger.info(f'Running task {task_id} with agent {agent_id}')
                agent_output = await task.run(agent)
                logger.info(f'Validating task {task_id}')
                validation_results = await task.validate(
                    agent_output, evaluation_runtime
                )
                end_time = datetime.now()

                metrics = task.metrics.get_all()
                run_result = RunResult(
                    success=validation_results.get('success', False),
                    validation_details=validation_results,
                    metrics=metrics,
                    start_time=start_time,
                    end_time=end_time,
                )

                # Reset metrics for next run
                task.metrics.reset()
                if metrics_callback is not None:
                    metrics_callback.reset()
                task.cleanup_task()

            except Exception as e:
                end_time = datetime.now()
                logger.exception(
                    'Error during run %d of task %s: %s', run_idx + 1, task_id, str(e)
                )
                run_result = RunResult(
                    success=False,
                    validation_details={},
                    metrics={},
                    start_time=start_time,
                    end_time=end_time,
                    error=str(e),
                )
            finally:
                task.cleanup_task()

            runs.append(run_result)

        result = EvaluationResult(
            task_id=task_id,
            agent_id=agent_id,
            categories=task.categories,
            runs=runs,
        )

        logger.info(
            'Completed %d runs for task %s. Success rate: %.2f, Avg duration: %.2fs',
            num_runs,
            task_id,
            result.success_rate,
            result.avg_duration,
        )
        return result

    async def evaluate_by_category_filter(
        self,
        category_filter: List[List[str]],
        agent_id: str,
        num_runs: int = 1,
        fail_fast: bool = False,
    ) -> List[EvaluationResult]:
        """Run evaluations for tasks matching category filter"""
        results = []
        task_ids = self.get_tasks_by_category_filter(category_filter)

        for task_id in task_ids:
            result = await self.evaluate(task_id, agent_id, num_runs)
            results.append(result)

            if fail_fast and result.success_rate == 0:
                break

        return results

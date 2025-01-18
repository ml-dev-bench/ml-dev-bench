from typing import Dict, Optional, Type

from .base import BaseAgent, BaseEvaluationTask


class EvalRegistry:
    """Registry for evaluation tasks and agents"""

    _tasks: Dict[str, Type[BaseEvaluationTask]] = {}
    _agents: Dict[str, Type[BaseAgent]] = {}

    @classmethod
    def register_task(
        cls, task_class: Type[BaseEvaluationTask]
    ) -> Type[BaseEvaluationTask]:
        """Register an evaluation task class"""
        cls._tasks[task_class.task_id] = task_class
        return task_class

    @classmethod
    def register_agent(cls, agent_class: Type[BaseAgent]) -> Type[BaseAgent]:
        """Register an agent class"""
        cls._agents[agent_class.agent_id] = agent_class
        return agent_class

    @classmethod
    def get_task(cls, task_id: str) -> Optional[Type[BaseEvaluationTask]]:
        return cls._tasks.get(task_id)

    @classmethod
    def get_agent(cls, agent_id: str) -> Optional[Type[BaseAgent]]:
        return cls._agents.get(agent_id)

    @classmethod
    def get_all_tasks(cls) -> Dict[str, Type[BaseEvaluationTask]]:
        return cls._tasks.copy()

    @classmethod
    def get_all_agents(cls) -> Dict[str, Type[BaseAgent]]:
        return cls._agents.copy()

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore[import-untyped]


@dataclass
class TaskConfig:
    """Configuration for a task"""

    id: str
    workspace_dir: Optional[Path] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Configuration for an agent"""

    id: str
    workspace_dir: Optional[Path] = None
    model_name: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def replace_workspace_dir(self, workspace_dir: Path) -> 'AgentConfig':
        """Replace the workspace_dir with a new value"""
        return AgentConfig(
            id=self.id,
            workspace_dir=workspace_dir,
            model_name=self.model_name,
            config=self.config,
        )


@dataclass
class EvaluationConfig:
    """Configuration for evaluation framework"""

    workspace_dir: Path
    agent: AgentConfig
    default_agent: Optional[AgentConfig] = None
    tasks: List[TaskConfig] = field(default_factory=list)
    num_runs: int = 1
    fail_fast: bool = False
    log_level: str = 'INFO'
    category_filters: Optional[List[List[str]]] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'EvaluationConfig':
        """Create config from dictionary"""
        workspace = Path(data.get('workspace_dir', ''))

        # Parse agent config
        agent_data = data.get('agent', {})
        agent = AgentConfig(
            id=agent_data.get('id'),
            workspace_dir=Path(agent_data.get('workspace_dir', workspace)),
            model_name=agent_data.get('model_name'),
            config=agent_data,
        )

        # Parse default agent if exists
        default_agent = None
        if 'default_agent' in data:
            default_data = data['default_agent']
            default_agent = AgentConfig(
                id=default_data.get('id'),
                workspace_dir=Path(default_data.get('workspace_dir', workspace)),
                model_name=default_data.get('model_name'),
                config=default_data,
            )

        # Parse task configs
        tasks = []
        for task_data in data.get('tasks', []):
            task = TaskConfig(
                id=task_data.get('id'),
                workspace_dir=Path(task_data.get('workspace_dir', workspace)),
                config=task_data,
            )
            tasks.append(task)

        return cls(
            workspace_dir=workspace,
            agent=agent,
            default_agent=default_agent,
            tasks=tasks,
            num_runs=data.get('num_runs', 1),
            fail_fast=data.get('fail_fast', False),
            log_level=data.get('log_level', 'INFO'),
            category_filters=data.get('category_filters'),
        )

    @classmethod
    def from_yaml(cls, path: str) -> 'EvaluationConfig':
        """Load config from YAML file"""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

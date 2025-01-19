"""Simple AIDE agent implementation."""

import os
from typing import Any, Dict

from agents.utils import AgentConfig, BaseAgent

try:
    import aide
except ImportError:
    raise ImportError(
        'AIDE agent requires aideml package. Please install it with: poetry install --with aide'
    )


class AIDEAgent(BaseAgent):
    """AIDE agent that uses WecoAI/aideml for ML tasks."""

    def __init__(self, config: AgentConfig):
        """Initialize AIDE agent with config."""
        super().__init__(config)
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError('OPENAI_API_KEY environment variable must be set')

    async def run(self, task: str) -> Dict[str, Any]:
        """Run AIDE agent on the given task.

        Args:
            task: Task description string

        Returns:
            Dict containing:
                - code: Generated solution code
                - metric: Validation metric value
                - success: Whether the solution was successful
        """
        # Create AIDE experiment with task description
        exp = aide.Experiment(
            data_dir=self.config.get('data_dir', '.'),  # Default to current directory
            goal=task,  # Use task description as goal
            eval=self.config.get('eval_metric', ''),  # Optional evaluation metric
        )

        # Run AIDE and get best solution
        best_solution = exp.run(steps=self.config.get('steps', 10))

        return {
            'code': best_solution.code,
            'metric': best_solution.valid_metric,
            'success': True if best_solution.valid_metric else False,
        }

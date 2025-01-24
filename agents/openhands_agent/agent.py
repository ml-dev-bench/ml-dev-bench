"""OpenHands agent implementation."""

import asyncio
from typing import Any, Dict

from openhands.controller.state.state import State
from openhands.core.config import AppConfig, LLMConfig
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import MessageAction

from agents.openhands_agent.utils import codeact_user_response
from calipers.framework.base import BaseAgent
from calipers.framework.config import AgentConfig
from calipers.framework.registry import EvalRegistry

AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
}

AGENT_CLS_TO_INST_SUFFIX = {
    'CodeActAgent': 'When you think you have completed the task, please finish the interaction using the "finish" tool.\n'
}


@EvalRegistry.register_agent
class OpenHandsAgent(BaseAgent):
    """Agent that uses OpenHands in headless mode."""

    agent_id = 'openhands_agent'

    def __init__(
        self,
        config: AgentConfig,
    ):
        """Initialize the OpenHands agent.

        Args:
            config: The agent configuration
        """
        super().__init__(config)
        self.config = config

    def _get_config(self) -> AppConfig:
        """Get OpenHands configuration using agent config.

        Returns:
            OpenHands AppConfig
        """
        app_config = AppConfig(
            default_agent='CodeActAgent',
            runtime='docker',
            workspace_base=self.config.workspace_dir,
            max_iterations=self.config.config.get('max_iterations', 100),
            **self.config.config.get('openhands', {}),  # Additional OpenHands config
        )
        app_config.set_llm_config(
            name=self.config.model_name,
            value=LLMConfig(
                model=self.config.model_name,
                **self.config.config.get('model_config', {}),
            ),
        )
        return app_config

    async def _run_task(self, task: str) -> Dict[str, Any]:
        """Run a task using OpenHands controller.

        Args:
            task: The task specification

        Returns:
            Task results
        """
        # Set up OpenHands config
        config = self._get_config()

        # Create and connect runtime
        runtime = create_runtime(config)
        await runtime.connect()

        # Create initial message action
        message = MessageAction(content=task)

        # Run the controller
        state: State | None = await run_controller(
            config=config,
            initial_user_action=message,
            runtime=runtime,
        )

        if state is None:
            return {
                'success': False,
                'output': None,
                'error': 'Failed to initialize OpenHands state',
            }

        # Get metrics and history
        metrics = state.metrics.get() if state.metrics else {}
        history = state.history.get_all() if state.history else []

        return {
            'success': True,
            'output': {
                'metrics': metrics,
                'history': history,
            },
            'error': None,
        }

    async def run(self, task: str) -> Dict[str, Any]:
        """Run the OpenHands agent on a task.

        Args:
            task: The task specification

        Returns:
            The result of running the task
        """
        try:
            # Run OpenHands task
            return asyncio.run(self._run_task(task))
        except Exception as e:
            return {
                'success': False,
                'output': None,
                'error': str(e),
            }

    def uses_litellm(self) -> bool:
        return True

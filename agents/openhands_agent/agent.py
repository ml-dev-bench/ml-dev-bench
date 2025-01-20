"""OpenHands agent implementation."""

import asyncio
import os
from typing import Any, Dict

from openhands.controller.state.state import State
from openhands.core.config import AppConfig
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import MessageAction


class OpenHandsAgent:
    """Agent that uses OpenHands in headless mode."""

    def __init__(
        self,
        agent_dir: str = os.getenv('AGENT_DIR', '/home/agent'),
        code_dir: str = os.getenv('CODE_DIR', '/home/code'),
        logs_dir: str = os.getenv('LOGS_DIR', '/home/logs'),
    ):
        """Initialize the OpenHands agent.

        Args:
            agent_dir: Directory where OpenHands is installed
            code_dir: Directory where code should be written
            logs_dir: Directory where logs should be written
        """
        self.agent_dir = agent_dir
        self.code_dir = code_dir
        self.logs_dir = logs_dir

        # Set required environment variables
        os.environ.update(
            {
                'WORKSPACE_BASE': self.code_dir,
                'LOG_ALL_EVENTS': 'true',
                'WORKSPACE_MOUNT_PATH': self.code_dir,
            }
        )

    def _get_config(self, model: str) -> AppConfig:
        """Get OpenHands configuration.

        Args:
            model: The LLM model to use

        Returns:
            OpenHands AppConfig
        """
        return AppConfig(
            default_agent='CodeActAgent',  # Using CodeActAgent as it's the standard agent
            run_as_openhands=False,
            runtime='local',  # Using local runtime since we're in the container
            llm={
                'model': model,
                'modify_params': False,  # Don't modify params for reproducibility
            },
        )

    async def _run_task(
        self, task: Dict[str, Any], model: str, api_key: str
    ) -> Dict[str, Any]:
        """Run a task using OpenHands controller.

        Args:
            task: The task specification
            model: The LLM model to use
            api_key: The API key for the model

        Returns:
            Task results
        """
        # Set up OpenHands config
        config = self._get_config(model)

        # Create and connect runtime
        runtime = create_runtime(config)
        await runtime.connect()

        # Create initial message action
        message = MessageAction(content=task.get('prompt', ''))

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

    def run(self, task: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Run the OpenHands agent on a task.

        Args:
            task: The task specification
            **kwargs: Additional arguments to pass to OpenHands

        Returns:
            The result of running the task
        """
        # Set model and API key from kwargs or environment
        model = kwargs.get(
            'model', os.getenv('LLM_MODEL', 'anthropic/claude-3-5-sonnet-20241022')
        )
        api_key = kwargs.get('api_key', os.getenv('LLM_API_KEY'))

        if not api_key:
            raise ValueError(
                'LLM_API_KEY must be provided either in kwargs or environment'
            )

        # Set OpenHands environment variables
        os.environ.update(
            {
                'LLM_API_KEY': api_key,
                'LLM_MODEL': model,
            }
        )

        try:
            # Run OpenHands task
            return asyncio.run(self._run_task(task, model, api_key))
        except Exception as e:
            return {
                'success': False,
                'output': None,
                'error': str(e),
            }

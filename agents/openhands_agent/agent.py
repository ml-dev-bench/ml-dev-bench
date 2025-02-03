"""OpenHands agent implementation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict

from openhands.controller.state.state import State
from openhands.core.config import AppConfig, LLMConfig, SandboxConfig, finalize_config
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import MessageAction
from openhands.runtime.base import Runtime

from agents.openhands_agent.utils import codeact_user_response
from calipers.framework.base import BaseAgent
from calipers.framework.config import AgentConfig
from calipers.framework.registry import EvalRegistry
from calipers.logger import logger

DEFAULT_RUNTIME_IMAGE = 'ml-dev-bench-runtime:latest'
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_AGENT_CLASS = 'CodeActAgent'


@dataclass
class TaskResult:
    """Result of running a task."""

    success: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AgentOutput(TypedDict):
    """Type definition for agent output."""

    metrics: Dict[str, Any]
    history: list[Any]


@EvalRegistry.register_agent
class OpenHandsAgent(BaseAgent):
    """Agent that uses OpenHands in headless mode."""

    agent_id = 'openhands_agent'

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the OpenHands agent.

        Args:
            config: The agent configuration containing model and runtime settings
        """
        super().__init__(config)
        self.config = config
        self._runtime: Optional[Runtime] = None

    def _create_sandbox_config(self) -> SandboxConfig:
        """Create sandbox configuration for the runtime.

        Returns:
            Configured SandboxConfig instance
        """
        return SandboxConfig(
            enable_auto_lint=False,
            use_host_network=True,
            runtime_container_image=DEFAULT_RUNTIME_IMAGE,
        )

    def _create_llm_config(self) -> LLMConfig:
        """Create LLM configuration from agent config.

        Returns:
            Configured LLMConfig instance
        """
        if not self.config.model_name:
            raise ValueError('Model name must be specified in agent config')

        return LLMConfig(
            model=self.config.model_name, **self.config.config.get('model_config', {})
        )

    def _get_config(self) -> AppConfig:
        """Get OpenHands configuration using agent config.

        Returns:
            Configured AppConfig instance

        Raises:
            ValueError: If required configuration is missing
        """
        if not self.config.workspace_dir:
            raise ValueError('Workspace directory must be specified in agent config')

        app_config = AppConfig(
            default_agent=DEFAULT_AGENT_CLASS,
            run_as_openhands=True,
            runtime='docker',
            workspace_base=str(self.config.workspace_dir),
            max_iterations=self.config.config.get(
                'max_iterations', DEFAULT_MAX_ITERATIONS
            ),
            sandbox=self._create_sandbox_config(),
            **self.config.config.get('openhands', {}),
        )

        app_config.set_llm_config(name='llm', value=self._create_llm_config())
        # Finalize config
        finalize_config(app_config)
        return app_config

    async def _setup_runtime(self) -> Runtime:
        """Set up and initialize the OpenHands runtime.

        Returns:
            Initialized Runtime instance

        Raises:
            RuntimeError: If runtime setup fails
        """
        try:
            runtime = create_runtime(self._get_config())
            await runtime.connect()

            if not runtime.container:
                raise RuntimeError(
                    'Container failed to initialize. Check sandbox configuration.'
                )

            self._runtime = runtime
            return runtime
        except Exception as e:
            logger.error(f'Failed to setup runtime: {str(e)}')
            raise RuntimeError(f'Runtime setup failed: {str(e)}') from e

    def _extract_state_output(self, state: Optional[State]) -> TaskResult:
        """Extract output from OpenHands state.

        Args:
            state: The OpenHands state to extract output from

        Returns:
            TaskResult containing success status and output/error
        """
        if state is None:
            return TaskResult(
                success=False, error='Failed to initialize OpenHands state'
            )

        # Convert MessageAction objects to serializable format
        history = []
        if state.history:
            for action in state.history:
                if isinstance(action, MessageAction):
                    history.append(
                        {
                            'type': 'message',
                            'content': action.content,
                            'role': action.role if hasattr(action, 'role') else None,
                        }
                    )
                else:
                    # For other action types, just store their string representation
                    history.append(str(action))

        return TaskResult(
            success=True,
            output={
                'metrics': state.metrics.get() if state.metrics else {},
                'history': history,
            },
        )

    async def _run_task(self, task: str) -> TaskResult:
        """Run a task using OpenHands controller.

        Args:
            task: The task specification

        Returns:
            TaskResult containing success status and output/error
        """
        try:
            runtime = await self._setup_runtime()
            message = MessageAction(content=task)

            state = await run_controller(
                config=self._get_config(),
                initial_user_action=message,
                runtime=runtime,
                fake_user_response_fn=codeact_user_response,
            )

            return self._extract_state_output(state)

        except Exception as e:
            logger.error('Task execution failed', exc_info=True)
            return TaskResult(success=False, error=str(e))

    async def run(self, task: str) -> Dict[str, Any]:
        """Run the OpenHands agent on a task.

        Args:
            task: The task specification

        Returns:
            Dictionary containing task results with success status and output/error
        """
        result = await self._run_task(task)
        return {
            'success': result.success,
            'output': result.output,
            'error': result.error,
        }

    def uses_litellm(self) -> bool:
        """Whether this agent uses litellm for model calls.

        Returns:
            True as OpenHands uses litellm
        """
        return True

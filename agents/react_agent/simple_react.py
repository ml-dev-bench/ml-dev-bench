from typing import Any, Dict

from composio import Action
from composio_langchain import ComposioToolSet
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from calipers.framework.base import BaseAgent, BaseRuntime
from calipers.framework.config import AgentConfig
from calipers.framework.registry import EvalRegistry
from runtime.environments import LocalConfig, RuntimeConfig
from runtime.runtime import (
    MLDevBenchRuntimeManager,
    RuntimeBackendType,
    RuntimeEnvironmentType,
)
from runtime.tools.filetool.tool import MLDevBenchFileTool  # noqa: F401
from runtime.tools.shelltool.tool import MLDevBenchShellTool  # noqa: F401

from ..utils import create_message_content

AGENT_TOOLS = [
    Action.ML_DEV_BENCH_FILE_TOOL_CREATE_AND_WRITE_FILE,
    Action.ML_DEV_BENCH_FILE_TOOL_GET_DIRECTORY_TREE,
    Action.ML_DEV_BENCH_FILE_TOOL_CWD,
    Action.ML_DEV_BENCH_SHELL_TOOL_EXEC_COMMAND,
    Action.ML_DEV_BENCH_SHELL_TOOL_SPAWN_PROCESS,
    Action.ML_DEV_BENCH_SHELL_TOOL_SLEEP_AND_EXECUTE,
    Action.FILETOOL_LIST_FILES,
    Action.FILETOOL_EDIT_FILE,
]


@EvalRegistry.register_agent
class SimpleReactAgent(BaseAgent):
    """A simple React agent implementation using LangChain."""

    agent_id = 'simple_react'
    description = 'A React agent that uses LangChain for reasoning and action execution'

    def __init__(self, config: AgentConfig):
        """Initialize the React agent with config."""
        super().__init__(config)
        self._runtime = None

        # Initialize LLM
        model_name = config.model_name or config.config.get('model_name', 'gpt-4o-mini')
        temperature = config.config.get('temperature', 0)
        tool_config = config.config.get('tool_config', {})
        self.llm = ChatLiteLLM(model_name=model_name, temperature=temperature)

        # Initialize tools from runtime actions
        runtime_manager = MLDevBenchRuntimeManager(
            backend_type=RuntimeBackendType.COMPOSIO
        )
        runtime_config = RuntimeConfig(
            persistent=True,
            environment={},
            local_config=LocalConfig(
                working_dir=str(self.config.workspace_dir),
                max_tree_items=2,
            ),
        )
        runtime_context = runtime_manager.get_runtime(
            runtime_type=RuntimeEnvironmentType.LOCAL, config=runtime_config
        )
        workspace = runtime_context.runtime
        self.workspace = workspace
        self.tool_set = ComposioToolSet(**tool_config)
        self.tool_set.set_workspace_id(workspace.id)
        self.tools = self.tool_set.get_tools(actions=AGENT_TOOLS)
        system_message = SystemMessage(
            content=create_message_content('You are a helpful assistant.')
        )

        # Create the React agent
        self.agent = create_react_agent(
            self.llm, tools=self.tools, state_modifier=system_message
        )

    def uses_litellm(self) -> bool:
        """Whether this agent uses LiteLLM."""
        return True

    async def run(self, task: str) -> Dict[str, Any]:
        """Run the agent on a task."""
        try:
            # Execute the agent
            result = await self.agent.ainvoke(
                {'messages': [('user', task)]},
                config={
                    'recursion_limit': self.config.config.get('recursion_limit', 30)
                },
            )

            # Extract the final response
            final_response = result.get('output', '')

            return {
                'success': True,
                'response': final_response,
                'intermediate_steps': result.get('intermediate_steps', []),
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
            }

    @property
    def runtime(self) -> BaseRuntime:
        """Get or create the runtime."""
        return self.workspace

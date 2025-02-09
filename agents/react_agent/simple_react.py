from typing import Any, Dict, Sequence

from composio import Action
from composio_langchain import ComposioToolSet
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
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
VERTEX_AI_PROMPT_CACHE_LIMIT = 4


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

        def add_cache_control_to_messages(
            state: Dict[str, Any],
        ) -> Sequence[BaseMessage]:
            """Add cache control to all messages in the state."""
            messages = state['messages']
            modified_messages = []

            # Add system message with cache control
            system_msg = SystemMessage(
                content=create_message_content('You are a helpful assistant.')
            )
            modified_messages.append(system_msg)

            # Process other messages
            for idx, msg in enumerate(messages):
                if isinstance(msg, (tuple, list)):
                    # Convert tuple messages to proper Message objects
                    role, content = msg
                    if role == 'user':
                        msg = HumanMessage(content=content)

                # Add cache control to first messages (since system takes 1 slot)
                if idx < (VERTEX_AI_PROMPT_CACHE_LIMIT - 1):
                    if not isinstance(msg.content, list):
                        msg.content = create_message_content(msg.content)
                else:
                    # For later messages, ensure content is a string
                    if isinstance(msg.content, list):
                        msg.content = msg.content[0]['text']

                modified_messages.append(msg)

            return modified_messages

        # Create the React agent with the state modifier
        self.agent = create_react_agent(
            self.llm, tools=self.tools, state_modifier=add_cache_control_to_messages
        )

    def uses_litellm(self) -> bool:
        """Whether this agent uses LiteLLM."""
        return True

    async def run(self, task: str) -> Dict[str, Any]:
        """Run the agent on a task."""
        try:
            # Execute the agent
            messages = [HumanMessage(content=create_message_content(task))]
            result = await self.agent.ainvoke(
                {'messages': messages},
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

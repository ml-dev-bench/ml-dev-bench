"""Cursor-style agent adapter for ML Dev Bench.

This adapter provides a Cursor-like tool-using agent wired to the
ML Dev Bench runtime via Composio actions, so it can be benchmarked
with the existing evaluation framework.
"""

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

# Tools the agent is allowed to use in the workspace
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
class CursorAgent(BaseAgent):
    """Cursor-style tool-using agent built on LangGraph + LiteLLM backend."""

    agent_id = 'cursor_agent'
    description = (
        'A Cursor-like agent that plans with ReAct and uses Composio tools to '
        'edit files and run shell commands in the workspace'
    )

    def __init__(self, config: AgentConfig):
        super().__init__(config)

        # LLM selection. Use model_name if provided, else fallback to config key
        model_name = config.model_name or config.config.get('model_name', 'gpt-4o-mini')
        temperature = config.config.get('temperature', 0)
        tool_config = config.config.get('tool_config', {})
        self.llm = ChatLiteLLM(model_name=model_name, temperature=temperature)

        # Create a persistent local runtime bound to the task workspace
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

        # Tools exposed to the agent
        self.tool_set = ComposioToolSet(**tool_config)
        self.tool_set.set_workspace_id(workspace.id)
        self.tools = self.tool_set.get_tools(actions=AGENT_TOOLS)

        # System prompt tailored for long-running ML tasks (Cursor-like behavior)
        cursor_system = (
            'You are a Cursor-style coding agent operating inside an ML task workspace. '
            'Follow these rules:\n'
            '- Prefer editing existing files minimally and explain changes in commit-like messages.\n'
            '- For long-running commands (training, pip installs), use SPAWN_PROCESS; do not block.\n'
            '- Periodically check logs/outputs using EXEC_COMMAND with short timeouts.\n'
            '- Assume working directory is the task workspace; do not hardcode absolute paths.\n'
            '- Finish by ensuring any required artifacts (e.g., checkpoints/metrics.json) are produced.\n'
        )

        def add_cursor_style_system_message(
            state: Dict[str, Any],
        ) -> Sequence[BaseMessage]:
            messages = state['messages']
            modified: list[BaseMessage] = []
            modified.append(
                SystemMessage(content=create_message_content(cursor_system))
            )
            for msg in messages:
                # Normalize tuple/list messages into proper message objects
                if isinstance(msg, (tuple, list)):
                    role, content = msg
                    if role == 'user':
                        msg = HumanMessage(content=content)
                # Ensure content format
                if not isinstance(msg.content, list):
                    msg.content = create_message_content(msg.content)
                modified.append(msg)
            return modified

        # Create the ReAct agent with state modifier
        self.agent = create_react_agent(
            self.llm,
            tools=self.tools,
            state_modifier=add_cursor_style_system_message,
        )

    def uses_litellm(self) -> bool:
        # Under the hood ChatLiteLLM uses LiteLLM routes
        return True

    async def run(self, task: str) -> Dict[str, Any]:
        """Execute the agent on a textual task specification."""
        try:
            messages = [HumanMessage(content=create_message_content(task))]
            result = await self.agent.ainvoke(
                {'messages': messages},
                config={
                    'recursion_limit': self.config.config.get('recursion_limit', 60)
                },
            )
            final_response = result.get('output', '')
            return {
                'success': True,
                'response': final_response,
                'intermediate_steps': result.get('intermediate_steps', []),
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    @property
    def runtime(self) -> BaseRuntime:
        # Expose the runtime to validators if needed
        return self.workspace

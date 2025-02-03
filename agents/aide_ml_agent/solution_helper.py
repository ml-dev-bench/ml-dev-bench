"""Solution helper agent that analyzes task descriptions and solutions."""

import asyncio
from typing import Any, Dict

from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatLiteLLM
from litellm import InternalServerError
from pydantic import BaseModel, Field

from calipers.framework.base import BaseAgent
from calipers.framework.config import AgentConfig
from calipers.framework.registry import EvalRegistry


class SolutionAnalysis(BaseModel):
    """Output schema for solution analysis."""

    output_file: str = Field(
        description=('Name of the Python file where the solution should be saved')
    )
    requires_execution: bool = Field(
        description=('Whether the solution needs to be executed to generate output')
    )
    explanation: str = Field(description='Brief explanation of the analysis')


@EvalRegistry.register_agent
class SolutionHelperAgent(BaseAgent):
    """Agent that analyzes task descriptions and solutions."""

    agent_id = 'solution_helper_agent'

    def __init__(self, config: AgentConfig):
        """Initialize solution helper agent with config."""
        super().__init__(config)

        # Initialize LLM
        model_name = config.model_name or config.config.get('model_name', 'gpt-4o-mini')
        temperature = config.config.get('temperature', 0)
        self.llm = ChatLiteLLM(
            model_name=model_name, temperature=temperature
        ).with_structured_output(SolutionAnalysis)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    """You analyze Python code and task descriptions to \
determine:

1. output_file: The name of the Python file where the solution should be saved based on the task description

2. requires_execution: Whether running the code is needed to generate output artifacts (e.g., generating files, computing results)
   - True if code needs to run to generate output artifacts required by the task description
   - False if the code added to the output_file is sufficient to meet the task requirements

3. explanation: Brief reason for your choices
""",
                ),
                (
                    'human',
                    """Analyze this task and solution:

Task Description: {task}

Solution Code: {solution}""",
                ),
            ]
        )

        # Retry configuration
        self.max_retries = config.config.get('max_retries', 3)
        self.initial_retry_delay = config.config.get('initial_retry_delay', 1)

    async def _invoke_with_retry(self, messages: list) -> SolutionAnalysis:
        """Invoke LLM with retry logic for InternalServerError.

        Args:
            messages: List of formatted messages to send to LLM

        Returns:
            SolutionAnalysis object from LLM response

        Raises:
            Exception: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                return self.llm.invoke(messages)
            except InternalServerError:
                if attempt == self.max_retries - 1:
                    raise  # Re-raise on last attempt

                # Exponential backoff
                delay = self.initial_retry_delay * (2**attempt)
                await asyncio.sleep(delay)

    async def run(self, task: str) -> Dict[str, Any]:
        """Analyze the task description and solution.

        Args:
            task: String containing both task description and solution code

        Returns:
            Dict containing analysis results
        """
        # Split input into task description and solution
        parts = task.split('Solution Code:', 1)
        if len(parts) != 2:
            raise ValueError("Input must contain 'Solution Code:' separator")

        task_desc = parts[0].strip()
        solution_code = parts[1].strip()

        # Format prompt and get response
        formatted_prompt = self.prompt.format_messages(
            task=task_desc, solution=solution_code
        )

        # Get response from LLM with retry logic
        analysis = await self._invoke_with_retry(formatted_prompt)

        return {
            'output_file': analysis.output_file,
            'requires_execution': analysis.requires_execution,
            'explanation': analysis.explanation,
        }

    def uses_litellm(self) -> bool:
        return True

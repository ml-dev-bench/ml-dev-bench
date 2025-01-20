"""Simple AIDE agent implementation."""

import os
import subprocess
from typing import Any, Dict

from calipers.framework.base import BaseAgent
from calipers.framework.config import AgentConfig
from calipers.framework.registry import EvalRegistry
from calipers.logger import logger
from runtime.environments.shell_setup import get_poetry_python_path

from .solution_helper import (
    SolutionHelperAgent,
)

try:
    import aide
except ImportError:
    raise ImportError(
        'AIDE agent requires aideml package. '
        'Please install it with: poetry install --with aide'
    )

AIDE_PROMPT_SUFFIX = '\n Note: When creating files, assume they are created in the current directory. Do not specify directory paths unless explicitly mentioned in the task description.'


@EvalRegistry.register_agent
class AIDEAgent(BaseAgent):
    """
    AIDE agent that uses WecoAI/aideml for ML tasks.
    Uses solution helper agent to determine output file and execute the solution.
    """

    agent_id = 'aide_ml_agent'

    def __init__(self, config: AgentConfig):
        """Initialize AIDE agent with config."""
        super().__init__(config)

        # Initialize solution helper agent
        self.solution_helper = SolutionHelperAgent(config)

        # Get Poetry's Python path
        try:
            poetry_bin = get_poetry_python_path()
            self.python_path = os.path.join(poetry_bin, 'python')
            if not os.path.exists(self.python_path):
                raise FileNotFoundError(
                    f'Python interpreter not found at {self.python_path}'
                )
        except Exception as e:
            raise RuntimeError(f'Failed to get Poetry Python path: {str(e)}')

    async def run(self, task: str) -> Dict[str, Any]:
        """Run AIDE agent on the given task.

        Args:
            task: Task description string

        Returns:
            Dict containing:
                - code: Generated solution code
                - metric: Validation metric value
                - success: Whether the solution was successful
                - output_file: Name of file where solution was saved
                - execution_results: Results of running the code (if executed)
        """
        task += AIDE_PROMPT_SUFFIX

        # Get model configurations, falling back to default model_name
        default_model = self.config.config.get('default_model', self.config.model_name)
        code_model = self.config.config.get('code_model', self.config.model_name)
        feedback_model = self.config.config.get(
            'feedback_model', self.config.model_name
        )
        report_model = self.config.config.get('report_model', self.config.model_name)

        # Create AIDE experiment with task description
        exp = aide.Experiment(
            # Default to current directory
            data_dir=str(self.config.workspace_dir),
            # Use task description as goal
            goal=task,
            # Optional evaluation metric
            eval=self.config.config.get('eval_metric', ''),
            # Use configured models
            default_model=default_model,
            code_model=code_model,
            feedback_model=feedback_model,
            report_model=report_model,
        )

        # Run AIDE and get best solution
        best_solution = exp.run(steps=self.config.config.get('steps', 10))
        logger.info(f'working dir files {os.listdir(self.config.workspace_dir)}')

        logger.info('Using solution helper agent to determine output file ...')
        # Use solution helper to determine output file
        helper_input = f'{task}\nSolution Code: {best_solution.code}'
        analysis = await self.solution_helper.run(helper_input)
        output_file = analysis['output_file']
        requires_execution = analysis['requires_execution']

        logger.info(f'Writing solution to {output_file} ...')
        logger.info(f'The solution will be executed: {requires_execution}')
        # Write solution to the determined file
        output_path = os.path.join(self.config.workspace_dir, output_file)
        with open(output_path, 'w') as f:
            f.write(best_solution.code)
        logger.info(f'Solution code:\n {best_solution.code}')

        execution_results = None
        if requires_execution:
            # TODO: Use the eval runtime to run the solution
            logger.info('Executing the solution ...')
            try:
                # Run the Python file using Poetry's Python
                process = subprocess.run(
                    [self.python_path, output_path],
                    capture_output=True,
                    text=True,
                    cwd=self.config.workspace_dir,
                    env={
                        **os.environ,
                        'PYTHONPATH': self.config.workspace_dir,
                    },
                )
                execution_results = {
                    'exit_code': process.returncode,
                    'stdout': process.stdout,
                    'stderr': process.stderr,
                }
                logger.info(f'Execution completed with exit code: {process.returncode}')
                logger.info(f'Execution results: {execution_results}')
            except Exception as e:
                execution_results = {
                    'exit_code': -1,
                    'stdout': '',
                    'stderr': f'Failed to execute: {str(e)}',
                }
                logger.error(f'Failed to execute solution: {e}')

        return {
            'code': best_solution.code,
            'metric': best_solution.valid_metric,
            'success': True if best_solution.valid_metric else False,
            'output_file': output_file,
            'execution_results': execution_results,
        }

    def uses_litellm(self) -> bool:
        return False

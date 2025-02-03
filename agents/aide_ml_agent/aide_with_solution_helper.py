"""Simple AIDE agent implementation."""

import os
import shutil
import subprocess
from pathlib import Path
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

    def _setup_experiment_dirs(self) -> tuple[str, str]:
        """Set up data and experiment directories."""
        data_dir = os.path.join(self.config.workspace_dir, 'data')
        exp_dir = os.path.join(self.config.workspace_dir, 'aide_working_dir')

        if Path(data_dir).exists():
            shutil.rmtree(data_dir)
        if Path(exp_dir).exists():
            shutil.rmtree(exp_dir)

        # Copy data and create experiment directory
        shutil.copytree(self.config.workspace_dir, data_dir)
        os.makedirs(exp_dir, exist_ok=True)

        return data_dir, exp_dir

    def _get_model_configs(self) -> tuple[str, str, str, str]:
        """Get model configurations with fallbacks."""
        default_model = self.config.config.get('default_model', self.config.model_name)
        code_model = self.config.config.get('code_model', self.config.model_name)
        feedback_model = self.config.config.get(
            'feedback_model', self.config.model_name
        )
        report_model = self.config.config.get('report_model', self.config.model_name)
        return default_model, code_model, feedback_model, report_model

    def _copy_output_files(self, output_dir: str) -> None:
        """Copy generated files from output directory to workspace."""
        for file in os.listdir(output_dir):
            src_path = os.path.join(output_dir, file)
            dst_path = os.path.join(self.config.workspace_dir, file)

            if os.path.isfile(src_path):
                shutil.copy(src_path, dst_path)
                logger.info(f'Copied file {file} to {self.config.workspace_dir}')
            else:  # is directory
                # if name of directory is 'working', copy all files in it
                if file == 'working':
                    # check if this is a file or directory and use appropriate shutil function
                    for item in os.listdir(src_path):
                        if os.path.isfile(os.path.join(src_path, item)):
                            shutil.copy(
                                os.path.join(src_path, item),
                                os.path.join(self.config.workspace_dir, item),
                            )
                        else:
                            shutil.copytree(
                                os.path.join(src_path, item),
                                os.path.join(self.config.workspace_dir, item),
                            )
                else:
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                logger.info(f'Copied directory {file} to {self.config.workspace_dir}')

    async def _execute_solution(self, output_path: str) -> Dict[str, Any]:
        """Execute the solution file and return results."""
        try:
            process = subprocess.run(
                [self.python_path, output_path],
                capture_output=True,
                text=True,
                cwd=self.config.workspace_dir,
                env={**os.environ, 'PYTHONPATH': self.config.workspace_dir},
            )
            return {
                'exit_code': process.returncode,
                'stdout': process.stdout,
                'stderr': process.stderr,
            }
        except Exception as e:
            return {
                'exit_code': -1,
                'stdout': '',
                'stderr': f'Failed to execute: {str(e)}',
            }

    async def run(self, task: str) -> Dict[str, Any]:
        """Run AIDE agent on the given task."""
        task += AIDE_PROMPT_SUFFIX

        # Check if output directory already has generated files
        output_dir = os.path.join(
            self.config.workspace_dir, 'aide_working_dir', '2-aide_exp'
        )
        # Set up directories and get model configs
        data_dir, exp_dir = self._setup_experiment_dirs()
        default_model, code_model, feedback_model, report_model = (
            self._get_model_configs()
        )

        # Create and run AIDE experiment
        exp = aide.Experiment(
            data_dir=data_dir,
            goal=task,
            eval=self.config.config.get('eval_metric', ''),
            default_model=default_model,
            code_model=code_model,
            feedback_model=feedback_model,
            report_model=report_model,
            workspace_dir=str(exp_dir),
            exp_name='aide_exp',
            copy_data=False,
        )

        best_solution = exp.run(steps=self.config.config.get('steps', 10))

        # Copy generated files to workspace
        logger.info(f'Experiment directory contents: {os.listdir(exp_dir)}')
        output_dir = os.path.join(exp_dir, '2-aide_exp')
        self._copy_output_files(output_dir)

        # Check if AIDE generated any files
        generated_files = [
            f
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
        ]
        if generated_files:
            logger.info(
                f'AIDE generated files: {generated_files}. Skipping solution helper.'
            )
            return {
                'code': best_solution.code,
                'metric': best_solution.valid_metric,
                'success': True if best_solution.valid_metric else False,
                'output_file': generated_files[0] if generated_files else None,
                'execution_results': None,
                'output_dir': output_dir,
            }

        # If no files were generated, use solution helper
        logger.info('No files generated by AIDE, using solution helper...')
        helper_input = f'{task}\nSolution Code: {best_solution.code}'
        analysis = await self.solution_helper.run(helper_input)

        output_file = analysis['output_file']
        output_path = os.path.join(self.config.workspace_dir, output_file)

        # Write and potentially execute solution
        with open(output_path, 'w') as f:
            f.write(best_solution.code)

        execution_results = None
        if analysis['requires_execution']:
            logger.info('Executing the solution ...')
            execution_results = await self._execute_solution(output_path)
            logger.info(f'Execution completed with results: {execution_results}')

        return {
            'code': best_solution.code,
            'metric': best_solution.valid_metric,
            'success': True if best_solution.valid_metric else False,
            'output_file': output_file,
            'execution_results': execution_results,
        }

    def uses_litellm(self) -> bool:
        return False

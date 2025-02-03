import os
from typing import Dict

from runtime.environments.config import (
    ContainerConfig,
    LocalConfig,
    RuntimeConfig,
)
from runtime.environments.shell_setup import (
    escape_value,
    get_poetry_python_path,
)

from .interface import RuntimeEnvironmentType, RuntimeEnvVarsLoader


class LocalEnvVarsLoader(RuntimeEnvVarsLoader[LocalConfig]):
    """Implementation of RuntimeEnvVarsLoader for local runtime environment"""

    def load_env_vars(
        self,
        user_env_vars: Dict[str, str],
        config: LocalConfig,  #
    ) -> Dict[str, str]:
        """Load environment variables for the local runtime environment"""
        if 'HOME' not in user_env_vars:
            user_env_vars['HOME'] = os.path.expanduser('~')

        # set the path to the poetry python path
        user_env_vars['PATH'] = (
            escape_value(get_poetry_python_path())
            + ':'
            + escape_value(user_env_vars.get('PATH', os.environ.get('PATH', '')))
        )

        if config is not None and config.working_dir is not None:
            # Set workspace env var
            user_env_vars['MLDevBench_WORKSPACE_DIR'] = config.working_dir

            # Add workspace dir to PYTHONPATH
            user_env_vars['PYTHONPATH'] = (
                escape_value(config.working_dir)
                + ':'
                + escape_value(
                    user_env_vars.get('PYTHONPATH', os.environ.get('PYTHONPATH', ''))
                )
            )

        return user_env_vars


class ContainerEnvVarsLoader(RuntimeEnvVarsLoader[ContainerConfig]):
    """Implementation of RuntimeEnvVarsLoader for container runtime environment"""

    def load_env_vars(
        self, user_env_vars: Dict[str, str], config: ContainerConfig
    ) -> Dict[str, str]:
        """Load environment variables for the container runtime environment"""
        raise NotImplementedError('ContainerEnvVarsLoader is not implemented')


class MLDevBenchRuntimeEnvVarsLoader(RuntimeEnvVarsLoader[RuntimeConfig]):
    """Implementation of RuntimeEnvVarsLoader for runtime environment"""

    def __init__(self, env_type: RuntimeEnvironmentType):
        self.env_type = env_type

    def load_env_vars(
        self, user_env_vars: Dict[str, str], config: RuntimeConfig
    ) -> Dict[str, str]:
        """Load environment variables for the runtime environment"""
        if self.env_type == RuntimeEnvironmentType.LOCAL:
            if config.local_config is None:
                raise ValueError(
                    'Local config is required for local runtime environment'
                )
            return LocalEnvVarsLoader().load_env_vars(
                user_env_vars, config.local_config
            )
        elif self.env_type == RuntimeEnvironmentType.DOCKER:
            if config.container_config is None:
                raise ValueError(
                    'Container config is required for docker runtime environment'
                )
            return ContainerEnvVarsLoader().load_env_vars(
                user_env_vars, config.container_config
            )
        else:
            raise ValueError('Invalid runtime configuration')

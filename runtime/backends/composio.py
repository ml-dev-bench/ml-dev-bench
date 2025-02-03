import logging
from typing import Dict, Optional

from composio import Action, WorkspaceFactory, WorkspaceType
from composio.tools.env.base import Workspace, WorkspaceConfigType

from runtime.backends.interface import (
    RuntimeBackendFactory,
    RuntimeBackendType,
    RuntimeContext,
    RuntimeFactory,
)
from runtime.environments import (
    MLDevBenchRuntimeEnvVarsLoader,
    RuntimeConfig,
    RuntimeEnvironmentType,
)
from runtime.tools.shelltool.tool import MLDevBenchShellTool  # noqa: F401

from .interface import runtime_backend_registry

logger = logging.getLogger(__name__)


class ComposioWorkspaceFactory(RuntimeFactory):
    """Factory for creating Composio workspaces"""

    def __init__(
        self,
        composio_api_key: Optional[str] = None,
        composio_base_url: Optional[str] = 'https://backend.composio.dev/api',
        github_access_token: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        persistent: bool = True,
    ):
        self.composio_api_key = composio_api_key
        self.composio_base_url = composio_base_url
        self.github_access_token = github_access_token
        self.environment = environment or {}
        self.persistent = persistent

    def get_runtime(
        self,
        runtime_type: RuntimeEnvironmentType,
        config: RuntimeConfig,
    ) -> RuntimeContext[Workspace]:
        workspace_config: Optional[WorkspaceConfigType] = None

        # Create a merged environment dictionary safely
        merged_environment = dict(self.environment)  # Create a copy of base environment
        if config.environment is not None:
            merged_environment.update(config.environment)

        env_vars_loader = MLDevBenchRuntimeEnvVarsLoader(runtime_type)
        merged_environment = env_vars_loader.load_env_vars(merged_environment, config)

        if runtime_type == RuntimeEnvironmentType.DOCKER:
            if config.container_config is None:
                raise ValueError('container_config is required for Docker runtime')

            workspace_config = WorkspaceType.Docker(
                image=config.container_config.image,
                composio_api_key=self.composio_api_key,
                composio_base_url=self.composio_base_url,
                github_access_token=self.github_access_token,
                environment=merged_environment,
                persistent=(
                    config.persistent
                    if config.persistent is not None
                    else self.persistent
                ),
                ports=config.container_config.ports,
                volumes=config.container_config.volumes,
            )
        elif runtime_type == RuntimeEnvironmentType.E2B:
            if config.e2b_config is None:
                raise ValueError('e2b_config is required for E2B runtime')

            workspace_config = WorkspaceType.E2B(
                composio_api_key=self.composio_api_key,
                composio_base_url=self.composio_base_url,
                github_access_token=self.github_access_token,
                environment=merged_environment,
                persistent=(
                    config.persistent
                    if config.persistent is not None
                    else self.persistent
                ),
                ports=config.e2b_config.ports,
                template=config.e2b_config.template,
            )
        elif runtime_type == RuntimeEnvironmentType.LOCAL:
            if config.local_config is None:
                raise ValueError('local_config is required for Local runtime')

            workspace_config = WorkspaceType.Host(
                composio_api_key=self.composio_api_key,
                composio_base_url=self.composio_base_url,
                github_access_token=self.github_access_token,
                environment=merged_environment,
                persistent=(
                    config.persistent
                    if config.persistent is not None
                    else self.persistent
                ),
            )
        else:
            raise ValueError(f'Unsupported runtime type: {runtime_type}')

        workspace = WorkspaceFactory.new(config=workspace_config)

        # set the working directory for local runtime
        if (
            runtime_type == RuntimeEnvironmentType.LOCAL
            and config.local_config
            and config.local_config.working_dir
        ):
            logger.info(
                f'Setting up working directory: {config.local_config.working_dir}'
            )
            # change the working directory
            out = workspace.execute_action(
                action=Action.FILETOOL_CHANGE_WORKING_DIRECTORY,
                request_data={
                    'path': config.local_config.working_dir,
                },
                metadata={},
            )
            if out['successful']:
                logger.debug(
                    f'Changed filetool working directory to {config.local_config.working_dir}'
                )

            # setup the shell environment
            out = workspace.execute_action(
                action=Action.ML_DEV_BENCH_SHELL_TOOL_EXEC_COMMAND,
                request_data={'cmd': f'cd {config.local_config.working_dir}'},
                metadata={},
            )
            if out['successful']:
                logger.info(
                    f'Shell setup complete, current working dir: {out["data"]["current_shell_pwd"]}'
                )
            else:
                logging.error(f'Ran into error while setting up shell: {out}')

        return RuntimeContext(runtime=workspace)


@runtime_backend_registry.register(RuntimeBackendType.COMPOSIO)
class ComposioBackendFactory(RuntimeBackendFactory):
    """Factory for creating Composio runtime backend instances"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @property
    def backend(self) -> RuntimeBackendType:
        return RuntimeBackendType.COMPOSIO

    def get_factory(self) -> ComposioWorkspaceFactory:
        """Returns a configured ComposioWorkspaceFactory instance"""
        return ComposioWorkspaceFactory(**self.kwargs)

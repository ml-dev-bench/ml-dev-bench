import logging
from typing import Any, Dict, Optional

from runtime.backends import (
    RuntimeBackendType,
    RuntimeFactory,
    runtime_backend_registry,
)
from runtime.backends.interface import RuntimeContext
from runtime.environments import RuntimeConfig, RuntimeEnvironmentType
from runtime.utils.env_loader import DotEnvLoader, EnvVarsLoader


class MLDevBenchRuntimeManager:
    """Runtime manager for creating and managing runtime environments"""

    def __init__(
        self,
        backend_type: RuntimeBackendType,
        backend_config: Optional[Dict[str, Any]] = None,
        env_loader: Optional[EnvVarsLoader] = None,
    ):
        self.backend_type = backend_type
        backend_config = backend_config or {}
        self.backend_factory: RuntimeFactory = runtime_backend_registry.get(
            backend_type
        )(**backend_config).get_factory()
        self.env_loader = env_loader or DotEnvLoader()

    def get_runtime(
        self,
        runtime_type: RuntimeEnvironmentType,
        config: Optional[RuntimeConfig] = None,
    ) -> RuntimeContext[Any]:
        """
        Get a runtime environment instance with environment variables loaded from .env.runtime

        Args:
            runtime_type: Type of runtime environment to create
            config: Optional RuntimeConfig instance. If not provided, one will be created

        Returns:
            Runtime environment instance with environment variables loaded
        """
        config = config or RuntimeConfig()
        config = self._load_runtime_env(config)
        return self.backend_factory.get_runtime(runtime_type, config)

    def _load_runtime_env(self, config: RuntimeConfig) -> RuntimeConfig:
        """Load environment variables and update config"""
        env_vars = self.env_loader.load()
        if env_vars:
            if config.environment is None:
                config.environment = env_vars
            else:
                config.environment.update(env_vars)
        logging.debug(f'Loaded environment variables: {env_vars.keys()}')
        return config

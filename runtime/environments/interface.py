from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Generic, TypeVar

from runtime.environments.config import EnvironmentConfig

# Add this near the top of the file
T = TypeVar('T', bound=EnvironmentConfig)


class RuntimeEnvironmentType(str, Enum):
    """Runtime environment type"""

    DOCKER = 'docker'
    LOCAL = 'local'
    E2B = 'e2b'


class RuntimeEnvVarsLoader(ABC, Generic[T]):
    """Interface for loading environment variables for a runtime environment"""

    @abstractmethod
    def load_env_vars(self, user_env_vars: Dict[str, str], config: T) -> Dict[str, str]:
        """Load environment variables for the runtime environment"""
        pass

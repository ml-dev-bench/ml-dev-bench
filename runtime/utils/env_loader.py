from abc import ABC, abstractmethod
from typing import Dict

from dotenv import dotenv_values, find_dotenv


class EnvVarsLoader(ABC):
    """Base class for environment variable loading strategies"""

    @abstractmethod
    def load(self) -> Dict[str, str]:
        """Load environment variables from a source"""
        pass


class DotEnvLoader(EnvVarsLoader):
    """Load environment variables from a .env file"""

    def __init__(self, env_file: str = '.env.runtime'):
        self.env_file = env_file

    def load(self) -> Dict[str, str]:
        """
        Load environment variables from the specified .env file

        Returns:
            Dictionary of environment variables
        """
        env_path = find_dotenv(self.env_file)
        if not env_path:
            return {}

        return {k: str(v) for k, v in dotenv_values(env_path).items() if v is not None}

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class EnvironmentConfig:
    """Base configuration class for all runtime configurations"""

    def __post_init__(self):
        pass


@dataclass
class ContainerConfig(EnvironmentConfig):
    """Configuration specific to container-based runtimes"""

    image: Optional[str] = None
    ports: Optional[Dict[int, Any]] = None
    volumes: Optional[Dict[str, str]] = None


@dataclass
class E2BConfig(EnvironmentConfig):
    """Configuration specific to E2B runtime"""

    api_key: Optional[str] = None
    template: Optional[str] = None
    ports: Optional[Dict[int, Any]] = None


@dataclass
class LocalConfig(EnvironmentConfig):
    """Configuration specific to local runtime"""

    working_dir: Optional[str] = None
    shell_path: Optional[str] = None
    max_tree_items: Optional[int] = (
        10  # Maximum items to show per directory level in tree output
    )


@dataclass
class RuntimeConfig(EnvironmentConfig):
    """Main configuration class for runtime environments"""

    persistent: Optional[bool] = None
    environment: Optional[Dict[str, str]] = None

    # Environment specific configs
    container_config: Optional[ContainerConfig] = None
    e2b_config: Optional[E2BConfig] = None
    local_config: Optional[LocalConfig] = None

    def __post_init__(self):
        if self.environment is None:
            self.environment = {}

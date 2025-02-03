from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Generic, Optional, TypeVar

from runtime.environments import RuntimeConfig, RuntimeEnvironmentType
from runtime.registry import BaseRegistry


class RuntimeBackendType(str, Enum):
    """Runtime backend type"""

    COMPOSIO = 'composio'


# Generic type for the actual runtime implementation
T = TypeVar('T')


class RuntimeContext(Generic[T]):
    """
    Container for a runtime that includes the runtime instance and additional context.
    """

    def __init__(
        self,
        runtime: T,
        additional_context: Optional[Dict[str, Any]] = None,
    ):
        self.runtime = runtime
        self.additional_context = additional_context


class RuntimeFactory(ABC):
    """
    Factory which creates runtimes of a given type
    """

    @abstractmethod
    def get_runtime(
        self, runtime_type: RuntimeEnvironmentType, config: RuntimeConfig
    ) -> RuntimeContext[Any]:
        raise NotImplementedError('Backend must implement get_runtime method')


class RuntimeBackendFactory(ABC):
    """
    Factory which creates the runtime factory corresponding to a given backend type
    """

    @property
    @abstractmethod
    def backend(self) -> RuntimeBackendType:
        raise NotImplementedError('Backend must implement backend method')

    @abstractmethod
    def get_factory(self) -> RuntimeFactory:
        raise NotImplementedError('Backend must implement get_factory method')


class RuntimeBackendRegistry(BaseRegistry[RuntimeBackendFactory]):
    """Registry for runtime backends"""

    pass


runtime_backend_registry = RuntimeBackendRegistry()

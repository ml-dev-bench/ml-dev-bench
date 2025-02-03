from typing import Callable, Dict, Generic, List, Type, TypeVar

T = TypeVar('T')


class BaseRegistry(Generic[T]):
    """Base registry class for managing registrations per type T"""

    def __init__(self):
        self._registry: Dict[str, Type[T]] = {}

    def register(self, key: str) -> Callable:
        """Register a class with the given key"""

        def decorator(registered_class: Type[T]) -> Type[T]:
            if key in self._registry:
                raise ValueError(
                    f"Key '{key}' is already registered to {self._registry[key].__name__}"
                )
            self._registry[key] = registered_class
            return registered_class

        return decorator

    def get(self, key: str) -> Type[T]:
        """Get a registered class by key"""
        if key not in self._registry:
            raise ValueError(f'Unknown key: {key}')
        return self._registry[key]

    def list(self) -> List[str]:
        """List all registered keys"""
        return list(self._registry.keys())

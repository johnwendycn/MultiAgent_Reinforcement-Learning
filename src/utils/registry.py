"""Component factory registry for modular MARL architecture."""

from typing import Any, Callable, Dict, Optional, Type


class Registry:
    """A generic key-to-class or key-to-factory registry.

    Example:
        >>> ENV_REGISTRY = Registry("Environments")
        >>> @ENV_REGISTRY.register("academy_3v1")
        ... class Academy3v1Env: ...
    """

    def __init__(self, name: str) -> None:
        """Initializes the registry.

        Args:
            name: Human-readable name for the registry (e.g. 'Environments', 'Algorithms').
        """
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, key: str, allow_override: bool = False) -> Callable[[Any], Any]:
        """Decorator to register a class or function under a unique key.

        Args:
            key: Unique string identifier.
            allow_override: If True, allows overriding previously registered key.

        Returns:
            The decorator function wrapping the target.
        """
        def decorator(target: Any) -> Any:
            if key in self._registry and not allow_override:
                existing = self._registry[key]
                if getattr(existing, "__name__", None) == getattr(target, "__name__", None):
                    self._registry[key] = target
                    return target
                raise KeyError(f"Key '{key}' is already registered in {self.name} registry.")
            self._registry[key] = target
            return target
        return decorator

    def get(self, key: str) -> Any:
        """Retrieves the registered item associated with key.

        Args:
            key: Unique string identifier.

        Returns:
            Registered class, function, or object.

        Raises:
            KeyError: If key is not registered.
        """
        if key not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(f"'{key}' not found in {self.name} registry. Available: {available}")
        return self._registry[key]

    def contains(self, key: str) -> bool:
        """Checks if a key is registered.

        Args:
            key: Unique string identifier.

        Returns:
            True if registered, False otherwise.
        """
        return key in self._registry

    def list_keys(self) -> list:
        """Returns all registered keys."""
        return list(self._registry.keys())


# Global registries
ENV_REGISTRY = Registry("Environments")
ALGO_REGISTRY = Registry("Algorithms")
MODEL_REGISTRY = Registry("Models")
FEATURE_REGISTRY = Registry("Features")
REWARD_REGISTRY = Registry("Rewards")

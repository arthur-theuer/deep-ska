"""Registry for plotting functions."""

from collections.abc import Callable

PLOT_REGISTRY: dict[str, Callable] = {}

def register_plot(name: str) -> Callable:
    """Decorator for registering a plotting function under a string key."""
    def decorator(func: Callable) -> Callable:
        if name in PLOT_REGISTRY:
            raise ValueError(f"Plot type '{name}' already registered.")
        PLOT_REGISTRY[name] = func
        return func
    return decorator

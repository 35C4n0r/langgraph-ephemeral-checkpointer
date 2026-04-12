from langgraph.checkpoint.base import BaseCheckpointSaver

from ._base import Strategy, ThreadTimestamps

__all__ = ["Strategy", "ThreadTimestamps", "detect"]


def detect(checkpointer: BaseCheckpointSaver):
    """Return the most efficient strategy for the given checkpointer type."""
    try:
        from langgraph.checkpoint.memory import InMemorySaver
        if isinstance(checkpointer, InMemorySaver):
            from .memory import MemoryStrategy
            return MemoryStrategy(checkpointer)
    except ImportError:
        pass

    raise TypeError(
        f"No strategy available for {type(checkpointer).__name__}. "
        "Supported backends: InMemorySaver."
    )

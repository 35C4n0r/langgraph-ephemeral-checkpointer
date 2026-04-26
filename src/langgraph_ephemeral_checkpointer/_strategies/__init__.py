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

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        if isinstance(checkpointer, SqliteSaver):
            from .sqlite import SqliteStrategy
            return SqliteStrategy(checkpointer)
    except ImportError:
        pass

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        if isinstance(checkpointer, AsyncSqliteSaver):
            from .sqlite import AsyncSqliteStrategy
            return AsyncSqliteStrategy(checkpointer)
    except ImportError:
        pass

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        if isinstance(checkpointer, PostgresSaver):
            from .postgres import PostgresStrategy
            return PostgresStrategy(checkpointer)
    except ImportError:
        pass

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        if isinstance(checkpointer, AsyncPostgresSaver):
            from .postgres import AsyncPostgresStrategy
            return AsyncPostgresStrategy(checkpointer)
    except ImportError:
        pass

    raise TypeError(
        f"No strategy available for {type(checkpointer).__name__}. "
        "Supported backends: InMemorySaver, SqliteSaver, AsyncSqliteSaver, "
        "PostgresSaver, AsyncPostgresSaver."
    )

"""Advisory lock helpers for multi-instance sweep coordination."""

import hashlib
import logging

logger = logging.getLogger(__name__)

# Deterministic signed int64 key: stable across library versions.
_LOCK_KEY: int = int.from_bytes(
    hashlib.md5(b"langgraph-ephemeral-checkpointer").digest()[:8],
    byteorder="big",
    signed=True,
)

_ACQUIRE_SQL = "SELECT pg_try_advisory_lock(%s) AS acquired"
_RELEASE_SQL = "SELECT pg_advisory_unlock(%s)"


class SyncAdvisoryLock:
    """Advisory lock for PostgresSaver (sync). Uses a sync cursor."""

    def __init__(self, checkpointer) -> None:
        self._checkpointer = checkpointer

    def try_acquire(self) -> bool:
        with self._checkpointer._cursor() as cur:
            cur.execute(_ACQUIRE_SQL, (_LOCK_KEY,))
            return bool(cur.fetchone()["acquired"])

    def release(self) -> None:
        with self._checkpointer._cursor() as cur:
            cur.execute(_RELEASE_SQL, (_LOCK_KEY,))

    async def atry_acquire(self) -> bool:
        return self.try_acquire()

    async def arelease(self) -> None:
        self.release()


class AsyncAdvisoryLock:
    """Advisory lock for AsyncPostgresSaver (async). Uses an async cursor."""

    def __init__(self, checkpointer) -> None:
        self._checkpointer = checkpointer

    def try_acquire(self) -> bool:
        raise NotImplementedError("Use atry_acquire() with AsyncPostgresSaver")

    def release(self) -> None:
        raise NotImplementedError("Use arelease() with AsyncPostgresSaver")

    async def atry_acquire(self) -> bool:
        async with self._checkpointer._cursor() as cur:
            await cur.execute(_ACQUIRE_SQL, (_LOCK_KEY,))
            row = await cur.fetchone()
            return bool(row["acquired"])

    async def arelease(self) -> None:
        async with self._checkpointer._cursor() as cur:
            await cur.execute(_RELEASE_SQL, (_LOCK_KEY,))


# Union type for type annotations in sweeper.py
AdvisoryLock = SyncAdvisoryLock | AsyncAdvisoryLock


def get_advisory_lock(checkpointer, enable: bool) -> SyncAdvisoryLock | AsyncAdvisoryLock | None:
    """Return an advisory lock for the given checkpointer, or None.

    Args:
        checkpointer: The LangGraph checkpointer to lock against.
        enable: If False, always returns None without checking the backend.

    Returns:
        SyncAdvisoryLock for PostgresSaver, AsyncAdvisoryLock for
        AsyncPostgresSaver, or None if the backend is unsupported or
        enable is False.
    """
    if not enable:
        return None

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        if isinstance(checkpointer, PostgresSaver):
            return SyncAdvisoryLock(checkpointer)
    except ImportError:
        pass

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        if isinstance(checkpointer, AsyncPostgresSaver):
            return AsyncAdvisoryLock(checkpointer)
    except ImportError:
        pass

    logger.warning(
        "enable_coordination=True has no effect for %s; "
        "advisory locks require PostgresSaver or AsyncPostgresSaver.",
        type(checkpointer).__name__,
    )
    return None

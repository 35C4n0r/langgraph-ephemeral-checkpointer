"""Advisory lock helpers for multi-instance sweep coordination.

Only PostgreSQL supports advisory locks.  For all other backends
get_advisory_lock() returns None and coordination is skipped.
"""

import hashlib
import logging

logger = logging.getLogger(__name__)

# Deterministic signed int64 key: stable across library versions.
_LOCK_KEY: int = int.from_bytes(
    hashlib.md5(b"langgraph-ephemeral-checkpointer").digest()[:8],
    byteorder="big",
    signed=True,
)

class AdvisoryLock:
    """Thin wrapper around PostgreSQL session-level advisory locks.

    Session-level locks are automatically released when the DB connection
    closes, so a crashed sweeper instance won't permanently block others.
    """

    def __init__(self, checkpointer) -> None:
        self._checkpointer = checkpointer

    def try_acquire(self) -> bool:
        with self._checkpointer._cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s) AS acquired", (_LOCK_KEY,))
            return bool(cur.fetchone()["acquired"])

    def release(self) -> None:
        with self._checkpointer._cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))

    async def atry_acquire(self) -> bool:
        async with self._checkpointer._cursor() as cur:
            await cur.execute(
                "SELECT pg_try_advisory_lock(%s) AS acquired", (_LOCK_KEY,)
            )
            row = await cur.fetchone()
            return bool(row["acquired"])

    async def arelease(self) -> None:
        async with self._checkpointer._cursor() as cur:
            await cur.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))

def get_advisory_lock(checkpointer, enable: bool) -> AdvisoryLock | None:
    """Return an AdvisoryLock if the backend supports it and enable=True, else None."""
    if not enable:
        return None

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        if isinstance(checkpointer, PostgresSaver):
            return AdvisoryLock(checkpointer)
    except ImportError:
        pass

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        if isinstance(checkpointer, AsyncPostgresSaver):
            return AdvisoryLock(checkpointer)
    except ImportError:
        pass

    logger.warning(
        "enable_coordination=True has no effect for %s; "
        "advisory locks require PostgresSaver or AsyncPostgresSaver.",
        type(checkpointer).__name__,
    )
    return None

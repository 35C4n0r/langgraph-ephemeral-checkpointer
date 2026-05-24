
import logging

from ._base import Strategy, ThreadTimestamps

logger = logging.getLogger(__name__)


def _parse_rows(rows) -> dict[str, ThreadTimestamps]:
    return {
        row["thread_id"]: ThreadTimestamps(
            latest_id=str(row["latest_id"]),
            earliest_id=str(row["earliest_id"]),
        )
        for row in rows
    }


class PostgresStrategy(Strategy):
    """Optimised strategy for PostgresSaver."""

    COLLECT_ALL = """
        SELECT thread_id, MIN(checkpoint_id) AS earliest_id, MAX(checkpoint_id) AS latest_id
        FROM checkpoints
        GROUP BY thread_id
    """
    # Postgres stores checkpoint_id as uuid; uuid byte-order matches temporal
    # order for UUIDv6, so > comparison is correct.
    COLLECT_SINCE = """
        SELECT thread_id, MIN(checkpoint_id) AS earliest_id, MAX(checkpoint_id) AS latest_id
        FROM checkpoints
        WHERE checkpoint_id > %s
        GROUP BY thread_id
    """
    BATCH_DELETE_WRITES = "DELETE FROM checkpoint_writes WHERE thread_id = ANY(%s)"
    BATCH_DELETE_BLOBS = "DELETE FROM checkpoint_blobs WHERE thread_id = ANY(%s)"
    BATCH_DELETE_CHECKPOINTS = "DELETE FROM checkpoints WHERE thread_id = ANY(%s)"

    def __init__(self, checkpointer) -> None:
        self._checkpointer = checkpointer

    def collect(
            self, cursor: str | None
    ) -> tuple[dict[str, ThreadTimestamps], str | None]:
        with self._checkpointer._cursor() as cur:
            if cursor is None:
                cur.execute(self.COLLECT_ALL)
            else:
                cur.execute(self.COLLECT_SINCE, (cursor,))
            rows = cur.fetchall()
        threads = _parse_rows(rows)
        new_cursor = max((ts.latest_id for ts in threads.values()), default=None)
        return threads, new_cursor

    async def acollect(
            self, cursor: str | None
    ) -> tuple[dict[str, ThreadTimestamps], str | None]:
        return self.collect(cursor)

    def batch_delete(self, thread_ids: list[str], checkpointer) -> None:
        if not thread_ids:
            return
        with self._checkpointer._cursor() as cur:
            cur.execute(self.BATCH_DELETE_WRITES, (thread_ids,))
            cur.execute(self.BATCH_DELETE_BLOBS, (thread_ids,))
            cur.execute(self.BATCH_DELETE_CHECKPOINTS, (thread_ids,))


class AsyncPostgresStrategy(Strategy):
    """Optimised strategy for AsyncPostgresSaver."""

    COLLECT_ALL = PostgresStrategy.COLLECT_ALL
    COLLECT_SINCE = PostgresStrategy.COLLECT_SINCE
    BATCH_DELETE_WRITES = PostgresStrategy.BATCH_DELETE_WRITES
    BATCH_DELETE_BLOBS = PostgresStrategy.BATCH_DELETE_BLOBS
    BATCH_DELETE_CHECKPOINTS = PostgresStrategy.BATCH_DELETE_CHECKPOINTS

    def __init__(self, checkpointer) -> None:
        self._checkpointer = checkpointer

    def collect(self, cursor: str | None) -> tuple[dict[str, ThreadTimestamps], str | None]:
        raise NotImplementedError("Use acollect() with AsyncPostgresSaver")

    async def acollect(
            self, cursor: str | None
    ) -> tuple[dict[str, ThreadTimestamps], str | None]:
        async with self._checkpointer._cursor() as cur:
            if cursor is None:
                await cur.execute(self.COLLECT_ALL)
            else:
                await cur.execute(self.COLLECT_SINCE, (cursor,))
            rows = await cur.fetchall()
        threads = _parse_rows(rows)
        new_cursor = max((ts.latest_id for ts in threads.values()), default=None)
        return threads, new_cursor

    async def abatch_delete(self, thread_ids: list[str], checkpointer) -> None:
        if not thread_ids:
            return
        async with self._checkpointer._cursor() as cur:
            await cur.execute(self.BATCH_DELETE_WRITES, (thread_ids,))
            await cur.execute(self.BATCH_DELETE_BLOBS, (thread_ids,))
            await cur.execute(self.BATCH_DELETE_CHECKPOINTS, (thread_ids,))

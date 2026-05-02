
import logging

from ._base import Strategy, ThreadTimestamps

logger = logging.getLogger(__name__)


def _parse_rows(rows) -> dict[str, ThreadTimestamps]:
    return {
        thread_id: ThreadTimestamps(
            latest_id=latest_id,
            earliest_id=earliest_id,
        )
        for thread_id, earliest_id, latest_id in rows
    }


class SqliteStrategy(Strategy):
    """Optimised strategy for SqliteSaver."""

    COLLECT_ALL = """
        SELECT thread_id, MIN(checkpoint_id) AS earliest_id, MAX(checkpoint_id) AS latest_id
        FROM checkpoints
        GROUP BY thread_id
    """
    # UUIDv6 strings are lexicographically ordered by time, so > works correctly.
    COLLECT_SINCE = """
        SELECT thread_id, MIN(checkpoint_id) AS earliest_id, MAX(checkpoint_id) AS latest_id
        FROM checkpoints
        WHERE checkpoint_id > ?
        GROUP BY thread_id
    """
    BATCH_DELETE_CHECKPOINTS = "DELETE FROM checkpoints WHERE thread_id IN ({placeholders})"
    BATCH_DELETE_WRITES = "DELETE FROM writes WHERE thread_id IN ({placeholders})"

    def __init__(self, checkpointer) -> None:
        self._checkpointer = checkpointer

    def collect(
            self, cursor: str | None
    ) -> tuple[dict[str, ThreadTimestamps], str | None]:
        with self._checkpointer.cursor(transaction=False) as cur:
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
        placeholders = ",".join("?" * len(thread_ids))
        with self._checkpointer.cursor(transaction=True) as cur:
            cur.execute(self.BATCH_DELETE_CHECKPOINTS.format(placeholders=placeholders), thread_ids)
            cur.execute(self.BATCH_DELETE_WRITES.format(placeholders=placeholders), thread_ids)


class AsyncSqliteStrategy(Strategy):
    """Optimised strategy for AsyncSqliteSaver."""

    COLLECT_ALL = SqliteStrategy.COLLECT_ALL
    COLLECT_SINCE = SqliteStrategy.COLLECT_SINCE
    BATCH_DELETE_CHECKPOINTS = SqliteStrategy.BATCH_DELETE_CHECKPOINTS
    BATCH_DELETE_WRITES = SqliteStrategy.BATCH_DELETE_WRITES

    def __init__(self, checkpointer) -> None:
        self._checkpointer = checkpointer

    def collect(self, cursor: str | None) -> tuple[dict[str, ThreadTimestamps], str | None]:
        raise NotImplementedError("Use acollect() with AsyncSqliteSaver")

    async def acollect(
            self, cursor: str | None
    ) -> tuple[dict[str, ThreadTimestamps], str | None]:
        await self._checkpointer.setup()
        async with self._checkpointer.lock:
            if cursor is None:
                async with self._checkpointer.conn.execute(self.COLLECT_ALL) as cur:
                    rows = await cur.fetchall()
            else:
                async with self._checkpointer.conn.execute(
                    self.COLLECT_SINCE, (cursor,)
                ) as cur:
                    rows = await cur.fetchall()
        threads = _parse_rows(rows)
        new_cursor = max((ts.latest_id for ts in threads.values()), default=None)
        return threads, new_cursor

    async def abatch_delete(self, thread_ids: list[str], checkpointer) -> None:
        if not thread_ids:
            return
        placeholders = ",".join("?" * len(thread_ids))
        async with self._checkpointer.lock:
            await self._checkpointer.conn.execute(self.BATCH_DELETE_CHECKPOINTS.format(placeholders=placeholders), thread_ids)
            await self._checkpointer.conn.execute(self.BATCH_DELETE_WRITES.format(placeholders=placeholders), thread_ids)
            await self._checkpointer.conn.commit()

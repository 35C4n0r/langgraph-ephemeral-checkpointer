
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

    def __init__(self, checkpointer) -> None:
        self._checkpointer = checkpointer

    def collect(self) -> dict[str, ThreadTimestamps]:
        with self._checkpointer.cursor(transaction=False) as cur:
            cur.execute(self.COLLECT_ALL)
            rows = cur.fetchall()
        return _parse_rows(rows)

    async def acollect(self) -> dict[str, ThreadTimestamps]:
        return self.collect()


class AsyncSqliteStrategy(Strategy):
    """Optimised strategy for AsyncSqliteSaver."""

    COLLECT_ALL = SqliteStrategy.COLLECT_ALL

    def __init__(self, checkpointer) -> None:
        self._checkpointer = checkpointer

    def collect(self) -> dict[str, ThreadTimestamps]:
        raise NotImplementedError("Use acollect() with AsyncSqliteSaver")

    async def acollect(self) -> dict[str, ThreadTimestamps]:
        await self._checkpointer.setup()
        async with self._checkpointer.lock:
            async with self._checkpointer.conn.execute(self.COLLECT_ALL) as cur:
                rows = await cur.fetchall()
        return _parse_rows(rows)

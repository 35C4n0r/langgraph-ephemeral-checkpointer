
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

    def __init__(self, checkpointer) -> None:
        self._checkpointer = checkpointer

    def collect(self) -> dict[str, ThreadTimestamps]:
        with self._checkpointer._cursor() as cur:
            cur.execute(self.COLLECT_ALL)
            rows = cur.fetchall()
        return _parse_rows(rows)

    async def acollect(self) -> dict[str, ThreadTimestamps]:
        return self.collect()


class AsyncPostgresStrategy(Strategy):
    """Optimised strategy for AsyncPostgresSaver."""

    COLLECT_ALL = PostgresStrategy.COLLECT_ALL

    def __init__(self, checkpointer) -> None:
        self._checkpointer = checkpointer

    def collect(self) -> dict[str, ThreadTimestamps]:
        raise NotImplementedError("Use acollect() with AsyncPostgresSaver")

    async def acollect(self) -> dict[str, ThreadTimestamps]:
        async with self._checkpointer._cursor() as cur:
            await cur.execute(self.COLLECT_ALL)
            rows = await cur.fetchall()
        return _parse_rows(rows)

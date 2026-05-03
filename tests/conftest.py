from datetime import datetime, timedelta, timezone

from langgraph.checkpoint.base import CheckpointTuple

from langgraph_ephemeral_checkpointer._strategies._base import (
    Strategy,
    ThreadTimestamps,
)
from langgraph_ephemeral_checkpointer._uuid6 import unix_to_uuid6


def iso_ts(offset_seconds: float = 0.0) -> str:
    """Return an ISO 8601 timestamp offset_seconds from now."""
    dt = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return dt.isoformat()


def make_checkpoint_tuple(
    thread_id: str,
    checkpoint_id: str | None,
    ts: str,
    checkpoint_ns: str = "",
) -> CheckpointTuple:
    if checkpoint_id is None:
        checkpoint_id = unix_to_uuid6(datetime.fromisoformat(ts).timestamp())
    return CheckpointTuple(
        config={
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        },
        checkpoint={
            "v": 1,
            "id": checkpoint_id,
            "ts": ts,
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "updated_channels": None,
        },
        metadata={},
        parent_config=None,
        pending_writes=None,
    )


class _ListStrategy(Strategy):
    """Test-only strategy: derives ThreadTimestamps from a static list of CheckpointTuples."""

    def __init__(self, tuples: list[CheckpointTuple]) -> None:
        self._tuples = tuples

    def collect(
            self, cursor: str | None
    ) -> tuple[dict[str, ThreadTimestamps], str | None]:
        threads: dict[str, ThreadTimestamps] = {}
        for tup in self._tuples:
            tid = tup.config["configurable"]["thread_id"]
            cp_id = tup.checkpoint["id"]
            if cursor is not None and cp_id <= cursor:
                continue
            if tid not in threads:
                threads[tid] = ThreadTimestamps(latest_id=cp_id, earliest_id=cp_id)
            else:
                entry = threads[tid]
                if cp_id > entry.latest_id:
                    entry.latest_id = cp_id
                if cp_id < entry.earliest_id:
                    entry.earliest_id = cp_id
        return threads, None

    async def acollect(
            self, cursor: str | None
    ) -> tuple[dict[str, ThreadTimestamps], str | None]:
        return self.collect(cursor)


def list_strategy(tuples: list[CheckpointTuple]) -> _ListStrategy:
    return _ListStrategy(tuples)

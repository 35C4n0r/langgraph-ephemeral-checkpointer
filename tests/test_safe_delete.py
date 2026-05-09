"""Tests for compare-and-delete safety."""
from collections.abc import AsyncIterator, Iterator

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple

from langgraph_ephemeral_checkpointer import Sweeper, TTLPolicy

from .conftest import iso_ts, list_strategy, make_checkpoint_tuple

FAKE_CP_ID = "1f1411e3-4e80-62b6-8001-eb2883608248"
OLD_CP_ID = "1e0000e3-4e80-62b6-8001-000000000000"

class MockCP(BaseCheckpointSaver):
    """Mock that returns a fresh tuple on get_tuple() to simulate a race."""

    def __init__(self, list_tuples, get_tuple_override=None):
        super().__init__()
        self._list_tuples = list_tuples
        self._get_override = get_tuple_override
        self.deleted: list[str] = []

    def list(self, config, **kwargs) -> Iterator[CheckpointTuple]:
        return iter(self._list_tuples)

    async def alist(self, config, **kwargs) -> AsyncIterator[CheckpointTuple]:
        for t in self._list_tuples:
            yield t

    def get_tuple(self, config) -> CheckpointTuple | None:
        if self._get_override:
            return self._get_override(config)
        tid = config["configurable"].get("thread_id")
        for t in reversed(self._list_tuples):
            if t.config["configurable"]["thread_id"] == tid:
                return t
        return None

    async def aget_tuple(self, config) -> CheckpointTuple | None:
        return self.get_tuple(config)

    def put(self, config, checkpoint, metadata, new_versions):
        return config

    def put_writes(self, config, writes, task_id, task_path=()):
        pass

    def delete_thread(self, thread_id):
        self.deleted.append(thread_id)

    async def adelete_thread(self, thread_id):
        self.deleted.append(thread_id)

def test_safe_delete_proceeds_when_timestamp_unchanged():
    tuples = [make_checkpoint_tuple("t", None, iso_ts(-3700))]
    cp = MockCP(tuples)
    sweeper = Sweeper(cp, TTLPolicy(idle_ttl_seconds=60), safe_delete=True, _strategy=list_strategy(cp._list_tuples))
    result = sweeper.sweep()

    assert "t" in result.deleted_thread_ids
    assert "t" in cp.deleted

def test_safe_delete_skips_when_new_checkpoint_arrived():
    """Simulate a new checkpoint arriving between the sweep read and the delete."""
    stale_tuple = make_checkpoint_tuple("t", None, iso_ts(-3700))
    fresh_tuple = make_checkpoint_tuple("t", None, iso_ts(-1))

    def get_override(config):
        return fresh_tuple

    cp = MockCP([stale_tuple], get_tuple_override=get_override)
    sweeper = Sweeper(cp, TTLPolicy(idle_ttl_seconds=60), safe_delete=True, _strategy=list_strategy(cp._list_tuples))
    result = sweeper.sweep()

    assert "t" not in result.deleted_thread_ids
    assert "t" not in cp.deleted

def test_safe_delete_skips_already_deleted_thread():
    """If get_tuple returns None (thread already gone), skip deletion gracefully."""
    stale_tuple = make_checkpoint_tuple("t", None, iso_ts(-3700))

    cp = MockCP([stale_tuple], get_tuple_override=lambda _: None)
    sweeper = Sweeper(cp, TTLPolicy(idle_ttl_seconds=60), safe_delete=True, _strategy=list_strategy(cp._list_tuples))
    result = sweeper.sweep()

    assert "t" not in result.deleted_thread_ids
    assert cp.deleted == []

def test_safe_delete_false_deletes_without_recheck():
    stale_tuple = make_checkpoint_tuple("t", None, iso_ts(-3700))
    fresh_tuple = make_checkpoint_tuple("t", None, iso_ts(-1))

    def get_override(config):
        return fresh_tuple

    cp = MockCP([stale_tuple], get_tuple_override=get_override)
    sweeper = Sweeper(cp, TTLPolicy(idle_ttl_seconds=60), safe_delete=False, _strategy=list_strategy(cp._list_tuples))
    result = sweeper.sweep()

    assert "t" in result.deleted_thread_ids
    assert "t" in cp.deleted

@pytest.mark.asyncio
async def test_async_safe_delete_skips_changed_thread():
    stale_tuple = make_checkpoint_tuple("t", None, iso_ts(-3700))
    fresh_tuple = make_checkpoint_tuple("t", None, iso_ts(-1))

    cp = MockCP([stale_tuple], get_tuple_override=lambda _: fresh_tuple)
    sweeper = Sweeper(cp, TTLPolicy(idle_ttl_seconds=60), safe_delete=True, _strategy=list_strategy(cp._list_tuples))
    result = await sweeper.asweep()

    assert "t" not in result.deleted_thread_ids

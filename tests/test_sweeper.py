import asyncio
from collections.abc import AsyncIterator, Iterator
from unittest.mock import patch

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple

from langgraph_ephemeral_checkpointer import Sweeper, TTLPolicy
from langgraph_ephemeral_checkpointer.result import SweepResult

from .conftest import iso_ts, list_strategy, make_checkpoint_tuple


class MockCheckpointer(BaseCheckpointSaver):
    def __init__(self, tuples: list[CheckpointTuple]) -> None:
        super().__init__()
        self._tuples = tuples
        self.deleted: list[str] = []

    def list(self, config, **kwargs) -> Iterator[CheckpointTuple]:
        return iter(self._tuples)

    async def alist(self, config, **kwargs) -> AsyncIterator[CheckpointTuple]:
        for t in self._tuples:
            yield t

    def get_tuple(self, config) -> CheckpointTuple | None:
        tid = config["configurable"].get("thread_id")
        for t in reversed(self._tuples):
            if t.config["configurable"]["thread_id"] == tid:
                return t
        return None

    async def aget_tuple(self, config) -> CheckpointTuple | None:
        return self.get_tuple(config)

    def put(self, config, checkpoint, metadata, new_versions):
        return config

    def put_writes(self, config, writes, task_id, task_path=()):
        pass

    def delete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)

FAKE_CP_ID = "1f1411e3-4e80-62b6-8001-eb2883608248"

def _make_sweeper(tuples, **policy_kwargs):
    cp = MockCheckpointer(tuples)
    policy = TTLPolicy(**policy_kwargs)
    return Sweeper(cp, policy, _strategy=list_strategy(tuples)), cp

def test_sweep_deletes_idle_expired():
    tuples = [
        make_checkpoint_tuple("expired", None, iso_ts(-3700)),
        make_checkpoint_tuple("active", None, iso_ts(-10)),
    ]
    sweeper, cp = _make_sweeper(tuples, idle_ttl_seconds=3600)
    result = sweeper.sweep()

    assert result.deleted_thread_ids == ["expired"]
    assert result.active_thread_count == 1
    assert "expired" in cp.deleted
    assert "active" not in cp.deleted

def test_sweep_keeps_active():
    tuples = [make_checkpoint_tuple("alive", None, iso_ts(-10))]
    sweeper, cp = _make_sweeper(tuples, idle_ttl_seconds=3600)
    result = sweeper.sweep()

    assert result.deleted_thread_ids == []
    assert result.active_thread_count == 1
    assert cp.deleted == []

def test_sweep_hard_age_expired():
    tuples = [
        make_checkpoint_tuple("old", None, iso_ts(-90000)),
        make_checkpoint_tuple("young", None, iso_ts(-100)),
    ]
    sweeper, cp = _make_sweeper(tuples, hard_age_ttl_seconds=86400)
    result = sweeper.sweep()

    assert "old" in result.deleted_thread_ids
    assert "young" not in result.deleted_thread_ids

def test_sweep_or_logic_idle_triggers():
    tuples = [make_checkpoint_tuple("t", None, iso_ts(-100))]
    sweeper, _ = _make_sweeper(tuples, idle_ttl_seconds=60, hard_age_ttl_seconds=86400)
    result = sweeper.sweep()
    assert "t" in result.deleted_thread_ids

def test_sweep_or_logic_hard_age_triggers():
    tuples = [
        make_checkpoint_tuple("t", None, iso_ts(-100000)),
        make_checkpoint_tuple("t", None, iso_ts(-10)),
    ]
    sweeper, _ = _make_sweeper(tuples, idle_ttl_seconds=3600, hard_age_ttl_seconds=86400)
    result = sweeper.sweep()
    assert "t" in result.deleted_thread_ids

def test_sweep_result_fields():
    tuples = [
        make_checkpoint_tuple("e", None, iso_ts(-3700)),
        make_checkpoint_tuple("a1", None, iso_ts(-10)),
        make_checkpoint_tuple("a2", None, iso_ts(-20)),
    ]
    sweeper, _ = _make_sweeper(tuples, idle_ttl_seconds=3600)
    result = sweeper.sweep()

    assert isinstance(result, SweepResult)
    assert result.active_thread_count == 2
    assert result.sweep_duration_seconds >= 0

def test_sweep_empty_checkpointer():
    sweeper, cp = _make_sweeper([], idle_ttl_seconds=60)
    result = sweeper.sweep()
    assert result.deleted_thread_ids == []
    assert result.active_thread_count == 0

@pytest.mark.asyncio
async def test_asweep_same_as_sweep():
    tuples = [
        make_checkpoint_tuple("expired", None, iso_ts(-3700)),
        make_checkpoint_tuple("active", None, iso_ts(-10)),
    ]
    cp = MockCheckpointer(tuples)
    sweeper = Sweeper(cp, TTLPolicy(idle_ttl_seconds=3600), _strategy=list_strategy(tuples))

    result = await sweeper.asweep()
    assert result.deleted_thread_ids == ["expired"]
    assert result.active_thread_count == 1
    assert "expired" in cp.deleted

@pytest.mark.asyncio
async def test_start_stop_lifecycle():
    tuples = [make_checkpoint_tuple("t", None, iso_ts(-10))]
    cp = MockCheckpointer(tuples)
    sweeper = Sweeper(cp, TTLPolicy(idle_ttl_seconds=3600), _strategy=list_strategy(tuples))

    await sweeper.start(interval_seconds=600)
    assert sweeper._task is not None
    assert not sweeper._task.done()

    await asyncio.sleep(0.05)

    await sweeper.stop()
    assert sweeper._task is None

@pytest.mark.asyncio
async def test_start_twice_raises():
    cp = MockCheckpointer([])
    sweeper = Sweeper(cp, TTLPolicy(idle_ttl_seconds=60), _strategy=list_strategy([]))
    await sweeper.start(interval_seconds=600)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            await sweeper.start(interval_seconds=600)
    finally:
        await sweeper.stop()

@pytest.mark.asyncio
async def test_loop_continues_after_exception():
    first_done = asyncio.Event()

    async def maybe_fail(self, *, dry_run=False):
        first_done.set()
        raise RuntimeError("transient failure")

    cp = MockCheckpointer([])
    sweeper = Sweeper(cp, TTLPolicy(idle_ttl_seconds=60), _strategy=list_strategy([]))

    with patch.object(Sweeper, "asweep", maybe_fail):
        await sweeper.start(interval_seconds=600)
        await asyncio.wait_for(first_done.wait(), timeout=2.0)

    assert sweeper._task is not None
    assert not sweeper._task.done()
    await sweeper.stop()


@pytest.mark.asyncio
async def test_loop_logs_exception(caplog):
    import logging

    async def failing_asweep(self, *, dry_run=False):
        raise RuntimeError("database timeout")

    cp = MockCheckpointer([])
    sweeper = Sweeper(cp, TTLPolicy(idle_ttl_seconds=60), _strategy=list_strategy([]))

    with patch.object(Sweeper, "asweep", failing_asweep):
        with caplog.at_level(logging.ERROR, logger="langgraph_ephemeral_checkpointer.sweeper"):
            await sweeper.start(interval_seconds=600)
            await asyncio.sleep(0.05)
            await sweeper.stop()

    assert any("Sweep cycle failed" in r.message for r in caplog.records)

@pytest.mark.asyncio
async def test_stop_before_start_is_noop():
    cp = MockCheckpointer([])
    sweeper = Sweeper(cp, TTLPolicy(idle_ttl_seconds=60), _strategy=list_strategy([]))
    await sweeper.stop()  # should not raise

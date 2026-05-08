"""Tests for on_before_delete and on_sweep_complete callbacks."""
from collections.abc import AsyncIterator, Iterator

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple

from langgraph_ephemeral_checkpointer import Sweeper, TTLPolicy
from langgraph_ephemeral_checkpointer.result import SweepResult

from .conftest import iso_ts, list_strategy, make_checkpoint_tuple

FAKE_CP_ID = "1f1411e3-4e80-62b6-8001-eb2883608248"

class MockCP(BaseCheckpointSaver):
    def __init__(self, tuples):
        super().__init__()
        self._tuples = tuples
        self.deleted: list[str] = []

    def list(self, config, **kwargs) -> Iterator[CheckpointTuple]:
        return iter(self._tuples)

    async def alist(self, config, **kwargs) -> AsyncIterator[CheckpointTuple]:
        for t in self._tuples:
            yield t

    def get_tuple(self, config):
        tid = config["configurable"].get("thread_id")
        for t in reversed(self._tuples):
            if t.config["configurable"]["thread_id"] == tid:
                return t
        return None

    async def aget_tuple(self, config):
        return self.get_tuple(config)

    def put(self, config, checkpoint, metadata, new_versions):
        return config

    def put_writes(self, config, writes, task_id, task_path=()):
        pass

    def delete_thread(self, thread_id):
        self.deleted.append(thread_id)

    async def adelete_thread(self, thread_id):
        self.deleted.append(thread_id)

def _expired(*thread_ids):
    return [make_checkpoint_tuple(tid, FAKE_CP_ID, iso_ts(-3700)) for tid in thread_ids]

def test_on_before_delete_true_proceeds():
    calls: list[tuple] = []

    def on_before(tid, policy, reason):
        calls.append((tid, policy, reason))
        return True

    cp = MockCP(_expired("t"))
    sweeper = Sweeper(
        cp,
        TTLPolicy(idle_ttl_seconds=60),
        on_before_delete=on_before,
        safe_delete=False,
        _strategy=list_strategy(cp._tuples),
    )
    result = sweeper.sweep()

    assert "t" in result.deleted_thread_ids
    assert len(calls) == 1
    tid, policy, reason = calls[0]
    assert tid == "t"
    assert reason == "idle_ttl"

def test_on_before_delete_false_skips():
    cp = MockCP(_expired("t"))
    sweeper = Sweeper(
        cp,
        TTLPolicy(idle_ttl_seconds=60),
        on_before_delete=lambda *_: False,
        safe_delete=False,
        _strategy=list_strategy(cp._tuples),
    )
    result = sweeper.sweep()

    assert "t" not in result.deleted_thread_ids
    assert cp.deleted == []

def test_on_before_delete_raises_propagates():
    def bad_callback(tid, policy, reason):
        raise RuntimeError("callback exploded")

    cp = MockCP(_expired("t"))
    sweeper = Sweeper(
        cp,
        TTLPolicy(idle_ttl_seconds=60),
        on_before_delete=bad_callback,
        safe_delete=False,
        _strategy=list_strategy(cp._tuples),
    )

    with pytest.raises(RuntimeError, match="callback exploded"):
        sweeper.sweep()

def test_on_before_delete_partial_veto():
    vetoed = {"bad"}

    cp = MockCP(_expired("bad", "good"))
    sweeper = Sweeper(
        cp,
        TTLPolicy(idle_ttl_seconds=60),
        on_before_delete=lambda tid, *_: tid not in vetoed,
        safe_delete=False,
        _strategy=list_strategy(cp._tuples),
    )
    result = sweeper.sweep()

    assert "good" in result.deleted_thread_ids
    assert "bad" not in result.deleted_thread_ids

def test_on_sweep_complete_receives_result():
    received: list[SweepResult] = []

    cp = MockCP(_expired("t"))
    sweeper = Sweeper(
        cp,
        TTLPolicy(idle_ttl_seconds=60),
        on_sweep_complete=received.append,
        safe_delete=False,
        _strategy=list_strategy(cp._tuples),
    )
    result = sweeper.sweep()

    assert len(received) == 1
    assert received[0] is result

def test_on_sweep_complete_raises_propagates():
    def bad_complete(result):
        raise RuntimeError("complete exploded")

    cp = MockCP(_expired("t"))
    sweeper = Sweeper(
        cp,
        TTLPolicy(idle_ttl_seconds=60),
        on_sweep_complete=bad_complete,
        safe_delete=False,
        _strategy=list_strategy(cp._tuples),
    )

    with pytest.raises(RuntimeError, match="complete exploded"):
        sweeper.sweep()

@pytest.mark.asyncio
async def test_async_callbacks_fire():
    before_calls: list[str] = []
    after_results: list[SweepResult] = []

    cp = MockCP(_expired("t1", "t2"))
    sweeper = Sweeper(
        cp,
        TTLPolicy(idle_ttl_seconds=60),
        on_before_delete=lambda tid, *_: before_calls.append(tid) or True,
        on_sweep_complete=after_results.append,
        safe_delete=False,
        _strategy=list_strategy(cp._tuples),
    )
    result = await sweeper.asweep()

    assert set(before_calls) == {"t1", "t2"}
    assert len(after_results) == 1
    assert after_results[0] is result

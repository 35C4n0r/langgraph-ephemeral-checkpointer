"""Tests for dry-run mode."""
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

def _make(tuples, **policy_kwargs):
    cp = MockCP(tuples)
    return Sweeper(cp, TTLPolicy(**policy_kwargs), safe_delete=False, _strategy=list_strategy(tuples)), cp

def test_dry_run_no_deletions():
    tuples = [
        make_checkpoint_tuple("e1", None, iso_ts(-3700)),
        make_checkpoint_tuple("e2", None, iso_ts(-7200)),
    ]
    sweeper, cp = _make(tuples, idle_ttl_seconds=60)
    sweeper.sweep(dry_run=True)

    assert cp.deleted == []

def test_dry_run_reports_would_be_deleted():
    tuples = [
        make_checkpoint_tuple("e", None, iso_ts(-3700)),
        make_checkpoint_tuple("a", None, iso_ts(-10)),
    ]
    sweeper, _ = _make(tuples, idle_ttl_seconds=60)
    result = sweeper.sweep(dry_run=True)

    assert "e" in result.deleted_thread_ids
    assert "a" not in result.deleted_thread_ids

def test_dry_run_threads_still_exist():
    from typing import TypedDict

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    class S(TypedDict):
        x: int

    saver = InMemorySaver()
    b = StateGraph(S)  # pyrefly: ignore[bad-specialization]
    b.add_node("n", lambda s: {"x": s["x"] + 1})  # pyrefly: ignore[bad-argument-type]
    b.add_edge(START, "n")
    b.add_edge("n", END)
    graph = b.compile(checkpointer=saver)

    graph.invoke({"x": 0}, {"configurable": {"thread_id": "t1"}})

    import time
    time.sleep(1.1)

    sweeper = Sweeper(saver, TTLPolicy(idle_ttl_seconds=1), safe_delete=False)
    result = sweeper.sweep(dry_run=True)

    assert "t1" in result.deleted_thread_ids
    assert "t1" in saver.storage

def test_dry_run_does_not_call_on_before_delete():
    before_calls: list[str] = []

    tuples = [make_checkpoint_tuple("e", None, iso_ts(-3700))]
    cp = MockCP(tuples)
    sweeper = Sweeper(
        cp,
        TTLPolicy(idle_ttl_seconds=60),
        on_before_delete=lambda tid, *_: before_calls.append(tid) or True,
        safe_delete=False,
        _strategy=list_strategy(tuples),
    )
    sweeper.sweep(dry_run=True)

    assert before_calls == []

def test_dry_run_calls_on_sweep_complete():
    received: list[SweepResult] = []

    tuples = [make_checkpoint_tuple("e", None, iso_ts(-3700))]
    cp = MockCP(tuples)
    sweeper = Sweeper(
        cp,
        TTLPolicy(idle_ttl_seconds=60),
        on_sweep_complete=received.append,
        safe_delete=False,
        _strategy=list_strategy(tuples),
    )
    sweeper.sweep(dry_run=True)

    assert len(received) == 1

@pytest.mark.asyncio
async def test_async_dry_run_no_deletions():
    tuples = [make_checkpoint_tuple("e", None, iso_ts(-3700))]
    sweeper, cp = _make(tuples, idle_ttl_seconds=60)
    result = await sweeper.asweep(dry_run=True)

    assert "e" in result.deleted_thread_ids
    assert cp.deleted == []

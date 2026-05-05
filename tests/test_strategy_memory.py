"""Tests for the InMemorySaver-optimized strategy."""
import time
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from langgraph_ephemeral_checkpointer import Sweeper, TTLPolicy
from langgraph_ephemeral_checkpointer._strategies import detect
from langgraph_ephemeral_checkpointer._strategies.memory import MemoryStrategy
from langgraph_ephemeral_checkpointer._uuid6 import uuid6_to_unix


class _State(TypedDict):
    x: int

def _build_graph(saver: InMemorySaver):
    builder = StateGraph(_State)
    builder.add_node("inc", lambda s: {"x": s["x"] + 1})
    builder.add_edge(START, "inc")
    builder.add_edge("inc", END)
    return builder.compile(checkpointer=saver)

def test_detect_returns_memory_strategy():
    saver = InMemorySaver()
    strategy = detect(saver)
    assert isinstance(strategy, MemoryStrategy)

def test_collect_empty():
    saver = InMemorySaver()
    strategy = MemoryStrategy(saver)
    assert strategy.collect(None)[0] == {}

def test_collect_single_thread():
    saver = InMemorySaver()
    graph = _build_graph(saver)
    graph.invoke({"x": 0}, {"configurable": {"thread_id": "t1"}})

    strategy = MemoryStrategy(saver)
    result, _ = strategy.collect(None)

    assert "t1" in result
    ts = result["t1"]
    assert ts.latest_id >= ts.earliest_id
    assert abs(uuid6_to_unix(ts.latest_id) - time.time()) < 5

def test_collect_multiple_threads():
    saver = InMemorySaver()
    graph = _build_graph(saver)
    graph.invoke({"x": 0}, {"configurable": {"thread_id": "t1"}})
    graph.invoke({"x": 0}, {"configurable": {"thread_id": "t2"}})

    strategy = MemoryStrategy(saver)
    result, _ = strategy.collect(None)

    assert set(result.keys()) == {"t1", "t2"}

def test_collect_multiple_checkpoints():
    """Multiple graph.invoke calls on the same thread -> latest > earliest."""
    saver = InMemorySaver()
    graph = _build_graph(saver)
    graph.invoke({"x": 0}, {"configurable": {"thread_id": "t1"}})
    graph.invoke({"x": 1}, {"configurable": {"thread_id": "t1"}})

    strategy = MemoryStrategy(saver)
    result, _ = strategy.collect(None)

    assert len(result) == 1
    ts = result["t1"]
    assert ts.latest_id >= ts.earliest_id

@pytest.mark.asyncio
async def test_acollect_same_as_collect():
    saver = InMemorySaver()
    graph = _build_graph(saver)
    graph.invoke({"x": 0}, {"configurable": {"thread_id": "t1"}})

    strategy = MemoryStrategy(saver)
    sync_result, _ = strategy.collect(None)
    async_result, _ = await strategy.acollect(None)

    assert set(sync_result.keys()) == set(async_result.keys())

def test_sweep_with_memory_saver_deletes_expired():
    saver = InMemorySaver()
    graph = _build_graph(saver)
    graph.invoke({"x": 0}, {"configurable": {"thread_id": "t1"}})

    policy = TTLPolicy(idle_ttl_seconds=1)
    sweeper = Sweeper(saver, policy)

    result = sweeper.sweep()
    assert "t1" not in result.deleted_thread_ids
    assert "t1" in saver.storage

    time.sleep(1.1)

    result = sweeper.sweep()
    assert "t1" in result.deleted_thread_ids
    assert "t1" not in saver.storage

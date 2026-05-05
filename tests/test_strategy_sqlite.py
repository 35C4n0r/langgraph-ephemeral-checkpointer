"""Tests for the SqliteSaver-optimized strategy."""
import time
from typing import TypedDict

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from langgraph_ephemeral_checkpointer import Sweeper, TTLPolicy
from langgraph_ephemeral_checkpointer._strategies import detect
from langgraph_ephemeral_checkpointer._strategies.sqlite import (
    AsyncSqliteStrategy,
    SqliteStrategy,
)
from langgraph_ephemeral_checkpointer._uuid6 import uuid6_to_unix


class _State(TypedDict):
    x: int

def _build_graph(saver):
    builder = StateGraph(_State)
    builder.add_node("inc", lambda s: {"x": s["x"] + 1})
    builder.add_edge(START, "inc")
    builder.add_edge("inc", END)
    return builder.compile(checkpointer=saver)

def test_detect_returns_sqlite_strategy():
    with SqliteSaver.from_conn_string(":memory:") as saver:
        strategy = detect(saver)
        assert isinstance(strategy, SqliteStrategy)

def test_collect_empty():
    with SqliteSaver.from_conn_string(":memory:") as saver:
        saver.setup()
        strategy = SqliteStrategy(saver)
        assert strategy.collect(None)[0] == {}

def test_collect_with_threads():
    with SqliteSaver.from_conn_string(":memory:") as saver:
        graph = _build_graph(saver)
        graph.invoke({"x": 0}, {"configurable": {"thread_id": "t1"}})
        graph.invoke({"x": 0}, {"configurable": {"thread_id": "t2"}})

        strategy = SqliteStrategy(saver)
        result, _ = strategy.collect(None)

        assert set(result.keys()) == {"t1", "t2"}
        for ts in result.values():
            assert abs(uuid6_to_unix(ts.latest_id) - time.time()) < 5
            assert ts.earliest_id <= ts.latest_id

def test_sweep_with_sqlite_deletes_expired():
    with SqliteSaver.from_conn_string(":memory:") as saver:
        graph = _build_graph(saver)
        graph.invoke({"x": 0}, {"configurable": {"thread_id": "t1"}})

        policy = TTLPolicy(idle_ttl_seconds=1)
        sweeper = Sweeper(saver, policy)

        result = sweeper.sweep()
        assert "t1" not in result.deleted_thread_ids

        time.sleep(1.1)
        result = sweeper.sweep()
        assert "t1" in result.deleted_thread_ids

        row = saver.conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", ("t1",)
        ).fetchone()
        assert row[0] == 0

def test_detect_returns_async_sqlite_strategy():
    pytest.importorskip("aiosqlite")

    async def _inner():
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            strategy = detect(saver)
            assert isinstance(strategy, AsyncSqliteStrategy)

    import asyncio
    asyncio.run(_inner())

@pytest.mark.asyncio
async def test_async_sqlite_collect():
    pytest.importorskip("aiosqlite")

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        graph = _build_graph(saver)
        await graph.ainvoke({"x": 0}, {"configurable": {"thread_id": "t1"}})

        strategy = AsyncSqliteStrategy(saver)
        result, _ = await strategy.acollect(None)

        assert "t1" in result
        assert abs(uuid6_to_unix(result["t1"].latest_id) - time.time()) < 5

@pytest.mark.asyncio
async def test_async_sweep_deletes_expired():
    pytest.importorskip("aiosqlite")

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        graph = _build_graph(saver)
        await graph.ainvoke({"x": 0}, {"configurable": {"thread_id": "t1"}})

        sweeper = Sweeper(saver, TTLPolicy(idle_ttl_seconds=1))

        result = await sweeper.asweep()
        assert "t1" not in result.deleted_thread_ids

        time.sleep(1.1)
        result = await sweeper.asweep()
        assert "t1" in result.deleted_thread_ids

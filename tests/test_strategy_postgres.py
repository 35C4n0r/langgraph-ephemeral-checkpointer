"""Integration tests for PostgresSaver / AsyncPostgresSaver strategies.

Requires Docker and testcontainers:
    pip install testcontainers[postgres]
"""
import time

import pytest

from langgraph_ephemeral_checkpointer import Sweeper, TTLPolicy
from langgraph_ephemeral_checkpointer._strategies import detect
from langgraph_ephemeral_checkpointer._strategies.postgres import (
    AsyncPostgresStrategy,
    PostgresStrategy,
)

testcontainers = pytest.importorskip("testcontainers")

from typing import TypedDict  # noqa: E402

from langgraph.graph import END, START, StateGraph  # noqa: E402
from testcontainers.postgres import PostgresContainer  # noqa: E402


class _State(TypedDict):
    x: int

def _build_graph(saver):
    builder = StateGraph(_State)
    builder.add_node("inc", lambda s: {"x": s["x"] + 1})
    builder.add_edge(START, "inc")
    builder.add_edge("inc", END)
    return builder.compile(checkpointer=saver)

@pytest.fixture(scope="module")
def postgres_dsn():
    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg.get_connection_url().replace("psycopg2", "psycopg")

def test_detect_returns_postgres_strategy(postgres_dsn):
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(postgres_dsn) as saver:
        saver.setup()
        strategy = detect(saver)
        assert isinstance(strategy, PostgresStrategy)

def test_postgres_collect_empty(postgres_dsn):
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(postgres_dsn) as saver:
        saver.setup()
        strategy = PostgresStrategy(saver)
        threads, _ = strategy.collect(None)
        assert isinstance(threads, dict)

def test_postgres_sweep_deletes_expired(postgres_dsn):
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(postgres_dsn) as saver:
        saver.setup()
        graph = _build_graph(saver)
        graph.invoke({"x": 0}, {"configurable": {"thread_id": "t1"}})

        sweeper = Sweeper(saver, TTLPolicy(idle_ttl_seconds=1))

        result = sweeper.sweep()
        assert "t1" not in result.deleted_thread_ids

        time.sleep(1.1)
        result = sweeper.sweep()
        assert "t1" in result.deleted_thread_ids

def test_detect_returns_async_postgres_strategy(postgres_dsn):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async def _inner():
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            strategy = detect(saver)
            assert isinstance(strategy, AsyncPostgresStrategy)

    import asyncio
    asyncio.run(_inner())

@pytest.mark.asyncio
async def test_async_postgres_sweep(postgres_dsn):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
        await saver.setup()
        graph = _build_graph(saver)
        await graph.ainvoke({"x": 0}, {"configurable": {"thread_id": "t1"}})

        sweeper = Sweeper(saver, TTLPolicy(idle_ttl_seconds=1))

        result = await sweeper.asweep()
        assert "t1" not in result.deleted_thread_ids

        time.sleep(1.1)
        result = await sweeper.asweep()
        assert "t1" in result.deleted_thread_ids

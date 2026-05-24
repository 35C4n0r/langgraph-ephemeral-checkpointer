"""Integration tests for PostgresSaver / AsyncPostgresSaver strategies.

Requires the test-postgres extra:
    uv sync --extra test --extra test-postgres
"""
import time
from typing import TypedDict

import pytest

from langgraph_ephemeral_checkpointer import Sweeper, TTLPolicy
from langgraph_ephemeral_checkpointer._coordination import (
    AsyncAdvisoryLock,
    SyncAdvisoryLock,
)
from langgraph_ephemeral_checkpointer._strategies import detect
from langgraph_ephemeral_checkpointer._strategies.postgres import (
    AsyncPostgresStrategy,
    PostgresStrategy,
)

testcontainers = pytest.importorskip("testcontainers")

from langgraph.graph import END, START, StateGraph  # noqa: E402
from testcontainers.postgres import PostgresContainer  # noqa: E402  # pyrefly: ignore[missing-import]


class _State(TypedDict):
    x: int


def _build_graph(saver):
    builder = StateGraph(_State)  # pyrefly: ignore[bad-specialization]
    builder.add_node("inc", lambda s: {"x": s["x"] + 1})  # pyrefly: ignore[bad-argument-type]
    builder.add_edge(START, "inc")
    builder.add_edge("inc", END)
    return builder.compile(checkpointer=saver)


def _row_count(saver, table: str, thread_id: str) -> int:
    with saver._cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS c FROM {table} WHERE thread_id = %s", (thread_id,))
        return cur.fetchone()["c"]


async def _arow_count(saver, table: str, thread_id: str) -> int:
    async with saver._cursor() as cur:
        await cur.execute(f"SELECT COUNT(*) AS c FROM {table} WHERE thread_id = %s", (thread_id,))
        return (await cur.fetchone())["c"]


@pytest.fixture(scope="module")
def postgres_dsn():
    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg.get_connection_url().replace("psycopg2", "psycopg")


def test_detect_returns_postgres_strategy(postgres_dsn):
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(postgres_dsn) as saver:
        saver.setup()
        assert isinstance(detect(saver), PostgresStrategy)


def test_detect_returns_async_postgres_strategy(postgres_dsn):
    import asyncio

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async def _inner():
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            assert isinstance(detect(saver), AsyncPostgresStrategy)

    asyncio.run(_inner())


def test_postgres_collect_empty(postgres_dsn):
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(postgres_dsn) as saver:
        saver.setup()
        threads, cursor = PostgresStrategy(saver).collect(None)
        assert isinstance(threads, dict)
        assert cursor is None or isinstance(cursor, str)


def test_postgres_collect_returns_thread_timestamps(postgres_dsn):
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(postgres_dsn) as saver:
        saver.setup()
        graph = _build_graph(saver)
        graph.invoke({"x": 0}, {"configurable": {"thread_id": "collect_sync"}})

        threads, cursor = PostgresStrategy(saver).collect(None)

        assert "collect_sync" in threads
        ts = threads["collect_sync"]
        assert ts.earliest_id <= ts.latest_id
        assert cursor == ts.latest_id


@pytest.mark.asyncio
async def test_async_postgres_collect_returns_thread_timestamps(postgres_dsn):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
        await saver.setup()
        graph = _build_graph(saver)
        await graph.ainvoke({"x": 0}, {"configurable": {"thread_id": "collect_async"}})

        threads, cursor = await AsyncPostgresStrategy(saver).acollect(None)

        assert "collect_async" in threads
        ts = threads["collect_async"]
        assert ts.earliest_id <= ts.latest_id
        assert cursor == ts.latest_id


def test_postgres_sweep_keeps_active_thread(postgres_dsn):
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(postgres_dsn) as saver:
        saver.setup()
        graph = _build_graph(saver)
        graph.invoke({"x": 0}, {"configurable": {"thread_id": "active_sync"}})

        result = Sweeper(saver, TTLPolicy(idle_ttl_seconds=60)).sweep()
        assert "active_sync" not in result.deleted_thread_ids


def test_postgres_sweep_deletes_expired(postgres_dsn):
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(postgres_dsn) as saver:
        saver.setup()
        graph = _build_graph(saver)
        graph.invoke({"x": 0}, {"configurable": {"thread_id": "expired_sync"}})

        sweeper = Sweeper(saver, TTLPolicy(idle_ttl_seconds=1))
        assert "expired_sync" not in sweeper.sweep().deleted_thread_ids

        time.sleep(1.1)
        assert "expired_sync" in sweeper.sweep().deleted_thread_ids


@pytest.mark.asyncio
async def test_async_postgres_sweep_deletes_expired(postgres_dsn):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
        await saver.setup()
        graph = _build_graph(saver)
        await graph.ainvoke({"x": 0}, {"configurable": {"thread_id": "expired_async"}})

        sweeper = Sweeper(saver, TTLPolicy(idle_ttl_seconds=1))
        assert "expired_async" not in (await sweeper.asweep()).deleted_thread_ids

        time.sleep(1.1)
        assert "expired_async" in (await sweeper.asweep()).deleted_thread_ids


def test_postgres_batch_delete_clears_all_tables(postgres_dsn):
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(postgres_dsn) as saver:
        saver.setup()
        graph = _build_graph(saver)
        graph.invoke({"x": 0}, {"configurable": {"thread_id": "all_tables_sync"}})

        assert _row_count(saver, "checkpoints", "all_tables_sync") > 0
        assert _row_count(saver, "checkpoint_blobs", "all_tables_sync") > 0

        sweeper = Sweeper(saver, TTLPolicy(idle_ttl_seconds=1))
        time.sleep(1.1)
        result = sweeper.sweep()

        assert "all_tables_sync" in result.deleted_thread_ids
        assert _row_count(saver, "checkpoints", "all_tables_sync") == 0
        assert _row_count(saver, "checkpoint_writes", "all_tables_sync") == 0
        assert _row_count(saver, "checkpoint_blobs", "all_tables_sync") == 0


@pytest.mark.asyncio
async def test_async_postgres_batch_delete_clears_all_tables(postgres_dsn):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
        await saver.setup()
        graph = _build_graph(saver)
        await graph.ainvoke({"x": 0}, {"configurable": {"thread_id": "all_tables_async"}})

        assert await _arow_count(saver, "checkpoints", "all_tables_async") > 0
        assert await _arow_count(saver, "checkpoint_blobs", "all_tables_async") > 0

        sweeper = Sweeper(saver, TTLPolicy(idle_ttl_seconds=1))
        time.sleep(1.1)
        result = await sweeper.asweep()

        assert "all_tables_async" in result.deleted_thread_ids
        assert await _arow_count(saver, "checkpoints", "all_tables_async") == 0
        assert await _arow_count(saver, "checkpoint_writes", "all_tables_async") == 0
        assert await _arow_count(saver, "checkpoint_blobs", "all_tables_async") == 0


def test_postgres_sweep_no_cross_contamination(postgres_dsn):
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(postgres_dsn) as saver:
        saver.setup()
        graph = _build_graph(saver)
        graph.invoke({"x": 0}, {"configurable": {"thread_id": "survivor"}})
        graph.invoke({"x": 0}, {"configurable": {"thread_id": "victim"}})

        time.sleep(1.1)
        graph.invoke({"x": 1}, {"configurable": {"thread_id": "survivor"}})

        result = Sweeper(saver, TTLPolicy(idle_ttl_seconds=1)).sweep()

        assert "victim" in result.deleted_thread_ids
        assert "survivor" not in result.deleted_thread_ids
        assert _row_count(saver, "checkpoints", "survivor") > 0
        assert _row_count(saver, "checkpoint_blobs", "survivor") > 0


def test_postgres_hard_age_ttl(postgres_dsn):
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(postgres_dsn) as saver:
        saver.setup()
        graph = _build_graph(saver)
        graph.invoke({"x": 0}, {"configurable": {"thread_id": "hard_age"}})

        sweeper = Sweeper(saver, TTLPolicy(hard_age_ttl_seconds=1))
        assert "hard_age" not in sweeper.sweep().deleted_thread_ids

        time.sleep(1.1)
        assert "hard_age" in sweeper.sweep().deleted_thread_ids


def test_advisory_lock_type_for_postgres(postgres_dsn):
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(postgres_dsn) as saver:
        saver.setup()
        sweeper = Sweeper(saver, TTLPolicy(idle_ttl_seconds=60), enable_coordination=True)
        assert isinstance(sweeper._advisory_lock, SyncAdvisoryLock)


@pytest.mark.asyncio
async def test_advisory_lock_type_for_async_postgres(postgres_dsn):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
        await saver.setup()
        sweeper = Sweeper(saver, TTLPolicy(idle_ttl_seconds=60), enable_coordination=True)
        assert isinstance(sweeper._advisory_lock, AsyncAdvisoryLock)


def test_postgres_advisory_lock_second_sweeper_skips(postgres_dsn):
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(postgres_dsn) as s1:
        with PostgresSaver.from_conn_string(postgres_dsn) as s2:
            s1.setup()

            sw1 = Sweeper(s1, TTLPolicy(idle_ttl_seconds=60), enable_coordination=True)
            sw2 = Sweeper(s2, TTLPolicy(idle_ttl_seconds=60), enable_coordination=True)

            assert sw1._advisory_lock.try_acquire()
            try:
                result = sw2.sweep()
                assert result.deleted_thread_ids == []
                assert result.active_thread_count == 0
            finally:
                sw1._advisory_lock.release()

            assert sw2.sweep() is not None


@pytest.mark.asyncio
async def test_async_postgres_advisory_lock_second_sweeper_skips(postgres_dsn):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as s1:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as s2:
            await s1.setup()

            sw1 = Sweeper(s1, TTLPolicy(idle_ttl_seconds=60), enable_coordination=True)
            sw2 = Sweeper(s2, TTLPolicy(idle_ttl_seconds=60), enable_coordination=True)

            assert await sw1._advisory_lock.atry_acquire()
            try:
                result = await sw2.asweep()
                assert result.deleted_thread_ids == []
                assert result.active_thread_count == 0
            finally:
                await sw1._advisory_lock.arelease()


def test_postgres_incremental_collect(postgres_dsn):
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(postgres_dsn) as saver:
        saver.setup()
        graph = _build_graph(saver)
        graph.invoke({"x": 0}, {"configurable": {"thread_id": "cursor_t1"}})

        strategy = PostgresStrategy(saver)
        threads1, cursor1 = strategy.collect(None)
        assert "cursor_t1" in threads1
        assert cursor1 is not None

        threads2, _ = strategy.collect(cursor1)
        assert "cursor_t1" not in threads2

        graph.invoke({"x": 0}, {"configurable": {"thread_id": "cursor_t2"}})
        threads3, _ = strategy.collect(cursor1)
        assert "cursor_t2" in threads3
        assert "cursor_t1" not in threads3

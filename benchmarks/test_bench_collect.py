"""Strategy.collect() benchmarks across all backends."""
from __future__ import annotations

import pytest

from conftest import (
    POSTGRES_DSN,
    clean_async,
    clean_sync,
    populate_async,
    populate_sync,
)


# ── Memory ───────────────────────────────────────────────────────────────────


def test_collect_memory_sync(benchmark, n_threads):
    from langgraph.checkpoint.memory import InMemorySaver

    from langgraph_ephemeral_checkpointer._strategies.memory import MemoryStrategy

    saver = InMemorySaver()
    populate_sync(saver, n_threads)
    strategy = MemoryStrategy(saver)

    benchmark(strategy.collect, None)


def test_collect_memory_async(benchmark, n_threads, aio_loop):
    from langgraph.checkpoint.memory import InMemorySaver

    from langgraph_ephemeral_checkpointer._strategies.memory import MemoryStrategy

    saver = InMemorySaver()
    populate_async(saver, n_threads, aio_loop)
    strategy = MemoryStrategy(saver)

    benchmark(lambda: aio_loop.run_until_complete(strategy.acollect(None)))


# ── SQLite sync ──────────────────────────────────────────────────────────────


def test_collect_sqlite_sync(benchmark, n_threads):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    from langgraph.checkpoint.sqlite import SqliteSaver

    from langgraph_ephemeral_checkpointer._strategies.sqlite import SqliteStrategy

    with SqliteSaver.from_conn_string(":memory:") as saver:
        saver.setup()
        populate_sync(saver, n_threads)
        strategy = SqliteStrategy(saver)

        benchmark(strategy.collect, None)


# ── SQLite async ─────────────────────────────────────────────────────────────


def test_collect_async_sqlite(benchmark, n_threads, aio_loop):
    pytest.importorskip("langgraph.checkpoint.sqlite.aio")
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from langgraph_ephemeral_checkpointer._strategies.sqlite import AsyncSqliteStrategy

    run = aio_loop.run_until_complete
    cm = AsyncSqliteSaver.from_conn_string(":memory:")
    saver = run(cm.__aenter__())
    try:
        run(saver.setup())
        populate_async(saver, n_threads, aio_loop)
        strategy = AsyncSqliteStrategy(saver)

        benchmark(lambda: run(strategy.acollect(None)))
    finally:
        run(cm.__aexit__(None, None, None))


# ── Postgres sync ────────────────────────────────────────────────────────────


def test_collect_postgres_sync(benchmark, n_threads, require_postgres):
    from langgraph.checkpoint.postgres import PostgresSaver

    from langgraph_ephemeral_checkpointer._strategies.postgres import PostgresStrategy

    with PostgresSaver.from_conn_string(POSTGRES_DSN) as saver:
        saver.setup()
        tids = populate_sync(saver, n_threads)
        strategy = PostgresStrategy(saver)

        benchmark(strategy.collect, None)

        clean_sync(saver, tids)


# ── Postgres async ───────────────────────────────────────────────────────────


def test_collect_async_postgres(benchmark, n_threads, require_async_postgres, aio_loop):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from langgraph_ephemeral_checkpointer._strategies.postgres import AsyncPostgresStrategy

    run = aio_loop.run_until_complete
    cm = AsyncPostgresSaver.from_conn_string(POSTGRES_DSN)
    saver = run(cm.__aenter__())
    try:
        run(saver.setup())
        tids = populate_async(saver, n_threads, aio_loop)
        strategy = AsyncPostgresStrategy(saver)

        benchmark(lambda: run(strategy.acollect(None)))

        clean_async(saver, tids, aio_loop)
    finally:
        run(cm.__aexit__(None, None, None))

"""Sweeper.sweep() / asweep() benchmarks across all backends and scenarios."""
from __future__ import annotations

import time

import pytest

from langgraph_ephemeral_checkpointer import Sweeper, TTLPolicy

from conftest import (
    EXPIRE_SLEEP_SECONDS,
    EXPIRE_TTL_SECONDS,
    PEDANTIC_ROUNDS,
    POSTGRES_DSN,
    clean_async,
    clean_sync,
    populate_async,
    populate_sync,
)

_EXPIRED_PARAMS = pytest.mark.parametrize(
    "safe_delete,dry_run",
    [(True, False), (False, False), (False, True)],
    ids=["safe", "fast", "dry_run"],
)


# ── Memory sync ─────────────────────────────────────────────────────────────


def test_sweep_active_memory_sync(benchmark, n_threads):
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    populate_sync(saver, n_threads)
    sw = Sweeper(saver, TTLPolicy(idle_ttl_seconds=999_999))

    benchmark(sw.sweep)


@_EXPIRED_PARAMS
def test_sweep_expired_memory_sync(benchmark, n_threads, safe_delete, dry_run):
    from langgraph.checkpoint.memory import InMemorySaver

    def setup():
        saver = InMemorySaver()
        populate_sync(saver, n_threads)
        time.sleep(EXPIRE_SLEEP_SECONDS)
        sw = Sweeper(
            saver,
            TTLPolicy(idle_ttl_seconds=EXPIRE_TTL_SECONDS),
            safe_delete=safe_delete,
        )
        return (sw,), {}

    benchmark.pedantic(
        lambda sw: sw.sweep(dry_run=dry_run),
        setup=setup,
        rounds=PEDANTIC_ROUNDS,
        warmup_rounds=0,
    )


# ── Memory async ────────────────────────────────────────────────────────────


def test_sweep_active_memory_async(benchmark, n_threads, aio_loop):
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    populate_async(saver, n_threads, aio_loop)
    sw = Sweeper(saver, TTLPolicy(idle_ttl_seconds=999_999))

    benchmark(lambda: aio_loop.run_until_complete(sw.asweep()))


@_EXPIRED_PARAMS
def test_sweep_expired_memory_async(benchmark, n_threads, safe_delete, dry_run, aio_loop):
    from langgraph.checkpoint.memory import InMemorySaver

    def setup():
        saver = InMemorySaver()
        populate_async(saver, n_threads, aio_loop)
        time.sleep(EXPIRE_SLEEP_SECONDS)
        sw = Sweeper(
            saver,
            TTLPolicy(idle_ttl_seconds=EXPIRE_TTL_SECONDS),
            safe_delete=safe_delete,
        )
        return (sw,), {}

    benchmark.pedantic(
        lambda sw: aio_loop.run_until_complete(sw.asweep(dry_run=dry_run)),
        setup=setup,
        rounds=PEDANTIC_ROUNDS,
        warmup_rounds=0,
    )


# ── SQLite sync ──────────────────────────────────────────────────────────────


def test_sweep_active_sqlite_sync(benchmark, n_threads):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    from langgraph.checkpoint.sqlite import SqliteSaver

    with SqliteSaver.from_conn_string(":memory:") as saver:
        saver.setup()
        populate_sync(saver, n_threads)
        sw = Sweeper(saver, TTLPolicy(idle_ttl_seconds=999_999))

        benchmark(sw.sweep)


@_EXPIRED_PARAMS
def test_sweep_expired_sqlite_sync(benchmark, n_threads, safe_delete, dry_run):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    from langgraph.checkpoint.sqlite import SqliteSaver

    with SqliteSaver.from_conn_string(":memory:") as saver:
        saver.setup()

        def setup():
            populate_sync(saver, n_threads)
            time.sleep(EXPIRE_SLEEP_SECONDS)
            sw = Sweeper(
                saver,
                TTLPolicy(idle_ttl_seconds=EXPIRE_TTL_SECONDS),
                safe_delete=safe_delete,
            )
            return (sw,), {}

        benchmark.pedantic(
            lambda sw: sw.sweep(dry_run=dry_run),
            setup=setup,
            rounds=PEDANTIC_ROUNDS,
            warmup_rounds=0,
        )


# ── SQLite async ─────────────────────────────────────────────────────────────


def test_sweep_active_async_sqlite(benchmark, n_threads, aio_loop):
    pytest.importorskip("langgraph.checkpoint.sqlite.aio")
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    run = aio_loop.run_until_complete
    cm = AsyncSqliteSaver.from_conn_string(":memory:")
    saver = run(cm.__aenter__())
    try:
        run(saver.setup())
        populate_async(saver, n_threads, aio_loop)
        sw = Sweeper(saver, TTLPolicy(idle_ttl_seconds=999_999))

        benchmark(lambda: run(sw.asweep()))
    finally:
        run(cm.__aexit__(None, None, None))


@_EXPIRED_PARAMS
def test_sweep_expired_async_sqlite(benchmark, n_threads, safe_delete, dry_run, aio_loop):
    pytest.importorskip("langgraph.checkpoint.sqlite.aio")
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    run = aio_loop.run_until_complete
    cm = AsyncSqliteSaver.from_conn_string(":memory:")
    saver = run(cm.__aenter__())
    try:
        run(saver.setup())

        def setup():
            populate_async(saver, n_threads, aio_loop)
            time.sleep(EXPIRE_SLEEP_SECONDS)
            sw = Sweeper(
                saver,
                TTLPolicy(idle_ttl_seconds=EXPIRE_TTL_SECONDS),
                safe_delete=safe_delete,
            )
            return (sw,), {}

        benchmark.pedantic(
            lambda sw: run(sw.asweep(dry_run=dry_run)),
            setup=setup,
            rounds=PEDANTIC_ROUNDS,
            warmup_rounds=0,
        )
    finally:
        run(cm.__aexit__(None, None, None))


# ── Postgres sync ────────────────────────────────────────────────────────────


def test_sweep_active_postgres_sync(benchmark, n_threads, require_postgres):
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(POSTGRES_DSN) as saver:
        saver.setup()
        tids = populate_sync(saver, n_threads)
        sw = Sweeper(saver, TTLPolicy(idle_ttl_seconds=999_999))

        benchmark(sw.sweep)

        clean_sync(saver, tids)


@_EXPIRED_PARAMS
def test_sweep_expired_postgres_sync(benchmark, n_threads, safe_delete, dry_run, require_postgres):
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(POSTGRES_DSN) as saver:
        saver.setup()
        prefix = "bench_"

        def setup():
            populate_sync(saver, n_threads, prefix=prefix)
            time.sleep(EXPIRE_SLEEP_SECONDS)
            sw = Sweeper(
                saver,
                TTLPolicy(idle_ttl_seconds=EXPIRE_TTL_SECONDS),
                safe_delete=safe_delete,
            )
            return (sw,), {}

        try:
            benchmark.pedantic(
                lambda sw: sw.sweep(dry_run=dry_run),
                setup=setup,
                rounds=PEDANTIC_ROUNDS,
                warmup_rounds=0,
            )
        finally:
            tids = [f"{prefix}{i}" for i in range(n_threads)]
            clean_sync(saver, tids)


# ── Postgres async ───────────────────────────────────────────────────────────


def test_sweep_active_async_postgres(benchmark, n_threads, require_async_postgres, aio_loop):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    run = aio_loop.run_until_complete
    cm = AsyncPostgresSaver.from_conn_string(POSTGRES_DSN)
    saver = run(cm.__aenter__())
    try:
        run(saver.setup())
        tids = populate_async(saver, n_threads, aio_loop)
        sw = Sweeper(saver, TTLPolicy(idle_ttl_seconds=999_999))

        benchmark(lambda: run(sw.asweep()))

        clean_async(saver, tids, aio_loop)
    finally:
        run(cm.__aexit__(None, None, None))


@_EXPIRED_PARAMS
def test_sweep_expired_async_postgres(
    benchmark, n_threads, safe_delete, dry_run, require_async_postgres, aio_loop
):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    run = aio_loop.run_until_complete
    cm = AsyncPostgresSaver.from_conn_string(POSTGRES_DSN)
    saver = run(cm.__aenter__())
    try:
        run(saver.setup())
        prefix = "abench_"

        def setup():
            populate_async(saver, n_threads, aio_loop, prefix=prefix)
            time.sleep(EXPIRE_SLEEP_SECONDS)
            sw = Sweeper(
                saver,
                TTLPolicy(idle_ttl_seconds=EXPIRE_TTL_SECONDS),
                safe_delete=safe_delete,
            )
            return (sw,), {}

        try:
            benchmark.pedantic(
                lambda sw: run(sw.asweep(dry_run=dry_run)),
                setup=setup,
                rounds=PEDANTIC_ROUNDS,
                warmup_rounds=0,
            )
        finally:
            tids = [f"{prefix}{i}" for i in range(n_threads)]
            clean_async(saver, tids, aio_loop)
    finally:
        run(cm.__aexit__(None, None, None))

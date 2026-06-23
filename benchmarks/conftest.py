"""Run: uv run --extra benchmark pytest benchmarks/ -v"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).parent))

THREAD_COUNTS = [10, 100, 1_000]
EXPIRE_TTL_SECONDS = 1
EXPIRE_SLEEP_SECONDS = 1.05
PEDANTIC_ROUNDS = 5

POSTGRES_DSN = os.environ.get(
    "BENCHMARK_POSTGRES_DSN",
    "postgresql://postgres:postgres@localhost:5432/bench",
)


class _State(TypedDict):
    x: int


def build_graph(saver):
    g = StateGraph(_State)
    g.add_node("inc", lambda s: {"x": s["x"] + 1})
    g.add_edge(START, "inc")
    g.add_edge("inc", END)
    return g.compile(checkpointer=saver)


def populate_sync(saver, n: int, prefix: str = "t") -> list[str]:
    tids = [f"{prefix}{i}" for i in range(n)]
    g = build_graph(saver)
    for tid in tids:
        g.invoke({"x": 0}, {"configurable": {"thread_id": tid}})
    return tids


def populate_async(
    saver, n: int, loop: asyncio.AbstractEventLoop, prefix: str = "t"
) -> list[str]:
    tids = [f"{prefix}{i}" for i in range(n)]
    g = build_graph(saver)
    for tid in tids:
        loop.run_until_complete(
            g.ainvoke({"x": 0}, {"configurable": {"thread_id": tid}})
        )
    return tids


def clean_sync(saver, tids: list[str]) -> None:
    for tid in tids:
        saver.delete_thread(tid)


def clean_async(
    saver, tids: list[str], loop: asyncio.AbstractEventLoop
) -> None:
    for tid in tids:
        loop.run_until_complete(saver.adelete_thread(tid))


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(params=THREAD_COUNTS, ids=[f"n={n}" for n in THREAD_COUNTS])
def n_threads(request):
    return request.param


@pytest.fixture
def aio_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Postgres availability (checked once per session) ─────────────────────────


@pytest.fixture(scope="session")
def _postgres_sync_check():
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(POSTGRES_DSN) as saver:
            saver.setup()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def _async_postgres_check():
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async def _check():
            async with AsyncPostgresSaver.from_conn_string(POSTGRES_DSN) as saver:
                await saver.setup()

        asyncio.run(_check())
        return True
    except Exception:
        return False


@pytest.fixture
def require_postgres(_postgres_sync_check):
    if not _postgres_sync_check:
        pytest.skip("Postgres unavailable")


@pytest.fixture
def require_async_postgres(_async_postgres_check):
    if not _async_postgres_check:
        pytest.skip("Async Postgres unavailable")

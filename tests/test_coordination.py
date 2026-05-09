"""Tests for multi-instance sweep coordination (advisory locks)."""
from collections.abc import AsyncIterator, Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple

from langgraph_ephemeral_checkpointer import Sweeper, TTLPolicy
from langgraph_ephemeral_checkpointer._coordination import _LOCK_KEY, get_advisory_lock

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
        return None

    async def aget_tuple(self, config):
        return None

    def put(self, config, checkpoint, metadata, new_versions):
        return config

    def put_writes(self, config, writes, task_id, task_path=()):
        pass

    def delete_thread(self, thread_id):
        self.deleted.append(thread_id)

    async def adelete_thread(self, thread_id):
        self.deleted.append(thread_id)

def _make_sweeper(tuples):
    cp = MockCP(tuples)
    sweeper = Sweeper(cp, TTLPolicy(idle_ttl_seconds=60), safe_delete=False, _strategy=list_strategy(tuples))
    return sweeper, cp

def _expired(*ids):
    return [make_checkpoint_tuple(tid, FAKE_CP_ID, iso_ts(-3700)) for tid in ids]

def test_lock_key_is_deterministic():
    import hashlib
    expected = int.from_bytes(
        hashlib.md5(b"langgraph-ephemeral-checkpointer").digest()[:8],
        byteorder="big",
        signed=True,
    )
    assert _LOCK_KEY == expected

def test_get_advisory_lock_disabled_returns_none():
    cp = MockCP([])
    assert get_advisory_lock(cp, enable=False) is None

def test_non_postgres_coordination_warns(caplog):
    import logging

    tuples = _expired("t")
    cp = MockCP(tuples)
    with caplog.at_level(logging.WARNING, logger="langgraph_ephemeral_checkpointer._coordination"):
        sweeper = Sweeper(
            cp,
            TTLPolicy(idle_ttl_seconds=60),
            enable_coordination=True,
            safe_delete=False,
            _strategy=list_strategy(tuples),
        )

    assert any("advisory locks require" in r.message for r in caplog.records)
    assert sweeper._advisory_lock is None

def test_non_postgres_coordination_sweep_still_proceeds():
    """Even with enable_coordination=True on a non-Postgres backend, sweep runs."""
    tuples = _expired("t")
    cp = MockCP(tuples)
    sweeper = Sweeper(
        cp,
        TTLPolicy(idle_ttl_seconds=60),
        enable_coordination=True,
        safe_delete=False,
        _strategy=list_strategy(tuples),
    )
    result = sweeper.sweep()
    assert "t" in result.deleted_thread_ids

def test_lock_acquired_sweep_proceeds():
    sweeper, cp = _make_sweeper(_expired("t"))

    mock_lock = MagicMock()
    mock_lock.try_acquire.return_value = True
    sweeper._advisory_lock = mock_lock

    result = sweeper.sweep()

    mock_lock.try_acquire.assert_called_once()
    mock_lock.release.assert_called_once()
    assert "t" in result.deleted_thread_ids
    assert "t" in cp.deleted

def test_lock_not_acquired_skips_sweep(caplog):
    import logging

    sweeper, cp = _make_sweeper(_expired("t"))

    mock_lock = MagicMock()
    mock_lock.try_acquire.return_value = False
    sweeper._advisory_lock = mock_lock

    with caplog.at_level(logging.INFO, logger="langgraph_ephemeral_checkpointer.sweeper"):
        result = sweeper.sweep()

    assert result.deleted_thread_ids == []
    assert result.active_thread_count == 0
    assert cp.deleted == []
    mock_lock.release.assert_not_called()
    assert any("Advisory lock held" in r.message for r in caplog.records)

def test_lock_released_on_sweep_exception():
    sweeper, cp = _make_sweeper([])

    mock_lock = MagicMock()
    mock_lock.try_acquire.return_value = True
    sweeper._advisory_lock = mock_lock

    def raising_run(**kwargs):
        raise RuntimeError("boom")

    sweeper._run_sweep = raising_run

    with pytest.raises(RuntimeError, match="boom"):
        sweeper.sweep()

    mock_lock.release.assert_called_once()

@pytest.mark.asyncio
async def test_async_lock_acquired_proceeds():
    sweeper, cp = _make_sweeper(_expired("t"))

    mock_lock = MagicMock()
    mock_lock.atry_acquire = AsyncMock(return_value=True)
    mock_lock.arelease = AsyncMock()
    sweeper._advisory_lock = mock_lock

    result = await sweeper.asweep()

    mock_lock.atry_acquire.assert_called_once()
    mock_lock.arelease.assert_called_once()
    assert "t" in result.deleted_thread_ids

@pytest.mark.asyncio
async def test_async_lock_not_acquired_skips(caplog):
    import logging

    sweeper, cp = _make_sweeper(_expired("t"))

    mock_lock = MagicMock()
    mock_lock.atry_acquire = AsyncMock(return_value=False)
    mock_lock.arelease = AsyncMock()
    sweeper._advisory_lock = mock_lock

    with caplog.at_level(logging.INFO, logger="langgraph_ephemeral_checkpointer.sweeper"):
        result = await sweeper.asweep()

    assert result.deleted_thread_ids == []
    assert cp.deleted == []
    mock_lock.arelease.assert_not_called()
    assert any("Advisory lock held" in r.message for r in caplog.records)

@pytest.mark.asyncio
async def test_async_lock_released_on_exception():
    sweeper, cp = _make_sweeper([])

    mock_lock = MagicMock()
    mock_lock.atry_acquire = AsyncMock(return_value=True)
    mock_lock.arelease = AsyncMock()
    sweeper._advisory_lock = mock_lock

    async def raising_arun(**kwargs):
        raise RuntimeError("async boom")

    sweeper._arun_sweep = raising_arun

    with pytest.raises(RuntimeError, match="async boom"):
        await sweeper.asweep()

    mock_lock.arelease.assert_called_once()

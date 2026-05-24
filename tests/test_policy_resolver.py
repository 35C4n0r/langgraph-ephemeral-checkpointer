"""Tests for per-thread policy overrides."""
from collections.abc import AsyncIterator, Iterator

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple

from langgraph_ephemeral_checkpointer import Sweeper, TTLPolicy
from langgraph_ephemeral_checkpointer.types import PolicyOverride

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

def _tuples(*thread_ids, offset=-3700):
    return [make_checkpoint_tuple(tid, None, iso_ts(offset)) for tid in thread_ids]

def test_exempt_thread_never_deleted():
    cp = MockCP(_tuples("vip", "regular"))
    policy = TTLPolicy(idle_ttl_seconds=60)

    def resolver(tid):
        return PolicyOverride.EXEMPT if tid == "vip" else PolicyOverride.USE_DEFAULT

    sweeper = Sweeper(cp, policy, policy_resolver=resolver, _strategy=list_strategy(cp._tuples))
    result = sweeper.sweep()

    assert "vip" not in result.deleted_thread_ids
    assert "regular" in result.deleted_thread_ids
    assert "vip" not in cp.deleted

def test_custom_policy_override_used():
    """Thread 'long' gets a 7-day TTL, thread 'short' gets the 60s default."""
    cp = MockCP(_tuples("long", "short", offset=-3700))
    default_policy = TTLPolicy(idle_ttl_seconds=60)
    long_policy = TTLPolicy(idle_ttl_seconds=604800)

    def resolver(tid):
        return long_policy if tid == "long" else PolicyOverride.USE_DEFAULT

    sweeper = Sweeper(cp, default_policy, policy_resolver=resolver, _strategy=list_strategy(cp._tuples))
    result = sweeper.sweep()

    assert "short" in result.deleted_thread_ids
    assert "long" not in result.deleted_thread_ids

def test_use_default_falls_back_to_global():
    cp = MockCP(_tuples("t"))
    policy = TTLPolicy(idle_ttl_seconds=60)

    def resolver(tid):
        return PolicyOverride.USE_DEFAULT

    sweeper = Sweeper(cp, policy, policy_resolver=resolver, _strategy=list_strategy(cp._tuples))
    result = sweeper.sweep()
    assert "t" in result.deleted_thread_ids

def test_resolver_exception_propagates():
    cp = MockCP(_tuples("t"))
    policy = TTLPolicy(idle_ttl_seconds=60)

    def bad_resolver(tid):
        raise ValueError("resolver exploded")

    sweeper = Sweeper(cp, policy, policy_resolver=bad_resolver, _strategy=list_strategy(cp._tuples))

    with pytest.raises(ValueError, match="resolver exploded"):
        sweeper.sweep()

def test_resolver_called_once_per_thread():
    call_counts: dict[str, int] = {}
    cp = MockCP(_tuples("a", "b", "c"))
    policy = TTLPolicy(idle_ttl_seconds=3600)

    def counting_resolver(tid):
        call_counts[tid] = call_counts.get(tid, 0) + 1
        return PolicyOverride.USE_DEFAULT

    sweeper = Sweeper(cp, policy, policy_resolver=counting_resolver, _strategy=list_strategy(cp._tuples))
    sweeper.sweep()

    assert call_counts == {"a": 1, "b": 1, "c": 1}

@pytest.mark.asyncio
async def test_async_sweep_respects_resolver():
    cp = MockCP(_tuples("exempt", "expired"))
    policy = TTLPolicy(idle_ttl_seconds=60)

    def resolver(tid):
        return PolicyOverride.EXEMPT if tid == "exempt" else PolicyOverride.USE_DEFAULT

    sweeper = Sweeper(cp, policy, policy_resolver=resolver, _strategy=list_strategy(cp._tuples))
    result = await sweeper.asweep()

    assert "exempt" not in result.deleted_thread_ids
    assert "expired" in result.deleted_thread_ids


def test_fresh_policy_objects_per_thread_applied_correctly():
    tuples = [
        make_checkpoint_tuple("strict", None, iso_ts(-70)),
        make_checkpoint_tuple("lenient", None, iso_ts(-70)),
    ]
    cp = MockCP(tuples)

    def resolver(tid):
        if tid == "strict":
            return TTLPolicy(idle_ttl_seconds=60)
        return TTLPolicy(idle_ttl_seconds=3600)

    sweeper = Sweeper(
        cp,
        TTLPolicy(idle_ttl_seconds=9999),
        policy_resolver=resolver,
        safe_delete=False,
        _strategy=list_strategy(tuples),
    )
    result = sweeper.sweep()

    assert "strict" in result.deleted_thread_ids
    assert "lenient" not in result.deleted_thread_ids


def test_many_threads_fresh_policies_all_evaluated_correctly():
    thread_ids = [f"t{i}" for i in range(50)]
    tuples = [make_checkpoint_tuple(tid, None, iso_ts(-70)) for tid in thread_ids]
    cp = MockCP(tuples)

    strict_ids = {tid for tid in thread_ids if int(tid[1:]) % 2 == 0}

    def resolver(tid):
        if tid in strict_ids:
            return TTLPolicy(idle_ttl_seconds=60)
        return TTLPolicy(idle_ttl_seconds=3600)

    sweeper = Sweeper(
        cp,
        TTLPolicy(idle_ttl_seconds=9999),
        policy_resolver=resolver,
        safe_delete=False,
        _strategy=list_strategy(tuples),
    )
    result = sweeper.sweep()

    deleted = set(result.deleted_thread_ids)
    assert deleted == strict_ids

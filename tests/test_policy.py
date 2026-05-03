import pytest

from langgraph_ephemeral_checkpointer import TTLPolicy


def test_no_rules_raises():
    with pytest.raises(ValueError, match="At least one of"):
        TTLPolicy()

def test_negative_idle_raises():
    with pytest.raises(ValueError, match="idle_ttl_seconds must be positive"):
        TTLPolicy(idle_ttl_seconds=-1)

def test_zero_idle_raises():
    with pytest.raises(ValueError, match="idle_ttl_seconds must be positive"):
        TTLPolicy(idle_ttl_seconds=0)

def test_negative_hard_age_raises():
    with pytest.raises(ValueError, match="hard_age_ttl_seconds must be positive"):
        TTLPolicy(hard_age_ttl_seconds=-1)

def test_zero_hard_age_raises():
    with pytest.raises(ValueError, match="hard_age_ttl_seconds must be positive"):
        TTLPolicy(hard_age_ttl_seconds=0)

def test_idle_only():
    p = TTLPolicy(idle_ttl_seconds=60)
    assert p.idle_ttl_seconds == 60
    assert p.hard_age_ttl_seconds is None

def test_hard_age_only():
    p = TTLPolicy(hard_age_ttl_seconds=3600)
    assert p.hard_age_ttl_seconds == 3600
    assert p.idle_ttl_seconds is None

def test_both_ttls():
    p = TTLPolicy(idle_ttl_seconds=60, hard_age_ttl_seconds=3600)
    assert p.idle_ttl_seconds == 60
    assert p.hard_age_ttl_seconds == 3600

def test_frozen():
    p = TTLPolicy(idle_ttl_seconds=60)
    with pytest.raises((AttributeError, TypeError)):
        p.idle_ttl_seconds = 999  # type: ignore[misc]

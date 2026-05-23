# langgraph-ephemeral-checkpointer

TTL-based thread retention for [LangGraph](https://github.com/langchain-ai/langgraph) checkpointers. Expire and delete old conversation threads based on idle time or absolute age.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [TTLPolicy](#ttlpolicy)
- [Sweeper](#sweeper)
  - [Running a sweep](#running-a-sweep)
  - [Background loop](#background-loop)
  - [Dry run](#dry-run)
  - [SweepResult](#sweepresult)
- [Per-thread policy overrides](#per-thread-policy-overrides)
- [Callbacks](#callbacks)
- [Safe delete](#safe-delete)
- [Multi-instance coordination](#multi-instance-coordination)
- [Backends](#backends)
- [API reference](#api-reference)

---

## Installation

```bash
pip install langgraph-ephemeral-checkpointer
```

Backend extras (install only what you use):

```bash
pip install "langgraph-ephemeral-checkpointer[sqlite]"    # SqliteSaver
pip install "langgraph-ephemeral-checkpointer[postgres]"  # PostgresSaver
```

Python 3.11+ is required.

---

## Quick Start

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph_ephemeral_checkpointer import TTLPolicy, Sweeper

checkpointer = SqliteSaver.from_conn_string("threads.db")
policy = TTLPolicy(idle_ttl_seconds=3600)  # delete threads idle for > 1 hour
sweeper = Sweeper(checkpointer, policy)

result = sweeper.sweep()
print(f"Deleted {len(result.deleted_thread_ids)} thread(s)")
```

---

## TTLPolicy

`TTLPolicy` defines when a thread is considered expired. At least one rule is required.

```python
from langgraph_ephemeral_checkpointer import TTLPolicy

# Idle TTL: delete threads with no activity for 2 hours
policy = TTLPolicy(idle_ttl_seconds=7200)

# Hard age TTL: delete threads older than 7 days regardless of activity
policy = TTLPolicy(hard_age_ttl_seconds=604800)

# Combine rules: a thread expires as soon as either fires
policy = TTLPolicy(
    idle_ttl_seconds=3600,
    hard_age_ttl_seconds=86400,
)
```

| Parameter | Type | Description |
|---|---|---|
| `idle_ttl_seconds` | `int \| None` | Expire threads with no checkpoint activity for this many seconds |
| `hard_age_ttl_seconds` | `int \| None` | Expire threads whose first checkpoint is older than this many seconds |

`TTLPolicy` is a frozen dataclass — instances are immutable.

---

## Sweeper

`Sweeper` is the main entry point.

```python
Sweeper(
    checkpointer,           # any LangGraph BaseCheckpointSaver
    policy,                 # TTLPolicy
    *,
    policy_resolver=None,   # per-thread overrides (see below)
    enable_coordination=False,  # PostgreSQL advisory locks
    safe_delete=True,       # re-verify timestamps before deleting
    on_before_delete=None,  # callback before each deletion
    on_sweep_complete=None, # callback after each sweep
)
```

### Running a sweep

```python
# synchronous
result = sweeper.sweep()

# asynchronous
result = await sweeper.asweep()
```

Both return a [`SweepResult`](#sweepresult).

### Background loop

For always-on applications, run a background sweep loop that fires on an interval:

```python
import asyncio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph_ephemeral_checkpointer import TTLPolicy, Sweeper

async def main():
    async with AsyncSqliteSaver.from_conn_string("threads.db") as checkpointer:
        policy = TTLPolicy(idle_ttl_seconds=3600)
        sweeper = Sweeper(checkpointer, policy)

        await sweeper.start(interval_seconds=300)  # sweep every 5 minutes

        # ... your application runs here ...

        await sweeper.stop()

asyncio.run(main())
```

### Dry run

Preview what would be deleted without actually deleting anything:

```python
result = sweeper.sweep(dry_run=True)
print(f"Would delete: {result.deleted_thread_ids}")
```

`on_before_delete` is skipped during dry runs. `on_sweep_complete` still fires.

### SweepResult

Every sweep returns a `SweepResult`:

```python
result = sweeper.sweep()

result.deleted_thread_ids      # list[str]: IDs of deleted threads
result.active_thread_count     # int: threads still alive after this sweep
result.sweep_duration_seconds  # float: wall-clock time for the sweep
```

---

## Per-thread policy overrides

Supply a `policy_resolver` to apply different rules to individual threads. The resolver receives a `thread_id` and returns either a custom `TTLPolicy` or a `PolicyOverride`.

```python
from langgraph_ephemeral_checkpointer import TTLPolicy, Sweeper
from langgraph_ephemeral_checkpointer.types import PolicyOverride

default_policy = TTLPolicy(idle_ttl_seconds=3600)

vip_policy = TTLPolicy(idle_ttl_seconds=604800)  # VIP threads last 7 days

def resolver(thread_id: str):
    if thread_id.startswith("vip:"):
        return vip_policy
    if thread_id.startswith("system:"):
        return PolicyOverride.EXEMPT  # never expire
    return PolicyOverride.USE_DEFAULT

sweeper = Sweeper(checkpointer, default_policy, policy_resolver=resolver)
```

| Return value | Behaviour |
|---|---|
| `TTLPolicy` | Use this policy for the thread instead of the global one |
| `PolicyOverride.USE_DEFAULT` | Apply the sweeper's global policy |
| `PolicyOverride.EXEMPT` | Never expire this thread |

If the resolver raises, the sweep aborts. It is called once per thread per sweep.

---

## Callbacks

### `on_before_delete`

Called before each thread is deleted. Return `False` to skip the deletion.

```python
def on_before_delete(thread_id: str, policy: TTLPolicy, reason: str) -> bool:
    print(f"Deleting {thread_id!r} (reason: {reason})")
    return True

sweeper = Sweeper(checkpointer, policy, on_before_delete=on_before_delete)
```

The `reason` parameter is one of `"idle_ttl"` or `"hard_age_ttl"`.

### `on_sweep_complete`

Called once after every sweep cycle with the final `SweepResult`.

```python
import logging

logger = logging.getLogger(__name__)

def on_sweep_complete(result: SweepResult) -> None:
    logger.info(
        "sweep complete",
        extra={
            "deleted": len(result.deleted_thread_ids),
            "active": result.active_thread_count,
            "duration_s": result.sweep_duration_seconds,
        },
    )

sweeper = Sweeper(checkpointer, policy, on_sweep_complete=on_sweep_complete)
```

Exceptions raised inside either callback propagate and abort the sweep.

---

## Safe delete

By default (`safe_delete=True`), the sweeper re-reads each thread's latest checkpoint immediately before deleting and skips it if a newer checkpoint has appeared since the scan started.

```python
sweeper = Sweeper(checkpointer, policy, safe_delete=False)
```

---

## Multi-instance coordination

If you run multiple application instances sharing a single PostgreSQL checkpointer, you can enable advisory locks so only one instance sweeps at a time:

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph_ephemeral_checkpointer import TTLPolicy, Sweeper

async with AsyncPostgresSaver.from_conn_string(dsn) as checkpointer:
    sweeper = Sweeper(
        checkpointer,
        TTLPolicy(idle_ttl_seconds=3600),
        enable_coordination=True,
    )
    result = await sweeper.asweep()
```

When `enable_coordination=True` and the backend is PostgreSQL, the sweeper acquires a session-level advisory lock before scanning. If another instance holds the lock, the sweep is skipped and an empty `SweepResult` is returned. The lock is tied to the database connection, so a crashed instance releases it automatically.

`enable_coordination=True` is a no-op for non-PostgreSQL backends — a warning is logged.

---

## Backends

The sweeper picks the most efficient strategy for your checkpointer automatically.

| Checkpointer | Strategy | Notes |
|---|---|---|
| `InMemorySaver` | `MemoryStrategy` | Reads storage dict directly; extracts timestamps from UUIDv6 checkpoint IDs |
| `SqliteSaver` | `SqliteStrategy` | Single `GROUP BY` query for all threads |
| `AsyncSqliteSaver` | `AsyncSqliteStrategy` | Async variant of the above |
| `PostgresSaver` | `PostgresStrategy` | `GROUP BY` query; advisory lock support |
| `AsyncPostgresSaver` | `AsyncPostgresStrategy` | Async variant; advisory lock support |

---

## API reference

### `TTLPolicy`

```python
@dataclass(frozen=True)
class TTLPolicy:
    idle_ttl_seconds: int | None = None
    hard_age_ttl_seconds: int | None = None
```

### `Sweeper`

```python
class Sweeper:
    def __init__(
        self,
        checkpointer: BaseCheckpointSaver,
        policy: TTLPolicy,
        *,
        policy_resolver: PolicyResolver | None = None,
        enable_coordination: bool = False,
        safe_delete: bool = True,
        on_before_delete: OnBeforeDelete | None = None,
        on_sweep_complete: OnSweepComplete | None = None,
    ) -> None: ...

    def sweep(self, *, dry_run: bool = False) -> SweepResult: ...
    async def asweep(self, *, dry_run: bool = False) -> SweepResult: ...
    async def start(self, interval_seconds: int = 300) -> None: ...
    async def stop(self) -> None: ...
```

### `SweepResult`

```python
@dataclass
class SweepResult:
    deleted_thread_ids: list[str]
    active_thread_count: int
    sweep_duration_seconds: float
```

### `PolicyOverride`

```python
class PolicyOverride(enum.Enum):
    USE_DEFAULT = "use_default"
    EXEMPT      = "exempt"
```

### Callable types

```python
PolicyResolver  = Callable[[str], TTLPolicy | PolicyOverride]
OnBeforeDelete  = Callable[[str, TTLPolicy, str], bool]
OnSweepComplete = Callable[[SweepResult], None]
```

# Changelog

## 0.0.2

### Fixed

- `PostgresStrategy` / `AsyncPostgresStrategy` - renamed `writes` to `checkpoint_writes` to match the langgraph-checkpoint-postgres v3 schema; the old name raised `UndefinedTable` on every sweep, meaning no Postgres threads were ever deleted
- `PostgresStrategy` / `AsyncPostgresStrategy` - added deletion of `checkpoint_blobs`; omitting this table left all serialised channel data behind as orphaned rows after every sweep
- `SyncAdvisoryLock` / `AsyncAdvisoryLock` - split the single `AdvisoryLock` class into two; the original mixed sync and async cursor usage, causing `TypeError` at runtime when advisory locks were enabled on `AsyncPostgresSaver`
- `_thresholds` cache in `_plan()` - changed cache key from `id(policy)` to the `TTLPolicy` value itself; CPython address reuse could cause a `policy_resolver` that creates fresh `TTLPolicy` instances per thread to receive thresholds computed for a different policy
- Background loop - exceptions from `asweep()` are now caught, logged at `ERROR` level, and the loop continues; previously any transient error permanently killed the background task

### Tests

- Postgres integration tests now run in CI against a real Postgres container via testcontainers
- Added table-level assertions for `checkpoint_writes` and `checkpoint_blobs` deletion on both sync and async Postgres paths
- Added advisory lock contention test with two real `PostgresSaver` connections
- Added incremental cursor collection tests for Postgres
- Added `writes` table deletion assertions for SQLite sync and async paths
- Added cross-contamination test verifying surviving threads are not affected by a sweep
- Added policy resolver tests that construct fresh `TTLPolicy` instances per call, covering the id()-cache collision path

## 0.0.1

Initial release.

### Added

- `TTLPolicy` - frozen dataclass with `idle_ttl_seconds` and `hard_age_ttl_seconds` fields; a thread expires as soon as either rule fires
- `Sweeper` - sidecar that sweeps expired threads from any `BaseCheckpointSaver` without intercepting reads or writes
- `sweep()` / `asweep()` - synchronous and asynchronous single-cycle sweep, both return a `SweepResult`
- `start()` / `stop()` - background loop that calls `asweep()` on a fixed interval
- `dry_run` mode - identifies expired threads without deleting them
- `safe_delete` - re-reads each thread's latest checkpoint immediately before deletion and skips it if a newer one has appeared since the scan
- `policy_resolver` - per-thread TTL override; return a `TTLPolicy`, `PolicyOverride.EXEMPT`, or `PolicyOverride.USE_DEFAULT`
- `on_before_delete` / `on_sweep_complete` callbacks
- `MemoryStrategy` - scans `InMemorySaver` storage directly
- `SqliteStrategy` / `AsyncSqliteStrategy` - single `GROUP BY` query against the SQLite checkpoints table
- `PostgresStrategy` / `AsyncPostgresStrategy` - same approach for PostgreSQL; supports `ANY` batch deletes
- Cursor-based incremental collection - subsequent sweeps only scan checkpoints newer than the previous cycle's max ID
- PostgreSQL advisory lock coordination - `enable_coordination=True` ensures only one sweeper instance runs at a time across multiple processes
- UUIDv6 timestamp encoding and decoding for threshold comparisons without deserialising checkpoint blobs
- CI across Python 3.11, 3.12, and 3.13; release and PyPI publish workflows
- MIT license

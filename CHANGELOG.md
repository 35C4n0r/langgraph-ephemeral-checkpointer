# Changelog

## 0.0.1 - 2026-05-24

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

import asyncio
import logging
import time

from langgraph.checkpoint.base import BaseCheckpointSaver

from . import _strategies
from ._coordination import AdvisoryLock, get_advisory_lock
from ._strategies import Strategy, ThreadTimestamps
from ._uuid6 import unix_to_uuid6, uuid6_to_unix
from .policy import TTLPolicy
from .result import SweepResult
from .types import OnBeforeDelete, OnSweepComplete, PolicyOverride, PolicyResolver

logger = logging.getLogger(__name__)

_REASON_IDLE = "idle_ttl"
_REASON_AGE = "hard_age_ttl"

# (thread_id, timestamps, human_description)
_DeleteItem = tuple[str, ThreadTimestamps, str]


class Sweeper:
    """Deletes expired LangGraph threads from any BaseCheckpointSaver.

    Runs as a sidecar: the graph talks directly to the checkpointer while the
    sweeper independently cleans threads on its own schedule based on a TTLPolicy.
    """

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
            _strategy: Strategy | None = None,
    ) -> None:
        """
        Args:
            checkpointer: The LangGraph checkpointer to sweep.
            policy: Default TTL policy applied to all threads.
            policy_resolver: Optional per-thread policy override. Return a
                TTLPolicy, PolicyOverride.EXEMPT, or PolicyOverride.USE_DEFAULT.
            enable_coordination: Acquire a PostgreSQL advisory lock before each
                sweep so only one instance runs at a time. No-op for non-Postgres
                backends.
            safe_delete: Re-fetch each thread's latest checkpoint immediately
                before deletion and skip it if a newer one has appeared.
            on_before_delete: Called with (thread_id, policy, reason) before
                each deletion. Return False to veto.
            on_sweep_complete: Called with the SweepResult after each sweep.
        """
        self._checkpointer = checkpointer
        self._policy = policy
        self._policy_resolver = policy_resolver
        self._safe_delete = safe_delete
        self._on_before_delete = on_before_delete
        self._on_sweep_complete = on_sweep_complete
        self._strategy: Strategy = _strategies.detect(checkpointer) if _strategy is None else _strategy
        self._advisory_lock: AdvisoryLock | None = get_advisory_lock(checkpointer, enable_coordination)
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None
        self._interval_seconds: int = 300

        # Incremental index carried across sweep cycles.
        # _index: per-thread timestamps accumulated from all sweeps so far.
        # _cursor: max checkpoint_id (UUIDv6 string) seen on the last sweep,
        #          or None → triggers a full scan on the next cycle.
        self._index: dict[str, ThreadTimestamps] = {}
        self._cursor: str | None = None


    def sweep(self, *, dry_run: bool = False) -> SweepResult:
        """Run one sweep cycle synchronously.

        Args:
            dry_run: Identify expired threads without deleting them. IDs still
                appear in the returned SweepResult.

        Returns:
            SweepResult with deleted thread IDs, active thread count, and
            wall-clock duration.
        """
        start = time.monotonic()
        lock_acquired = False
        if self._advisory_lock is not None:
            lock_acquired = self._advisory_lock.try_acquire()
            if not lock_acquired:
                logger.info("Advisory lock held by another sweeper instance; skipping this cycle")
                return SweepResult(
                    deleted_thread_ids=[],
                    active_thread_count=0,
                    sweep_duration_seconds=time.monotonic() - start,
                )
        try:
            return self._run_sweep(dry_run=dry_run)
        finally:
            if lock_acquired and self._advisory_lock is not None:
                self._advisory_lock.release()

    async def asweep(self, *, dry_run: bool = False) -> SweepResult:
        """Async variant of sweep().

        Args:
            dry_run: Identify expired threads without deleting them. IDs still
                appear in the returned SweepResult.

        Returns:
            SweepResult with deleted thread IDs, active thread count, and
            wall-clock duration.
        """
        start = time.monotonic()
        lock_acquired = False
        if self._advisory_lock is not None:
            lock_acquired = await self._advisory_lock.atry_acquire()
            if not lock_acquired:
                logger.info("Advisory lock held by another sweeper instance; skipping this cycle")
                return SweepResult(
                    deleted_thread_ids=[],
                    active_thread_count=0,
                    sweep_duration_seconds=time.monotonic() - start,
                )
        try:
            return await self._arun_sweep(dry_run=dry_run)
        finally:
            if lock_acquired and self._advisory_lock is not None:
                await self._advisory_lock.arelease()

    async def start(self, interval_seconds: int = 300) -> None:
        """Start a background task that calls asweep() on a fixed interval.

        Args:
            interval_seconds: Seconds to wait between sweep cycles. Defaults to 300.

        Raises:
            RuntimeError: If the sweeper is already running. Call stop() first.
        """
        if self._task is not None and not self._task.done():
            raise RuntimeError("Sweeper already running; call stop() first")
        self._interval_seconds = interval_seconds
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop the background sweep loop and wait for the current cycle to finish."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None and not self._task.done():
            await self._task
        self._task = None
        self._stop_event = None


    async def _loop(self) -> None:
        assert self._stop_event is not None
        while True:
            await self.asweep()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._interval_seconds,
                )
                break
            except TimeoutError:
                pass


    def _merge_updates(self, updates: dict[str, ThreadTimestamps]) -> None:
        for tid, ts in updates.items():
            if tid in self._index:
                old = self._index[tid]
                self._index[tid] = ThreadTimestamps(
                    latest_id=max(old.latest_id, ts.latest_id),
                    earliest_id=old.earliest_id,
                )
            else:
                self._index[tid] = ts

    def _drop_from_index(self, thread_ids: list[str]) -> None:
        for tid in thread_ids:
            self._index.pop(tid, None)


    def _run_sweep(self, *, dry_run: bool) -> SweepResult:
        start = time.monotonic()

        updates, new_cursor = self._strategy.collect(self._cursor)
        if self._cursor is None:
            self._index = updates
        else:
            self._merge_updates(updates)
        if new_cursor is not None:
            self._cursor = new_cursor

        now = time.time()
        dry_ids, to_delete = self._plan(self._index, now, dry_run)

        deleted_ids = list(dry_ids)

        survivors: list[str] = []
        for tid, ts, human in to_delete:
            if self._safe_delete and not self._safe_check_sync(tid, ts):
                logger.debug("Skipping thread_id=%s: timestamp changed since sweep start", tid)
                continue
            logger.debug("Deleting thread_id=%s (%s)", tid, human)
            survivors.append(tid)

        if survivors:
            self._strategy.batch_delete(survivors, self._checkpointer)
            deleted_ids.extend(survivors)

        if not dry_run:
            self._drop_from_index(deleted_ids)

        return self._build_result(deleted_ids, start, dry_run)

    async def _arun_sweep(self, *, dry_run: bool) -> SweepResult:
        start = time.monotonic()

        updates, new_cursor = await self._strategy.acollect(self._cursor)
        if self._cursor is None:
            self._index = updates
        else:
            self._merge_updates(updates)
        if new_cursor is not None:
            self._cursor = new_cursor

        now = time.time()
        dry_ids, to_delete = self._plan(self._index, now, dry_run)

        deleted_ids = list(dry_ids)

        survivors: list[str] = []
        for tid, ts, human in to_delete:
            if self._safe_delete and not await self._safe_check_async(tid, ts):
                logger.debug("Skipping thread_id=%s: timestamp changed since sweep start", tid)
                continue
            logger.debug("Deleting thread_id=%s (%s)", tid, human)
            survivors.append(tid)

        if survivors:
            await self._strategy.abatch_delete(survivors, self._checkpointer)
            deleted_ids.extend(survivors)

        if not dry_run:
            self._drop_from_index(deleted_ids)

        return self._build_result(deleted_ids, start, dry_run)


    def _plan(
            self,
            timestamps: dict[str, ThreadTimestamps],
            now: float,
            dry_run: bool,
    ) -> tuple[list[str], list[_DeleteItem]]:
        dry_ids: list[str] = []
        to_delete: list[_DeleteItem] = []

        # Threshold UUIDs are computed once per distinct policy object, not once per thread.
        _cache: dict[int, tuple[str | None, str | None]] = {}

        def _thresholds(policy: TTLPolicy) -> tuple[str | None, str | None]:
            pid = id(policy)
            if pid not in _cache:
                idle = unix_to_uuid6(now - policy.idle_ttl_seconds) if policy.idle_ttl_seconds is not None else None
                age = unix_to_uuid6(now - policy.hard_age_ttl_seconds) if policy.hard_age_ttl_seconds is not None else None
                _cache[pid] = (idle, age)
            return _cache[pid]

        for tid, ts in timestamps.items():
            policy = self._resolve_policy(tid)
            if policy is None:
                continue

            idle_threshold, age_threshold = _thresholds(policy)

            reason_code: str | None = None
            if idle_threshold is not None and ts.latest_id < idle_threshold:
                reason_code = _REASON_IDLE
            elif age_threshold is not None and ts.earliest_id < age_threshold:
                reason_code = _REASON_AGE

            if reason_code:
                human = self._expiry_human(ts, now, policy)
                if dry_run:
                    logger.debug("[DRY RUN] Would delete thread_id=%s (%s)", tid, human)
                    dry_ids.append(tid)
                    continue
                if not self._fire_on_before_delete(tid, policy, reason_code):
                    continue
                to_delete.append((tid, ts, human))

        return dry_ids, to_delete

    def _build_result(
            self,
            deleted_ids: list[str],
            start: float,
            dry_run: bool,
    ) -> SweepResult:
        result = SweepResult(
            deleted_thread_ids=deleted_ids,
            active_thread_count=len(self._index),
            sweep_duration_seconds=time.monotonic() - start,
        )
        self._log_result(result, dry_run)
        self._fire_on_sweep_complete(result)
        return result


    def _resolve_policy(self, thread_id: str) -> TTLPolicy | None:
        if self._policy_resolver is None:
            return self._policy
        override = self._policy_resolver(thread_id)
        if isinstance(override, TTLPolicy):
            return override
        if override is PolicyOverride.EXEMPT:
            return None
        return self._policy

    @staticmethod
    def _expiry_human(
            ts: ThreadTimestamps, now: float, policy: TTLPolicy
    ) -> str:
        if policy.idle_ttl_seconds is not None:
            idle = now - uuid6_to_unix(ts.latest_id)
            if idle > policy.idle_ttl_seconds:
                return f"idle {idle:.0f}s > {policy.idle_ttl_seconds}s limit"
        if policy.hard_age_ttl_seconds is not None:
            age = now - uuid6_to_unix(ts.earliest_id)
            if age > policy.hard_age_ttl_seconds:
                return f"age {age:.0f}s > {policy.hard_age_ttl_seconds}s limit"
        return "expired"

    def _safe_check_sync(self, thread_id: str, original: ThreadTimestamps) -> bool:
        current = self._checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})
        if current is None:
            return False
        return current.checkpoint["id"] <= original.latest_id

    async def _safe_check_async(self, thread_id: str, original: ThreadTimestamps) -> bool:
        current = await self._checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
        if current is None:
            return False
        return current.checkpoint["id"] <= original.latest_id

    def _fire_on_before_delete(self, thread_id: str, policy: TTLPolicy, reason: str) -> bool:
        if self._on_before_delete is None:
            return True
        return self._on_before_delete(thread_id, policy, reason)

    def _fire_on_sweep_complete(self, result: SweepResult) -> None:
        if self._on_sweep_complete is None:
            return
        self._on_sweep_complete(result)

    @staticmethod
    def _log_result(result: SweepResult, dry_run: bool) -> None:
        if dry_run:
            logger.info(
                "[DRY RUN] Would expire %d threads, %d active (%.2fs)",
                len(result.deleted_thread_ids),
                result.active_thread_count,
                result.sweep_duration_seconds,
            )
        else:
            logger.info(
                "Sweep complete: %d expired, %d active (%.2fs)",
                len(result.deleted_thread_ids),
                result.active_thread_count,
                result.sweep_duration_seconds,
            )

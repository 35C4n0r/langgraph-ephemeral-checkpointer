import logging
import time

from langgraph.checkpoint.base import BaseCheckpointSaver

from . import _strategies
from ._strategies import Strategy, ThreadTimestamps
from ._uuid6 import uuid6_to_unix, unix_to_uuid6
from .policy import TTLPolicy
from .result import SweepResult
from .types import OnBeforeDelete, OnSweepComplete

logger = logging.getLogger(__name__)

_REASON_IDLE = "idle_ttl"
_REASON_AGE = "hard_age_ttl"

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
            safe_delete: bool = True,
            on_before_delete: OnBeforeDelete | None = None,
            on_sweep_complete: OnSweepComplete | None = None,
            _strategy: Strategy | None = None,
    ) -> None:
        self._checkpointer = checkpointer
        self._policy = policy
        self._safe_delete = safe_delete
        self._on_before_delete = on_before_delete
        self._on_sweep_complete = on_sweep_complete
        self._strategy: Strategy = _strategies.detect(checkpointer) if _strategy is None else _strategy

    def sweep(self, *, dry_run: bool = False) -> SweepResult:
        """Run one sweep cycle synchronously."""
        return self._run_sweep(dry_run=dry_run)

    async def asweep(self, *, dry_run: bool = False) -> SweepResult:
        """Async variant of sweep(). Prefer this when using an async checkpointer."""
        return await self._arun_sweep(dry_run=dry_run)

    def _run_sweep(self, *, dry_run: bool) -> SweepResult:
        start = time.monotonic()
        threads = self._strategy.collect()
        now = time.time()
        dry_ids, to_delete = self._plan(threads, now, dry_run)
        deleted_ids = list(dry_ids)
        if not dry_run:
            tids: list[str] = []
            for tid, ts, human in to_delete:
                if self._safe_delete and not self._safe_check_sync(tid, ts):
                    logger.debug("Skipping thread_id=%s: timestamp changed since sweep start", tid)
                    continue
                logger.debug("Deleting thread_id=%s (%s)", tid, human)
                tids.append(tid)
            if tids:
                self._strategy.batch_delete(tids, self._checkpointer)
            deleted_ids.extend(tids)
        return self._build_result(threads, deleted_ids, start, dry_run)

    async def _arun_sweep(self, *, dry_run: bool) -> SweepResult:
        start = time.monotonic()
        threads = await self._strategy.acollect()
        now = time.time()
        dry_ids, to_delete = self._plan(threads, now, dry_run)
        deleted_ids = list(dry_ids)
        if not dry_run:
            tids: list[str] = []
            for tid, ts, human in to_delete:
                if self._safe_delete and not await self._safe_check_async(tid, ts):
                    logger.debug("Skipping thread_id=%s: timestamp changed since sweep start", tid)
                    continue
                logger.debug("Deleting thread_id=%s (%s)", tid, human)
                tids.append(tid)
            if tids:
                await self._strategy.abatch_delete(tids, self._checkpointer)
            deleted_ids.extend(tids)
        return self._build_result(threads, deleted_ids, start, dry_run)

    def _build_result(self, threads, deleted_ids, start, dry_run) -> SweepResult:
        result = SweepResult(
            deleted_thread_ids=deleted_ids,
            active_thread_count=len(threads) - len(deleted_ids),
            sweep_duration_seconds=time.monotonic() - start,
        )
        if dry_run:
            logger.info("[DRY RUN] Would expire %d threads, %d active (%.2fs)",
                        len(result.deleted_thread_ids), result.active_thread_count,
                        result.sweep_duration_seconds)
        else:
            logger.info("Sweep complete: %d expired, %d active (%.2fs)",
                        len(result.deleted_thread_ids), result.active_thread_count,
                        result.sweep_duration_seconds)
        self._fire_on_sweep_complete(result)
        return result

    def _plan(self, timestamps, now, dry_run):
        dry_ids: list[str] = []
        to_delete: list[_DeleteItem] = []
        for tid, ts in timestamps.items():
            reason_code = self._expiry_reason_code(ts, now, self._policy)
            if reason_code:
                human = self._expiry_human(ts, now, self._policy)
                if dry_run:
                    logger.debug("[DRY RUN] Would delete thread_id=%s (%s)", tid, human)
                    dry_ids.append(tid)
                else:
                    if not self._fire_on_before_delete(tid, self._policy, reason_code):
                        continue
                    to_delete.append((tid, ts, human))
        return dry_ids, to_delete

    def _expiry_reason_code(self, ts, now, policy) -> str | None:
        if policy.idle_ttl_seconds is not None and ts.latest_id < unix_to_uuid6(now - policy.idle_ttl_seconds):
            return _REASON_IDLE
        if policy.hard_age_ttl_seconds is not None and ts.earliest_id < unix_to_uuid6(now - policy.hard_age_ttl_seconds):
            return _REASON_AGE
        return None

    @staticmethod
    def _expiry_human(ts, now, policy) -> str:
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

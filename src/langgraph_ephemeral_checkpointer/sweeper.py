import logging
import time

from langgraph.checkpoint.base import BaseCheckpointSaver

from . import _strategies
from ._strategies import Strategy, ThreadTimestamps
from ._uuid6 import uuid6_to_unix, unix_to_uuid6
from .policy import TTLPolicy
from .result import SweepResult

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
            _strategy: Strategy | None = None,
    ) -> None:
        self._checkpointer = checkpointer
        self._policy = policy
        self._strategy: Strategy = _strategies.detect(checkpointer) if _strategy is None else _strategy

    def sweep(self) -> SweepResult:
        """Run one sweep cycle synchronously."""
        return self._run_sweep()

    def _run_sweep(self) -> SweepResult:
        start = time.monotonic()
        threads = self._strategy.collect()
        now = time.time()
        to_delete = self._plan(threads, now)
        deleted_ids: list[str] = []
        for tid, ts, human in to_delete:
            logger.debug("Deleting thread_id=%s (%s)", tid, human)
            deleted_ids.append(tid)
        if deleted_ids:
            self._strategy.batch_delete(deleted_ids, self._checkpointer)
        result = SweepResult(
            deleted_thread_ids=deleted_ids,
            active_thread_count=len(threads) - len(deleted_ids),
            sweep_duration_seconds=time.monotonic() - start,
        )
        logger.info(
            "Sweep complete: %d expired, %d active (%.2fs)",
            len(result.deleted_thread_ids),
            result.active_thread_count,
            result.sweep_duration_seconds,
        )
        return result

    def _plan(
            self,
            timestamps: dict[str, ThreadTimestamps],
            now: float,
    ) -> list[_DeleteItem]:
        to_delete: list[_DeleteItem] = []
        for tid, ts in timestamps.items():
            reason_code = self._expiry_reason_code(ts, now, self._policy)
            if reason_code:
                human = self._expiry_human(ts, now, self._policy)
                to_delete.append((tid, ts, human))
        return to_delete

    def _expiry_reason_code(
            self, ts: ThreadTimestamps, now: float, policy: TTLPolicy
    ) -> str | None:
        if policy.idle_ttl_seconds is not None and ts.latest_id < unix_to_uuid6(now - policy.idle_ttl_seconds):
            return _REASON_IDLE
        if policy.hard_age_ttl_seconds is not None and ts.earliest_id < unix_to_uuid6(now - policy.hard_age_ttl_seconds):
            return _REASON_AGE
        return None

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

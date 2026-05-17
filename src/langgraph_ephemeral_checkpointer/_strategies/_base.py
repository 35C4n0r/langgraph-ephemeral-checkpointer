from abc import ABC, abstractmethod
from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver


@dataclass
class ThreadTimestamps:
    """UUIDv6 checkpoint IDs bracketing a thread's activity window.

    Both fields are UUIDv6 strings and compare correctly with plain ``<`` / ``>``,
    so expiry thresholds can be applied without decoding to unix timestamps.
    """

    latest_id: str
    earliest_id: str


class Strategy(ABC):
    """Backend-specific implementation of checkpoint scanning and deletion.

    Subclass this to support a new checkpointer type. The Sweeper calls these
    methods; it never touches the checkpointer directly except through the
    batch_delete/abatch_delete helpers.
    """

    @abstractmethod
    def collect(
            self, cursor: str | None
    ) -> tuple[dict[str, ThreadTimestamps], str | None]:
        """Return thread timestamps, optionally filtered to activity since cursor.

        Args:
            cursor: Max checkpoint_id (UUIDv6 string) seen on the previous sweep,
                or None for a full scan.

        Returns:
            A tuple of (threads, new_cursor) where threads maps thread_id to
            ThreadTimestamps for threads active since cursor (or all threads when
            cursor is None), and new_cursor is the max checkpoint_id seen this
            cycle (None if the backend has no cursor support).
        """
        ...

    @abstractmethod
    async def acollect(
            self, cursor: str | None
    ) -> tuple[dict[str, ThreadTimestamps], str | None]:
        """Async variant of collect()."""
        ...

    def batch_delete(
            self,
            thread_ids: list[str],
            checkpointer: BaseCheckpointSaver,
    ) -> None:
        """Delete multiple threads in one operation.

        Default: sequential delete_thread() loop, correct for any backend.
        SQL strategies override with a single IN / ANY statement.
        """
        for tid in thread_ids:
            checkpointer.delete_thread(tid)

    async def abatch_delete(
            self,
            thread_ids: list[str],
            checkpointer: BaseCheckpointSaver,
    ) -> None:
        """Async variant of batch_delete()."""
        for tid in thread_ids:
            await checkpointer.adelete_thread(tid)

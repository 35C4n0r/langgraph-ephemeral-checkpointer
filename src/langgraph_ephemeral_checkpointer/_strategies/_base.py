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
    """Backend-specific checkpoint scanning and deletion. Subclass to add a new backend."""

    @abstractmethod
    def collect(
            self, cursor: str | None
    ) -> tuple[dict[str, ThreadTimestamps], str | None]:
        """Scan checkpoints and return per-thread timestamps.

        Args:
            cursor: Max checkpoint_id seen on the previous sweep, or None for a
                full scan.

        Returns:
            Tuple of (threads, new_cursor). threads maps thread_id to
            ThreadTimestamps for threads active since cursor. new_cursor is the
            max checkpoint_id seen this cycle, or None if unsupported.
        """
        ...

    @abstractmethod
    async def acollect(
            self, cursor: str | None
    ) -> tuple[dict[str, ThreadTimestamps], str | None]:
        """Async variant of collect().

        Args:
            cursor: Max checkpoint_id seen on the previous sweep, or None for a
                full scan.

        Returns:
            Tuple of (threads, new_cursor). threads maps thread_id to
            ThreadTimestamps for threads active since cursor. new_cursor is the
            max checkpoint_id seen this cycle, or None if unsupported.
        """
        ...

    def batch_delete(
            self,
            thread_ids: list[str],
            checkpointer: BaseCheckpointSaver,
    ) -> None:
        """Delete threads. SQL strategies override with a single IN/ANY query.

        Args:
            thread_ids: IDs of threads to delete.
            checkpointer: The checkpointer to delete from.
        """
        for tid in thread_ids:
            checkpointer.delete_thread(tid)

    async def abatch_delete(
            self,
            thread_ids: list[str],
            checkpointer: BaseCheckpointSaver,
    ) -> None:
        """Async variant of batch_delete().

        Args:
            thread_ids: IDs of threads to delete.
            checkpointer: The checkpointer to delete from.
        """
        for tid in thread_ids:
            await checkpointer.adelete_thread(tid)

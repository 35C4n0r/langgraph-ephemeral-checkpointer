from abc import ABC, abstractmethod
from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver


@dataclass
class ThreadTimestamps:
    latest_id: str
    earliest_id: str


class Strategy(ABC):
    @abstractmethod
    def collect(self) -> dict[str, "ThreadTimestamps"]: ...

    @abstractmethod
    async def acollect(self) -> dict[str, "ThreadTimestamps"]: ...

    def batch_delete(
            self,
            thread_ids: list[str],
            checkpointer: BaseCheckpointSaver,
    ) -> None:
        for tid in thread_ids:
            checkpointer.delete_thread(tid)

    async def abatch_delete(
            self,
            thread_ids: list[str],
            checkpointer: BaseCheckpointSaver,
    ) -> None:
        for tid in thread_ids:
            await checkpointer.adelete_thread(tid)

from langgraph.checkpoint.memory import InMemorySaver

from ._base import Strategy, ThreadTimestamps


class MemoryStrategy(Strategy):
    """Optimised strategy for InMemorySaver: reads storage dict directly,
    extracting timestamps from UUIDv6 checkpoint IDs without deserialising blobs."""

    def __init__(self, checkpointer: InMemorySaver) -> None:
        self._checkpointer = checkpointer

    def collect(
            self, cursor: str | None
    ) -> tuple[dict[str, ThreadTimestamps], str | None]:
        updates: dict[str, ThreadTimestamps] = {}
        global_max: str | None = None

        for thread_id, ns_dict in self._checkpointer.storage.items():
            thread_max: str | None = None
            thread_min: str | None = None

            for cp_dict in ns_dict.values():
                for cp_id in cp_dict:
                    cp_str = str(cp_id)
                    if thread_max is None or cp_str > thread_max:
                        thread_max = cp_str
                    if thread_min is None or cp_str < thread_min:
                        thread_min = cp_str

            if thread_max is None:
                continue
            assert thread_min is not None

            if global_max is None or thread_max > global_max:
                global_max = thread_max

            if cursor is None or thread_max > cursor:
                updates[thread_id] = ThreadTimestamps(
                    latest_id=thread_max,
                    earliest_id=thread_min,
                )

        return updates, global_max

    async def acollect(
            self, cursor: str | None
    ) -> tuple[dict[str, ThreadTimestamps], str | None]:
        return self.collect(cursor)

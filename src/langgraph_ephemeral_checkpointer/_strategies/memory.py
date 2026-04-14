from langgraph.checkpoint.memory import InMemorySaver

from ._base import Strategy, ThreadTimestamps


class MemoryStrategy(Strategy):
    """Optimised strategy for InMemorySaver: reads storage dict directly,
    extracting timestamps from UUIDv6 checkpoint IDs without deserialising blobs."""

    def __init__(self, checkpointer: InMemorySaver) -> None:
        self._checkpointer = checkpointer

    def collect(self) -> dict[str, ThreadTimestamps]:
        result: dict[str, ThreadTimestamps] = {}

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

            result[thread_id] = ThreadTimestamps(
                latest_id=thread_max,
                earliest_id=thread_min,
            )

        return result

    async def acollect(self) -> dict[str, ThreadTimestamps]:
        return self.collect()

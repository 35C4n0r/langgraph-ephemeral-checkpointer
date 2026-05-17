from dataclasses import dataclass


@dataclass
class SweepResult:
    """Outcome of a single sweep cycle.

    Attributes:
        deleted_thread_ids: IDs of threads that were deleted (or would have
            been deleted when running in dry-run mode).
        active_thread_count: Number of threads still alive after the sweep.
        sweep_duration_seconds: Wall-clock time the sweep took, in seconds.
    """

    deleted_thread_ids: list[str]
    active_thread_count: int
    sweep_duration_seconds: float

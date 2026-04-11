from dataclasses import dataclass


@dataclass
class SweepResult:
    deleted_thread_ids: list[str]
    active_thread_count: int
    sweep_duration_seconds: float

from dataclasses import dataclass


@dataclass(frozen=True)
class TTLPolicy:
    """Immutable retention policy applied to LangGraph threads.

    At least one TTL field must be set. Both may be combined; a thread
    expires as soon as either rule fires.

    Attributes:
        idle_ttl_seconds: Expire a thread whose most-recent checkpoint is
            older than this many seconds.
        hard_age_ttl_seconds: Expire a thread whose oldest checkpoint is
            older than this many seconds, regardless of recent activity.
    """

    idle_ttl_seconds: int | None = None
    hard_age_ttl_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.idle_ttl_seconds is None and self.hard_age_ttl_seconds is None:
            raise ValueError("At least one of idle_ttl_seconds or hard_age_ttl_seconds must be set")
        if self.idle_ttl_seconds is not None and self.idle_ttl_seconds <= 0:
            raise ValueError("idle_ttl_seconds must be positive")
        if self.hard_age_ttl_seconds is not None and self.hard_age_ttl_seconds <= 0:
            raise ValueError("hard_age_ttl_seconds must be positive")

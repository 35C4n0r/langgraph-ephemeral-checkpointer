from dataclasses import dataclass


@dataclass(frozen=True)
class TTLPolicy:
    idle_ttl_seconds: int | None = None
    hard_age_ttl_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.idle_ttl_seconds is None and self.hard_age_ttl_seconds is None:
            raise ValueError("At least one of idle_ttl_seconds or hard_age_ttl_seconds must be set")
        if self.idle_ttl_seconds is not None and self.idle_ttl_seconds <= 0:
            raise ValueError("idle_ttl_seconds must be positive")
        if self.hard_age_ttl_seconds is not None and self.hard_age_ttl_seconds <= 0:
            raise ValueError("hard_age_ttl_seconds must be positive")

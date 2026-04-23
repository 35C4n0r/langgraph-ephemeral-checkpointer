from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .policy import TTLPolicy
    from .result import SweepResult


class PolicyOverride(Enum):
    """Sentinel values returned by a PolicyResolver."""

    USE_DEFAULT = "use_default"
    """Apply the sweeper's global TTLPolicy to this thread."""

    EXEMPT = "exempt"
    """This thread never expires."""


# Called before deleting a thread.  Returns True to proceed, False to skip.
# Args: thread_id, effective TTLPolicy, reason string (e.g. "idle_ttl", "hard_age_ttl").
OnBeforeDelete = Callable[[str, "TTLPolicy", str], bool]

# Called once after every sweep cycle with the final SweepResult.
OnSweepComplete = Callable[["SweepResult"], None]

__all__ = ["PolicyOverride", "OnBeforeDelete", "OnSweepComplete"]

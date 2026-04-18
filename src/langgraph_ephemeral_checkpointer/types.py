from enum import Enum


class PolicyOverride(Enum):
    """Sentinel values returned by a PolicyResolver."""

    USE_DEFAULT = "use_default"
    """Apply the sweeper's global TTLPolicy to this thread."""

    EXEMPT = "exempt"
    """This thread never expires."""

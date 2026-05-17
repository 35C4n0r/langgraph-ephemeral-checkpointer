
import uuid as _uuid_mod

# UUIDv6 bit layout: https://www.rfc-editor.org/rfc/rfc9562#section-5.6
# Epoch offset (100-ns intervals, 1582-10-15 → 1970-01-01): https://www.rfc-editor.org/rfc/rfc4122#section-4.1.4
# Bit manipulation derived from CPython Lib/uuid.py (uuid6, added in 3.14):
# CopyPasta from https://github.com/python/cpython/blob/acefff95eab3db6b7cf837f3ce2707bbf9199376/Lib/uuid.py#L796
_UUID_EPOCH_OFFSET = 0x01b21dd213814000
_RFC_4122_VERSION_6_FLAGS = (0x6 << 76) | (0b10 << 62)  # version=6, variant=RFC 4122


def uuid6_to_unix(uuid_str: str | object) -> float:
    """Extract a unix timestamp (seconds) from a UUIDv6 string."""
    u = _uuid_mod.UUID(str(uuid_str))
    i = u.int
    time_hi_and_mid = (i >> 80) & 0xffff_ffff_ffff
    time_lo = (i >> 64) & 0x0fff
    timestamp = (time_hi_and_mid << 12) | time_lo
    return (timestamp - _UUID_EPOCH_OFFSET) * 100 / 1e9


def unix_to_uuid6(ts: float) -> str:
    """Create a UUIDv6 string from a unix timestamp for threshold comparisons.

    node and clock_seq are zeroed so this is the smallest valid UUID for the
    given timestamp; any real checkpoint at that moment compares greater.
    """
    timestamp = int(ts * 1e9 / 100) + _UUID_EPOCH_OFFSET
    time_hi_and_mid = (timestamp >> 12) & 0xffff_ffff_ffff
    time_lo = timestamp & 0x0fff
    int_uuid_6 = time_hi_and_mid << 80
    int_uuid_6 |= time_lo << 64
    # node=0, clock_seq=0: minimum UUID for this timestamp
    int_uuid_6 |= _RFC_4122_VERSION_6_FLAGS
    return str(_uuid_mod.UUID(int=int_uuid_6))

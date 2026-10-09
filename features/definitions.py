"""Feature names and constants, declared once for the online and the batch implementations.

Every feature describes a transaction using only what the user did BEFORE it.
"""

from schema import MERCHANT, P2P, TOPUP

WINDOWS = {"5m": 5 * 60, "1h": 3600, "24h": 24 * 3600}  # seconds
HISTORY_SECONDS = max(WINDOWS.values())  # how much recent history a user's state must keep

NIGHT_END_HOUR = 6  # is_night = hour in [0, 6)
MAX_GAP_SECONDS = 30 * 24 * 3600  # cap for seconds_since_prev, also used when there is no previous

TYPE_FLAGS = {"type_p2p": P2P, "type_merchant": MERCHANT, "type_topup": TOPUP}

FEATURES = [
    "amount_tnd",
    *TYPE_FLAGS,
    *(f"{stat}_{w}" for w in WINDOWS for stat in ("count", "sum", "mean")),
    "amount_ratio",  # amount / mean of the user's previous amounts (1 if no history)
    "is_new_device",  # 0 when the user has no history: nothing to compare to
    "is_new_city",
    "is_new_recipient",
    "hour",
    "is_night",
    "seconds_since_prev",
]

"""Features for ONE transaction given the user's state. This is the reference implementation,
used for online scoring; `features.batch` must give the same values (see the parity test).

Usage, in this order, for each incoming transaction:
    features = compute_features(tx, state)   # reads the past only
    update_state(state, tx)                  # then the transaction becomes part of the past
"""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from features.definitions import (
    FEATURES,
    HISTORY_SECONDS,
    MAX_GAP_SECONDS,
    NIGHT_END_HOUR,
    TYPE_FLAGS,
    WINDOWS,
)
from simulator.schema import P2P


@dataclass
class UserState:
    """Everything the features need to know about a user's past."""

    recent: deque = field(default_factory=deque)  # (epoch seconds, amount), last 24 h, oldest first
    devices: set = field(default_factory=set)
    cities: set = field(default_factory=set)
    recipients: set = field(default_factory=set)
    tx_count: int = 0  # all-time
    amount_sum: float = 0.0  # all-time
    last_ts: float | None = None


def epoch_seconds(timestamp) -> float:
    return pd.Timestamp(timestamp).value / 1e9


def compute_features(tx: Mapping, state: UserState) -> dict[str, float]:
    """Features of `tx`, using only `state` (the user's transactions before `tx`)."""
    timestamp = pd.Timestamp(tx["timestamp"])
    ts = timestamp.value / 1e9
    amount = float(tx["amount_tnd"])
    has_history = state.tx_count > 0

    f = {"amount_tnd": amount}
    for name, tx_type in TYPE_FLAGS.items():
        f[name] = float(tx["type"] == tx_type)

    for name, seconds in WINDOWS.items():
        amounts = [a for t, a in state.recent if t >= ts - seconds]
        total = float(sum(amounts))
        f[f"count_{name}"] = float(len(amounts))
        f[f"sum_{name}"] = total
        f[f"mean_{name}"] = total / len(amounts) if amounts else 0.0

    f["amount_ratio"] = amount / (state.amount_sum / state.tx_count) if has_history else 1.0
    f["is_new_device"] = float(has_history and tx["device_id"] not in state.devices)
    f["is_new_city"] = float(has_history and tx["city"] not in state.cities)
    f["is_new_recipient"] = float(
        has_history and tx["type"] == P2P and tx["recipient_id"] not in state.recipients
    )
    f["hour"] = float(timestamp.hour)
    f["is_night"] = float(timestamp.hour < NIGHT_END_HOUR)
    gap = MAX_GAP_SECONDS if state.last_ts is None else max(0.0, ts - state.last_ts)
    f["seconds_since_prev"] = float(min(gap, MAX_GAP_SECONDS))
    return {name: f[name] for name in FEATURES}


def update_state(state: UserState, tx: Mapping) -> None:
    """Add `tx` to the user's past. Call AFTER compute_features."""
    ts = epoch_seconds(tx["timestamp"])
    amount = float(tx["amount_tnd"])
    state.recent.append((ts, amount))
    while state.recent[0][0] < ts - HISTORY_SECONDS:
        state.recent.popleft()
    state.devices.add(tx["device_id"])
    state.cities.add(tx["city"])
    if tx["type"] == P2P:
        state.recipients.add(tx["recipient_id"])
    state.tx_count += 1
    state.amount_sum += amount
    state.last_ts = ts


def replay(df: pd.DataFrame) -> pd.DataFrame:
    """Run transactions one by one, in time order, through the online functions.

    Slow; meant for tests and small samples. Returns features aligned with `df.index`.
    """
    states: dict[str, UserState] = {}
    rows = []
    ordered = df.sort_values("timestamp", kind="stable")
    for tx in ordered.to_dict("records"):
        state = states.setdefault(tx["user_id"], UserState())
        rows.append(compute_features(tx, state))
        update_state(state, tx)
    return pd.DataFrame(rows, index=ordered.index, columns=FEATURES).loc[df.index]

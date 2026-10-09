"""Offline, vectorized version of `features.online` for training speed.

No leakage by construction: each user's transactions are put in time order and every
feature of the row at position k only reads positions < k.
"""

import numpy as np
import pandas as pd

from features.definitions import FEATURES, MAX_GAP_SECONDS, NIGHT_END_HOUR, TYPE_FLAGS, WINDOWS
from simulator.schema import P2P


def compute_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Features for every transaction of `df`, aligned with `df.index`."""
    n = len(df)
    timestamps = df["timestamp"].astype("datetime64[ns]")
    ts_all = timestamps.astype("int64").to_numpy() / 1e9
    user_all = pd.factorize(df["user_id"])[0]

    # sort by (user, time); lexsort is stable, so ties keep the arrival order
    order = np.lexsort((ts_all, user_all))
    user, ts = user_all[order], ts_all[order]
    amount = df["amount_tnd"].to_numpy(dtype="float64")[order]

    n_prior = np.empty(n)  # number of previous transactions of the user
    prior_sum = np.empty(n)  # sum of their amounts
    gap = np.empty(n)
    window_count = {w: np.empty(n) for w in WINDOWS}
    window_sum = {w: np.empty(n) for w in WINDOWS}

    starts = np.flatnonzero(np.r_[True, user[1:] != user[:-1]])
    for s, e in zip(starts, np.r_[starts[1:], n], strict=True):
        t, a = ts[s:e], amount[s:e]
        k = np.arange(e - s)
        before = np.r_[0.0, np.cumsum(a)[:-1]]  # sum of the amounts at positions < k
        n_prior[s:e] = k
        prior_sum[s:e] = before
        gap[s:e] = np.r_[MAX_GAP_SECONDS, np.diff(t)]
        for w, seconds in WINDOWS.items():
            first = np.searchsorted(t, t - seconds, side="left")  # first position in the window
            window_count[w][s:e] = k - first
            window_sum[w][s:e] = before - before[first]

    has_history = n_prior > 0
    tx_type = df["type"].to_numpy()[order]
    keys = pd.DataFrame(
        {
            "user": user,
            "device": df["device_id"].to_numpy()[order],
            "city": df["city"].to_numpy()[order],
            "recipient": df["recipient_id"].to_numpy()[order],
        }
    )
    is_p2p = tx_type == P2P
    seen_recipient = np.zeros(n, dtype=bool)
    seen_recipient[is_p2p] = keys.loc[is_p2p, ["user", "recipient"]].duplicated().to_numpy()

    hour = timestamps.dt.hour.to_numpy()[order]
    out = {"amount_tnd": amount}
    for name, value in TYPE_FLAGS.items():
        out[name] = tx_type == value
    for w in WINDOWS:
        count = window_count[w]
        out[f"count_{w}"] = count
        out[f"sum_{w}"] = window_sum[w]
        out[f"mean_{w}"] = np.divide(window_sum[w], count, out=np.zeros(n), where=count > 0)
    out["amount_ratio"] = np.divide(amount * n_prior, prior_sum, out=np.ones(n), where=has_history)
    out["is_new_device"] = has_history & ~keys[["user", "device"]].duplicated().to_numpy()
    out["is_new_city"] = has_history & ~keys[["user", "city"]].duplicated().to_numpy()
    out["is_new_recipient"] = has_history & is_p2p & ~seen_recipient
    out["hour"] = hour
    out["is_night"] = hour < NIGHT_END_HOUR
    out["seconds_since_prev"] = np.minimum(gap, MAX_GAP_SECONDS)

    features = pd.DataFrame(out, index=df.index[order], columns=FEATURES).astype("float64")
    return features.loc[df.index]

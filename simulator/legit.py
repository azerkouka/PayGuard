"""Legitimate transactions, generated in a vectorized way.

`build_legit` works for any users and timestamps, so the streaming simulator can reuse it.
"""

import numpy as np
import pandas as pd

from simulator.profiles import N_CONTACTS, N_FAV_MERCHANTS, LifeEvents, Population
from simulator.schema import MERCHANT, P2P, TOPUP

IN_ACTIVE_HOURS_RATE = 0.93  # the rest happens at any hour
SECOND_DEVICE_RATE = 0.2  # for users owning two devices
NEW_RECIPIENT_RATE = 0.12  # P2P transfers to someone outside the usual contacts
NEW_MERCHANT_RATE = 0.15  # payments outside the favourite merchants
LARGE_RATE = 0.02  # legit large purchases / transfers (rent, appliances, tuition)
LARGE_MULTIPLIER = (3, 15)
MICRO_RATE = 0.06  # merchant payments of a few TND (coffee, transport)
MICRO_AMOUNT = (0.5, 5)
BURST_RATE = 0.02  # transactions followed by a few quick ones (vending, top-ups, bill splitting)
BURST_FOLLOW_UPS = (2, 6)
BURST_GAP_S = (10, 300)
BURST_INFLATION = 1 + BURST_RATE * sum(BURST_FOLLOW_UPS) / 2  # rows after / rows before bursts
MIN_AMOUNT = 0.5

TOPUP_AMOUNTS = [1, 2, 5, 10, 20, 50]
TOPUP_P = [0.15, 0.2, 0.3, 0.2, 0.1, 0.05]


def sample_users_and_times(
    pop: Population, n: int, start: np.datetime64, days: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Pick who transacts (by activity level) and when (mostly within their active hours)."""
    users = pop.users
    u = rng.choice(pop.n_users, n, p=users["activity"].to_numpy())
    active_start = users["active_start"].to_numpy()[u]
    active_len = (users["active_end"].to_numpy()[u] - active_start) % 24

    in_window = rng.random(n) < IN_ACTIVE_HOURS_RATE
    hour = np.where(in_window, (active_start + rng.random(n) * active_len) % 24, rng.random(n) * 24)
    seconds = rng.integers(0, days, n) * 86400 + (hour * 3600).astype(np.int64)
    return u, start + seconds.astype("timedelta64[s]")


def _cities(pop: Population, events: LifeEvents, u: np.ndarray, ts: np.ndarray):
    """Home city, replaced by the trip city while the user is travelling."""
    city = pop.users["home_city"].to_numpy()[u].copy()
    on_trip = np.zeros(len(u), dtype=bool)
    if len(events.trips):
        left = pd.DataFrame({"u": u, "ts": ts, "row": np.arange(len(u))}).sort_values("ts")
        trips = events.trips.sort_values("trip_start")
        merged = pd.merge_asof(left, trips, left_on="ts", right_on="trip_start", by="u")
        travelling = merged[merged["ts"] < merged["trip_end"]]
        rows = travelling["row"].to_numpy()
        city[rows] = travelling["trip_city"].to_numpy()
        on_trip[rows] = True
    return city, on_trip


def build_legit(
    pop: Population, events: LifeEvents, u: np.ndarray, ts: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """Legit transactions for user indices `u` at timestamps `ts` (datetime64[s])."""
    users = pop.users
    n = len(u)

    # type, from each user's preferences
    r = rng.random(n)
    p_p2p = users["p_p2p"].to_numpy()[u]
    p_merchant = users["p_merchant"].to_numpy()[u]
    tx_type = np.where(r < p_p2p, P2P, np.where(r < p_p2p + p_merchant, MERCHANT, TOPUP))
    is_p2p, is_merchant, is_topup = tx_type == P2P, tx_type == MERCHANT, tx_type == TOPUP

    # amount: log-normal with [amount_low, amount_high] as the 5th-95th percentile range
    log_low = np.log(users["amount_low"].to_numpy()[u])
    log_high = np.log(users["amount_high"].to_numpy()[u])
    sigma = (log_high - log_low) / (2 * 1.645)
    amount = np.exp((log_low + log_high) / 2 + sigma * rng.standard_normal(n))
    large = rng.random(n) < LARGE_RATE
    amount[large] *= rng.uniform(*LARGE_MULTIPLIER, large.sum())
    micro = is_merchant & (rng.random(n) < MICRO_RATE)
    amount[micro] = rng.uniform(*MICRO_AMOUNT, micro.sum())
    amount[is_topup] = rng.choice(TOPUP_AMOUNTS, is_topup.sum(), p=TOPUP_P)
    amount = np.round(np.maximum(amount, MIN_AMOUNT), 3)  # TND has 3 decimals (millimes)

    city, on_trip = _cities(pop, events, u, ts)

    # device: usual one, sometimes the second one, the new phone after a phone change
    use_second = (users["n_devices"].to_numpy()[u] == 2) & (rng.random(n) < SECOND_DEVICE_RATE)
    device = np.where(use_second, users["device_2"].to_numpy()[u], users["device_1"].to_numpy()[u])
    changed = ts >= events.phone_change_time[u]  # False when NaT
    device = np.where(changed, events.new_device[u], device)

    # recipient: a usual contact, sometimes someone new (never a mule account)
    recipient = pop.contacts[u, rng.integers(0, N_CONTACTS, n)]
    new_recipient = rng.random(n) < NEW_RECIPIENT_RATE
    recipient[new_recipient] = rng.choice(pop.non_mules, new_recipient.sum())
    to_self = recipient == u
    recipient[to_self] = pop.contacts[u[to_self], 0]

    # merchant: a favourite one, or any shop of the current city when travelling or exploring
    merchant = pop.fav_merchants[u, rng.integers(0, N_FAV_MERCHANTS, n)]
    elsewhere = on_trip | (rng.random(n) < NEW_MERCHANT_RATE)
    for c in np.unique(city[elsewhere]):
        mask = elsewhere & (city == c)
        merchant[mask] = rng.choice(pop.shops_by_city[c], mask.sum())
    merchant[is_topup] = pop.operator[u[is_topup]]

    user_ids = users["user_id"].to_numpy()
    merchant_ids = pop.merchants["merchant_id"].to_numpy()
    return pd.DataFrame(
        {
            "timestamp": ts,
            "user_id": user_ids[u],
            "merchant_id": np.where(is_p2p, None, merchant_ids[merchant]),
            "recipient_id": np.where(is_p2p, user_ids[recipient], None),
            "amount_tnd": amount,
            "type": tx_type,
            "device_id": device,
            "city": city,
            "is_fraud": False,
            "fraud_scenario": None,
        }
    )


def add_bursts(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Append legit bursts: a few quick follow-ups to some transactions.

    Small payments after a payment or top-up, similar transfers after a P2P transfer.
    """
    seeds = rng.choice(len(df), int(len(df) * BURST_RATE), replace=False)
    k = rng.integers(BURST_FOLLOW_UPS[0], BURST_FOLLOW_UPS[1] + 1, len(seeds))
    follow = df.iloc[np.repeat(seeds, k)].reset_index(drop=True)

    # cumulative gap since the seed payment, restarting at each burst
    gaps = rng.integers(BURST_GAP_S[0], BURST_GAP_S[1] + 1, len(follow))
    total = np.cumsum(gaps)
    first = np.cumsum(k) - k
    offset = total - np.repeat(total[first] - gaps[first], k)
    follow["timestamp"] = follow["timestamp"].to_numpy() + offset.astype("timedelta64[s]")

    is_topup = (follow["type"] == TOPUP).to_numpy()
    is_p2p = (follow["type"] == P2P).to_numpy()
    amount = rng.uniform(*MICRO_AMOUNT, len(follow))
    amount[is_topup] = rng.choice(TOPUP_AMOUNTS[:3], is_topup.sum())
    amount[is_p2p] = follow["amount_tnd"].to_numpy()[is_p2p] * rng.uniform(0.5, 1.5, is_p2p.sum())
    follow["amount_tnd"] = np.round(amount, 3)
    return pd.concat([df, follow], ignore_index=True)

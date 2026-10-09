"""Fraud scenarios. Each function returns the transactions of ONE incident on ONE victim.

Common signature: (pop, events, u, day, rng, seen) where `u` is the victim's user index,
`day` is midnight of the incident day and `seen` maps a user index to the mule accounts
already used against them (so a fraud recipient is always new for the victim).
"""

from collections import defaultdict
from collections.abc import Callable

import numpy as np
import pandas as pd

from simulator.profiles import (
    CITY_NAMES,
    CITY_P,
    LifeEvents,
    Population,
    current_device,
    new_device_ids,
    other_cities,
)
from simulator.schema import ACCOUNT_TAKEOVER, CARD_TESTING, MERCHANT, NIGHT_DRAIN, P2P

Seen = dict[int, set[int]]

# share of fraudulent TRANSACTIONS per scenario (card testing has many rows per incident)
SCENARIO_SHARES = {ACCOUNT_TAKEOVER: 0.4, CARD_TESTING: 0.4, NIGHT_DRAIN: 0.2}

CAUTIOUS_FRAUDSTER_RATE = 0.35  # keeps amounts within the victim's usual range
REMOTE_ACCESS_RATE = 0.25  # takeovers operated from the victim's own device (malware)
DAYTIME_RATE = 0.7  # incidents timed during the day, when legit activity is high


def _seconds(n: int | float) -> np.timedelta64:
    return np.timedelta64(int(n), "s")


def _unseen_mule(pop: Population, u: int, rng: np.random.Generator, seen: Seen) -> int:
    candidates = [m for m in pop.mules if m != u and m not in seen[u]]
    mule = int(rng.choice(candidates))
    seen[u].add(mule)
    return mule


def _row(pop, u, ts, amount, tx_type, device, city, scenario, merchant=None, recipient=None):
    return {
        "timestamp": ts,
        "user_id": pop.users["user_id"].iat[u],
        "merchant_id": None if merchant is None else pop.merchants["merchant_id"].iat[merchant],
        "recipient_id": None if recipient is None else pop.users["user_id"].iat[recipient],
        "amount_tnd": round(float(amount), 3),
        "type": tx_type,
        "device_id": device,
        "city": city,
        "is_fraud": True,
        "fraud_scenario": scenario,
    }


def _large_amount(pop, u, rng, multiplier: tuple[float, float]) -> float:
    """A large amount relative to the victim's usual range, unless the fraudster is cautious."""
    high = pop.users["amount_high"].iat[u]
    if rng.random() < CAUTIOUS_FRAUDSTER_RATE:
        return high * rng.uniform(0.3, 1.2)
    return high * rng.uniform(*multiplier)


def _incident_time(day, rng) -> np.datetime64:
    if rng.random() < DAYTIME_RATE:
        return day + _seconds(rng.integers(8 * 3600, 23 * 3600))
    return day + _seconds(rng.integers(0, 23 * 3600))


def account_takeover(pop, events, u, day, rng, seen) -> list[dict]:
    """Stolen credentials: new device, usually a new city, several large transfers in minutes."""
    ts = _incident_time(day, rng)
    home = pop.users["home_city"].iat[u]
    if rng.random() < REMOTE_ACCESS_RATE:
        city, device = home, current_device(pop, events, u, ts)
    else:
        city = other_cities(np.array([home]), rng)[0] if rng.random() < 0.85 else home
        device = new_device_ids(1, rng)[0]
    mules = [_unseen_mule(pop, u, rng, seen) for _ in range(rng.integers(1, 3))]
    base = _large_amount(pop, u, rng, (1.2, 8))

    rows = []
    for _ in range(rng.integers(3, 7)):
        amount = base * rng.uniform(0.7, 1.3)
        mule = mules[rng.integers(len(mules))]
        rows.append(_row(pop, u, ts, amount, P2P, device, city, ACCOUNT_TAKEOVER, recipient=mule))
        ts = ts + _seconds(rng.integers(20, 400))
    return rows


def card_testing(pop, events, u, day, rng, seen) -> list[dict]:
    """Stolen payment credentials checked with a burst of tiny payments within minutes."""
    ts = _incident_time(day, rng)
    home = pop.users["home_city"].iat[u]
    city = rng.choice(CITY_NAMES, p=CITY_P) if rng.random() < 0.5 else home
    if rng.random() < 0.6:
        device = new_device_ids(1, rng)[0]
    else:
        device = current_device(pop, events, u, ts)
    n_shops = len(pop.merchants) - len(np.unique(pop.operator))
    shops = rng.integers(0, n_shops, rng.integers(1, 3))

    rows = []
    for _ in range(rng.integers(5, 16)):
        shop = int(shops[rng.integers(len(shops))])
        amount = rng.uniform(0.5, 3)
        rows.append(_row(pop, u, ts, amount, MERCHANT, device, city, CARD_TESTING, merchant=shop))
        ts = ts + _seconds(rng.integers(5, 61))
    return rows


def night_drain(pop, events, u, day, rng, seen) -> list[dict]:
    """Large P2P transfer in the middle of the night to a never-seen recipient, usual device."""
    ts = day + _seconds(rng.integers(0, 5.5 * 3600))
    home = pop.users["home_city"].iat[u]
    device = current_device(pop, events, u, ts)
    mule = _unseen_mule(pop, u, rng, seen)
    amount = _large_amount(pop, u, rng, (1.5, 10))

    rows = [_row(pop, u, ts, amount, P2P, device, home, NIGHT_DRAIN, recipient=mule)]
    if rng.random() < 0.15:  # a second transfer shortly after
        ts = ts + _seconds(rng.integers(60, 180))
        amount = amount * rng.uniform(0.3, 0.8)
        rows.append(_row(pop, u, ts, amount, P2P, device, home, NIGHT_DRAIN, recipient=mule))
    return rows


Scenario = Callable[..., list[dict]]
SCENARIOS: dict[str, Scenario] = {
    ACCOUNT_TAKEOVER: account_takeover,
    CARD_TESTING: card_testing,
    NIGHT_DRAIN: night_drain,
}


def inject_fraud(
    pop: Population,
    events: LifeEvents,
    n_target: int,
    start: np.datetime64,
    days: int,
    rng: np.random.Generator,
    shares: dict[str, float] = SCENARIO_SHARES,
    seen: Seen | None = None,
) -> pd.DataFrame:
    """Generate incidents on random victims until each scenario reaches its share of `n_target`.

    Pass the same `seen` across calls (streaming) to keep fraud recipients new for each victim.
    """
    seen = defaultdict(set) if seen is None else seen
    rows: list[dict] = []
    for name, share in shares.items():
        budget, count = round(n_target * share), 0
        while count < budget:
            u = int(rng.integers(pop.n_users))
            day = start + np.timedelta64(int(rng.integers(days)), "D")
            incident = SCENARIOS[name](pop, events, u, day, rng, seen)
            rows.extend(incident)
            count += len(incident)
    return pd.DataFrame(rows)

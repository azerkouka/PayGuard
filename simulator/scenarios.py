"""Fraud scenarios. Each scenario function returns the transactions of ONE incident on ONE victim.

A scenario takes (ctx, u, day): the shared `FraudContext`, the victim's user index and
midnight of the incident day.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from schema import ACCOUNT_TAKEOVER, CARD_TESTING, MERCHANT, NIGHT_DRAIN, P2P
from simulator.profiles import (
    CITY_NAMES,
    CITY_P,
    LifeEvents,
    Population,
    current_device,
    new_device_ids,
    other_cities,
)

# user index -> mule accounts already used against that user
Seen = dict[int, set[int]]

# share of fraudulent TRANSACTIONS per scenario (card testing has many rows per incident)
SCENARIO_SHARES = {ACCOUNT_TAKEOVER: 0.4, CARD_TESTING: 0.4, NIGHT_DRAIN: 0.2}

CAUTIOUS_FRAUDSTER_RATE = 0.35  # keeps amounts within the victim's usual range
REMOTE_ACCESS_RATE = 0.25  # takeovers operated from the victim's own device (malware)
DAYTIME_RATE = 0.7  # incidents timed during the day, when legit activity is high


@dataclass
class FraudContext:
    """What every scenario needs: who exists, what they are doing, and the random source."""

    pop: Population
    events: LifeEvents
    rng: np.random.Generator
    # kept across incidents so that a fraud recipient is always new for the victim
    seen: Seen = field(default_factory=lambda: defaultdict(set))


def _seconds(n: float) -> np.timedelta64:
    return np.timedelta64(int(n), "s")


def _unseen_mule(ctx: FraudContext, u: int) -> int:
    """A mule account that user `u` has never sent money to."""
    candidates = [m for m in ctx.pop.mules if m != u and m not in ctx.seen[u]]
    mule = int(ctx.rng.choice(candidates))
    ctx.seen[u].add(mule)
    return mule


def _row(
    ctx: FraudContext,
    u: int,
    ts: np.datetime64,
    amount: float,
    *,
    tx_type: str,
    device: str,
    city: str,
    scenario: str,
    merchant: int | None = None,
    recipient: int | None = None,
) -> dict:
    users, merchants = ctx.pop.users, ctx.pop.merchants
    return {
        "timestamp": ts,
        "user_id": users["user_id"].iat[u],
        "merchant_id": None if merchant is None else merchants["merchant_id"].iat[merchant],
        "recipient_id": None if recipient is None else users["user_id"].iat[recipient],
        "amount_tnd": round(float(amount), 3),
        "type": tx_type,
        "device_id": device,
        "city": city,
        "is_fraud": True,
        "fraud_scenario": scenario,
    }


def _large_amount(ctx: FraudContext, u: int, multiplier: tuple[float, float]) -> float:
    """A large amount relative to the victim's usual range, unless the fraudster is cautious."""
    high = ctx.pop.users["amount_high"].iat[u]
    if ctx.rng.random() < CAUTIOUS_FRAUDSTER_RATE:
        return high * ctx.rng.uniform(0.3, 1.2)
    return high * ctx.rng.uniform(*multiplier)


def _incident_time(ctx: FraudContext, day: np.datetime64) -> np.datetime64:
    if ctx.rng.random() < DAYTIME_RATE:
        return day + _seconds(ctx.rng.integers(8 * 3600, 23 * 3600))
    return day + _seconds(ctx.rng.integers(0, 23 * 3600))


def account_takeover(ctx: FraudContext, u: int, day: np.datetime64) -> list[dict]:
    """Stolen credentials: new device, usually a new city, several large transfers in minutes."""
    pop, rng = ctx.pop, ctx.rng
    ts = _incident_time(ctx, day)
    home = pop.users["home_city"].iat[u]
    if rng.random() < REMOTE_ACCESS_RATE:
        city, device = home, current_device(pop, ctx.events, u, ts)
    else:
        city = other_cities(np.array([home]), rng)[0] if rng.random() < 0.85 else home
        device = new_device_ids(1, rng)[0]
    mules = [_unseen_mule(ctx, u) for _ in range(rng.integers(1, 3))]
    base = _large_amount(ctx, u, (1.2, 8))

    rows = []
    for _ in range(rng.integers(3, 7)):
        amount = base * rng.uniform(0.7, 1.3)
        mule = mules[rng.integers(len(mules))]
        rows.append(
            _row(
                ctx,
                u,
                ts,
                amount,
                tx_type=P2P,
                device=device,
                city=city,
                scenario=ACCOUNT_TAKEOVER,
                recipient=mule,
            )
        )
        ts = ts + _seconds(rng.integers(20, 400))
    return rows


def card_testing(ctx: FraudContext, u: int, day: np.datetime64) -> list[dict]:
    """Stolen payment credentials checked with a burst of tiny payments within minutes."""
    pop, rng = ctx.pop, ctx.rng
    ts = _incident_time(ctx, day)
    home = pop.users["home_city"].iat[u]
    city = rng.choice(CITY_NAMES, p=CITY_P) if rng.random() < 0.5 else home
    if rng.random() < 0.6:
        device = new_device_ids(1, rng)[0]
    else:
        device = current_device(pop, ctx.events, u, ts)
    shops = rng.integers(0, pop.n_shops, rng.integers(1, 3))

    rows = []
    for _ in range(rng.integers(5, 16)):
        shop = int(shops[rng.integers(len(shops))])
        amount = rng.uniform(0.5, 3)
        rows.append(
            _row(
                ctx,
                u,
                ts,
                amount,
                tx_type=MERCHANT,
                device=device,
                city=city,
                scenario=CARD_TESTING,
                merchant=shop,
            )
        )
        ts = ts + _seconds(rng.integers(5, 61))
    return rows


def night_drain(ctx: FraudContext, u: int, day: np.datetime64) -> list[dict]:
    """Large P2P transfer in the middle of the night to a never-seen recipient, usual device."""
    pop, rng = ctx.pop, ctx.rng
    ts = day + _seconds(rng.integers(0, 5.5 * 3600))
    transfer = {
        "tx_type": P2P,
        "device": current_device(pop, ctx.events, u, ts),
        "city": pop.users["home_city"].iat[u],
        "scenario": NIGHT_DRAIN,
        "recipient": _unseen_mule(ctx, u),
    }
    amount = _large_amount(ctx, u, (1.5, 10))

    rows = [_row(ctx, u, ts, amount, **transfer)]
    if rng.random() < 0.15:  # a second transfer shortly after
        ts = ts + _seconds(rng.integers(60, 180))
        amount = amount * rng.uniform(0.3, 0.8)
        rows.append(_row(ctx, u, ts, amount, **transfer))
    return rows


Scenario = Callable[[FraudContext, int, np.datetime64], list[dict]]
SCENARIOS: dict[str, Scenario] = {
    ACCOUNT_TAKEOVER: account_takeover,
    CARD_TESTING: card_testing,
    NIGHT_DRAIN: night_drain,
}


def inject_fraud(
    ctx: FraudContext,
    n_target: int,
    start: np.datetime64,
    days: int,
    shares: dict[str, float] = SCENARIO_SHARES,
) -> pd.DataFrame:
    """Generate incidents on random victims until each scenario reaches its share of `n_target`."""
    rows: list[dict] = []
    for name, share in shares.items():
        budget, count = round(n_target * share), 0
        while count < budget:
            u = int(ctx.rng.integers(ctx.pop.n_users))
            day = start + np.timedelta64(int(ctx.rng.integers(days)), "D")
            incident = SCENARIOS[name](ctx, u, day)
            rows.extend(incident)
            count += len(incident)
    return pd.DataFrame(rows)

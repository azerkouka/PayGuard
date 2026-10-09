"""User and merchant profiles, plus the legit "life events" (trips, phone changes)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Illustrative weights, roughly proportional to population. Not official statistics.
CITIES = {
    "Tunis": 20, "Sfax": 10, "Sousse": 8, "Ariana": 6, "Ben Arous": 6, "Nabeul": 6,
    "Bizerte": 5, "Kairouan": 5, "Monastir": 5, "Gabes": 4, "Medenine": 4, "La Manouba": 4,
    "Gafsa": 3, "Kasserine": 3, "Mahdia": 3, "Sidi Bouzid": 3, "Jendouba": 3, "Beja": 2,
    "Le Kef": 2, "Zaghouan": 1.5, "Siliana": 1.5, "Tataouine": 1.5, "Kebili": 1.5, "Tozeur": 1,
}  # fmt: skip
CITY_NAMES = np.array(list(CITIES))
CITY_P = np.array(list(CITIES.values())) / sum(CITIES.values())

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "pharmacy", "fuel", "clothing", "electronics", "online", "utilities",
]  # fmt: skip
N_TELECOM_OPERATORS = 3  # top-ups are paid to these merchants

N_CONTACTS = 5  # usual P2P recipients per user
N_FAV_MERCHANTS = 8

TRIPS_PER_90_DAYS = 0.5  # mean number of trips per user
PHONE_CHANGE_PER_90_DAYS = 0.15  # share of users who get a new phone


@dataclass(frozen=True)
class Population:
    """Static profiles. Users and merchants are referred to by row index internally."""

    users: pd.DataFrame
    merchants: pd.DataFrame
    contacts: np.ndarray  # (n_users, N_CONTACTS) user indices
    fav_merchants: np.ndarray  # (n_users, N_FAV_MERCHANTS) merchant indices
    operator: np.ndarray  # (n_users,) merchant index of the user's telecom operator
    mules: np.ndarray  # user indices receiving fraudulent transfers
    non_mules: np.ndarray
    shops_by_city: dict[str, np.ndarray]  # city -> merchant indices (telecom excluded)

    @property
    def n_users(self) -> int:
        return len(self.users)


@dataclass(frozen=True)
class LifeEvents:
    """Legit behaviour changes over a time window."""

    trips: pd.DataFrame  # u, trip_start, trip_end, trip_city
    phone_change_time: np.ndarray  # (n_users,) datetime64[s], NaT if no change
    new_device: np.ndarray  # (n_users,) device used after the change


def new_device_ids(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.array([f"dev_{x:010x}" for x in rng.integers(0, 2**40, n)], dtype=object)


def other_cities(home: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One random city per entry, different from `home`."""
    city = rng.choice(CITY_NAMES, len(home), p=CITY_P)
    while (same := city == home).any():
        city[same] = rng.choice(CITY_NAMES, same.sum(), p=CITY_P)
    return city


def _build_merchants(n_merchants: int, rng: np.random.Generator) -> pd.DataFrame:
    n_shops = n_merchants - N_TELECOM_OPERATORS
    if n_shops < len(CITY_NAMES):
        raise ValueError(f"need at least {len(CITY_NAMES) + N_TELECOM_OPERATORS} merchants")
    # one shop per city first, so every city has somewhere to pay
    extra = rng.choice(CITY_NAMES, n_shops - len(CITY_NAMES), p=CITY_P)
    city = np.concatenate([CITY_NAMES, extra, ["Tunis"] * N_TELECOM_OPERATORS])
    category = np.concatenate(
        [rng.choice(MERCHANT_CATEGORIES, n_shops), ["telecom"] * N_TELECOM_OPERATORS]
    )
    return pd.DataFrame(
        {
            "merchant_id": [f"M{i:04d}" for i in range(n_merchants)],
            "city": city,
            "category": category,
        }
    )


def build_population(n_users: int, n_merchants: int, rng: np.random.Generator) -> Population:
    merchants = _build_merchants(n_merchants, rng)
    n_shops = n_merchants - N_TELECOM_OPERATORS
    shop_city = merchants["city"].to_numpy()[:n_shops]
    shops_by_city = {c: np.flatnonzero(shop_city == c) for c in CITY_NAMES}

    home_city = rng.choice(CITY_NAMES, n_users, p=CITY_P)
    n_devices = rng.choice([1, 2], n_users, p=[0.7, 0.3])
    device_2 = new_device_ids(n_users, rng)
    device_2[n_devices == 1] = None

    # typical amount range: [low, high] covers ~90% of the user's amounts
    amount_low = np.round(np.exp(rng.normal(np.log(8), 0.5, n_users)), 1)
    amount_high = np.round(amount_low * rng.uniform(4, 12, n_users), 1)

    # active hours: most users are day people, some are night owls (window wraps past midnight)
    night_owl = rng.random(n_users) < 0.15
    active_start = np.where(night_owl, rng.integers(12, 17, n_users), rng.integers(6, 10, n_users))
    active_len = np.where(night_owl, rng.integers(11, 15, n_users), rng.integers(12, 17, n_users))

    type_prefs = rng.dirichlet([3, 4, 2], n_users)  # p2p, merchant, top-up
    activity = rng.gamma(2.0, 1.0, n_users)

    users = pd.DataFrame(
        {
            "user_id": [f"U{i:05d}" for i in range(n_users)],
            "home_city": home_city,
            "n_devices": n_devices,
            "device_1": new_device_ids(n_users, rng),
            "device_2": device_2,
            "amount_low": amount_low,
            "amount_high": amount_high,
            "active_start": active_start,
            "active_end": (active_start + active_len) % 24,
            "p_p2p": type_prefs[:, 0],
            "p_merchant": type_prefs[:, 1],
            "p_topup": type_prefs[:, 2],
            "activity": activity / activity.sum(),
        }
    )

    mules = np.sort(rng.choice(n_users, max(5, n_users * 3 // 100), replace=False))
    non_mules = np.setdiff1d(np.arange(n_users), mules)

    me = np.arange(n_users)[:, None]
    contacts = rng.choice(non_mules, (n_users, N_CONTACTS))
    while (own := contacts == me).any():
        contacts[own] = rng.choice(non_mules, own.sum())

    fav_merchants = np.empty((n_users, N_FAV_MERCHANTS), dtype=np.int64)
    for i, city in enumerate(home_city):
        pool = shops_by_city[city]
        fav_merchants[i] = rng.choice(pool, N_FAV_MERCHANTS, replace=len(pool) < N_FAV_MERCHANTS)

    return Population(
        users=users,
        merchants=merchants,
        contacts=contacts,
        fav_merchants=fav_merchants,
        operator=n_shops + rng.integers(0, N_TELECOM_OPERATORS, n_users),
        mules=mules,
        non_mules=non_mules,
        shops_by_city=shops_by_city,
    )


def plan_life_events(
    pop: Population, start: np.datetime64, days: int, rng: np.random.Generator
) -> LifeEvents:
    """Schedule legit trips and phone changes between `start` and `start + days`."""
    span_s = days * 86400
    scale = days / 90

    n_trips = rng.poisson(TRIPS_PER_90_DAYS * scale, pop.n_users)
    u = np.repeat(np.arange(pop.n_users), n_trips)
    trip_start = start + rng.integers(0, span_s, len(u)).astype("timedelta64[s]")
    duration = rng.integers(1 * 86400, 6 * 86400, len(u)).astype("timedelta64[s]")
    trips = pd.DataFrame(
        {
            "u": u,
            "trip_start": trip_start,
            "trip_end": trip_start + duration,
            "trip_city": other_cities(pop.users["home_city"].to_numpy()[u], rng),
        }
    )

    changes = rng.random(pop.n_users) < min(1.0, PHONE_CHANGE_PER_90_DAYS * scale)
    change_time = start + rng.integers(0, span_s, pop.n_users).astype("timedelta64[s]")
    change_time[~changes] = np.datetime64("NaT", "s")

    return LifeEvents(
        trips=trips,
        phone_change_time=change_time,
        new_device=new_device_ids(pop.n_users, rng),
    )


def current_device(pop: Population, events: LifeEvents, u: int, ts: np.datetime64) -> str:
    """The device user `u` normally uses at time `ts`."""
    if ts >= events.phone_change_time[u]:  # False when NaT
        return events.new_device[u]
    return pop.users["device_1"].iat[u]

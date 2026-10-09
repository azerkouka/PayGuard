import pandas as pd
import pytest

from schema import (
    ACCOUNT_TAKEOVER,
    CARD_TESTING,
    COLUMNS,
    FRAUD_SCENARIOS,
    NIGHT_DRAIN,
    P2P,
    TRANSACTION_TYPES,
)
from simulator.generate import generate_dataset

SMALL = {"days": 14, "n_users": 300, "n_merchants": 60, "n_transactions": 20_000}


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(seed=42, **SMALL)


@pytest.fixture(scope="module")
def df(dataset):
    return dataset[0]


@pytest.fixture(scope="module")
def with_profile(dataset):
    """Transactions joined with the profile of their user."""
    df, pop = dataset
    return df.merge(pop.users, on="user_id")


def test_schema(df):
    assert list(df.columns) == COLUMNS
    assert df["timestamp"].dtype == "datetime64[ns]"
    assert df["amount_tnd"].dtype == "float64"
    assert df["is_fraud"].dtype == bool
    assert df["transaction_id"].is_unique
    assert df["timestamp"].is_monotonic_increasing
    assert (df["amount_tnd"] > 0).all()
    assert set(df["type"]) == set(TRANSACTION_TYPES)
    required = ["transaction_id", "timestamp", "user_id", "amount_tnd", "type", "device_id", "city"]
    assert df[required].notna().all().all()


def test_counterparty_matches_type(df):
    is_p2p = df["type"] == P2P
    assert (df["recipient_id"].notna() == is_p2p).all()
    assert (df["merchant_id"].notna() == ~is_p2p).all()
    assert (df["recipient_id"] != df["user_id"]).all()


def test_fraud_label_consistency(df):
    assert (df["fraud_scenario"].notna() == df["is_fraud"]).all()


def test_fraud_rate_within_tolerance(df):
    assert df["is_fraud"].mean() == pytest.approx(0.015, abs=0.003)


def test_each_scenario_present(df):
    assert set(df["fraud_scenario"].dropna()) == set(FRAUD_SCENARIOS)


def test_reproducible():
    small = {"days": 5, "n_users": 100, "n_merchants": 40, "n_transactions": 3000}
    a, _ = generate_dataset(seed=7, **small)
    b, _ = generate_dataset(seed=7, **small)
    c, _ = generate_dataset(seed=8, **small)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_account_takeover_mostly_uses_unknown_device(with_profile):
    ato = with_profile[with_profile["fraud_scenario"] == ACCOUNT_TAKEOVER]
    assert (ato["type"] == P2P).all()
    unknown = (ato["device_id"] != ato["device_1"]) & (ato["device_id"] != ato["device_2"])
    assert unknown.mean() > 0.5  # the rest is operated from the victim's own device
    assert ato.groupby(["user_id", "device_id"]).size().min() >= 3


def test_card_testing_is_a_burst_of_small_payments(df):
    ct = df[df["fraud_scenario"] == CARD_TESTING]
    assert (ct["amount_tnd"] <= 3).all()
    incidents = ct.groupby(["user_id", "device_id", ct["timestamp"].dt.date])["timestamp"]
    assert (incidents.size() >= 5).all()
    assert (incidents.max() - incidents.min()).max() <= pd.Timedelta(minutes=15)


def test_night_drain_is_at_night_to_a_never_seen_recipient(df):
    nd = df[df["fraud_scenario"] == NIGHT_DRAIN]
    assert (nd["timestamp"].dt.hour < 6).all()
    first_transfer = df[df["type"] == P2P].drop_duplicates(["user_id", "recipient_id"])
    pairs = nd[["user_id", "recipient_id"]].drop_duplicates()
    first = pairs.merge(first_transfer, on=["user_id", "recipient_id"])
    assert (first["fraud_scenario"] == NIGHT_DRAIN).all()


def test_legit_edge_cases_exist(with_profile):
    """Fraud signals must also appear on legit transactions, or the problem is trivial."""
    legit = with_profile[~with_profile["is_fraud"]]
    known_device = (legit["device_id"] == legit["device_1"]) | (
        legit["device_id"] == legit["device_2"]
    )
    assert (legit["city"] != legit["home_city"]).any()  # travel
    assert (~known_device).any()  # new phone
    assert (legit["amount_tnd"] > 3 * legit["amount_high"]).any()  # large purchase
    assert (legit["timestamp"].dt.hour < 5).any()  # night activity

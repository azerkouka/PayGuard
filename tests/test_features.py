import pandas as pd
import pytest

from features import FEATURES, compute_features_batch, replay
from schema import MERCHANT, P2P
from simulator.generate import generate_dataset


@pytest.fixture(scope="module")
def df():
    return generate_dataset(seed=1, days=10, n_users=150, n_merchants=40, n_transactions=8000)[0]


def test_no_leakage_from_future_transactions(df):
    """Features of a transaction must not change when later transactions are added."""
    cut = int(len(df) * 0.6)
    past_only = compute_features_batch(df.iloc[:cut])
    with_future = compute_features_batch(df).iloc[:cut]
    pd.testing.assert_frame_equal(past_only, with_future, check_exact=True)


def test_batch_matches_online(df):
    pd.testing.assert_frame_equal(compute_features_batch(df), replay(df), rtol=1e-9, atol=1e-6)


def test_batch_ignores_row_order(df):
    shuffled = df.sample(frac=1, random_state=0)
    expected = compute_features_batch(df)
    pd.testing.assert_frame_equal(compute_features_batch(shuffled).loc[df.index], expected)


def _tx(time, amount, tx_type=MERCHANT, device="d1", city="Tunis", recipient=None):
    return {
        "timestamp": pd.Timestamp(f"2026-01-01 {time}"),
        "user_id": "U1",
        "merchant_id": None if tx_type == P2P else "M1",
        "recipient_id": recipient,
        "amount_tnd": amount,
        "type": tx_type,
        "device_id": device,
        "city": city,
    }


@pytest.mark.parametrize("compute", [compute_features_batch, replay])
def test_known_values(compute):
    df = pd.DataFrame(
        [
            _tx("02:00:00", 10.0),
            _tx("12:00:00", 20.0),
            _tx("12:02:00", 30.0, tx_type=P2P, recipient="U2"),
            _tx("12:30:00", 60.0, tx_type=P2P, recipient="U2", device="d2", city="Sfax"),
        ]
    )
    f = compute(df)
    assert list(f.columns) == FEATURES

    first = f.iloc[0]  # no history: neutral values
    assert first["count_24h"] == 0 and first["amount_ratio"] == 1
    assert first["is_new_device"] == 0 and first["is_new_city"] == 0
    assert first["is_night"] == 1 and first["hour"] == 2

    third = f.iloc[2]
    assert (third["count_5m"], third["sum_5m"], third["mean_5m"]) == (1, 20, 20)
    assert (third["count_24h"], third["sum_24h"]) == (2, 30)
    assert third["amount_ratio"] == 30 / 15
    assert third["is_new_recipient"] == 1 and third["is_new_device"] == 0
    assert third["seconds_since_prev"] == 120
    assert third["type_p2p"] == 1 and third["type_merchant"] == 0

    last = f.iloc[3]
    assert (last["count_5m"], last["count_1h"], last["sum_1h"]) == (0, 2, 50)
    assert last["amount_ratio"] == 60 / 20
    assert last["is_new_recipient"] == 0
    assert last["is_new_device"] == 1 and last["is_new_city"] == 1

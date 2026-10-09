import fakeredis
import pandas as pd
import pytest

from features import FEATURES, compute_features_batch
from features.store import FeatureStore, user_keys
from simulator.generate import generate_dataset
from streaming.feature_consumer import compute_batch_features


@pytest.fixture(scope="module")
def df():
    return generate_dataset(seed=2, days=6, n_users=80, n_merchants=40, n_transactions=2500)[0]


@pytest.fixture
def store():
    return FeatureStore(fakeredis.FakeRedis(decode_responses=True))


def online_features(store, df) -> pd.DataFrame:
    rows = compute_batch_features(store, df.to_dict("records"))
    return pd.DataFrame(rows, index=df.index, columns=FEATURES)


def test_online_path_matches_batch_features(store, df):
    """Parity: historical transactions through Redis give the offline training features."""
    pd.testing.assert_frame_equal(
        online_features(store, df), compute_features_batch(df), rtol=1e-9, atol=1e-6
    )


def test_replayed_transaction_changes_nothing(store, df):
    head, replayed, tail = df.iloc[:1500], df.iloc[1400:1500], df.iloc[1500:1800]
    online_features(store, head)
    assert all(store.process(tx) is None for tx in replayed.to_dict("records"))
    assert all(not store.apply(tx) for tx in replayed.to_dict("records"))
    expected = compute_features_batch(df.iloc[:1800]).iloc[1500:]
    pd.testing.assert_frame_equal(online_features(store, tail), expected, rtol=1e-9, atol=1e-6)


def test_bootstrap_equals_processing_the_history(store, df):
    cut = int(len(df) * 0.7)
    assert store.bootstrap(df.iloc[:cut]) == df.iloc[:cut]["user_id"].nunique()
    expected = compute_features_batch(df).iloc[cut:]
    pd.testing.assert_frame_equal(
        online_features(store, df.iloc[cut:]), expected, rtol=1e-9, atol=1e-6
    )


def test_bootstrap_replaces_existing_state(store, df):
    online_features(store, df)
    store.bootstrap(df.iloc[:100])
    users = df.iloc[:100]["user_id"].nunique()
    assert len(store.client.keys("u:*:stats")) == users


def test_state_expires_and_recent_history_is_trimmed(store, df):
    online_features(store, df)
    user = df["user_id"].value_counts().index[0]
    recent, devices, cities, _, stats = user_keys(user)
    assert 0 < store.client.ttl(recent) <= 24 * 3600
    for key in (devices, cities, stats):
        assert 24 * 3600 < store.client.ttl(key) <= 90 * 86400
    scores = [score for _, score in store.client.zrange(recent, 0, -1, withscores=True)]
    assert max(scores) - min(scores) <= 24 * 3600
    assert int(store.client.hget(stats, "tx_count")) == (df["user_id"] == user).sum()

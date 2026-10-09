from itertools import islice

import pandas as pd

from simulator.schema import COLUMNS, decode_transaction, encode_transaction
from simulator.stream import publish, transaction_stream

SMALL = dict(tx_per_day=2000, n_users=200, n_merchants=40)


def take(n, seed=1):
    return list(islice(transaction_stream(seed=seed, **SMALL), n))


def test_stream_is_ordered_across_days():
    df = pd.DataFrame(take(5000))  # more than two simulated days
    assert list(df.columns) == COLUMNS
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].dt.date.nunique() >= 3
    assert df["transaction_id"].is_unique
    assert df["timestamp"].min() >= pd.Timestamp("2026-04-01")
    assert 0 < df["is_fraud"].mean() < 0.05


def test_stream_seed():
    assert take(300, seed=5) == take(300, seed=5)
    assert take(300, seed=5) != take(300, seed=6)
    random_a = next(transaction_stream(**SMALL))
    random_b = next(transaction_stream(**SMALL))
    assert random_a["transaction_id"] != random_b["transaction_id"]


def test_encode_decode_roundtrip():
    for tx in take(200):
        key, value = encode_transaction(tx)
        assert key == tx["user_id"].encode()
        assert decode_transaction(value) == tx


class FakeProducer:
    def __init__(self):
        self.messages = []

    def produce(self, topic, key, value, on_delivery):
        self.messages.append((topic, key, value))

    def poll(self, timeout):
        return 0

    def flush(self, timeout):
        return 0


def test_publish_keys_by_user_and_keeps_order():
    txs = take(200)
    producer = FakeProducer()
    sent = publish(txs, producer, "transactions", rate=10_000)
    assert sent == 200
    assert [key for _, key, _ in producer.messages] == [tx["user_id"].encode() for tx in txs]
    assert [decode_transaction(v) for _, _, v in producer.messages] == txs


def test_publish_stops_after_duration():
    producer = FakeProducer()
    sent = publish(transaction_stream(seed=1, **SMALL), producer, "t", rate=200, duration=0.5)
    assert 50 <= sent <= 110

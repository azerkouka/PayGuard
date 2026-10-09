import json

import pandas as pd

from streaming.consumer import decode_batch


class FakeMessage:
    def __init__(self, value: bytes, error=None):
        self._value, self._error = value, error

    def error(self):
        return self._error

    def value(self):
        return self._value

    def timestamp(self):
        return (1, 1_700_000_000_123)

    def topic(self):
        return "transactions"

    def partition(self):
        return 0

    def offset(self):
        return 42


def test_decode_batch_skips_unreadable_messages():
    good = json.dumps({"transaction_id": "t1", "timestamp": "2026-04-01T10:00:00"}).encode()
    messages = [
        FakeMessage(good),
        FakeMessage(b"not json"),
        FakeMessage(json.dumps({"transaction_id": "no-timestamp"}).encode()),
        FakeMessage(b"", error="broker went away"),
    ]

    transactions = decode_batch(messages)

    assert [tx["transaction_id"] for tx in transactions] == ["t1"]
    assert transactions[0]["timestamp"] == pd.Timestamp("2026-04-01 10:00:00")
    assert transactions[0]["produced_at_ms"] == 1_700_000_000_123

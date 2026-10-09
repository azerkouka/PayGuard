import json
import time

import fakeredis
import numpy as np
import pandas as pd
import pytest

from features import compute_features_batch
from features.store import FeatureStore
from ml.scoring import APPROVE, Scorer
from streaming.scoring_consumer import ScoringPipeline


class FakeSink:
    def __init__(self):
        self.transactions, self.results = [], []

    def write(self, transactions, results):
        self.transactions += transactions
        self.results += results


class FakeProducer:
    def __init__(self):
        self.messages = []

    def produce(self, topic, key, value):
        self.messages.append((topic, key, json.loads(value)))

    def flush(self, timeout):
        return 0


@pytest.fixture(scope="module")
def history(trained_models):
    return pd.read_parquet(trained_models[0] / "transactions.parquet")


@pytest.fixture
def pipeline(trained_models):
    store = FeatureStore(fakeredis.FakeRedis(decode_responses=True))
    return ScoringPipeline(store, Scorer(trained_models[0]), FakeSink(), FakeProducer(), "alerts")


def as_messages(df) -> list[dict]:
    now_ms = time.time() * 1000
    return [{**tx, "produced_at_ms": now_ms} for tx in df.to_dict("records")]


def test_pipeline_scores_like_the_offline_model(pipeline, history):
    """End-to-end parity: stream scores equal scores computed from the batch features."""
    df = history.iloc[:1500]
    stats = pipeline.handle_batch(as_messages(df))

    results = pipeline.sink.results
    assert [r["transaction_id"] for r in results] == list(df["transaction_id"])
    features = compute_features_batch(df).to_dict("records")
    expected = [pipeline.scorer.score(row).score for row in features]
    np.testing.assert_allclose([r["score"] for r in results], expected, atol=1e-6)
    assert all(r["latency_ms"] >= 0 for r in results)
    assert stats == {"alerts": len(pipeline.producer.messages), "unscored_replays": 0}


def test_alerts_are_review_and_block_only_without_ground_truth(pipeline, history):
    df = history.iloc[:1500]
    pipeline.handle_batch(as_messages(df))

    flagged = {r["transaction_id"]: r for r in pipeline.sink.results if r["decision"] != APPROVE}
    alerts = pipeline.producer.messages
    assert len(alerts) == len(flagged) > 0
    users = df.set_index("transaction_id")["user_id"]
    for topic, key, alert in alerts:
        assert topic == "alerts"
        assert key.decode() == users[alert["transaction_id"]] == alert["user_id"]
        assert alert["decision"] == flagged[alert["transaction_id"]]["decision"]
        assert "is_fraud" not in alert and "fraud_scenario" not in alert


def test_replayed_batch_gives_the_same_results(pipeline, history):
    """A crash before the offset commit replays the batch: nothing changes, nothing is lost."""
    first, second = as_messages(history.iloc[:800]), as_messages(history.iloc[800:1000])
    pipeline.handle_batch(first)
    before = list(pipeline.sink.results)

    # a replay only ever concerns the last, uncommitted batch
    stats = pipeline.handle_batch(first[-200:])
    assert stats["unscored_replays"] == 0
    assert pipeline.sink.results[len(before) :] == before[-200:]

    pipeline.sink.results.clear()
    pipeline.handle_batch(second)
    features = compute_features_batch(history.iloc[:1000]).iloc[800:]
    expected = [pipeline.scorer.score(row).score for row in features.to_dict("records")]
    np.testing.assert_allclose([r["score"] for r in pipeline.sink.results], expected, atol=1e-6)

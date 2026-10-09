"""Full online pipeline: `python -m streaming.scoring_consumer`.

For each transaction read from Kafka:
    features (from Redis, past only) -> score, decision, reasons -> Redis state update
then, per batch:
    raw transactions and scores -> PostgreSQL, review/block decisions -> Kafka topic `alerts`,
    and only then the Kafka offsets are committed.

The model is called as a library (`ml.scoring.Scorer`), not through the HTTP API.
"""

import argparse
import json
import logging
import time

import psycopg
from confluent_kafka import Producer
from psycopg.types.json import Jsonb

from config import settings
from features.online import compute_features
from features.store import FeatureStore
from ml.scoring import APPROVE, Scorer
from streaming.feature_consumer import consume, insert_transactions, make_store

log = logging.getLogger("consumer")

INSERT_SCORE = """
INSERT INTO scores (transaction_id, score, decision, top_reasons, latency_ms)
VALUES (%(transaction_id)s, %(score)s, %(decision)s, %(top_reasons)s, %(latency_ms)s)
ON CONFLICT (transaction_id) DO NOTHING
"""

# what an alert carries: enough for an analyst, and never the simulator's ground truth
ALERT_FIELDS = ("transaction_id", "user_id", "amount_tnd", "type", "city", "device_id",
                "merchant_id", "recipient_id")  # fmt: skip


class PostgresSink:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def write(self, transactions: list[dict], results: list[dict]) -> None:
        """Raw transactions and their scores, in one database transaction."""
        scores = [{**r, "top_reasons": Jsonb(r["reasons"])} for r in results]
        with self.conn.cursor() as cur:
            insert_transactions(cur, transactions)
            cur.executemany(INSERT_SCORE, scores)
        self.conn.commit()


class ScoringPipeline:
    def __init__(self, store: FeatureStore, scorer: Scorer, sink, producer, alerts_topic: str):
        self.store = store
        self.scorer = scorer
        self.sink = sink
        self.producer = producer
        self.alerts_topic = alerts_topic

    def score_transaction(self, tx: dict) -> dict | None:
        """Score `tx` and add it to its user's state.

        A replayed transaction is not scored again (its "before" state is gone): the result
        saved with the state update is returned instead. None if that result has expired.
        """
        state, already_applied = self.store.load(tx)
        if already_applied:
            saved = self.store.saved_result(tx)
            return json.loads(saved) if saved else None

        outcome = self.scorer.score(compute_features(tx, state))
        result = {
            "transaction_id": tx["transaction_id"],
            "score": outcome.score,
            "decision": outcome.decision,
            "reasons": outcome.reasons,
            # end to end: from the moment the transaction was published to Kafka
            "latency_ms": round(time.time() * 1000 - tx["produced_at_ms"], 3),
        }
        self.store.apply(tx, json.dumps(result))
        return result

    def alert(self, tx: dict, result: dict) -> tuple[bytes, bytes]:
        value = {
            **{name: tx[name] for name in ALERT_FIELDS},
            "timestamp": tx["timestamp"].isoformat(),
            "score": result["score"],
            "decision": result["decision"],
            "reasons": result["reasons"],
        }
        return tx["user_id"].encode(), json.dumps(value).encode()

    def handle_batch(self, transactions: list[dict]) -> dict[str, int]:
        results = [self.score_transaction(tx) for tx in transactions]
        scored = [(tx, r) for tx, r in zip(transactions, results, strict=True) if r is not None]

        self.sink.write(transactions, [r for _, r in scored])
        alerts = [(tx, r) for tx, r in scored if r["decision"] != APPROVE]
        for tx, result in alerts:
            key, value = self.alert(tx, result)
            self.producer.produce(self.alerts_topic, key=key, value=value)
        if self.producer.flush(10):  # alerts must be delivered before the offsets are committed
            raise RuntimeError("alerts could not be delivered to Kafka")
        return {"alerts": len(alerts), "unscored_replays": len(transactions) - len(scored)}


def run(idle_exit: float | None = None):
    scorer = Scorer(settings.models_dir)
    producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers, "linger.ms": 5})
    with psycopg.connect(settings.postgres_dsn) as conn:
        pipeline = ScoringPipeline(
            make_store(), scorer, PostgresSink(conn), producer, settings.kafka_alerts_topic
        )
        log.info("model %s, review >= %s, block >= %s",
                 scorer.version, scorer.review_threshold, scorer.block_threshold)  # fmt: skip
        return consume(pipeline.handle_batch, idle_exit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score transactions from Kafka.")
    parser.add_argument("--idle-exit", type=float, help="exit after this many seconds idle")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run(args.idle_exit)


if __name__ == "__main__":
    main()

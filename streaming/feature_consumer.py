"""Kafka consumer loop, and a features-only consumer: `python -m streaming.feature_consumer`.

For each batch read from Kafka:
    1. compute each transaction's features from the user's Redis state, then update that state
    2. insert the raw transactions into PostgreSQL
    3. commit the Kafka offsets
Offsets are committed LAST, so a crash at any point replays the batch (at-least-once delivery).
Replays are harmless: the Redis update and the PostgreSQL insert are both idempotent.

The full pipeline (features + scoring + alerts) is `streaming.scoring_consumer`; it reuses the
loop below. Run one or the other, not both: they share the same consumer group and user state.
"""

import argparse
import logging
import signal
import time
from collections import Counter
from collections.abc import Callable

import psycopg
import redis
from confluent_kafka import Consumer

from config import settings
from features.store import FeatureStore
from simulator.schema import decode_transaction

log = logging.getLogger("consumer")

INSERT_TRANSACTION = """
INSERT INTO transactions (transaction_id, "timestamp", user_id, merchant_id, recipient_id,
                          amount_tnd, type, device_id, city, is_fraud, fraud_scenario)
VALUES (%(transaction_id)s, %(timestamp)s, %(user_id)s, %(merchant_id)s, %(recipient_id)s,
        %(amount_tnd)s, %(type)s, %(device_id)s, %(city)s, %(is_fraud)s, %(fraud_scenario)s)
ON CONFLICT (transaction_id) DO NOTHING
"""

# handles one batch of decoded transactions and returns counters to add to the run's statistics
BatchHandler = Callable[[list[dict]], dict[str, int]]


def make_store() -> FeatureStore:
    client = redis.Redis(
        host=settings.redis_host, port=settings.redis_port, decode_responses=True
    )
    return FeatureStore(client, state_ttl_seconds=settings.state_ttl_days * 86400)


def decode_batch(messages) -> list[dict]:
    """Decode Kafka messages, skipping the ones that cannot be read (logged, not retried).

    Each transaction also gets `produced_at_ms`, the time Kafka recorded when it was published.
    """
    transactions = []
    for message in messages:
        if message.error():
            log.warning("kafka error: %s", message.error())
            continue
        try:
            tx = decode_transaction(message.value())
        except (ValueError, KeyError, TypeError):
            # a message that can never be decoded must not block the partition forever
            log.exception(
                "skipping malformed message at %s[%s]@%s",
                message.topic(), message.partition(), message.offset(),
            )  # fmt: skip
            continue
        tx["produced_at_ms"] = message.timestamp()[1]
        transactions.append(tx)
    return transactions


def compute_batch_features(store: FeatureStore, transactions: list[dict]) -> list[dict | None]:
    """Features for each transaction, in order. None marks a replayed transaction."""
    return [store.process(tx) for tx in transactions]


def insert_transactions(cur: psycopg.Cursor, transactions: list[dict]) -> None:
    rows = [{**tx, "timestamp": tx["timestamp"].to_pydatetime()} for tx in transactions]
    cur.executemany(INSERT_TRANSACTION, rows)


def consume(handle_batch: BatchHandler, idle_exit: float | None = None) -> Counter:
    """Read batches until interrupted (or until no message arrives for `idle_exit` seconds).

    The offsets of a batch are committed only after `handle_batch` returns.
    """
    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": settings.kafka_consumer_group,
            "enable.auto.commit": False,  # we commit ourselves, after processing
            "auto.offset.reset": "earliest",
            # a crashed consumer is declared dead, and its partitions reassigned, after 10 s
            "session.timeout.ms": 10_000,
        }
    )
    assigned = False

    def on_assign(_consumer, partitions):
        nonlocal assigned
        assigned = True
        log.info("assigned partitions %s", sorted(p.partition for p in partitions))

    consumer.subscribe([settings.kafka_transactions_topic], on_assign=on_assign)

    running = True

    def stop(signum, _frame):
        nonlocal running
        log.info("signal %s received, finishing the current batch", signum)
        running = False

    # SIGINT = Ctrl+C, SIGTERM = `docker stop`, SIGBREAK = Ctrl+Break on Windows
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop)

    stats = Counter()
    last_message = last_report = time.monotonic()
    try:
        while running:
            # wait for one message, then take whatever else has already arrived: a batch of 1
            # when traffic is light (lowest latency), bigger batches when the consumer is behind
            first = consumer.poll(0.5)
            now = time.monotonic()
            if not assigned:  # still joining the group: not idle yet
                last_message = now
            if first is None:
                if idle_exit is not None and now - last_message >= idle_exit:
                    log.info("no message for %.0fs, exiting", idle_exit)
                    break
                continue
            last_message = now
            messages = [first, *consumer.consume(settings.consumer_batch_size - 1, timeout=0)]

            transactions = decode_batch(messages)
            stats.update(handle_batch(transactions))
            consumer.commit(asynchronous=False)

            stats.update(processed=len(transactions), batches=1)
            if now - last_report >= 5:
                last_report = now
                log.info("progress: %s", dict(stats))
    finally:
        consumer.close()  # leaves the group cleanly so partitions are reassigned at once
        log.info("stopped: %s", dict(stats))
    return stats


def run(idle_exit: float | None = None) -> Counter:
    """Features only: update Redis and store raw transactions, no scoring."""
    store = make_store()
    with psycopg.connect(settings.postgres_dsn) as conn:

        def handle_batch(transactions: list[dict]) -> dict[str, int]:
            features = compute_batch_features(store, transactions)
            with conn.cursor() as cur:
                insert_transactions(cur, transactions)
            conn.commit()
            return {"replayed": sum(f is None for f in features)}

        return consume(handle_batch, idle_exit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute online features from Kafka.")
    parser.add_argument("--idle-exit", type=float, help="exit after this many seconds idle")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run(args.idle_exit)


if __name__ == "__main__":
    main()

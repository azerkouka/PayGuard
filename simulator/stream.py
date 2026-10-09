"""Streaming simulator: `python -m simulator.stream --rate 50`.

Publishes transactions to Kafka, keyed by user_id. The stream runs on a SIMULATED clock:
it continues the batch history with the same users and the same per-user activity, so the
features seen online look like the ones the model was trained on. `--rate` only sets how
fast simulated time is published (at 50 tx/s, a simulated day takes about two minutes).
"""

import argparse
import itertools
import time
from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd
from confluent_kafka import Producer

from config import settings
from schema import encode_transaction
from simulator.generate import (
    DEFAULT_DAYS,
    DEFAULT_FRAUD_RATE,
    DEFAULT_MERCHANTS,
    DEFAULT_START_DATE,
    DEFAULT_TRANSACTIONS,
    DEFAULT_USERS,
    finalize,
    generate_window,
)
from simulator.profiles import build_population, plan_life_events
from simulator.scenarios import FraudContext

# by default the stream starts the day after the batch history ends
STREAM_START_DATE = str(np.datetime64(DEFAULT_START_DATE) + np.timedelta64(DEFAULT_DAYS, "D"))
TX_PER_DAY = DEFAULT_TRANSACTIONS // DEFAULT_DAYS
HORIZON_DAYS = 365  # trips and phone changes are planned this far ahead


def transaction_stream(
    seed: int | None = None,
    start_date: str = STREAM_START_DATE,
    tx_per_day: int = TX_PER_DAY,
    fraud_rate: float = DEFAULT_FRAUD_RATE,
    n_users: int = DEFAULT_USERS,
    n_merchants: int = DEFAULT_MERCHANTS,
    population_seed: int = settings.seed,
) -> Iterator[dict]:
    """Endless transactions in time order, generated one simulated day at a time.

    The population comes from `population_seed` (same users as the batch history); the events
    come from `seed`, random by default so that two runs never emit the same transaction ids.
    """
    pop = build_population(n_users, n_merchants, np.random.default_rng(population_seed))
    rng = np.random.default_rng(seed)
    start = np.datetime64(start_date, "s")
    ctx = FraudContext(pop, plan_life_events(pop, start, HORIZON_DAYS, rng), rng)

    carry = None  # transactions generated for a day but dated after its end
    for day in itertools.count():
        day_start = start + np.timedelta64(day, "D")
        day_end = pd.Timestamp(day_start + np.timedelta64(1, "D"))
        df = generate_window(ctx, day_start, 1, tx_per_day, fraud_rate)
        if carry is not None and len(carry):
            df = pd.concat([carry, df], ignore_index=True)
        df = finalize(df, rng)
        late = df["timestamp"] >= day_end
        carry = df[late]
        yield from df[~late].to_dict("records")


def publish(
    transactions: Iterable[dict],
    producer: Producer,
    topic: str,
    rate: float,
    duration: float | None = None,
    report_every: float = 5.0,
) -> int:
    """Send transactions at `rate` per second until `duration` seconds have passed (or forever)."""
    errors = []

    def on_delivery(err, _msg):
        if err is not None:
            errors.append(err)

    started = last_report = time.perf_counter()
    sent = frauds = 0
    try:
        for tx in transactions:
            due = started + sent / rate
            while (now := time.perf_counter()) < due:
                producer.poll(0)
                time.sleep(min(due - now, 0.05))
            if duration is not None and now - started >= duration:
                break

            key, value = encode_transaction(tx)
            while True:
                try:
                    producer.produce(topic, key=key, value=value, on_delivery=on_delivery)
                    break
                except BufferError:  # local queue full: let the producer drain it
                    producer.poll(0.5)
            producer.poll(0)
            sent += 1
            frauds += tx["is_fraud"]

            if now - last_report >= report_every:
                last_report = now
                print(
                    f"sent {sent:>8,}  {sent / (now - started):6.1f} tx/s  frauds {frauds:>5,}  "
                    f"simulated time {tx['timestamp']}  delivery errors {len(errors)}"
                )
    except KeyboardInterrupt:
        print("stopping...")
    finally:
        producer.flush(10)
    elapsed = time.perf_counter() - started
    print(
        f"sent {sent:,} transactions in {elapsed:.1f}s ({sent / elapsed:.1f} tx/s), "
        f"{frauds:,} frauds, {len(errors)} delivery errors"
    )
    return sent


def make_producer() -> Producer:
    return Producer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "enable.idempotence": True,  # retries can neither duplicate nor reorder messages
            "linger.ms": 10,  # small batches: a few ms of delay for far fewer requests
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish simulated transactions to Kafka.")
    parser.add_argument("--rate", type=float, default=settings.stream_rate, help="tx per second")
    parser.add_argument("--duration", type=float, help="stop after this many seconds")
    parser.add_argument("--seed", type=int, help="event seed (random by default)")
    parser.add_argument("--start-date", default=STREAM_START_DATE, help="simulated start date")
    parser.add_argument("--fraud-rate", type=float, default=DEFAULT_FRAUD_RATE)
    parser.add_argument("--topic", default=settings.kafka_transactions_topic)
    args = parser.parse_args()

    transactions = transaction_stream(args.seed, args.start_date, fraud_rate=args.fraud_rate)
    print(f"publishing to '{args.topic}' at {args.rate:g} tx/s (Ctrl+C to stop)")
    publish(transactions, make_producer(), args.topic, args.rate, args.duration)


if __name__ == "__main__":
    main()

"""Run the whole pipeline and measure it: `python -m scripts.benchmark --duration 120 --rate 50`.

Starts the scoring consumer, streams transactions for the given duration, waits until every
one of them is scored, then reads latency and throughput from the `scores` table.
Needs `make up`, `make generate` and `make train` first.

Latency is end to end: from the moment a transaction is published to Kafka to the moment its
score is computed (queueing + features + model), as recorded in scores.latency_ms.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime

import psycopg
from psycopg.rows import dict_row

from config import settings
from scripts.run_pipeline import check_prerequisites, start_consumer, stop_consumer
from simulator.stream import make_producer, publish, transaction_stream

DRAIN_TIMEOUT_SECONDS = 600

RESULTS_SQL = """
SELECT count(*)                                                        AS scored,
       percentile_cont(0.50) WITHIN GROUP (ORDER BY s.latency_ms)      AS p50,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY s.latency_ms)      AS p95,
       percentile_cont(0.99) WITHIN GROUP (ORDER BY s.latency_ms)      AS p99,
       max(s.latency_ms)                                               AS max,
       extract(epoch FROM max(s.scored_at) - min(s.scored_at))         AS seconds,
       count(*) FILTER (WHERE s.decision = 'review')                   AS review,
       count(*) FILTER (WHERE s.decision = 'block')                    AS block,
       count(*) FILTER (WHERE t.is_fraud)                              AS frauds,
       count(*) FILTER (WHERE t.is_fraud AND s.decision <> 'approve')  AS frauds_flagged
FROM scores s JOIN transactions t USING (transaction_id)
WHERE s.scored_at >= %s
"""


def wait_until_scored(conn: psycopg.Connection, expected: int, since: datetime) -> None:
    """Block until `expected` transactions have a score, or give up after the drain timeout."""
    deadline = time.monotonic() + DRAIN_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        scored = conn.execute(
            "SELECT count(*) AS n FROM scores WHERE scored_at >= %s", (since,)
        ).fetchone()["n"]
        if scored >= expected:
            return
        time.sleep(0.5)
    sys.exit(f"only {scored:,} of {expected:,} transactions were scored before the timeout")


def summarize(row: dict, duration: float, rate: float) -> dict:
    seconds = float(row["seconds"]) or 1.0
    return {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "duration_s": duration,
        "target_rate_tx_s": rate,
        "transactions_scored": row["scored"],
        "throughput_tx_s": round(row["scored"] / seconds, 1),
        "latency_ms": {k: round(float(row[k]), 1) for k in ("p50", "p95", "p99", "max")},
        "decisions": {"review": row["review"], "block": row["block"]},
        "alert_rate": round((row["review"] + row["block"]) / row["scored"], 4),
        "frauds": row["frauds"],
        "frauds_flagged": row["frauds_flagged"],
    }


def print_report(results: dict) -> None:
    latency, decisions = results["latency_ms"], results["decisions"]
    print(f"\nscored       {results['transactions_scored']:,} transactions")
    print(
        f"throughput   {results['throughput_tx_s']:,} tx/s (target {results['target_rate_tx_s']:g})"
    )
    print(
        f"latency      p50 {latency['p50']} ms   p95 {latency['p95']} ms   "
        f"p99 {latency['p99']} ms   max {latency['max']} ms"
    )
    print(
        f"decisions    {decisions['review']:,} review, {decisions['block']:,} block "
        f"({results['alert_rate']:.1%} of traffic)"
    )
    print(f"frauds       {results['frauds_flagged']:,} of {results['frauds']:,} flagged")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the streaming pipeline.")
    parser.add_argument("--duration", type=float, default=120, help="seconds of streaming")
    parser.add_argument("--rate", type=float, default=settings.stream_rate, help="tx per second")
    parser.add_argument("--no-bootstrap", action="store_true", help="keep the current Redis state")
    parser.add_argument("--output", default="benchmark.json", help="file name in models/")
    args = parser.parse_args()

    check_prerequisites()
    if not args.no_bootstrap:
        # every run restarts the simulated clock, so start from the same known state
        print("loading the history into Redis and PostgreSQL...")
        subprocess.run([sys.executable, "-m", "streaming.bootstrap"], check=True)

    with psycopg.connect(settings.postgres_dsn, row_factory=dict_row, autocommit=True) as conn:
        started_at = conn.execute("SELECT now() AS now").fetchone()["now"]
        # the consumer is ready before the first message: its startup is not counted as latency
        consumer = start_consumer()
        try:
            print(f"streaming at {args.rate:g} tx/s for {args.duration:g}s...")
            sent = publish(
                transaction_stream(),
                make_producer(),
                settings.kafka_transactions_topic,
                args.rate,
                args.duration,
            )
            wait_until_scored(conn, sent, started_at)
        finally:
            stop_consumer(consumer)
        row = conn.execute(RESULTS_SQL, (started_at,)).fetchone()

    results = summarize(row, args.duration, args.rate)
    out = settings.models_dir / args.output
    out.write_text(json.dumps(results, indent=2))
    print_report(results)
    print(f"written to {out}")


if __name__ == "__main__":
    main()

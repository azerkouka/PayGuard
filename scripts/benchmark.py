"""Run the whole pipeline and measure it: `python -m scripts.benchmark --duration 120 --rate 50`.

Starts the scoring consumer and the streaming simulator as separate processes, waits for the
stream to end and the consumer to drain, then reads latency and throughput from the `scores`
table. Needs `make up`, `make generate` and `make train` first.

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

from config import settings

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


def run_module(module: str, *args: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", module, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )  # fmt: skip


def last_line(process: subprocess.Popen) -> str:
    lines = [line for line in process.communicate()[0].strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the streaming pipeline.")
    parser.add_argument("--duration", type=float, default=120, help="seconds of streaming")
    parser.add_argument("--rate", type=float, default=settings.stream_rate, help="tx per second")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="keep the current Redis state")  # fmt: skip
    parser.add_argument("--output", default="benchmark.json", help="file name in models/")
    args = parser.parse_args()

    if not args.no_bootstrap:
        # every run restarts the simulated clock, so start from the same known state
        print("bootstrapping Redis from the batch history...")
        subprocess.run([sys.executable, "-m", "streaming.bootstrap"], check=True)

    with psycopg.connect(settings.postgres_dsn) as conn:
        started_at = conn.execute("SELECT now()").fetchone()[0]

    consumer = run_module("streaming.scoring_consumer", "--idle-exit", "8")
    # wait until the consumer has loaded the model and joined the group, so that its startup
    # time is not counted as latency
    for line in consumer.stdout:
        if "assigned partitions" in line:
            break
    else:
        sys.exit("the consumer stopped before it was ready")

    print(f"streaming at {args.rate:g} tx/s for {args.duration:g}s...")
    stream = run_module("simulator.stream", "--rate", str(args.rate),
                        "--duration", str(args.duration))  # fmt: skip
    stream_summary = last_line(stream)
    print(f"  simulator: {stream_summary}")
    t0 = time.perf_counter()
    consumer_summary = last_line(consumer)
    print(f"  consumer:  {consumer_summary.split('consumer ', 1)[-1]}")
    drain_seconds = time.perf_counter() - t0 - 8  # the consumer waits 8 s idle before exiting
    if stream.returncode or consumer.returncode:
        sys.exit("a process failed; see the lines above")

    with psycopg.connect(settings.postgres_dsn, row_factory=psycopg.rows.dict_row) as conn:
        r = conn.execute(RESULTS_SQL, (started_at,)).fetchone()
    if not r["scored"]:
        sys.exit("no transaction was scored")

    seconds = float(r["seconds"]) or 1.0
    results = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "duration_s": args.duration,
        "target_rate_tx_s": args.rate,
        "transactions_scored": r["scored"],
        "throughput_tx_s": round(r["scored"] / seconds, 1),
        "latency_ms": {k: round(float(r[k]), 1) for k in ("p50", "p95", "p99", "max")},
        "decisions": {"review": r["review"], "block": r["block"]},
        "alert_rate": round((r["review"] + r["block"]) / r["scored"], 4),
        "frauds": r["frauds"],
        "frauds_flagged": r["frauds_flagged"],
        "consumer_drain_s": round(max(drain_seconds, 0.0), 1),
    }
    out = settings.models_dir / args.output
    out.write_text(json.dumps(results, indent=2))

    lat = results["latency_ms"]
    print(f"\nscored       {r['scored']:,} transactions")
    print(f"throughput   {results['throughput_tx_s']:,} tx/s (target {args.rate:g})")
    print(f"latency      p50 {lat['p50']} ms   p95 {lat['p95']} ms   p99 {lat['p99']} ms   "
          f"max {lat['max']} ms")  # fmt: skip
    print(f"decisions    {r['review']:,} review, {r['block']:,} block "
          f"({results['alert_rate']:.1%} of traffic)")  # fmt: skip
    print(f"frauds       {r['frauds_flagged']:,} of {r['frauds']:,} flagged")
    print(f"written to {out}")


if __name__ == "__main__":
    main()

"""Run the live pipeline with one command: `python -m scripts.run_pipeline --rate 50`.

1. loads the batch history into Redis and PostgreSQL (the stream continues from there)
2. starts the scoring consumer in a second process and waits until it is ready
3. publishes simulated transactions until Ctrl+C, then stops the consumer cleanly

Needs `make up`, `make generate` and `make train` first. Open the dashboard in another
terminal with `make dashboard`.
"""

import argparse
import signal
import subprocess
import sys
import threading

from config import settings
from scripts.check_infra import check_kafka, check_postgres, check_redis, describe_error
from simulator.stream import make_producer, publish, transaction_stream

DATA = settings.data_dir / "transactions.parquet"
MODEL = settings.models_dir / "model.json"
CONSUMER_STOP_TIMEOUT_SECONDS = 20


def check_prerequisites() -> None:
    """Stop with a message saying what to run, rather than with a stack trace later."""
    problems = []
    if not DATA.exists():
        problems.append(f"{DATA} is missing: run `make generate`")
    if not MODEL.exists():
        problems.append(f"{MODEL} is missing: run `make train`")
    services = {"Kafka": check_kafka, "Redis": check_redis, "PostgreSQL": check_postgres}
    for name, check in services.items():
        try:
            check()
        except Exception as exc:  # any failure means "not usable"; the reason is reported
            problems.append(f"{name} is not usable ({describe_error(exc)}): run `make up`")
    if problems:
        sys.exit("cannot start the pipeline:\n  - " + "\n  - ".join(problems))


def start_consumer() -> subprocess.Popen:
    """Start the scoring consumer and return once it has joined the Kafka group."""
    # its own process group on Windows, so that Ctrl+C reaches it through stop_consumer only
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    consumer = subprocess.Popen(
        [sys.executable, "-m", "streaming.scoring_consumer"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=flags,
    )
    for line in consumer.stdout:
        if "assigned partitions" in line:
            break
    else:
        sys.exit(
            "the consumer stopped before it was ready; run it alone to see why: "
            "python -m streaming.scoring_consumer"
        )

    def forward_output() -> None:
        for line in consumer.stdout:
            if " consumer " in line:  # its own log lines, not library warnings
                print(f"  [consumer] {line.split(' consumer ', 1)[1].rstrip()}")

    threading.Thread(target=forward_output, daemon=True).start()
    return consumer


def stop_consumer(consumer: subprocess.Popen) -> None:
    """Ask the consumer to finish its batch and commit, then wait for it."""
    if consumer.poll() is not None:
        return
    consumer.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGTERM)
    try:
        consumer.wait(timeout=CONSUMER_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        consumer.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run consumer + simulator together.")
    parser.add_argument("--rate", type=float, default=settings.stream_rate, help="tx per second")
    parser.add_argument("--duration", type=float, help="stop after this many seconds")
    parser.add_argument("--no-bootstrap", action="store_true", help="keep the current Redis state")
    args = parser.parse_args()

    check_prerequisites()
    if not args.no_bootstrap:
        # every run restarts the simulated clock, so start from the end of the history
        print("loading the history into Redis and PostgreSQL...")
        subprocess.run([sys.executable, "-m", "streaming.bootstrap"], check=True)

    print("starting the scoring consumer...")
    consumer = start_consumer()
    try:
        print(
            f"streaming at {args.rate:g} tx/s (Ctrl+C to stop). "
            "Dashboard: `make dashboard` in another terminal."
        )
        publish(
            transaction_stream(),
            make_producer(),
            settings.kafka_transactions_topic,
            args.rate,
            args.duration,
        )
    finally:
        print("stopping the consumer...")
        stop_consumer(consumer)


if __name__ == "__main__":
    main()

"""Check that Kafka, Redis and PostgreSQL are reachable: `python -m scripts.check_infra`."""

import sys

import psycopg
import redis
from confluent_kafka.admin import AdminClient

from config import settings

TABLES = {"transactions", "scores", "analyst_labels"}


def check_kafka() -> str:
    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap_servers})
    topics = admin.list_topics(timeout=5).topics
    expected = {settings.kafka_transactions_topic, settings.kafka_alerts_topic}
    if missing := expected - set(topics):
        raise RuntimeError(f"missing topics: {sorted(missing)}")
    return ", ".join(f"{t} ({len(topics[t].partitions)} partitions)" for t in sorted(expected))


def check_redis() -> str:
    client = redis.Redis(host=settings.redis_host, port=settings.redis_port, socket_timeout=5)
    client.ping()
    return f"version {client.info('server')['redis_version']}"


def check_postgres() -> str:
    with psycopg.connect(settings.postgres_dsn, connect_timeout=5) as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()
    if missing := TABLES - {r[0] for r in rows}:
        raise RuntimeError(f"missing tables: {sorted(missing)}")
    return "tables " + ", ".join(sorted(TABLES))


def main() -> None:
    checks = {"kafka": check_kafka, "redis": check_redis, "postgres": check_postgres}
    failed = False
    for name, check in checks.items():
        try:
            print(f"OK    {name:<9} {check()}")
        except Exception as exc:  # report every service, whatever the failure
            failed = True
            reason = str(exc).strip().splitlines()[0].encode("ascii", "replace").decode()
            print(f"FAIL  {name:<9} {type(exc).__name__}: {reason}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

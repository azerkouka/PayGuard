"""Load the batch history into the online stores: `python -m streaming.bootstrap`.

- Redis: every user's state at the end of the history. Without it the stream starts with users
  who have no past, and their first features (and scores) are not comparable to training.
  Replaces any state already in Redis.
- PostgreSQL: the historical transactions, so the dashboard can show a user's past activity.
  Rows already present are left untouched.
"""

import argparse
import io
import time
from pathlib import Path

import pandas as pd
import psycopg

from config import settings
from features.store import FeatureStore
from schema import COLUMNS


def load_history_into_postgres(df: pd.DataFrame) -> int:
    """Bulk-insert `df` into the transactions table. Returns the number of new rows."""
    buffer = io.StringIO()
    df[COLUMNS].to_csv(buffer, index=False, header=False)
    columns = ", ".join(f'"{c}"' for c in COLUMNS)
    with psycopg.connect(settings.postgres_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE history (LIKE transactions INCLUDING DEFAULTS) ON COMMIT DROP"
        )
        with cur.copy(f"COPY history ({columns}) FROM STDIN WITH (FORMAT csv)") as copy:
            copy.write(buffer.getvalue())
        cur.execute(
            f"INSERT INTO transactions ({columns}) SELECT {columns} FROM history "
            "ON CONFLICT (transaction_id) DO NOTHING"
        )
        return cur.rowcount


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the batch history into Redis and Postgres.")
    parser.add_argument("--data", type=Path, default=settings.data_dir / "transactions.parquet")
    parser.add_argument("--skip-postgres", action="store_true", help="only load Redis")
    args = parser.parse_args()

    t0 = time.perf_counter()
    df = pd.read_parquet(args.data)
    n_users = FeatureStore.from_settings().bootstrap(df)
    print(
        f"redis: state of {n_users:,} users from {len(df):,} transactions "
        f"(up to {df['timestamp'].max()}) in {time.perf_counter() - t0:.1f}s"
    )
    if not args.skip_postgres:
        t0 = time.perf_counter()
        inserted = load_history_into_postgres(df)
        print(
            f"postgres: {inserted:,} historical transactions added "
            f"in {time.perf_counter() - t0:.1f}s"
        )


if __name__ == "__main__":
    main()

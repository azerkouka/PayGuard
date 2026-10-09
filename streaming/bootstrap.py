"""Preload Redis with every user's state at the end of the batch history:
`python -m streaming.bootstrap`.

Without it the stream starts with users who have no past, and their first features (and
scores) are not comparable to training. Replaces any state already in Redis.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from config import settings
from streaming.feature_consumer import make_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Load historical user state into Redis.")
    parser.add_argument("--data", type=Path, default=settings.data_dir / "transactions.parquet")
    args = parser.parse_args()

    t0 = time.perf_counter()
    df = pd.read_parquet(args.data)
    n_users = make_store().bootstrap(df)
    print(
        f"loaded the state of {n_users:,} users from {len(df):,} transactions "
        f"(up to {df['timestamp'].max()}) in {time.perf_counter() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
